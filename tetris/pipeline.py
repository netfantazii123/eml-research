"""
pipeline.py — reusable orchestration of EML distillation.

One code path for the CLI (main.py distill) and the GUI (Distillation tab):
    oracle -> dataset (afterstate features, logits) -> EML formula -> evaluation
    -> saving.

Modes:
    full_distill()  — a single formula.
    batch_distill() — a batch of formulas with a controlled spread of GA
                      parameters; the dataset is collected once, and the best
                      formula becomes the canonical one (models/best_eml.json).

Every run creates runs/<date>_distill/ with a config snapshot, the formulas, a
summary and plots (see runs.py).

`on_event(dict)` is a progress callback for the GUI, `should_stop()` cancels.
"""

import os
import csv
import random

import numpy as np
import torch

import config
import storage
import runs as runs_mod
from cnn_oracle import TetrisCNN, get_device, obs_to_tensors
from env import TetrisEnv
from dataset_collector import (
    collect_dataset, save_dataset, load_dataset,
    compute_sample_weights, summarize,
)
from eml_distiller import distill, score_episodes, pool_slice, static_metrics


def oracle_score(oracle, seeds, max_placements: int, device) -> dict:
    """
    Play the oracle (deterministic argmax) on one game per seed.

    A fresh env per seed, exactly like eml_distiller.play_episodes — the oracle
    and the formula must see the same piece sequences, or the ratio between them
    compares two different games.
    """
    oracle.eval()
    lines: list[int] = []
    steps: list[int] = []
    for s in seeds:
        env = TetrisEnv(seed=int(s))
        obs = env.reset()
        f = 0
        while not env.done and f < max_placements:
            grid, scalars, afeats, mask = obs_to_tensors(obs, device)
            with torch.no_grad():
                logits = oracle.get_logits(grid, scalars, afeats, mask)
            a = int(torch.argmax(logits, dim=-1).item())
            obs, _, _, _ = env.step_placement(a)
            f += 1
        lines.append(env.score)
        steps.append(f)
    la = np.asarray(lines, dtype=np.float64)
    sa = np.asarray(steps, dtype=np.float64)
    return {
        'lines': lines, 'steps': steps, 'seeds': [int(s) for s in seeds],
        'avg_lines': float(la.mean()),
        'std_lines': float(la.std(ddof=1)) if len(la) > 1 else 0.0,
        'avg_steps': float(sa.mean()),
    }


def evaluate_holdout(tree, oracle, device, *, n_games: int | None = None,
                     max_placements: int | None = None) -> dict:
    """
    The reported number, and the only place the TEST pool is ever touched.

    Both policies run on the SAME held-out seeds; nothing in the pipeline may
    select on the result, which is what makes it an estimate rather than a
    maximum over noise.
    """
    if n_games is None:
        n_games = config.EML_TEST_GAMES
    if max_placements is None:
        max_placements = config.EML_INGAME_MAX_PLACEMENTS
    seeds = pool_slice(config.EML_SEED_POOL_TEST, n_games)

    eml = score_episodes(tree, seeds, max_placements)
    orc = oracle_score(oracle, seeds, max_placements, device)
    ratio = (eml['avg_lines'] / orc['avg_lines'] * 100.0
             if orc['avg_lines'] > 0 else 0.0)

    # Paired over seeds: both policies played the same games, so the per-seed
    # difference is the honest spread of the comparison.
    diff = np.asarray(eml['lines'], dtype=np.float64) - \
        np.asarray(orc['lines'], dtype=np.float64)
    return {
        'n_games': n_games,
        'seeds': [int(s) for s in seeds],
        'eml': eml,
        'oracle': orc,
        'eml_lines': eml['avg_lines'],
        'eml_std': eml['std_lines'],
        'oracle_lines': orc['avg_lines'],
        'oracle_std': orc['std_lines'],
        'ratio_pct': ratio,
        'paired_diff_mean': float(diff.mean()),
        'paired_diff_std': float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
    }


# ── Shared stages (oracle, dataset) ───────────────────────────────────────────

def _load_oracle(oracle, device, emit):
    if oracle is None:
        oracle = TetrisCNN().to(device)
        meta = storage.load_oracle(oracle, device=device)
        emit({'type': 'log', 'msg': f"Oracle loaded ({meta})"})
    oracle.eval()
    return oracle


def _get_dataset(oracle, device, *, n_episodes, reuse_data, data_path,
                 emit, should_stop, verbose):
    """Collect or reuse the dataset. None -> cancelled by the user."""
    emit({'type': 'phase', 'phase': 'dataset'})
    if reuse_data and os.path.exists(data_path):
        features, logits, groups = load_dataset(data_path)
        emit({'type': 'log', 'msg': f"Reused dataset {data_path}"})
    else:
        eps = n_episodes if n_episodes is not None else config.EML_DATASET_EPISODES
        emit({'type': 'log', 'msg': f"Collecting dataset ({eps} episodes)..."})

        def _progress(done, total, n_samples):
            emit({'type': 'collect', 'episode': done, 'total': total,
                  'samples': n_samples})
            if should_stop is not None and should_stop():
                raise _Cancelled()

        try:
            features, logits, groups = collect_dataset(
                oracle, n_episodes=eps, device=device,
                verbose=verbose, progress_cb=_progress)
        except _Cancelled:
            emit({'type': 'cancelled'})
            return None
        save_dataset(features, logits, groups, data_path)
        emit({'type': 'log', 'msg': f"Saved dataset -> {data_path}"})

    emit({'type': 'log', 'msg': summarize(features, logits, groups)})
    weights = compute_sample_weights(logits, groups)
    emit({'type': 'dataset_ready', 'samples': int(len(features)),
          'counts': [int(len(np.unique(groups)))]})
    return features, logits, groups, weights


# ── Raw per-episode log ───────────────────────────────────────────────────────
#
# One long CSV, one row per EPISODE, never an aggregate. Every plot in the thesis
# (mean, spread, R(K) curve, paired EML-vs-oracle tests) is a groupby over this
# file, so a new figure never means re-running the experiments. The schema
# already carries basis/k_budget so rows written before the ClassicBasis and the
# node-budget work stay comparable with rows written after it.

RAW_EPISODES_COLUMNS = [
    'task', 'basis', 'k_budget', 'config_id', 'ga_seed', 'variant',
    'pool', 'policy', 'episode', 'env_seed', 'lines', 'placements',
    'nodes', 'depth', 'n_vars', 'n_exp', 'n_log',
    'commit', 'config_hash', 'run_dir',
]


def episode_rows(res, *, pool, policy, static=None, task='tetris', basis='eml',
                 k_budget=None, config_id=0, ga_seed=None, variant='single',
                 prov=None, run_dir=None) -> list[dict]:
    """One row per episode of a score_episodes()/oracle_score() result."""
    prov = prov or {}
    static = static or {}
    base = {
        'task': task, 'basis': basis, 'k_budget': k_budget,
        'config_id': config_id, 'ga_seed': ga_seed, 'variant': variant,
        'pool': pool, 'policy': policy,
        'nodes': static.get('nodes'), 'depth': static.get('depth'),
        'n_vars': static.get('n_vars_unique'),
        'n_exp': static.get('n_exp'), 'n_log': static.get('n_log'),
        'commit': prov.get('commit'), 'config_hash': prov.get('config_hash'),
        'run_dir': run_dir,
    }
    return [
        {**base, 'episode': i, 'env_seed': res['seeds'][i],
         'lines': ln, 'placements': st}
        for i, (ln, st) in enumerate(zip(res['lines'], res['steps']))
    ]


def write_rows(run_dir, rows) -> str:
    """Append rows to results/raw_episodes.csv (and the run directory copy)."""
    if not rows:
        return ''
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    targets = [os.path.join(config.RESULTS_DIR, 'raw_episodes.csv')]
    if run_dir:
        targets.append(os.path.join(run_dir, 'raw_episodes.csv'))
    for path in targets:
        new = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=RAW_EPISODES_COLUMNS)
            if new:
                w.writeheader()
            w.writerows(rows)
    return targets[0]


def write_val_rows(run_dir, result, **kw) -> str:
    """
    Per-episode VAL scores of one GA run.

    Written for EVERY run of a batch, not just the winner: the spread across
    seeds at a fixed config is the thing that says whether a difference between
    configs is real, and it cannot be recovered later from the winner alone.
    """
    val = result.get('val_result')
    if not val:
        return ''
    kw.setdefault('run_dir', run_dir)
    kw.setdefault('basis', result.get('basis', 'eml'))
    kw.setdefault('k_budget', result.get('max_nodes'))
    return write_rows(run_dir, episode_rows(
        val, pool='val', policy='eml', static=result.get('static'), **kw))


PARETO_COLUMNS = [
    'task', 'basis', 'config_id', 'ga_seed', 'variant', 'k',
    'train_fitness', 'val_fitness', 'val_lines', 'val_lines_std',
    'nodes', 'depth', 'n_vars', 'n_exp', 'n_log', 'n_transcendental',
    'est_code_bytes', 'est_stack_bytes',
    'commit', 'config_hash', 'run_dir', 'formula',
]


def write_pareto_rows(run_dir, result, *, task='tetris', config_id=0,
                      ga_seed=None, variant='single', prov=None) -> str:
    """
    The R(K) curve of one GA run -> results/pareto_curve.csv.

    One row per node budget, carrying both the regression fitness the search
    optimised and the in-game lines the thesis reports, so the two can be
    plotted against each other rather than assumed to agree.
    """
    rows = result.get('pareto') or []
    if not rows:
        return ''
    prov = prov or runs_mod.provenance()
    base = {
        'task': task, 'basis': result.get('basis', 'eml'),
        'config_id': config_id, 'ga_seed': ga_seed, 'variant': variant,
        'commit': prov.get('commit'), 'config_hash': prov.get('config_hash'),
        'run_dir': run_dir,
    }
    out = [{**base, **{k: v for k, v in row.items() if k in PARETO_COLUMNS}}
           for row in rows]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    targets = [os.path.join(config.RESULTS_DIR, 'pareto_curve.csv')]
    if run_dir:
        targets.append(os.path.join(run_dir, 'pareto_curve.csv'))
    for path in targets:
        new = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=PARETO_COLUMNS)
            if new:
                w.writeheader()
            w.writerows(out)
    return targets[0]


def write_test_rows(run_dir, holdout, result, **kw) -> str:
    """Per-episode TEST scores: the formula and the oracle on the same seeds."""
    kw.setdefault('run_dir', run_dir)
    kw.setdefault('basis', result.get('basis', 'eml'))
    kw.setdefault('k_budget', result.get('max_nodes'))
    rows = episode_rows(holdout['eml'], pool='test', policy='eml',
                        static=result.get('static'), **kw)
    # Static metrics describe the formula, so they stay blank on oracle rows.
    rows += episode_rows(holdout['oracle'], pool='test', policy='oracle', **kw)
    return write_rows(run_dir, rows)


def _save_reports(result, run_dir, emit, meta):
    """Distillation history + plots -> results/ and the run directory."""
    try:
        import reports
        hist_path = reports.save_distill_history(result, meta=meta)
        data_hist = (result['data_results'][0].get('history', [])
                     if result.get('data_results') else [])
        joint_hist = (result['joint_result'].get('history', [])
                      if result.get('joint_result') else [])
        png_path = reports.plot_distill_history(data_hist, joint_hist)
        emit({'type': 'log', 'msg': f"Reports -> {hist_path}, {png_path}"})
        if run_dir:
            runs_mod.copy_into(run_dir, hist_path, png_path)
    except Exception as exc:  # noqa: BLE001 — reports must not break the pipeline
        emit({'type': 'log', 'msg': f"report save failed: {exc!r}"})


# ── Single distillation ───────────────────────────────────────────────────────

def full_distill(
    *,
    oracle=None,
    device=None,
    n_episodes: int | None = None,
    reuse_data: bool = False,
    data_path: str | None = None,
    depth_penalty_name: str = 'medium',
    data_generations: int | None = None,
    data_population: int | None = None,
    joint: bool = True,
    basis_name: str | None = None,
    max_nodes: int | None = None,
    multi_objective: bool | None = None,
    eval_games: int | None = None,
    run_dir: str | None = None,
    on_event=None,
    should_stop=None,
    verbose: bool = False,
) -> dict:
    """
    The full single-distillation pipeline. None parameters -> from config.

    The 'done' event carries the final metrics; the function also returns them.
    """
    def _emit(ev):
        if on_event is not None:
            on_event(ev)

    if device is None:
        device = get_device()
    if data_path is None:
        data_path = os.path.join(config.RESULTS_DIR, 'eml_dataset.npz')
    if eval_games is None:
        eval_games = config.EML_INGAME_GAMES
    if run_dir is None:
        run_dir = runs_mod.create_run_dir('distill')
    _emit({'type': 'log', 'msg': f"Run dir: {run_dir}"})

    oracle = _load_oracle(oracle, device, _emit)
    ds = _get_dataset(oracle, device, n_episodes=n_episodes,
                      reuse_data=reuse_data, data_path=data_path,
                      emit=_emit, should_stop=should_stop, verbose=verbose)
    if ds is None:
        return {'cancelled': True}
    features, logits, groups, weights = ds

    if should_stop is not None and should_stop():
        _emit({'type': 'cancelled'})
        return {'cancelled': True}

    # ── Formula evolution ────────────────────────────────────────────────
    result = distill(
        features, logits, weights,
        groups=groups,
        depth_penalty_name=depth_penalty_name,
        data_generations=data_generations,
        data_population=data_population,
        joint=joint,
        basis_name=basis_name, max_nodes=max_nodes,
        multi_objective=multi_objective,
        verbose=verbose,
        should_stop=should_stop, on_event=on_event,
    )

    # ── Evaluation vs the oracle + saving ────────────────────────────────
    _emit({'type': 'phase', 'phase': 'eval'})
    trees = result['trees']
    holdout = evaluate_holdout(trees[0], oracle, device, n_games=eval_games)
    eml_lines = holdout['eml_lines']
    orc_lines = holdout['oracle_lines']
    ratio = holdout['ratio_pct']
    _emit({'type': 'log', 'msg':
           f"Holdout ({holdout['n_games']} unseen seeds): "
           f"EML {eml_lines:.2f}±{holdout['eml_std']:.2f} vs "
           f"oracle {orc_lines:.2f}±{holdout['oracle_std']:.2f} "
           f"-> {ratio:.1f}%  (val was {result['val_lines']:.2f})"})

    meta = {
        'final_lines': eml_lines,        # holdout mean — the reportable number
        'val_lines': result['val_lines'],  # selection metric, biased upward
        'base_lines': result['base_lines'],
        'oracle_lines': orc_lines,
        'ratio_pct': ratio,
        'holdout': {k: v for k, v in holdout.items() if k not in ('eml', 'oracle')},
        'static': result['static'],
        'basis': result.get('basis'),
        'max_nodes': result.get('max_nodes'),
        'multi_objective': result.get('multi_objective'),
        'pareto': [{k: v for k, v in r.items() if k != 'formula'}
                   for r in result.get('pareto', [])],
        'dataset_size': result['dataset_size'],
        'depth_penalty': depth_penalty_name,
        'joint': joint,
        'provenance': runs_mod.provenance(),
    }
    prov = meta['provenance']
    write_val_rows(run_dir, result, prov=prov)
    write_test_rows(run_dir, holdout, result, prov=prov)
    write_pareto_rows(run_dir, result, prov=prov)
    path = storage.save_formulas(trees, meta=meta)
    _save_reports(result, run_dir, _emit, meta)
    runs_mod.copy_into(run_dir, path)
    runs_mod.save_summary(run_dir, meta)

    total_size = sum(t.size() for t in trees)
    summary = {
        'trees': trees,
        'eml_lines': eml_lines,
        'base_lines': result['base_lines'],
        'oracle_lines': orc_lines,
        'ratio_pct': ratio,
        'total_size': total_size,
        'pareto': meta['pareto'],
        'path': path,
        'run_dir': run_dir,
        'formulas': [t.to_string() for t in trees],
        'sizes': [t.size() for t in trees],
        'depths': [t.depth() for t in trees],
        'nvars': [t.n_unique_vars() for t in trees],
    }
    _emit({'type': 'done', **{k: v for k, v in summary.items() if k != 'trees'}})
    return summary


# ── Batch distillation with a controlled spread ───────────────────────────────

_DP_CYCLE = ['medium', 'weak', 'strong']


def _make_config_params(i: int, spread: float, rng,
                        base_generations: int, base_population: int) -> dict:
    """
    GA hyper-parameters of configuration i: a controlled spread around the base.

    Configuration 0 is always the clean base (medium, no jitter): the comparison
    anchor.

    Deliberately contains NO seed. The seed is an independent axis of the batch:
    the previous version jittered the seed together with five hyper-parameters,
    which made every run a different configuration AND a different seed at once,
    so the seed-to-seed spread — the thing that decides whether a difference
    between configurations is real — could not be estimated from a batch at all.
    """
    if i == 0 or spread <= 0:
        return {
            'generations': base_generations,
            'population': base_population,
            'depth_penalty': 'medium',
            'max_depth': config.EML_MAX_DEPTH,
            'var_bonus': config.EML_VAR_BONUS,
        }

    def _jit(v):
        return max(1, int(round(v * rng.uniform(1.0 - spread, 1.0 + spread))))

    return {
        'generations': _jit(base_generations),
        'population': _jit(base_population),
        'depth_penalty': _DP_CYCLE[i % len(_DP_CYCLE)],
        'max_depth': int(np.clip(
            config.EML_MAX_DEPTH + rng.integers(-1, 2), 4, 10)),
        'var_bonus': float(config.EML_VAR_BONUS
                           * rng.uniform(1.0 - spread, 1.0 + spread)),
    }


def batch_distill(
    *,
    n_configs: int | None = None,
    n_seeds: int | None = None,
    spread: float | None = None,
    oracle=None,
    device=None,
    n_episodes: int | None = None,
    reuse_data: bool = False,
    data_path: str | None = None,
    data_generations: int | None = None,
    data_population: int | None = None,
    joint: bool = True,
    basis_name: str | None = None,
    max_nodes: int | None = None,
    multi_objective: bool | None = None,
    eval_games: int | None = None,
    on_event=None,
    should_stop=None,
    verbose: bool = False,
) -> dict:
    """
    A batch of GA runs over one dataset: n_configs configurations x n_seeds seeds.

    The two axes are independent on purpose. n_seeds > 1 at a fixed configuration
    measures how much of a difference is just the GA's own randomness; without it
    a batch can only rank configurations it cannot tell apart. Set n_configs=1,
    n_seeds=5 for a pure replicate study — that is the default shape.

    Selection (best run -> models/best_eml.json) uses the VAL pool. The reported
    score comes afterwards from the TEST pool, once, on the winner only.
    """
    def _emit(ev):
        if on_event is not None:
            on_event(ev)

    if n_configs is None:
        n_configs = config.EML_BATCH_CONFIGS
    if n_seeds is None:
        n_seeds = config.EML_BATCH_SEEDS
    if spread is None:
        spread = config.EML_BATCH_SPREAD
    if device is None:
        device = get_device()
    if data_path is None:
        data_path = os.path.join(config.RESULTS_DIR, 'eml_dataset.npz')
    if eval_games is None:
        eval_games = config.EML_TEST_GAMES
    if data_generations is None:
        data_generations = config.EML_GENERATIONS
    if data_population is None:
        data_population = config.EML_POPULATION

    run_dir = runs_mod.create_run_dir('distill_batch')
    prov = runs_mod.provenance()
    total_runs = n_configs * n_seeds
    _emit({'type': 'log', 'msg': f"Run dir: {run_dir}  "
                                 f"({n_configs} configs x {n_seeds} seeds "
                                 f"= {total_runs} runs, spread ±{spread:.0%})"})

    oracle = _load_oracle(oracle, device, _emit)
    ds = _get_dataset(oracle, device, n_episodes=n_episodes,
                      reuse_data=reuse_data, data_path=data_path,
                      emit=_emit, should_stop=should_stop, verbose=verbose)
    if ds is None:
        return {'cancelled': True}
    features, logits, groups, weights = ds

    rng = np.random.default_rng(12345)
    configs = [_make_config_params(c, spread, rng, data_generations,
                                   data_population)
               for c in range(n_configs)]
    variants: list[dict] = []
    cfg_backup = (config.EML_MAX_DEPTH, config.EML_VAR_BONUS)
    idx = 0

    try:
        for c_id, params in enumerate(configs):
            for s_i in range(n_seeds):
                if should_stop is not None and should_stop():
                    _emit({'type': 'cancelled'})
                    raise _Cancelled()
                ga_seed = config.EML_GA_SEED_BASE + s_i
                label = f"c{c_id:02d}s{ga_seed}"
                _emit({'type': 'variant_start', 'variant': idx,
                       'total': total_runs, 'params': {**params,
                                                       'config_id': c_id,
                                                       'ga_seed': ga_seed}})
                if verbose:
                    print(f"\n=== RUN {idx + 1}/{total_runs}  "
                          f"config {c_id} seed {ga_seed}: {params} ===")

                # Spread of the parameters the GA reads from config.
                config.EML_MAX_DEPTH = params['max_depth']
                config.EML_VAR_BONUS = params['var_bonus']
                random.seed(ga_seed)   # evolution seed (the GA random module)

                result = distill(
                    features, logits, weights,
                    groups=groups,
                    depth_penalty_name=params['depth_penalty'],
                    data_generations=params['generations'],
                    data_population=params['population'],
                    joint=joint,
                    basis_name=basis_name, max_nodes=max_nodes,
                    multi_objective=multi_objective,
                    verbose=verbose,
                    should_stop=should_stop, on_event=on_event,
                )
                tree = result['trees'][0]
                fpath = os.path.join(run_dir, f"formula_{label}.json")
                storage.save_formulas(result['trees'], path=fpath, meta={
                    'config_id': c_id, 'ga_seed': ga_seed, 'params': params,
                    'val_lines': result['val_lines'],
                    'base_lines': result['base_lines'],
                    'static': result['static'],
                    'provenance': prov,
                })
                write_val_rows(run_dir, result, config_id=c_id,
                               ga_seed=ga_seed, variant=label, prov=prov)
                write_pareto_rows(run_dir, result, config_id=c_id,
                                  ga_seed=ga_seed, variant=label, prov=prov)
                variants.append({
                    'variant': idx, 'label': label,
                    'config_id': c_id, 'ga_seed': ga_seed, 'params': params,
                    'val_lines': result['val_lines'],
                    'val_std': result['val_result']['std_lines'],
                    'base_lines': result['base_lines'],
                    'size': tree.size(), 'depth': tree.depth(),
                    'nvars': tree.n_unique_vars(),
                    'static': result['static'],
                    'path': fpath, 'result': result,
                })
                _emit({'type': 'variant_done', 'variant': idx,
                       'total': total_runs,
                       'val_lines': result['val_lines'],
                       'size': tree.size(),
                       'formula': tree.to_string()[:80]})
                idx += 1
    except _Cancelled:
        pass
    finally:
        config.EML_MAX_DEPTH, config.EML_VAR_BONUS = cfg_backup

    if not variants:
        return {'cancelled': True}

    # ── The best variant -> the canonical best_eml.json ──────────────────
    _emit({'type': 'phase', 'phase': 'eval'})
    best = max(variants, key=lambda v: v['val_lines'])
    holdout = evaluate_holdout(best['result']['trees'][0], oracle, device,
                               n_games=eval_games)
    eml_lines = holdout['eml_lines']
    orc_lines = holdout['oracle_lines']
    ratio = holdout['ratio_pct']

    per_config = summarize_by_config(variants)
    _emit({'type': 'log', 'msg':
           f"Winner {best['label']} — val {best['val_lines']:.2f}, "
           f"holdout {eml_lines:.2f}±{holdout['eml_std']:.2f} vs oracle "
           f"{orc_lines:.2f}±{holdout['oracle_std']:.2f} -> {ratio:.1f}%"})
    for row in per_config:
        _emit({'type': 'log', 'msg':
               f"  config {row['config_id']}: val {row['val_mean']:.2f}"
               f"±{row['val_std']:.2f} over {row['n_seeds']} seeds "
               f"(min {row['val_min']:.2f}, max {row['val_max']:.2f})"})

    meta = {
        'final_lines': eml_lines,          # holdout mean — the reportable number
        'val_lines': best['val_lines'],    # selection metric, biased upward
        'base_lines': best['base_lines'],
        'oracle_lines': orc_lines,
        'ratio_pct': ratio,
        'holdout': {k: v for k, v in holdout.items() if k not in ('eml', 'oracle')},
        'static': best['static'],
        'basis': best['result'].get('basis'),
        'max_nodes': best['result'].get('max_nodes'),
        'dataset_size': len(features),
        'variant': best['variant'],
        'label': best['label'],
        'config_id': best['config_id'],
        'ga_seed': best['ga_seed'],
        'params': best['params'],
        'batch': {'n_configs': n_configs, 'n_seeds': n_seeds,
                  'n_runs': len(variants), 'spread': spread,
                  'run_dir': run_dir},
        'per_config': per_config,
        'provenance': prov,
    }
    path = storage.save_formulas(best['result']['trees'], meta=meta)
    write_test_rows(run_dir, holdout, best['result'],
                    config_id=best['config_id'], ga_seed=best['ga_seed'],
                    variant=best['label'], prov=prov)
    _save_reports(best['result'], run_dir, _emit, meta)
    runs_mod.copy_into(run_dir, path)
    runs_mod.save_summary(run_dir, {
        'oracle_lines': orc_lines,
        'best_variant': best['variant'],
        'best_label': best['label'],
        'best_ratio_pct': ratio,
        'holdout': meta['holdout'],
        'per_config': per_config,
        'provenance': prov,
        'variants': [{k: v for k, v in vr.items() if k != 'result'}
                     for vr in variants],
    })

    trees = best['result']['trees']
    summary = {
        'trees': trees,
        'eml_lines': eml_lines,
        'val_lines': best['val_lines'],
        'base_lines': best['base_lines'],
        'oracle_lines': orc_lines,
        'ratio_pct': ratio,
        'total_size': best['size'],
        'pareto': [{k: v for k, v in r.items() if k != 'formula'}
                   for r in best['result'].get('pareto', [])],
        'path': path,
        'run_dir': run_dir,
        'per_config': per_config,
        'variants': [{k: v for k, v in vr.items() if k != 'result'}
                     for vr in variants],
        'formulas': [t.to_string() for t in trees],
        'sizes': [t.size() for t in trees],
        'depths': [t.depth() for t in trees],
        'nvars': [t.n_unique_vars() for t in trees],
    }
    _emit({'type': 'done', **{k: v for k, v in summary.items() if k != 'trees'}})
    return summary


def summarize_by_config(variants: list[dict]) -> list[dict]:
    """
    Per-configuration spread across GA seeds.

    This is the table that says whether a configuration difference is worth
    anything: if val_std within a configuration is as large as the gap between
    configurations, the ranking is noise and the batch has ranked coin flips.
    """
    by_cfg: dict[int, list[dict]] = {}
    for v in variants:
        by_cfg.setdefault(v['config_id'], []).append(v)

    rows = []
    for c_id, group in sorted(by_cfg.items()):
        vals = np.asarray([g['val_lines'] for g in group], dtype=np.float64)
        sizes = np.asarray([g['size'] for g in group], dtype=np.float64)
        rows.append({
            'config_id': c_id,
            'n_seeds': len(group),
            'seeds': [g['ga_seed'] for g in group],
            'val_mean': float(vals.mean()),
            'val_std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            'val_min': float(vals.min()),
            'val_max': float(vals.max()),
            'size_mean': float(sizes.mean()),
            'params': group[0]['params'],
        })
    return rows


class _Cancelled(Exception):
    """Internal cancellation signal during dataset collection."""
