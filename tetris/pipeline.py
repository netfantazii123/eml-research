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
from eml_distiller import distill


def oracle_score(oracle, n_games: int, max_placements: int,
                 device, seed: int = 1000) -> tuple[float, float]:
    """Play with the oracle (deterministic argmax). Returns (avg_lines, avg_placements)."""
    oracle.eval()
    env = TetrisEnv(seed=seed)
    total_lines = 0
    total_steps = 0
    for _ in range(n_games):
        obs = env.reset()
        f = 0
        while not env.done and f < max_placements:
            grid, scalars, afeats, mask = obs_to_tensors(obs, device)
            with torch.no_grad():
                logits = oracle.get_logits(grid, scalars, afeats, mask)
            a = int(torch.argmax(logits, dim=-1).item())
            obs, _, _, _ = env.step_placement(a)
            f += 1
        total_lines += env.score
        total_steps += f
    return total_lines / n_games, total_steps / n_games


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
        joint=joint, verbose=verbose,
        should_stop=should_stop, on_event=on_event,
    )

    # ── Evaluation vs the oracle + saving ────────────────────────────────
    _emit({'type': 'phase', 'phase': 'eval'})
    orc_lines, _ = oracle_score(
        oracle, eval_games, config.EML_INGAME_MAX_PLACEMENTS, device)
    eml_lines = result['final_lines']
    ratio = (eml_lines / orc_lines * 100.0) if orc_lines > 0 else 0.0

    trees = result['trees']
    meta = {
        'final_lines': eml_lines,
        'base_lines': result['base_lines'],
        'oracle_lines': orc_lines,
        'ratio_pct': ratio,
        'dataset_size': result['dataset_size'],
        'depth_penalty': depth_penalty_name,
        'joint': joint,
    }
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


def _make_variant_params(i: int, spread: float, rng,
                         base_generations: int, base_population: int) -> dict:
    """
    Parameters of variant i: a controlled spread around the base values.

    Variant 0 is always the clean base (medium, no jitter): the comparison
    anchor.
    """
    if i == 0 or spread <= 0:
        return {
            'seed': 1000 + i,
            'generations': base_generations,
            'population': base_population,
            'depth_penalty': 'medium',
            'max_depth': config.EML_MAX_DEPTH,
            'var_bonus': config.EML_VAR_BONUS,
        }

    def _jit(v):
        return max(1, int(round(v * rng.uniform(1.0 - spread, 1.0 + spread))))

    return {
        'seed': 1000 + i,
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
    n_variants: int | None = None,
    spread: float | None = None,
    oracle=None,
    device=None,
    n_episodes: int | None = None,
    reuse_data: bool = False,
    data_path: str | None = None,
    data_generations: int | None = None,
    data_population: int | None = None,
    joint: bool = True,
    eval_games: int | None = None,
    on_event=None,
    should_stop=None,
    verbose: bool = False,
) -> dict:
    """
    Distillation of a batch of formulas: one dataset, N evolutions with a
    parameter spread.

    The best variant (by in-game lines) is saved to models/best_eml.json.
    All variants + summary -> runs/<date>_distill_batch/.
    """
    def _emit(ev):
        if on_event is not None:
            on_event(ev)

    if n_variants is None:
        n_variants = config.EML_BATCH_VARIANTS
    if spread is None:
        spread = config.EML_BATCH_SPREAD
    if device is None:
        device = get_device()
    if data_path is None:
        data_path = os.path.join(config.RESULTS_DIR, 'eml_dataset.npz')
    if eval_games is None:
        eval_games = config.EML_INGAME_GAMES
    if data_generations is None:
        data_generations = config.EML_GENERATIONS
    if data_population is None:
        data_population = config.EML_POPULATION

    run_dir = runs_mod.create_run_dir('distill_batch')
    _emit({'type': 'log', 'msg': f"Run dir: {run_dir}  "
                                 f"({n_variants} variants, spread ±{spread:.0%})"})

    oracle = _load_oracle(oracle, device, _emit)
    ds = _get_dataset(oracle, device, n_episodes=n_episodes,
                      reuse_data=reuse_data, data_path=data_path,
                      emit=_emit, should_stop=should_stop, verbose=verbose)
    if ds is None:
        return {'cancelled': True}
    features, logits, groups, weights = ds

    rng = np.random.default_rng(12345)
    variants: list[dict] = []
    cfg_backup = (config.EML_MAX_DEPTH, config.EML_VAR_BONUS)

    try:
        for i in range(n_variants):
            if should_stop is not None and should_stop():
                _emit({'type': 'cancelled'})
                break
            params = _make_variant_params(
                i, spread, rng, data_generations, data_population)
            _emit({'type': 'variant_start', 'variant': i,
                   'total': n_variants, 'params': params})
            if verbose:
                print(f"\n=== VARIANT {i + 1}/{n_variants}: {params} ===")

            # Spread of the parameters the GA reads from config.
            config.EML_MAX_DEPTH = params['max_depth']
            config.EML_VAR_BONUS = params['var_bonus']
            random.seed(params['seed'])   # evolution seed (the GA random module)

            result = distill(
                features, logits, weights,
                groups=groups,
                depth_penalty_name=params['depth_penalty'],
                data_generations=params['generations'],
                data_population=params['population'],
                joint=joint, verbose=verbose,
                should_stop=should_stop, on_event=on_event,
            )
            tree = result['trees'][0]
            fpath = os.path.join(run_dir, f"formula_{i:02d}.json")
            storage.save_formulas(result['trees'], path=fpath, meta={
                'variant': i, 'params': params,
                'final_lines': result['final_lines'],
                'base_lines': result['base_lines'],
            })
            variants.append({
                'variant': i, 'params': params,
                'final_lines': result['final_lines'],
                'base_lines': result['base_lines'],
                'size': tree.size(), 'depth': tree.depth(),
                'nvars': tree.n_unique_vars(),
                'path': fpath, 'result': result,
            })
            _emit({'type': 'variant_done', 'variant': i,
                   'total': n_variants,
                   'final_lines': result['final_lines'],
                   'size': tree.size(),
                   'formula': tree.to_string()[:80]})
    finally:
        config.EML_MAX_DEPTH, config.EML_VAR_BONUS = cfg_backup

    if not variants:
        return {'cancelled': True}

    # ── The best variant -> the canonical best_eml.json ──────────────────
    _emit({'type': 'phase', 'phase': 'eval'})
    orc_lines, _ = oracle_score(
        oracle, eval_games, config.EML_INGAME_MAX_PLACEMENTS, device)
    best = max(variants, key=lambda v: v['final_lines'])
    ratio = (best['final_lines'] / orc_lines * 100.0) if orc_lines > 0 else 0.0
    meta = {
        'final_lines': best['final_lines'],
        'base_lines': best['base_lines'],
        'oracle_lines': orc_lines,
        'ratio_pct': ratio,
        'dataset_size': len(features),
        'variant': best['variant'],
        'params': best['params'],
        'batch': {'n_variants': len(variants), 'spread': spread,
                  'run_dir': run_dir},
    }
    path = storage.save_formulas(best['result']['trees'], meta=meta)
    _save_reports(best['result'], run_dir, _emit, meta)
    runs_mod.copy_into(run_dir, path)
    runs_mod.save_summary(run_dir, {
        'oracle_lines': orc_lines,
        'best_variant': best['variant'],
        'best_ratio_pct': ratio,
        'variants': [{k: v for k, v in vr.items() if k != 'result'}
                     for vr in variants],
    })

    trees = best['result']['trees']
    summary = {
        'trees': trees,
        'eml_lines': best['final_lines'],
        'base_lines': best['base_lines'],
        'oracle_lines': orc_lines,
        'ratio_pct': ratio,
        'total_size': best['size'],
        'path': path,
        'run_dir': run_dir,
        'variants': [{k: v for k, v in vr.items() if k != 'result'}
                     for vr in variants],
        'formulas': [t.to_string() for t in trees],
        'sizes': [t.size() for t in trees],
        'depths': [t.depth() for t in trees],
        'nvars': [t.n_unique_vars() for t in trees],
    }
    _emit({'type': 'done', **{k: v for k, v in summary.items() if k != 'trees'}})
    return summary


class _Cancelled(Exception):
    """Internal cancellation signal during dataset collection."""
