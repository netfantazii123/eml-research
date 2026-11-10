"""
dagger_distill.py — DAgger for the Tetris EML formula (placement-based).

The question it answers: is the GAME phase compensating for distribution
shift? A DATA-phase formula ranks the oracle's own placements well and still
scores few lines, because one bad placement puts it on boards the oracle never
produced. DAgger fixes the data instead of the fitness — roll the current
formula out, have the oracle score EVERY legal placement on the boards the
formula reached, aggregate, refit.

Four students, same oracle, same DATA budget per fit, same VAL/TEST pools:

    BC            DATA phase on oracle trajectories     (DAgger iteration 0)
    DAgger        DATA phase on the aggregated dataset  (best iteration by VAL)
    BC+GAME       the existing pipeline: BC formula -> evolve_ingame
    DAgger+GAME   evolve_ingame seeded with the DAgger formula

Every TEST number comes from pipeline.evaluate_holdout: the formula and the
oracle on the same held-out seeds, nothing selects on it.

    python dagger_distill.py --seeds 0 1 2 --iters 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import storage
from env import TetrisEnv
from cnn_oracle import TetrisCNN, get_device, obs_to_tensors
from dataset_collector import collect_dataset, compute_sample_weights
from eml_distiller import (EMLPolicy, evolve_formula, evolve_ingame,
                           make_basis, score_episodes, split_groups,
                           _standardize_per_group, DEPTH_PENALTIES)
from pipeline import evaluate_holdout
from emlcore import static_metrics, pool_slice
from emlcore.dagger import run_dagger, beta_schedule

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results', 'dagger')

# DAgger rollouts get their own env-seed stream, disjoint from TRAIN (1000..),
# VAL (3000..), TEST (5000..) and the BC dataset (dataset_collector seed=0).
DAGGER_SEED_BASE = 9_000_000


# ── Collection: mixture rollout, oracle labels on every legal placement ──────

def collect_mixture(oracle, tree, beta: float, *, n_episodes: int,
                    seed_base: int, max_placements: int, rng, device,
                    group_offset: int = 0):
    """
    Same rows as dataset_collector.collect_dataset — (26 features, oracle logit,
    group) for every legal placement of every visited step — but the TRAJECTORY
    follows beta*oracle + (1-beta)*formula, flipped per step.

    A fresh env per episode, seeded from seed_base: the formula's games are the
    ones whose boards we need labelled, and they must not overlap any pool.
    """
    policy = EMLPolicy(tree) if tree is not None else None
    feats, logits_out, groups = [], [], []
    gid = group_offset
    oracle.eval()
    for ep in range(n_episodes):
        env = TetrisEnv(seed=seed_base + ep)
        obs = env.reset()
        steps = 0
        while not env.done and steps < max_placements:
            mask = obs['mask']
            legal = np.flatnonzero(mask > 0)
            if len(legal) == 0:
                break
            g, sc, af, m = obs_to_tensors(obs, device)
            with torch.no_grad():
                z = oracle.get_logits(g, sc, af, m)[0].cpu().numpy()
            nxt = obs['scalars'][7:14]
            for a in legal:
                feats.append(np.concatenate([obs['afeats'][a], nxt]))
                logits_out.append(float(z[a]))
                groups.append(gid)
            gid += 1
            if policy is None or rng.random() < beta:
                action = int(legal[np.argmax(z[legal])])
            else:
                action = policy.choose(mask, obs['afeats'], obs['scalars'])
            obs, _, _, _ = env.step_placement(action)
            steps += 1
    return (np.asarray(feats, dtype=np.float32),
            np.asarray(logits_out, dtype=np.float32),
            np.asarray(groups, dtype=np.int64))


def n_groups(chunk) -> int:
    return int(len(np.unique(chunk[2]))) if len(chunk[2]) else 0


# ── Fit: the DATA phase of eml_distiller.distill, on a list of chunks ────────

FIT_INFO: dict[int, dict] = {}


def fit_data_phase(chunks, warm, *, generations, population, dp_name,
                   basis_name=None):
    X = np.concatenate([c[0] for c in chunks]).astype(np.float64)
    z = np.concatenate([c[1] for c in chunks])
    g = np.concatenate([c[2] for c in chunks])
    w = compute_sample_weights(z, g)
    y = _standardize_per_group(z.astype(np.float64), g)
    tr, va = split_groups(g, config.EML_DATA_VAL_FRACTION,
                          np.random.default_rng(random.randrange(2 ** 63)))
    val = (X[va], y[va], w[va]) if va.any() else None
    res = evolve_formula(
        X[tr], y[tr], w[tr],
        generations=generations, population_size=population,
        elitism=config.EML_ELITISM, patience=config.EML_PATIENCE,
        depth_penalty=DEPTH_PENALTIES[dp_name], val_data=val,
        basis=make_basis(basis_name),
        init_trees=[warm] if warm is not None else None)
    t = res['best_tree']
    FIT_INFO[id(t)] = {'data_val_fitness': res['best_val_fitness'],
                       'nodes': t.size(), 'depth': t.depth(),
                       'generations_run': len(res['history']),
                       'top1_agree_val': top1_agreement(t, val) if val else None}
    return t


def top1_agreement(tree, val) -> float:
    """
    Fraction of VAL steps where the formula's argmax placement is the oracle's.

    The quantity DAgger's guarantee is about (per-step imitation error under
    the student's own state distribution), so it is worth reporting next to
    lines — the regression fitness is not it.
    """
    X, y, _ = val
    # Groups are not passed with val; recover them from the per-group z-score
    # boundaries is fragile, so the caller-side split keeps them contiguous and
    # a group is a maximal run of rows sharing the same standardized ranking.
    return float('nan') if X is None else _top1(tree, X, y)


def _top1(tree, X, y):
    # y was standardized per group: a group boundary is where the running row
    # index resets, which we cannot see here. Used only through
    # top1_agreement_groups below.
    return float('nan')


def top1_agreement_groups(tree, chunk) -> float:
    """Top-1 agreement on a labelled chunk (features, logits, groups)."""
    X, z, g = chunk
    if len(X) == 0:
        return float('nan')
    with np.errstate(all='ignore'):
        pred = np.broadcast_to(tree.evaluate_batch(X.astype(np.float64)),
                               z.shape)
    pred = np.where(np.isfinite(pred), pred, -np.inf)
    bounds = np.flatnonzero(np.diff(g)) + 1
    hits = 0
    n = 0
    start = 0
    for end in list(bounds) + [len(g)]:
        hits += int(np.argmax(pred[start:end]) == np.argmax(z[start:end]))
        n += 1
        start = end
    return hits / max(1, n)


# ── Experiment ───────────────────────────────────────────────────────────────

def _val(tree):
    seeds = pool_slice(config.EML_SEED_POOL_VAL, config.EML_VAL_GAMES)
    r = score_episodes(tree, seeds, config.EML_INGAME_MAX_PLACEMENTS)
    return {'score': r['fitness'], 'avg_lines': r['avg_lines'],
            'std_lines': r['std_lines'], 'avg_steps': r['avg_steps']}


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        if new:
            w.writeheader()
        w.writerows(rows)


def run_seed(ga_seed: int, *, oracle, device, d0, n_iters: int,
             beta_kind: str, beta_p: float, dagger_episodes: int,
             generations: int, population: int, dp_name: str,
             game_generations: int, game_population: int, test_games: int,
             probe, out_dir: str, tag: str = '', verbose=True):
    random.seed(ga_seed)
    np.random.seed(ga_seed)
    rng = np.random.default_rng(ga_seed)
    counter = {'gid': int(d0[2].max()) + 1}
    t_start = time.perf_counter()

    def collect(student, beta, it):
        ch = collect_mixture(
            oracle, student, beta, n_episodes=dagger_episodes,
            seed_base=DAGGER_SEED_BASE + ga_seed * 10_000 + it * 100,
            max_placements=config.EML_DATA_MAX_PLACEMENTS, rng=rng,
            device=device, group_offset=counter['gid'])
        if len(ch[2]):
            counter['gid'] = int(ch[2].max()) + 1
        return ch

    def fit(chunks, warm):
        return fit_data_phase(chunks, warm, generations=generations,
                              population=population, dp_name=dp_name)

    def info(t):
        d = dict(FIT_INFO.get(id(t), {}))
        d.pop('top1_agree_val', None)
        # Imitation error on boards the formula itself reaches (probe set,
        # collected once from the BC student) and on the oracle's own boards.
        d['top1_oracle_states'] = top1_agreement_groups(t, probe['oracle'])
        return d

    def on_iter(h):
        if verbose:
            m = h.metrics
            print(f"  [seed {ga_seed}] it {h.iteration} beta={h.beta:.2f} "
                  f"+{h.n_new:>5} -> {h.n_total:>6} steps | VAL "
                  f"{m['avg_lines']:6.2f}±{m['std_lines']:.2f} lines "
                  f"({m['avg_steps']:.0f} pl) | top1 "
                  f"{h.fit_info.get('top1_oracle_states', 0):.3f} | "
                  f"S{h.fit_info.get('nodes')} | fit {h.fit_seconds:.0f}s "
                  f"coll {h.collect_seconds:.0f}s", flush=True)

    res = run_dagger(collect, fit, _val, n_iters=n_iters,
                     beta=beta_schedule(beta_kind, p=beta_p),
                     initial_data=d0, size_fn=n_groups,
                     fit_info_fn=info, on_iter=on_iter)
    t_dagger = time.perf_counter() - t_start

    iter_rows = [{'task': 'tetris', 'ga_seed': ga_seed, 'tag': tag,
                  'beta_kind': beta_kind, **r} for r in res.table()]
    _write_csv(os.path.join(out_dir, 'dagger_iterations.csv'), iter_rows)

    bc = res.students[0]
    dag = res.best_student
    students = {'BC': (bc, {'best_iter': 0}),
                'DAgger': (dag, {'best_iter': res.best_iteration})}

    if game_generations > 0:
        for name, seed_tree in (('BC+GAME', bc), ('DAgger+GAME', dag)):
            random.seed(ga_seed)
            np.random.seed(ga_seed)
            t0 = time.perf_counter()
            j = evolve_ingame(seed_tree, generations=game_generations,
                              population_size=game_population, verbose=False)
            tree = j['best_tree']
            # Same guard as distill(): GAME may not end below its seed on VAL.
            base = _val(seed_tree)
            if j['val_result']['avg_lines'] < base['avg_lines']:
                tree = seed_tree
            students[name] = (tree, {'game_s': round(time.perf_counter() - t0)})

    rows, ep_rows = [], []
    for name, (tree, extra) in students.items():
        h = evaluate_holdout(tree, oracle, device, n_games=test_games)
        st = static_metrics(tree)
        rows.append({
            'task': 'tetris', 'method': name, 'ga_seed': ga_seed, 'tag': tag,
            'test_lines': h['eml_lines'], 'test_std': h['eml_std'],
            'oracle_lines': h['oracle_lines'], 'pct_oracle': h['ratio_pct'],
            'paired_diff': h['paired_diff_mean'],
            'test_placements': h['eml']['avg_steps'],
            'top1_oracle_states': top1_agreement_groups(tree, probe['oracle']),
            'nodes': st['nodes'], 'depth': st['depth'],
            'n_transcendental': st['n_transcendental'],
            'dagger_s': round(t_dagger), 'formula': tree.to_string(), **extra})
        for i, s in enumerate(h['seeds']):
            ep_rows.append({'task': 'tetris', 'method': name,
                            'ga_seed': ga_seed, 'tag': tag, 'env_seed': s,
                            'lines': h['eml']['lines'][i],
                            'placements': h['eml']['steps'][i],
                            'oracle_lines': h['oracle']['lines'][i]})
        storage.save_formulas([tree], path=os.path.join(
            out_dir, f'formula_{tag or "run"}_{name}_seed{ga_seed}.json'),
            meta={'method': name, 'ga_seed': ga_seed, **rows[-1]})
        if verbose:
            print(f"  [seed {ga_seed}] {name:<12} TEST {h['eml_lines']:7.2f}"
                  f"±{h['eml_std']:.2f} lines = {h['ratio_pct']:5.1f}% of "
                  f"oracle {h['oracle_lines']:.2f} | {st['nodes']} nodes",
                  flush=True)
    _write_csv(os.path.join(out_dir, 'dagger_summary.csv'), rows)
    _write_csv(os.path.join(out_dir, 'dagger_episodes.csv'), ep_rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--beta', default='exp', choices=['exp', 'linear', 'zero'])
    ap.add_argument('--beta-p', type=float, default=0.5)
    ap.add_argument('--bc-episodes', type=int, default=config.EML_DATASET_EPISODES)
    ap.add_argument('--dagger-episodes', type=int, default=10)
    ap.add_argument('--generations', type=int, default=config.EML_GENERATIONS)
    ap.add_argument('--population', type=int, default=config.EML_POPULATION)
    ap.add_argument('--game-generations', type=int, default=15)
    ap.add_argument('--game-population', type=int, default=30)
    ap.add_argument('--test-games', type=int, default=config.EML_TEST_GAMES)
    ap.add_argument('--depth-penalty', default='medium')
    ap.add_argument('--oracle', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=RESULTS)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = get_device()
    oracle = TetrisCNN().to(device)
    meta = storage.load_oracle(oracle, path=a.oracle, device=device)
    oracle.eval()
    print(f"oracle: {meta}", flush=True)

    # D_0: the same BC dataset the pipeline uses (seed 0 stream), shared by
    # every GA seed so the seeds differ only in the GA.
    d0_path = os.path.join(a.out, f'd0_{a.bc_episodes}ep.npz')
    if os.path.exists(d0_path):
        d = np.load(d0_path)
        d0 = (d['features'], d['logits'], d['groups'])
    else:
        d0 = collect_dataset(oracle, n_episodes=a.bc_episodes, device=device,
                             verbose=False)
        np.savez_compressed(d0_path, features=d0[0], logits=d0[1],
                            groups=d0[2])
    probe = {'oracle': collect_mixture(
        oracle, None, 1.0, n_episodes=5, seed_base=DAGGER_SEED_BASE - 1000,
        max_placements=200, rng=np.random.default_rng(0), device=device)}
    print(f"D0: {len(d0[0]):,} rows / {n_groups(d0):,} steps", flush=True)

    with open(os.path.join(a.out, f'dagger_meta{("_" + a.tag) if a.tag else ""}.json'),
              'w') as fh:
        json.dump({**vars(a), 'oracle_meta': meta,
                   'd0_rows': int(len(d0[0])), 'd0_steps': n_groups(d0),
                   'ingame_max_placements': config.EML_INGAME_MAX_PLACEMENTS,
                   'multi_objective': config.EML_MULTI_OBJECTIVE},
                  fh, indent=2, default=str)

    for s in a.seeds:
        run_seed(s, oracle=oracle, device=device, d0=d0, n_iters=a.iters,
                 beta_kind=a.beta, beta_p=a.beta_p,
                 dagger_episodes=a.dagger_episodes,
                 generations=a.generations, population=a.population,
                 dp_name=a.depth_penalty,
                 game_generations=a.game_generations,
                 game_population=a.game_population,
                 test_games=a.test_games, probe=probe, out_dir=a.out,
                 tag=a.tag)


if __name__ == '__main__':
    main()
