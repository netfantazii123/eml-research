"""
dagger_distill.py — DAgger for the Flappy Bird EML formula.

Compares, on the same oracle, the same GA budget per fit and the same VAL/TEST
pools:

    BC        one supervised fit on oracle trajectories (DAgger iteration 0)
    DAgger    the same fit, repeated on the aggregated dataset of states the
              CURRENT formula visits, each labelled by the oracle
    EML-GAME  the existing pipeline (eml_distiller.evolve_eml: a short DATA
              warm-up, then in-game fitness) — the reference the thesis uses

The supervised fit is a vectorised version of the DATA-phase fitness of
eml_distiller (1 / (MSE + dp*depth) * var bonus, sigmoid targets), so BC and
DAgger differ only in WHICH states the data comes from.

    python dagger_distill.py [--seeds 3] [--iters 6] [--quick]
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
from config import DEPTH_PENALTIES
from env import FlappyEnv
from storage import load_ga_model
from eml_distiller import (make_basis, EMLAgent, score_episodes, evolve_eml,
                           collect_dataset)
from emlcore import search, static_metrics, pool_slice
from emlcore.dagger import run_dagger, beta_schedule

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# DAgger rollouts get their own seed stream: disjoint from the BC dataset
# (EML_DATA_SEED * 100_000 + ep) and from TRAIN/VAL/TEST (1000..5063).
DAGGER_SEED_BASE = 9_000_000


# ── Oracle labels, batched ───────────────────────────────────────────────────

def oracle_logit(oracle, states: np.ndarray) -> np.ndarray:
    """Pre-sigmoid output: flap <=> logit > 0."""
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(states, dtype=np.float32))
        return oracle.fc2(torch.relu(oracle.fc1(x))).numpy().reshape(-1)


def make_target(logits: np.ndarray, scale: float) -> np.ndarray:
    """
    Regression target with the decision boundary kept at 0.5.

    The oracle sigmoid lives in [0.44, 0.56] and flaps on ~4 % of frames, so an
    MSE fit to it is won by a near-constant that never flaps — the "narrow
    targets" problem noted in eml_distiller. 0.5 + logit/scale is an affine
    rescale of the logit: EMLAgent's `> 0.5` rule then means exactly
    `oracle logit > 0`, and the target has unit spread. scale is fixed once
    per run (std of the BC logits) so all DAgger iterations share it.
    """
    return 0.5 + logits / scale


def collect_mixture(oracle, tree, beta: float, *, n_episodes: int,
                    seed_base: int, max_frames: int, rng) -> tuple:
    """
    Roll beta*oracle + (1-beta)*formula out and label every state by the oracle.

    The coin is flipped per step, as in Ross et al. tree=None -> pure oracle.
    """
    agent = EMLAgent(tree) if tree is not None else None
    states, oracle_act = [], []
    for ep in range(n_episodes):
        env = FlappyEnv(seed=seed_base + ep)
        s = env.reset()
        f = 0
        while not env.done and f < max_frames:
            states.append(s.copy())
            o = oracle.get_action(s)
            oracle_act.append(o)
            if agent is None or rng.random() < beta:
                a = o
            else:
                a = agent.get_action(s)
            s, _, _ = env.step(a)
            f += 1
    X = np.asarray(states, dtype=np.float32)
    return X, oracle_logit(oracle, X)


# ── Supervised fit (vectorised DATA-phase fitness) ───────────────────────────

FIT_INFO: dict[int, dict] = {}
LOSS = 'balacc'     # 'balacc' (default) or 'mse' (the DATA-phase regression)

def class_weights(y: np.ndarray) -> np.ndarray:
    """Balance flap / no-flap so the rare flap frames are not noise."""
    pos = y > 0.5
    p = pos.mean()
    if p in (0.0, 1.0):
        return np.ones(len(y))
    return np.where(pos, 0.5 / p, 0.5 / (1 - p))


def _fitness(tree, X, y, dp, w=None) -> float:
    n_vars = tree.n_unique_vars()
    if n_vars == 0:
        return 1e-6
    with np.errstate(all='ignore'):
        try:
            pred = tree.evaluate_batch(X)
        except (OverflowError, ValueError, FloatingPointError):
            return 1e-6
    pred = np.broadcast_to(pred, y.shape)
    if not np.all(np.isfinite(pred)):
        return 1e-6
    var_bonus = 1.0 + 0.05 * n_vars
    if LOSS == 'mse':
        mse = float(np.average((pred - y) ** 2, weights=w))
        return var_bonus / (mse + dp * tree.depth() + 1e-8)
    # Balanced accuracy of the ACTION. Imitation is a classification problem
    # here: the formula only matters through `> 0.5`. A regression fit to the
    # logit is dominated by the far-from-boundary frames — on the BC data even
    # least squares, R^2 = 0.81, reaches only 0.65 balanced accuracy, while a
    # linear classifier reaches 0.94. The depth penalty is scaled down 10x
    # against the regression form because accuracy lives in [0, 1].
    act = pred > 0.5
    t = y > 0.5
    tpr = (act & t).sum() / max(1, t.sum())
    tnr = (~act & ~t).sum() / max(1, (~t).sum())
    return max(1e-6, (tpr + tnr) / 2 * var_bonus - 0.1 * dp * tree.depth())


def agreement(tree, X, y) -> float:
    """Balanced accuracy of the formula's action against the oracle's."""
    with np.errstate(all='ignore'):
        pred = np.broadcast_to(tree.evaluate_batch(X), y.shape)
    act = np.where(np.isfinite(pred), pred > 0.5, False)
    t = y > 0.5
    tpr = (act & t).sum() / max(1, t.sum())
    tnr = (~act & ~t).sum() / max(1, (~t).sum())
    return float((tpr + tnr) / 2)


def fit_supervised(X, y, warm=None, *, generations: int, population: int,
                   dp: float, subsample: int = 5000, patience: int = 60,
                   rng=None, row_weights=None) -> object:
    """
    GA symbolic regression to the oracle sigmoid. warm -> half the initial
    population are mutants of the previous student (the rest random, so a bad
    warm start cannot take the whole run hostage).
    """
    basis = make_basis()
    rng = rng or np.random.default_rng(random.randrange(2 ** 63))
    n = len(X)
    # Row-level holdout is fine here: rows are single states, there are no
    # groups, and the holdout only picks the hall-of-fame champion.
    perm = rng.permutation(n)
    n_val = max(1, int(0.15 * n))
    va, tr = perm[:n_val], perm[n_val:]
    Xv, yv = X[va].astype(np.float64), y[va].astype(np.float64)
    Xt, yt = X[tr].astype(np.float64), y[tr].astype(np.float64)

    if len(Xv) > 20000:
        k = rng.choice(len(Xv), 20000, replace=False)
        Xv, yv = Xv[k], yv[k]
    wt, wv = class_weights(yt), class_weights(yv)
    if row_weights is not None:
        # Sampling weights for the fitness subsample (chunk balance): rows are
        # drawn in proportion, so the fitness itself stays unweighted by them.
        pr = np.asarray(row_weights, dtype=np.float64)[tr]
        pr = pr / pr.sum()
    else:
        pr = None

    pop = search.seed_population(basis, population, max_depth=3)
    if warm is not None:
        warm_pop = search.seed_population(basis, population // 2,
                                          seed_tree=warm, intensity=0.5)
        pop = warm_pop + pop[:population - len(warm_pop)]

    best, best_val, stall = None, -np.inf, 0
    for gen in range(generations):
        intensity = max(0.1, 1.0 - gen / (generations * 0.7))
        if len(Xt) > subsample:
            idx = rng.choice(len(Xt), subsample, replace=pr is not None, p=pr)
            Xs, ys, ws = Xt[idx], yt[idx], wt[idx]
        else:
            Xs, ys, ws = Xt, yt, wt
        fit = [_fitness(t, Xs, ys, dp, ws) for t in pop]
        order = np.argsort(fit)[::-1]
        pop = [pop[i] for i in order]
        fit = [fit[i] for i in order]
        v = _fitness(pop[0], Xv, yv, dp, wv)
        if v > best_val:
            best_val, best, stall = v, pop[0].clone(), 0
        else:
            stall += 1
        if patience and stall >= patience:
            break
        pop = search.breed(pop, fit, basis, size=population,
                           elitism=config.EML_ELITISM, intensity=intensity,
                           p_mutate=0.45, p_cross=0.75, p_cross_mutate=0.3,
                           tournament_k=5)
    # Side table, not attributes: Node has __slots__.
    FIT_INFO[id(best)] = {'fit_val': best_val,
                          'agree': agreement(best, Xv, yv),
                          'nodes': best.size()}
    return best


# ── Experiment ───────────────────────────────────────────────────────────────

def _val(tree, max_frames):
    seeds = pool_slice(config.EML_SEED_POOL_VAL, config.EML_VAL_GAMES)
    r = score_episodes(tree, seeds, max_frames)
    return {'score': r['fitness'], 'avg_pipes': r['avg_score'],
            'avg_frames': r['avg_frames'],
            'std_pipes': r['std_score']}


def _test(tree, max_frames, n_games):
    seeds = pool_slice(config.EML_SEED_POOL_TEST, n_games)
    return score_episodes(tree, seeds, max_frames)


def run(seeds=(0, 1, 2), n_iters=6, generations=200, population=300,
        bc_episodes=60, dagger_episodes=30, max_frames=None, dp_name='medium',
        test_games=None, with_game_baseline=True, out_dir=RESULTS,
        oracle_path=None, chunk_balance=True, game_generations=150,
        game_population=200, verbose=True):
    max_frames = max_frames or config.GA_MAX_FRAMES
    test_games = test_games or config.EML_TEST_GAMES
    dp = DEPTH_PENALTIES[dp_name]
    oracle_path = oracle_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'models', 'best_ga.pt')
    oracle = load_ga_model(oracle_path)
    os.makedirs(out_dir, exist_ok=True)

    # Oracle reference on TEST (same seeds and frame cap as every student).
    test_seeds = pool_slice(config.EML_SEED_POOL_TEST, test_games)
    o_scores = []
    for s in test_seeds:
        env = FlappyEnv(seed=int(s)); st = env.reset(); f = 0
        while not env.done and f < max_frames:
            st, _, _ = env.step(oracle.get_action(st)); f += 1
        o_scores.append(env.score)
    oracle_avg = float(np.mean(o_scores))
    if verbose:
        print(f"oracle TEST: {oracle_avg:.2f} pipes ({test_games} games, "
              f"cap {max_frames} frames)")

    rows, iter_rows, episode_rows = [], [], []

    def _record(method, ga_seed, tree, extra=None, elapsed=None):
        t = _test(tree, max_frames, test_games)
        st = static_metrics(tree)
        row = {'task': 'flappy', 'method': method, 'ga_seed': ga_seed,
               'test_pipes': t['avg_score'], 'test_std': t['std_score'],
               'pct_oracle': 100 * t['avg_score'] / oracle_avg
               if oracle_avg > 0 else 0.0,
               'nodes': st['nodes'], 'depth': st['depth'],
               'n_transcendental': st['n_transcendental'],
               'elapsed_s': round(elapsed or 0, 1),
               'formula': tree.to_string(), **(extra or {})}
        rows.append(row)
        for i, (sc, fr) in enumerate(zip(t['scores'], t['frames'])):
            episode_rows.append({'task': 'flappy', 'method': method,
                                 'ga_seed': ga_seed, 'env_seed': t['seeds'][i],
                                 'pipes': sc, 'frames': fr})
        if verbose:
            print(f"  [{method:<10} seed {ga_seed}] TEST {t['avg_score']:6.2f}"
                  f"±{t['std_score']:.2f} pipes ({row['pct_oracle']:.0f}%), "
                  f"{st['nodes']} nodes")
        return row

    for gs in seeds:
        random.seed(gs); np.random.seed(gs)
        rng = np.random.default_rng(gs)
        t0 = time.perf_counter()

        def collect(student, beta, it):
            return collect_mixture(
                oracle, student, beta, n_episodes=dagger_episodes,
                seed_base=DAGGER_SEED_BASE + gs * 10_000 + it * 100,
                max_frames=max_frames, rng=rng)

        def fit(chunks, warm):
            X = np.concatenate([c[0] for c in chunks])
            y = make_target(np.concatenate([c[1] for c in chunks]), scale)
            # Every iteration's chunk gets the same total weight. Without it
            # the 90k oracle frames of D_0 drown the ~2k frames a failing
            # student produces, and DAgger degenerates into BC.
            rw = (np.concatenate([np.full(len(c[0]), 1.0 / len(c[0]))
                                  for c in chunks]) if chunk_balance else None)
            return fit_supervised(X, y, warm, generations=generations,
                                  population=population, dp=dp, rng=rng,
                                  row_weights=rw)

        # D_0 = the oracle's own trajectories (the BC dataset).
        X0, _ = collect_dataset(oracle, n_episodes=bc_episodes,
                                mode='sigmoid', max_frames=max_frames)
        d0 = (X0.astype(np.float32), oracle_logit(oracle, X0))
        scale = float(np.std(d0[1])) + 1e-8

        def on_iter(h):
            if verbose:
                print(f"    it {h.iteration} beta={h.beta:.2f} "
                      f"+{h.n_new:>7,} -> {h.n_total:>8,} rows | "
                      f"VAL {h.metrics['avg_pipes']:6.2f} pipes "
                      f"({h.metrics['avg_frames']:.0f} fr) | "
                      f"agree {h.fit_info.get('agree', 0):.3f} | "
                      f"fit {h.fit_seconds:.0f}s")

        res = run_dagger(
            collect, fit, lambda t: _val(t, max_frames),
            n_iters=n_iters, beta=beta_schedule('exp', p=0.5),
            initial_data=d0, size_fn=lambda c: len(c[0]),
            fit_info_fn=lambda t: FIT_INFO.get(id(t), {}),
            on_iter=on_iter)
        elapsed = time.perf_counter() - t0
        for r in res.table():
            iter_rows.append({'task': 'flappy', 'ga_seed': gs, **r})

        # BC = iteration 0 (beta=1, oracle data only).
        _record('BC', gs, res.students[0],
                {'iterations': 1, 'rows': res.history[0].n_total})
        _record('DAgger', gs, res.best_student,
                {'iterations': n_iters, 'best_iter': res.best_iteration,
                 'rows': res.history[-1].n_total}, elapsed)

        if with_game_baseline:
            # BC+GAME / DAgger+GAME: the same GAME pipeline, warm-started from
            # the supervised formula — the comparison that matters in Tetris.
            for name, seed_tree in (('BC+GAME', res.students[0]),
                                    ('DAgger+GAME', res.best_student)):
                random.seed(gs); np.random.seed(gs)
                t0 = time.perf_counter()
                g = evolve_eml(X0, 1 / (1 + np.exp(-d0[1])), mode='sigmoid',
                               depth_penalty_name=dp_name,
                               generations=game_generations,
                               population_size=game_population,
                               init_trees=[seed_tree], verbose=False)
                _record(name, gs, g['best_tree'], {'iterations': 1},
                        time.perf_counter() - t0)
            random.seed(gs); np.random.seed(gs)
            t0 = time.perf_counter()
            # The reference pipeline exactly as the thesis runs it: sigmoid
            # targets, its own DATA warm-up and GAME phase.
            g = evolve_eml(X0, 1 / (1 + np.exp(-d0[1])), mode='sigmoid', depth_penalty_name=dp_name,
                           generations=game_generations,
                           population_size=game_population, verbose=False)
            _record('EML-GAME', gs, g['best_tree'], {'iterations': 1},
                    time.perf_counter() - t0)

    _write_csv(os.path.join(out_dir, 'dagger_summary.csv'), rows)
    _write_csv(os.path.join(out_dir, 'dagger_iterations.csv'), iter_rows)
    _write_csv(os.path.join(out_dir, 'dagger_episodes.csv'), episode_rows)
    with open(os.path.join(out_dir, 'dagger_meta.json'), 'w') as fh:
        json.dump({'oracle_test_pipes': oracle_avg, 'test_games': test_games,
                   'max_frames': max_frames, 'seeds': list(seeds),
                   'n_iters': n_iters, 'generations': generations,
                   'population': population, 'game_generations': game_generations,
                   'game_population': game_population, 'loss': LOSS,
                   'chunk_balance': chunk_balance, 'bc_episodes': bc_episodes,
                   'dagger_episodes': dagger_episodes, 'depth_penalty': dp_name,
                   'beta': 'exp p=0.5'}, fh, indent=2)
    return rows, iter_rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--generations', type=int, default=200)
    ap.add_argument('--population', type=int, default=300)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-game', action='store_true')
    a = ap.parse_args()
    if a.quick:
        run(seeds=(0,), n_iters=3, generations=30, population=60,
            bc_episodes=10, dagger_episodes=5, test_games=5,
            with_game_baseline=not a.no_game,
            out_dir=os.path.join(RESULTS, 'quick'))
    else:
        run(seeds=tuple(range(a.seeds)), n_iters=a.iters,
            generations=a.generations, population=a.population,
            with_game_baseline=not a.no_game)
