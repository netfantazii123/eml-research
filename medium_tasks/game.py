"""
game.py — the GAME phase for any medium task: formulas evolve on the return.

The same seed discipline as tetris/eml_distiller.evolve_ingame:
  * every generation plays `games` TRAIN seeds at a rotating offset, the same
    seeds for the whole population (common random numbers);
  * the generation champion is re-scored on the disjoint VAL pool and the
    returned tree is the best BY VAL;
  * seed_tree (the DATA-phase formula) starts the population as itself plus
    mutants; seed_tree=None starts from random trees (the no-oracle control).
The whole population plays in one vectorised batch.
"""

from __future__ import annotations

import math
import random

import numpy as np

from emlcore import search
from medium_tasks import pools
from medium_tasks.tasks import play


def _score_population(task, trees, seeds):
    P, n = len(trees), len(seeds)
    all_seeds = [s for _ in range(P) for s in seeds]

    def score(flat):
        F = flat.shape[-1]
        X = flat.reshape(P, -1, F)
        out = np.empty(X.shape[:2])
        with np.errstate(all='ignore'):
            for i, t in enumerate(trees):
                out[i] = t.evaluate_batch(X[i])
        return out.reshape(-1)
    ret = play(task, score, all_seeds, batch=len(all_seeds))
    return ret.reshape(P, n).mean(1)


def score_tree(task, tree, seeds):
    with np.errstate(all='ignore'):
        return play(task, lambda X: tree.evaluate_batch(X), seeds)


def game_phase(task, basis, seed_tree=None, *, population=60, generations=40,
               elitism=6, games=4, val_games=16, patience=15, verbose=False):
    if seed_tree is not None:
        pop = search.seed_population(basis, population, seed_tree=seed_tree,
                                     intensity=0.7)
    else:
        pop = search.seed_population(basis, population, max_depth=3)
    val_seeds = pools.VAL[:val_games]
    val_cache: dict[str, float] = {}

    def val(t):
        k = t.key()
        if k not in val_cache:
            val_cache[k] = float(score_tree(task, t, val_seeds).mean())
        return val_cache[k]

    best_tree, best_val, stale, hist = None, -math.inf, 0, []
    if seed_tree is not None:
        best_tree, best_val = seed_tree.clone(), val(seed_tree)
    for gen in range(generations):
        seeds = pools.slice_(pools.TRAIN, games, gen * games)
        fit = _score_population(task, pop, seeds)
        fit = np.where(np.isfinite(fit), fit, -np.inf)
        order = np.argsort(-fit, kind='stable')
        pop = [pop[i] for i in order]
        fit = [float(fit[i]) for i in order]
        v = val(pop[0])
        hist.append({'gen': gen, 'train': fit[0], 'val': v})
        if verbose:
            print(f'    [GAME {task.name}] gen {gen:2d} train {fit[0]:9.2f} '
                  f'val {v:9.2f}', flush=True)
        if v > best_val:
            best_tree, best_val, stale = pop[0].clone(), v, 0
        else:
            stale += 1
            if stale >= patience:
                break
        intensity = max(0.15, 1.0 - gen / (generations * 0.8))
        fresh = None
        if seed_tree is not None:
            def fresh(b=basis, it=intensity):
                return search.mutate_tree(seed_tree, b, intensity=it)
        pop = search.breed(pop, fit, basis, size=population, elitism=elitism,
                           intensity=intensity, fresh=fresh)
    return {'best_tree': best_tree, 'best_val': best_val, 'history': hist,
            'generations_run': len(hist)}
