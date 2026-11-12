"""
train_oracle_vec.py — train the GANet oracle on any difficulty profile, fast.

The same neuroevolution as ga_net.train_ga (4-16-1 net, tournament of 3,
per-weight mutation, elitism), with three differences that matter for the
'ramp' profile, where the game ends on its own and episodes are long:

* the whole population flies through one pipe stream at once (vec_env), so a
  generation costs one simulation per game, not one per bird;
* fitness is the mean over EML_INGAME_GAMES seeded TRAIN layouts, the same
  layouts for every bird in a generation (common random numbers), rotating
  across generations; train_ga gave each bird one unseeded game;
* the returned net is chosen on the VAL pool (all 32 layouts), never on TEST.

Saves a regular GANet state_dict, so every existing loader reads it.

    EML_FLAPPY_ENV=ramp python train_oracle_vec.py --out models/ramp_ga.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config                                  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
from emlcore.metrics import pool_slice          # noqa: E402
from vec_env import play_population            # noqa: E402

SHAPES = {'W1': (16, 4), 'b1': (16,), 'W2': (16,), 'b2': ()}


def init_population(n, rng):
    """torch.nn.Linear's default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    return {
        'W1': rng.uniform(-0.5, 0.5, (n, 16, 4)),
        'b1': rng.uniform(-0.5, 0.5, (n, 16)),
        'W2': rng.uniform(-0.25, 0.25, (n, 16)),
        'b2': rng.uniform(-0.25, 0.25, (n,)),
    }


def make_policy(pop):
    W1 = pop['W1'].astype(np.float32)
    b1 = pop['b1'].astype(np.float32)
    W2 = pop['W2'].astype(np.float32)
    b2 = pop['b2'].astype(np.float32)

    def policy(st, alive_idx):
        h = np.maximum(np.einsum('nij,nj->ni', W1[alive_idx], st)
                       + b1[alive_idx], 0)
        return (np.einsum('ni,ni->n', W2[alive_idx], h) + b2[alive_idx]) > 0
    return policy


def fitness(pop, seeds, max_frames):
    n = len(pop['b2'])
    tot = np.zeros(n)
    scores = np.zeros(n)
    pol = make_policy(pop)
    for s in seeds:
        sc, fr = play_population(pol, n, int(s), max_frames=max_frames)
        tot += sc * 100.0 + fr
        scores += sc
    return tot / len(seeds), scores / len(seeds)


def take(pop, idx):
    return {k: v[idx].copy() for k, v in pop.items()}


def to_state_dict(pop, i):
    import torch
    return {'fc1.weight': torch.tensor(pop['W1'][i], dtype=torch.float32),
            'fc1.bias': torch.tensor(pop['b1'][i], dtype=torch.float32),
            'fc2.weight': torch.tensor(pop['W2'][i][None, :], dtype=torch.float32),
            'fc2.bias': torch.tensor([pop['b2'][i]], dtype=torch.float32)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--population', type=int, default=config.GA_POPULATION)
    p.add_argument('--generations', type=int, default=200)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--games', type=int, default=config.EML_INGAME_GAMES)
    p.add_argument('--max-frames', type=int, default=200_000,
                   help='safety cap only; ramp games end long before it')
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    pop = init_population(a.population, rng)
    elit, rate, std = config.GA_ELITISM, config.GA_MUTATION_RATE, config.GA_MUTATION_STD
    best_fit, stale, hist = -np.inf, 0, []
    hof = []                      # (gen, params of the generation champion)
    t0 = time.time()
    train = config.EML_SEED_POOL_TRAIN
    for gen in range(a.generations):
        seeds = pool_slice(train, a.games, offset=gen * a.games)
        fit, sc = fitness(pop, seeds, a.max_frames)
        order = np.argsort(-fit, kind='stable')
        pop, fit, sc = take(pop, order), fit[order], sc[order]
        hof.append((gen, take(pop, np.arange(min(3, len(fit))))))
        hist.append({'gen': gen, 'best_fit': float(fit[0]),
                     'best_score': float(sc[0]), 'mean_score': float(sc.mean())})
        print(f'gen {gen:3d}  best {sc[0]:7.1f} pipes  mean {sc.mean():6.1f}  '
              f'{time.time() - t0:6.0f}s', flush=True)
        # Plateau on the generation champion's TRAIN fitness is noisy (the
        # layouts rotate), so patience counts generations without a new
        # all-time best.
        if fit[0] > best_fit:
            best_fit, stale = fit[0], 0
        else:
            stale += 1
            if stale >= a.patience:
                break
        n = a.population
        children = {k: np.empty_like(v) for k, v in pop.items()}
        for k in pop:
            children[k][:elit] = pop[k][:elit]
        cand = rng.integers(0, n, size=(n - elit, 3))
        parent = cand.min(axis=1)                 # sorted: lower index = fitter
        for k, v in pop.items():
            c = v[parent].copy()
            mask = rng.random(c.shape) < rate
            c += mask * rng.normal(0.0, std, c.shape)
            children[k][elit:] = c
        pop = children

    # Choose on VAL: the champions of the last 20 generations, all 32 layouts.
    cands = [h for _, h in hof[-20:]]
    flat = {k: np.concatenate([c[k] for c in cands]) for k in pop}
    val_fit, val_sc = fitness(flat, config.EML_SEED_POOL_VAL, a.max_frames)
    i = int(np.argmax(val_fit))
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(to_state_dict(flat, i), a.out)
    meta = {'profile': config.FLAPPY_ENV, 'seed': a.seed,
            'population': a.population, 'generations_run': len(hist),
            'games_per_eval': a.games, 'val_pipes': float(val_sc[i]),
            'elapsed_sec': time.time() - t0, 'history': hist}
    with open(os.path.splitext(a.out)[0] + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'saved {a.out}: VAL {val_sc[i]:.1f} pipes over '
          f'{len(config.EML_SEED_POOL_VAL)} layouts')


if __name__ == '__main__':
    main()
