"""
oracle.py — the neural teacher of every medium task.

An MLP value net over afterstate features (F -> H -> H -> 1, tanh), trained by
a simple evolution strategy on the game return: truncation selection, Gaussian
mutation, elitism, the same TRAIN layouts for the whole population within a
generation (rotating across generations), and the final net chosen on the VAL
pool. No gradients, so the same trainer works for every task, and the whole
population plays in one vectorised batch.

    python -m medium_tasks.oracle --task highway
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from medium_tasks.tasks import TASKS, play           # noqa: E402
from medium_tasks import pools                         # noqa: E402

MODELS = os.path.join(HERE, 'models')


class MLP:
    """A batch of P nets; params is a dict of arrays with a leading P axis."""

    def __init__(self, params):
        self.p = params

    @staticmethod
    def init(P, F, H, rng):
        return {'W1': rng.normal(0, 1 / np.sqrt(F), (P, F, H)),
                'b1': np.zeros((P, H)),
                'W2': rng.normal(0, 1 / np.sqrt(H), (P, H, H)),
                'b2': np.zeros((P, H)),
                'W3': rng.normal(0, 1 / np.sqrt(H), (P, H)),
                'b3': np.zeros(P)}

    def __call__(self, X):
        """X: (P, N, F) -> (P, N)."""
        p = self.p
        h = np.tanh(np.einsum('pnf,pfh->pnh', X, p['W1']) + p['b1'][:, None])
        h = np.tanh(np.einsum('pnh,phk->pnk', h, p['W2']) + p['b2'][:, None])
        return np.einsum('pnh,ph->pn', h, p['W3']) + p['b3'][:, None]

    def one(self, i):
        return MLP({k: v[i:i + 1] for k, v in self.p.items()})

    def score_fn(self):
        """The policy of a single net (P = 1) as a play() score function."""
        return lambda X: self(X[None])[0]


def evaluate_population(task, params, seeds):
    P = len(params['b3'])
    net = MLP(params)
    B = len(seeds)
    all_seeds = [s for _ in range(P) for s in seeds]

    def score(flat):
        F = flat.shape[-1]
        return net(flat.reshape(P, -1, F)).reshape(-1)
    ret = play(task, score, all_seeds, batch=len(all_seeds))
    return ret.reshape(P, B).mean(1)


def train(task, *, P=200, H=16, generations=150, games=6, sigma=0.1,
          elite_frac=0.2, patience=40, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    F = len(task.feature_names)
    pop = MLP.init(P, F, H, rng)
    n_par = max(2, int(P * elite_frac))
    best, stale, hist, champs = -np.inf, 0, [], []
    t0 = time.time()
    for gen in range(generations):
        seeds = pools.slice_(pools.TRAIN, games, gen * games)
        fit = evaluate_population(task, pop, seeds)
        order = np.argsort(-fit, kind='stable')
        pop = {k: v[order] for k, v in pop.items()}
        fit = fit[order]
        champs.append({k: v[:1].copy() for k, v in pop.items()})
        hist.append({'gen': gen, 'best': float(fit[0]), 'mean': float(fit.mean())})
        if verbose and gen % 5 == 0:
            print(f'  [{task.name}] gen {gen:3d} best {fit[0]:9.2f} '
                  f'mean {fit.mean():9.2f}  {time.time() - t0:5.0f}s', flush=True)
        if fit[0] > best:
            best, stale = fit[0], 0
        else:
            stale += 1
            if stale >= patience:
                break
        parents = rng.integers(0, n_par, P - 2)
        new = {}
        for k, v in pop.items():
            child = v[parents] + sigma * rng.normal(size=v[parents].shape)
            new[k] = np.concatenate([v[:2], child])
        pop = new
    cand = {k: np.concatenate([c[k] for c in champs[-20:]]) for k in pop}
    val = evaluate_population(task, cand, pools.VAL)
    i = int(np.argmax(val))
    best_params = {k: v[i:i + 1] for k, v in cand.items()}
    return best_params, {'val': float(val[i]), 'generations_run': len(hist),
                         'elapsed_sec': time.time() - t0, 'history': hist,
                         'P': P, 'H': H, 'games': games, 'sigma': sigma,
                         'seed': seed}


def path_for(name):
    return os.path.join(MODELS, f'{name}_oracle.npz')


def save(name, params, meta):
    os.makedirs(MODELS, exist_ok=True)
    np.savez(path_for(name), **params)
    with open(os.path.splitext(path_for(name))[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=1)


def load(name) -> MLP:
    d = np.load(path_for(name))
    return MLP({k: d[k] for k in d.files})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=sorted(TASKS))
    p.add_argument('--generations', type=int, default=150)
    p.add_argument('--population', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    task = TASKS[a.task]()
    params, meta = train(task, P=a.population, generations=a.generations,
                         seed=a.seed)
    test = play(task, MLP(params).score_fn(), pools.TEST)
    meta.update({'test_mean': float(test.mean()),
                 'test_std': float(test.std(ddof=1))})
    save(a.task, params, meta)
    print(f'[{a.task}] oracle VAL {meta["val"]:.2f}  TEST {test.mean():.2f} '
          f'± {test.std(ddof=1):.2f}  ({meta["elapsed_sec"]:.0f}s)')


if __name__ == '__main__':
    main()
