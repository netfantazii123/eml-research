"""
One full-budget Flappy Bird EML run for one GA seed, scored on TEST.

The thesis pipeline (benchmark.py / eml_distiller.distill) with the oracle held
fixed: the dataset is collected once from that oracle (its seeds are fixed by
config.EML_DATA_SEED, so it is the same dataset in every process) and only the
GA seed changes. benchmark.py itself seeds nothing and retrains the GA oracle
every time, so its runs mix oracle noise with GA noise and cannot be
replicated; this script isolates the GA noise.

    python seed_study/flappy_seed.py --seed 0 --out seed_study/runs/flappy

Every seed is scored on the same 30 TEST layouts (5000 frames, as benchmark.py),
whatever its VAL score: no seed is dropped.
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.environ.get('EML_CODE_ROOT') or \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = os.path.join(ROOT, 'flappy_bird_with_ai')
sys.path.insert(0, ROOT)
sys.path.insert(0, TASK)

import numpy as np                                     # noqa: E402

import config                                          # noqa: E402
from storage import load_ga_model                      # noqa: E402
from eml_distiller import (collect_dataset, evolve_eml, EMLAgent,  # noqa: E402
                           EMLNode, pool_slice)
from benchmark import test_agent_score                 # noqa: E402

DEFAULT_ORACLE = os.path.join(TASK, 'models', 'best_ga.pt')


def _file_sha(path):
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def get_dataset(oracle_path, cache):
    # The cache is keyed by the oracle file: reusing --out with another --oracle
    # used to train silently on the old oracle's labels.
    sha = _file_sha(oracle_path)
    if os.path.exists(cache):
        d = np.load(cache)
        cached = str(d['oracle_sha']) if 'oracle_sha' in d.files else None
        if cached is not None and cached != sha:
            raise SystemExit(f'{cache} was collected with another oracle '
                             f'({cached} != {sha}); pass a new --dataset')
        return d['states'], d['targets']
    states, targets = collect_dataset(load_ga_model(oracle_path), mode='sigmoid')
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    np.savez_compressed(cache, states=states, targets=targets,
                        oracle_sha=np.array(sha))
    return states, targets


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--oracle', default=DEFAULT_ORACLE)
    p.add_argument('--dataset', default=None, help='cache .npz (default <out>/dataset.npz)')
    p.add_argument('--penalty', default='medium')
    p.add_argument('--test-games', type=int, default=config.EML_TEST_GAMES)
    p.add_argument('--basis', default=config.EML_BASIS, choices=['eml', 'classic'])
    p.add_argument('--no-oracle', action='store_true',
                   help='control: no DATA phase, GAME from random trees')
    a = p.parse_args(argv)
    config.EML_BASIS = a.basis
    a.out = os.path.abspath(a.out)   # before the chdir below
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, f'seed_{a.seed}.json')
    if os.path.exists(res_path):
        print(f'{res_path} exists, skipping')
        return
    os.chdir(TASK)   # the task code resolves models/ and results/ relative to cwd
    states, targets = get_dataset(a.oracle, os.path.abspath(
        a.dataset or os.path.join(a.out, 'dataset.npz')))

    random.seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()
    res = evolve_eml(states.astype(np.float32), targets, mode='sigmoid',
                     depth_penalty_name=a.penalty, verbose=True,
                     data_fraction=0.0 if a.no_oracle else None)
    fit_sec = time.time() - t0
    tree = res['best_tree']
    seeds = pool_slice(config.EML_SEED_POOL_TEST, a.test_games)
    scores = test_agent_score(EMLAgent(tree), seeds=seeds)
    val = res.get('val_result') or {}
    out = {
        'task': 'flappy', 'ga_seed': a.seed, 'penalty': a.penalty,
        'profile': config.FLAPPY_ENV, 'basis': a.basis,
        'no_oracle': bool(a.no_oracle),
        'max_frames': {'train_val': config.GA_MAX_FRAMES,
                       'test': config.EML_TEST_MAX_FRAMES},
        'val_games': config.EML_VAL_GAMES,
        'oracle': os.path.relpath(os.path.abspath(a.oracle), ROOT),
        'budget': {'population': config.EML_POPULATION,
                   'generations': config.EML_GENERATIONS,
                   'patience': config.EML_PATIENCE,
                   'dataset_episodes': config.EML_DATASET_EPISODES},
        'dataset_size': int(len(states)),
        'val_score': val.get('avg_score'), 'val_scores': val.get('scores'),
        'test_score': float(np.mean(scores)),
        'test_std': float(np.std(scores, ddof=1)),
        'test_per_game': [int(s) for s in scores],
        'test_seeds': [int(s) for s in seeds],
        'size': tree.size(), 'depth': tree.depth(),
        'nvars': tree.n_unique_vars(), 'formula': tree.to_string(),
        'tree': tree.to_dict(), 'stop_reason': res.get('stop_reason'),
        'generations_run': len(res.get('history') or []),
        'fit_sec': fit_sec, 'code_root': ROOT,
    }
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'seed {a.seed}: VAL {out["val_score"]}  TEST {out["test_score"]:.1f}'
          f' ± {out["test_std"]:.1f}  size {out["size"]}  {fit_sec/60:.1f} min')


if __name__ == '__main__':
    main()
