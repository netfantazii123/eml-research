"""
One full-budget Tetris EML run for one GA seed, scored on TEST.

The same run as configuration 0 of `main.py distill --batch` (base parameters,
medium depth penalty, DATA + GAME phase, random.seed(ga_seed)), but one seed per
process so seeds can run side by side, and with the TEST pool scored for EVERY
seed instead of only for the VAL winner. That last part is the point: the batch
reports the winner, and a seed that looks fine on 8 VAL games can still collapse
on TEST (seed 23 of the 2026-08-20 batch: VAL 115.9, TEST 78.1).

    python seed_study/tetris_seed.py --seed 20 --dataset tetris/results/eml_dataset.npz \
        --out seed_study/runs/tetris_A

Writes <out>/seed_<s>.json (formula, VAL, TEST per game, size) and keeps the GA
checkpoints in <out>/ckpt_<s>/ so an interrupted seed resumes with --resume.
"""
import argparse
import inspect
import json
import os
import random
import sys
import time

# EML_CODE_ROOT runs the study against another checkout of the pipeline (for
# instance the commit an earlier batch was made with); default: this repository.
ROOT = os.environ.get('EML_CODE_ROOT') or \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = os.path.join(ROOT, 'tetris')
sys.path.insert(0, ROOT)
sys.path.insert(0, TASK)

import numpy as np                                   # noqa: E402

import config                                        # noqa: E402
import storage                                       # noqa: E402
import runs as runs_mod                              # noqa: E402
from dataset_collector import load_dataset, compute_sample_weights  # noqa: E402
from eml_distiller import distill, score_episodes, pool_slice       # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True, help='GA seed')
    p.add_argument('--dataset', required=True, help='eml_dataset.npz')
    p.add_argument('--out', required=True)
    p.add_argument('--basis', default=config.EML_BASIS)
    p.add_argument('--test-games', type=int, default=config.EML_TEST_GAMES)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, f'seed_{a.seed}.json')
    if os.path.exists(res_path):
        print(f'{res_path} exists, skipping')
        return
    ckpt = os.path.join(a.out, f'ckpt_{a.seed}')

    features, logits, groups = load_dataset(a.dataset)
    weights = compute_sample_weights(logits, groups)

    # Configuration 0 of batch_distill: the clean base.
    params = {'generations': config.EML_GENERATIONS,
              'population': config.EML_POPULATION,
              'depth_penalty': 'medium',
              'max_depth': config.EML_MAX_DEPTH,
              'var_bonus': config.EML_VAR_BONUS}
    random.seed(a.seed)
    kw = {}
    if 'checkpoint_dir' in inspect.signature(distill).parameters:
        kw = {'checkpoint_dir': ckpt,
              'resume': ckpt if a.resume and os.path.isdir(ckpt) else None}
    t0 = time.time()
    result = distill(features, logits, weights, groups=groups,
                     depth_penalty_name=params['depth_penalty'],
                     data_generations=params['generations'],
                     data_population=params['population'],
                     joint=True, basis_name=a.basis, verbose=True, **kw)
    fit_sec = time.time() - t0
    tree = result['trees'][0]

    seeds = pool_slice(config.EML_SEED_POOL_TEST, a.test_games)
    test = score_episodes(tree, seeds, config.EML_INGAME_MAX_PLACEMENTS)

    fpath = os.path.join(a.out, f'formula_s{a.seed}.json')
    storage.save_formulas(result['trees'], path=fpath, meta={
        'ga_seed': a.seed, 'params': params, 'basis': a.basis,
        'val_lines': result['val_lines'], 'static': result['static'],
        'provenance': runs_mod.provenance()})
    out = {
        'task': 'tetris', 'ga_seed': a.seed, 'basis': a.basis,
        'params': params, 'dataset': os.path.abspath(a.dataset),
        'val_lines': result['val_lines'],
        'val_std': result['val_result']['std_lines'],
        'test_lines': test['avg_lines'], 'test_std': test['std_lines'],
        'test_per_game': [int(x) for x in test['lines']],
        'test_seeds': [int(s) for s in seeds],
        'size': tree.size(), 'depth': tree.depth(),
        'nvars': tree.n_unique_vars(), 'static': result['static'],
        'formula': tree.to_string(), 'fit_sec': fit_sec,
        'code_root': ROOT, 'provenance': runs_mod.provenance(),
    }
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'seed {a.seed}: VAL {out["val_lines"]:.2f}  TEST {out["test_lines"]:.2f}'
          f' ± {out["test_std"]:.2f}  size {out["size"]}  {fit_sec/60:.1f} min')


if __name__ == '__main__':
    main()
