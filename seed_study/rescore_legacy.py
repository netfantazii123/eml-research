"""
Re-score the formulas of earlier runs on today's TEST pool.

The 2026-07-12 and 2026-08-20 Tetris batches and the 2026-05-29 / 2026-08-19
Flappy benchmark runs were made before every seed was scored on TEST (the batch
scored only its VAL winner; benchmark.py scored each formula, but on whatever
pool it had then). Their formula files are copied into results/legacy/; this
plays every one of them on the same 30 TEST layouts the new runs use and writes
results/legacy/<run>/rescored.json, one entry per formula, none dropped.

    python seed_study/rescore_legacy.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEGACY = os.path.join(HERE, 'results', 'legacy')


def _task_path(task):
    for t in ('tetris', 'flappy_bird_with_ai'):
        p = os.path.join(ROOT, t)
        if p in sys.path:
            sys.path.remove(p)
    for m in ('config', 'storage', 'eml_distiller', 'env', 'benchmark'):
        sys.modules.pop(m, None)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, task))


def tetris(run):
    _task_path('tetris')
    import config
    import storage
    from eml_distiller import EMLNode, score_episodes, pool_slice
    seeds = pool_slice(config.EML_SEED_POOL_TEST, config.EML_TEST_GAMES)
    out = []
    for p in sorted(glob.glob(os.path.join(LEGACY, run, 'formula_*.json'))):
        trees, meta = storage.load_formulas(EMLNode, p)
        r = score_episodes(trees[0], seeds, config.EML_INGAME_MAX_PLACEMENTS)
        out.append({'file': os.path.basename(p), 'ga_seed': meta.get('ga_seed'),
                    'params': meta.get('params'), 'val_lines': meta.get('val_lines'),
                    'test_lines': r['avg_lines'], 'test_std': r['std_lines'],
                    'test_per_game': [int(x) for x in r['lines']],
                    'size': trees[0].size(), 'formula': trees[0].to_string()})
        print(run, out[-1]['file'], round(out[-1]['test_lines'], 2), flush=True)
    return out


def flappy(run):
    _task_path('flappy_bird_with_ai')
    import numpy as np
    import config
    from storage import load_eml_formula
    from benchmark import test_agent_score
    from eml_distiller import pool_slice
    seeds = pool_slice(config.EML_SEED_POOL_TEST, config.EML_TEST_GAMES)
    out = []
    for p in sorted(glob.glob(os.path.join(LEGACY, run, 'eml-*.json'))):
        d = load_eml_formula(p)
        sc = test_agent_score(d['agent'], seeds=seeds)
        out.append({'file': os.path.basename(p),
                    'penalty': d.get('depth_penalty'), 'oracle': d.get('oracle_ref'),
                    'stored_test_avg': (float(np.mean(d['test_scores']))
                                        if d.get('test_scores') else None),
                    'test_score': float(np.mean(sc)),
                    'test_std': float(np.std(sc, ddof=1)),
                    'test_per_game': [int(s) for s in sc],
                    'size': d['tree_node'].size(), 'formula': d['tree_node'].to_string()})
        print(run, out[-1]['file'], round(out[-1]['test_score'], 2), flush=True)
    return out


def main():
    for run, fn in (('tetris_2026-08-20_batch', tetris),
                    ('tetris_2026-07-12_batch', tetris),
                    ('flappy', flappy)):
        res = fn(run)
        with open(os.path.join(LEGACY, run, 'rescored.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
