"""
run.py — every distiller on one medium task, one seed per call.

For seed s the oracle's demonstrations come from DATA episodes
[s*E, (s+1)*E) of the DATA pool, so every method of that seed is fit on the
same data, and data noise enters the seed-to-seed spread for every method
(not only for the GA ones). Every method is scored on the same TEST episodes.

Methods
  linear, tree_d3, tree_d6, kd_mlp_h16, gplearn_sr  classical students
  eml_data, classic_gp           DATA phase only (EML / classical basis)
  eml_task                       full pipeline: DATA (EML) -> GAME phase
  classic_task                   full pipeline on the classical basis
  nooracle_task                  GAME phase from random EML trees, no oracle

    python -m medium_tasks.run --task highway --seed 0
Writes medium_tasks/results/<task>/seed_<s>.jsonl (one row per method).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from emlcore import make_basis, static_metrics                     # noqa: E402
from distill_compare.students import make_student                 # noqa: E402
from medium_tasks import oracle as O, pools                         # noqa: E402
from medium_tasks.game import game_phase, score_tree               # noqa: E402
from medium_tasks.tasks import TASKS, play                          # noqa: E402

RESULTS = os.path.join(HERE, 'results')
STUDENTS = ('linear', 'tree_d3', 'tree_d6', 'kd_mlp_h16', 'gplearn_sr',
            'eml_data', 'classic_gp')
FULL = ('eml_task', 'classic_task', 'nooracle_task')
DATA_EPISODES = 20
MAX_GROUPS = 20_000          # decisions kept for fitting (uniform subsample)


def fidelity(score_fn, data):
    """Share of decisions where the student's argmax is the teacher's."""
    with np.errstate(all='ignore'):
        s = np.asarray(score_fn(data['X']), dtype=np.float64)
    s = np.where(np.isfinite(s), s, -np.inf)
    g = data['groups']
    hit = n = 0
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    for a, b in zip(starts, ends):
        hit += int(np.argmax(s[a:b]) == np.argmax(data['T'][a:b]))
        n += 1
    return hit / max(n, 1)


def subsample_groups(data, n, rng):
    g = data['groups']
    uniq = np.unique(g)
    if len(uniq) <= n:
        return data
    keep = np.isin(g, rng.choice(uniq, n, replace=False))
    return {k: v[keep] for k, v in data.items()}


class _Tree:
    def __init__(self, tree):
        self.tree = tree

    def scores(self, X):
        with np.errstate(all='ignore'):
            v = self.tree.evaluate_batch(np.asarray(X, dtype=np.float64))
        return np.where(np.isfinite(v), v, -np.inf)

    def complexity(self):
        m = static_metrics(self.tree)
        return {'size': m['nodes'], 'size_unit': 'nodes',
                'flops': m['n_arith'] + m['n_transcendental'],
                'transcendental': m['n_transcendental']}

    def describe(self):
        return self.tree.to_string()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=sorted(TASKS))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--methods', default=','.join(STUDENTS + FULL))
    a = p.parse_args(argv)
    task = TASKS[a.task]()
    teacher = O.load(a.task).score_fn()
    out_dir = os.path.join(RESULTS, a.task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'seed_{a.seed}.jsonl')
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)['method'] for l in f if l.strip()}

    rng = np.random.default_rng(a.seed)
    data_seeds = pools.slice_(pools.DATA, DATA_EPISODES, a.seed * DATA_EPISODES)
    _, data = play(task, teacher, data_seeds, collect=True, teacher_fn=teacher)
    data = subsample_groups(data, MAX_GROUPS, rng)
    _, held = play(task, teacher, pools.HELDOUT, collect=True, teacher_fn=teacher)
    t_test = play(task, teacher, pools.TEST)
    r_rng = np.random.default_rng(12345)
    r_test = play(task, lambda X: r_rng.random(len(X)), pools.TEST)
    t_mean, r_mean = float(t_test.mean()), float(r_test.mean())
    span = t_mean - r_mean if abs(t_mean - r_mean) > 1e-9 else 1.0

    def write(method, scorer, fit_sec, extra=None):
        ret = play(task, scorer.scores, pools.TEST)
        row = {'task': a.task, 'level': task.level, 'method': method,
               'seed': a.seed, 'metric_name': task.metric_name,
               'metric': float(ret.mean()), 'metric_std': float(ret.std(ddof=1)),
               'per_episode': [float(x) for x in ret],
               'teacher_metric': t_mean, 'random_metric': r_mean,
               'pct_of_teacher': 100.0 * (ret.mean() - r_mean) / span,
               'fidelity': fidelity(scorer.scores, held),
               'n_test': len(pools.TEST), 'n_data_rows': int(len(data['X'])),
               'fit_sec': fit_sec, **scorer.complexity(),
               'formula': scorer.describe()[:2000] if hasattr(scorer, 'describe') else '',
               **(extra or {})}
        with open(out_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(f"[{a.task} s{a.seed}] {method:14s} {row['metric']:9.2f} "
              f"({row['pct_of_teacher']:6.1f}% of oracle {t_mean:.2f}) "
              f"fid {row['fidelity']:.3f} size {row.get('size')} "
              f"{fit_sec:.0f}s", flush=True)

    methods = a.methods.split(',')
    F = len(task.feature_names)
    data_trees = {}
    for m in methods:
        need_tree = m in ('eml_task', 'classic_task')
        if m in done and not need_tree:
            continue
        t0 = time.time()
        if m in STUDENTS or need_tree:
            name = {'eml_task': 'eml_data', 'classic_task': 'classic_gp'}.get(m, m)
            if name in data_trees:
                s = data_trees[name]
            else:
                random.seed(a.seed)
                np.random.seed(a.seed)
                s = make_student(name, seed=a.seed).fit(
                    data['X'], data['T'], kind='rank', groups=data['groups'])
                if name in ('eml_data', 'classic_gp'):
                    data_trees[name] = s
            if not need_tree:
                if m not in done:
                    write(m, s, time.time() - t0)
                continue
            if m in done:
                continue
            basis_name = 'eml' if m == 'eml_task' else 'classic'
            seed_tree = s.regs[0].tree
            basis = seed_tree.basis
        else:                                       # nooracle_task
            basis_name = 'eml'
            basis = make_basis('eml', F, task.feature_names, max_depth=6)
            seed_tree = None
        random.seed(a.seed)
        np.random.seed(a.seed)
        g = game_phase(task, basis, seed_tree)
        write(m, _Tree(g['best_tree']), time.time() - t0,
              {'basis': basis_name, 'val': g['best_val'],
               'game_generations': g['generations_run']})


if __name__ == '__main__':
    main()
