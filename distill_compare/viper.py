"""
viper.py — VIPER (Bastani, Pu & Solar-Lezama, NeurIPS 2018): verifiable
policy extraction into a decision tree.

    D <- teacher rollouts, labelled by the teacher
    for i in 1..N:
        resample D with probability proportional to the loss weight l(s)
        fit tree_i on the resample
        roll out tree_i, label the visited states with the teacher, D <- D + them
    return the tree_i with the best validation score

The rollout/aggregate half is DAgger (Ross et al., 2011); the resampling is what
makes it VIPER. The paper's weight is l(s) = V(s) - min_a Q(s, a), which needs a
Q-function; the oracles here are policies without one (Flappy's GA-MLP has no
critic at all), so the gap between the teacher's best and worst logit is used
instead — the same "how costly is a mistake in this state" signal, read off the
policy. For ranking tasks the weight is per decision (group), and whole groups
are resampled so a placement is never ranked without its competitors.

Validation episodes come from the VAL seed pool and the reported score from the
TEST pool, never the same seeds that picked the tree.
"""

from __future__ import annotations

import numpy as np

from .envs import concat_data, rollout, student_policy, teacher_policy
from .students import TreeStudent, group_gap


def _weights(kind: str, data: dict) -> np.ndarray:
    T = data['T']
    if kind == 'classify':
        return T.max(1) - T.min(1)
    return group_gap(T, data['groups'])


def _resample(kind: str, data: dict, n: int, rng) -> dict:
    w = _weights(kind, data) + 1e-6
    if kind == 'classify':
        idx = rng.choice(len(w), size=min(n, len(w)), replace=True, p=w / w.sum())
        return {'X': data['X'][idx], 'T': data['T'][idx],
                'groups': np.arange(len(idx))}
    g = data['groups']
    uniq, first = np.unique(g, return_index=True)
    gw = w[first]
    pick = rng.choice(len(uniq), size=min(n, len(uniq)), replace=True,
                      p=gw / gw.sum())
    order = np.argsort(g, kind='stable')
    gs = g[order]
    starts = np.searchsorted(gs, uniq)
    ends = np.searchsorted(gs, uniq, side='right')
    rows, new_g = [], []
    for k, j in enumerate(pick):
        r = order[starts[j]:ends[j]]
        rows.append(r)
        new_g.append(np.full(len(r), k))
    rows = np.concatenate(rows)
    return {'X': data['X'][rows], 'T': data['T'][rows],
            'groups': np.concatenate(new_g)}


def viper(adapter, *, max_depth: int = 6, n_iters: int = 8,
          episodes_per_iter: int = 10, train_seeds, val_seeds,
          resample_n: int = 20000, seed: int = 0, initial: dict | None = None,
          label_every: int = 1, verbose: bool = True):
    """
    Run VIPER. Returns (best_tree_student, history). `initial` is an already
    collected teacher dataset (saves re-rolling the teacher).
    """
    rng = np.random.default_rng(seed)
    kind = adapter.kind
    chunks = []
    if initial is not None:
        chunks.append(initial)
    else:
        _, d0 = rollout(adapter, teacher_policy(adapter),
                        train_seeds[:episodes_per_iter], collect=True,
                        label_every=label_every)
        chunks.append(d0)
    best, best_val, history = None, -np.inf, []
    n_tr = len(train_seeds)
    for it in range(n_iters):
        data = concat_data(chunks)
        sample = _resample(kind, data, resample_n, rng)
        tree = TreeStudent(max_depth=max_depth, seed=seed).fit(
            sample['X'], sample['T'], kind=kind, groups=sample['groups'])
        val, _ = rollout(adapter, student_policy(adapter, tree), val_seeds)
        v = float(np.mean(val))
        history.append({'iter': it, 'rows': len(data['X']), 'val': v})
        if verbose:
            print(f"    VIPER it {it}: {len(data['X']):>7} rows | val {v:.2f}",
                  flush=True)
        if v > best_val:
            best, best_val = tree, v
        if it < n_iters - 1:
            off = (it * episodes_per_iter) % n_tr
            seeds = [train_seeds[(off + k) % n_tr] for k in range(episodes_per_iter)]
            _, d = rollout(adapter, student_policy(adapter, tree), seeds,
                           collect=True, label_every=label_every)
            chunks.append(d)
    best.name = f'viper_d{max_depth}'
    return best, history
