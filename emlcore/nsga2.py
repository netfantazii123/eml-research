"""
nsga2.py — multi-objective selection over (quality, size).

Why this rather than one run per node budget K: a sweep over 9 budgets x 2 bases
x 8 tasks x 10 seeds is roughly 1440 GA runs, which does not fit in the time
available. NSGA-II reads the whole quality-vs-size curve off a single run,
because the population spreads along the Pareto front instead of collapsing onto
whatever size the pressure happens to favour.

Why not the cheaper trick — running an ordinary single-objective GA and keeping
an archive of the best tree seen at each size: that archive is biased downward
exactly where it matters. Small trees enter it in early generations, when the
search has not converged for any size, and nothing ever revisits them; the
result answers "the best small tree met on the way to a big one", not "the best
small tree". Under NSGA-II small trees keep selection pressure of their own for
the whole run, because they are non-dominated by construction.

That said, the curve this produces still has to be checked against dedicated
fixed-K runs on at least one task before it is used as a result. `k_budget` on
the Basis exists for exactly those anchor runs.

Objectives are maximised. Size is passed as -size.
"""

from __future__ import annotations

import random

import numpy as np


def nondominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    Split indices into Pareto fronts, best first (all objectives maximised).

    The straightforward O(M N^2) formulation from Deb et al. 2002; population
    sizes here are in the hundreds, so the quadratic term is not the bottleneck
    next to evaluating the trees.
    """
    n = len(objectives)
    if n == 0:
        return []
    obj = np.asarray(objectives, dtype=np.float64)

    # dominates[i, j] = i dominates j
    ge = (obj[:, None, :] >= obj[None, :, :]).all(axis=2)
    gt = (obj[:, None, :] > obj[None, :, :]).any(axis=2)
    dominates = ge & gt

    domination_count = dominates.sum(axis=0)      # how many dominate i
    fronts: list[list[int]] = []
    remaining = np.ones(n, dtype=bool)
    counts = domination_count.copy()

    while remaining.any():
        current = np.flatnonzero(remaining & (counts <= 0))
        if len(current) == 0:            # numerical safety net: take the rest
            current = np.flatnonzero(remaining)
        fronts.append([int(i) for i in current])
        remaining[current] = False
        # Removing this front lowers the domination count of everyone it beat.
        counts -= dominates[current].sum(axis=0)
    return fronts


def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance within one front — the diversity tie-breaker."""
    k = len(front)
    dist = np.zeros(k, dtype=np.float64)
    if k <= 2:
        return np.full(k, np.inf)

    obj = np.asarray(objectives, dtype=np.float64)[front]
    for m in range(obj.shape[1]):
        order = np.argsort(obj[:, m])
        lo, hi = obj[order[0], m], obj[order[-1], m]
        dist[order[0]] = dist[order[-1]] = np.inf
        span = hi - lo
        if span <= 0:
            continue
        prev = obj[order[:-2], m]
        nxt = obj[order[2:], m]
        dist[order[1:-1]] += (nxt - prev) / span
    return dist


def rank_and_crowding(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-individual front index (0 = best) and crowding distance."""
    n = len(objectives)
    ranks = np.zeros(n, dtype=np.int64)
    crowd = np.zeros(n, dtype=np.float64)
    for r, front in enumerate(nondominated_sort(objectives)):
        d = crowding_distance(objectives, front)
        for pos, i in enumerate(front):
            ranks[i] = r
            crowd[i] = d[pos]
    return ranks, crowd


def environmental_selection(objectives: np.ndarray, n: int) -> list[int]:
    """The (mu + lambda) survivor step: fill by front, break ties by crowding."""
    chosen: list[int] = []
    for front in nondominated_sort(objectives):
        if len(chosen) + len(front) <= n:
            chosen.extend(front)
            continue
        d = crowding_distance(objectives, front)
        order = np.argsort(-d)
        chosen.extend(front[i] for i in order[:n - len(chosen)])
        break
    return chosen[:n]


def crowded_tournament(ranks: np.ndarray, crowd: np.ndarray, k: int = 3) -> int:
    """Tournament by (front, then crowding) — the NSGA-II parent selector."""
    n = len(ranks)
    cands = random.sample(range(n), min(k, n))
    best = cands[0]
    for i in cands[1:]:
        if ranks[i] < ranks[best] or (
                ranks[i] == ranks[best] and crowd[i] > crowd[best]):
            best = i
    return best


class ParetoArchive:
    """
    Best-scoring tree seen at every node count.

    Kept alongside NSGA-II rather than instead of it: the archive is what the
    R(K) curve is read from, while NSGA-II is what keeps small sizes genuinely
    searched. Entries are replaced only on a strict improvement, so the curve is
    monotone in what was actually found.
    """

    def __init__(self):
        self.best: dict[int, tuple[float, object]] = {}

    def update(self, tree, score: float) -> bool:
        size = tree.size()
        cur = self.best.get(size)
        if cur is None or score > cur[0]:
            self.best[size] = (score, tree.clone())
            return True
        return False

    def update_many(self, trees, scores) -> None:
        for t, s in zip(trees, scores):
            self.update(t, s)

    def curve(self) -> list[dict]:
        """
        The R(K) curve: for every budget K, the best score at size <= K.

        Cumulative on purpose — a formula of 9 nodes is a legal answer to "what
        can 15 nodes do", and reporting the raw per-size best would show dips
        that are artefacts of which exact sizes the search happened to visit.
        """
        rows = []
        best_so_far = -np.inf
        best_tree = None
        for size in sorted(self.best):
            score, tree = self.best[size]
            if score > best_so_far:
                best_so_far, best_tree = score, tree
            rows.append({'k': size, 'score_at_k': score,
                         'best_score_upto_k': best_so_far,
                         'tree': best_tree})
        return rows

    def front(self) -> list[tuple[int, float, object]]:
        """(size, score, tree) for sizes that are not dominated by a smaller one."""
        out = []
        best_so_far = -np.inf
        for size in sorted(self.best):
            score, tree = self.best[size]
            if score > best_so_far:
                best_so_far = score
                out.append((size, score, tree))
        return out
