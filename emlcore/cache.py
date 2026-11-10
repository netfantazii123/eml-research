"""
cache.py — memoised fitness.

A GA re-evaluates the same formula far more often than it looks:

  * the validation re-scoring of the generation champion — the champion is
    usually an elite, i.e. the same tree as last generation, scored again on
    the same fixed validation data (in the GAME phase: the same 8 games);
  * duplicates inside one generation — crossover of two copies of an elite, or
    a mutation that rewrote a constant back to a value already present.

Fitness is a pure function of (tree, data the generation is scored on), so both
are cached by (Node.key(), context). The context is whatever identifies that
data: the generation index when the subsample is redrawn every generation, the
seed tuple of the generation's games, or a constant such as 'val' for fixed
validation data. A cache hit returns exactly the value a re-evaluation would,
so caching changes how long a run takes and nothing else.
"""

from __future__ import annotations

from collections import OrderedDict


class FitnessCache:
    def __init__(self, maxsize: int = 100_000):
        self.maxsize = int(maxsize)
        self._d: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, tree, context, compute):
        k = (tree.key(), context)
        if k in self._d:
            self._d.move_to_end(k)
            self.hits += 1
            return self._d[k]
        self.misses += 1
        v = compute(tree)
        self._d[k] = v
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)
        return v

    def map(self, trees, context, compute, *, batch_compute=None) -> list:
        """
        Fitness for a whole population.

        batch_compute(list_of_trees) -> list, if given, scores all the misses at
        once — that is the hook for evaluating in parallel. Duplicates within
        the call are computed once.
        """
        keys = [t.key() for t in trees]
        out = [None] * len(trees)
        todo: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            ck = (k, context)
            if ck in self._d:
                self._d.move_to_end(ck)
                self.hits += 1
                out[i] = self._d[ck]
            else:
                todo.setdefault(k, []).append(i)
        if todo:
            firsts = [idx[0] for idx in todo.values()]
            uniq = [trees[i] for i in firsts]
            self.misses += len(uniq)
            self.hits += sum(len(idx) - 1 for idx in todo.values())
            vals = (batch_compute(uniq) if batch_compute is not None
                    else [compute(t) for t in uniq])
            for (k, idx), v in zip(todo.items(), vals):
                self._d[(k, context)] = v
                for i in idx:
                    out[i] = v
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)
        return out

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def stats(self) -> dict:
        return {'hits': self.hits, 'misses': self.misses,
                'hit_rate': round(self.hit_rate, 4)}
