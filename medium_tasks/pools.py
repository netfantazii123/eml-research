"""Disjoint episode-seed pools, the same discipline as the games.

DATA (oracle demonstrations), TRAIN (GA / oracle fitness, rotating), VAL
(selection), HELDOUT (fidelity states) and TEST (the reported number, never
selected on) never share a seed.
"""
DATA = list(range(10_000, 10_000 + 200))
TRAIN = list(range(20_000, 20_000 + 600))
VAL = list(range(30_000, 30_000 + 32))
HELDOUT = list(range(40_000, 40_000 + 20))
TEST = list(range(50_000, 50_000 + 50))


def slice_(pool, n, offset=0):
    return [pool[(offset + i) % len(pool)] for i in range(n)]
