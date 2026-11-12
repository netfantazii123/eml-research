"""
parallel.py — a shared process pool for fitness evaluations that are expensive.

Only worth it where one evaluation is expensive: a GAME-phase fitness plays
several full games (tens of milliseconds to seconds), so a population of 60
spread over 8 cores is close to 8x faster. A DATA-phase fitness is one numpy
pass over a subsample — sub-millisecond — and shipping trees to a process would
cost more than evaluating them, so it stays in-process.

The pool is created once and reused (evaluate_pareto with refine runs one GAME
phase per front point; a pool per call would pay the spawn cost every time).
'spawn' on every platform: the GUI runs the distillation in a thread, and
forking a process that has threads is undefined behaviour.

Workers receive trees as dicts and rebuild them; results come back in order, so
a parallel run returns exactly what a serial one would.
"""

from __future__ import annotations

import atexit
import os

_POOL = None
_POOL_WORKERS = 0


def resolve_workers(workers: int | None) -> int:
    """0 or None -> all cores; anything else is taken as is (min 1)."""
    if not workers:
        return max(1, os.cpu_count() or 1)
    return max(1, int(workers))


def get_pool(workers: int):
    """The shared pool with `workers` processes, or None for serial."""
    global _POOL, _POOL_WORKERS
    if workers <= 1:
        return None
    if _POOL is not None and _POOL_WORKERS == workers:
        return _POOL
    shutdown()
    import multiprocessing as mp
    _POOL = mp.get_context('spawn').Pool(workers)
    _POOL_WORKERS = workers
    return _POOL


def shutdown() -> None:
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.terminate()
        _POOL.join()
    _POOL, _POOL_WORKERS = None, 0


atexit.register(shutdown)


def map_ordered(fn, payloads: list, workers: int) -> list:
    """fn over payloads, in order; serial when workers <= 1."""
    pool = get_pool(workers)
    if pool is None or len(payloads) <= 1:
        return [fn(p) for p in payloads]
    chunk = max(1, len(payloads) // (workers * 4))
    return pool.map(fn, payloads, chunksize=chunk)
