"""
metrics.py — static cost of a formula, and the seed-pool helper.

Static metrics are the resource proxies available without any hardware: node
count, depth, and how many transcendental calls the compiled formula makes. On a
Cortex-M without an FPU the transcendental count is the number that decides
whether the formula runs at all, so it belongs in the results table from the
start rather than after the first hardware trip.
"""

from __future__ import annotations

from collections import Counter

from .node import Node
from .search import collect_nodes

# Which C library calls each operator emits. Used to report the transcendental
# cost of a formula before it ever reaches a compiler.
TRANSCENDENTAL_CALLS = {
    'eml': ('exp', 'log'),
    'exp': ('exp',),
    'log': ('log',),
    'tanh': ('tanh',),
    'div': ('div',),
}


def static_metrics(tree: Node) -> dict:
    """Node counts, depth, per-operator counts and transcendental calls."""
    ops: Counter[str] = Counter()
    n_var = n_const = 0
    for node in collect_nodes(tree):
        if node.kind == 'op':
            ops[node.op.name] += 1
        elif node.kind == 'var':
            n_var += 1
        else:
            n_const += 1

    calls: Counter[str] = Counter()
    for name, count in ops.items():
        for call in TRANSCENDENTAL_CALLS.get(name, ()):
            calls[call] += count

    return {
        'nodes': sum(ops.values()) + n_var + n_const,
        'depth': tree.depth(),
        'n_ops': sum(ops.values()),
        'ops': dict(ops),
        'n_leaves': n_var + n_const,
        'n_const': n_const,
        'n_vars_unique': tree.n_unique_vars(),
        'n_exp': calls.get('exp', 0),
        'n_log': calls.get('log', 0),
        'n_transcendental': sum(calls.values()),
        # Kept for the EML basis, where one operator is one exp and one log.
        'n_eml': ops.get('eml', 0),
    }


def pool_slice(pool, n: int, offset: int = 0) -> list[int]:
    """n seeds from the pool starting at offset (wraps around)."""
    if not pool:
        raise ValueError("empty seed pool")
    return [pool[(offset + i) % len(pool)] for i in range(n)]
