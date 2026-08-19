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

    n_ops = sum(ops.values())
    n_transcendental = sum(calls.values())
    return {
        'nodes': n_ops + n_var + n_const,
        'depth': tree.depth(),
        'n_ops': n_ops,
        'ops': dict(ops),
        'n_leaves': n_var + n_const,
        'n_const': n_const,
        'n_vars_unique': tree.n_unique_vars(),
        'n_exp': calls.get('exp', 0),
        'n_log': calls.get('log', 0),
        'n_transcendental': n_transcendental,
        # Kept for the EML basis, where one operator is one exp and one log.
        'n_eml': ops.get('eml', 0),
        **estimate_cost(n_ops, n_transcendental, n_const, tree.depth()),
    }


# Rough per-item costs for a Cortex-M class target, in bytes. These are order-of
# -magnitude proxies for ranking formulas against each other before any hardware
# is involved, NOT predictions: a real figure needs a real toolchain, and the
# gap between these numbers and the measured ones is itself worth reporting.
_BYTES_PER_CALL = 8        # a libm call: argument setup + bl
_BYTES_PER_ARITH = 4       # one inline FPU/soft-float op
_BYTES_PER_CONST = 4       # a float literal in .rodata


def estimate_cost(n_ops: int, n_transcendental: int, n_const: int,
                  depth: int) -> dict:
    """
    Code-size and stack proxies for the exported C.

    The expression compiles to a single nested statement, so peak stack tracks
    the tree DEPTH (live intermediates along one root-to-leaf path) rather than
    the node count — which is why a deep 40-node tree can cost more stack than a
    wide 100-node one.
    """
    arith = max(0, n_ops - n_transcendental)
    return {
        'est_code_bytes': (n_transcendental * _BYTES_PER_CALL
                           + arith * _BYTES_PER_ARITH
                           + n_const * _BYTES_PER_CONST),
        'est_stack_floats': depth + 1,
        'est_stack_bytes': (depth + 1) * 4,
    }


def pool_slice(pool, n: int, offset: int = 0) -> list[int]:
    """n seeds from the pool starting at offset (wraps around)."""
    if not pool:
        raise ValueError("empty seed pool")
    return [pool[(offset + i) % len(pool)] for i in range(n)]
