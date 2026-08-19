"""
node.py — the formula AST, independent of which operators it is built from.

A node is a constant, an input variable, or an operator applied to children. The
operator is stored as the Op object itself rather than a name, so evaluation is
a direct call with no lookup, and a node carries its Basis so that printing,
serialisation and mutation do not need it threaded through every call.

Serialisation reads the old {'kind': 'eml', 'left': ..., 'right': ...} layout as
well as the current one, so formulas saved before this refactor still load.
"""

from __future__ import annotations

import numpy as np

from .basis import Basis, Op


class Node:
    """One AST node: kind is 'const', 'var' or 'op'."""

    __slots__ = ('kind', 'op', 'children', 'var_idx', 'value', 'basis')

    def __init__(self, kind: str, basis: Basis, *, op: Op | None = None,
                 children: tuple["Node", ...] = (), var_idx: int = 0,
                 value: float = 1.0):
        self.kind = kind
        self.basis = basis
        self.op = op
        self.children = tuple(children)
        self.var_idx = var_idx
        self.value = value

    # ── Constructors ──────────────────────────────────────────────────────
    @classmethod
    def const(cls, basis: Basis, value: float) -> "Node":
        return cls('const', basis, value=float(value))

    @classmethod
    def var(cls, basis: Basis, idx: int) -> "Node":
        return cls('var', basis, var_idx=int(idx))

    @classmethod
    def apply(cls, basis: Basis, op: Op | str, *children: "Node") -> "Node":
        if isinstance(op, str):
            op = basis.op(op)
        if len(children) != op.arity:
            raise ValueError(f"{op.name} takes {op.arity} children, "
                             f"got {len(children)}")
        return cls('op', basis, op=op, children=children)

    # ── Binary-tree compatibility ─────────────────────────────────────────
    # Plenty of task code (GUI tree rendering, the C exporter) walks .left/.right.
    # Kept as properties so that code keeps working for binary operators.
    @property
    def left(self):
        return self.children[0] if self.children else None

    @property
    def right(self):
        return self.children[1] if len(self.children) > 1 else None

    def replace_child(self, i: int, node: "Node") -> None:
        kids = list(self.children)
        kids[i] = node
        self.children = tuple(kids)

    def become(self, other: "Node") -> None:
        """Turn this node into a copy of another, in place (used by mutation)."""
        self.kind = other.kind
        self.op = other.op
        self.children = other.children
        self.var_idx = other.var_idx
        self.value = other.value

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(self, inputs) -> float:
        """Scalar evaluation for one input vector (inference)."""
        if self.kind == 'const':
            return self.value
        if self.kind == 'var':
            return float(inputs[self.var_idx])
        return self.op.scalar(*(c.evaluate(inputs) for c in self.children))

    def evaluate_batch(self, X: np.ndarray, dtype=np.float64) -> np.ndarray:
        """
        Vectorised evaluation over a whole dataset, (N, n_vars) -> (N,).

        dtype selects the working precision of the whole expression, not just of
        the inputs — numpy keeps the dtype through exp/log, so dtype=np.float32
        reproduces what the exported C actually computes.
        """
        if self.kind == 'const':
            return np.full(X.shape[0], self.value, dtype=dtype)
        if self.kind == 'var':
            return X[:, self.var_idx].astype(dtype)
        return self.op.batch(*(c.evaluate_batch(X, dtype) for c in self.children))

    # ── Metrics ───────────────────────────────────────────────────────────
    def depth(self) -> int:
        if self.kind != 'op':
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        if self.kind != 'op':
            return 1
        return 1 + sum(c.size() for c in self.children)

    def _var_mask(self) -> int:
        if self.kind == 'var':
            return 1 << self.var_idx
        if self.kind == 'op':
            m = 0
            for c in self.children:
                m |= c._var_mask()
            return m
        return 0

    def n_unique_vars(self) -> int:
        return bin(self._var_mask()).count('1')

    # ── Printing / serialisation ──────────────────────────────────────────
    def to_string(self) -> str:
        if self.kind == 'const':
            return f"{self.value:.3f}"
        if self.kind == 'var':
            return self.basis.var_name(self.var_idx)
        return self.op.text.format(*(c.to_string() for c in self.children))

    def clone(self) -> "Node":
        if self.kind == 'const':
            return Node.const(self.basis, self.value)
        if self.kind == 'var':
            return Node.var(self.basis, self.var_idx)
        return Node('op', self.basis, op=self.op,
                    children=tuple(c.clone() for c in self.children))

    def to_dict(self) -> dict:
        if self.kind == 'const':
            return {'kind': 'const', 'value': self.value}
        if self.kind == 'var':
            return {'kind': 'var', 'var_idx': self.var_idx}
        return {'kind': 'op', 'op': self.op.name,
                'children': [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, d: dict, basis: Basis) -> "Node":
        kind = d['kind']
        if kind == 'const':
            return cls.const(basis, float(d['value']))
        if kind == 'var':
            return cls.var(basis, int(d['var_idx']))
        if kind == 'op':
            return cls.apply(basis, d['op'],
                             *(cls.from_dict(c, basis) for c in d['children']))
        # Legacy layout: the operator name was the kind and children were named.
        if 'left' in d and 'right' in d:
            return cls.apply(basis, kind,
                             cls.from_dict(d['left'], basis),
                             cls.from_dict(d['right'], basis))
        raise ValueError(f"Unknown node kind: {kind}")

    def __repr__(self) -> str:
        return f"<Node {self.to_string()[:60]}>"
