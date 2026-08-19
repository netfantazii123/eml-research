"""
basis.py — the set of primitives a formula may be built from.

A Basis is everything the search needs to know about the language of formulas:
which operators exist, how many inputs there are, how deep a tree may grow. The
GA in search.py never mentions a specific operator, so swapping the basis swaps
the language without touching the search — which is the point: an EML result
means nothing until the same pipeline has been run on an ordinary basis (+ - *
/ exp log tanh) at the same budget and the same seeds.

Numerical guards (the exp clamp and the ln epsilon) live inside the operator
closures rather than in the evaluator. They are properties of THIS operator, not
of formulas in general, and they are exactly what separates the operator
implemented here from the one in the literature: eml(x, y) = exp(x) - ln(y) is
complete over the complex numbers, while exp(clamp(x, +-C)) - ln(|y| + eps) is a
different function that happens to be computable over the reals. Keeping C and
eps as basis parameters is what makes that gap measurable instead of hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Op:
    """
    One operator: how to evaluate it, print it, and compile it to C.

    text and c_expr are format strings positional in the children, e.g.
    "eml({0}, {1})". Keeping the C form next to the numeric form is deliberate —
    when they are written in different files they drift, and a formula that
    scores well in Python but compiles to something subtly different is the kind
    of bug that only shows up on the hardware.
    """
    name: str
    arity: int
    batch: Callable[..., np.ndarray]   # vectorised: arrays -> array
    scalar: Callable[..., float]       # single sample
    text: str
    c_expr: str
    c_helpers: tuple[str, ...] = ()    # C function definitions this op needs


def make_eml_op(epsilon: float = 1e-10, clamp: float = 10.0) -> Op:
    """eml(x, y) = exp(clamp(x, +-clamp)) - ln(|y| + epsilon)."""
    def _batch(x, y):
        return np.exp(np.clip(x, -clamp, clamp)) - np.log(np.abs(y) + epsilon)

    def _scalar(x, y):
        xc = -clamp if x < -clamp else (clamp if x > clamp else x)
        return math.exp(xc) - math.log(abs(y) + epsilon)

    helper = (
        f"#define EML_EPS {epsilon:g}f\n"
        f"#define EML_CLAMP {clamp:g}f\n"
        "static inline float eml_op(float x, float y) {\n"
        "    if (x > EML_CLAMP) x = EML_CLAMP;\n"
        "    if (x < -EML_CLAMP) x = -EML_CLAMP;\n"
        "    return expf(x) - logf(fabsf(y) + EML_EPS);\n"
        "}"
    )
    return Op(name='eml', arity=2, batch=_batch, scalar=_scalar,
              text="eml({0}, {1})", c_expr="eml_op({0}, {1})",
              c_helpers=(helper,))


@dataclass
class Basis:
    """The language of formulas plus the shape limits the search must respect."""
    name: str
    ops: tuple[Op, ...]
    n_vars: int
    var_names: tuple[str, ...] = ()
    max_depth: int = 7
    const_range: tuple[float, float] = (-2.0, 2.0)
    const_abs_max: float = 5.0
    var_leaf_prob: float = 0.7
    # Node budget: None = unbounded (the historical behaviour, where the size of
    # a formula is whatever the run happened to produce). Set it to sweep K.
    max_nodes: int | None = None
    _by_name: dict[str, Op] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._by_name = {o.name: o for o in self.ops}
        if not self.var_names:
            self.var_names = tuple(f"x{i}" for i in range(self.n_vars))

    def op(self, name: str) -> Op:
        try:
            return self._by_name[name]
        except KeyError:
            raise ValueError(
                f"operator {name!r} is not in basis {self.name!r} "
                f"(have: {sorted(self._by_name)})") from None

    def ops_with_arity(self, arity: int) -> tuple[Op, ...]:
        return tuple(o for o in self.ops if o.arity == arity)

    @property
    def default_op(self) -> Op:
        """The operator used where the old code hardcoded 'eml'."""
        return self.ops[0]

    def var_name(self, idx: int) -> str:
        return self.var_names[idx] if idx < len(self.var_names) else f"x{idx}"


def eml_basis(n_vars: int, var_names=(), *, max_depth: int = 7,
              epsilon: float = 1e-10, clamp: float = 10.0,
              max_nodes: int | None = None) -> Basis:
    """The single-operator EML basis — the one this project is about."""
    return Basis(name='eml', ops=(make_eml_op(epsilon, clamp),),
                 n_vars=n_vars, var_names=tuple(var_names),
                 max_depth=max_depth, max_nodes=max_nodes)
