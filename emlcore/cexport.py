"""
cexport.py — compile a formula tree to a C expression.

The C form of every operator lives in the Op next to its numeric form, so the
exporter is basis-agnostic and cannot drift from the evaluator.

One caveat this module cannot fix: the generated C is float32 (expf/logf) while
the search runs in float64. With nested exp the two can disagree, and a formula
that ranks placements differently on the target is not the formula the results
describe. check_float32_agreement() measures that disagreement, and it should be
run before any hardware timing — timing the wrong policy is worse than not
timing at all.
"""

from __future__ import annotations

import numpy as np

from .basis import Basis, c_float
from .node import Node


def c_expr(node: Node, var_fmt: str = "f[{idx}]") -> str:
    """The formula as a single C expression."""
    if node.kind == 'const':
        return c_float(node.value)
    if node.kind == 'var':
        return var_fmt.format(idx=node.var_idx,
                              name=node.basis.var_name(node.var_idx))
    return node.op.c_expr.format(*(c_expr(c, var_fmt) for c in node.children))


def c_helpers(basis: Basis, used_ops: set[str] | None = None) -> list[str]:
    """Helper definitions for the operators actually used (or the whole basis)."""
    out: list[str] = []
    for op in basis.ops:
        if used_ops is not None and op.name not in used_ops:
            continue
        for helper in op.c_helpers:
            if helper not in out:
                out.append(helper)
    return out


def used_ops(node: Node) -> set[str]:
    names: set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.kind == 'op':
            names.add(n.op.name)
            stack.extend(n.children)
    return names


def generate_header(tree: Node, *, guard: str = 'EML_FORMULA_H',
                    func_name: str = 'eml_formula',
                    var_fmt: str = "f[{idx}]",
                    title: str = 'distilled EML policy',
                    notes: list[str] | None = None,
                    metrics: dict | None = None) -> str:
    """A self-contained C header holding the formula and its helpers."""
    basis = tree.basis
    lines = [
        "/*",
        f" * {func_name}.h — {title} (auto-generated).",
        " *",
        f" * Basis: {basis.name}   inputs: {basis.n_vars}",
    ]
    if metrics:
        lines.append(
            f" * AST: nodes={metrics.get('nodes')} depth={metrics.get('depth')} "
            f"unique_vars={metrics.get('n_vars_unique')}")
        lines.append(
            f" * Transcendental calls per evaluation: "
            f"expf={metrics.get('n_exp')} logf={metrics.get('n_log')}")
    for note in (notes or []):
        lines.append(f" * {note}")
    lines.append(" *")
    lines.append(" * Inputs:")
    for i in range(basis.n_vars):
        lines.append(f" *   f[{i:2d}] = {basis.var_name(i)}")
    lines += [
        " */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <math.h>",
        "",
    ]
    lines += c_helpers(basis, used_ops(tree))
    lines += [
        "",
        f"static inline float {func_name}(const float f[{basis.n_vars}]) {{",
        f"    return {c_expr(tree, var_fmt)};",
        "}",
        "",
        f"#endif /* {guard} */",
        "",
    ]
    return "\n".join(lines)


def check_float32_agreement(tree: Node, X: np.ndarray,
                            group_size: int | None = None,
                            groups: np.ndarray | None = None) -> dict:
    """
    How much the float32 (target) and float64 (search) evaluations disagree.

    A relative error is not the interesting quantity: the formula only ever
    picks an argmax, so what matters is how often the two precisions pick a
    different candidate. Pass `groups` (the per-row candidate-set id from the
    dataset) to measure that on real decisions; group_size falls back to fixed
    windows when no grouping is available.
    """
    # Both the inputs and the arithmetic run at the target precision — casting
    # only the inputs and still computing in float64 measures nothing, since the
    # error this is looking for accumulates inside the nested exp/log.
    f64 = tree.evaluate_batch(np.asarray(X, dtype=np.float64), np.float64)
    f32 = tree.evaluate_batch(np.asarray(X, dtype=np.float32),
                              np.float32).astype(np.float64)

    finite = np.isfinite(f64) & np.isfinite(f32)
    rel = np.zeros_like(f64)
    denom = np.maximum(np.abs(f64), 1e-12)
    rel[finite] = np.abs(f64[finite] - f32[finite]) / denom[finite]

    out = {
        'n_samples': int(len(f64)),
        'max_rel_error': float(rel.max()) if len(rel) else 0.0,
        'mean_rel_error': float(rel.mean()) if len(rel) else 0.0,
        'n_nonfinite': int((~finite).sum()),
    }

    if groups is not None:
        groups = np.asarray(groups)
        # Groups are contiguous in the collected dataset, so boundaries are
        # where the id changes.
        bounds = np.flatnonzero(np.diff(groups)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [len(groups)]))
        flips = 0
        n_groups = 0
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue          # a single candidate cannot flip
            n_groups += 1
            if int(np.argmax(f64[s:e])) != int(np.argmax(f32[s:e])):
                flips += 1
        out['n_groups'] = n_groups
        out['argmax_flip_rate'] = float(flips / n_groups) if n_groups else 0.0
    elif group_size and group_size > 1 and len(f64) >= group_size:
        n_groups = len(f64) // group_size
        a = f64[:n_groups * group_size].reshape(n_groups, group_size)
        b = f32[:n_groups * group_size].reshape(n_groups, group_size)
        flips = (a.argmax(axis=1) != b.argmax(axis=1)).mean()
        out['n_groups'] = int(n_groups)
        out['argmax_flip_rate'] = float(flips)
    return out
