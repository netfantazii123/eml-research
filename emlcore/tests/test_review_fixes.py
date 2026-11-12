"""Regression tests for the 2026-11 implementation review (each one failed before its fix)."""

import math
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from emlcore import (Node, eml_basis, classic_basis, search, nsga2,  # noqa: E402
                     checkpoint, ParetoArchive)
from emlcore import cexport, metrics  # noqa: E402
from emlcore.basis import c_float  # noqa: E402


# ── C export ────────────────────────────────────────────────────────────────

def test_c_float_literals_parse_and_keep_precision():
    assert c_float(10.0) == '10.0f'                 # was '10f': not valid C
    assert float(c_float(1.2345678e-7)[:-1]) == pytest.approx(1.2345678e-7, rel=1e-7)
    with pytest.raises(ValueError):
        c_float(float('nan'))


def _compile_and_run(header: str, X: np.ndarray, tmp_path) -> np.ndarray:
    (tmp_path / 'f.h').write_text(header)
    rows = ",\n".join("{" + ", ".join(c_float(v) for v in r) + "}" for r in X)
    (tmp_path / 'm.c').write_text(
        '#include <stdio.h>\n#include "f.h"\n'
        f"static const float X[{len(X)}][{X.shape[1]}] = {{{rows}}};\n"
        "int main(void){for(int i=0;i<%d;i++) printf(\"%%.9g\\n\", "
        "(double)eml_formula(X[i])); return 0;}\n" % len(X))
    exe = tmp_path / 'm'
    subprocess.run(['gcc', '-std=c99', '-Wall', '-Werror', '-O0',
                    str(tmp_path / 'm.c'), '-o', str(exe), '-lm'],
                   check=True, capture_output=True, text=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    return np.array([float(v) for v in out.stdout.split()])


@pytest.mark.skipif(shutil.which('gcc') is None, reason='needs gcc')
@pytest.mark.parametrize('make', [eml_basis, classic_basis])
def test_generated_header_compiles_and_matches_python(make, tmp_path):
    basis = make(3, ('a', 'b', 'c'), max_depth=4, max_nodes=15)
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(25, 3))
    random.seed(1)
    # A tiny constant: '{:.6f}' turned it into 0.000000f and changed the value.
    op = basis.ops[0] if make is eml_basis else next(
        o for o in basis.ops if o.name == 'log')
    kids = ((Node.var(basis, 0), Node.const(basis, 1.2345678e-7))
            if op.arity == 2 else (Node.const(basis, 1.2345678e-7),))
    tiny = Node.apply(basis, op, *kids)
    trees = [search.random_tree(basis, 3, force_op=True) for _ in range(15)]
    trees.append(tiny)
    for t in trees:
        py = t.evaluate_batch(X.astype(np.float32), np.float32)
        c = _compile_and_run(cexport.generate_header(t), X.astype(np.float32),
                             tmp_path)
        ok = np.isfinite(py)
        assert np.allclose(c[ok], py[ok], rtol=1e-3, atol=1e-3), t.to_string()


# ── NaN in selection ────────────────────────────────────────────────────────

def test_nan_objective_is_dominated_not_front_zero():
    obj = [[1, -3], [np.nan, -1], [0.5, -2], [0.9, -1]]
    fronts = nsga2.nondominated_sort(obj)
    assert 1 not in fronts[0]
    d = nsga2.crowding_distance(obj, [0, 1, 2, 3])
    assert not np.isnan(d).any()


def test_archive_ignores_nan_and_keeps_improving():
    basis = eml_basis(2, ('a', 'b'))
    random.seed(0)
    t = search.random_tree(basis, 2, force_op=True)
    a = ParetoArchive()
    assert a.update(t, float('nan')) is False
    assert a.update(t, 0.9) is True
    assert a.best[t.size()][0] == 0.9


def test_tournament_never_picks_nan():
    random.seed(0)
    scores = [float('nan'), 0.1, float('nan')]
    picks = {search.tournament_select(scores, k=3) for _ in range(20)}
    assert picks == {1}


# ── Checkpoint directories ──────────────────────────────────────────────────

def _run(d, basis, gens, improved_at, start=0, resume_gen=None, score=None):
    ck = checkpoint.RunCheckpointer(d, phase='data', signature={},
                                    generations=gens, every=5,
                                    snapshot_every=10, keep_best=2,
                                    resume_gen=resume_gen)
    random.seed(0)
    pop = [search.random_tree(basis, 3, force_op=True) for _ in range(4)]
    for gen in range(start, gens):
        imp = gen in improved_at
        if imp:
            ck.record(pop[gen % 4], score if score is not None else float(gen), gen)
        ck.on_generation(checkpoint.capture(gen, pop, counters={}), improved=imp)
    ck.finish(checkpoint.capture(gens, pop, counters={}))


def test_fresh_run_in_used_directory_does_not_inherit_old_states(tmp_path):
    basis = eml_basis(2, ('a', 'b'))
    d = str(tmp_path / 'ck')
    _run(d, basis, 300, {298, 299}, start=290, score=0.99)   # an old, long run
    _run(d, basis, 5, {0, 1, 2}, score=0.1)                    # a fresh short run
    best = checkpoint.resolve_point(d, 'best')
    assert best.endswith('gen_000002.json'), best
    hof = checkpoint.load_hall_of_fame(d, basis)
    assert all(h['score'] == 0.1 for h in hof)
    assert os.path.isdir(os.path.join(d, 'superseded'))        # moved, not deleted


def test_resume_from_earlier_point_sets_aside_abandoned_timeline(tmp_path):
    basis = eml_basis(2, ('a', 'b'))
    d = str(tmp_path / 'ck')
    _run(d, basis, 30, {5, 25})
    _run(d, basis, 30, {12}, start=10, resume_gen=10)
    pts = {(p['kind'], p['generation']) for p in checkpoint.list_points(d)}
    assert ('best', 25) not in pts and ('best', 12) in pts and ('best', 5) in pts


# ── Cost proxy ──────────────────────────────────────────────────────────────

def test_division_is_not_a_transcendental_call():
    basis = classic_basis(1, ('a',))
    div = next(o for o in basis.ops if o.name == 'div')
    x = Node.var(basis, 0)
    m = metrics.static_metrics(Node.apply(basis, div, x, x.clone()))
    assert m['n_transcendental'] == 0 and m['n_arith'] > 0
