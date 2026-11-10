"""emlcore: search refactor, key(), cache, constant folding, checkpoints."""

import json
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from emlcore import (Node, eml_basis, classic_basis, search, nsga2,  # noqa: E402
                     checkpoint, FitnessCache, ParetoArchive)


@pytest.fixture
def basis():
    return eml_basis(4, ('a', 'b', 'c', 'd'), max_depth=6, max_nodes=30)


def _pop(basis, n=40, seed=0):
    random.seed(seed)
    return [search.random_tree(basis, 3, force_op=True) for _ in range(n)]


def test_key_distinguishes_constants_to_string_merges(basis):
    a = Node.apply(basis, 'eml', Node.var(basis, 0), Node.const(basis, 1.00001))
    b = Node.apply(basis, 'eml', Node.var(basis, 0), Node.const(basis, 1.00002))
    assert a.to_string() == b.to_string()
    assert a.key() != b.key()
    assert a.key() == a.clone().key()


@pytest.mark.parametrize('make', [eml_basis, classic_basis])
def test_fold_constants_preserves_value_and_shrinks(make):
    b = make(3, max_depth=7)
    X = np.random.default_rng(0).normal(size=(200, 3))
    random.seed(1)
    shrunk = 0
    for _ in range(200):
        t = search.random_tree(b, 5, force_op=True)
        f = search.fold_constants(t)
        assert f.size() <= t.size()
        shrunk += f.size() < t.size()
        p, q = t.evaluate_batch(X), f.evaluate_batch(X)
        ok = np.isfinite(p)
        np.testing.assert_allclose(q[ok], p[ok], rtol=1e-9, atol=1e-9)
    assert shrunk > 0


def test_unique_breeding_has_no_duplicates(basis):
    pop = _pop(basis)
    fit = [random.random() for _ in pop]
    order = sorted(range(len(pop)), key=lambda i: -fit[i])
    pop = [pop[i] for i in order]
    fit = [fit[i] for i in order]
    new = search.breed(pop, fit, basis, size=40, elitism=4, intensity=0.3,
                       unique=True)
    keys = [t.key() for t in new]
    assert len(set(keys)) == len(keys)
    obj = np.array([[f, -t.size()] for f, t in zip(fit, pop)])
    r, c = nsga2.rank_and_crowding(obj)
    off = search.breed_nsga2(pop, r, c, basis, size=40, intensity=0.3,
                             unique=True)
    parents = {t.key() for t in pop}
    assert not parents & {t.key() for t in off}
    assert all(search.within_budget(t, basis) for t in new + off)


def test_tune_constants_never_worsens(basis):
    X = np.random.default_rng(0).normal(size=(300, 4))
    y = np.exp(X[:, 0] * 0.3) - np.log(np.abs(X[:, 1]) + 1.7)
    def score(t):
        p = t.evaluate_batch(X)
        return -float(np.mean((p - y) ** 2)) if np.all(np.isfinite(p)) else -1e9
    random.seed(0)
    for t in _pop(basis, 20):
        s0 = score(t)
        t2, s2 = search.tune_constants(t, score, steps=10)
        assert s2 >= s0 and s2 == pytest.approx(score(t2))


def test_cache_returns_exact_values_and_counts(basis):
    pop = _pop(basis, 10)
    pop = pop + [pop[0].clone(), pop[1].clone()]
    calls = []
    def f(t):
        calls.append(1)
        return float(t.size())
    c = FitnessCache()
    v1 = c.map(pop, 'g0', f)
    assert v1 == [float(t.size()) for t in pop]
    assert len(calls) == len({t.key() for t in pop})
    v2 = c.map(pop, 'g0', f)
    assert v2 == v1 and len(calls) == len({t.key() for t in pop})
    c.map(pop, 'g1', f)     # another context is another dataset
    assert len(calls) == 2 * len({t.key() for t in pop})


def test_rng_state_roundtrip_through_json():
    random.seed(5)
    rng = np.random.default_rng(9)
    s_py = json.loads(json.dumps(checkpoint.python_rng_state()))
    s_np = json.loads(json.dumps(checkpoint.numpy_rng_state(rng)))
    a = [random.random() for _ in range(5)], rng.random(5).tolist()
    checkpoint.set_python_rng_state(s_py)
    rng2 = checkpoint.numpy_rng_from_state(s_np)
    b = [random.random() for _ in range(5)], rng2.random(5).tolist()
    assert a == b


def test_capture_serialise_restore(tmp_path, basis):
    pop = _pop(basis, 12)
    arch = ParetoArchive()
    arch.update_many(pop, [float(i) for i in range(len(pop))])
    hist = [1.0, 2.0]
    cap = checkpoint.capture(7, pop, counters={'history': hist, 'best': pop[3],
                                               'p': 2, 'v': -np.inf},
                             rng=np.random.default_rng(1), archive=arch)
    hist.append(3.0)                        # mutation after capture must not leak
    path = tmp_path / 's.json'
    checkpoint.atomic_write_json(str(path), checkpoint.serialise(
        cap, phase='data', signature={'x': 1}, generations=50))
    st = checkpoint.restore(checkpoint.read_json(str(path)), basis)
    assert st['generation'] == 7 and st['generations'] == 50
    assert [t.key() for t in st['population']] == [t.key() for t in pop]
    assert st['counters']['history'] == [1.0, 2.0]
    assert st['counters']['best'].key() == pop[3].key()
    assert st['counters']['v'] == -np.inf
    assert {k: v[0] for k, v in st['archive'].best.items()} == \
        {k: v[0] for k, v in arch.best.items()}


def test_signature_check():
    assert checkpoint.check_signature({'a': 1}, {'a': 1}) == []
    with pytest.raises(checkpoint.SignatureMismatch):
        checkpoint.check_signature({'a': 1, 'data': 'x'}, {'a': 1, 'data': 'y'})
    assert checkpoint.check_signature({'data': 'x'}, {'data': 'y'},
                                      allow=('data',)) == ['data']


def test_checkpointer_layout_and_points(tmp_path, basis):
    d = str(tmp_path / 'run')
    ck = checkpoint.RunCheckpointer(d, phase='data', signature={},
                                    generations=30, every=5, snapshot_every=10,
                                    keep_best=2, hof_size=3)
    pop = _pop(basis, 6)
    for gen in range(30):
        cap = checkpoint.capture(gen, pop, counters={})
        improved = gen in (1, 4, 12, 20)
        if improved:
            ck.record(pop[gen % 6], float(gen), gen)
        ck.on_generation(cap, improved=improved)
    ck.finish(checkpoint.capture(30, pop, counters={}))
    pts = checkpoint.list_points(d)
    kinds = {(p['kind'], p['generation']) for p in pts}
    assert ('snapshot', 0) in kinds and ('snapshot', 20) in kinds
    assert {g for k, g in kinds if k == 'best'} == {12, 20}      # keep_best=2
    assert ('latest', 30) in kinds
    assert checkpoint.resolve_point(d, 'best').endswith('gen_000020.json')
    assert checkpoint.resolve_point(d, 15).endswith('gen_000012.json')
    assert checkpoint.resolve_point(d, 'latest').endswith('latest.json')
    hof = checkpoint.load_hall_of_fame(d, basis)
    assert len(hof) <= 3 and hof[0]['score'] == 20.0
    assert isinstance(hof[0]['node'], Node)
    with pytest.raises(FileNotFoundError):
        checkpoint.resolve_point(str(tmp_path / 'nope'))
