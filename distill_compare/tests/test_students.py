"""Fast checks of the student interface, the target transforms and VIPER."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from distill_compare import students as S  # noqa: E402
from distill_compare.viper import _resample  # noqa: E402


def _toy_classify(n=400, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    W = rng.normal(size=(4, k))
    return X, X @ W * 3.0


def _toy_rank(n_groups=60, per=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * per, 3))
    T = 2.0 * X[:, 0] - X[:, 1] + 5.0      # offset must not matter
    g = np.repeat(np.arange(n_groups), per)
    return X, T, g


def test_zscore_rows_keeps_argmax():
    _, T = _toy_classify()
    assert np.array_equal(S.zscore_rows(T).argmax(1), T.argmax(1))


def test_zscore_groups_keeps_argmax_and_unit_scale():
    _, T, g = _toy_rank()
    Z = S.zscore_groups(T, g)
    for k in range(5):
        m = g == k
        assert np.argmax(Z[m]) == np.argmax(T[m])
        assert abs(Z[m].mean()) < 1e-9 and abs(Z[m].std() - 1) < 1e-6


def test_binary_margin_keeps_sign():
    T = np.array([[0.0, 2.0], [0.0, -1.0], [1.0, 0.5]])
    assert np.array_equal(np.sign(S.binary_margin(T)), [1, -1, -1])


def test_group_gap():
    T = np.array([1.0, 3.0, 0.0, 10.0, 4.0])
    g = np.array([0, 0, 0, 1, 1])
    assert np.allclose(S.group_gap(T, g), [3, 3, 3, 6, 6])


FAST = {'eml_data': dict(generations=8, population=40),
        'classic_gp': dict(generations=8, population=40),
        'gplearn_sr': dict(generations=2, population=50),
        'kd_mlp_h16': dict(epochs=20)}


@pytest.mark.parametrize('name', S.DEFAULT_STUDENTS)
def test_classify_interface(name):
    X, T = _toy_classify()
    s = S.make_student(name, seed=0, **FAST.get(name, {})).fit(X, T)
    sc = s.scores(X)
    assert sc.shape == (len(X), 3)
    c = s.complexity()
    assert {'size', 'size_unit', 'flops', 'transcendental'} <= set(c)
    assert c['size'] > 0


@pytest.mark.parametrize('name', S.DEFAULT_STUDENTS)
def test_rank_interface(name):
    X, T, g = _toy_rank()
    s = S.make_student(name, seed=0, **FAST.get(name, {})).fit(
        X, T, kind='rank', groups=g)
    assert s.scores(X).shape == (len(X),)


@pytest.mark.parametrize('name', ['tree_d6', 'linear', 'kd_mlp_h16'])
def test_easy_students_reach_high_fidelity(name):
    X, T = _toy_classify(n=1500, k=2)
    s = S.make_student(name, seed=0).fit(X, T)
    assert np.mean(s.actions(X) == T.argmax(1)) > 0.85


def test_linear_ranks_a_linear_teacher_exactly():
    X, T, g = _toy_rank()
    s = S.make_student('linear').fit(X, T, kind='rank', groups=g)
    p = s.scores(X)
    hits = [np.argmax(p[g == k]) == np.argmax(T[g == k]) for k in range(60)]
    assert np.mean(hits) > 0.95


def test_ga_student_is_reproducible():
    X, T = _toy_classify(k=2)
    a = S.make_student('eml_data', seed=3, **FAST['eml_data']).fit(X, T)
    b = S.make_student('eml_data', seed=3, **FAST['eml_data']).fit(X, T)
    assert a.describe() == b.describe()


def test_viper_resample_keeps_groups_whole():
    X, T, g = _toy_rank(n_groups=20, per=5)
    out = _resample('rank', {'X': X, 'T': T, 'groups': g}, 50,
                    np.random.default_rng(0))
    sizes = np.bincount(out['groups'])
    assert set(sizes) == {5}
