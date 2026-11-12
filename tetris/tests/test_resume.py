"""
Tetris DATA / GAME phases: the refactor changes nothing, resume replays exactly.

Uses a synthetic dataset (a hand-written evaluation function as the "oracle"),
so no trained CNN and no torch are needed.
"""

import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import config                         # noqa: E402
import eml_distiller as E             # noqa: E402
from env import TetrisEnv             # noqa: E402
from emlcore import checkpoint        # noqa: E402


@pytest.fixture(scope='module')
def data():
    names = config.FEATURE_NAMES
    W = np.zeros(config.N_AFTERSTATE_FEATURES)
    for n, v in {'holes': -4.0, 'bumpiness': -0.2, 'agg_h': -0.5,
                 'landing_h': -1.0, 'cleared': 3.4}.items():
        W[names.index(n)] = v
    rng = np.random.default_rng(0)
    F, L, G = [], [], []
    g = 0
    for ep in range(3):
        env = TetrisEnv(seed=ep)
        obs = env.reset()
        for _ in range(60):
            if env.done:
                break
            legal = np.flatnonzero(obs['mask'] > 0)
            af = obs['afeats'][legal]
            z = af @ W + rng.normal(0, 0.3, len(legal))
            X = np.zeros((len(legal), config.N_FEATURES))
            X[:, :af.shape[1]] = af
            X[:, af.shape[1]:] = obs['scalars'][7:14]
            F.append(X); L.append(z); G.append(np.full(len(legal), g)); g += 1
            obs, *_ = env.step_placement(int(legal[np.argmax(z)]))
    X = np.concatenate(F); g = np.concatenate(G)
    y = E._standardize_per_group(np.concatenate(L), g)
    w = np.ones(len(y))
    va = g % 5 == 0
    return X[~va], y[~va], w[~va], (X[va], y[va], w[va])


KW = dict(generations=12, population_size=30, elitism=4, patience=0,
          depth_penalty=0.05, subsample=1500)


@pytest.mark.parametrize('mo', [True, False])
def test_data_phase_resume_is_exact(tmp_path, data, mo):
    X, y, w, val = data
    random.seed(3)
    full = E.evolve_formula(X, y, w, val_data=val, multi_objective=mo,
                            checkpoint_dir=str(tmp_path / 'a'), **KW)
    n = [0]
    def stop():
        n[0] += 1
        return n[0] > 5
    random.seed(3)
    E.evolve_formula(X, y, w, val_data=val, multi_objective=mo,
                     checkpoint_dir=str(tmp_path / 'b'), should_stop=stop, **KW)
    random.seed(12345)            # the caller's RNG must not matter
    res = E.evolve_formula(X, y, w, val_data=val, multi_objective=mo,
                           resume=str(tmp_path / 'b'), **KW)
    assert res['history'] == full['history']
    assert res['best_tree'].key() == full['best_tree'].key()

    # From a chosen generation, and past the end.
    again = E.evolve_formula(X, y, w, val_data=val, multi_objective=mo,
                             resume=str(tmp_path / 'a'), resume_point=0, **KW)
    assert again['history'] == full['history']
    more = E.evolve_formula(X, y, w, val_data=val, multi_objective=mo,
                            resume=str(tmp_path / 'a'), extra_generations=4,
                            **KW)
    assert len(more['history']) == len(full['history']) + 4
    assert more['best_val_fitness'] >= full['best_val_fitness']


def test_data_phase_refuses_other_data(tmp_path, data):
    X, y, w, val = data
    random.seed(1)
    E.evolve_formula(X, y, w, val_data=val, checkpoint_dir=str(tmp_path), **KW)
    with pytest.raises(checkpoint.SignatureMismatch):
        E.evolve_formula(X[:-3], y[:-3], w[:-3], val_data=val,
                         resume=str(tmp_path), **KW)
    r = E.evolve_formula(X[:-3], y[:-3], w[:-3], val_data=val,
                         resume=str(tmp_path), allow_changes=('data',),
                         extra_generations=2, **KW)
    assert len(r['history']) == KW['generations'] + 2


def test_fine_tune_from_hall_of_fame(tmp_path, data):
    X, y, w, val = data
    random.seed(2)
    E.evolve_formula(X, y, w, val_data=val, checkpoint_dir=str(tmp_path), **KW)
    hof = checkpoint.load_hall_of_fame(str(tmp_path), E.TASK_BASIS)
    assert hof and hof == sorted(hof, key=lambda r: -r['score'])
    r = E.evolve_formula(X, y, w, val_data=val,
                         init_trees=[h['node'] for h in hof], **KW)
    # The hall of fame enters the population unchanged, so the fine-tuned run
    # can only match or beat the best formula it started from.
    assert r['best_val_fitness'] >= hof[0]['score'] - 1e-9


def test_game_phase_resume_and_parallel_are_exact(tmp_path):
    seed = E.EMLNode.from_dict({'kind': 'op', 'op': 'eml', 'children': [
        {'kind': 'const', 'value': 0.5},
        {'kind': 'op', 'op': 'eml', 'children': [
            {'kind': 'var', 'var_idx': 10}, {'kind': 'var', 'var_idx': 12}]}]})
    kw = dict(generations=5, population_size=10, elitism=2, patience=0,
              n_games=1, max_placements=40, val_games=2, verbose=False)
    random.seed(5)
    full = E.evolve_ingame(seed, **kw)
    n = [0]
    def stop():
        n[0] += 1
        return n[0] > 2
    random.seed(5)
    E.evolve_ingame(seed, checkpoint_dir=str(tmp_path), should_stop=stop, **kw)
    res = E.evolve_ingame(seed, resume=str(tmp_path), **kw)
    assert res['history'] == full['history']
    random.seed(5)
    par = E.evolve_ingame(seed, workers=2, **kw)
    assert par['history'] == full['history']
