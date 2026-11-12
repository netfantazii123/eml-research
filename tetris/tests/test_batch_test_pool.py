"""
batch_distill scores EVERY seed on TEST, not only the VAL winner.

Synthetic dataset and an untrained CNN as the oracle: only the bookkeeping is
under test, so the budgets are tiny.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

torch = pytest.importorskip('torch')

import config                         # noqa: E402
import pipeline                       # noqa: E402
import runs as runs_mod               # noqa: E402
from env import TetrisEnv             # noqa: E402
from cnn_oracle import TetrisCNN      # noqa: E402
from dataset_collector import save_dataset  # noqa: E402


def _synthetic_dataset(path):
    names = config.FEATURE_NAMES
    W = np.zeros(config.N_AFTERSTATE_FEATURES)
    for n, v in {'holes': -4.0, 'agg_h': -0.5, 'cleared': 3.4}.items():
        W[names.index(n)] = v
    F, L, G, g = [], [], [], 0
    env = TetrisEnv(seed=0)
    obs = env.reset()
    for _ in range(40):
        if env.done:
            break
        legal = np.flatnonzero(obs['mask'] > 0)
        af = obs['afeats'][legal]
        X = np.zeros((len(legal), config.N_FEATURES))
        X[:, :af.shape[1]] = af
        X[:, af.shape[1]:] = obs['scalars'][7:14]
        z = af @ W
        F.append(X); L.append(z); G.append(np.full(len(legal), g)); g += 1
        obs, *_ = env.step_placement(int(legal[np.argmax(z)]))
    save_dataset(np.concatenate(F).astype(np.float32),
                 np.concatenate(L).astype(np.float32), np.concatenate(G), path)


def test_every_seed_gets_a_test_score(tmp_path, monkeypatch):
    monkeypatch.setattr(runs_mod, 'RUNS_DIR', str(tmp_path / 'runs'), raising=False)
    monkeypatch.setattr(config, 'MODELS_DIR', str(tmp_path / 'models'))
    monkeypatch.setattr(config, 'RESULTS_DIR', str(tmp_path / 'results'))
    monkeypatch.setattr(config, 'EML_INGAME_MAX_PLACEMENTS', 15)
    monkeypatch.setattr(config, 'EML_JOINT_GENERATIONS', 1)
    monkeypatch.setattr(config, 'EML_JOINT_POPULATION', 6)
    monkeypatch.setattr(config, 'EML_VAL_GAMES', 1)
    monkeypatch.setattr(config, 'EML_INGAME_GAMES', 1)
    monkeypatch.setattr(config, 'EML_PARETO_MAX_POINTS', 2)
    data = str(tmp_path / 'ds.npz')
    _synthetic_dataset(data)
    oracle = TetrisCNN().eval()
    s = pipeline.batch_distill(n_configs=1, n_seeds=2, oracle=oracle,
                               device='cpu', reuse_data=True, data_path=data,
                               data_generations=3, data_population=12,
                               eval_games=2)
    assert len(s['variants']) == 2
    for v in s['variants']:
        assert v['test_lines'] is not None
    row = s['per_config'][0]
    tests = [v['test_lines'] for v in s['variants']]
    assert row['test_mean'] == pytest.approx(np.mean(tests))
    assert row['test_min'] == pytest.approx(min(tests))
