"""A single `distill` run reports TEST over EML_TEST_GAMES and is seeded.

Before the 2026-11 review full_distill defaulted its holdout to
EML_INGAME_GAMES (3 games, the GAME-phase fitness size) and never seeded the GA.
"""

import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

pytest.importorskip('torch')

import config      # noqa: E402
import pipeline    # noqa: E402


class _Stop(Exception):
    pass


def _patched(monkeypatch, tmp_path, seen):
    monkeypatch.setattr(pipeline, '_load_oracle', lambda o, d, e: object())
    monkeypatch.setattr(pipeline, '_get_dataset',
                        lambda *a, **k: (np.zeros((2, 2)), np.zeros(2),
                                         np.zeros(2), np.ones(2)))

    def fake_distill(*a, **k):
        seen.setdefault('draws', []).append(random.random())
        return {'trees': [None]}

    def fake_holdout(tree, oracle, device, n_games):
        seen['n_games'] = n_games
        raise _Stop

    monkeypatch.setattr(pipeline, 'distill', fake_distill)
    monkeypatch.setattr(pipeline, 'evaluate_holdout', fake_holdout)


def test_holdout_uses_test_games_and_run_is_seeded(monkeypatch, tmp_path):
    seen: dict = {}
    _patched(monkeypatch, tmp_path, seen)
    for _ in range(2):
        random.seed(12345)                    # unrelated prior state
        random.random()
        with pytest.raises(_Stop):
            pipeline.full_distill(device='cpu', run_dir=str(tmp_path / 'r'))
    assert seen['n_games'] == config.EML_TEST_GAMES
    assert seen['draws'][0] == seen['draws'][1]
    random.seed(config.EML_GA_SEED_BASE)
    assert seen['draws'][0] == random.random()
