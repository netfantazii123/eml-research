"""An episode depends only on its seed and actions, not on its batch mates."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from medium_tasks.tasks import TASKS, play  # noqa: E402


@pytest.mark.parametrize('name', sorted(TASKS))
def test_episode_independent_of_batch(name):
    task = TASKS[name]()
    w = np.random.default_rng(0).normal(size=len(task.feature_names))
    pol = lambda X: X @ w            # noqa: E731
    together = play(task, pol, [5, 6, 7, 8], batch=4)
    alone = np.array([play(task, pol, [s], batch=1)[0] for s in (5, 6, 7, 8)])
    assert np.allclose(together, alone)


@pytest.mark.parametrize('name', sorted(TASKS))
def test_collect_rows_match_groups(name):
    task = TASKS[name]()
    pol = lambda X: X[:, 0]          # noqa: E731
    _, d = play(task, pol, [1, 2], collect=True, teacher_fn=pol)
    assert len(d['X']) == len(d['T']) == len(d['groups'])
    assert d['X'].shape[1] == len(task.feature_names)
