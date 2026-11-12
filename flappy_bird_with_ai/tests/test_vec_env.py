"""vec_env.play_population gives exactly the scores of separate FlappyEnv games."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip('torch')

import config                     # noqa: E402
from env import FlappyEnv         # noqa: E402
from ga_net import GANet          # noqa: E402
from vec_env import play_population  # noqa: E402


def _nets(n, seed=0):
    torch.manual_seed(seed)
    nets = [GANet() for _ in range(n)]
    for k, net in enumerate(nets):           # a spread of behaviours
        with torch.no_grad():
            net.fc2.bias.add_((k - n / 2) * 0.3)
    path = os.path.join(os.path.dirname(HERE), 'models', 'best_ga.pt')
    if os.path.exists(path):
        from storage import load_ga_model
        nets.append(load_ga_model(path))
    return nets


def _batched(nets):
    W1 = np.stack([n.fc1.weight.detach().numpy() for n in nets]).astype(np.float32)
    b1 = np.stack([n.fc1.bias.detach().numpy() for n in nets]).astype(np.float32)
    W2 = np.stack([n.fc2.weight.detach().numpy()[0] for n in nets]).astype(np.float32)
    b2 = np.array([n.fc2.bias.item() for n in nets], dtype=np.float32)

    def policy(st, alive_idx):
        h = np.maximum(np.einsum('nij,nj->ni', W1[alive_idx], st) + b1[alive_idx], 0)
        z = np.einsum('ni,ni->n', W2[alive_idx], h) + b2[alive_idx]
        return z > 0.0            # sigmoid(z) > 0.5
    return policy


@pytest.mark.parametrize('profile', ['classic', 'ramp'])
def test_matches_scalar_env(profile):
    nets = _nets(12)
    for seed in (5000, 5001, 3003):
        sc, fr = play_population(_batched(nets), len(nets), seed,
                                 max_frames=4000, profile=profile)
        for k, net in enumerate(nets):
            env = FlappyEnv(seed=seed, profile=profile)
            f = 0
            while not env.done and f < 4000:
                env.step(net.get_action(env.get_state()))
                f += 1
            assert (sc[k], fr[k]) == (env.score, f), (profile, seed, k)


def test_classic_profile_keeps_the_pipe_stream():
    env = FlappyEnv(seed=5000, profile='classic')
    gaps = []
    for _ in range(900):
        env.step(0 if env.bird_y < 300 else 1)
    gaps = {p['gap'] for p in env.pipes}
    assert gaps == {config.PIPE_GAP}
