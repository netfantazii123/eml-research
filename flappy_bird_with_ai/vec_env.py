"""
vec_env.py — many birds on ONE seeded pipe stream, simulated together.

The pipes of a FlappyEnv(seed) do not depend on the bird, so N policies played
on the same seed are N birds flying through the same pipes. Simulating them in
one numpy step gives exactly the scores N separate FlappyEnv(seed) games give
(tests/test_vec_env.py checks that bird by bird), at a fraction of the cost.
That is what makes training an oracle on the 'ramp' profile, and scoring whole
GA populations without a frame cap, affordable.
"""

from __future__ import annotations

import numpy as np

import config
from config import (SCREEN_W, SCREEN_H, PIPE_GAP, PIPE_SPEED, PIPE_FREQ,
                    GRAVITY, FLAP_POWER, BIRD_X, BIRD_RADIUS, PIPE_WIDTH)


def play_population(policy, n_birds: int, seed: int, *,
                    max_frames: int | None = None,
                    profile: str | None = None):
    """
    Fly n_birds through the pipe stream of FlappyEnv(seed).

    policy(states) -> actions: states is (n_alive, 4) float32 in FlappyEnv's
    get_state() layout, plus `alive_idx` keyword with the bird indices; returns
    an int/bool array of flaps.
    Returns (score, frames) arrays of shape (n_birds,): pipes passed and frames
    survived, as FlappyEnv reports them after `while not done and f < max`.
    """
    prof = config.FLAPPY_PROFILES[profile or config.FLAPPY_ENV]
    rng = np.random.default_rng(seed)
    margin = 80
    pipes_x: list[float] = []
    pipes_gy: list[float] = []
    pipes_gap: list[int] = []
    pipes_scored: list[np.ndarray] = []      # per pipe, which birds scored it
    n_spawned = 0

    def spawn():
        nonlocal n_spawned
        gap = int(max(prof['gap_min'],
                      prof['gap'] - prof['gap_shrink'] * n_spawned))
        gy = rng.integers(margin + PIPE_GAP // 2,
                          SCREEN_H - margin - PIPE_GAP // 2)
        n_spawned += 1
        pipes_x.append(float(SCREEN_W))
        pipes_gy.append(float(gy))
        pipes_gap.append(gap)
        pipes_scored.append(np.zeros(n_birds, dtype=bool))

    y = np.full(n_birds, SCREEN_H / 2, dtype=np.float64)
    vel = np.zeros(n_birds, dtype=np.float64)
    alive = np.ones(n_birds, dtype=bool)
    score = np.zeros(n_birds, dtype=np.int64)
    frames = np.zeros(n_birds, dtype=np.int64)
    spawn()
    frame = 0
    limit = max_frames if max_frames is not None else np.iinfo(np.int64).max

    while alive.any() and frame < limit:
        idx = np.flatnonzero(alive)
        # state, exactly as FlappyEnv.get_state()
        nxt = next((i for i, x in enumerate(pipes_x)
                    if x + PIPE_WIDTH > BIRD_X), None)
        if nxt is None:
            dx, gy = 1.0, 0.5
        else:
            dx = (pipes_x[nxt] - BIRD_X) / SCREEN_W
            gy = pipes_gy[nxt] / SCREEN_H
        st = np.empty((len(idx), 4), dtype=np.float32)
        st[:, 0] = (y[idx] / SCREEN_H).astype(np.float32)
        st[:, 1] = (vel[idx] / 10.0).astype(np.float32)
        st[:, 2] = np.float32(dx)
        st[:, 3] = np.float32(gy)
        act = np.asarray(policy(st, alive_idx=idx)).astype(bool).reshape(-1)

        v = vel[idx]
        v = np.where(act, float(FLAP_POWER), v) + GRAVITY
        vel[idx] = v
        y[idx] = y[idx] + v
        frame += 1
        frames[idx] += 1
        for i in range(len(pipes_x)):
            pipes_x[i] -= PIPE_SPEED
        if frame % PIPE_FREQ == 0:
            spawn()
        keep = [i for i, x in enumerate(pipes_x) if x + PIPE_WIDTH > 0]
        if len(keep) != len(pipes_x):
            pipes_x[:] = [pipes_x[i] for i in keep]
            pipes_gy[:] = [pipes_gy[i] for i in keep]
            pipes_gap[:] = [pipes_gap[i] for i in keep]
            pipes_scored[:] = [pipes_scored[i] for i in keep]

        yy = y[idx]
        dead = (yy - BIRD_RADIUS <= 0) | (yy + BIRD_RADIUS >= SCREEN_H)
        for x, g, gp in zip(pipes_x, pipes_gy, pipes_gap):
            if x < BIRD_X + BIRD_RADIUS and x + PIPE_WIDTH > BIRD_X - BIRD_RADIUS:
                dead |= (yy - BIRD_RADIUS < g - gp // 2) | \
                        (yy + BIRD_RADIUS > g + gp // 2)
        alive[idx[dead]] = False
        surv = idx[~dead]
        for x, sc in zip(pipes_x, pipes_scored):
            if x + PIPE_WIDTH < BIRD_X:
                new = surv[~sc[surv]]
                if len(new):
                    sc[new] = True
                    score[new] += 1
    return score, frames
