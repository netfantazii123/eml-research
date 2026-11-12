"""
FlappyEnv — a headless Flappy Bird environment with a gym-like interface.
~330K FPS on CPU.
"""

import numpy as np
from config import (
    SCREEN_W, SCREEN_H, PIPE_GAP, PIPE_SPEED, PIPE_FREQ,
    GRAVITY, FLAP_POWER, BIRD_X, BIRD_RADIUS, PIPE_WIDTH,
)


class FlappyEnv:
    """
    A headless Flappy Bird environment with a gym-like interface.

    Interface:
        reset()    -> state (np.ndarray shape (4,))
        step(action) -> (state, reward, done)
        get_state() -> state

    State (4 normalized numbers):
        [bird_y/H, bird_vel/10, dx_to_pipe/W, gap_y/H]

    Action:
        1 = flap, 0 = do nothing (fall).

    Rewards:
        +0.1  per surviving frame
        +1.0  for passing a pipe
        -1.0  for a collision (death)
    """

    def __init__(self, seed: int | None = None):
        """
        seed: pipe-layout seed. None keeps the old behaviour (a fresh random
        layout every episode) and is right for interactive play; evaluation must
        pass an explicit seed. Without one, two formulas compared over N games
        never play the same pipes, so at the handful of games a GA fitness can
        afford the ranking is mostly which individual drew the easier layouts.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.bird_y: float = 0.0
        self.bird_vel: float = 0.0
        self.pipes: list[dict] = []
        self.score: int = 0
        self.frame: int = 0
        self.done: bool = False
        self.reset()

    # ── Public interface ─────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state. Returns the state.

        A seeded env restarts its pipe stream here, so every episode of that env
        is the same episode. Evaluating N different games therefore means N envs
        with N seeds, not N resets of one env.
        """
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        self.bird_y = SCREEN_H / 2
        self.bird_vel = 0.0
        self.pipes = []
        self.score = 0
        self.frame = 0
        self.done = False
        self._spawn_pipe()
        return self.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Run one simulation frame.

        Args:
            action: 1 = flap, 0 = nothing.

        Returns:
            (state, reward, done)
        """
        if self.done:
            return self.get_state(), 0.0, True

        if action == 1:
            self.bird_vel = FLAP_POWER

        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel

        self.frame += 1
        for pipe in self.pipes:
            pipe['x'] -= PIPE_SPEED
        if self.frame % PIPE_FREQ == 0:
            self._spawn_pipe()
        self.pipes = [p for p in self.pipes if p['x'] + PIPE_WIDTH > 0]

        reward = 0.1
        if self._check_collision():
            self.done = True
            reward = -1.0
            return self.get_state(), reward, self.done

        for pipe in self.pipes:
            if not pipe['scored'] and pipe['x'] + PIPE_WIDTH < BIRD_X:
                pipe['scored'] = True
                self.score += 1
                reward = 1.0

        return self.get_state(), reward, self.done

    def get_state(self) -> np.ndarray:
        """Normalized state vector [4 numbers]."""
        next_pipe = self._get_next_pipe()
        if next_pipe is None:
            dx = 1.0
            gap_y = 0.5
        else:
            dx = (next_pipe['x'] - BIRD_X) / SCREEN_W
            gap_y = next_pipe['gap_y'] / SCREEN_H

        return np.array([
            self.bird_y / SCREEN_H,
            self.bird_vel / 10.0,
            dx,
            gap_y
        ], dtype=np.float32)

    # ── Private methods ──────────────────────────────────────────────────

    def _spawn_pipe(self):
        """Create a new pipe with a random gap_y."""
        margin = 80
        gap_y = self.rng.integers(
            margin + PIPE_GAP // 2,
            SCREEN_H - margin - PIPE_GAP // 2
        )
        self.pipes.append({
            'x': float(SCREEN_W),
            'gap_y': float(gap_y),
            'scored': False
        })

    def _get_next_pipe(self) -> dict | None:
        """The nearest pipe ahead of the bird."""
        for pipe in self.pipes:
            if pipe['x'] + PIPE_WIDTH > BIRD_X:
                return pipe
        return None

    def _check_collision(self) -> bool:
        """Collision with the floor/ceiling and the pipes."""
        if self.bird_y - BIRD_RADIUS <= 0 or self.bird_y + BIRD_RADIUS >= SCREEN_H:
            return True

        for pipe in self.pipes:
            if pipe['x'] < BIRD_X + BIRD_RADIUS and \
               pipe['x'] + PIPE_WIDTH > BIRD_X - BIRD_RADIUS:
                top_pipe_bottom = pipe['gap_y'] - PIPE_GAP // 2
                bottom_pipe_top = pipe['gap_y'] + PIPE_GAP // 2
                if self.bird_y - BIRD_RADIUS < top_pipe_bottom or \
                   self.bird_y + BIRD_RADIUS > bottom_pipe_top:
                    return True

        return False
