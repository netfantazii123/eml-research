"""
TetrisEnv — a headless Tetris environment with a gym-like interface.

Two control modes:

1. PLACEMENT (the main one, for the oracle and EML) — an action is a piece
   placement:
       action = rot * BOARD_W + xi
   where rot is the rotation and xi is the column of the piece's leftmost
   occupied cell. One step_placement() = rotate + shift + hard drop + lock +
   clear. The reward is dense and issued on every step. Illegal actions are
   closed off by a mask.

       reset()                   -> obs (placement, see get_placement_state)
       step_placement(a)         -> (obs, reward, done, info)
       placement_mask()          -> (N_PLACEMENTS,) float32 (1 = legal)
       afterstate(rot, xi)       -> (board_after, cleared, landing_h), no mutation

2. FRAME (legacy, for the visual demo) — 6 per-frame actions:
       step(action) — 0 NO-OP, 1 left, 2 right, 3 rotate CW, 4 soft, 5 hard
       get_state()  — obs for frame-by-frame viewing

Bag-7 randomization, line clears. No rendering — pure NumPy logic.
"""

import numpy as np

import config
from config import (
    BOARD_W, BOARD_H, BOARD_BUFFER, BOARD_H_TOTAL,
    GRAVITY_TICKS, MAX_EPISODE_FRAMES, N_PIECES, N_PLACEMENTS,
)
# The reward constants are NOT imported by name but read through config.* in
# step(): this lets the GUI change them live (config.REWARD_W_HOLES = ... during
# training).
from pieces import piece_cells, SPAWN_X, SPAWN_Y, DISTINCT_ROTATIONS, SHAPES
import features as features_mod


# Precomputed geometry: _GEOM[ptype][rot] = (cells, c_min, span, r_max),
# where span = c_max - c_min (piece width in occupied columns minus 1).
_GEOM: list[list[tuple[tuple[tuple[int, int], ...], int, int, int]]] = []
for _pt in range(N_PIECES):
    _rots = []
    for _r in range(4):
        _cells = SHAPES[_pt][_r]
        _cols = [c for (_, c) in _cells]
        _rows = [r for (r, _) in _cells]
        _rots.append((_cells, min(_cols), max(_cols) - min(_cols), max(_rows)))
    _GEOM.append(_rots)


class TetrisEnv:
    """Headless Tetris. Board (24x10): rows 0..3 are the spawn buffer, 4..23 visible."""

    def __init__(self, seed: int | None = None,
                 max_placements: int | None = None,
                 profile: str | None = None):
        """
        Args:
            seed: bag randomization seed.
            max_placements: placement limit per episode; None -> from config,
                0 -> no limit (Play mode: play until a real game over).
        """
        self._max_placements = max_placements
        self.profile = config.TETRIS_PROFILES[profile or config.TETRIS_ENV]
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.garbage_rows = 0
        self.board = np.zeros((BOARD_H_TOTAL, BOARD_W), dtype=np.int8)
        self._bag: list[int] = []
        self.cur_type = 0
        self.cur_rot = 0
        self.cur_x = SPAWN_X
        self.cur_y = SPAWN_Y
        self.next_type = 0
        self.gravity_counter = 0
        self.score = 0          # total lines cleared in the episode
        self.frame = 0          # frames (frame mode)
        self.placements = 0     # pieces placed (placement mode)
        self.done = False
        self._lock_dholes = 0   # metric deltas of the last lock (reward shaping)
        self._lock_dbump = 0
        self._lock_dagg = 0
        self.reset()

    # ── Public interface ─────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset the environment. Returns the first placement obs."""
        self.board[:] = 0
        self._bag = []
        self.score = 0
        self.frame = 0
        self.placements = 0
        self.gravity_counter = 0
        self.done = False
        self.garbage_rows = 0
        # Garbage holes: a separate stream, so the pieces do not change. Like
        # the piece RNG it is NOT re-seeded by reset() of a seeded env, and it
        # is created on the first reset, which the constructor makes.
        if not hasattr(self, '_grng'):
            self._grng = np.random.default_rng(
                None if self._seed is None else [int(self._seed), 0x6A7B])
        self.next_type = self._draw_from_bag()
        self._spawn()   # on a clean board the spawn cannot hit game over
        return self.get_placement_state()

    # ── PLACEMENT mode ───────────────────────────────────────────────────

    def placement_mask(self) -> np.ndarray:
        """
        Mask of legal placements of the current piece, (N_PLACEMENTS,) float32.

        Legal means: the rotation is geometrically unique for the piece, the
        piece fits the board by width, and it does not collide at spawn height.
        """
        mask = np.zeros(N_PLACEMENTS, dtype=np.float32)
        for rot in range(DISTINCT_ROTATIONS[self.cur_type]):
            cells, c_min, span, _ = _GEOM[self.cur_type][rot]
            for xi in range(BOARD_W - span):
                x = xi - c_min
                if not self._collides(self.cur_type, rot, x, SPAWN_Y):
                    mask[rot * BOARD_W + xi] = 1.0
        return mask

    def afterstate(self, rot: int, xi: int) -> tuple[np.ndarray, int, int] | None:
        """
        A trial placement WITHOUT mutating the environment.

        Returns:
            (board_after, cleared, landing_h), or None if the placement is
            illegal. board_after is a copy of the board after lock and clear.
        """
        cells, c_min, span, r_max = _GEOM[self.cur_type][rot]
        x = xi - c_min
        if xi + span >= BOARD_W or self._collides(self.cur_type, rot, x, SPAWN_Y):
            return None
        y = self._drop_y(self.cur_type, rot, x, SPAWN_Y)
        board = self.board.copy()
        for (r, c) in cells:
            board[y + r, x + c] = 1
        landing_h = BOARD_H_TOTAL - (y + r_max + 1)
        full = np.all(board == 1, axis=1)
        cleared = int(np.count_nonzero(full))
        if cleared:
            kept = board[~full]
            board = np.zeros_like(board)
            board[cleared:] = kept
        return board, cleared, landing_h

    def afterstate_features(self, rot: int, xi: int) -> np.ndarray | None:
        """19 afterstate features for EML (None if the placement is illegal)."""
        result = self.afterstate(rot, xi)
        if result is None:
            return None
        board, cleared, landing_h = result
        return features_mod.extract_afterstate(board, cleared, landing_h)

    def all_afterstate_features(self,
                                mask: np.ndarray | None = None) -> np.ndarray:
        """
        Afterstate features of ALL legal placements, (N_PLACEMENTS, 19).

        Illegal rows are zeros. Used by the oracle: the policy head scores every
        placement from its afterstate features. Feature extraction is vectorized
        over the candidates (extract_afterstate_batch).
        """
        if mask is None:
            mask = self.placement_mask()
        feats = np.zeros((N_PLACEMENTS, features_mod.N_AFTERSTATE),
                         dtype=np.float32)
        legal = np.flatnonzero(mask > 0)
        if len(legal) == 0:
            return feats
        boards = np.empty((len(legal), BOARD_H_TOTAL, BOARD_W), dtype=np.int8)
        cleared = np.empty(len(legal), dtype=np.int32)
        landing = np.empty(len(legal), dtype=np.int32)
        for k, a in enumerate(legal):
            rot, xi = divmod(int(a), BOARD_W)
            boards[k], cleared[k], landing[k] = self.afterstate(rot, xi)
        feats[legal] = features_mod.extract_afterstate_batch(
            boards, cleared, landing)
        return feats

    def step_placement(self, action: int) -> tuple[dict, float, bool, dict]:
        """
        Place a piece: action = rot * BOARD_W + xi. One step = one lock.

        An illegal action (which does not occur with a correct mask) is treated
        as a loss — the network is required to respect the mask.
        """
        if self.done:
            return self.get_placement_state(), 0.0, True, {'score': self.score}

        rot, xi = divmod(int(action), BOARD_W)
        cells, c_min, span, _ = _GEOM[self.cur_type][rot]
        x = xi - c_min
        if (rot >= DISTINCT_ROTATIONS[self.cur_type] or xi + span >= BOARD_W
                or self._collides(self.cur_type, rot, x, SPAWN_Y)):
            self.done = True
            return (self.get_placement_state(), config.REWARD_GAME_OVER, True,
                    {'score': self.score, 'lines': 0, 'illegal': True})

        self.cur_rot = rot
        self.cur_x = x
        self.cur_y = self._drop_y(self.cur_type, rot, x, SPAWN_Y)
        n_cleared = self._lock_piece()
        self.placements += 1
        overflow = self._maybe_add_garbage()

        reward = config.REWARD_PIECE
        if n_cleared > 0:
            reward += config.REWARD_LINE_BASE * (n_cleared ** 2)
            self.score += n_cleared
        shaping = (config.REWARD_W_HOLES * max(0, self._lock_dholes)
                   + config.REWARD_W_BUMP * max(0, self._lock_dbump)
                   + config.REWARD_W_AGG * max(0, self._lock_dagg))
        reward -= min(shaping, config.REWARD_SHAPING_CLIP)

        cap = (self._max_placements if self._max_placements is not None
               else config.MAX_EPISODE_PLACEMENTS)
        if overflow or not self._spawn():
            self.done = True
            reward += config.REWARD_GAME_OVER
        elif cap > 0 and self.placements >= cap:
            self.done = True

        return (self.get_placement_state(), reward, self.done,
                {'score': self.score, 'lines': n_cleared})

    def _maybe_add_garbage(self) -> bool:
        """Garbage profile: push a holed row in from the bottom when due.

        Returns True when the push shoves a block out of the top of the board
        (game over)."""
        p = self.profile
        if p is None:
            return False
        every = max(p['min_every'],
                    p['every'] - self.placements // p['faster_every'])
        if self.placements % every:
            return False
        overflow = bool(self.board[0].any())
        self.board[:-1] = self.board[1:]
        row = np.ones(BOARD_W, dtype=self.board.dtype)
        row[int(self._grng.integers(BOARD_W))] = 0
        self.board[-1] = row
        self.garbage_rows += 1
        return overflow

    def get_placement_state(self) -> dict:
        """
        Observation for the placement oracle.

        'grid':    (1, 24, 10) float32 — locked cells.
        'scalars': (14,) float32 — one-hot(cur 7) + one-hot(next 7).
        'mask':    (N_PLACEMENTS,) float32 — legal placements.
        'afeats':  (N_PLACEMENTS, 19) float32 — afterstate features of every
                   legal placement (illegal ones are zeros).
        """
        grid = np.zeros((1, BOARD_H_TOTAL, BOARD_W), dtype=np.float32)
        grid[0] = self.board
        scalars = np.zeros(14, dtype=np.float32)
        scalars[self.cur_type] = 1.0               # 0..6
        scalars[7 + self.next_type] = 1.0          # 7..13
        mask = self.placement_mask()
        return {'grid': grid, 'scalars': scalars, 'mask': mask,
                'afeats': self.all_afterstate_features(mask)}

    # ── FRAME mode (legacy, for the visual demo) ─────────────────────────

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        """One simulation frame (6 actions, gravity). For the demo/visualization."""
        if self.done:
            return self.get_state(), 0.0, True, {'score': self.score}

        reward = config.REWARD_SURVIVE   # usually 0.0 — survival is rewarded per event
        locked = False
        n_cleared = 0

        # ── Apply the action ────────────────────────────────────────────
        if action == 1:                              # left
            self._try_move(dx=-1, dy=0)
        elif action == 2:                            # right
            self._try_move(dx=1, dy=0)
        elif action == 3:                            # rotate CW
            self._try_rotate()
        elif action == 4:                            # soft drop
            if self._try_move(dx=0, dy=1):
                self.gravity_counter = 0             # a soft drop resets the timer
        elif action == 5:                            # hard drop
            while self._try_move(dx=0, dy=1):
                pass
            n_cleared = self._lock_piece()
            locked = True

        # ── Gravity (unless already locked by a hard drop) ───────────────
        if not locked:
            self.gravity_counter += 1
            if self.gravity_counter >= GRAVITY_TICKS:
                self.gravity_counter = 0
                if not self._try_move(dx=0, dy=1):
                    n_cleared = self._lock_piece()
                    locked = True

        # ── Lock reward: dense shaping + lines + spawning a new piece ────
        if locked:
            reward += config.REWARD_PIECE                # placed a piece and survived
            if n_cleared > 0:
                reward += config.REWARD_LINE_BASE * (n_cleared ** 2)
                self.score += n_cleared
            shaping = (config.REWARD_W_HOLES * max(0, self._lock_dholes)
                       + config.REWARD_W_BUMP * max(0, self._lock_dbump)
                       + config.REWARD_W_AGG * max(0, self._lock_dagg))
            reward -= min(shaping, config.REWARD_SHAPING_CLIP)
            if not self._spawn():
                self.done = True
                reward += config.REWARD_GAME_OVER

        self.frame += 1
        if self.frame >= MAX_EPISODE_FRAMES:
            self.done = True

        return self.get_state(), reward, self.done, {'score': self.score,
                                                     'lines': n_cleared}

    def get_state(self) -> dict:
        """
        Frame mode observation (legacy).

        'grid':    (3, 24, 10) float32 — locked / current piece / ghost.
        'scalars': (19,) float32 — one-hot(cur 7) + one-hot(next 7)
                   + one-hot(rot 4) + drop_progress(1).
        """
        grid = np.zeros((3, BOARD_H_TOTAL, BOARD_W), dtype=np.float32)
        grid[0] = self.board

        for (r, c) in piece_cells(self.cur_type, self.cur_rot):
            grid[1, self.cur_y + r, self.cur_x + c] = 1.0

        ghost_y = self._ghost_y()
        for (r, c) in piece_cells(self.cur_type, self.cur_rot):
            grid[2, ghost_y + r, self.cur_x + c] = 1.0

        scalars = np.zeros(19, dtype=np.float32)
        scalars[self.cur_type] = 1.0               # 0..6
        scalars[7 + self.next_type] = 1.0          # 7..13
        scalars[14 + self.cur_rot] = 1.0           # 14..17
        scalars[18] = self.gravity_counter / GRAVITY_TICKS

        return {'grid': grid, 'scalars': scalars}

    # ── Bag-7 randomization ──────────────────────────────────────────────

    def _draw_from_bag(self) -> int:
        """Draw the next piece from the bag of 7."""
        if not self._bag:
            self._bag = list(range(N_PIECES))
            self.rng.shuffle(self._bag)
        return self._bag.pop()

    def _spawn(self) -> bool:
        """
        Spawn the current piece from next_type at the start position.
        Returns False if the spawn collides (-> game over).
        """
        self.cur_type = self.next_type
        self.cur_rot = 0
        self.cur_x = SPAWN_X
        self.cur_y = SPAWN_Y
        self.next_type = self._draw_from_bag()
        self.gravity_counter = 0
        return not self._collides(self.cur_type, self.cur_rot,
                                  self.cur_x, self.cur_y)

    # ── Movement / rotation ──────────────────────────────────────────────

    def _collides(self, ptype: int, rot: int, x: int, y: int) -> bool:
        """Collision of a piece (type, rotation) at (x, y) with walls/floor/blocks."""
        for (r, c) in piece_cells(ptype, rot):
            br, bc = y + r, x + c
            if bc < 0 or bc >= BOARD_W or br >= BOARD_H_TOTAL or br < 0:
                return True
            if self.board[br, bc]:
                return True
        return False

    def _drop_y(self, ptype: int, rot: int, x: int, from_y: int) -> int:
        """The Y at which the piece stops when dropped from from_y."""
        y = from_y
        while not self._collides(ptype, rot, x, y + 1):
            y += 1
        return y

    def _try_move(self, dx: int, dy: int) -> bool:
        """Try to move the piece. True if it succeeded."""
        nx, ny = self.cur_x + dx, self.cur_y + dy
        if not self._collides(self.cur_type, self.cur_rot, nx, ny):
            self.cur_x, self.cur_y = nx, ny
            return True
        return False

    def _try_rotate(self) -> bool:
        """Clockwise rotation (without wall kicks). True if it succeeded."""
        nrot = (self.cur_rot + 1) % 4
        if not self._collides(self.cur_type, nrot, self.cur_x, self.cur_y):
            self.cur_rot = nrot
            return True
        return False

    def _ghost_y(self) -> int:
        """The Y at which the piece will lock on a hard drop."""
        return self._drop_y(self.cur_type, self.cur_rot, self.cur_x, self.cur_y)

    # ── Lock and line clearing ───────────────────────────────────────────

    def _lock_piece(self) -> int:
        """
        Lock the current piece onto the board and clear full lines.

        As a side effect it computes the board metric deltas (before placement
        -> after clear) and stores them in self._lock_d* for reward shaping.
        """
        before = self._board_metrics()
        for (r, c) in piece_cells(self.cur_type, self.cur_rot):
            self.board[self.cur_y + r, self.cur_x + c] = 1
        n = self._clear_lines()
        after = self._board_metrics()
        self._lock_dholes = after[0] - before[0]
        self._lock_dbump = after[1] - before[1]
        self._lock_dagg = after[2] - before[2]
        return n

    def _board_metrics(self) -> tuple[int, int, int]:
        """(holes, bumpiness, agg_h) of the current board — for shaping (features.py)."""
        heights = features_mod.column_heights(self.board)
        holes = features_mod.count_holes(self.board, heights)
        bumpiness = int(np.abs(np.diff(heights)).sum())
        agg_h = int(heights.sum())
        return holes, bumpiness, agg_h

    def _clear_lines(self) -> int:
        """Remove full rows and shift the rest down. Returns the count."""
        full = np.all(self.board == 1, axis=1)
        n = int(np.count_nonzero(full))
        if n == 0:
            return 0
        kept = self.board[~full]
        new_board = np.zeros_like(self.board)
        new_board[n:] = kept     # the surviving rows drop down
        self.board = new_board
        return n
