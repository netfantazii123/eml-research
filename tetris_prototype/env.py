"""
TetrisEnv — headless-среда Tetris с gym-like интерфейсом.

Frame-by-frame управление (6 дискретных действий), bag-7 рандомизация,
gravity, line-clear. Без отрисовки — чистая логика на NumPy для скорости.

Интерфейс:
    reset()           -> obs (dict: 'grid' (3,24,10), 'scalars' (19,))
    step(action: int) -> (obs, reward, done, info)
    get_features()    -> np.ndarray (28,) инженерных признаков для EML

Action space:
    0 = NO-OP   1 = left   2 = right
    3 = rotate CW   4 = soft drop   5 = hard drop
"""

import numpy as np

import config
from config import (
    BOARD_W, BOARD_H, BOARD_BUFFER, BOARD_H_TOTAL,
    GRAVITY_TICKS, MAX_EPISODE_FRAMES, N_PIECES,
    REWARD_SURVIVE, REWARD_GAME_OVER, REWARD_LINE_BASE, REWARD_HEIGHT_PENALTY,
)
from pieces import piece_cells, SPAWN_X, SPAWN_Y
import features as features_mod


class TetrisEnv:
    """Headless Tetris. Доска (24×10): строки 0..3 — буфер спавна, 4..23 — видимые."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((BOARD_H_TOTAL, BOARD_W), dtype=np.int8)
        self._bag: list[int] = []
        self.cur_type = 0
        self.cur_rot = 0
        self.cur_x = SPAWN_X
        self.cur_y = SPAWN_Y
        self.next_type = 0
        self.gravity_counter = 0
        self.score = 0          # суммарно очищенных линий за эпизод
        self.frame = 0
        self.done = False
        self.reset()

    # ── Публичный интерфейс ──────────────────────────────────────────────

    def reset(self) -> dict:
        """Сброс среды. Возвращает первый obs."""
        self.board[:] = 0
        self._bag = []
        self.score = 0
        self.frame = 0
        self.gravity_counter = 0
        self.done = False
        self.next_type = self._draw_from_bag()
        self._spawn()   # на чистой доске спавн не может зайти в game over
        return self.get_state()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        """Один кадр симуляции."""
        if self.done:
            return self.get_state(), 0.0, True, {'score': self.score}

        reward = REWARD_SURVIVE
        height_before = self._aggregate_height() if REWARD_HEIGHT_PENALTY else 0
        locked = False
        n_cleared = 0

        # ── Применить действие ──────────────────────────────────────────
        if action == 1:                              # left
            self._try_move(dx=-1, dy=0)
        elif action == 2:                            # right
            self._try_move(dx=1, dy=0)
        elif action == 3:                            # rotate CW
            self._try_rotate()
        elif action == 4:                            # soft drop
            if self._try_move(dx=0, dy=1):
                self.gravity_counter = 0             # soft drop сбрасывает таймер
        elif action == 5:                            # hard drop
            while self._try_move(dx=0, dy=1):
                pass
            n_cleared = self._lock_piece()
            locked = True

        # ── Gravity (если ещё не залочились хард-дропом) ─────────────────
        if not locked:
            self.gravity_counter += 1
            if self.gravity_counter >= GRAVITY_TICKS:
                self.gravity_counter = 0
                if not self._try_move(dx=0, dy=1):
                    n_cleared = self._lock_piece()
                    locked = True

        # ── Награда за лок/линии + спавн новой фигуры ────────────────────
        if locked:
            if n_cleared > 0:
                reward += REWARD_LINE_BASE * (n_cleared ** 2)
                self.score += n_cleared
            if REWARD_HEIGHT_PENALTY:
                delta_h = self._aggregate_height() - height_before
                reward -= REWARD_HEIGHT_PENALTY * max(0, delta_h)
            if not self._spawn():
                self.done = True
                reward += REWARD_GAME_OVER

        self.frame += 1
        if self.frame >= MAX_EPISODE_FRAMES:
            self.done = True

        return self.get_state(), reward, self.done, {'score': self.score,
                                                     'lines': n_cleared}

    def get_state(self) -> dict:
        """
        Observation для CNN.

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

    def get_features(self) -> np.ndarray:
        """28 инженерных признаков для EML (см. features.py)."""
        return features_mod.extract(self.board, self.cur_type, self.cur_rot)

    # ── Bag-7 рандомизация ───────────────────────────────────────────────

    def _draw_from_bag(self) -> int:
        """Достать следующую фигуру из bag-7."""
        if not self._bag:
            self._bag = list(range(N_PIECES))
            self.rng.shuffle(self._bag)
        return self._bag.pop()

    def _spawn(self) -> bool:
        """
        Заспавнить текущую фигуру из next_type на стартовой позиции.
        Возвращает False, если спавн коллизирует (→ game over).
        """
        self.cur_type = self.next_type
        self.cur_rot = 0
        self.cur_x = SPAWN_X
        self.cur_y = SPAWN_Y
        self.next_type = self._draw_from_bag()
        self.gravity_counter = 0
        return not self._collides(self.cur_type, self.cur_rot,
                                  self.cur_x, self.cur_y)

    # ── Движение / повороты ──────────────────────────────────────────────

    def _collides(self, ptype: int, rot: int, x: int, y: int) -> bool:
        """Коллизия фигуры (тип, поворот) в позиции (x, y) со стенами/полом/блоками."""
        for (r, c) in piece_cells(ptype, rot):
            br, bc = y + r, x + c
            if bc < 0 or bc >= BOARD_W or br >= BOARD_H_TOTAL or br < 0:
                return True
            if self.board[br, bc]:
                return True
        return False

    def _try_move(self, dx: int, dy: int) -> bool:
        """Попробовать сдвинуть фигуру. True, если удалось."""
        nx, ny = self.cur_x + dx, self.cur_y + dy
        if not self._collides(self.cur_type, self.cur_rot, nx, ny):
            self.cur_x, self.cur_y = nx, ny
            return True
        return False

    def _try_rotate(self) -> bool:
        """Поворот по часовой (без wall-kicks). True, если удалось."""
        nrot = (self.cur_rot + 1) % 4
        if not self._collides(self.cur_type, nrot, self.cur_x, self.cur_y):
            self.cur_rot = nrot
            return True
        return False

    def _ghost_y(self) -> int:
        """Y, на котором фигура зафиксируется при hard drop."""
        y = self.cur_y
        while not self._collides(self.cur_type, self.cur_rot, self.cur_x, y + 1):
            y += 1
        return y

    # ── Лок и очистка линий ──────────────────────────────────────────────

    def _lock_piece(self) -> int:
        """Зафиксировать текущую фигуру на доске и очистить полные линии."""
        for (r, c) in piece_cells(self.cur_type, self.cur_rot):
            self.board[self.cur_y + r, self.cur_x + c] = 1
        return self._clear_lines()

    def _clear_lines(self) -> int:
        """Удалить заполненные строки, сдвинуть остальное вниз. Возвращает кол-во."""
        full = np.all(self.board == 1, axis=1)
        n = int(np.count_nonzero(full))
        if n == 0:
            return 0
        kept = self.board[~full]
        new_board = np.zeros_like(self.board)
        new_board[n:] = kept     # сохранённые строки опускаются вниз
        self.board = new_board
        return n

    def _aggregate_height(self) -> int:
        """Сумма высот колонок (для height-penalty reward)."""
        filled = self.board != 0
        heights = BOARD_H_TOTAL - np.argmax(
            np.vstack([filled, np.ones((1, BOARD_W), dtype=bool)]), axis=0
        )
        return int(heights.sum())
