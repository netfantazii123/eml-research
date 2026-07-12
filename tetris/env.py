"""
TetrisEnv — headless-среда Tetris с gym-like интерфейсом.

Два режима управления:

1. PLACEMENT (основной, для оракула и EML) — действие = постановка фигуры:
       action = rot * BOARD_W + xi
   где rot — поворот, xi — колонка самой левой занятой ячейки фигуры.
   Один step_placement() = поворот+сдвиг+hard drop+лок+клир. Награда плотная,
   выдаётся на каждом шаге. Нелегальные действия закрываются маской.

       reset()                   -> obs (placement, см. get_placement_state)
       step_placement(a)         -> (obs, reward, done, info)
       placement_mask()          -> (N_PLACEMENTS,) float32 (1 = легально)
       afterstate(rot, xi)       -> (board_after, cleared, landing_h) без мутации

2. FRAME (legacy, для визуального демо) — 6 покадровых действий:
       step(action) — 0 NO-OP, 1 left, 2 right, 3 rotate CW, 4 soft, 5 hard
       get_state()  — obs для покадрового просмотра

Bag-7 рандомизация, line-clear. Без отрисовки — чистая логика на NumPy.
"""

import numpy as np

import config
from config import (
    BOARD_W, BOARD_H, BOARD_BUFFER, BOARD_H_TOTAL,
    GRAVITY_TICKS, MAX_EPISODE_FRAMES, N_PIECES, N_PLACEMENTS,
)
# Reward-константы НЕ импортируются по имени, а читаются через config.* в step():
# так GUI может менять их вживую (config.REWARD_W_HOLES = ... во время обучения).
from pieces import piece_cells, SPAWN_X, SPAWN_Y, DISTINCT_ROTATIONS, SHAPES
import features as features_mod


# Предвычисленная геометрия: _GEOM[ptype][rot] = (cells, c_min, span, r_max),
# где span = c_max - c_min (ширина фигуры в занятых колонках минус 1).
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
    """Headless Tetris. Доска (24×10): строки 0..3 — буфер спавна, 4..23 — видимые."""

    def __init__(self, seed: int | None = None,
                 max_placements: int | None = None):
        """
        Args:
            seed: сид bag-рандомизации.
            max_placements: лимит постановок за эпизод; None → из config,
                0 → без лимита (Play-режим: игра до реального game over).
        """
        self._max_placements = max_placements
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
        self.frame = 0          # кадров (frame-режим)
        self.placements = 0     # поставленных фигур (placement-режим)
        self.done = False
        self._lock_dholes = 0   # дельты метрик последнего лока (reward shaping)
        self._lock_dbump = 0
        self._lock_dagg = 0
        self.reset()

    # ── Публичный интерфейс ──────────────────────────────────────────────

    def reset(self) -> dict:
        """Сброс среды. Возвращает первый placement-obs."""
        self.board[:] = 0
        self._bag = []
        self.score = 0
        self.frame = 0
        self.placements = 0
        self.gravity_counter = 0
        self.done = False
        self.next_type = self._draw_from_bag()
        self._spawn()   # на чистой доске спавн не может зайти в game over
        return self.get_placement_state()

    # ── PLACEMENT-режим ──────────────────────────────────────────────────

    def placement_mask(self) -> np.ndarray:
        """
        Маска легальных постановок текущей фигуры, (N_PLACEMENTS,) float32.

        Легально: поворот геометрически уникален для фигуры, фигура помещается
        в доску по ширине и не коллизирует на спавн-высоте.
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
        Пробная постановка БЕЗ мутации среды.

        Returns:
            (board_after, cleared, landing_h) или None, если постановка нелегальна.
            board_after — копия доски после лока и клира.
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
        """19 признаков afterstate для EML (None, если постановка нелегальна)."""
        result = self.afterstate(rot, xi)
        if result is None:
            return None
        board, cleared, landing_h = result
        return features_mod.extract_afterstate(board, cleared, landing_h)

    def all_afterstate_features(self,
                                mask: np.ndarray | None = None) -> np.ndarray:
        """
        Признаки afterstate ВСЕХ легальных постановок, (N_PLACEMENTS, 19).

        Нелегальные строки — нули. Используется оракулом: голова политики
        оценивает каждую постановку по признакам её afterstate.
        Извлечение признаков векторизовано по кандидатам (extract_afterstate_batch).
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
        Поставить фигуру: action = rot * BOARD_W + xi. Один шаг = один лок.

        Нелегальное действие (при корректной маске не встречается) трактуется
        как проигрыш — сеть обязана уважать маску.
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
        if not self._spawn():
            self.done = True
            reward += config.REWARD_GAME_OVER
        elif cap > 0 and self.placements >= cap:
            self.done = True

        return (self.get_placement_state(), reward, self.done,
                {'score': self.score, 'lines': n_cleared})

    def get_placement_state(self) -> dict:
        """
        Observation для placement-оракула.

        'grid':    (1, 24, 10) float32 — зафиксированные ячейки.
        'scalars': (14,) float32 — one-hot(cur 7) + one-hot(next 7).
        'mask':    (N_PLACEMENTS,) float32 — легальные постановки.
        'afeats':  (N_PLACEMENTS, 19) float32 — признаки afterstate каждой
                   легальной постановки (нелегальные — нули).
        """
        grid = np.zeros((1, BOARD_H_TOTAL, BOARD_W), dtype=np.float32)
        grid[0] = self.board
        scalars = np.zeros(14, dtype=np.float32)
        scalars[self.cur_type] = 1.0               # 0..6
        scalars[7 + self.next_type] = 1.0          # 7..13
        mask = self.placement_mask()
        return {'grid': grid, 'scalars': scalars, 'mask': mask,
                'afeats': self.all_afterstate_features(mask)}

    # ── FRAME-режим (legacy, для визуального демо) ───────────────────────

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        """Один кадр симуляции (6 действий, gravity). Для демо/визуализации."""
        if self.done:
            return self.get_state(), 0.0, True, {'score': self.score}

        reward = config.REWARD_SURVIVE   # обычно 0.0 — выживание награждается событийно
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

        # ── Награда за лок: плотный shaping + линии + спавн новой фигуры ──
        if locked:
            reward += config.REWARD_PIECE                # положил фигуру, не умер
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
        Observation frame-режима (legacy).

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

    def _drop_y(self, ptype: int, rot: int, x: int, from_y: int) -> int:
        """Y, на котором фигура останавливается при падении из from_y."""
        y = from_y
        while not self._collides(ptype, rot, x, y + 1):
            y += 1
        return y

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
        return self._drop_y(self.cur_type, self.cur_rot, self.cur_x, self.cur_y)

    # ── Лок и очистка линий ──────────────────────────────────────────────

    def _lock_piece(self) -> int:
        """
        Зафиксировать текущую фигуру на доске и очистить полные линии.

        Побочно вычисляет дельты метрик доски (до постановки → после клира)
        и сохраняет их в self._lock_d* для reward shaping.
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
        """(holes, bumpiness, agg_h) текущей доски — для shaping (см. features.py)."""
        heights = features_mod.column_heights(self.board)
        holes = features_mod.count_holes(self.board, heights)
        bumpiness = int(np.abs(np.diff(heights)).sum())
        agg_h = int(heights.sum())
        return holes, bumpiness, agg_h

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
