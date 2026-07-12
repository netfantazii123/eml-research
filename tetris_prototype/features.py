"""
features.py — экстрактор инженерных признаков доски для EML-дистилляции.

28 признаков (порядок = config.FEATURE_NAMES):
    h0..h9      — высоты 10 колонок
    holes       — кол-во "дыр" (пустых ячеек под занятыми)
    bumpiness   — сумма |разниц высот соседних колонок|
    agg_h       — суммарная высота
    max_h       — максимальная высота
    wells       — суммарная глубина "колодцев"
    row_trans   — горизонтальные переходы filled/empty (Dellacherie)
    col_trans   — вертикальные переходы filled/empty
    piece_I..L  — one-hot текущей фигуры (7)
    rot_0..3    — one-hot текущего поворота (4)

Числовые признаки нормализованы в [-1, 1]; one-hot остаются 0/1.
EML stateless — все признаки вычисляются из текущего board state снаружи.
"""

import numpy as np

from config import BOARD_W, BOARD_H, BOARD_H_TOTAL, N_PIECES, N_ROTATIONS


def column_heights(board: np.ndarray) -> np.ndarray:
    """Высота каждой колонки: расстояние от пола до верхней занятой ячейки."""
    filled = board != 0
    any_filled = filled.any(axis=0)
    first = np.argmax(filled, axis=0)               # индекс верхней занятой строки
    return np.where(any_filled, BOARD_H_TOTAL - first, 0).astype(np.int32)


def count_holes(board: np.ndarray, heights: np.ndarray) -> int:
    """Пустые ячейки, над которыми в той же колонке есть занятая."""
    filled = board != 0
    total = 0
    for c in range(BOARD_W):
        if heights[c] == 0:
            continue
        top = BOARD_H_TOTAL - heights[c]            # строка верхней занятой ячейки
        total += int(np.count_nonzero(~filled[top + 1:, c]))
    return total


def count_wells(heights: np.ndarray) -> int:
    """Суммарная глубина колодцев (стены трактуются как бесконечно высокие)."""
    total = 0
    for c in range(BOARD_W):
        left = heights[c - 1] if c > 0 else BOARD_H_TOTAL
        right = heights[c + 1] if c < BOARD_W - 1 else BOARD_H_TOTAL
        depth = min(left, right) - heights[c]
        if depth > 0:
            total += int(depth)
    return total


def row_transitions(board: np.ndarray) -> int:
    """Горизонтальные переходы filled↔empty по строкам (стены = занятые)."""
    filled = (board != 0).astype(np.int8)
    padded = np.pad(filled, ((0, 0), (1, 1)), constant_values=1)
    return int(np.count_nonzero(padded[:, 1:] != padded[:, :-1]))


def column_transitions(board: np.ndarray) -> int:
    """Вертикальные переходы filled↔empty по колонкам (пол = занятый, верх = пустой)."""
    filled = (board != 0).astype(np.int8)
    padded = np.pad(filled, ((1, 1), (0, 0)), constant_values=0)
    padded[-1, :] = 1                                # пол под доской — занятый
    return int(np.count_nonzero(padded[1:, :] != padded[:-1, :]))


def extract(board: np.ndarray, piece_type: int, rotation: int) -> np.ndarray:
    """
    Собрать 28-мерный вектор признаков из доски + текущей фигуры.

    Args:
        board: (BOARD_H_TOTAL, BOARD_W) int — занятые ячейки (locked).
        piece_type: 0..6.
        rotation: 0..3.
    """
    heights = column_heights(board)
    holes = count_holes(board, heights)
    bumpiness = int(np.abs(np.diff(heights)).sum())
    agg_h = int(heights.sum())
    max_h = int(heights.max())
    wells = count_wells(heights)
    row_t = row_transitions(board)
    col_t = column_transitions(board)

    feats = np.empty(28, dtype=np.float32)

    # Высоты колонок (10): нормализация по видимой высоте → [-1, 1]
    feats[0:10] = np.clip(heights / BOARD_H, 0.0, 1.0) * 2.0 - 1.0

    # Скалярные признаки доски: нормализация по эмпирическим максимумам
    feats[10] = _norm(holes, 40.0)
    feats[11] = _norm(bumpiness, 60.0)
    feats[12] = _norm(agg_h, BOARD_W * BOARD_H)
    feats[13] = _norm(max_h, BOARD_H_TOTAL)
    feats[14] = _norm(wells, 40.0)
    feats[15] = _norm(row_t, 2 * (BOARD_W + 1) * BOARD_H_TOTAL)
    feats[16] = _norm(col_t, 2 * BOARD_W * BOARD_H_TOTAL)

    # One-hot фигуры (7) и поворота (4)
    feats[17:17 + N_PIECES] = 0.0
    feats[17 + piece_type] = 1.0
    feats[24:24 + N_ROTATIONS] = 0.0
    feats[24 + rotation] = 1.0

    return feats


def _norm(value: float, max_value: float) -> float:
    """Нормализация value/[0..max] → [-1, 1] с клиппингом."""
    x = min(max(value / max_value, 0.0), 1.0)
    return x * 2.0 - 1.0
