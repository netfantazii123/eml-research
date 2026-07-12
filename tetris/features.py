"""
features.py — экстрактор инженерных признаков afterstate для EML-дистилляции.

EML-формула работает как оценочная функция Dellacherie: ей показывают доску
ПОСЛЕ пробной постановки фигуры (afterstate) и она возвращает скаляр —
насколько эта постановка хороша. Постановка с максимальной оценкой выбирается.

19 признаков (порядок = config.FEATURE_NAMES):
    h0..h9      — высоты 10 колонок
    holes       — кол-во "дыр" (пустых ячеек под занятыми)
    bumpiness   — сумма |разниц высот соседних колонок|
    agg_h       — суммарная высота
    max_h       — максимальная высота
    wells       — суммарная глубина "колодцев"
    row_trans   — горизонтальные переходы filled/empty (Dellacherie)
    col_trans   — вертикальные переходы filled/empty
    cleared     — линий очищено этой постановкой (0..4)
    landing_h   — высота, на которую легла фигура

Числовые признаки нормализованы в [-1, 1].
EML stateless — все признаки вычисляются из afterstate снаружи.
"""

import numpy as np

from config import BOARD_W, BOARD_H, BOARD_H_TOTAL

N_AFTERSTATE = 19   # размер вектора extract_afterstate()


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


def extract_afterstate(board: np.ndarray, cleared: int,
                       landing_h: int) -> np.ndarray:
    """
    Собрать 19-мерный вектор признаков afterstate.

    Args:
        board: (BOARD_H_TOTAL, BOARD_W) int — доска ПОСЛЕ лока фигуры и клира.
        cleared: сколько линий очистила эта постановка (0..4).
        landing_h: высота нижней ячейки фигуры над полом в момент лока.
    """
    heights = column_heights(board)
    holes = count_holes(board, heights)
    bumpiness = int(np.abs(np.diff(heights)).sum())
    agg_h = int(heights.sum())
    max_h = int(heights.max())
    wells = count_wells(heights)
    row_t = row_transitions(board)
    col_t = column_transitions(board)

    feats = np.empty(19, dtype=np.float32)

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
    feats[17] = _norm(cleared, 4.0)
    feats[18] = _norm(landing_h, BOARD_H_TOTAL)

    return feats


def _norm(value: float, max_value: float) -> float:
    """Нормализация value/[0..max] → [-1, 1] с клиппингом."""
    x = min(max(value / max_value, 0.0), 1.0)
    return x * 2.0 - 1.0


def extract_afterstate_batch(boards: np.ndarray, cleared: np.ndarray,
                             landing_h: np.ndarray) -> np.ndarray:
    """
    Векторная версия extract_afterstate для K afterstate-досок разом.

    Args:
        boards: (K, BOARD_H_TOTAL, BOARD_W) int — доски после лока+клира.
        cleared: (K,) int — линий очищено каждой постановкой.
        landing_h: (K,) int — высота посадки фигуры.

    Returns:
        (K, 19) float32 — те же признаки, что extract_afterstate, построчно.
    """
    K = boards.shape[0]
    filled = boards != 0                                   # (K, H, W)
    any_f = filled.any(axis=1)                             # (K, W)
    first = np.argmax(filled, axis=1)                      # (K, W)
    heights = np.where(any_f, BOARD_H_TOTAL - first, 0)    # (K, W)

    # Дыры: пустые ячейки ниже верхней занятой = высота − занятых в колонке.
    filled_cnt = filled.sum(axis=1)                        # (K, W)
    holes = np.where(any_f, heights - filled_cnt, 0).sum(axis=1)   # (K,)

    bump = np.abs(np.diff(heights, axis=1)).sum(axis=1)
    agg = heights.sum(axis=1)
    max_h = heights.max(axis=1)

    wall = np.full((K, 1), BOARD_H_TOTAL, dtype=heights.dtype)
    left = np.concatenate([wall, heights[:, :-1]], axis=1)
    right = np.concatenate([heights[:, 1:], wall], axis=1)
    wells = np.clip(np.minimum(left, right) - heights, 0, None).sum(axis=1)

    padded = np.pad(filled, ((0, 0), (0, 0), (1, 1)), constant_values=True)
    row_t = (padded[:, :, 1:] != padded[:, :, :-1]).sum(axis=(1, 2))

    padc = np.pad(filled, ((0, 0), (1, 1), (0, 0)), constant_values=False)
    padc[:, -1, :] = True                                  # пол — занятый
    col_t = (padc[:, 1:, :] != padc[:, :-1, :]).sum(axis=(1, 2))

    feats = np.empty((K, 19), dtype=np.float32)
    feats[:, 0:10] = np.clip(heights / BOARD_H, 0.0, 1.0) * 2.0 - 1.0
    feats[:, 10] = _norm_arr(holes, 40.0)
    feats[:, 11] = _norm_arr(bump, 60.0)
    feats[:, 12] = _norm_arr(agg, BOARD_W * BOARD_H)
    feats[:, 13] = _norm_arr(max_h, BOARD_H_TOTAL)
    feats[:, 14] = _norm_arr(wells, 40.0)
    feats[:, 15] = _norm_arr(row_t, 2 * (BOARD_W + 1) * BOARD_H_TOTAL)
    feats[:, 16] = _norm_arr(col_t, 2 * BOARD_W * BOARD_H_TOTAL)
    feats[:, 17] = _norm_arr(np.asarray(cleared), 4.0)
    feats[:, 18] = _norm_arr(np.asarray(landing_h), BOARD_H_TOTAL)
    return feats


def _norm_arr(values: np.ndarray, max_value: float) -> np.ndarray:
    """Векторная нормализация [0..max] → [-1, 1] с клиппингом."""
    return np.clip(values / max_value, 0.0, 1.0) * 2.0 - 1.0
