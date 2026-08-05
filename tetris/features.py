"""
features.py — extractor of engineered afterstate features for EML distillation.

The EML formula works as a Dellacherie evaluation function: it is shown the
board AFTER a trial piece placement (the afterstate) and returns a scalar — how
good that placement is. The placement with the highest score is chosen.

19 features (order = config.FEATURE_NAMES):
    h0..h9      — heights of the 10 columns
    holes       — number of "holes" (empty cells under occupied ones)
    bumpiness   — sum of |height differences of adjacent columns|
    agg_h       — total height
    max_h       — maximum height
    wells       — total depth of "wells"
    row_trans   — horizontal filled/empty transitions (Dellacherie)
    col_trans   — vertical filled/empty transitions
    cleared     — lines cleared by this placement (0..4)
    landing_h   — height at which the piece landed

The numeric features are normalized to [-1, 1].
EML is stateless — all features are computed from the afterstate outside it.
"""

import numpy as np

from config import BOARD_W, BOARD_H, BOARD_H_TOTAL

N_AFTERSTATE = 19   # size of the extract_afterstate() vector


def column_heights(board: np.ndarray) -> np.ndarray:
    """Height of each column: distance from the floor to the topmost filled cell."""
    filled = board != 0
    any_filled = filled.any(axis=0)
    first = np.argmax(filled, axis=0)               # index of the topmost filled row
    return np.where(any_filled, BOARD_H_TOTAL - first, 0).astype(np.int32)


def count_holes(board: np.ndarray, heights: np.ndarray) -> int:
    """Empty cells that have a filled cell above them in the same column."""
    filled = board != 0
    total = 0
    for c in range(BOARD_W):
        if heights[c] == 0:
            continue
        top = BOARD_H_TOTAL - heights[c]            # row of the topmost filled cell
        total += int(np.count_nonzero(~filled[top + 1:, c]))
    return total


def count_wells(heights: np.ndarray) -> int:
    """Total well depth (the walls are treated as infinitely high)."""
    total = 0
    for c in range(BOARD_W):
        left = heights[c - 1] if c > 0 else BOARD_H_TOTAL
        right = heights[c + 1] if c < BOARD_W - 1 else BOARD_H_TOTAL
        depth = min(left, right) - heights[c]
        if depth > 0:
            total += int(depth)
    return total


def row_transitions(board: np.ndarray) -> int:
    """Horizontal filled<->empty transitions along rows (walls = filled)."""
    filled = (board != 0).astype(np.int8)
    padded = np.pad(filled, ((0, 0), (1, 1)), constant_values=1)
    return int(np.count_nonzero(padded[:, 1:] != padded[:, :-1]))


def column_transitions(board: np.ndarray) -> int:
    """Vertical filled<->empty transitions along columns (floor = filled, top = empty)."""
    filled = (board != 0).astype(np.int8)
    padded = np.pad(filled, ((1, 1), (0, 0)), constant_values=0)
    padded[-1, :] = 1                                # the floor under the board is filled
    return int(np.count_nonzero(padded[1:, :] != padded[:-1, :]))


def extract_afterstate(board: np.ndarray, cleared: int,
                       landing_h: int) -> np.ndarray:
    """
    Assemble the 19-dimensional afterstate feature vector.

    Args:
        board: (BOARD_H_TOTAL, BOARD_W) int — the board AFTER the piece lock and
            line clear.
        cleared: how many lines this placement cleared (0..4).
        landing_h: height of the piece's lowest cell above the floor at lock time.
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

    # Column heights (10): normalized by the visible height -> [-1, 1]
    feats[0:10] = np.clip(heights / BOARD_H, 0.0, 1.0) * 2.0 - 1.0

    # Scalar board features: normalized by empirical maxima
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
    """Normalize value/[0..max] -> [-1, 1] with clipping."""
    x = min(max(value / max_value, 0.0), 1.0)
    return x * 2.0 - 1.0


def extract_afterstate_batch(boards: np.ndarray, cleared: np.ndarray,
                             landing_h: np.ndarray) -> np.ndarray:
    """
    Vectorized version of extract_afterstate for K afterstate boards at once.

    Args:
        boards: (K, BOARD_H_TOTAL, BOARD_W) int — boards after lock + clear.
        cleared: (K,) int — lines cleared by each placement.
        landing_h: (K,) int — landing height of the piece.

    Returns:
        (K, 19) float32 — the same features as extract_afterstate, row by row.
    """
    K = boards.shape[0]
    filled = boards != 0                                   # (K, H, W)
    any_f = filled.any(axis=1)                             # (K, W)
    first = np.argmax(filled, axis=1)                      # (K, W)
    heights = np.where(any_f, BOARD_H_TOTAL - first, 0)    # (K, W)

    # Holes: empty cells below the topmost filled one = height - filled in column.
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
    padc[:, -1, :] = True                                  # the floor is filled
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
    """Vectorized normalization [0..max] -> [-1, 1] with clipping."""
    return np.clip(values / max_value, 0.0, 1.0) * 2.0 - 1.0
