"""
features.py — extractor of engineered board features for EML distillation.

28 features (order = config.FEATURE_NAMES):
    h0..h9      — heights of the 10 columns
    holes       — number of "holes" (empty cells under occupied ones)
    bumpiness   — sum of |height differences of adjacent columns|
    agg_h       — total height
    max_h       — maximum height
    wells       — total depth of "wells"
    row_trans   — horizontal filled/empty transitions (Dellacherie)
    col_trans   — vertical filled/empty transitions
    piece_I..L  — one-hot of the current piece (7)
    rot_0..3    — one-hot of the current rotation (4)

The numeric features are normalized to [-1, 1]; one-hots stay 0/1.
EML is stateless — all features are computed from the board state outside it.
"""

import numpy as np

from config import BOARD_W, BOARD_H, BOARD_H_TOTAL, N_PIECES, N_ROTATIONS


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


def extract(board: np.ndarray, piece_type: int, rotation: int) -> np.ndarray:
    """
    Assemble the 28-dimensional feature vector from the board + current piece.

    Args:
        board: (BOARD_H_TOTAL, BOARD_W) int — occupied (locked) cells.
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

    # One-hot of the piece (7) and the rotation (4)
    feats[17:17 + N_PIECES] = 0.0
    feats[17 + piece_type] = 1.0
    feats[24:24 + N_ROTATIONS] = 0.0
    feats[24 + rotation] = 1.0

    return feats


def _norm(value: float, max_value: float) -> float:
    """Normalize value/[0..max] -> [-1, 1] with clipping."""
    x = min(max(value / max_value, 0.0), 1.0)
    return x * 2.0 - 1.0
