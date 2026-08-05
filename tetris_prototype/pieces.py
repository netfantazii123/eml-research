"""
Tetromino pieces: 7 types x 4 rotations.

Every rotation is given as a list of (row, col) occupied cells inside a
bounding box whose top-left corner is at (0, 0). The coordinates are the
standard guideline rotations (without SRS wall kicks).

The type indexing is fixed and matches FEATURE_NAMES in config.py:
    0=I, 1=O, 2=T, 3=S, 4=Z, 5=J, 6=L
"""

PIECE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']

# SHAPES[piece_type][rotation] = tuple of (row, col) cells.
SHAPES: list[list[tuple[tuple[int, int], ...]]] = [
    # ── I ──────────────────────────────────────────────────────────────
    [
        ((1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 1), (1, 1), (2, 1), (3, 1)),
    ],
    # ── O ── (all rotations are identical) ─────────────────────────────
    [
        ((0, 1), (0, 2), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)),
    ],
    # ── T ──────────────────────────────────────────────────────────────
    [
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
    ],
    # ── S ──────────────────────────────────────────────────────────────
    [
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 1), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
    ],
    # ── Z ──────────────────────────────────────────────────────────────
    [
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 2), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ],
    # ── J ──────────────────────────────────────────────────────────────
    [
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ],
    # ── L ──────────────────────────────────────────────────────────────
    [
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (1, 2), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ],
]

# Spawn offset along X (top-left bbox column) for centering on a board of
# width 10. All pieces spawn near the center.
SPAWN_X = 3
SPAWN_Y = 0   # top of the buffer


def piece_cells(piece_type: int, rotation: int) -> tuple[tuple[int, int], ...]:
    """Cells of the piece in its bounding box for the given rotation."""
    return SHAPES[piece_type][rotation % 4]
