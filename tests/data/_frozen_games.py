"""Frozen WP9 replay / Q13 input games (committed test data).

Verbatim (id, moves, winner) for the 4 games the O4b replay-parity and O5 Q13-parity
oracles reuse. Sourced once from the dispatcher frozen-input set; embedded as code so the
committed tests are self-contained (the full old-side capture is not in this repo).
"""
from __future__ import annotations

# id -> (moves, winner). moves are (q, r) axial tuples.
FROZEN_GAMES: dict[str, tuple[list[tuple[int, int]], int]] = {
    'g0': (
        [
            (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (2, -1),
            (1, -2), (-1, 2), (-1, -1), (-2, 1), (1, 1), (0, 2), (2, -2), (-2, 0),
            (0, -2), (-1, 3), (2, -3), (-2, -1), (-2, 2), (-1, -2),
        ],
        -1,
    ),
    'g1': (
        [
            (0, 0), (1, -1), (-1, 1), (0, -1), (1, 0), (-1, 0), (0, 1), (-1, 2),
            (-1, -1), (-2, 1), (1, 1), (2, 0), (1, -2), (2, -1), (0, 2), (0, -2),
            (2, -2), (-2, 0), (-3, 1), (1, -3), (-2, 2), (3, -1), (2, 1),
        ],
        -1,
    ),
    'g2': (
        [
            (0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1),
            (-1, 2), (2, -1), (1, -2), (-1, -1), (-2, 1), (0, -2), (2, -2), (-2, 2),
            (0, 2), (2, 0), (-3, 1), (2, 1), (1, 2), (-2, -1), (-2, 3), (-1, 3),
            (-2, 0), (3, -1), (2, -3), (-3, 2), (3, -2), (0, 3), (0, -3), (-3, 3),
            (3, 0), (-1, -2), (1, -3), (2, 2),
        ],
        1,
    ),
    'g3': (
        [
            (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (2, -1),
            (6, 0), (11, 0), (16, -1), (21, -1), (21, 1), (23, 0), (22, 1), (23, -1),
        ],
        1,
    ),
}

# The three registered corpus encodings replayed for parity.
ENCODINGS = ("v6", "v6w25", "v6_live2_ls")
