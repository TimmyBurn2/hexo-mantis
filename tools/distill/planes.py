"""Six's network input planes for a position in move order, its root-tactics screen, and the teacher target on our legal set."""
# Replicates engine/src/planes.cpp of Six (MIT, Copyright 2026 CixMango), vendored at the pin in vendor/pins.toml.
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CROP = 25
HALF = CROP // 2
CELLS = CROP * CROP
PLANES = 8
RECENT = 4
WIN = 6
AXES = ((1, 0), (0, 1), (1, -1))
#: A stone's colour as the ring stores it: the opener (stone 0) is +1.
P1, P2 = 1, -1


def turn_for_stone(i: int) -> int:
    """The 1-based turn that places stone `i`: the opener alone, then two stones a turn."""
    return 1 if i == 0 else (i - 1) // 2 + 2


def player_for_stone(i: int) -> int:
    """The colour that places stone `i` (+1 opens)."""
    return P1 if turn_for_stone(i) % 2 == 1 else P2


def stones_left_before(i: int) -> int:
    """Stones the mover still has in its turn before placing stone `i`."""
    return 1 if i == 0 else (2 if (i - 1) % 2 == 0 else 1)


def crop_center(moves: np.ndarray) -> tuple[int, int]:
    """floor(mean + 1/2) of the last four stones on each axis; the origin on an empty board."""
    n_all = len(moves)
    if n_all == 0:
        return 0, 0
    recent = moves[-RECENT:]
    n = len(recent)
    return (2 * int(recent[:, 0].sum()) + n) // (2 * n), (2 * int(recent[:, 1].sum()) + n) // (2 * n)


def crop_index(q: np.ndarray, r: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    """Flat crop index (row from r, column from q) of each cell, -1 outside the crop."""
    row = np.asarray(r) - center[1] + HALF
    col = np.asarray(q) - center[0] + HALF
    inside = (row >= 0) & (row < CROP) & (col >= 0) & (col < CROP)
    return np.where(inside, row * CROP + col, -1)


_ROWS, _COLS = np.meshgrid(np.arange(CROP), np.arange(CROP), indexing="ij")


def crop_cells(center: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The (q, r) of every crop index, flat in index order."""
    return (_COLS - HALF + center[0]).reshape(-1), (_ROWS - HALF + center[1]).reshape(-1)


def near_stones(q: np.ndarray, r: np.ndarray, moves: np.ndarray, radius: int) -> np.ndarray:
    """Cells within hex distance `radius` of some stone, or of the origin before the first stone."""
    anchors = moves if len(moves) else np.zeros((1, 2), dtype=np.int64)
    dq = q[None, :] - anchors[:, 0, None]
    dr = r[None, :] - anchors[:, 1, None]
    return (np.maximum(np.maximum(np.abs(dq), np.abs(dr)), np.abs(dq + dr)) <= radius).any(axis=0)


def six_planes(moves: np.ndarray, radius: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Planes `[8, 25, 25]` float32 for the side about to place stone `len(moves)`, and the crop centre."""
    moves = np.asarray(moves, dtype=np.int64).reshape(-1, 2)
    n = len(moves)
    center = crop_center(moves)
    out = np.zeros((PLANES, CELLS), dtype=np.float32)
    out[0] = 1.0
    mover = player_for_stone(n)
    second = n > 0 and stones_left_before(n) == 1
    if n:
        idx = crop_index(moves[:, 0], moves[:, 1], center)
        own = np.array([player_for_stone(i) == mover for i in range(n)])
        out[1, idx[(idx >= 0) & own]] = 1.0
        out[2, idx[(idx >= 0) & ~own]] = 1.0
    cq, cr = crop_cells(center)
    out[3] = near_stones(cq, cr, moves, radius) & (out[1] == 0) & (out[2] == 0)
    if second:
        i = crop_index(moves[n - 1, 0], moves[n - 1, 1], center)
        if i >= 0:
            out[4, i] = 1.0
    turn_start = n - 1 if second else n
    for i in (turn_start - 2, turn_start - 1):
        if i >= 0 and player_for_stone(i) != mover:
            j = crop_index(moves[i, 0], moves[i, 1], center)
            if j >= 0:
                out[5, j] = 1.0
    out[6] = 1.0 if second else 0.0
    out[7] = 1.0 if radius == 9 else 0.0
    return out.reshape(PLANES, CROP, CROP), center


def _window_counts(moves: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    """Every six-cell line window holding a stone: (axis, start q, start r) -> [P1 count, P2 count]."""
    colour = {(int(q), int(r)): player_for_stone(i) for i, (q, r) in enumerate(moves)}
    windows: dict[tuple[int, int, int], list[int]] = {}
    for q, r in colour:
        for a, (dq, dr) in enumerate(AXES):
            for k in range(WIN):
                key = (a, q - dq * k, r - dr * k)
                if key not in windows:
                    windows[key] = [0, 0]
                    for j in range(WIN):
                        c = colour.get((key[1] + dq * j, key[2] + dr * j))
                        if c is not None:
                            windows[key][0 if c == P1 else 1] += 1
    return windows


def root_tactics_act(moves: Sequence[tuple[int, int]] | np.ndarray) -> bool:
    """True where Six's exact root tactics override its network: a mover win in this turn, or an opponent window of four or more with none of the mover's."""
    moves = np.asarray(moves, dtype=np.int64).reshape(-1, 2)
    n = len(moves)
    mover = player_for_stone(n)
    left = stones_left_before(n)
    me, opp = (0, 1) if mover == P1 else (1, 0)
    for counts in _window_counts(moves).values():
        if counts[opp] == 0 and counts[me] >= WIN - left:
            return True
        if counts[me] == 0 and counts[opp] >= WIN - 2:
            return True
    return False


def softmax_on(logits: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """softmax of `logits[idx]`, float64."""
    x = logits[idx].astype(np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def target_support(moves: np.ndarray, legal_q: np.ndarray, legal_r: np.ndarray,
                   center: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Our legal cells inside the crop: (positions in the legal list, their crop indices)."""
    idx = crop_index(legal_q, legal_r, center)
    keep = np.nonzero(idx >= 0)[0]
    return keep, idx[keep]
