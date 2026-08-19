"""Selfplay axis-distribution metric (WP10 §a.3 PORT — behaviour-exact).

For each hex axis, computes the fraction of adjacent stone pairs that share the same color.
Values near 0.5 = balanced opponent interleaving; near 1.0 = same-color clustering along
that axis — a potential degenerate-strategy signal.

Three axes (matching env.game_state._HEX_AXES order):
  axis_q  E-W      (dq=+1, dr= 0)
  axis_r  NW-SE    (dq= 0, dr=+1)
  axis_s  NE-SW    (dq=+1, dr=-1)
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np

# Mirrors env.game_state._HEX_AXES — do not reorder.
_AXES = ((1, 0), (0, 1), (1, -1))
AXIS_LABELS = ("axis_q", "axis_r", "axis_s")


class AxisFractions(TypedDict):
    """Per-axis same-color fractions plus the max-axis LABEL (a str, not a fraction)."""

    axis_q: float
    axis_r: float
    axis_s: float
    axis_max: str


def _assign_colors(move_history: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Return {(q, r): color} for a game's move list.

    color = +1 for P1, -1 for P2. Assignment matches pool.py compound-move rule: ply 0 → P1;
    compound_idx = (ply - 1) // 2; P2 when even, P1 odd.
    """
    stone_color: dict[tuple[int, int], int] = {}
    for ply, pos in enumerate(move_history):
        is_p1 = (ply == 0) or (((ply - 1) // 2) % 2 == 1)
        stone_color[pos] = 1 if is_p1 else -1
    return stone_color


def compute_axis_fractions(games: Sequence[list[tuple[int, int]]]) -> AxisFractions:
    """Compute axis-distribution fractions from completed self-play games.

    Aggregates total same-color adjacent pairs / total adjacent pairs across all games for
    each axis. Returns 0.0 on empty input. Keys: axis_q, axis_r, axis_s (floats in [0,1]),
    axis_max (label of the max-fraction axis).
    """
    same = [0, 0, 0]
    total = [0, 0, 0]

    for game in games:
        if len(game) < 2:
            continue
        stone_color = _assign_colors(game)
        for i, (dq, dr) in enumerate(_AXES):
            for (q, r), color in stone_color.items():
                nbr = (q + dq, r + dr)
                if nbr in stone_color:
                    total[i] += 1
                    if stone_color[nbr] == color:
                        same[i] += 1

    fracs = [s / t if t > 0 else 0.0 for s, t in zip(same, total, strict=True)]
    max_idx = int(np.argmax(fracs)) if any(t > 0 for t in total) else 0
    return {
        "axis_q": fracs[0],
        "axis_r": fracs[1],
        "axis_s": fracs[2],
        "axis_max": AXIS_LABELS[max_idx],
    }
