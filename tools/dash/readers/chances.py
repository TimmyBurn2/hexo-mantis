"""The win-chance strip: Light's chance at each turn's first stone from the recorded search, each turn's cost, the turning point."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hexlogic import first_of_turn, owner, turn_of

#: A turn costing its mover at least this much chance is marked; at least `RED` is marked red.
AMBER = 0.30
RED = 0.50


@dataclass(frozen=True)
class Point:
    """One turn's start: its first stone's ply, Light's chance there, and the turn's cost to its mover (None at the last point or a gap)."""

    ply: int
    turn: int
    light: float
    cost: float | None


def _value(entry: dict[str, Any]) -> float | None:
    v = entry.get("root_value")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return max(-1.0, min(1.0, float(v)))


def points(stats_by_ply: dict[int, dict[str, Any]]) -> list[Point]:
    """Light's chance per turn from the root value at its first stone (the second stone's W/N is biased low), with costs."""
    raw: list[tuple[int, int, float]] = []
    for ply in sorted(stats_by_ply):
        v = _value(stats_by_ply[ply])
        if v is None or not first_of_turn(ply):
            continue
        raw.append((ply, turn_of(ply), (v + 1) / 2 if owner(ply) == 0 else (1 - v) / 2))
    out: list[Point] = []
    for i, (ply, turn, light) in enumerate(raw):
        cost = None
        if i + 1 < len(raw) and raw[i + 1][1] == turn + 1:
            nxt = raw[i + 1][2]
            cost = (light - nxt) if owner(ply) == 0 else (nxt - light)
        out.append(Point(ply=ply, turn=turn, light=light, cost=cost))
    return out


def turning_point(pts: list[Point]) -> Point | None:
    """The turn with the largest cost of at least `AMBER`, or None when no turn cost that much."""
    marked = [p for p in pts if p.cost is not None and p.cost >= AMBER]
    return max(marked, key=lambda p: (p.cost or 0.0, -p.ply)) if marked else None
