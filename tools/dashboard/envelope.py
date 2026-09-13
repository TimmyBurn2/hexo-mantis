"""Per-pixel bucketing: a long series drawn at chart width keeps its exact extremes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """Every value of one pixel column, `x` being the column's last x."""

    x: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class Bucket:
    """One pixel column: its min, max and last value."""

    x: float
    lo: float
    hi: float
    last: float


def windows(series: list[tuple[float, float]], *, width: int) -> list[Window]:
    """Group `(x, y)` points into at most `width` columns by x; empty columns are dropped."""
    if not series:
        return []
    ordered = sorted(series, key=lambda p: p[0])
    x0, x1 = ordered[0][0], ordered[-1][0]
    span = x1 - x0
    columns: dict[int, list[float]] = {}
    last_x: dict[int, float] = {}
    for x, y in ordered:
        col = int((x - x0) / span * width) if span > 0 else 0
        col = min(col, width - 1)
        columns.setdefault(col, []).append(y)
        last_x[col] = x
    return [Window(x=last_x[col], values=tuple(columns[col])) for col in sorted(columns)]


def envelope(series: list[tuple[float, float]], *, width: int) -> list[Bucket]:
    """The min–max–last envelope of `series` over at most `width` columns."""
    return [Bucket(x=w.x, lo=min(w.values), hi=max(w.values), last=w.values[-1])
            for w in windows(series, width=width)]
