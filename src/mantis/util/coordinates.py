"""Hex coordinate helpers, pure (no numpy, no `mantis` imports) so any caller can use them."""
from __future__ import annotations


def axial_distance(a: tuple[float, float], b: tuple[float, float]):
    """Hex Manhattan distance between two axial points, as `max(|dq|, |dr|, |dq + dr|)`.

    Accepts `int` or `float` tuples and returns the input's type, so a float centroid gets the
    exact sub-unit distance without flooring.
    """
    dq = abs(a[0] - b[0])
    dr = abs(a[1] - b[1])
    ds = abs((a[0] + a[1]) - (b[0] + b[1]))
    return max(dq, dr, ds)
