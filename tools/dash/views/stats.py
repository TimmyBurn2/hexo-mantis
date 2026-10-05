"""The statistics behind the verdict words: a word appears only when its 95 % interval excludes the null."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..readers.sidecars import logit

Z95 = 1.96
__all__ = ["Separation", "expit", "logit", "separation", "wilson"]


@dataclass(frozen=True)
class Separation:
    """`a − b` with its approximate 95 % interval; `sign` is +1 or −1 only when the interval excludes zero."""

    d: float
    lo: float
    hi: float

    @property
    def sign(self) -> int:
        return 1 if self.lo > 0 else -1 if self.hi < 0 else 0


def separation(p_a: float, n_a: int, p_b: float, n_b: int) -> Separation:
    """Two win rates over their distinct-game counts; the pairing is ignored, which widens the interval, never narrows it."""
    var = (p_a * (1 - p_a) / max(n_a, 1)) + (p_b * (1 - p_b) / max(n_b, 1))
    se, d = math.sqrt(var), p_a - p_b
    return Separation(d, d - Z95 * se, d + Z95 * se)


def wilson(k: int, n: int) -> tuple[float, float]:
    """The Wilson 95 % interval of k successes in n; (0, 1) for n = 0."""
    if n <= 0:
        return 0.0, 1.0
    p, z2 = k / n, Z95 * Z95
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def expit(x: float) -> float:
    """The inverse of `logit`: a logit back to a probability."""
    return 1.0 / (1.0 + math.exp(-x))
