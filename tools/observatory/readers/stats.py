"""Hand-checkable arithmetic: Wilson interval, Elo of a win rate, OLS slope with a t interval."""
from __future__ import annotations

import math
from dataclasses import dataclass

#: z for a two-sided 95 % interval.
Z95 = 1.959964

#: t(0.975, df) at anchor degrees of freedom; the next-lower anchor is used between them,
#: which widens rather than narrows an interval.
_T975 = (
    (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (6, 2.447), (7, 2.365),
    (8, 2.306), (9, 2.262), (10, 2.228), (11, 2.201), (12, 2.179), (13, 2.160), (14, 2.145),
    (15, 2.131), (16, 2.120), (17, 2.110), (18, 2.101), (19, 2.093), (20, 2.086), (21, 2.080),
    (22, 2.074), (23, 2.069), (24, 2.064), (25, 2.060), (26, 2.056), (27, 2.052), (28, 2.048),
    (29, 2.045), (30, 2.042), (40, 2.021), (60, 2.000), (120, 1.980),
)


def t975(df: int) -> float:
    """Two-sided 95 % t quantile for `df` degrees of freedom (conservative between anchors)."""
    value = Z95
    for anchor, quantile in _T975:
        if df >= anchor:
            value = quantile
    return value


def wilson_interval(p: float, n: int, *, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for an observed rate `p` over `n` trials.

    Raises:
        ValueError: `n` is not positive, or `p` is outside [0, 1].
    """
    if n <= 0:
        raise ValueError(f"wilson_interval needs n > 0, got {n}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"wilson_interval needs 0 <= p <= 1, got {p}")
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def elo_of_wr(p: float) -> float:
    """Elo difference implied by a win rate: 400·log10(p / (1 − p)); ±inf at the ends."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    return 400.0 * math.log10(p / (1.0 - p))


def clamp_wr(p: float, n: int) -> float:
    """Keep a rate inside [1/(2n), 1 − 1/(2n)] — half a game — so a 0 % round stays finite."""
    edge = 1.0 / (2.0 * n)
    return min(max(p, edge), 1.0 - edge)


@dataclass(frozen=True)
class SlopeFit:
    """An OLS slope with its two-sided 95 % interval over `n` points."""

    slope: float
    se: float
    lo: float
    hi: float
    n: int

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0


def ols_slope(xs: list[float], ys: list[float]) -> SlopeFit | None:
    """Least-squares slope of `ys` on `xs`; `None` below three points or on a degenerate x."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = mean_y - slope * mean_x
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    se = math.sqrt(sse / (n - 2) / sxx)
    half = t975(n - 2) * se
    return SlopeFit(slope=slope, se=se, lo=slope - half, hi=slope + half, n=n)


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile over an already-sorted, non-empty list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
