"""The dashboard's arithmetic, hand-checked: a wrong interval on the hero strip is fiction with a
decimal point."""
from __future__ import annotations

import importlib
import math

import pytest


@pytest.fixture(scope="module")
def stats(dashboard):
    return importlib.import_module("dashboard.stats")


@pytest.mark.parametrize(
    ("p", "n", "lo", "hi"),
    [
        # 50 of 100: the textbook symmetric case.
        (0.5, 100, 0.4038, 0.5962),
        # 0 of 32: the lower bound is exactly 0 and the upper is z^2 / (n + z^2).
        (0.0, 32, 0.0, 0.1072),
        # 6 of 32: run6's first round, worked by hand.
        (0.1875, 32, 0.0889, 0.3531),
    ],
)
def test_the_wilson_interval_matches_three_hand_checked_cases(stats, p, n, lo, hi):
    got_lo, got_hi = stats.wilson_interval(p, n)
    assert got_lo == pytest.approx(lo, abs=5e-4)
    assert got_hi == pytest.approx(hi, abs=5e-4)


def test_the_wilson_interval_refuses_a_zero_sample(stats):
    with pytest.raises(ValueError):
        stats.wilson_interval(0.5, 0)


def test_elo_of_an_even_win_rate_is_zero_and_the_map_is_antisymmetric(stats):
    assert stats.elo_of_wr(0.5) == 0.0
    assert stats.elo_of_wr(0.75) == pytest.approx(400 * math.log10(3))
    assert stats.elo_of_wr(0.25) == pytest.approx(-stats.elo_of_wr(0.75))
    assert stats.elo_of_wr(0.0) == -math.inf and stats.elo_of_wr(1.0) == math.inf


def test_the_half_game_clamp_keeps_a_zero_round_finite_and_leaves_an_interior_rate_alone(stats):
    assert stats.clamp_wr(0.0, 32) == pytest.approx(1 / 64)
    assert stats.clamp_wr(1.0, 32) == pytest.approx(1 - 1 / 64)
    assert stats.clamp_wr(0.1875, 32) == 0.1875
    assert math.isfinite(stats.elo_of_wr(stats.clamp_wr(0.0, 32)))


def test_an_exact_line_has_slope_two_and_a_zero_width_interval(stats):
    fit = stats.ols_slope([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert fit.slope == pytest.approx(2.0)
    assert fit.lo == pytest.approx(2.0) and fit.hi == pytest.approx(2.0)
    assert fit.n == 4


def test_a_noisy_fit_carries_the_t_based_interval_worked_by_hand(stats):
    # slope 0.6, SSE 2.4 on 3 df, Sxx 10 -> SE 0.28284, t(0.975, 3) = 3.182 -> 0.6 +/- 0.900
    fit = stats.ols_slope([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 5.0, 4.0, 5.0])
    assert fit.slope == pytest.approx(0.6)
    assert fit.lo == pytest.approx(-0.300, abs=2e-3)
    assert fit.hi == pytest.approx(1.500, abs=2e-3)
    assert fit.excludes_zero is False


def test_a_slope_needs_three_points_and_says_so(stats):
    assert stats.ols_slope([1.0, 2.0], [1.0, 2.0]) is None
    assert stats.ols_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_quantiles_interpolate_linearly_over_the_sorted_values(stats):
    values = [10.0, 20.0, 30.0, 40.0]
    assert stats.quantile(values, 0.5) == pytest.approx(25.0)
    assert stats.quantile(values, 0.0) == 10.0 and stats.quantile(values, 1.0) == 40.0
    assert stats.quantile([7.0], 0.9) == 7.0
