"""The observatory's stats are the dashboard's, copied; this guard reds the day they drift."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def new(observatory):
    return importlib.import_module("observatory.readers.stats")


@pytest.fixture(scope="module")
def old(dashboard):
    return importlib.import_module("dashboard.stats")


@pytest.mark.parametrize("p,n", [(0.0, 32), (0.1875, 32), (0.5, 288), (1.0, 20)])
def test_wilson_elo_and_clamp_agree_with_the_dashboard(new, old, p, n):
    assert new.wilson_interval(p, n) == old.wilson_interval(p, n)
    assert new.clamp_wr(p, n) == old.clamp_wr(p, n)
    assert new.elo_of_wr(new.clamp_wr(p, n)) == old.elo_of_wr(old.clamp_wr(p, n))


def test_the_slope_and_quantile_agree_with_the_dashboard(new, old):
    xs, ys = [1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 12.5, 11.0, 15.0, 14.0]
    a, b = new.ols_slope(xs, ys), old.ols_slope(xs, ys)
    assert a is not None and b is not None and (a.slope, a.se, a.lo, a.hi, a.n) == (b.slope, b.se, b.lo, b.hi, b.n)
    assert new.ols_slope(xs[:2], ys[:2]) is None and old.ols_slope(xs[:2], ys[:2]) is None
    assert new.quantile([1.0, 2.0, 4.0], 0.5) == old.quantile([1.0, 2.0, 4.0], 0.5)


def test_the_copy_is_byte_identical_to_the_original_until_phase_6_deletes_it(new, old):
    assert Path(new.__file__).read_text(encoding="utf-8") == Path(old.__file__).read_text(encoding="utf-8")
