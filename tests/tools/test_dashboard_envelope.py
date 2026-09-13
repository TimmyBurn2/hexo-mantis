"""The per-pixel envelope: what is drawn at 640 px must carry the same extremes as the full
series, or a spike the reader is looking for is averaged out of the picture."""
from __future__ import annotations

import importlib
import math
import random

import pytest


@pytest.fixture(scope="module")
def envelope(dashboard):
    return importlib.import_module("dashboard.envelope")


def _noisy_series(n: int, seed: int = 7) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(float(i), math.sin(i / 50.0) + rng.random() * 0.2) for i in range(n)]


def test_the_envelope_preserves_the_global_extremes_exactly(envelope):
    series = _noisy_series(35_000)
    series[12_345] = (12_345.0, 99.0)
    series[999] = (999.0, -99.0)
    buckets = envelope.envelope(series, width=640)
    assert max(b.hi for b in buckets) == 99.0
    assert min(b.lo for b in buckets) == -99.0


def test_the_envelope_preserves_the_last_value(envelope):
    series = _noisy_series(10_001)
    series[-1] = (10_000.0, 42.5)
    buckets = envelope.envelope(series, width=640)
    assert buckets[-1].last == 42.5


def test_there_are_at_most_width_buckets_and_their_x_is_monotone(envelope):
    series = _noisy_series(35_000)
    buckets = envelope.envelope(series, width=640)
    assert 0 < len(buckets) <= 640
    xs = [b.x for b in buckets]
    assert xs == sorted(xs) and len(set(xs)) == len(xs)


def test_a_short_series_is_one_bucket_per_point(envelope):
    series = [(0.0, 1.0), (5.0, 3.0), (7.0, 2.0)]
    buckets = envelope.envelope(series, width=640)
    assert [(b.x, b.lo, b.hi, b.last) for b in buckets] == [
        (0.0, 1.0, 1.0, 1.0), (5.0, 3.0, 3.0, 3.0), (7.0, 2.0, 2.0, 2.0),
    ]


def test_unsorted_input_is_bucketed_by_x_not_by_arrival(envelope):
    series = [(3.0, 30.0), (1.0, 10.0), (2.0, 20.0)]
    buckets = envelope.envelope(series, width=2)
    assert [b.x for b in buckets] == sorted(b.x for b in buckets)
    assert min(b.lo for b in buckets) == 10.0 and max(b.hi for b in buckets) == 30.0


def test_an_empty_series_is_an_empty_envelope(envelope):
    assert envelope.envelope([], width=640) == []


def test_windows_carry_every_value_so_quantiles_are_over_the_full_series(envelope):
    series = _noisy_series(5_000)
    windows = envelope.windows(series, width=100)
    assert sum(len(w.values) for w in windows) == 5_000
    assert len(windows) <= 100
