"""Chart primitives: what a figure prints beside itself equals what it draws."""
from __future__ import annotations

import importlib
import math
import re

import pytest


@pytest.fixture(scope="module")
def svg(dashboard):
    return importlib.import_module("dashboard.svg")


def _series(svg, n: int):
    pts = [(float(i), math.sin(i / 40.0)) for i in range(n)]
    pts[min(777, n - 1)] = (float(min(777, n - 1)), 9.5)
    return svg.Series(label="loss", points=pts)


def test_an_envelope_chart_states_and_draws_the_same_extremes(svg):
    fig = svg.envelope_chart([_series(svg, 20_000)], x_label="step")
    assert "<svg" in fig and 'viewBox="0 0 640' in fig
    assert 'preserveAspectRatio="xMidYMid meet"' in fig
    drawn_max = float(re.search(r'data-max="([^"]+)"', fig).group(1))
    assert drawn_max == 9.5, "the tallest envelope pixel must be the printed max"
    assert "max 9.5" in fig and "last " in fig and "min " in fig
    band_points = re.search(r'<polygon class="band[^"]*" points="([^"]+)"', fig).group(1)
    assert len(band_points.split()) <= 2 * 640


def test_a_series_shorter_than_the_width_draws_no_band_only_the_line(svg):
    fig = svg.envelope_chart([svg.Series(label="wr", points=[(0.0, 1.0), (1.0, 2.0),
                                                             (2.0, 1.5)])], x_label="x")
    assert "<polygon" not in fig and "<polyline" in fig


def test_fewer_than_two_points_is_an_absence_not_a_chart(svg):
    fig = svg.envelope_chart([svg.Series(label="wr", points=[(0.0, 1.0)])], x_label="x")
    assert "<svg" not in fig and "too few" in fig


def test_axis_labels_are_html_text_outside_the_svg(svg):
    fig = svg.envelope_chart([_series(svg, 100)], x_label="step")
    assert "<text" not in fig, "labels scale with the SVG; HTML labels keep their pixel size"
    assert 'class="x-labels"' in fig and 'class="y-labels"' in fig


def test_a_rule_is_drawn_at_its_value_and_named(svg):
    fig = svg.envelope_chart([_series(svg, 100)], x_label="step", rules=[(5.0, "abort at 5")])
    assert 'class="rule"' in fig and "abort at 5" in fig


def test_a_quantile_chart_carries_the_three_quantiles_it_draws(svg):
    windows = [svg.QuantileWindow(x=float(i), p10=1.0, p50=2.0 + i, p90=3.0 + i)
               for i in range(50)]
    fig = svg.quantile_chart(windows, x_label="games", label="plies")
    assert "<polygon" in fig and "<polyline" in fig
    assert "p10" in fig and "median" in fig and "p90" in fig


def test_a_tick_chart_draws_one_tick_per_event_with_its_class(svg):
    fig = svg.tick_chart([(10.0, "alert"), (20.0, "fire")], x_range=(0.0, 100.0),
                         x_label="step", label="ticks")
    assert fig.count('<line class="tick alert"') == 1 and fig.count('<line class="tick fire"') == 1
