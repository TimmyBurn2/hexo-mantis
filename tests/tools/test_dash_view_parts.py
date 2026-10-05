"""The dash view parts: formatting, the verdict statistics and the SVG primitives, each against hand values."""
from __future__ import annotations

import importlib
import json
import re

import pytest


@pytest.fixture(scope="module")
def fmt(dash):
    return importlib.import_module("dash.views.fmt")


@pytest.fixture(scope="module")
def stats(dash):
    return importlib.import_module("dash.views.stats")


@pytest.fixture(scope="module")
def svg(dash):
    return importlib.import_module("dash.views.svg")


@pytest.mark.parametrize(("value", "digits", "text"), [(55170, 0, "55 170"), (-0.25, 2, "−0.25"),
                                                       (-0.0001, 2, "0.00"), (None, 0, "—"), (float("nan"), 0, "—")])
def test_numbers_read_with_thin_thousands_and_a_real_minus(fmt, value, digits, text):
    assert fmt.num(value, digits) == text


@pytest.mark.parametrize(("step", "text"), [(45000, "45k"), (50800, "50.8k"), (900, "900"), (None, "—")])
def test_steps_read_the_way_the_axes_do(fmt, step, text):
    assert fmt.short(step) == text


def test_shares_signs_and_cells(fmt):
    assert fmt.pct(0.4712, 1) == "47.1 %" and fmt.pct(None) == "—"
    assert fmt.signed(0.17) == "+0.17" and fmt.signed(-0.04) == "−0.04"
    assert fmt.cell(-6, -1) == "(−6, −1)" and fmt.sig(0.51749) == "0.517" and fmt.sig(20.44) == "20.4"


def test_a_wall_time_is_labelled_cet_or_cest(fmt):
    assert fmt.when(1791198000.0).endswith("CEST") and fmt.when(1798761600.0).endswith("CET")
    assert fmt.when(None) == "—"


def test_escaping_covers_quotes(fmt):
    assert fmt.esc('"<a>&\'') == "&quot;&lt;a&gt;&amp;&#x27;"


@pytest.mark.parametrize(("pa", "pb", "n", "sign"), [(0.76, 0.59, 576, 1), (0.61, 0.59, 576, 0), (0.40, 0.59, 576, -1),
                                                     (0.142, 0.111, 288, 0)])
def test_a_separation_has_a_sign_only_past_its_interval(stats, pa, pb, n, sign):
    assert stats.separation(pa, n, pb, n).sign == sign


def test_wilson_contains_the_share_and_widens_with_fewer_games(stats):
    lo, hi = stats.wilson(940, 2000)
    assert lo < 0.47 < hi and hi - lo < 0.05
    lo2, hi2 = stats.wilson(47, 100)
    assert hi2 - lo2 > hi - lo and stats.wilson(0, 0) == (0.0, 1.0)


def test_logit_and_expit_invert(stats):
    assert stats.expit(stats.logit(0.594)) == pytest.approx(0.594)


def test_an_empty_bucket_is_absent_never_a_zero(svg):
    pairs = [(0.0, 1.0), (1.0, 3.0), (100.0, 5.0)]
    out = svg.bucket(pairs, n=10)
    assert [round(b[1], 3) for b in out] == [2.0, 5.0] and svg.bucket([], 10) == []


def test_smoothing_and_the_median_half_range(svg):
    assert svg.smooth([0.0, 3.0, 6.0], k=3) == [1.5, 3.0, 4.5]
    assert svg.median_half_range([(0, 0, -1, 1), (1, 0, -3, 3), (2, 0, -2, 2)]) == 2.0


def test_ticks_are_round_and_cover_the_span(svg):
    assert svg.ticks(0.0, 36670.0, 4) == [0.0, 10000.0, 20000.0, 30000.0]
    assert svg.ticks(0.47, 0.53, 3) == pytest.approx([0.475, 0.5, 0.525])


def test_a_one_point_series_is_a_dot_and_the_readout_data_cannot_close_its_script(svg):
    chart = svg.Chart("t</script>", lines=[svg.Line("a</script>", "c1", [(1.0, 0.5)])])
    out = chart.render()
    assert '<circle class="dot c1"' in out and "<polyline" not in out
    blob = re.search(r'<script type="application/json" class="xh">(.*)</script>$', out).group(1)
    assert "</" not in blob and json.loads(blob)["series"][0]["name"] == "a</script>"


def test_a_marked_reference_is_dashed_and_labelled_at_the_end(svg):
    out = svg.Chart("t", lines=[svg.Line("a", "c1", [(0.0, 0.1), (1.0, 0.2)])],
                    refs=[svg.Ref(0.15, label="line", marked=True, end=True)]).render()
    assert 'class="refmark"' in out and 'text-anchor="end">line<' in out
