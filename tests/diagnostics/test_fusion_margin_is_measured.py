"""AUDIT-1 F-07 — `margin_achieved` is a measurement, and `margin_requested` is the input.

THE DEFECT. `run` wrote `"margin_achieved": float(args.margin) if recommending else None` —
the `--margin` argument echoed back under a name that says the tool measured it. The achieved
ratio (`predicted_peak_bytes / budget_bytes`) was not a field at all. `RECAL_SITTING4_RECORD`
and `RECAL_SITTING5_RECORD` both carry `| margin_achieved | 0.85 |`, read off exactly that,
and R327(c)'s `k = 0.849998` was hand-derived from elsewhere because the report could not
supply it.

WHY THE TWO NUMBERS ARE GENUINELY DIFFERENT, and not a rename for its own sake: `nodes` is a
FLOOR division of the usable budget, so the emitted pair is the largest INTEGER pair under the
requested fraction. What it is predicted to occupy is therefore at or below the request, and
the gap is exactly the granularity of one node — which is the number an operator sizing a card
wants, and the one the report was not carrying.

CPU-ONLY. `_recommend` is a pure function over the fit, the budget, the margin and the
measured points, so every row here runs without the GPU the recommending path needs.
"""
from __future__ import annotations

from typing import Any

import pytest

from mantis.diagnostics.fusion_calibrate import CalibrationRefusal, _recommend

#: A three-term fit whose numbers are round enough to check by hand: a fixed cost, a per-edge
#: term, a per-node term, and an operating ratio of 4 edges per node.
FIT: dict[str, Any] = {
    "a_bytes": 1_000_000.0,
    "b_bytes_per_edge": 100.0,
    "c_bytes_per_node": 200.0,
    "operating_edges_per_node": 4.0,
}
POINTS: list[dict[str, Any]] = [
    {"largest_graph_nodes": 10, "largest_graph_edges": 40},
    {"largest_graph_nodes": 12, "largest_graph_edges": 48},
]
BUDGET = 1_000_000_000


def test_the_report_carries_BOTH_the_request_and_the_measurement() -> None:
    rec = _recommend(FIT, BUDGET, 0.85, POINTS)
    assert rec["margin"] == 0.85, "the requested ceiling keeps its own name"
    assert "margin_achieved" in rec, "the measured fraction must be a field, not a derivation"


def test_the_achieved_margin_is_the_prediction_over_the_budget() -> None:
    """THE PIN. Computed from the report's own two fields — if the achieved value is ever
    `--margin` again, this is the row that says so."""
    rec = _recommend(FIT, BUDGET, 0.85, POINTS)
    assert rec["margin_achieved"] == pytest.approx(
        rec["predicted_peak_bytes"] / rec["budget_bytes"]
    )


@pytest.mark.parametrize("margin", [0.5, 0.6, 0.75, 0.85, 0.95])
def test_the_achieved_margin_TRACKS_the_request_without_ever_equalling_it(
    margin: float,
) -> None:
    """The audit's criterion, generalised: a run whose ratio is well below the request must
    report a value different from `--margin`. Floor division means the achieved fraction is
    at or just under the request at every setting, never above it."""
    rec = _recommend(FIT, BUDGET, margin, POINTS)
    assert rec["margin_achieved"] <= margin + 1e-12, (
        "the emitted pair is predicted to exceed the budget fraction it was solved against"
    )
    assert rec["margin_achieved"] == pytest.approx(margin, abs=1e-6), (
        "at this budget the granularity is one node in a billion bytes, so the two agree to "
        "six places — the row that separates them is the coarse-granularity one below"
    )


def test_a_COARSE_budget_separates_the_two_numbers_visibly() -> None:
    """The load-bearing row: at a budget where one node is a large fraction of the whole, the
    achieved margin is measurably below the request. If the field were still the echoed input
    it would read 0.85 exactly here, which is the misreport the sitting records carry."""
    coarse_fit = {**FIT, "a_bytes": 100.0, "b_bytes_per_edge": 1000.0,
                  "c_bytes_per_node": 3000.0}
    rec = _recommend(coarse_fit, 100_000, 0.85, POINTS)
    assert rec["margin_achieved"] < 0.85, rec
    assert rec["margin_achieved"] != pytest.approx(0.85, abs=1e-6), (
        "the field still reports the --margin input rather than what the pair occupies"
    )
    # and it is a real ratio, not a token
    assert 0.0 < rec["margin_achieved"] < 1.0


def test_the_requested_margin_is_never_silently_the_achieved_one() -> None:
    """Two different requests that land on the SAME integer pair must still report two
    different achieved values only if the pair differs — the achieved number is a function of
    the pair, and this row states that dependency rather than assuming it."""
    a = _recommend(FIT, BUDGET, 0.85, POINTS)
    b = _recommend(FIT, BUDGET, 0.42, POINTS)
    assert a["max_fused_nodes"] != b["max_fused_nodes"]
    assert a["margin_achieved"] != b["margin_achieved"]
    assert a["margin"] != b["margin"]


def test_a_budget_that_buys_no_pair_still_REFUSES_rather_than_reporting_a_margin() -> None:
    """The achieved margin must never be computed for a pair that was not emitted."""
    with pytest.raises(CalibrationRefusal):
        _recommend(FIT, 1_000, 0.85, POINTS)
