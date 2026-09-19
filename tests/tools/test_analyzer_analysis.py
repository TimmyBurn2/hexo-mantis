"""The record: values are the mover's (pinned on a WIN1 position), terminal positions call no engine, verdicts name the class."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="module")
def analysis(analyzer):
    return importlib.import_module("analyzer.analysis")


def test_the_head_reads_the_solver_override_on_a_win1_position_in_the_movers_view(analysis, mantis_engine, positions):
    one = analysis.analyze(mantis_engine, positions["WIN1"], 1)
    assert one["perspective"] == "to_move" and one["position"]["to_move"] == "p1"
    assert one["search"]["root_value"] == pytest.approx(1.0), "1 leaf: the root's own value is the quiescence override"
    assert one["search"]["quiescence_fires"] >= 1
    sixteen = analysis.analyze(mantis_engine, positions["WIN1"], 16)
    assert sixteen["search"]["root_value"] > 0.5
    assert -1.0 <= sixteen["raw"]["value"] <= 1.0 and sixteen["raw"]["derivation"].startswith("quiescence-off")
    # k=2: the census's win set is W1 ∪ W2 — a first stone two cells out still completes six with the second.
    assert sixteen["tactics"]["class"] == "win"
    assert sorted(sixteen["tactics"]["cells"]) == [[-2, 0], [-1, 0], [5, 0], [6, 0]]


def test_the_raw_argmax_is_the_max_prior_cell_not_the_returned_move(analysis, mantis_engine, positions):
    rec = analysis.analyze(mantis_engine, positions["CHECK"], 0)
    best = max(rec["raw"]["policy"], key=lambda row: row[2])
    assert rec["raw"]["argmax"] == best[:2]
    assert rec["search"] == {"absent": "sims=0 (raw only)"}
    assert rec["tactics"]["verdict"]["search"] is None


def test_a_check_position_reads_block_with_the_fives_ends(analysis, mantis_engine, positions):
    rec = analysis.analyze(mantis_engine, positions["CHECK"], 0)
    t = rec["tactics"]
    assert t["class"] == "block" and t["k"] == 2 and sorted(t["cells"]) == [[-2, 0], [4, 0]]
    assert t["opp_fours"] and t["derivation"] == f"mantis.diagnostics.tactics.analyze(k=2, radius={mantis_engine.radius})"
    assert t["verdict"]["raw"] == analysis.verdict("block", {(-2, 0), (4, 0)}, rec["raw"]["argmax"])


class _Untouchable:
    """An engine that must not be called: a terminal position ends before any engine work."""

    card = {"id": "stub", "encoding": "gnn_axis_v1", "radius": 8}
    encoding = "gnn_axis_v1"
    radius = 8
    raw_derivation = head_derivation = "stub"

    def raw_read(self, board):
        raise AssertionError("raw_read called on a terminal position")

    def search(self, board, sims):
        raise AssertionError("search called on a terminal position")


def test_a_terminal_position_is_absent_by_name_and_calls_no_engine(analysis, positions):
    rec = analysis.analyze(_Untouchable(), positions["SIX"], 64)
    assert rec["position"]["winner"] == "p1"
    assert rec["raw"] == rec["search"] == {"absent": "position is terminal (p1 wins)"}
    assert rec["tactics"] == {"class": "terminal", "winner": "p1"}
    assert rec["symmetry"] == {"absent": "not requested"}
    with_sweep = analysis.analyze(_Untouchable(), positions["SIX"], 0, symmetry=True)
    assert with_sweep["symmetry"] == {"absent": "position is terminal (p1 wins)"}


@pytest.mark.parametrize("cls,cells,argmax,expected", [
    ("win", {(5, 0), (-1, 0)}, (5, 0), "argmax WINS"),
    ("win", {(5, 0), (-1, 0)}, (2, 2), "argmax MISSES WIN at [(-1, 0), (5, 0)]"),
    ("block", {(4, 0)}, (4, 0), "argmax BLOCKS"),
    ("block", {(4, 0)}, (0, 4), "argmax MISSES THE BLOCK — block set [(4, 0)]"),
    ("lost1", set(), (0, 0), "no block exists (LOST1)"),
    ("quiet", set(), (0, 0), "quiet"),
])
def test_the_verdict_names_the_class_and_the_miss(analysis, cls, cells, argmax, expected):
    assert analysis.verdict(cls, cells, argmax) == expected


def test_a_refused_position_propagates_as_the_position_refusal(analysis, mantis_engine):
    position = importlib.import_module("analyzer.position")
    with pytest.raises(position.PositionRefused, match="ply 1"):
        analysis.analyze(mantis_engine, [(0, 0), (0, 0)], 0)
