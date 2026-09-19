"""The record: values are the mover's (pinned on a WIN1 position), terminal positions call no engine, verdicts name the class."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="module")
def analysis(analyzer):
    return importlib.import_module("analyzer.analysis")


@pytest.fixture(scope="module")
def engine(analyzer, tmp_path_factory, mint_stamp):
    engines = importlib.import_module("analyzer.engines")
    stamp = mint_stamp(tmp_path_factory.mktemp("ckpt"))
    eng = engines.MantisEngine(engines.discover([stamp.parent])[0], device="cpu", threads=2)
    yield eng
    eng.close()


# p1 holds five on the q axis with both ends open; p1 to move at ply 11 with two stones: WIN1 at (-1,0) / (5,0).
WIN1 = [(0, 0), (0, 5), (1, 5), (1, 0), (2, 0), (0, 6), (1, 6), (3, 0), (4, 0), (0, 7), (1, 7)]
# p1 holds an open four; p2 to move at ply 9 with two stones: CHECK, the block set is the four's two ends.
CHECK = [(0, 0), (0, 3), (1, 3), (1, 0), (2, 0), (0, 4), (1, 4), (3, 0), (-1, 0)]
SIX = [(0, 0), (0, 5), (1, 5), (1, 0), (2, 0), (0, 6), (1, 6), (3, 0), (4, 0), (0, 7), (1, 7), (5, 0)]


def test_values_are_the_movers_pinned_on_a_win1_position(analysis, engine):
    one = analysis.analyze(engine, WIN1, 1)
    assert one["perspective"] == "to_move" and one["position"]["to_move"] == "p1"
    assert one["search"]["root_value"] == pytest.approx(1.0), "1 leaf: the root's own value is the quiescence override"
    assert one["search"]["quiescence_fires"] >= 1
    sixteen = analysis.analyze(engine, WIN1, 16)
    assert sixteen["search"]["root_value"] > 0.5
    assert -1.0 <= sixteen["raw"]["value"] <= 1.0 and sixteen["raw"]["derivation"].startswith("quiescence-off")
    # k=2: the census's win set is W1 ∪ W2 — a first stone two cells out still completes six with the second.
    assert sixteen["tactics"]["class"] == "win"
    assert sorted(sixteen["tactics"]["cells"]) == [[-2, 0], [-1, 0], [5, 0], [6, 0]]


def test_the_raw_argmax_is_the_max_prior_cell_not_the_returned_move(analysis, engine):
    rec = analysis.analyze(engine, CHECK, 0)
    best = max(rec["raw"]["policy"], key=lambda row: row[2])
    assert rec["raw"]["argmax"] == best[:2]
    assert rec["search"] == {"absent": "sims=0 (raw only)"}
    assert rec["tactics"]["verdict"]["search"] is None


def test_a_check_position_reads_block_with_the_fours_ends(analysis, engine):
    rec = analysis.analyze(engine, CHECK, 0)
    t = rec["tactics"]
    assert t["class"] == "block" and t["k"] == 2 and sorted(t["cells"]) == [[-2, 0], [4, 0]]
    assert t["opp_fours"] and t["derivation"] == f"mantis.diagnostics.tactics.analyze(k=2, radius={engine.radius})"
    assert t["verdict"]["raw"] in ("argmax BLOCKS", "argmax MISSES THE BLOCK — block set [(-2, 0), (4, 0)]")


class _Untouchable:
    """An engine that must not be called: a terminal position ends before any engine work."""

    card = {"id": "stub", "encoding": "gnn_axis_v1", "radius": 8}
    encoding = "gnn_axis_v1"
    radius = 8

    def raw_read(self, board):
        raise AssertionError("raw_read called on a terminal position")

    def search(self, board, sims):
        raise AssertionError("search called on a terminal position")


def test_a_terminal_position_is_absent_by_name_and_calls_no_engine(analysis):
    rec = analysis.analyze(_Untouchable(), SIX, 64)
    assert rec["position"]["winner"] == "p1"
    assert rec["raw"] == rec["search"] == {"absent": "position is terminal (p1 wins)"}
    assert rec["tactics"] == {"class": "terminal", "winner": "p1"}
    assert rec["symmetry"] == {"absent": "not requested"}


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


def test_a_refused_position_propagates_as_the_position_refusal(analysis, engine):
    position = importlib.import_module("analyzer.position")
    with pytest.raises(position.PositionRefused, match="ply 1"):
        analysis.analyze(engine, [(0, 0), (0, 0)], 0)
