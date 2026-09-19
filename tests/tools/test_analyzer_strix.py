"""The strix engine: its driver's `analyze` op (skipped loudly without the vendor), its rows, its stated gaps."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from mantis._engine import Board
from mantis.bots.strix import strix_availability

AVAILABLE, WHY = strix_availability()
pytestmark = pytest.mark.skipif(not AVAILABLE, reason=f"strix vendor not built: {WHY} — run `make vendor.strix`")
REPO_ROOT = Path(__file__).resolve().parents[2]
ENC = "gnn_axis_r8"


@pytest.fixture(scope="module")
def strix(analyzer):
    return importlib.import_module("analyzer.strix")


@pytest.fixture(scope="module")
def engine(strix):
    eng = strix.StrixEngine(strix.strix_info(), encoding=ENC)
    yield eng
    eng.close()


def _board(moves: list[tuple[int, int]]) -> Board:
    board = Board.with_encoding_name(ENC)
    for q, r in moves:
        board.apply_move(q, r)
    return board


def test_the_driver_that_runs_is_this_trees(strix):
    assert strix.DRIVER == REPO_ROOT / "tools" / "strix_driver.py"
    assert strix.strix_info().kind == "strix" and strix.strix_info().note == ""


def test_the_empty_board_is_a_stated_refusal(strix, engine):
    with pytest.raises(strix.StrixRefused, match="p1 stone"):
        engine.raw_read(_board([]))


def test_the_raw_read_and_the_search_carry_full_children_rows(engine):
    board = _board([(0, 0), (1, 0), (0, 1)])
    raw = engine.raw_read(board)
    assert -1.0 <= raw.value <= 1.0 and len(raw.children) == len(board.legal_moves())
    assert sum(c[2] for c in raw.children) == pytest.approx(1.0, abs=1e-3)
    assert all(c[3] == 0 for c in raw.children)
    s = engine.search(board, 16)
    assert s.root_visits >= 1 and s.argmax in {c[0] for c in s.children} and -1.0 <= s.root_value <= 1.0
    assert engine.card["gaps"] and engine.card["radius"] == 8 and engine.card["forcing_solver"] == "on"


def test_the_raw_value_is_the_movers(engine):
    # p2 to move holding an open four on the q axis: the mover wins, so the mover's value is ≈ +1.
    board = _board([(0, 0), (5, 0), (6, 0), (0, 1), (0, 2), (7, 0), (8, 0), (0, 3), (0, 4)])
    assert board.current_player == -1
    assert engine.raw_read(board).value > 0.5


def test_the_dispatcher_lists_strix_and_analyzes_through_it(analyzer):
    dispatch = importlib.import_module("analyzer.dispatch")
    disp = dispatch.Dispatcher([], device="cpu", threads=1, strix=True)
    try:
        rows = disp.handle({"op": "engines"})["body"]["engines"]
        assert [r["id"] for r in rows] == ["strix"] and rows[0]["note"] == ""
        out = disp.handle({"op": "analyze", "engine": "strix", "moves": "0,0;1,0;0,1", "sims": 8, "seq": 1})
        assert out["status"] == 200, out
        rec = out["body"]["record"]
        assert rec["engine"]["encoding"] == dispatch.STRIX_FALLBACK_ENCODING and rec["search"]["quiescence_fires"] == 0
        assert rec["raw"]["derivation"].startswith("strix") and rec["search"]["derivation"].startswith("strix")
        empty = disp.handle({"op": "analyze", "engine": "strix", "moves": "", "sims": 0, "seq": 2})
        assert empty["status"] == 400 and "p1 stone" in empty["body"]["refused"]
    finally:
        disp.close()
