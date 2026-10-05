"""Turns, not stones: the Games payload's turn sentences, each net's whole turn in the Analyzer, the import route, a framed board."""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from _dash_record import game, six_in_a_row_for_p1

WEB = Path(__file__).resolve().parents[2] / "tools" / "dash" / "web"

_STATS = [{"ply": 3, "root_value": 0.2, "visits": [[1, 0, 9], [8, 8, 1]]},
          {"ply": 4, "root_value": 0.3, "visits": [[2, 0, 4], [9, 9, 6]]}]


@pytest.fixture(scope="module")
def gv(dash):
    return importlib.import_module("dash.views.games")


def test_a_turn_start_carries_both_stones_and_a_sentence_for_each(gv, dash):
    GameView = importlib.import_module("dash.readers.games").GameView
    body = gv.payload(GameView.from_record(game("g", six_in_a_row_for_p1(), stats=_STATS)), "r1", None)
    turn = body["pos"][3]["turn"]
    assert turn["stones"] == [[1, 0], [2, 0]] and [t[0] for t in turn["texts"]] == ["Stone 1", "Stone 2"]
    assert "top move" in turn["texts"][0][1] and "40 % of the visits" in turn["texts"][1][1]
    assert "turn" not in body["pos"][4] and body["pos"][0]["turn"]["texts"][0][0] == "Stone"
    assert body["pos"][3]["where"] == "<b>Turn 3</b> of 7, Light to place two stones"
    assert body["htttx"]["head"] == "version[1];" and tuple(body["htttx"]["turns"][1]) == (3, 5, "2. [1,0][2,0];", "2. [1,0];")


def test_the_transport_steps_from_turn_start_to_turn_start(gv):
    starts = [0, 1, 3, 5, 7]
    assert gv._neighbours(starts, 9, 3) == (1, 5)
    assert gv._neighbours(starts, 9, 4) == (3, 5)
    assert gv._neighbours(starts, 9, 9) == (7, 9) and gv._neighbours(starts, 9, 0) == (0, 1)


def test_a_few_stones_are_framed_on_a_board_of_at_least_the_minimum_area(dash):
    board = importlib.import_module("dash.views.board")
    svg = board.render(board.Scene(moves=[(0, 0)], ply=1, ghosts=[((1, 0), "1", "ghost"), ((2, 0), "2", "ghost")]))
    _x, _y, w, h = (float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split())
    assert w >= board.MIN_W - 1e-6 and h >= board.MIN_H - 1e-6
    assert '<text class="ghostnum"' in svg and ">1<" in svg and ">2<" in svg


def test_a_numbered_stone_takes_its_cells_label_from_the_policy_percent(dash):
    board = importlib.import_module("dash.views.board")
    svg = board.render(board.Scene(moves=[(0, 0)], ply=1, heat=[((1, 0), 1.0, "42"), ((3, 0), 0.5, "21")],
                                   ghosts=[((1, 0), "1", "ghost"), ((2, 0), "2", "ghost")]))
    assert ">42<" not in svg and ">21<" in svg and ">1<" in svg


def test_a_frame_past_the_cell_cap_draws_its_stones_without_the_empty_grid(dash):
    board = importlib.import_module("dash.views.board")
    svg = board.render(board.Scene(moves=[(0, 0), (400, 0), (0, 400)], ply=3))
    assert 'class="cell"' not in svg and svg.count('class="s') == 3


def test_the_browsers_board_constants_are_the_servers(dash):
    board = importlib.import_module("dash.views.board")
    js = (WEB / "board.js").read_text(encoding="utf-8")
    assert "const MIN_W = SQ3 * 15, MIN_H = 1.5 * 13;" in js and f"const MAX_CELLS = {board.MAX_CELLS};" in js
    assert (board.MIN_W, board.MIN_H) == (3 ** 0.5 * 15, 1.5 * 13)


@pytest.fixture(scope="module")
def text(dash):
    return importlib.import_module("dash.views.analyzer_text")


def _rec(policy, run_id="r", step=45000, to_move="p2", k=2):
    return {"engine": {"id": f"{run_id}_{step}", "run_id": run_id, "step": step},
            "position": {"to_move": to_move, "moves_remaining": k}, "raw": {"value": 0.0, "argmax": [0, 1],
                                                                            "policy": [[q, r, p] for (q, r), p in policy.items()]},
            "search": {"absent": "sims=0"}, "tactics": {"class": "quiet", "cells": []}}


def test_each_nets_whole_turn_is_named_and_the_same_turn_said_once(text):
    pa, turn = {(0, 1): 0.6, (1, 1): 0.4}, text.Turn
    both = text.compose(_rec(pa), _rec(pa, step=3000), None, None, None, turn_a=turn([(0, 1), (1, 1)], searched=True),
                        turn_b=turn([(0, 1), (1, 1)]))
    assert "Both nets play (0, 1) then (1, 1)." in both["verdict"] and both["turn"]["a"] == [[0, 1], [1, 1]]
    assert both["turn"]["note"] == "Numbered: each net's turn, the reading net's from its search."
    assert both["turn"]["ghosts"] == [[0, 1, "1", "gab"], [1, 1, "2", "gab"]]
    split = text.compose(_rec(pa), _rec(pa, step=3000), None, None, None, turn_a=turn([(0, 1), (1, 1)]),
                         turn_b=turn([(0, 1), (2, 2)]))
    assert ('<span class="c1">r\u00a0at\u00a045k</span> plays (0, 1) then (1, 1); <span class="c2">r\u00a0at\u00a03k</span> plays (0, 1) then (2, 2).'
            in split["verdict"])
    cut = text.compose(_rec(pa), None, None, None, None, turn_a=turn([(0, 1)], unread="the analyst did not answer"))
    assert "r\u00a0at\u00a045k</span> plays (0, 1), its second stone not read (the analyst did not answer)." in cut["verdict"]


class _Analyst:
    """Answers each read with the moves' count as the chosen cell; `fail` makes the follow-up reads answer that instead."""

    def __init__(self, fail=None):
        self.seen, self.fail = [], fail

    def submit(self, req):
        self.seen.append((req["engine"], req["moves"], req["sims"], req["client"]))
        if self.fail and req["client"].endswith(":turn"):
            return self.fail
        n = len([m for m in req["moves"].split(";") if m])
        search = {"argmax": [n, 1], "ms": 1500.0} if req["sims"] else {"absent": "x"}
        return {"status": 200, "body": {"record": {"position": {"moves_remaining": 2 if n % 2 == 1 else 1},
                                                   "raw": {"argmax": [n, 0]}, "search": search}}}


def _desk(fail=None):
    desk = importlib.import_module("dash.desk")
    fake = object.__new__(desk.Desk)
    fake.analyst = _Analyst(fail)
    return fake


def test_the_second_stone_is_read_by_the_same_net_at_the_same_depth_under_its_own_key(dash):
    d = _desk()
    got = d.turn("e", [(0, 0)], {"position": {"moves_remaining": 2}, "search": {"argmax": [5, 5]}}, 256, "c")
    assert got.stones == [(5, 5), (2, 1)] and got.searched and got.ms == 1500.0 and got.unread is None
    assert d.analyst.seen == [("e", "0,0;5,5", 256, "c:turn")]
    assert d.turn("e", [(0, 0), (1, 0)], {"position": {"moves_remaining": 1}, "raw": {"argmax": [7, 7]}}, 0, "c").stones == [(7, 7)]
    assert d.turn("e", [], {"position": {"winner": "p1"}, "raw": {"argmax": [1, 1]}}, 0, "c").stones == []


def test_a_second_read_that_fails_is_named_and_one_superseded_makes_the_read_stale(dash):
    timeout = {"status": 504, "body": {"ok": False, "refused": "the analyst did not answer within 120.0 s"}}
    got = _desk(timeout).turn("e", [(0, 0)], {"position": {"moves_remaining": 2}, "raw": {"argmax": [5, 5]}}, 0, "c")
    assert got.stones == [(5, 5)] and got.unread == "the analyst did not answer within 120.0 s"
    desk = importlib.import_module("dash.desk")
    stale = _desk({"status": 200, "body": {"superseded": True}})
    assert stale.read("a", None, [(0, 0)], desk.Context(None, None, False, 0, None, None, None)) == (200, {"superseded": True})


def test_the_read_composes_both_nets_turns_and_the_positions_htttx(dash):
    desk = importlib.import_module("dash.desk")
    status, out = _desk().read("a", "b", [(0, 0)], desk.Context(None, None, False, 0, None, None, None), sims=64)
    turn = out["panel"]["turn"]
    assert status == 200 and turn["a"] == [[1, 1], [2, 1]] and turn["b"] == [[1, 0], [2, 0]] and turn["second_ms"] == 1500.0
    assert out["panel"]["htttx"] is None and out["panel"]["turn_starts"] == [0] and not out["panel"]["htttx_moved"]
    _, two = _desk().read("a", None, [(0, 0), (1, 0)], desk.Context(None, None, False, 0, None, None, None))
    assert two["panel"]["htttx"] == "version[1];\n1. [1,0];\n"


def test_a_won_position_says_it_is_final_not_whose_turn(dash):
    analyzer = importlib.import_module("dash.views.analyzer")
    line = [tuple(m) for m in six_in_a_row_for_p1()]
    assert analyzer.where(line, "p1") == "<b>Final position</b>, Light has six in a row"
    assert analyzer.where(line[:3]) == "<b>Turn 3</b>, Light to place two stones"
    assert analyzer.where(line[:4]) == "<b>Turn 3</b>, Light places stone 2 of 2"


def test_an_imported_line_gives_the_context_without_a_recorded_search(dash, dash_game_hub):
    desk = importlib.import_module("dash.desk")
    line = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    on = desk.context(dash_game_hub, None, None, line[:1], line)
    assert on.on_line and on.entry is None and on.nxt == (1, 0) and on.second == (2, 0)
    off = desk.context(dash_game_hub, None, None, [(0, 0), (9, 9)], line)
    assert not off.on_line and off.off == 1


def test_the_import_form_opens_the_game_or_keeps_the_text_and_reads_nothing(dash, dash_game_hub):
    routes, built = importlib.import_module("dash.analyzer_routes"), []

    def desk_of():
        built.append(1)
        raise AssertionError("a refused import must not build the desk")

    good = routes.post(dash_game_hub, None)("/analyzer", b"htttx=version%5B1%5D%3B%0A1.+%5B1%2C0%5D%5B2%2C0%5D%3B")
    html = good.body.decode("utf-8")
    assert good.status == 200 and "Imported game, 3 stones" in html and '"id":"imported"' in html
    bad = routes.post(dash_game_hub, desk_of)("/analyzer", b"htttx=1.+%5B0%2C0%5D%5B1%2C0%5D%3B%3Cb%3E").body.decode("utf-8")
    assert '<p class="refused">turn 1: (0, 0) is already occupied</p>' in bad and "1. [0,0][1,0];&lt;b&gt;</textarea>" in bad
    huge = routes.post(dash_game_hub, desk_of)("/analyzer", b"htttx=" + b"x" * (routes.MAX_IMPORT + 1)).body.decode("utf-8")
    assert "longer than" in huge and not built
