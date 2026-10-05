"""Turns, not stones: the Games payload's turn sentences, each net's whole turn in the Analyzer, the import route, a framed board."""
from __future__ import annotations

import importlib
import re
import time

import pytest

from _dash_record import game, segment_start, six_in_a_row_for_p1, trainer_rows, write_config, write_heartbeat, write_segment, write_shard

_STATS = [{"ply": 3, "root_value": 0.2, "visits": [[1, 0, 9], [8, 8, 1]]},
          {"ply": 4, "root_value": 0.3, "visits": [[2, 0, 4], [9, 9, 6]]}]


@pytest.fixture(scope="module")
def gv(dash):
    return importlib.import_module("dash.views.games")


def test_a_turn_start_carries_both_stones_and_a_sentence_for_each(gv, dash):
    GameView = importlib.import_module("dash.readers.games").GameView
    body = gv.payload(GameView.from_record(game("g", six_in_a_row_for_p1(), stats=_STATS)), "r1", None)
    turn = body["pos"][3]["turn"]
    assert turn["stones"] == [[1, 0], [2, 0]] and [t[0] for t in turn["texts"]] == ["First stone", "Second stone"]
    assert "its most-visited move" in turn["texts"][0][1] and "with 40 % of the visits" in turn["texts"][1][1]
    assert "turn" not in body["pos"][4] and body["pos"][0]["turn"]["texts"][0][0] == "The stone"
    assert body["pos"][3]["where"] == "<b>Turn 3</b> of 7, Light to place two stones"


def test_the_transport_steps_from_turn_start_to_turn_start(gv):
    starts = [0, 1, 3, 5, 7]
    assert gv._neighbours(starts, 9, 3) == (1, 5)
    assert gv._neighbours(starts, 9, 4) == (3, 5)
    assert gv._neighbours(starts, 9, 9) == (7, 9) and gv._neighbours(starts, 9, 0) == (0, 1)


def test_a_few_stones_are_framed_on_a_board_of_at_least_the_minimum_area(dash):
    board = importlib.import_module("dash.views.board")
    svg = board.render(board.Scene(moves=[(0, 0)], ply=1, ghosts=[((1, 0), "1"), ((2, 0), "2")]))
    _x, _y, w, h = (float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split())
    assert w >= board.MIN_W - 1e-6 and h >= board.MIN_H - 1e-6
    assert '<text class="ghostnum"' in svg and ">1<" in svg and ">2<" in svg


@pytest.fixture(scope="module")
def text(dash):
    return importlib.import_module("dash.views.analyzer_text")


def _rec(policy, run_id="r", step=45000, to_move="p2", k=2):
    return {"engine": {"id": f"{run_id}_{step}", "run_id": run_id, "step": step},
            "position": {"to_move": to_move, "moves_remaining": k}, "raw": {"value": 0.0, "argmax": [0, 1],
                                                                            "policy": [[q, r, p] for (q, r), p in policy.items()]},
            "search": {"absent": "sims=0"}, "tactics": {"class": "quiet", "cells": []}}


def test_each_nets_whole_turn_is_named_and_the_same_turn_said_once(text):
    pa = {(0, 1): 0.6, (1, 1): 0.4}
    both = text.compose(_rec(pa), _rec(pa, step=3000), None, None, None, turn_a=[(0, 1), (1, 1)], turn_b=[(0, 1), (1, 1)])
    assert "Both nets play (0, 1) then (1, 1)." in both["verdict"] and both["turn"]["a"] == [[0, 1], [1, 1]]
    split = text.compose(_rec(pa), _rec(pa, step=3000), None, None, None, turn_a=[(0, 1), (1, 1)], turn_b=[(0, 1), (2, 2)])
    assert "r at 45k plays (0, 1) then (1, 1); r at 3k plays (0, 1) then (2, 2)." in split["verdict"]


class _Analyst:
    """Answers each read with the moves' count as the chosen cell, so a turn's second read is visible in its answer."""

    def __init__(self):
        self.seen = []

    def submit(self, req):
        self.seen.append(req["moves"])
        n = len([m for m in req["moves"].split(";") if m])
        return {"status": 200, "body": {"record": {"position": {"moves_remaining": 2 if n % 2 == 1 else 1},
                                                   "raw": {"argmax": [n, 0]}, "search": {"absent": "x"}}}}


def test_the_desks_turn_reads_the_second_stone_on_the_position_after_the_first(dash):
    desk = importlib.import_module("dash.desk")
    fake = object.__new__(desk.Desk)
    fake.analyst = _Analyst()
    rec = {"position": {"moves_remaining": 2}, "raw": {"argmax": [5, 5]}}
    assert fake.turn("e", [(0, 0)], rec, 0, "c") == [(5, 5), (2, 0)]
    assert fake.analyst.seen == ["0,0;5,5"]
    assert fake.turn("e", [(0, 0), (1, 0)], {"position": {"moves_remaining": 1}, "raw": {"argmax": [7, 7]}}, 0, "c") == [(7, 7)]
    assert fake.turn("e", [], {"position": {"winner": "p1"}, "raw": {"argmax": [1, 1]}}, 0, "c") == []


def _hub(tmp_path):
    serve = importlib.import_module("dash.serve")
    record = importlib.import_module("dash.readers.record")
    run = tmp_path / "r1"
    write_segment(run / "logs", "r1", 1, [segment_start("r1", 1), *trainer_rows(range(1, 5))])
    write_config(run, "r1", None)
    write_heartbeat(run / "logs", "r1", time.time())
    write_shard(run / "logs" / "games", "r1", 1, "2026100510", [game("g1", six_in_a_row_for_p1())])
    hub = serve.Hub([record.RunRecord("r1", run)])
    hub.poll_once()
    return hub


def test_an_imported_line_gives_the_context_without_a_recorded_search(dash, tmp_path):
    desk = importlib.import_module("dash.desk")
    hub = _hub(tmp_path)
    line = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    on = desk.context(hub, None, None, line[:1], line)
    assert on.on_line and on.entry is None and on.nxt == (1, 0) and on.second == (2, 0)
    off = desk.context(hub, None, None, [(0, 0), (9, 9)], line)
    assert not off.on_line and off.off == 1


def test_the_import_form_opens_the_game_or_names_its_refusal(dash, tmp_path):
    routes = importlib.import_module("dash.analyzer_routes")
    hub = _hub(tmp_path)
    good = routes.post(hub, None)("/analyzer", b"htttx=version%5B1%5D%3B%0A1.+%5B1%2C0%5D%5B2%2C0%5D%3B")
    html = good.body.decode("utf-8")
    assert good.status == 200 and "An imported game, 3 stones" in html and '"id":"imported"' in html
    bad = routes.post(hub, None)("/analyzer", b"htttx=1.+%5B0%2C0%5D%5B1%2C0%5D%3B").body.decode("utf-8")
    assert '<p class="refused">turn 1: (0, 0) is already occupied</p>' in bad
    huge = routes.post(hub, None)("/analyzer", b"htttx=" + b"x" * (routes.MAX_IMPORT + 1)).body.decode("utf-8")
    assert "longer than" in huge


def test_the_scripts_step_by_turn_and_frame_the_board(dash):
    from pathlib import Path

    web = Path(__file__).resolve().parents[2] / "tools" / "dash" / "web"
    games, analyzer, board = ((web / n).read_text(encoding="utf-8") for n in ("games.js", "analyzer.js", "board.js"))
    assert "if (e.shiftKey) go(ply + 1); else stepTurn(1);" in games and "H.viewport(" in games and "H.viewport(" in analyzer
    assert "line_starts" in analyzer and "function viewport(" in board and "MIN_W" in board
