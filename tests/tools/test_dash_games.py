"""The dash game view: tactics against the engine's oracle, the six checked against the record, arms and stats as recorded."""
from __future__ import annotations

import importlib
import inspect
import json

import pytest

from _dash_record import game, six_in_a_row_for_p1


@pytest.fixture(scope="module")
def tactics(dash):
    return importlib.import_module("dash.readers.tactics")


@pytest.fixture(scope="module")
def games(dash):
    return importlib.import_module("dash.readers.games")


#: Hand-built games covering a five to finish, a four to block, a double four and a quiet board.
_FIXTURES = {
    "six": six_in_a_row_for_p1(),
    "check": [[0, 0], [0, 3], [1, 3], [1, 0], [2, 0], [0, 4], [1, 4], [3, 0], [-1, 0], [5, 5], [6, 6]],
    "double": [[0, 0], [9, 9], [9, 8], [1, 0], [2, 0], [9, 7], [8, 9], [0, 1], [0, 2], [7, 7], [6, 6], [3, 0], [0, 3]],
    "quiet": [[0, 0], [2, 2], [-2, -2], [4, 0], [0, 4]],
}


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_tactics_agree_with_the_engines_winning_moves_on_every_position(tactics, name):
    assert tactics.oracle_disagreements([tuple(m) for m in _FIXTURES[name]]) == []


def test_stones_left_follow_the_turn_structure(tactics):
    assert [tactics.stones_left(p) for p in range(6)] == [1, 2, 1, 2, 1, 2]


def test_a_four_must_be_blocked_and_a_five_is_won(tactics):
    check = [tuple(m) for m in _FIXTURES["check"]]
    block = tactics.read_position(check, 9)
    assert block.mover == 1 and block.cls == "block" and block.fours >= 1 and block.cells
    won = tactics.read_position([tuple(m) for m in _FIXTURES["six"]], 11)
    assert won.cls == "win" and (5, 0) in won.cells


def test_a_quiet_board_has_no_cells(tactics):
    assert tactics.read_position([tuple(m) for m in _FIXTURES["quiet"]], 5).cls == "quiet"


def test_the_six_through_the_last_stone_is_checked_against_the_record(games):
    ok = games.GameView.from_record(game("a", six_in_a_row_for_p1(), result="p1"))
    assert ok.win == [(k, 0) for k in range(6)] and ok.finding is None
    wrong = games.GameView.from_record(game("b", six_in_a_row_for_p1(), result="p2"))
    assert "belongs to p1" in wrong.finding
    no_line = games.GameView.from_record(game("c", six_in_a_row_for_p1()[:-1], termination="six_in_a_row"))
    assert "no line of six" in no_line.finding


def test_arms_are_read_only_when_parallel_to_the_moves(games):
    moves = six_in_a_row_for_p1()
    good = games.GameView.from_record(game("a", moves, move_arms=["opening"] + ["full", "fast"] * 5 + ["full"],
                                           move_sims=[0] + [320, 64] * 5 + [320]))
    assert good.arms[1] == "full" and good.sims[2] == 64
    short = games.GameView.from_record(game("b", moves, move_arms=["full"], move_sims=[320]))
    assert short.arms is None and short.sims is None


@pytest.mark.parametrize(("stats", "state"), [("absent", "absent"), (None, "none"), ([], "empty"),
                                              ([{"ply": 0, "root_value": 0.1, "visits": [[0, 0, 3]]}], "recorded")])
def test_the_search_stats_field_states_are_kept_apart(games, stats, state):
    rec = game("a", [[0, 0]])
    if stats != "absent":
        rec["search_stats"] = stats
    assert games.GameView.from_record(rec).stats_field == state


def test_the_payload_carries_visits_by_count_tactics_per_position_and_the_strip(games):
    stats = [{"ply": 0, "root_value": 0.0, "visits": [[1, 1, 2], [0, 1, 9]]},
             {"ply": 1, "root_value": 0.6, "visits": [[2, 2, 4]]}, {"ply": 3, "root_value": 0.0, "visits": []}]
    view = games.GameView.from_record(game("a", six_in_a_row_for_p1(), stats=stats, step=-1))
    body = view.payload()
    assert body["stats"]["0"]["top"] == [[0, 1, 9], [1, 1, 2]] and body["stats"]["0"]["n"] == 11
    assert len(body["tactics"]) == len(view.moves) and body["tactics"][11]["cls"] == "win"
    assert [c["turn"] for c in body["chances"]] == [1, 2, 3] and body["turns"] == 7
    assert view.step == -1 and json.dumps(body)


def test_the_hex_facts_are_the_viewers_until_it_is_retired(dash, viewer):
    ours = importlib.import_module("dash.readers.hexlogic")
    theirs = importlib.import_module("viewer.hexlogic")
    for name in ("owner", "win_line"):
        assert inspect.getsource(getattr(ours, name)) == inspect.getsource(getattr(theirs, name))
