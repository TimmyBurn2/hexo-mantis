"""One game's view: moves, owners, the six, the arms, the stats by ply, and every absence named."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="module")
def games(observatory):
    return importlib.import_module("observatory.readers.games")


def _six_for_p1() -> list[list[int]]:
    p1 = [[k, 0] for k in range(6)]
    p2 = [[2 * k, 5 + k % 2] for k in range(6)]
    return [p1[0], p2[0], p2[1], p1[1], p1[2], p2[2], p2[3], p1[3], p1[4], p2[4], p2[5], p1[5]]


def _record(**over) -> dict:
    row = {"contract": "game-record-v1", "game_id": "g", "run_id": "t", "channel": "selfplay", "step": 7,
           "step_kind": "actor", "served_sims": 64, "plies": 12, "result": "p1",
           "termination": "six_in_a_row", "moves": _six_for_p1()}
    row.update(over)
    return row


def test_a_self_play_game_carries_owners_the_six_and_the_stats_gap(games):
    view = games.GameView.from_record(_record(), "t")
    assert view.owners[:5] == [0, 1, 1, 0, 0] and view.win == [(k, 0) for k in range(6)]
    assert view.stats_by_ply == {} and "no per-position search stats" in (view.stats_absent_reason or "")
    assert view.arms is None and view.candidate is None and view.finding is None


def test_the_arms_and_sims_per_arm_are_read_from_the_record_never_inferred(games):
    arms = ["opening"] * 4 + ["full"] * 2 + ["fast"] * 6
    sims = [0] * 4 + [320] * 2 + [64] * 6
    view = games.GameView.from_record(_record(move_arms=arms, move_sims=sims), "t")
    assert view.arms == "ooooffqqqqqq" and view.sims_by_arm == {"f": 320, "q": 64}
    broken = games.GameView.from_record(_record(move_arms=arms[:3], move_sims=sims), "t")
    assert broken.arms is None and broken.sims_by_arm == {}


def test_an_eval_game_keys_its_stats_by_ply_and_names_an_empty_root_list(games):
    stats = [{"ply": 0, "by": "candidate", "root_value": 0.2, "visits": [[1, 0, 3]]},
             {"ply": 3, "by": "candidate", "root_value": -0.1, "visits": []}]
    view = games.GameView.from_record(_record(channel="promotion", step_kind="round",
                                              colors={"candidate": 1, "opponent": -1},
                                              served_sims=512, search_stats=stats), "t")
    assert set(view.stats_by_ply) == {0, 3} and view.stats_by_ply[3]["visits"] == []
    assert view.stats_absent_reason is None and view.candidate == 1 and view.served_sims == 512
    none = games.GameView.from_record(_record(channel="external", colors={"candidate": -1, "opponent": 1},
                                              search_stats=None), "t")
    assert none.stats_absent_reason == "no player exposed a search root on this game"


def test_a_record_whose_six_belongs_to_the_other_side_is_a_finding(games):
    view = games.GameView.from_record(_record(result="p2"), "t")
    assert view.finding is not None and "belongs to p1" in view.finding
    view = games.GameView.from_record(_record(moves=_six_for_p1()[:-1], plies=11, termination="six_in_a_row"), "t")
    assert view.finding is not None and "no line of six" in view.finding
