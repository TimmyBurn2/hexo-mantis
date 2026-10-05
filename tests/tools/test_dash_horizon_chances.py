"""The knowledge horizon and the win-chance strip against hand-computed fixtures; a turn's second stone is never read."""
from __future__ import annotations

import importlib

import pytest

from _dash_record import game


@pytest.fixture(scope="module")
def horizon(dash):
    return importlib.import_module("dash.readers.horizon")


@pytest.fixture(scope="module")
def chances(dash):
    return importlib.import_module("dash.readers.chances")


def _stats(values: dict[int, float]) -> list[dict]:
    return [{"ply": p, "root_value": v, "visits": []} for p, v in values.items()]


def test_a_p1_win_reads_agreement_by_turns_to_the_end(horizon):
    red = horizon.HorizonReducer()
    # 7 plies = turns 1..4; p1 owns plies 0, 3, 4; p2 owns 1, 2, 5, 6. First stones: 0 (t1), 1 (t2), 3 (t3), 5 (t4).
    red(game("a", [[i, 0] for i in range(7)], result="p1", stats=_stats({0: 0.2, 1: -0.5, 3: 0.4, 5: 0.3})))
    rows = list(red._games[0])
    assert [tuple(rows[i:i + 3]) for i in range(0, len(rows), 3)] == [(3, 1, 1), (2, 0, 1), (1, 1, 1), (0, 0, 0)]


def test_a_turns_second_stone_is_never_read(horizon):
    red = horizon.HorizonReducer()
    red(game("a", [[i, 0] for i in range(7)], result="p1", stats=_stats({2: 0.9, 4: 0.9, 6: 0.9})))
    assert red.read() is None


@pytest.mark.parametrize("over", [{"channel": "promotion"}, {"result": "draw"}, {"stats": None}])
def test_only_sampled_decided_self_play_games_count(horizon, over):
    red = horizon.HorizonReducer()
    kwargs = {"result": "p1", "stats": _stats({0: 0.5}), **over}
    red(game("a", [[0, 0], [1, 0]], **kwargs))
    assert red.read() is None


def test_the_windows_are_the_first_and_last_fifth_and_the_reach_needs_every_turn_at_ninety(horizon, monkeypatch):
    monkeypatch.setattr(horizon, "MIN_POSITIONS", 1)
    red = horizon.HorizonReducer()
    moves = [[i, 0] for i in range(7)]
    for i in range(10):
        good = i >= 5
        red(game(f"g{i}", moves, result="p1", step=i,
                 stats=_stats({0: 0.2, 1: -0.5, 3: 0.4 if good else -0.4, 5: -0.3})))
    read = red.read()
    assert read.games == 10 and read.window_games == 2
    assert read.early_steps == (0, 1) and read.late_steps == (8, 9)
    assert read.late_reach == 3 and read.early_reach == 0
    assert read.late_won.share[1] == 1.0 and read.early_won.share[1] == 0.0


def test_a_turn_below_the_minimum_count_is_a_gap_not_a_share(horizon):
    red = horizon.HorizonReducer()
    red(game("a", [[i, 0] for i in range(7)], result="p1", stats=_stats({0: 0.2})))
    read = red.read()
    assert read.late_won.share[3] is None and read.late_won.n[3] == 1 and read.late_reach is None


def test_light_chance_and_costs_follow_the_root_value_of_each_turns_first_stone(chances):
    # ply 0 Light v=0.0 -> 0.5; ply 1 Dark v=0.6 -> Light 0.2; ply 3 Light v=0.0 -> 0.5; ply 2 is a second stone.
    stats = {0: {"root_value": 0.0}, 1: {"root_value": 0.6}, 2: {"root_value": -1.0}, 3: {"root_value": 0.0}}
    pts = chances.points(stats)
    assert [(p.ply, p.turn, round(p.light, 3)) for p in pts] == [(0, 1, 0.5), (1, 2, 0.2), (3, 3, 0.5)]
    assert [None if p.cost is None else round(p.cost, 3) for p in pts] == [0.3, 0.3, None]


def test_the_turning_point_is_the_largest_cost_at_or_above_the_amber_mark(chances):
    # Light 0.5 (t1), 0.5 (t2, Dark), 0.8 (t3, Light), 0.1 (t4, Dark): t2 costs Dark 0.3, t3 costs Light 0.7.
    stats = {0: {"root_value": 0.0}, 1: {"root_value": 0.0}, 3: {"root_value": 0.6}, 5: {"root_value": 0.8}}
    pts = chances.points(stats)
    assert [None if p.cost is None else round(p.cost, 3) for p in pts] == [0.0, 0.3, 0.7, None]
    tp = chances.turning_point(pts)
    assert tp.ply == 3 and tp.cost == pytest.approx(0.7) and tp.cost >= chances.RED


def test_no_turn_costs_enough_means_no_turning_point(chances):
    pts = chances.points({0: {"root_value": 0.0}, 1: {"root_value": 0.1}, 3: {"root_value": 0.05}})
    assert chances.turning_point(pts) is None


def test_a_gap_in_the_recorded_turns_carries_no_cost(chances):
    pts = chances.points({0: {"root_value": 0.0}, 5: {"root_value": 0.9}})
    assert all(p.cost is None for p in pts)


def test_a_non_numeric_root_value_is_skipped_never_read_as_zero(chances):
    pts = chances.points({0: {"root_value": None}, 1: {"root_value": True}, 3: {"root_value": 0.2}})
    assert [p.ply for p in pts] == [3]
