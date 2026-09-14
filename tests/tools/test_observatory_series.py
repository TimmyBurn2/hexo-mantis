"""The reducers: the same numbers the dashboard reads, held as arrays, never as parsed dicts."""
from __future__ import annotations

import importlib
from array import array

import pytest


@pytest.fixture(scope="module")
def series(observatory):
    return importlib.import_module("observatory.readers.series")


def _feed(series, rows):
    red = series.Reducers()
    for row in rows:
        red.feed(row)
    return red.snapshot()


def test_a_trainer_step_row_is_reduced_to_arrays_and_not_kept(series):
    snap = _feed(series, [{"event": "trainer_step", "step": 1, "value_loss": 0.5, "policy_loss": 2.0,
                           "loss": 2.5, "grad_norm": 3.0, "lr": 1e-3, "ts": 10.0},
                          {"event": "trainer_step", "step": 2, "value_loss": 0.4, "policy_loss": float("nan"),
                           "loss": None, "grad_norm": 2.0, "lr": 1e-3, "ts": 11.0}])
    s = snap.series("trainer_step", "step", "value_loss")
    assert isinstance(s.x, array) and s.x.typecode == "d"
    assert s.pairs() == [(1.0, 0.5), (2.0, 0.4)]
    assert snap.series("trainer_step", "step", "policy_loss").pairs() == [(1.0, 2.0)], "NaN is not a point"
    assert snap.series("trainer_step", "step", "loss").pairs() == [(1.0, 2.5)], "None is not a point"
    assert snap.rows("trainer_step") == [], "per-step rows are reduced, never kept"
    last = snap.last("trainer_step")
    assert last is not None and last["step"] == 2 and last["grad_norm"] == 2.0
    assert snap.counts == {"trainer_step": 2}
    assert snap.steps_max == 2 and snap.first_ts == 10.0 and snap.last_ts == 11.0


def test_an_unreduced_series_is_a_key_error_not_an_empty_series(series):
    snap = _feed(series, [{"event": "trainer_step", "step": 1, "value_loss": 0.5}])
    with pytest.raises(KeyError):
        snap.series("trainer_step", "step", "not_a_reduced_key")


def test_game_complete_becomes_columns_and_the_heavy_fields_never_land(series):
    rows = [{"event": "game_complete", "moves": 11, "winner": 0, "terminal_reason": "six_in_a_row",
             "moves_list": ["x"] * 11, "moves_detail": [1], "value_trace": [0.1], "game_id_byte_hash": "h1"},
            {"event": "game_complete", "moves": 256, "winner": -1, "terminal_reason": "ply_cap",
             "game_id_byte_hash": "h1"},
            {"event": "game_complete", "moves": 9, "winner": None, "terminal_reason": "unknown"}]
    snap = _feed(series, rows)
    g = snap.games
    assert g.count == 3 and list(g.plies) == [11.0, 256.0, 9.0]
    assert list(g.winner) == [0, -1, -2] and list(g.cap) == [0, 1, 0]
    assert g.hashes == ["h1", "h1"]
    assert snap.dropped_fields == {"game_complete": 3}
    assert snap.rows("game_complete") == []


def test_the_windowed_cap_rate_matches_the_dashboards_arithmetic(series, dashboard):
    health = importlib.import_module("dashboard.health")
    flags = [0, 1, 1, 0, 0, 1, 0]
    rows = [{"event": "game_complete", "moves": 5, "winner": 0,
             "terminal_reason": "ply_cap" if f else "six_in_a_row"} for f in flags]
    snap = _feed(series, rows)
    assert snap.games.cap_windowed(3) == health.ply_cap_windowed(
        [{"terminal_reason": "ply_cap" if f else "six_in_a_row"} for f in flags], 3)
    assert snap.games.cap_windowed(10) == []


def test_round_level_events_are_kept_whole_and_the_rest_are_counted_and_dropped(series):
    snap = _feed(series, [{"event": "eval_round_complete", "round_id": "r000001_1000", "step": 1000, "ts": 1.0},
                          {"event": "system_stats", "cpu": 1.0}, {"event": "actor_sync", "step": 3}])
    assert snap.rows("eval_round_complete")[0]["round_id"] == "r000001_1000"
    assert snap.rows("system_stats") == [] and snap.counts["system_stats"] == 1
    assert snap.counts["actor_sync"] == 1 and snap.steps_max == 1000


def test_iteration_complete_reduces_the_rates_the_fill_and_the_alpha_block(series):
    snap = _feed(series, [{"event": "iteration_complete", "step": 5, "games_per_hour": 100.0,
                           "positions_per_hour": None, "steps_per_hour": 90.0, "sims_per_sec": 1.5,
                           "buffer_size": 50, "buffer_capacity": 200,
                           "gumbel_alpha_full": {"rows": 1, "graph_rows": 500, "per_1000": 2.0}},
                          {"event": "iteration_complete", "step": 6, "games_per_hour": 110.0,
                           "buffer_size": 60, "buffer_capacity": 0, "gumbel_alpha_full": None}])
    assert snap.series("iteration_complete", "step", "games_per_hour").pairs() == [(5.0, 100.0), (6.0, 110.0)]
    assert snap.series("iteration_complete", "step", "positions_per_hour").pairs() == []
    assert snap.series("iteration_complete", "step", "buffer_fill").pairs() == [(5.0, 0.25)]
    assert snap.series("iteration_complete", "step", "alpha_per_1000").pairs() == [(5.0, 2.0)]
    last = snap.last("iteration_complete")
    assert last is not None and last["step"] == 6


def test_the_memory_shares_follow_the_dashboards_microbatch_rule(series):
    snap = _feed(series, [{"event": "trainer_step", "step": 1, "edges": 200, "caps_max_edges": 400,
                           "nodes": 50, "caps_max_nodes": 100, "microbatches": 2},
                          {"event": "trainer_step", "step": 2, "edges": 200, "caps_max_edges": 0,
                           "nodes": 50, "caps_max_nodes": 100}])
    assert snap.series("trainer_step", "step", "edges_share").pairs() == [(1.0, 0.25)]
    assert snap.series("trainer_step", "step", "nodes_share").pairs() == [(1.0, 0.25)]


def test_wall_hours_and_a_snapshot_is_frozen(series):
    snap = _feed(series, [{"event": "run_boot_identity", "ts": 0.0},
                          {"event": "disk_free", "ts": 7200.0, "disk_free_gb": 80.5}])
    assert snap.wall_hours() == 2.0
    assert snap.series("disk_free", "ts", "disk_free_gb").pairs() == [(7200.0, 80.5)]
    with pytest.raises((AttributeError, TypeError)):
        snap.counts = {}  # type: ignore[misc]
