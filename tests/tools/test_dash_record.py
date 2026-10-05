"""The dash run record: series equal to the record's own rows, liveness from the heartbeat, one snapshot over every input."""
from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import pytest

from _dash_record import (
    game,
    game_rows,
    iteration_rows,
    segment_start,
    sidecar,
    six_in_a_row_for_p1,
    trainer_rows,
    write_config,
    write_heartbeat,
    write_segment,
    write_shard,
)

_KEYS = [("trainer_step", "step", k) for k in ("value_loss", "policy_loss", "loss", "grad_norm", "lr")] + [
    ("iteration_complete", "step", k) for k in ("games_per_hour", "positions_per_hour", "steps_per_hour")]


@pytest.fixture(scope="module")
def record(dash):
    return importlib.import_module("dash.readers.record")


@pytest.fixture(scope="module")
def liveness(dash):
    return importlib.import_module("dash.readers.liveness")


def _expected(rows: list[dict], event: str, key: str) -> list[tuple[float, float]]:
    return [(float(r["step"]), float(r[key])) for r in rows if r.get("event") == event and key in r]


def test_the_series_and_game_columns_equal_the_records_own_rows(dash, tmp_path):
    events = importlib.import_module("dash.readers.events")
    rows = [segment_start("r1", 1), *trainer_rows(range(1, 200)), *iteration_rows(range(1, 200, 4)),
            *game_rows(120, cap_every=9)]
    write_segment(tmp_path / "logs", "r1", 1, rows)
    snap = events.EventTail(tmp_path / "logs", "r1").poll()
    for event, x, key in _KEYS:
        assert snap.series(event, x, key).pairs() == _expected(rows, event, key), key
    games = [r for r in rows if r["event"] == "game_complete"]
    assert snap.games.count == len(games)
    assert list(snap.games.cap) == [1 if g["terminal_reason"] == "ply_cap" else 0 for g in games]
    assert list(snap.games.winner) == [g["winner"] for g in games]


@pytest.mark.parametrize(("age", "state"), [(30.0, "live"), (1200.0, "stale"), (7200.0, "stopped")])
def test_the_heartbeats_age_says_live_stale_or_stopped(liveness, tmp_path, age, state):
    now = time.time()
    path = write_heartbeat(tmp_path, "r1", now - age)
    os.utime(path, (now - 99999, now - 99999))
    assert liveness.read(tmp_path, "r1").state(now) == state


def test_a_newer_mtime_wins_over_an_older_wall_ts_and_no_file_is_unknown(liveness, tmp_path):
    now = time.time()
    write_heartbeat(tmp_path, "r1", now - 99999)
    beat = liveness.read(tmp_path, "r1")
    assert beat.state(now) == "live" and "mtime" in beat.source
    assert liveness.read(tmp_path, "absent").state(now) == "unknown"


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    logs = run / "logs"
    write_segment(logs, "r1", 1, [segment_start("r1", 1, pid=1), *trainer_rows(range(1, 50))])
    write_segment(logs, "r1", 2, [segment_start("r1", 2, pid=2), *trainer_rows(range(50, 60))])
    write_config(run, "r1", "p0_00045000_abcd1234")
    write_heartbeat(logs, "r1", time.time())
    stats = [{"ply": 0, "root_value": 0.3, "visits": []}, {"ply": 1, "root_value": -0.4, "visits": []}]
    write_shard(logs / "games", "r1", 2, "2026100510", [game("a", six_in_a_row_for_p1(), stats=stats)])
    return run


def test_one_poll_reads_events_games_records_and_cells_into_one_snapshot(record, tmp_path):
    run = _run(tmp_path)
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 30, 0.70)
    sidecar(cells, "p0", 45000, 0.59)
    snap = record.RunRecord("label", run, tmp_path / "records", (cells,)).poll()
    assert snap.run_id == "r1" and snap.label == "label" and snap.events.live_segment.started["pid"] == 2
    assert snap.games_indexed == 1 and snap.horizon.games == 1
    (ruler,) = snap.rulers
    assert ruler.parent.wr == 0.59 and [c.step for c in ruler.line] == [30] and ruler.family == "six"
    assert snap.records.saves == () and snap.beat.state(time.time()) == "live"


def test_a_run_dir_with_no_segment_is_refused(record, tmp_path):
    events = importlib.import_module("dash.readers.events")
    (tmp_path / "logs").mkdir()
    with pytest.raises(events.EmptyRunRecord):
        record.RunRecord("x", tmp_path)


def test_several_runs_in_one_logs_dir_are_resolved_by_the_configs_run_id(record, tmp_path):
    run = _run(tmp_path)
    write_segment(run / "logs", "zz", 1, trainer_rows(range(1, 3)))
    assert record.RunRecord("x", run).run_id == "r1"
    write_config(run, "neither", None)
    events = importlib.import_module("dash.readers.events")
    with pytest.raises(events.EmptyRunRecord, match="names none"):
        record.RunRecord("x", run)


def test_the_freeze_reads_the_writers_clock_never_the_files_mtime(dash, tmp_path):
    view = importlib.import_module("dash.views.run")
    record = importlib.import_module("dash.readers.record")
    run = _run(tmp_path)
    beat = run / "logs" / "heartbeat_r1.json"
    os.utime(beat, (1.0e9, 1.0e9))
    first = view.status(record.RunRecord("x", run).poll(), None)
    os.utime(beat, (1.5e9, 1.5e9))
    assert view.status(record.RunRecord("x", run).poll(), None) == first


def test_a_position_further_out_than_the_chart_is_not_counted(dash):
    horizon = importlib.import_module("dash.readers.horizon")
    red = horizon.HorizonReducer()
    moves = [[i, 0] for i in range(2 * (horizon.K_MAX + 3))]
    red(game("long", moves, result="p1", stats=[{"ply": 0, "root_value": 0.5}, {"ply": len(moves) - 1, "root_value": 0.5}]))
    rows = list(red._games[0])  # noqa: SLF001
    assert len(rows) == 3 and rows[0] == 0
