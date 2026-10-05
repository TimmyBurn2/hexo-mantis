"""The dash run record: parity with tools/dashboard's reader, liveness from the heartbeat, one snapshot over every input."""
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


def _parity(dash_snapshot, old) -> None:
    for key in _KEYS:
        assert dash_snapshot.series(*key).pairs() == old.series(*key), key
    games = old.rows("game_complete")
    assert dash_snapshot.games.count == len(games)
    assert list(dash_snapshot.games.cap) == [1 if g.get("terminal_reason") == "ply_cap" else 0 for g in games]
    assert sum(1 for w in dash_snapshot.games.winner if w == 0) == sum(1 for g in games if g.get("winner") == 0)


def test_the_series_and_game_columns_equal_the_dashboards_on_a_full_shaped_record(dash, reader, tmp_path):
    events = importlib.import_module("dash.readers.events")
    rows = [segment_start("r1", 1), *trainer_rows(range(1, 200)), *iteration_rows(range(1, 200, 4)),
            *game_rows(120, cap_every=9)]
    path = write_segment(tmp_path / "logs", "r1", 1, rows)
    _parity(events.EventTail(tmp_path / "logs", "r1").poll(), reader.load_record(path))


def test_the_mirror_record_matches_the_dashboard_when_it_is_present(dash, reader):
    named = os.environ.get("MANTIS_DASH_FIXTURE_EVENTS")
    if not named:
        pytest.skip("MANTIS_DASH_FIXTURE_EVENTS names no event segment")
    events = importlib.import_module("dash.readers.events")
    path = Path(named)
    run_id = events.SEGMENT_RE.match(path.name).group("run")
    snap = events.EventTail(path.parent, run_id).poll()
    if len(snap.segments) != 1:
        pytest.skip("the named record has several segments; the dashboard reads one file")
    _parity(snap, reader.load_record(path))


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
    assert snap.six.parent.wr == 0.59 and [c.step for c in snap.six.line] == [30] and not snap.strix.line
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
