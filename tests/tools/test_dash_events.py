"""The dash event tail and reducers: segments in order, a torn tail held, incremental equal to whole, snapshots frozen."""
from __future__ import annotations

import importlib
import json

import pytest

from _dash_record import game_rows, iteration_rows, segment_start, trainer_rows, write_segment


@pytest.fixture(scope="module")
def events(dash):
    return importlib.import_module("dash.readers.events")


def _rows(run_id: str = "r1") -> list[dict]:
    return [segment_start(run_id, 1), *trainer_rows(range(1, 60)), *iteration_rows(range(1, 60, 3)),
            *game_rows(40), {"event": "training_alert", "rule": "grad_norm_spike", "step": 30, "ts": 1030.0}]


def _state(snap) -> tuple:
    keys = [("trainer_step", "step", "policy_entropy"), ("iteration_complete", "step", "games_per_hour")]
    return (tuple(tuple(snap.series(*k).pairs()) for k in keys), tuple(snap.games.plies), tuple(snap.games.step),
            snap.counts.get("game_complete"), len(snap.rows("training_alert")), snap.steps_max)


def test_an_incremental_read_equals_one_whole_read(events, tmp_path):
    rows = _rows()
    whole = tmp_path / "whole" / "logs"
    write_segment(whole, "r1", 1, rows)
    expected = _state(events.EventTail(whole, "r1").poll())
    grown = tmp_path / "grown" / "logs"
    path = write_segment(grown, "r1", 1, rows[:37])
    tail = events.EventTail(grown, "r1")
    tail.poll()
    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows[37:]))
    assert _state(tail.poll()) == expected


def test_a_torn_tail_is_held_until_its_newline_arrives(events, tmp_path):
    logs = tmp_path / "logs"
    path = write_segment(logs, "r1", 1, trainer_rows(range(1, 4)))
    torn = json.dumps(trainer_rows(range(4, 5))[0])
    with path.open("a", encoding="utf-8") as fh:
        fh.write(torn[:20])
    tail = events.EventTail(logs, "r1")
    assert len(tail.poll().series("trainer_step", "step", "value_loss")) == 3
    with path.open("a", encoding="utf-8") as fh:
        fh.write(torn[20:] + "\n")
    assert len(tail.poll().series("trainer_step", "step", "value_loss")) == 4
    assert tail.unparseable == 0


def test_every_segment_is_read_in_number_order_and_the_newest_is_the_live_one(events, tmp_path):
    logs = tmp_path / "logs"
    write_segment(logs, "r1", 10, [segment_start("r1", 10, pid=30), *trainer_rows(range(200, 202))])
    write_segment(logs, "r1", 9, [segment_start("r1", 9, pid=20), *trainer_rows(range(100, 102))])
    write_segment(logs, "r1", 1, [segment_start("r1", 1, pid=10), *trainer_rows(range(1, 3))])
    snap = events.EventTail(logs, "r1").poll()
    assert [s.number for s in snap.segments] == [1, 9, 10]
    assert snap.series("trainer_step", "step", "value_loss").pairs()[0][0] == 1.0
    assert snap.live_segment.number == 10 and snap.live_segment.started["pid"] == 30


def test_a_new_segment_is_a_resume_and_is_appended(events, tmp_path):
    logs = tmp_path / "logs"
    write_segment(logs, "r1", 1, [segment_start("r1", 1), *trainer_rows(range(1, 5))])
    tail = events.EventTail(logs, "r1")
    assert tail.poll().live_segment.number == 1
    write_segment(logs, "r1", 2, [segment_start("r1", 2, pid=99), *trainer_rows(range(5, 7))])
    snap = tail.poll()
    assert snap.live_segment.number == 2 and snap.live_segment.rows == 3
    assert len(snap.series("trainer_step", "step", "value_loss")) == 6


def test_a_failed_boot_segment_without_a_start_row_still_counts_its_rows(events, tmp_path):
    logs = tmp_path / "logs"
    write_segment(logs, "r1", 1, trainer_rows(range(1, 3)))
    snap = events.EventTail(logs, "r1").poll()
    assert snap.live_segment.started is None and snap.live_segment.rows == 2


def test_a_record_with_no_parseable_row_is_refused(events, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "events_r1_seg0001.jsonl").write_text("not json\n[1, 2]\n", encoding="utf-8")
    tail = events.EventTail(logs, "r1")
    with pytest.raises(events.EmptyRunRecord, match="2 unparseable"):
        tail.poll()


def test_another_runs_segments_are_never_read(events, tmp_path):
    logs = tmp_path / "logs"
    write_segment(logs, "r1", 1, trainer_rows(range(1, 3)))
    write_segment(logs, "other", 1, trainer_rows(range(1, 9)))
    assert len(events.EventTail(logs, "r1").poll().series("trainer_step", "step", "lr")) == 2
    assert events.record_run_ids(logs) == ["other", "r1"]


def test_a_game_carries_the_newest_step_seen_before_it_and_minus_one_before_any(events, tmp_path):
    logs = tmp_path / "logs"
    write_segment(logs, "r1", 1, [*game_rows(1), *trainer_rows(range(5, 8)), *game_rows(2)])
    games = events.EventTail(logs, "r1").poll().games
    assert list(games.step) == [-1, 7, 7]


def test_heavy_game_fields_are_dropped_and_counted(events, tmp_path):
    logs = tmp_path / "logs"
    write_segment(logs, "r1", 1, game_rows(5))
    snap = events.EventTail(logs, "r1").poll()
    assert snap.dropped_fields["game_complete"] == 5 and snap.games.count == 5


def test_a_snapshot_never_shows_a_row_fed_after_it(events, tmp_path):
    logs = tmp_path / "logs"
    path = write_segment(logs, "r1", 1, [*trainer_rows(range(1, 4)), *game_rows(3)])
    tail = events.EventTail(logs, "r1")
    first = tail.poll()
    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in [*trainer_rows(range(4, 9)), *game_rows(4)]))
    tail.poll()
    assert len(first.series("trainer_step", "step", "lr")) == 3 and first.games.count == 3


def test_a_series_the_reducers_do_not_keep_is_a_key_error_never_an_empty_series(events, tmp_path):
    logs = tmp_path / "logs"
    write_segment(logs, "r1", 1, trainer_rows(range(1, 3)))
    with pytest.raises(KeyError):
        events.EventTail(logs, "r1").poll().series("trainer_step", "step", "no_such_key")


def test_a_record_without_policy_entropy_keeps_that_series_empty_not_zero(events, tmp_path):
    logs = tmp_path / "logs"
    write_segment(logs, "r1", 1, trainer_rows(range(1, 5), entropy=False))
    snap = events.EventTail(logs, "r1").poll()
    assert len(snap.series("trainer_step", "step", "policy_entropy")) == 0
    assert snap.series("trainer_step", "step", "policy_entropy").last() is None
