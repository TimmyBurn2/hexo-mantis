"""The tail: incremental equals whole, a torn tail is held, a new segment is appended, nothing refuses twice."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def events(observatory):
    return importlib.import_module("observatory.readers.events")


def _write(logs: Path, rows: list[dict], name: str = "events_t_seg0001.jsonl", mode: str = "w") -> Path:
    logs.mkdir(exist_ok=True)
    path = logs / name
    with path.open(mode, encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _rows(n: int, start: int = 1) -> list[dict]:
    return [{"event": "trainer_step", "step": s, "value_loss": s / 10, "ts": float(s)}
            for s in range(start, start + n)]


def test_incremental_polls_equal_one_whole_read(events, tmp_path):
    logs = tmp_path / "logs"
    _write(logs, _rows(5))
    tail = events.EventTail(logs, "t")
    first = tail.poll()
    assert first.series("trainer_step", "step", "value_loss").pairs() == [(float(s), s / 10) for s in range(1, 6)]
    _write(logs, _rows(3, start=6), mode="a")
    second = tail.poll()
    whole = events.EventTail(logs, "t").poll()
    assert (second.series("trainer_step", "step", "value_loss").pairs()
            == whole.series("trainer_step", "step", "value_loss").pairs())
    assert second.counts == whole.counts and second.steps_max == 8


def test_a_torn_tail_line_is_held_not_skipped_and_completes_on_the_next_poll(events, tmp_path):
    logs = tmp_path / "logs"
    path = _write(logs, _rows(2))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "trainer_step", "step": 3, "val')
    tail = events.EventTail(logs, "t")
    snap = tail.poll()
    assert snap.steps_max == 2 and tail.unparseable == 0 and tail.held[path].startswith(b'{"event"')
    with path.open("a", encoding="utf-8") as fh:
        fh.write('ue_loss": 0.3, "ts": 3.0}\n')
    snap = tail.poll()
    assert snap.steps_max == 3 and tail.held.get(path, b"") == b""


def test_a_complete_non_json_line_is_counted_unparseable(events, tmp_path):
    logs = tmp_path / "logs"
    path = _write(logs, _rows(1))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
    tail = events.EventTail(logs, "t")
    tail.poll()
    assert tail.unparseable == 1


def test_a_new_segment_is_discovered_and_read_after_the_old_one(events, tmp_path):
    logs = tmp_path / "logs"
    _write(logs, [{"event": "run_segment_started", "segment": 1, "ts": 1.0}] + _rows(2))
    tail = events.EventTail(logs, "t")
    tail.poll()
    _write(logs, [{"event": "run_segment_started", "segment": 2, "ts": 5.0}] + _rows(2, start=3),
           name="events_t_seg0002.jsonl")
    snap = tail.poll()
    assert [s.get("segment") for s in snap.rows("run_segment_started")] == [1, 2]
    assert [p.name for p in tail.segments] == ["events_t_seg0001.jsonl", "events_t_seg0002.jsonl"]
    assert snap.steps_max == 4


def test_another_runs_segment_is_ignored(events, tmp_path):
    logs = tmp_path / "logs"
    _write(logs, _rows(1))
    _write(logs, _rows(9), name="events_other_seg0001.jsonl")
    assert events.EventTail(logs, "t").poll().steps_max == 1


def test_a_record_with_no_parseable_row_refuses(events, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "events_t_seg0001.jsonl").write_text("\n\nnot json\n", encoding="utf-8")
    with pytest.raises(events.EmptyRunRecord):
        events.EventTail(logs, "t").poll()
    with pytest.raises(events.EmptyRunRecord):
        events.EventTail(logs, "missing").poll()
