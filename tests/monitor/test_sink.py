"""O-24 / P-24 (+ sink emit/close/counter/ts contract) — the REAL JsonlEventSink.

RED-at-import until IMPL writes `mantis.monitor.sink`. Asserts the §c.1 public API exactly:
`JsonlEventSink(*, log_dir, run_id)`, `.emit`, `.close`, `.path`, `.persist_errors_total`.

Covers:
  * emit stamps a `ts` iff absent, and PRESERVES a producer-supplied `ts` (behaviour parity
    with the old `emit_event` `{"ts": time.time(), **payload}` later-key-wins funnel);
  * a missing `"event"` key is a loud ValueError (producer bug, not a persistence failure);
  * a write/serialize failure ⇒ `persist_errors_total += 1`, NO raise (LAW-14 — the
    watchdog delivers run-fatality by observing the counter, from ANY thread);
  * O-24: N writer threads × M events → exactly N×M (+1 header) well-formed JSON lines,
    no torn/interleaved lines (single `write` under the instance lock, line-buffered).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from mantis.monitor.sink import JsonlEventSink


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_first_line_is_run_segment_started(tmp_path: Path) -> None:
    """Every segment opens with a `run_segment_started` header naming the run + segment."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="runa")
    sink.close()
    lines = _read_lines(sink.path)
    assert lines, "a fresh segment must carry its run_segment_started header line"
    assert lines[0]["event"] == "run_segment_started"
    assert lines[0]["run_id"] == "runa"


def test_emit_appends_a_json_line(tmp_path: Path) -> None:
    sink = JsonlEventSink(log_dir=tmp_path, run_id="runa")
    sink.emit({"event": "training_step", "step": 7})
    sink.close()
    events = [ln for ln in _read_lines(sink.path) if ln["event"] == "training_step"]
    assert len(events) == 1
    assert events[0]["step"] == 7


def test_emit_stamps_ts_when_absent(tmp_path: Path) -> None:
    """A payload without `ts` gets a wall stamp added by the sink."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="runa")
    sink.emit({"event": "e"})
    sink.close()
    ev = [ln for ln in _read_lines(sink.path) if ln["event"] == "e"][0]
    assert "ts" in ev and isinstance(ev["ts"], (int, float))


def test_emit_preserves_producer_supplied_ts(tmp_path: Path) -> None:
    """A producer-supplied `ts` WINS (parity with the old later-key-wins funnel) — the sink
    stamp never clobbers an explicit producer timestamp."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="runa")
    sink.emit({"event": "e", "ts": 123.5})
    sink.close()
    ev = [ln for ln in _read_lines(sink.path) if ln["event"] == "e"][0]
    assert ev["ts"] == 123.5


def test_emit_missing_event_key_raises_valueerror(tmp_path: Path) -> None:
    """A missing `event` key is a producer BUG → loud ValueError (NOT a persist failure, so
    the counter must NOT move)."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="runa")
    with pytest.raises(ValueError):
        sink.emit({"step": 1})
    assert sink.persist_errors_total == 0


def test_write_failure_counts_and_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    """LAW-14 — a serialize/IO failure ⇒ persist_errors_total += 1, and emit does NOT raise
    (emits run on daemon threads where a raise would only kill the feeder; the counter makes
    the failure fatal from any thread, via the watchdog)."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="runa")
    before = sink.persist_errors_total

    def _boom(*_a, **_k):
        raise OSError("simulated disk write failure")

    # Force the underlying line write to fail (implementation-agnostic: patch json.dumps in
    # the sink module so serialization raises for the next emit).
    import mantis.monitor.sink as sink_mod

    monkeypatch.setattr(sink_mod, "json", type("J", (), {"dumps": staticmethod(_boom)}))
    sink.emit({"event": "e", "step": 1})  # must NOT raise
    assert sink.persist_errors_total == before + 1


def test_persist_errors_total_starts_at_zero(tmp_path: Path) -> None:
    sink = JsonlEventSink(log_dir=tmp_path, run_id="runa")
    assert sink.persist_errors_total == 0
    sink.emit({"event": "ok"})
    assert sink.persist_errors_total == 0
    sink.close()


def test_concurrent_writers_produce_well_formed_non_interleaved_lines(tmp_path: Path) -> None:
    """O-24 / P-24 — 8 threads × 250 events → exactly 2000 payload lines (+1 header), every
    line a well-formed JSON object, zero torn/interleaved lines, persist_errors_total == 0.
    Bites a torn/interleaved JSONL from an unlocked multi-write."""
    sink = JsonlEventSink(log_dir=tmp_path, run_id="runa")
    n_threads, per_thread = 8, 250

    def _worker(wid: int) -> None:
        for i in range(per_thread):
            sink.emit({"event": "training_step", "worker": wid, "i": i})

    threads = [threading.Thread(target=_worker, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert all(not t.is_alive() for t in threads), "writer threads must finish (no deadlock)"
    sink.close()

    lines = _read_lines(sink.path)  # every line parses ⇒ no torn/interleaved line
    payload = [ln for ln in lines if ln["event"] == "training_step"]
    assert len(payload) == n_threads * per_thread
    assert sink.persist_errors_total == 0
