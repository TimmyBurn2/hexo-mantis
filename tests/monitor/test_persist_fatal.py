"""⊕ O-07 / P-07 — persist-fatal: a killed sink write ⇒ counter++ ⇒ the INDEPENDENT
watchdog observes the counter and aborts with code 43 (LAW-14; pays the WP6 debt).

RED-at-import until IMPL writes `mantis.monitor.sink` AND
`mantis.train.lifecycle.heartbeat_watchdog`. ORACLE-FIRST (⊕): both top-level imports raise
ModuleNotFoundError before any port code exists. Torch-free: `heartbeat_watchdog` imports
`mantis.monitor.heartbeat` only.

PASS bars (PREREG P-07): one failed emit ⇒ `persist_errors_total` +1 exactly; the watchdog
fires on the next poll after observation; exit code 43; a `.watchdog` snapshot is attempted
(best_effort — counted if it fails). A swallowed persistence failure is the exact bug this
bites. Deterministic via `poll_once()` + a fake clock (the IMPL-pinned single-poll seam,
see ORACLE_NOTES).
"""
from __future__ import annotations

from pathlib import Path

from mantis.monitor.heartbeat import HEARTBEAT_SOURCES, HeartbeatRegistry
from mantis.monitor.sink import JsonlEventSink
from mantis.train.lifecycle.heartbeat_watchdog import HeartbeatWatchdog


def _make_watchdog(*, counters_fn, sink, clock, exit_spy, snapshot_spy, hb_file: Path):
    reg = HeartbeatRegistry(clock=clock)
    return HeartbeatWatchdog(
        registry=reg,
        deadlines={s: 1800.0 for s in HEARTBEAT_SOURCES},
        sink=sink,
        counters_fn=counters_fn,
        heartbeat_file=hb_file,
        file_interval_sec=0.0,
        poll_interval_sec=0.1,
        clock=clock,
        save_snapshot=snapshot_spy,
        exit_fn=exit_spy,
    )


def test_sink_failure_counts_and_aborts(tmp_path: Path, spy_sink, fake_clock,
                                        exit_spy, snapshot_spy, monkeypatch) -> None:
    """O-07 / P-07 — the manifest `persist_fatal` producer test. A monkeypatched sink write
    failure mid-run bumps `persist_errors_total` exactly once; the watchdog's `counters_fn`
    observes it and fires exit 43 with a best-effort snapshot. NOT a silent except: pass."""
    real_sink = JsonlEventSink(log_dir=tmp_path, run_id="runa")
    before = real_sink.persist_errors_total

    # The watchdog reads the LIVE sink counter every poll.
    wd = _make_watchdog(
        counters_fn=lambda: real_sink.persist_errors_total,
        sink=spy_sink, clock=fake_clock, exit_spy=exit_spy, snapshot_spy=snapshot_spy,
        hb_file=tmp_path / "hb.json",
    )
    wd.poll_once()                       # counter still 0 → no fire
    assert not exit_spy.fired

    # Kill the underlying serialization so the next emit fails (LAW-14: count, do not raise).
    import mantis.monitor.sink as sink_mod

    def _boom(*_a, **_k):
        raise OSError("disk write failure")

    monkeypatch.setattr(sink_mod, "json", type("J", (), {"dumps": staticmethod(_boom)}))
    real_sink.emit({"event": "training_step", "step": 1})   # counted, not raised
    assert real_sink.persist_errors_total == before + 1     # +1 EXACTLY

    wd.poll_once()                       # counter > 0 → fire on this poll
    assert exit_spy.codes and exit_spy.codes[0] == 43, "persist-fatal must exit 43"
    assert snapshot_spy.count >= 1, "the fire must attempt a best-effort snapshot"
    fired = spy_sink.named("heartbeat_watchdog_fired")
    assert fired, "the watchdog must emit a loud heartbeat_watchdog_fired event"
    assert "persist" in str(fired[-1].get("reason", "")), (
        f"the fire reason must name persist_fatal, got {fired[-1]}"
    )


def test_zero_counter_never_fires_persist(tmp_path: Path, spy_sink, fake_clock,
                                          exit_spy, snapshot_spy) -> None:
    """O-07 — with the persist counter held at 0, repeated polls never fire 43 (no phantom
    persist-fatal from a healthy sink)."""
    wd = _make_watchdog(
        counters_fn=lambda: 0, sink=spy_sink, clock=fake_clock,
        exit_spy=exit_spy, snapshot_spy=snapshot_spy, hb_file=tmp_path / "hb.json",
    )
    for _ in range(5):
        fake_clock.advance(1.0)
        wd.poll_once()
    assert not exit_spy.fired
