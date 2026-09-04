"""AUDIT-1 F-11 — a disk guard whose every tick raises is no longer indistinguishable from a
healthy one.

THE DEFECT, and it is three facts reinforcing each other. `DiskGuard._loop` wrapped
`check_once` in `except Exception → _LOG.warning("disk_guard_error…")` and continued. A
`check_once` that raises on every tick therefore:

* emitted no `disk_free` and no `disk_alert` — and an ABSENCE of disk alerts reads as "plenty
  of space", which is exactly the state the guard exists to deny;
* incremented no counter and never set `_critical_fired`;
* left the guard outside `HEARTBEAT_SOURCES`, so the stall watchdog had nothing to notice;
* and the warning itself was invisible, because `mantis.run` installs no logging handler at
  all (F-08), so Python's lastResort handler prints WARNING+ unformatted or not at all.

Meanwhile gate 12's REQUIRED `disk_space_exhausted` row audits ARMED off `monitor.disk_guard.
fail_gb > 0` — a CONFIG NUMBER — so every instrument reported the guard armed. The rc-47
abort R132 closed could be dead for a whole run while the volume filled and the supervisor
relaunched into it.

WHAT IS AND IS NOT CLOSED HERE. The event and the counters are: a monitor reading the ONE
channel can now tell a failing guard from a quiet one. The ARMING mechanism is not — gate 12
still reads a config number for this row, and changing that is a mint-affecting decision
about `armed_aborts.MANIFEST`, not a repair. Registering the guard as a heartbeat source would
need a new collaborator threaded through the composition root and is likewise not this.

THE ARMING HALF IS NOW CLOSED, by R334(b), in
`tests/train/test_monitor_liveness_arming.py` — the paragraph above is kept as the record of
what this file's scope WAS. The row's mechanism carries a producer-liveness operand and gate 12
stays byte-untouched; the heartbeat-source route was REJECTED on the mechanism this paragraph
only guessed at, namely that the stall code is 42, the class the supervisor relaunches on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mantis.train.lifecycle.disk_guard import DiskGuard


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


def _guard(tmp_path: Path, sink: _Sink, *, interval: float = 0.01) -> DiskGuard:
    return DiskGuard(watch_path=tmp_path, interval_sec=interval, warn_gb=0.0, fail_gb=0.0,
                     keep_all=True, sink=sink)


def test_a_fresh_guard_has_measured_nothing_and_says_so(tmp_path: Path) -> None:
    guard = _guard(tmp_path, _Sink())
    assert guard.checks_total == 0
    assert guard.errors_total == 0


def test_a_healthy_check_counts_itself(tmp_path: Path) -> None:
    """The control. `checks_total` is the denominator every other reading is taken over."""
    sink = _Sink()
    guard = _guard(tmp_path, sink)
    guard.check_once()
    guard.check_once()
    assert guard.checks_total == 2 and guard.errors_total == 0
    assert len(sink.named("disk_free")) == 2
    assert not sink.named("disk_guard_error")


def test_a_tick_that_RAISES_emits_a_named_event_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE PIN. Before this, the only trace was a `logger.warning` into a run with no handler."""
    import mantis.train.lifecycle.disk_guard as dg

    sink = _Sink()
    guard = _guard(tmp_path, sink)

    def _boom(_path: Any) -> Any:
        raise OSError("watch_path vanished")

    monkeypatch.setattr(dg.shutil, "disk_usage", _boom)
    guard.start()
    try:
        _wait_for(lambda: guard.errors_total >= 2)
    finally:
        guard.stop()

    errors = sink.named("disk_guard_error")
    assert errors, "a guard failing on every tick emitted nothing into the ONE channel"
    last = errors[-1]
    assert last["error_class"] == "OSError"
    assert "watch_path vanished" in last["detail"]
    assert last["errors_total"] >= 2
    assert last["checks_total"] == 0, (
        "no check completed, and the payload must say so — a reader cannot otherwise tell "
        "'the guard measured plenty of space' from 'the guard measured nothing'"
    )
    assert not sink.named("disk_free"), "a failing guard must not appear to be reporting"
    assert guard.critical_fired is False


def test_the_guard_keeps_running_after_a_failed_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the original design, preserved: a monitor thread must not crash the
    run. The repair makes the failure LOUD, not fatal."""
    import mantis.train.lifecycle.disk_guard as dg

    calls = {"n": 0}
    real = dg.shutil.disk_usage

    def _flaky(path: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return real(path)

    sink = _Sink()
    guard = _guard(tmp_path, sink)
    monkeypatch.setattr(dg.shutil, "disk_usage", _flaky)
    guard.start()
    try:
        _wait_for(lambda: guard.checks_total >= 1)
    finally:
        guard.stop()

    assert guard.errors_total == 1 and guard.checks_total >= 1
    assert sink.named("disk_guard_error") and sink.named("disk_free")


def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached within the timeout")
