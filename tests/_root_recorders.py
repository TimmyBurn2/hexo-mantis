"""The real-subsystem recording harness the root-composition suites share."""
from __future__ import annotations

from typing import Any

import mantis.run as mantis_run
from mantis.train.lifecycle.disk_guard import DiskGuard

#: The bounded burst every shared drive runs; 3 is the smallest legal run at cadence 1.
DRIVE_STEPS = 3

#: Disk-guard values, deliberately three DISTINCT numbers: an assertion that the guard received
#: the resolver's values is vacuous if two are equal, and the transposition is exactly a swap of
#: two. Low enough that the critical arm can NEVER fire on a real filesystem.
DRIVE_DISK_GUARD = {"interval_sec": 0.02, "warn_gb": 0.001, "fail_gb": 0.0005}


class RecordedDiskGuard(DiskGuard):
    """The REAL guard with two observation points. Every behaviour is `super()`'s."""

    instances: list = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ctor_kwargs = dict(kwargs)
        self.stop_calls = 0
        type(self).instances.append(self)

    def stop(self) -> None:
        self.stop_calls += 1
        super().stop()


class Recorders:
    """What the real subsystems did, observed without standing any of them in."""

    def __init__(self) -> None:
        self.run_safety: Any = None
        self.watchdog_stops = 0
        self.sink_closes = 0
        self.disk_guards: list[RecordedDiskGuard] = []


def install_recorders(monkeypatch, request) -> Recorders:
    """Close the REAL sink after every drive: on the COMPLETED path `close_out` never touches
    it, which is bounded in production because both real callers exit the process right after."""
    rec = Recorders()
    real_build = mantis_run.build_run_safety
    RecordedDiskGuard.instances = rec.disk_guards

    def _recording_build(**kwargs):
        run_safety = real_build(**kwargs)          # the REAL builder, unmodified
        rec.run_safety = run_safety
        real_stop, real_close = run_safety.watchdog.stop, run_safety.sink.close

        def _stop() -> None:
            rec.watchdog_stops += 1
            real_stop()

        def _close() -> None:
            rec.sink_closes += 1
            real_close()

        run_safety.watchdog.stop = _stop
        run_safety.sink.close = _close
        request.addfinalizer(run_safety.sink.close)
        return run_safety

    monkeypatch.setattr(mantis_run, "build_run_safety", _recording_build)
    monkeypatch.setattr(mantis_run, "DiskGuard", RecordedDiskGuard)
    return rec


def bounded(smoke_run_config, **over):
    """A REAL minted graph config, bounded so the drive terminates; `eval_enabled` stays the
    CONFIG's own value, no parameter being able to force it."""
    monitor = {"actor_lag_threshold_steps": DRIVE_STEPS - 1,
               "disk_guard": dict(DRIVE_DISK_GUARD)}
    monitor.update(over.pop("monitor", {}))
    over.setdefault("eval_enabled", False)
    return smoke_run_config(
        "dev_example.yaml",
        train={"actor_sync_cadence_steps": 1, "max_train_steps": DRIVE_STEPS,
               "batch_size": 8},
        monitor=monitor, **over,
    )
