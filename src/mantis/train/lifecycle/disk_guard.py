"""Disk-space guard: emits ``disk_free``, WARNs below ``warn_gb``, SIGTERMs self below
``fail_gb`` (→ the signal handler's save-then-exit).

The thresholds are a SAFETY guard, independent of the ``keep_all`` pruning knob. The critical
arm is LATCHED and publishes ``critical_fired``, which the composition root turns into the
registered abort rule and rc 47; before that, a run the guard killed exited 0 and the supervisor
relaunched into the same full volume, while the unlatched arm supplied the two-press force-exit
itself and killed its own save.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import threading
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

# Decimal GB: the WARN/FAIL thresholds are calibrated against this exact divisor.
_GB = 1_000_000_000


class DiskGuard:
    """Background thread monitoring disk free space: emits ``disk_free`` every ``interval_sec``,
    WARNs below ``warn_gb`` and SIGTERMs self below ``fail_gb``, so the lifecycle signal handler
    saves the buffer before exit. ``keep_all`` does NOT disable the thresholds.
    """

    def __init__(
        self,
        *,
        watch_path: str | Path,
        interval_sec: float,
        warn_gb: float,
        fail_gb: float,
        keep_all: bool,
        sink: Any,
    ) -> None:
        # NO PARAMETER DEFAULTS: these four are minted config keys read by ONE resolver, and a
        # default here would migrate that authority back into this signature.
        self._path = Path(watch_path)
        self._interval = interval_sec
        self._warn_gb = warn_gb
        self._fail_gb = fail_gb
        self.keep_all = keep_all
        self._sink = sink
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # A FACT this guard produced, not a config proxy. Written only in `check_once` and read
        # by the composition root after `stop()` joined that thread, so no cross-thread write to
        # the run's stop state exists here; the two counters below follow the same rule.
        self._critical_fired = False
        self._errors_total = 0
        self._checks_total = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="disk-guard")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def critical_fired(self) -> bool:
        """True iff this guard's critical arm has signalled the process.

        The guard publishes the FACT, not the abort rule's NAME: the name is a manifest row
        `mantis.train` must not import, so the composition root does the naming.
        """
        return self._critical_fired

    def check_once(self) -> float:
        """Check disk free, emit ``disk_free``, handle thresholds; returns free_gb."""
        usage = shutil.disk_usage(self._path)
        self._checks_total += 1
        free_gb = usage.free / _GB
        self._sink.emit({"event": "disk_free", "disk_free_gb": round(free_gb, 2)})

        if free_gb < self._fail_gb:
            # The SIGTERM is LATCHED; the alert is not. The condition never clears itself, so an
            # unlatched arm supplied the two-press force-exit's SECOND press and exited mid-save.
            first_fire = not self._critical_fired
            self._critical_fired = True
            _LOG.error(
                "disk_critical: free=%.2f GB < fail_threshold=%.2f GB — %s",
                free_gb,
                self._fail_gb,
                "sending SIGTERM to halt training cleanly" if first_fire else
                "SIGTERM already sent; NOT re-signalling (a second press force-exits mid-save)",
            )
            self._sink.emit(
                {"event": "disk_alert", "level": "critical", "disk_free_gb": round(free_gb, 2)}
            )
            if first_fire:
                os.kill(os.getpid(), signal.SIGTERM)
        elif free_gb < self._warn_gb:
            _LOG.warning(
                "disk_low_warn: free=%.2f GB < warn_threshold=%.2f GB",
                free_gb,
                self._warn_gb,
            )
            self._sink.emit(
                {"event": "disk_alert", "level": "warn", "disk_free_gb": round(free_gb, 2)}
            )

        return free_gb

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            try:
                self.check_once()
            except Exception as exc:  # noqa: BLE001 — a monitor thread must not crash the run
                # The counter and the event are the two channels a monitor can read: a bare log
                # warning was silent, and the guard still reported as armed.
                self._errors_total += 1
                _LOG.warning("disk_guard_error: %s", exc)
                self._sink.emit({
                    "event": "disk_guard_error",
                    "error_class": type(exc).__name__,
                    "detail": str(exc)[:300],
                    "errors_total": self._errors_total,
                    "checks_total": self._checks_total,
                    "watch_path": str(self._path),
                })

    @property
    def errors_total(self) -> int:
        """How many ticks of this guard's loop raised: a `check_once` raising every tick emits
        no `disk_free`, so absence-of-alert reads as "plenty of space"."""
        return self._errors_total

    @property
    def interval_sec(self) -> float:
        """This guard's own poll period, exposed so a liveness reader can denominate a stall
        deadline in it rather than holding a second copy of a minted config value."""
        return self._interval

    @property
    def checks_total(self) -> int:
        """How many ticks completed; 0 on a started guard means nothing was measured."""
        return self._checks_total
