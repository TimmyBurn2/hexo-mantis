"""Disk-space guard (repo_design §11; WP10 §a.2).

Emits ``disk_free`` events, WARNs below ``warn_gb``, and SIGTERMs self below ``fail_gb``
(→ the signal handler's save-then-exit). Relocated out of the old `monitoring/disk_guard.py`
into the lifecycle subsystem; the ``emit_event`` call is re-pointed at the injected
``EventSink`` (repo_design §11 seam). Thresholds are a SAFETY guard, independent of
``keep_all`` (a pruning knob) — verbatim invariant.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import threading
from pathlib import Path
from typing import Any, Optional

_LOG = logging.getLogger(__name__)

# Decimal GB (matching the old `monitoring/disk_guard.py:59` `usage.free / 1e9`) — the
# run-safety WARN/FAIL thresholds are calibrated against this exact divisor; behaviour-exact.
_GB = 1_000_000_000


class DiskGuard:
    """Background thread monitoring disk free space.

    Emits ``disk_free`` every ``interval_sec`` seconds. WARNs (``disk_alert`` level=warn)
    when free < ``warn_gb``; SIGTERMs self (``disk_alert`` level=critical) when free <
    ``fail_gb`` — SIGTERM triggers the lifecycle signal handler so the buffer is saved
    before exit. ``keep_all`` is carried for the caller's pruning policy; it does NOT
    disable the disk thresholds.
    """

    def __init__(
        self,
        *,
        watch_path: str | Path = ".",
        interval_sec: float = 60.0,
        warn_gb: float = 10.0,
        fail_gb: float = 5.0,
        keep_all: bool = False,
        sink: Any,
    ) -> None:
        self._path = Path(watch_path)
        self._interval = interval_sec
        self._warn_gb = warn_gb
        self._fail_gb = fail_gb
        self.keep_all = keep_all
        self._sink = sink
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="disk-guard")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def check_once(self) -> float:
        """Check disk free, emit ``disk_free``, handle thresholds. Returns free_gb (decimal
        GB, matching old ``/1e9``)."""
        usage = shutil.disk_usage(self._path)
        free_gb = usage.free / _GB
        self._sink.emit({"event": "disk_free", "disk_free_gb": round(free_gb, 2)})

        if free_gb < self._fail_gb:
            _LOG.error(
                "disk_critical: free=%.2f GB < fail_threshold=%.2f GB — sending SIGTERM "
                "to halt training cleanly",
                free_gb,
                self._fail_gb,
            )
            self._sink.emit(
                {"event": "disk_alert", "level": "critical", "disk_free_gb": round(free_gb, 2)}
            )
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
                _LOG.warning("disk_guard_error: %s", exc)
