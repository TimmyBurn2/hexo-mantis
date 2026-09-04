"""Disk-space guard (repo_design §11; WP10 §a.2).

Emits ``disk_free`` events, WARNs below ``warn_gb``, and SIGTERMs self below ``fail_gb``
(→ the signal handler's save-then-exit). Relocated out of the old `monitoring/disk_guard.py`
into the lifecycle subsystem; the ``emit_event`` call is re-pointed at the injected
``EventSink`` (repo_design §11 seam). Thresholds are a SAFETY guard, independent of
``keep_all`` (a pruning knob) — verbatim invariant.

WPMAIN RED-TEAM RT-2 / R132: that SIGTERM used to leave NO trace a supervisor could read.
The handler sets ``shutdown_save``/``running`` and never ``abort_rule``, so a run the guard
killed exited **0** and the supervisor above relaunched into the same full volume. The arm is
now LATCHED (it fired every ``interval_sec`` and supplied the two-press force-exit itself,
killing its own save) and it publishes ``critical_fired``, which the composition root turns
into the registered abort rule -> process rc 47. Delivery is unchanged: still a SIGTERM, still
save-then-exit, because that is what the guard exists for.
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
        watch_path: str | Path,
        interval_sec: float,
        warn_gb: float,
        fail_gb: float,
        keep_all: bool,
        sink: Any,
    ) -> None:
        # NO PARAMETER DEFAULTS (MF-2, WPMAIN). The four values below used to default to
        # `"."` / 60 / 10 / 5 here AND to be re-defaulted by `build_subsystems`' `.get(...)`
        # over a key no schema carried — two authorities for numbers no operator could see,
        # in a guard nothing constructed. They are minted config keys now
        # (`monitor.disk_guard.*`, R122), read by ONE resolver; a parameter default here
        # would MIGRATE the authority back into this signature, leaving every field census
        # green while a caller that omits an argument silently inherits a posture.
        self._path = Path(watch_path)
        self._interval = interval_sec
        self._warn_gb = warn_gb
        self._fail_gb = fail_gb
        self.keep_all = keep_all
        self._sink = sink
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # RT-2b's latch, and RT-2's carrier. NOT a config proxy (R79): it is a FACT this guard
        # produced — "my critical arm signalled this process" — with no config value beside it.
        # Written only in `check_once`, i.e. only on this object's own thread (or on the
        # caller's, for a direct `check_once()`), and read by the composition root AFTER
        # `stop()` has joined that thread, so no cross-thread write to the run's stop state
        # exists anywhere on this path.
        self._critical_fired = False
        # F-11's two counters. Written only on this guard's own thread (or the caller's, for
        # a direct `check_once`), read by the coordinator's `monitor_gates` emit.
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
        """True iff this guard's critical arm has signalled the process (WPMAIN RT-2/R132).

        The guard cannot record WHY the run is stopping — the rule name is a row of
        `mantis.config.armed_aborts.MANIFEST`, and `mantis.train` must not import that module
        (the same rule `_fire_hard_abort` obeys; spelling the name here would be a second
        authority for a manifest row's `name`). So the guard publishes the FACT and the
        composition root, which already imports the manifest for its own rc resolution, does
        the naming. `mantis.run.compose_run` reads this after `stop()` and records the rule on
        the `ShutdownState` it owns; a guard nothing composed reads inert-but-truthful.
        """
        return self._critical_fired

    def check_once(self) -> float:
        """Check disk free, emit ``disk_free``, handle thresholds. Returns free_gb (decimal
        GB, matching old ``/1e9``)."""
        usage = shutil.disk_usage(self._path)
        self._checks_total += 1
        free_gb = usage.free / _GB
        self._sink.emit({"event": "disk_free", "disk_free_gb": round(free_gb, 2)})

        if free_gb < self._fail_gb:
            # RT-2b. The SIGTERM is LATCHED; the alert is not. The guard polls every
            # `interval_sec` (minted 60 s) and the condition it fires on does not clear itself,
            # so an unlatched arm supplied the SECOND press of LAW-16's two-press force-exit
            # ITSELF: `_stop` hits `stop_count >= 2` and `sys.exit(1)`s from a signal handler,
            # at an arbitrary point in the main thread, MID-SAVE. Against `close_out`'s
            # 14400 s drain caps a save-then-exit that must finish inside 60 s is close to
            # certain to be killed — the guard destroying the save it asked for. The two-press
            # force-exit is the OPERATOR's affordance (signals.py:56) and it stays theirs.
            # The alert keeps firing every tick because the condition persists and an operator
            # watching the stream must see that; only the escalation is once-per-run.
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
                # AUDIT-1 F-11. This was a bare `_LOG.warning`, and three facts made it a
                # SILENT total failure: the run installs no logging handler at all
                # (F-08), the guard is not a heartbeat source, and gate 12's REQUIRED
                # `disk_space_exhausted` row audits ARMED off a CONFIG NUMBER — so a
                # `check_once` raising every tick emitted no `disk_free`, incremented no
                # counter, never set `_critical_fired`, and every instrument still reported
                # the guard armed. The rc-47 abort R132 closed can be dead for a whole run
                # while the volume fills and the supervisor relaunches into it.
                # The counter and the event are the two channels a monitor can actually
                # read; the WARNING stays for a terminal an operator is watching live.
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
        """How many ticks of this guard's loop raised (AUDIT-1 F-11).

        A guard whose `check_once` raises on every tick is indistinguishable from a healthy
        one on every OTHER observable — it emits no `disk_free`, so absence-of-alert reads as
        "plenty of space". This is the counter that separates them.
        """
        return self._errors_total

    @property
    def interval_sec(self) -> float:
        """This guard's own poll period (AUDIT-1 F-11 / R334(b)).

        Exposed so a liveness reader can denominate a stall deadline in the guard's OWN
        interval instead of holding a second copy of a minted config value — the duplicated-
        authority shape R1 exists to kill. A reader that captured 60.0 would judge a 5 s
        smoke guard at twelve times its period.
        """
        return self._interval

    @property
    def checks_total(self) -> int:
        """How many ticks completed. `checks_total == 0` with a started guard is the shape a
        reader must be able to see: nothing was measured, so nothing is known."""
        return self._checks_total
