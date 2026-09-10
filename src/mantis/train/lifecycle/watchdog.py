"""Self-play stall watchdog: fire when self-play stops completing games.

Always armed (the arm-log is unconditional); ``tick`` resets the stall clock on new games, and
a fire is a loud log plus a best-effort snapshot to a DISTINCT ``.watchdog`` path plus
``exit_fn``. Origin: a wedged self-play/eval GPU deadlock froze games for ~45 h while the main
loop looped forever. ``timeout_sec <= 0`` disables the fire; the arm-log still emits.
"""
from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mantis.monitor.best_effort import BestEffortCounters, best_effort
from mantis.train.emit import emit_via

_LOG = logging.getLogger(__name__)

# Wall-clock seconds with NO new self-play game after which the run fails fast.
DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC: float = 1800.0
# Distinct non-zero exit code so a launch/restart wrapper can key on a watchdog abort.
SELFPLAY_STALL_EXIT_CODE: int = 42


def watchdog_snapshot_path(canonical: Path) -> Path:
    """The fire-time buffer-snapshot path: ``<canonical>.watchdog``. The watchdog never writes
    the canonical resume buffer, whose non-atomic save could be killed mid-write."""
    return Path(str(canonical) + ".watchdog")


class StallWatchdog:
    """Fail-fast watchdog: fire when ``games_completed`` stops advancing for ``timeout_sec``.

    Collaborators are injected, and ``exit_fn`` defaults to ``os._exit`` so a wedged
    clean-shutdown attempt is avoided.
    """

    def __init__(
        self,
        *,
        timeout_sec: float,
        clock: Callable[[], float],
        sink: Any,
        exit_fn: Callable[[int], None] = os._exit,
        save_snapshot: Callable[[], None],
        save_model: Callable[[], None] | None = None,
        counters: BestEffortCounters | None = None,
    ) -> None:
        self._timeout = timeout_sec
        self._clock = clock
        self._sink = sink
        self._exit_fn = exit_fn
        self._save_snapshot = save_snapshot
        # `save_model` is what makes a stall abort SURVIVABLE: the fire path saved the replay
        # buffer and nothing else, so a wedged run kept its positions and lost its WEIGHTS.
        self._save_model = save_model
        # An optional effect in a fire path is COUNTED, never swallowed. Owned here when not
        # injected, so the count is readable via `.counters` even in a harness.
        self._counters = counters if counters is not None else BestEffortCounters()
        self._last_games = 0
        self._last_progress_time = 0.0

    @property
    def counters(self) -> BestEffortCounters:
        """Best-effort failure counts for the fire path; 0 for a label that never failed."""
        return self._counters

    def arm(self, games_completed: int) -> None:
        """Seed the stall clock + games count and emit ``selfplay_stall_watchdog_armed``, which
        fires regardless of config so a disabled watchdog is VISIBLE rather than silent."""
        self._last_games = games_completed
        self._last_progress_time = self._clock()
        # None-sink tolerant: a harness coordinator built with sink=None must still arm.
        emit_via(
            self._sink,
            {
                "event": "selfplay_stall_watchdog_armed",
                "timeout_sec": self._timeout,
                "enabled": bool(math.isfinite(self._timeout) and self._timeout > 0),
            }
        )

    def tick(self, games_completed: int, now: float) -> None:
        """Advance the watchdog: reset the stall clock on new games, else fire on stall."""
        if games_completed > self._last_games:
            self._last_games = games_completed
            self._last_progress_time = now
        elif self._timeout > 0:
            stalled = now - self._last_progress_time
            if stalled >= self._timeout:
                self._fire(stalled)

    def _fire(self, stalled_for: float) -> None:
        """LOUD log → best-effort snapshot → exit with a distinct code; a clean shutdown would
        hang on the wedged GPU."""
        emit_via(
            self._sink,
            {
                "event": "selfplay_stall_watchdog",
                "stalled_for_sec": round(stalled_for, 1),
                "threshold_sec": self._timeout,
            }
        )
        _LOG.error(
            "selfplay_stall_watchdog: self-play produced no new games for %.0fs "
            "(>= %.0fs threshold) — likely a wedged self-play/eval GPU deadlock; "
            "failing fast so the run can be restarted",
            stalled_for,
            self._timeout,
        )
        # MODEL FIRST, then buffer: the weights are the expensive half to regenerate, so they
        # are written before the buffer can consume the remaining time or disk. Both are
        # best-effort but COUNTED, never swallowed.
        if self._save_model is not None:
            best_effort("watchdog_model_save", self._save_model, counters=self._counters)
        best_effort("watchdog_snapshot", self._save_snapshot, counters=self._counters)
        failures = self._counters.snapshot()
        if failures:
            emit_via(self._sink, {"event": "selfplay_stall_watchdog_save_failed", **failures})
        self._exit_fn(SELFPLAY_STALL_EXIT_CODE)
