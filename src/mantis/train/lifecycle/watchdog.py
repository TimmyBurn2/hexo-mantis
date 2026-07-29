"""Self-play stall watchdog (repo_design §11; WP10 §a.2 / §c.3).

Extracted from the old `training/step_coordinator.py` stall-watchdog slice (L193-198,
435-448, 678-716, 842-850) into a standalone armable/fireable unit so the ⊕ lifecycle
suite tests it without a full StepCoordinator; the coordinator drives it via
``watchdog.tick(...)`` (Slice-2 wiring). Behaviour byte-identical: always armed
(arm-log unconditional), ``tick`` resets the stall clock on new games, fire → loud log +
best-effort snapshot to a DISTINCT ``.watchdog`` path + ``exit_fn(SELFPLAY_STALL_EXIT_CODE)``.

Origin (2026-07-11 run2 eval-boundary wedge): a wedged self-play/eval GPU deadlock froze
games for ~45h while the main loop looped forever. 30 min ≫ any legitimate zero-games gap.
``timeout_sec <= 0`` disables the fire (documented off switch); the arm-log still emits.
"""
from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mantis.train.emit import emit_via

_LOG = logging.getLogger(__name__)

# Wall-clock seconds with NO new self-play game after which the run fails fast.
DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC: float = 1800.0
# Distinct non-zero exit code so a launch/restart wrapper can key on a watchdog abort.
SELFPLAY_STALL_EXIT_CODE: int = 42


def watchdog_snapshot_path(canonical: Path) -> Path:
    """The fire-time buffer-snapshot path: ``<canonical>.watchdog``.

    The watchdog NEVER writes the canonical resume buffer — its non-atomic save fires
    exactly in the abnormal-exit regime where a mid-write kill is most likely, and a
    separate path can never truncate the known-good ``replay_buffer.bin``.
    """
    return Path(str(canonical) + ".watchdog")


class StallWatchdog:
    """Fail-fast watchdog: fire when ``games_completed`` stops advancing for
    ``timeout_sec``.

    Collaborators are injected (repo_design §11): ``clock`` (a callable returning the
    current monotonic time), ``sink`` (the EventSink), ``exit_fn`` (defaults to
    ``os._exit`` so a wedged clean-shutdown attempt is avoided), ``save_snapshot`` (the
    best-effort buffer snapshot closure).
    """

    def __init__(
        self,
        *,
        timeout_sec: float,
        clock: Callable[[], float],
        sink: Any,
        exit_fn: Callable[[int], None] = os._exit,
        save_snapshot: Callable[[], None],
    ) -> None:
        self._timeout = timeout_sec
        self._clock = clock
        self._sink = sink
        self._exit_fn = exit_fn
        self._save_snapshot = save_snapshot
        self._last_games = 0
        self._last_progress_time = 0.0

    def arm(self, games_completed: int) -> None:
        """Seed the stall clock + games count and emit ``selfplay_stall_watchdog_armed``.

        Always armed (context law): the arm-log fires regardless of config so a
        disabled/misconfigured watchdog (``timeout_sec <= 0`` or a non-finite value that
        silently never fires) is VISIBLE, not silent.
        """
        self._last_games = games_completed
        self._last_progress_time = self._clock()
        # None-sink tolerant (house emit convention): a harness coordinator built with
        # sink=None must still construct/arm; a real run always injects the JSONL sink.
        emit_via(
            self._sink,
            {
                "event": "selfplay_stall_watchdog_armed",
                "timeout_sec": self._timeout,
                "enabled": bool(math.isfinite(self._timeout) and self._timeout > 0),
            }
        )

    def tick(self, games_completed: int, now: float) -> None:
        """Advance the watchdog: reset the stall clock on new games, else fire on stall.

        A new game (``games_completed`` increased) resets ``_last_games`` +
        ``_last_progress_time`` to ``now``. Otherwise, when ``timeout_sec > 0`` and the
        stall has reached the timeout, fire.
        """
        if games_completed > self._last_games:
            self._last_games = games_completed
            self._last_progress_time = now
        elif self._timeout > 0:
            stalled = now - self._last_progress_time
            if stalled >= self._timeout:
                self._fire(stalled)

    def _fire(self, stalled_for: float) -> None:
        """LOUD log → best-effort snapshot → exit with a distinct code. A clean shutdown
        is avoided on purpose (it would hang on the wedged GPU)."""
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
        try:
            self._save_snapshot()
        except Exception:  # noqa: BLE001 — fail-fast must not be blocked by a save
            pass
        self._exit_fn(SELFPLAY_STALL_EXIT_CODE)
