"""The ONE sanctioned optional-effect path (LAW-14; WP13-A §c.2).

`best_effort(label, fn, *, counters)` is the only way an effect in `monitor/**` (and in
the watchdog fire path) may fail without taking the run down: the failure is COUNTED
under a named label and WARN-logged, never swallowed. The mandatory counter is enforced
BY SIGNATURE — `counters` is keyword-only with no default, so omitting it is a
`TypeError` at the call site, not a silently uncounted swallow.

The other outcome for an optional effect is "fail loud". `except …: pass` is banned
outright and mechanically censused over `monitor/**` (O-20).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, TypeVar

_LOG = logging.getLogger(__name__)

T = TypeVar("T")


class BestEffortCounters:
    """A TOTAL named-counter registry: an untouched label reads 0, never a KeyError.

    Thread-safe (the watchdog thread, the drain loop and the main thread all count into
    one registry); `snapshot()` is what the LAW-18 in-run summary events publish.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def increment(self, label: str) -> int:
        """Count one failure under ``label`` and return the new value."""
        with self._lock:
            value = self._counts.get(label, 0) + 1
            self._counts[label] = value
            return value

    def get(self, label: str) -> int:
        """The count for ``label`` — 0 for a never-incremented label (total registry)."""
        with self._lock:
            return self._counts.get(label, 0)

    def snapshot(self) -> dict[str, int]:
        """A copy of every non-zero label count (for the in-run visibility events)."""
        with self._lock:
            return dict(self._counts)

    def total(self) -> int:
        """Sum over every label — the single number a gate/summary can key on."""
        with self._lock:
            return sum(self._counts.values())


def best_effort(
    label: str,
    fn: Callable[[], T],
    *,
    counters: BestEffortCounters,
    logger: logging.Logger | None = None,
) -> tuple[bool, T | None]:
    """Run ``fn`` as an optional effect: ``(True, value)`` on success, ``(False, None)``
    on failure with ``counters[label] += 1`` and a WARN.

    NEVER raises (the caller is a fire path or a teardown step that must continue), and
    NEVER fails silently. ``counters`` is keyword-only with NO default on purpose (O-09).
    """
    log = logger if logger is not None else _LOG
    try:
        return True, fn()
    except Exception as exc:  # noqa: BLE001 — an optional effect must not break the caller
        counters.increment(label)
        log.warning("best_effort_failed label=%s exc=%r", label, exc)
        return False, None
