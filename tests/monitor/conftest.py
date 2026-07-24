"""Shared fixtures for the WP13-A monitor oracle suite (tests/monitor/).

Torch-free by construction (§b O-18): this conftest imports ONLY stdlib and NEVER
`mantis.monitor.*` (which does not exist until IMPL) — so tests/monitor collects cleanly
while the ⊕ oracle files are RED-at-import (their own top-level `import mantis.monitor.*`
raises ModuleNotFoundError; that is the oracle-first proof, PREREG abort condition 1).

The spies are plain duck-typed classes: the injected `EventSink` is a structural Protocol
(single `emit` method) so a bare class with `.emit` satisfies it without importing the
not-yet-written concrete sink. No `sys.path` writes, no torch (R5/LAW-17).
"""
from __future__ import annotations

from typing import Any, Callable

import pytest


class SpyEventSink:
    """Records every emitted event Mapping (structural `EventSink`: single `emit`)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]

    def has(self, name: str) -> bool:
        return any(e.get("event") == name for e in self.events)


class FakeClock:
    """Controllable monotonic clock. `clock()` returns the current fake time `t`; the
    injected registry/watchdog read ONLY this, so a real wall-clock jump changes nothing."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


class ExitSpy:
    """Injected `exit_fn`: records every code instead of calling `os._exit`, and sets a
    `threading.Event`-like flag so a bounded real-thread test can wait on the first fire."""

    def __init__(self) -> None:
        self.codes: list[int] = []

    def __call__(self, code: int) -> None:
        self.codes.append(int(code))

    @property
    def fired(self) -> bool:
        return bool(self.codes)

    @property
    def first(self) -> int | None:
        return self.codes[0] if self.codes else None


class CallSpy:
    """A zero-arg callable that counts invocations (snapshot / sink-close / drain spies)."""

    def __init__(self, ret: Any = None, raises: BaseException | None = None) -> None:
        self.count = 0
        self._ret = ret
        self._raises = raises

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        if self._raises is not None:
            raise self._raises
        return self._ret


@pytest.fixture
def spy_sink() -> SpyEventSink:
    return SpyEventSink()


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def exit_spy() -> ExitSpy:
    return ExitSpy()


@pytest.fixture
def snapshot_spy() -> CallSpy:
    return CallSpy()


@pytest.fixture
def mutable_counter() -> "Callable[[], int]":
    """A live-reading counter closure over a 1-element list, so a test can bump the counter
    AFTER the watchdog is constructed and prove the watchdog re-reads it every poll."""
    box = [0]

    def _fn() -> int:
        return box[0]

    _fn.box = box  # type: ignore[attr-defined]  # tests bump _fn.box[0]
    return _fn
