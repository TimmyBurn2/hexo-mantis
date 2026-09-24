"""Shared fixtures for the monitor suite (tests/monitor/): plain duck-typed spies, torch-free."""
from __future__ import annotations

from typing import Any

import pytest
from _drivable import FakeClock


class SpyEventSink:
    """Records every emitted event Mapping (structural `EventSink`: single `emit`)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


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


class CallSpy:
    """A zero-arg callable that counts invocations (snapshot / sink-close / drain spies)."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.count += 1


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
