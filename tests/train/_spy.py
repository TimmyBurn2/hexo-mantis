"""The ONE recording event sink for tests/train drives; the conftest spy_sink wraps it."""
from __future__ import annotations

from typing import Any


class SpyEventSink:
    """Record every emitted event Mapping; the event name travels under the `event` key."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        # `event` is subscripted, not `.get`-ed: a missing key must be loud.
        return [e for e in self.events if e["event"] == name]

    def has(self, name: str) -> bool:
        return any(e["event"] == name for e in self.events)

    def order(self) -> list[str]:
        return [e["event"] for e in self.events]
