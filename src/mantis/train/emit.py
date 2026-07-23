"""The injected event-emit seam (repo_design §11 / WP10 §c.4).

`monitoring/events.emit_event` was the ONE funnel every training-side event flowed
through. `mantis/monitor/` is EMPTY until WP13, so the trainer/coordinator/lifecycle
emit through a LOCAL structural `EventSink` Protocol (single `emit(event)` method) with
an explicit `NullEventSink` no-op default (the WP9 `BotLike` precedent). WP13 supplies
the real JSONL sink + the LAW-07 producer tests; the DAG stays clean — no `train → monitor`
hard edge.

The emitted event `Mapping` carries its NAME under the ``"event"`` key (the mantis emit
convention, cf. `mantis.config.emit.ResolvedConfig.to_event_payload`); warning fields
(`knob`/`base_default`/`checkpoint_baked`, `level`, `disk_free_gb`, …) are top-level keys.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class EventSink(Protocol):
    """The emit surface the trainer / coordinator / lifecycle use.

    A single method — every higher-level builder constructs a payload dict and calls
    ``.emit(payload)``. A bare duck-typed class with an ``emit`` method satisfies it.
    """

    def emit(self, event: Mapping[str, Any]) -> None: ...


class NullEventSink:
    """Explicit no-op default (WP9 BotLike precedent). Injected everywhere until WP13
    wires the real sink."""

    def emit(self, event: Mapping[str, Any]) -> None:
        return None


def emit_via(sink: EventSink | None, event: Mapping[str, Any]) -> None:
    """Emit ``event`` through ``sink`` when one is injected; a ``None`` sink is a no-op.

    The single helper every builder in `train/` routes an optional-sink emission through,
    so no call site grows its own ``if sink is not None`` boilerplate."""
    if sink is not None:
        sink.emit(event)
