"""The production iteration_complete payload driven with stub collaborators."""
from __future__ import annotations

from typing import Any

from _spy import SpyEventSink


def iteration_complete_payload(pool: Any, rstats: Any, *, gph: Any = None,
                               sph: Any = None) -> dict[str, Any]:
    """Raises: AssertionError — when the builder does not emit exactly one event."""
    from mantis.train.events import emit_iteration_complete_event

    class _Buffer:
        size = 0
        capacity = 1024

    sink = SpyEventSink()
    emit_iteration_complete_event(
        train_step=0, games_played=0, last_iter_games=0, pool=pool,
        buffer=_Buffer(),
        games_per_hour_fn=lambda: gph, steps_per_hour_fn=(lambda: sph) if sph is not None else None,
        target_integrity={}, rstats=rstats, sink=sink, search_levers={},
    )
    assert len(sink.events) == 1, sink.events
    return sink.events[0]
