"""Dispatch-only veneer over the engine HEXG graph replay buffer.

The facade resolves the kind ONCE from `spec.representation`, so a config naming no graph
representation dies at construction rather than downstream. It is a veneer and nothing more:
zero copies (no `numpy` import and no array operation at all; identity is test-pinned), zero
storage change (capacity, eviction, weighting and the on-disk formats are the engine's, whose
byte-level cross-magic rejection it re-asserts only by letting it propagate), and zero new
metrics — `outcome_in_range_count` is absent on the graph buffer, so the caller's
missing-attribute fallback stays reachable.
"""
from __future__ import annotations

import enum
from typing import Any

from mantis._engine import HexgBuffer
from mantis.selfplay.hparams import is_graph_representation


class BufferKindMismatch(TypeError):
    """The buffer handle does not match the representation it is being used as; a `TypeError`
    subclass because it is a wiring error, not a data error."""


class BufferKind(enum.Enum):
    """The replay-storage kind. Closed set — no second arm and no default."""

    GRAPH = "graph"  # engine `HexgBuffer` (HEXG)

    @classmethod
    def from_spec(cls, spec: Any) -> BufferKind:
        """Resolve the kind from an encoding spec, delegating the closed match to the single
        representation-dispatch authority: an unknown or absent `spec.representation` raises
        rather than falling back to a dense arm."""
        is_graph_representation(spec)
        return cls.GRAPH


#: The engine class the graph kind is backed by, kept as a NAME rather than an isinstance gate
#: so the facade stays duck-typed for the recording/stub buffers the drain oracles push into.
_RAW_FOR: dict[BufferKind, type] = {BufferKind.GRAPH: HexgBuffer}


class ReplayFacade:
    """Dispatch-only veneer holding the raw engine buffer.

    NEVER copies, slices or re-dtypes an array. Passthroughs are explicit, one method per
    forwarded member, so the surface is greppable and a missing member raises from the raw
    object, unswallowed.
    """

    def __init__(self, spec: Any, raw: Any) -> None:
        self.kind = BufferKind.from_spec(spec)
        self.raw = raw

    def __repr__(self) -> str:
        return f"ReplayFacade(kind={self.kind.value!r}, raw={type(self.raw).__name__})"

    def push_graph_position(self, *record: Any, game_id: int = -1) -> None:
        """Forward one graph row. The record tuple travels verbatim and is not inspected."""
        self.raw.push_graph_position(*record, game_id=game_id)

    # passthrough surface
    @property
    def size(self) -> int:
        return self.raw.size

    @property
    def capacity(self) -> int:
        return self.raw.capacity

    def resize(self, new_capacity: int) -> None:
        self.raw.resize(new_capacity)

    def save_to_path(self, path: str) -> None:
        self.raw.save_to_path(path)

    def load_from_path(self, path: str) -> int:
        """Load from disk; a cross-format file raises the engine's own byte-level magic error,
        neither re-implemented nor swallowed here."""
        return self.raw.load_from_path(path)

    def set_weight_schedule(
        self, thresholds: list[int], weights: list[float], default_weight: float
    ) -> None:
        self.raw.set_weight_schedule(thresholds, weights, default_weight)

    def next_game_id(self) -> int:
        """Allocate the next buffer-global game id.

        Forwarded explicitly — this class has no `__getattr__` by design — and `push_graph`
        calls it through the FACADE, so omitting it would have broken the self-play write path
        on its first drained game while every raw-buffer unit test stayed green."""
        return self.raw.next_game_id()

    def last_batch_composition(self) -> dict[str, int]:
        """The last sampled batch's rows-per-game and age quantiles. Forwarded even though the
        production SAMPLING path reaches the raw buffer: the dispatcher probes with `getattr`
        and an absent instrument looks exactly like a healthy zero."""
        return self.raw.last_batch_composition()

    def outcome_in_range_count(self, lo: float, hi: float) -> int:
        """Count buffered outcomes in `[lo, hi)`; absent on a graph buffer, and the resulting
        `AttributeError` propagates so the caller's NaN fallback stays reachable."""
        return self.raw.outcome_in_range_count(lo, hi)


__all__ = ["BufferKind", "BufferKindMismatch", "ReplayFacade"]
