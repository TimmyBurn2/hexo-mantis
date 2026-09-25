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

from mantis.selfplay.hparams import is_graph_representation


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

    def push_graph_position(self, *record: Any, game_id: int = -1, tail_mass: float = 0.0) -> None:
        """Forward one graph row. The record tuple travels verbatim and is not inspected."""
        self.raw.push_graph_position(*record, game_id=game_id, tail_mass=tail_mass)

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

    def sym_draw_counts(self) -> tuple[list[int], int]:
        """The augmentation bins and empty-board skips since boot; forwarded as `last_batch_composition` is."""
        return self.raw.sym_draw_counts()

    def samples_consumed_total(self) -> int:
        """Rows the ring handed the trainer since boot, the replay ratio's numerator."""
        return self.raw.samples_consumed_total()

    def outcome_in_range_count(self, lo: float, hi: float) -> int:
        """Count buffered outcomes in `[lo, hi)`; absent on a graph buffer, and the resulting
        `AttributeError` propagates so the caller's NaN fallback stays reachable."""
        return self.raw.outcome_in_range_count(lo, hi)


__all__ = ["BufferKind", "ReplayFacade"]
