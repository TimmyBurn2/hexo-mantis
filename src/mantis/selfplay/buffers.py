"""Dispatch-only veneer over the engine HEXG graph replay buffer.

The facade resolves the kind ONCE from `spec.representation` (closed match, LAW-11), so a
config that names no graph representation dies at construction rather than downstream. The
cross-kind mislabel guard it was built for went with the grid path (R346(f)): with one
storage kind there is no other buffer to hand in by mistake.

It is a veneer and nothing more:

  * zero copies — this module contains no `numpy` import and no array operation at all;
    every push forwards the caller's ndarray objects unchanged (identity is test-pinned).
    The only array work on the push path stays where it was: the drain-side f16
    cast/reshape, which is behaviour, not facade.
  * zero storage change — capacity, eviction, weighting and the on-disk HEXB/HEXG formats
    are entirely the engine's (WP5/WP7). The byte-level cross-magic rejection on
    `load_from_path` is the engine's crate gate; the facade only re-asserts it at the seam
    by letting the engine error propagate unswallowed.
  * zero new metrics — `outcome_in_range_count` is a plain passthrough, absent on the graph
    buffer; the caller's missing-attribute fallback (a NaN `draw_target_fraction`) must stay
    reachable, so the absence is propagated, never papered over.
"""
from __future__ import annotations

import enum
from typing import Any

from mantis._engine import HexgBuffer
from mantis.selfplay.hparams import is_graph_representation


class BufferKindMismatch(TypeError):
    """The buffer handle does not match the representation it is being used as.

    Raised at construction (raw handle of the wrong engine class for the resolved kind)
    and at every push (a graph push on a dense facade, or the inverse). A `TypeError`
    subclass because it is a wiring error, not a data error.
    """


class BufferKind(enum.Enum):
    """The replay-storage kind. Closed set — there is no second arm and no default. The
    HEXB dense kind went with the grid path (R346(f))."""

    GRAPH = "graph"  # engine `HexgBuffer` (HEXG)

    @classmethod
    def from_spec(cls, spec: Any) -> BufferKind:
        """Resolve the kind from an encoding spec.

        Delegates the closed match to `hparams.is_graph_representation`, the single
        representation-dispatch authority in this package: an unknown or absent
        `spec.representation` raises `RepresentationMismatch` (LAW-11 — no
        dense-by-default arm anywhere).
        """
        is_graph_representation(spec)
        return cls.GRAPH


#: The engine class the graph kind is backed by. Kept as a NAME rather than an isinstance
#: gate: the facade stays duck-typed for the recording/stub buffers the drain oracles push
#: into, and with one kind left there is no cross-kind mislabel to exclude.
_RAW_FOR: dict[BufferKind, type] = {BufferKind.GRAPH: HexgBuffer}


class ReplayFacade:
    """Dispatch-only veneer holding the raw engine buffer.

    NEVER copies, slices or re-dtypes an array — every push forwards the caller's
    ndarray objects unchanged. Attribute passthroughs are explicit (one method per
    forwarded member) so the forwarded surface is greppable, and a member the raw
    handle does not have raises `AttributeError` from the raw object, unswallowed.
    """

    def __init__(self, spec: Any, raw: Any) -> None:
        self.kind = BufferKind.from_spec(spec)
        self.raw = raw

    def __repr__(self) -> str:
        return f"ReplayFacade(kind={self.kind.value!r}, raw={type(self.raw).__name__})"

    def push_graph_position(self, *record: Any, game_id: int = -1) -> None:
        """Forward one graph row. The record tuple travels verbatim and is not inspected."""
        self.raw.push_graph_position(*record, game_id=game_id)

    # ── passthrough surface ─────────────────────────────────────────────────────
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
        """Load from disk. A cross-format file raises the engine's own loud error —
        the HEXB/HEXG magic check is byte-level in the engine and is NOT re-implemented
        or swallowed here."""
        return self.raw.load_from_path(path)

    def set_weight_schedule(
        self, thresholds: list[int], weights: list[float], default_weight: float
    ) -> None:
        self.raw.set_weight_schedule(thresholds, weights, default_weight)

    def next_game_id(self) -> int:
        """Allocate the next buffer-global game id (R345(b)(6)).

        FORWARDED EXPLICITLY, like everything else here — this class has no `__getattr__`, by
        design, *"so the forwarded surface is greppable"*. `pool_push.push_graph` calls this
        once per distinct runner game to translate the runner's own restart-at-zero sequence
        into an id that cannot collide with a resumed ring's history, and the push arm sees
        the FACADE, not the raw handle. Omitting it here would have made the whole self-play
        write path raise `AttributeError` on its first drained game while every unit test that
        used a raw `HexgBuffer` stayed green.
        """
        return self.raw.next_game_id()

    def last_batch_composition(self) -> dict[str, int]:
        """The last sampled batch's rows-per-game and age quantiles (R345(b)(6)).

        Forwarded even though the production SAMPLING path reaches the raw buffer rather than
        this facade (`run.py` hands the coordinator the raw handle and the pool the wrapped
        one). The dispatcher probes for this member with `getattr` and publishes nothing when
        it is absent — so if the two ever converge on the facade, the instrument must not go
        silently missing, and an absent instrument looks exactly like a healthy zero.
        """
        return self.raw.last_batch_composition()

    def outcome_in_range_count(self, lo: float, hi: float) -> int:
        """Count buffered outcomes in `[lo, hi)`.

        Present on the dense buffer only. On a graph buffer the attribute is genuinely
        absent — old-side truth — and the resulting `AttributeError` propagates so the
        caller's documented fallback (NaN `draw_target_fraction`) stays reachable.
        """
        return self.raw.outcome_in_range_count(lo, hi)


__all__ = ["BufferKind", "BufferKindMismatch", "ReplayFacade"]
