"""Dispatch-only veneer over the engine replay buffers (HEXB dense / HEXG graph).

The facade exists for ONE reason: the mislabel class. Old-side the pool held a raw
`ReplayBuffer` / `HexgBuffer` and chose the push arm from the encoding spec; nothing
cross-checked that the buffer handed in actually matched the representation the pool
dispatched on, so a graph payload could be pushed into a dense buffer (or vice versa)
and only surface later as corrupt training data. `ReplayFacade` resolves the kind ONCE
from `spec.representation` (closed match, LAW-11) and cross-checks the raw handle, so a
mislabel dies at construction with a named error.

It is a veneer and nothing more:

  * zero copies — this module contains no `numpy` import and no array operation at all;
    every push forwards the caller's ndarray objects unchanged (identity is test-pinned).
    The only array work on the push path stays where it was: the drain-side f16
    cast/reshape, which is behaviour, not facade.
  * zero storage change — capacity, eviction, weighting and the on-disk HEXB/HEXG formats
    are entirely the engine's (WP5/WP7). The byte-level cross-magic rejection on
    `load_from_path` is the engine's crate gate; the facade only re-asserts it at the seam
    by letting the engine error propagate unswallowed.
  * zero new metrics — `outcome_in_range_count` is a plain passthrough. It exists on the
    dense buffer and NOT on the graph buffer, which is old-side truth: the caller's
    missing-attribute fallback (a NaN `draw_target_fraction` on the graph path) must stay
    reachable, so the absence is propagated, never papered over.
"""
from __future__ import annotations

import enum
from typing import Any

from mantis._engine import HexgBuffer, ReplayBuffer
from mantis.selfplay.hparams import is_graph_representation


class BufferKindMismatch(TypeError):
    """The buffer handle does not match the representation it is being used as.

    Raised at construction (raw handle of the wrong engine class for the resolved kind)
    and at every push (a graph push on a dense facade, or the inverse). A `TypeError`
    subclass because it is a wiring error, not a data error.
    """


class BufferKind(enum.Enum):
    """The two replay-storage kinds. Closed set — there is no third arm and no default."""

    GRID = "grid"  # engine `ReplayBuffer` (HEXB)
    GRAPH = "graph"  # engine `HexgBuffer` (HEXG)

    @classmethod
    def from_spec(cls, spec: Any) -> BufferKind:
        """Resolve the kind from an encoding spec.

        Delegates the closed match to `hparams.is_graph_representation`, the single
        representation-dispatch authority in this package: an unknown or absent
        `spec.representation` raises `RepresentationMismatch` (LAW-11 — no
        dense-by-default arm anywhere).
        """
        return cls.GRAPH if is_graph_representation(spec) else cls.GRID


# The engine class that must NOT appear under each kind. Checked by exclusion rather
# than by allowlist so the facade stays duck-typed for the recording/stub buffers the
# drain oracles push into, while the real mislabel (HEXG under grid, HEXB under graph)
# still dies loudly.
_WRONG_RAW_FOR: dict[BufferKind, type] = {
    BufferKind.GRID: HexgBuffer,
    BufferKind.GRAPH: ReplayBuffer,
}


class ReplayFacade:
    """Dispatch-only veneer holding the raw engine buffer.

    NEVER copies, slices or re-dtypes an array — every push forwards the caller's
    ndarray objects unchanged. Attribute passthroughs are explicit (one method per
    forwarded member) so the forwarded surface is greppable, and a member the raw
    handle does not have raises `AttributeError` from the raw object, unswallowed.
    """

    def __init__(self, spec: Any, raw: Any) -> None:
        kind = BufferKind.from_spec(spec)
        wrong = _WRONG_RAW_FOR[kind]
        if isinstance(raw, wrong):
            raise BufferKindMismatch(
                f"replay buffer is a {type(raw).__name__} but the resolved encoding "
                f"representation is {kind.value!r}; a "
                f"{'graph' if kind is BufferKind.GRID else 'dense'} buffer cannot back "
                f"the {kind.value} self-play write path. Build the buffer from the same "
                "encoding the pool resolves."
            )
        self.kind = kind
        self.raw = raw

    def __repr__(self) -> str:
        return f"ReplayFacade(kind={self.kind.value!r}, raw={type(self.raw).__name__})"

    def _require(self, kind: BufferKind, method: str) -> None:
        if self.kind is not kind:
            raise BufferKindMismatch(
                f"{method} is the {kind.value} write path but this facade wraps a "
                f"{self.kind.value} buffer ({type(self.raw).__name__}). The push arm and "
                "the resolved representation disagree."
            )

    # ── push arms (the only two write paths) ────────────────────────────────────
    def push_dense_many(
        self,
        states: Any,
        chain_planes: Any,
        policies: Any,
        outcomes: Any,
        ownership: Any,
        winning_line: Any,
        game_lengths: Any,
        is_full_search: Any,
        position_indices: Any = None,
        value_target_valid: Any = None,
    ) -> None:
        """Forward one bulk dense push. Argument objects travel unchanged."""
        self._require(BufferKind.GRID, "push_dense_many")
        self.raw.push_many(
            states,
            chain_planes,
            policies,
            outcomes,
            ownership,
            winning_line,
            game_lengths,
            is_full_search,
            position_indices,
            value_target_valid=value_target_valid,
        )

    def push_graph_position(self, *record: Any, game_id: int = -1) -> None:
        """Forward one graph row. The record tuple travels verbatim and is not inspected."""
        self._require(BufferKind.GRAPH, "push_graph_position")
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

    def outcome_in_range_count(self, lo: float, hi: float) -> int:
        """Count buffered outcomes in `[lo, hi)`.

        Present on the dense buffer only. On a graph buffer the attribute is genuinely
        absent — old-side truth — and the resulting `AttributeError` propagates so the
        caller's documented fallback (NaN `draw_target_fraction`) stays reachable.
        """
        return self.raw.outcome_in_range_count(lo, hi)


__all__ = ["BufferKind", "BufferKindMismatch", "ReplayFacade"]
