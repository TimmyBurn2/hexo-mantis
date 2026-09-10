"""PRE-COLLATE graph-batch splitting — the mechanism `train.microbatch_caps` bounds with.

A post-collate split leaves the FULL-E input tensors resident for the whole step: at run5's
measured `E = 18_735_930` the first two allocations alone are 300 MB and 375 MB, both unbounded
in E, which IS the defect. The Rust wire getters COPY OUT, so the caller converts the wire to a
payload EXACTLY ONCE per step and this module slices numpy views of it.

The partition is order-preserving, sequential and greedy, deliberately NOT bin packing. Every
part is handed to the real `collate_graph_batch` afterwards, so the slice is validated.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.selfplay.graph_collate import GraphWirePayload

_KEY = "train.microbatch_caps"
_MEMBERS = ("max_edges", "max_nodes")

#: The INFERENCE arm's naming of the same partition. One greedy loop, one over-cap check, TWO
#: name authorities: an inference-side failure that said `train.microbatch_caps` would be FALSE
#: PROVENANCE sending an operator to re-mint a key that had nothing to do with it. The block path
#: and its member spellings are ONE naming fact and are never passed apart.
_FUSED_KEY = "inference.fused_graph_caps"
_FUSED_MEMBERS = ("max_fused_edges", "max_fused_nodes")


class GraphMicroBatchOverCap(ValueError):
    """ONE graph exceeds a member of `train.microbatch_caps` on its own, so no split can rescue
    it: micro-batching partitions at GRAPH boundaries. Raised at the partition, before any device
    allocation, naming the graph, both counts, the member breached and the config key. Never a
    silent drop, which would change the batch composition and both loss denominators while
    reporting a normal step, and never a runtime clamp, which makes the peak bound unprovable."""


class FusedGraphOverCap(ValueError):
    """The inference twin of `GraphMicroBatchOverCap`, deliberately NOT a subclass in either
    direction: a trainer-side `except GraphMicroBatchOverCap` would otherwise swallow an
    inference-side refusal, turning a run-fatal memory refusal into a skipped forward on the
    wrong seam. No OOM handler, no retry, no catch-and-degrade."""


class GraphEmptyBatchError(ValueError):
    """A graph training step was handed ZERO micro-batches, which cannot produce a gradient, so
    the only honest outcomes are a raise or a no-op that lets a run report steps it never took.
    HEAD already fails here uninformatively, naming neither the condition nor the subsystem.
    DECLARED DEFENSIVE: the path's reachability through the warmup gate is UNVERIFIED."""


@dataclass(frozen=True)
class GraphTargetSlice:
    """One micro-batch's slice of the target arrays plus the argmax-cell sequence, which is
    sliced here because the collate LENGTH-CHECKS it against the part's own `B` — an unsliced
    full-length list makes EVERY part raise."""

    policy_target: np.ndarray
    explicit_mask: np.ndarray
    tail_mass: np.ndarray
    outcomes: np.ndarray
    value_valid: np.ndarray
    is_full_search: np.ndarray
    target_argmax_cells: list[Any]


def plan_microbatches(
    edge_offsets: Any,
    node_offsets: Any,
    max_edges: int,
    max_nodes: int,
    *,
    key: str = _KEY,
    members: tuple[str, str] = _MEMBERS,
) -> tuple[tuple[int, int], ...]:
    """Partition `[0, B)` into contiguous ordered `(g0, g1)` micro-batches under BOTH members.

    Returns `()` when `B == 0`: a naive greedy loop appends a trailing part unconditionally and
    would collate a zero-graph batch. A pure function of `(edge counts, node counts, caps)`.
    `key`/`members` name the CONFIG BLOCK the caps came from, so the refusal tells the truth on
    both arms; their defaults are behaviour-preserving and hide no authority.
    """
    eo = np.asarray(edge_offsets, dtype=np.int64)
    no = np.asarray(node_offsets, dtype=np.int64)
    ec = np.diff(eo)
    nc = np.diff(no)
    b = int(ec.shape[0])
    if b == 0:
        return ()
    # Out of domain FIRST, before any packing: a single graph over either member has no split
    # that rescues it, and a bound that admits one over-bound part is not a bound.
    for i in range(b):
        # BOTH members are tested before raising: two sequential `if ... raise` statements report
        # only the first, so an operator whose graph breaks both fixes one and meets the other as
        # an apparently-new surprise.
        breaches = [(m, cap) for m, cap, count in
                    ((members[0], max_edges, int(ec[i])), (members[1], max_nodes, int(nc[i])))
                    if count > cap]
        if breaches:
            exceeds = " AND ".join(f"{m}={cap} ({key}.{m})" for m, cap in breaches)
            raise GraphMicroBatchOverCap(
                f"graph {i} needs {int(ec[i])} edges and {int(nc[i])} nodes on its own, which "
                f"exceeds {exceeds}. Micro-batching "
                "partitions at GRAPH boundaries, so a single graph is the atom and no split "
                "reduces it — this is out of the domain the caps were sized for. Never a "
                "silent truncation and never a silent drop (R114)."
            )
    parts: list[tuple[int, int]] = []
    start, acc_e, acc_n = 0, 0, 0
    for i in range(b):
        if ((acc_e + int(ec[i]) > max_edges or acc_n + int(nc[i]) > max_nodes)
                and i > start):
            parts.append((start, i))
            start, acc_e, acc_n = i, 0, 0
        acc_e += int(ec[i])
        acc_n += int(nc[i])
    parts.append((start, b))
    return tuple(parts)


def plan_fused_forwards(
    edge_offsets: Any,
    node_offsets: Any,
    caps: FusedGraphCapsSpec,
) -> tuple[tuple[int, int], ...]:
    """Partition ONE fused inference pop into bounded forwards — an ADAPTER over
    `plan_microbatches`, not a second transcription, since two implementations of one partition
    agree right up until they diverge. What differs is the NAME AUTHORITY and the exception TYPE,
    both so a refusal sends the operator to the key they have to re-mint."""
    try:
        return plan_microbatches(
            edge_offsets, node_offsets,
            caps.max_fused_edges, caps.max_fused_nodes,
            key=_FUSED_KEY, members=_FUSED_MEMBERS,
        )
    except GraphMicroBatchOverCap as exc:
        raise FusedGraphOverCap(str(exc)) from exc


def slice_graph_wire(payload: GraphWirePayload, g0: int, g1: int) -> GraphWirePayload:
    """The sub-wire holding graphs `[g0, g1)`, re-based so it is a valid wire on its own.

    THE FLAT-`edge_index` TRAP: it is flat of size `2E` and reshaped `(2, E)` by the collate, so
    an edge RANGE is TWO disjoint ranges — `[e0:e1]` and `[E+e0:E+e1]` — both shifted down by
    `node_offsets[g0]`.
    """
    no = np.asarray(payload.node_offsets, dtype=np.int64)
    eo = np.asarray(payload.edge_offsets, dtype=np.int64)
    lo = np.asarray(payload.legal_offsets, dtype=np.int64)
    n0, n1 = int(no[g0]), int(no[g1])
    e0, e1 = int(eo[g0]), int(eo[g1])
    l0, l1 = int(lo[g0]), int(lo[g1])
    total_n = int(no[-1])
    total_e = int(eo[-1])

    node_feat = np.asarray(payload.node_feat)
    feat_dim = (node_feat.size // total_n) if total_n else 0
    edge_attr = np.asarray(payload.edge_attr)
    edge_dim = (edge_attr.size // total_e) if total_e else 0
    ei = np.asarray(payload.edge_index, dtype=np.int64)
    sliced_ei = (np.concatenate([ei[e0:e1], ei[total_e + e0:total_e + e1]]) - n0
                 if total_e else ei[:0])

    return GraphWirePayload(
        contract_version=int(payload.contract_version),
        builder_impl=int(payload.builder_impl),
        n_graphs=int(g1 - g0),
        node_feat=node_feat[n0 * feat_dim:n1 * feat_dim],
        node_coords=np.asarray(payload.node_coords)[n0 * 2:n1 * 2],
        edge_index=sliced_ei,
        edge_attr=edge_attr[e0 * edge_dim:e1 * edge_dim],
        node_offsets=no[g0:g1 + 1] - n0,
        edge_offsets=eo[g0:g1 + 1] - e0,
        legal_offsets=lo[g0:g1 + 1] - l0,
        legal_node_gather=np.asarray(payload.legal_node_gather, dtype=np.int64)[l0:l1] - n0,
        policy_dst_slot=np.asarray(payload.policy_dst_slot)[l0:l1],
        n_nodes_checksum=np.asarray(payload.n_nodes_checksum)[g0:g1],
        n_stones=np.asarray(payload.n_stones)[g0:g1],
        window_center=np.asarray(payload.window_center)[g0 * 2:g1 * 2],
        current_player=np.asarray(payload.current_player)[g0:g1],
    )


def slice_targets(targets: Any, legal_offsets: Any, g0: int, g1: int) -> GraphTargetSlice:
    """Slice the target arrays and the argmax-cell sequence for graphs `[g0, g1)`.
    `policy_target` and `explicit_mask` are flat PER LEGAL NODE, so their bounds come from the
    wire's own `legal_offsets`; the rest, `tail_mass` included, are per-graph."""
    lo = np.asarray(legal_offsets, dtype=np.int64)
    l0, l1 = int(lo[g0]), int(lo[g1])
    cells: Sequence[Any] = targets.target_argmax_cells
    return GraphTargetSlice(
        policy_target=np.asarray(targets.policy_target)[l0:l1],
        explicit_mask=np.asarray(targets.explicit_mask)[l0:l1],
        tail_mass=np.asarray(targets.tail_mass)[g0:g1],
        outcomes=np.asarray(targets.outcomes)[g0:g1],
        value_valid=np.asarray(targets.value_valid)[g0:g1],
        is_full_search=np.asarray(targets.is_full_search)[g0:g1],
        target_argmax_cells=list(cells)[g0:g1],
    )


__all__ = [
    "FusedGraphOverCap",
    "GraphEmptyBatchError",
    "GraphMicroBatchOverCap",
    "GraphTargetSlice",
    "plan_fused_forwards",
    "plan_microbatches",
    "slice_graph_wire",
    "slice_targets",
]
