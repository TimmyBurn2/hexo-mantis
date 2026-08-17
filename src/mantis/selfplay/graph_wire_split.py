"""PRE-COLLATE graph-batch splitting — the mechanism `train.microbatch_caps` bounds with
(WP12-R dispatch 6 phase F2, CARD-RUN5-GPU-OOM, R179).

WHY PRE-COLLATE, AND NOT POST-COLLATE. The seam is the choice; the rest follows from it. A
post-collate split (collate once, slice and re-index the tensors) leaves the FULL-E input
tensors resident for the whole step — at run5's measured `E = 18 735 930` the first two
allocations alone are `edge_index (2,E) int64 = 300 MB` and `edge_attr (E,5) fp32 = 375 MB`,
both unbounded in E, which IS the defect. A design whose first allocation is proportional to
the uncapped quantity cannot meet a criterion that says "bounded by the caps". Partitioning
the WIRE, before any torch tensor exists, makes one micro-batch's tensors the only ones
resident.

THE COPY-OUT TRAP, RECORDED. The Rust `GraphWire` getters COPY OUT
(`crates/mantis-bridge/src/hexg.rs:192-194`), so reading a getter once per micro-batch would
copy the whole array M times. The caller converts the wire to a `GraphWirePayload` EXACTLY
ONCE per step (`graph_collate.graph_wire_from_rust`) and this module slices numpy views of
that payload, so host cost matches HEAD, which reads each getter exactly once.

THE PARTITION IS ORDER-PRESERVING, SEQUENTIAL AND GREEDY — deliberately NOT bin packing. A
packer would reorder graphs as a function of the whole batch, be non-obvious to a reader, and
save at most one micro-batch. The refusal is recorded here so a later "optimisation" argues
against a decision rather than filling a silence.

Pure numpy on CSR offsets; no torch, no config, no device. Every part is handed to the real
`collate_graph_batch(semantic="full")` afterwards and therefore passes the full 18-check
contract on its own — the slice is not trusted, it is validated.
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

#: The INFERENCE arm's naming of the same partition (F-816-10, D-2). One greedy loop, one
#: over-cap check, TWO name authorities: the shared planner's refusal must name the key an
#: operator would actually edit, and an inference-side failure that said
#: `train.microbatch_caps` would be a FALSE PROVENANCE RECORD (R73) sending them to re-mint a
#: key that had nothing to do with it. The block path and its member spellings are ONE naming
#: fact with two components and travel together — they are never passed apart.
_FUSED_KEY = "inference.fused_graph_caps"
_FUSED_MEMBERS = ("max_fused_edges", "max_fused_nodes")


class GraphMicroBatchOverCap(ValueError):
    """ONE graph exceeds a member of `train.microbatch_caps` on its own.

    No split can rescue it: micro-batching partitions at GRAPH boundaries, so a single graph
    is the atom. Raised at the partition — before any device allocation and before
    `optimizer.zero_grad()` — naming the offending graph, its edge count, its node count,
    WHICH member it exceeded, that member's value and the config key path.

    R114's original clause: never a silent truncation, never a silent drop. A skip would
    silently change the batch composition AND both loss denominators, and let a run train on
    a biased sub-population while reporting a normal step. Clamping the cap up at runtime is
    refused for a different reason: it is tune-to-green at runtime (R61) and it makes the
    peak-allocation bound unprovable.
    """


class FusedGraphOverCap(ValueError):
    """ONE graph exceeds a member of `inference.fused_graph_caps` on its own (F-816-10).

    The inference arm's twin of `GraphMicroBatchOverCap`, and deliberately NOT a subclass of it
    in either direction (D-2). The two seams must stay diagnosable APART: every trainer-side
    `except GraphMicroBatchOverCap` would otherwise silently swallow an inference-side refusal
    — turning a run-fatal memory refusal into a skipped forward on the wrong seam, with a
    message that would still read correctly and no assertion anywhere that could notice.

    Raised at the partition, BEFORE any device allocation, and it travels the existing R276
    seam unchanged: the inner `except Exception` in `_run_graph_loop` logs
    `graph_inference_forward_failed` and routes every waiter to
    `submit_graph_inference_failure`. No OOM handler, no retry, no catch-and-degrade, no new
    failure path — a retry on a memory failure is the silent catch-and-retry R276(f) forbids by
    name, and clamping the cap up at runtime is tune-to-green (R61) that makes the
    peak-allocation bound unprovable.
    """


class GraphEmptyBatchError(ValueError):
    """A graph training step was handed ZERO micro-batches (`B == 0`).

    A training step with no graphs cannot produce a gradient, so the only honest outcomes are
    a raise or a silent no-op — and a silent no-op would let a run report steps it never took
    (LAW-14's posture). HEAD already fails here, but incidentally and uninformatively:
    `ragged_policy_ce` early-returns a `torch.zeros(())` with no `grad_fn`
    (`train/losses.py:89-90`), so `loss.backward()` raises `RuntimeError: element 0 of tensors
    does not require grad` — loud, but naming neither the condition nor the subsystem.

    DECLARED DEFENSIVE: whether `sample_graph_batch` can return `n_graphs == 0` through the
    coordinator's `min_buf_size` gate is UNVERIFIED (DESIGN_DFIX §3.3 names the settling
    measurement). This is a named failure on a path of unverified reachability, not a claim
    that the path is live.
    """


@dataclass(frozen=True)
class GraphTargetSlice:
    """One micro-batch's slice of the four target arrays plus the argmax-cell sequence.

    `target_argmax_cells` is here and not left to the caller because `collate_graph_batch`
    LENGTH-CHECKS it against the part's own `B` (`graph_collate.py:586-590`,
    `AugRoundTripMismatch`) and indexes it per graph: an unsliced full-length list makes
    EVERY part raise.
    """

    policy_target: np.ndarray
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

    Returns `()` when `B == 0`: a naive reading of the greedy loop appends a trailing part
    unconditionally and yields `[(0, 0)]`, one EMPTY part, which would then collate a
    zero-graph batch. Zero parts is the honest answer and the trainer raises on it by name.

    A pure function of `(edge counts, node counts, caps)` — no host state, no RNG, no device.

    `key`/`members` name the CONFIG BLOCK the caps came from, and they exist only so the
    refusal below tells the truth on both arms (F-816-10 D-2). They are ONE naming fact with
    two components — the block path and the two member spellings under it — so they are passed
    together or not at all, and `plan_fused_forwards` is the only caller that passes them. The
    defaults are BEHAVIOUR-PRESERVING: every existing caller keeps its exact current message
    byte for byte, which is what makes a defaulted parameter here hide no authority. A REQUIRED
    parameter was the design's original shape and was overruled: it churns every direct caller
    and moves `tests/train/test_graph_microbatch_authority.py`, whose frozen AST census is the
    whole reason the inference members are not spelled `max_edges`/`max_nodes`.
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
        if int(ec[i]) > max_edges:
            raise GraphMicroBatchOverCap(
                f"graph {i} needs {int(ec[i])} edges and {int(nc[i])} nodes on its own, which "
                f"exceeds {members[0]}={max_edges} ({key}.{members[0]}). Micro-batching "
                "partitions at GRAPH boundaries, so a single graph is the atom and no split "
                "reduces it — this is out of the domain the caps were sized for. Never a "
                "silent truncation and never a silent drop (R114)."
            )
        if int(nc[i]) > max_nodes:
            raise GraphMicroBatchOverCap(
                f"graph {i} needs {int(ec[i])} edges and {int(nc[i])} nodes on its own, which "
                f"exceeds {members[1]}={max_nodes} ({key}.{members[1]}). Micro-batching "
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
    """Partition ONE fused inference pop into bounded forwards (F-816-10, verdict V-A).

    An ADAPTER over `plan_microbatches`, not a second transcription: ONE greedy loop, ONE
    over-cap check, ONE algorithm. Two implementations of one partition agree right up until
    they diverge, and the divergence would be a memory bound that is correct on one arm only.
    What differs between the arms is the NAME AUTHORITY and the exception TYPE, and both differ
    for the same reason — a refusal must send the operator to the key they actually have to
    re-mint.

    Called PRE-COLLATE, on the wire's own CSR offsets, before any torch tensor exists: a
    post-collate split leaves the full-E tensors resident for the whole forward, and a design
    whose first allocation is proportional to the uncapped quantity cannot meet a bound.
    """
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

    THE FLAT-`edge_index` TRAP, RECORDED. The wire's `edge_index` is flat of size `2E` and is
    reshaped `(2, E)` by the collate (`graph_collate.py:339-341`, and `_check_structural`'s
    `len(edge_index) != 2E` check at `:406-407`). An edge RANGE is therefore **two disjoint
    ranges of the flat array**, not one contiguous slice: `[e0:e1]` from the source row and
    `[E+e0:E+e1]` from the destination row, both shifted down by `node_offsets[g0]`.

    Every array keeps its contract dtype (numpy slicing preserves dtype; the three re-basings
    are int64-minus-int64), because `_check_structural` re-validates all thirteen dtypes on
    every part.
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
    """Slice the four target arrays and the argmax-cell sequence for graphs `[g0, g1)`.

    `policy_target` is flat PER LEGAL NODE, so its bounds come from the wire's own
    `legal_offsets` (a first-class payload field, `graph_collate.py:153`) — derivable
    pre-collate without touching torch. The other three are per-graph.
    """
    lo = np.asarray(legal_offsets, dtype=np.int64)
    l0, l1 = int(lo[g0]), int(lo[g1])
    cells: Sequence[Any] = targets.target_argmax_cells
    return GraphTargetSlice(
        policy_target=np.asarray(targets.policy_target)[l0:l1],
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
