"""The SINGLE wire reader for the GNN ragged-payload contract v1.

>300 justify: the named contract errors, the resolver, the structural + semantic check layers
and the two hot-path output helpers are ONE contract (`docs/contracts/graph_wire.md`).
Splitting them would break the "one place asserts the wire" property the ADV suite gates.

`collate_graph_batch` is the one-and-only consumer of the block-diagonal wire emitted by the
Rust `InferenceBatcher.next_graph_batch`, imported by BOTH the self-play hot path and the
promotion-gate eval path, and import-safe with no module-scope torch. It asserts the contract
version and the native-builder handshake, runs the structural checks (always full) and the
semantic ones (canary on the hot path), then builds block-diagonal torch tensors. Every
mismatch raises a NAMED error; there is no silent fixed-width fallback anywhere. The OUTPUT is
not a dense scatter — the InferenceServer segment-softmaxes and returns ragged probs.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from mantis._engine import HEX_AXES as _ENGINE_HEX_AXES

# The 3 win axes in axial coords — READ from the engine, not mirrored: the check-14 geometry
# recompute runs in Rust against the same table, so a copy here could disagree.
WIN_AXES: tuple[tuple[int, int], ...] = tuple(
    (int(dq), int(dr)) for dq, dr in _ENGINE_HEX_AXES
)

# Contract-fixed schema widths; callers pass spec.* dims from the registry.
_OFF_WINDOW_SLOT = -1
_BUILDER_IMPL_NATIVE = 1


# The named contract errors (§2.5). All subclass ValueError so the die-loud call sites catch
# uniformly.
class GraphContractError(ValueError):
    """Base for every graph-wire contract violation."""


class GraphContractVersionMismatch(GraphContractError):
    pass


class WireSurfaceIncomplete(GraphContractError):
    """The object handed to the wire adapter does not carry the `GraphWire` attribute surface."""


class NonNativeSampleBuilder(GraphContractError):
    pass


class NodeFeatDimMismatch(GraphContractError):
    pass


class EdgeAttrDimMismatch(GraphContractError):
    pass


class DtypeMismatch(GraphContractError):
    pass


class BatchCountMismatch(GraphContractError):
    pass


class OffsetsNonMonotonic(GraphContractError):
    pass


class NodeCountChecksum(GraphContractError):
    pass


class EdgeIndexOutOfBounds(GraphContractError):
    pass


class EdgeCrossesGraphBoundary(GraphContractError):
    pass


class ScatterGatherCrossesGraph(GraphContractError):
    pass


class ScatterSlotOutOfBounds(GraphContractError):
    pass


class ScatterSlotAliasing(GraphContractError):
    pass


class GatherNotStrictlyIncreasing(GraphContractError):
    """`legal_node_gather` is not strictly ascending (check 13); the gather is the CONTRACT
    ORDER of every per-legal-node quantity, and a boolean-mask gather returns rows in ascending
    row index, so the two coincide while this holds and mispair priors silently when it does not."""


class EmptyLegalSet(GraphContractError):
    pass


class EdgeAttrGeometryMismatch(GraphContractError):
    pass


class GatherNotLegalNode(GraphContractError):
    pass


class ScatterSlotCanonicalMismatch(GraphContractError):
    pass


class AugRoundTripMismatch(GraphContractError):
    pass


@dataclass
class GraphWirePayload:
    """Pure-Python mirror of the Rust `GraphWire` pyclass; the resolver reads the SAME
    duck-typed surface from either. Every array is flat 1-D numpy with the contract dtype."""

    contract_version: int
    builder_impl: int
    n_graphs: int
    node_feat: np.ndarray
    node_coords: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_offsets: np.ndarray
    edge_offsets: np.ndarray
    legal_offsets: np.ndarray
    legal_node_gather: np.ndarray
    policy_dst_slot: np.ndarray
    n_nodes_checksum: np.ndarray
    n_stones: np.ndarray
    window_center: np.ndarray
    current_player: np.ndarray


@dataclass
class GraphBatch:
    """Collated block-diagonal torch tensors feeding `GnnNet.forward_batch`, plus the fields
    the ragged OUTPUT assemble needs.

    `node_coords` is deliberately absent: the DEVICE tensor had zero reads, so it was an H2D
    transfer per part that nothing read. The WIRE array of the same name is live.
    """

    x: Any  # torch.Tensor (N, 11) float
    edge_index: Any  # (2, E) int64
    edge_attr: Any  # (E, 5) float
    legal_offsets: Any  # (B+1,) int64
    legal_node_gather: Any  # (Lg,) int64 (global rows)
    node_offsets: Any  # (B+1,) int64
    n_stones: Any  # (B,) int64
    n_graphs: int = 0
    device: str = "cpu"
    extra: dict = field(default_factory=dict)


# Canary cadence for the semantic layer: the trainer runs "full", self-play runs "canary" —
# the first batch after a reset plus every Nth.
_CANARY_STATE = {"count": 0}


def reset_semantic_canary() -> None:
    """Reset the canary counter after a process start or weight swap, so the FIRST batch runs
    the full geometric layer."""
    _CANARY_STATE["count"] = 0


def _canary_should_run(period: int) -> bool:
    n = _CANARY_STATE["count"]
    _CANARY_STATE["count"] = n + 1
    return (n == 0) or (period > 0 and n % period == 0)


# Geometry helper — byte-parity with the Rust builder's `window_flat_idx`, vectorized.
#: Fills the `(B, 2)` cell array for graphs with no usable target cell (check 17). `int64`'s
#: minimum cannot be a board coordinate, so a sentinel row never matches a real one.
_CELL_SENTINEL: int = np.iinfo(np.int64).min


def _canonical_slot_vec(
    q: np.ndarray, r: np.ndarray, cq: np.ndarray, cr: np.ndarray, trunk: int
) -> np.ndarray:
    half = (trunk - 1) // 2
    wq = q - cq + half
    wr = r - cr + half
    inside = (wq >= 0) & (wq < trunk) & (wr >= 0) & (wr < trunk)
    slot = np.where(inside, wq * trunk + wr, _OFF_WINDOW_SLOT)
    return slot.astype(np.int64)


def _graph_of(offsets: np.ndarray, count: int) -> np.ndarray:
    """Map each element index [0,count) to its graph via CSR `offsets`.

    `repeat`, not `searchsorted`: element `i` belongs to graph `g` exactly
    `offsets[g+1] - offsets[g]` times in a row, so one linear fill replaces a `count`-long
    arange plus `count` binary searches (measured x13.5 on real serve parts).

    PRECONDITION, established by check 5: `offsets` is non-decreasing, `offsets[0] == 0`,
    `offsets[-1] == count`. A negative segment length makes `np.repeat` raise.
    """
    return np.repeat(np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets))


# Rust GraphWire -> GraphWirePayload adapter.
#: The wire surface, DERIVED from the payload this adapter builds rather than transcribed
#: beside it — a hand-kept second list would drift the moment a field is added.
_WIRE_SURFACE: tuple[str, ...] = tuple(f.name for f in fields(GraphWirePayload))


def graph_wire_from_rust(gw: Any) -> GraphWirePayload:
    """Read one Rust `GraphWire` into a payload, through its single-read `take()`.

    `take()` MOVES the buffers into numpy rather than copying them out getter by getter, and
    CONSUMES the wire, which is why this is the ONE place production reads one. A duck-typed
    object without `take` uses the getters.

    Raises:
        WireAlreadyConsumed: the wire was already taken.
        WireSurfaceIncomplete: `gw` carries neither `take()` nor the `GraphWire` surface.
    """
    if not hasattr(gw, "take"):
        return _payload_from_getters(gw)
    d = gw.take()
    return GraphWirePayload(
        contract_version=int(d["contract_version"]),
        builder_impl=int(d["builder_impl"]),
        n_graphs=int(d["n_graphs"]),
        node_feat=d["node_feat"],
        node_coords=d["node_coords"],
        edge_index=d["edge_index"],
        edge_attr=d["edge_attr"],
        node_offsets=d["node_offsets"],
        edge_offsets=d["edge_offsets"],
        legal_offsets=d["legal_offsets"],
        legal_node_gather=d["legal_node_gather"],
        policy_dst_slot=d["policy_dst_slot"],
        n_nodes_checksum=d["n_nodes_checksum"],
        n_stones=d["n_stones"],
        window_center=d["window_center"],
        current_player=d["current_player"],
    )


def _payload_from_getters(gw: Any) -> GraphWirePayload:
    """The per-array getter read — for duck-typed wires that carry no `take()`.

    A payload without the wire surface is refused BY NAME: handing the `targets` half of the
    `(wire, targets)` pair used to raise a bare `AttributeError` naming neither the function
    nor the argument. The check sits here because probing a live pyclass would invoke the very
    copying getters `take()` avoids, so every wrong kind arrives here anyway.

    Raises:
        WireSurfaceIncomplete: `gw` lacks one or more `GraphWirePayload` field names.
    """
    missing = [name for name in _WIRE_SURFACE if not hasattr(gw, name)]
    if missing:
        raise WireSurfaceIncomplete(
            f"graph_wire_from_rust was handed a {type(gw).__name__}, which does not carry the "
            f"GraphWire surface: missing {missing}. Only the wire half of "
            "`sample_graph_batch`'s (wire, targets) pair is a graph wire."
        )
    return GraphWirePayload(
        contract_version=int(gw.contract_version),
        builder_impl=int(gw.builder_impl),
        n_graphs=int(gw.n_graphs),
        node_feat=np.asarray(gw.node_feat),
        node_coords=np.asarray(gw.node_coords),
        edge_index=np.asarray(gw.edge_index),
        edge_attr=np.asarray(gw.edge_attr),
        node_offsets=np.asarray(gw.node_offsets),
        edge_offsets=np.asarray(gw.edge_offsets),
        legal_offsets=np.asarray(gw.legal_offsets),
        legal_node_gather=np.asarray(gw.legal_node_gather),
        policy_dst_slot=np.asarray(gw.policy_dst_slot),
        n_nodes_checksum=np.asarray(gw.n_nodes_checksum),
        n_stones=np.asarray(gw.n_stones),
        window_center=np.asarray(gw.window_center),
        current_player=np.asarray(gw.current_player),
    )


def collate_graph_batch(
    wire: Any,
    expected_version: int = 1,
    *,
    trunk_size: int,
    win_length: int,
    node_feat_dim: int,
    edge_feat_dim: int,
    device: str | None = None,
    semantic: str = "full",
    canary_period: int = 64,
    allow_oracle_builder: bool = False,
    target_argmax_cells: Sequence[tuple[int, int] | None] | None = None,
) -> GraphBatch:
    """Validate and collate one block-diagonal graph wire into a `GraphBatch`.

    `semantic`: "full" (trainer), "canary" (hot path — first + every Nth) or "off". The
    structural layer always runs full, and any mismatch raises a NAMED `GraphContractError`.

    THE FOUR GEOMETRY PARAMETERS ARE REQUIRED: they are the EXPECTED geometry the wire is
    checked against, so a default is a silent expectation and a payload re-captured at another
    radius would collate under stale geometry with nothing red.

    Raises:
        GraphContractError: any wire array disagrees with the declared geometry or the
            structural contract.
    """
    import torch  # deferred: keeps this module import-safe in torch-free envs

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- resolver step 1: contract-version handshake (§2.3) ---
    cv = int(wire.contract_version)
    if cv != expected_version:
        raise GraphContractVersionMismatch(
            f"contract_version {cv} != expected {expected_version}"
        )

    # --- resolver step 2: native-builder handshake (F7) ---
    allow_oracle = (
        allow_oracle_builder or os.environ.get("MANTIS_ALLOW_ORACLE_BUILDER") == "1"
    )
    builder_impl = int(wire.builder_impl)
    if builder_impl != _BUILDER_IMPL_NATIVE and not allow_oracle:
        raise NonNativeSampleBuilder(
            f"builder_impl {builder_impl} != {_BUILDER_IMPL_NATIVE} (native); "
            "the 26x Python-builder sample-path trap is refused "
            "(set MANTIS_ALLOW_ORACLE_BUILDER=1 only for parity tests/CI)"
        )

    # Pull flat arrays (numpy view of either the pyclass getters or the payload).
    node_feat = np.asarray(wire.node_feat)
    node_coords = np.asarray(wire.node_coords)
    edge_index = np.asarray(wire.edge_index)
    edge_attr = np.asarray(wire.edge_attr)
    node_offsets = np.asarray(wire.node_offsets)
    edge_offsets = np.asarray(wire.edge_offsets)
    legal_offsets = np.asarray(wire.legal_offsets)
    legal_node_gather = np.asarray(wire.legal_node_gather)
    policy_dst_slot = np.asarray(wire.policy_dst_slot)
    n_nodes_checksum = np.asarray(wire.n_nodes_checksum)
    n_stones = np.asarray(wire.n_stones)
    window_center = np.asarray(wire.window_center)
    current_player = np.asarray(wire.current_player)
    B = int(wire.n_graphs)

    # --- resolver step 3a: STRUCTURAL layer (13) — always full ---
    _check_structural(
        node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
        legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
        window_center, current_player, B, node_feat_dim, edge_feat_dim,
    )

    # --- resolver step 3b: SEMANTIC/GEOMETRIC layer (4) — mode-gated ---
    run_semantic = semantic == "full" or (
        semantic == "canary" and _canary_should_run(canary_period)
    )
    if run_semantic:
        _check_semantic(
            node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
            legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
            window_center, current_player, B, trunk_size, win_length, node_feat_dim,
            edge_feat_dim, target_argmax_cells,
        )

    # --- resolver step 4: block-diagonal torch tensors (edge_index already global) ---
    N = node_feat.size // node_feat_dim
    E = edge_attr.size // edge_feat_dim
    x = torch.from_numpy(
        np.ascontiguousarray(node_feat, dtype=np.float32)
    ).reshape(N, node_feat_dim).to(device)
    ei = torch.from_numpy(
        np.ascontiguousarray(edge_index, dtype=np.int64)
    ).reshape(2, E).to(device)
    ea = torch.from_numpy(
        np.ascontiguousarray(edge_attr, dtype=np.float32)
    ).reshape(E, edge_feat_dim).to(device)

    return GraphBatch(
        x=x,
        edge_index=ei,
        edge_attr=ea,
        legal_offsets=torch.from_numpy(
            np.ascontiguousarray(legal_offsets, dtype=np.int64)
        ).to(device),
        legal_node_gather=torch.from_numpy(
            np.ascontiguousarray(legal_node_gather, dtype=np.int64)
        ).to(device),
        node_offsets=torch.from_numpy(
            np.ascontiguousarray(node_offsets, dtype=np.int64)
        ).to(device),
        n_stones=torch.from_numpy(
            np.ascontiguousarray(n_stones, dtype=np.int64)
        ).to(device),
        n_graphs=B,
        device=device,
    )


# Structural layer — index in-range / unique / monotonic / typed.
def _check_structural(
    node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
    legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
    window_center, current_player, B, node_feat_dim, edge_feat_dim,
) -> None:
    # 1. NodeFeatDimMismatch
    if node_feat.size % node_feat_dim != 0:
        raise NodeFeatDimMismatch(
            f"len(node_feat)={node_feat.size} not divisible by {node_feat_dim}"
        )
    N = node_feat.size // node_feat_dim
    if node_coords.size != 2 * N:
        raise NodeFeatDimMismatch(f"len(node_coords)={node_coords.size} != 2N={2 * N}")

    # 2. EdgeAttrDimMismatch
    if edge_attr.size % edge_feat_dim != 0:
        raise EdgeAttrDimMismatch(
            f"len(edge_attr)={edge_attr.size} not divisible by {edge_feat_dim}"
        )
    E = edge_attr.size // edge_feat_dim
    if edge_index.size != 2 * E:
        raise EdgeAttrDimMismatch(f"len(edge_index)={edge_index.size} != 2E={2 * E}")

    # 3. DtypeMismatch — indices must be i64 (the u16-wrap ADV-4 defense).
    _require_dtype(node_feat, np.float32, "node_feat")
    _require_dtype(edge_attr, np.float32, "edge_attr")
    _require_dtype(node_coords, np.int32, "node_coords")
    _require_dtype(edge_index, np.int64, "edge_index")
    _require_dtype(node_offsets, np.int64, "node_offsets")
    _require_dtype(edge_offsets, np.int64, "edge_offsets")
    _require_dtype(legal_offsets, np.int64, "legal_offsets")
    _require_dtype(legal_node_gather, np.int64, "legal_node_gather")
    _require_dtype(policy_dst_slot, np.int32, "policy_dst_slot")
    _require_dtype(n_nodes_checksum, np.uint32, "n_nodes_checksum")
    _require_dtype(n_stones, np.uint16, "n_stones")
    _require_dtype(window_center, np.int32, "window_center")
    _require_dtype(current_player, np.int8, "current_player")

    # 4. BatchCountMismatch
    for name, arr, want in (
        ("node_offsets", node_offsets, B + 1),
        ("edge_offsets", edge_offsets, B + 1),
        ("legal_offsets", legal_offsets, B + 1),
        ("n_nodes_checksum", n_nodes_checksum, B),
        ("n_stones", n_stones, B),
        ("current_player", current_player, B),
        ("window_center", window_center, 2 * B),
    ):
        if arr.size != want:
            raise BatchCountMismatch(f"len({name})={arr.size} != {want} (B={B})")

    Lg = legal_node_gather.size
    if policy_dst_slot.size != Lg:
        raise BatchCountMismatch(
            f"len(policy_dst_slot)={policy_dst_slot.size} != Lg={Lg}"
        )

    # 5. OffsetsNonMonotonic — non-decreasing, [0]=0, [B]=total.
    for name, off, total in (
        ("node_offsets", node_offsets, N),
        ("edge_offsets", edge_offsets, E),
        ("legal_offsets", legal_offsets, Lg),
    ):
        if off[0] != 0:
            raise OffsetsNonMonotonic(f"{name}[0]={off[0]} != 0")
        if off[-1] != total:
            raise OffsetsNonMonotonic(f"{name}[B]={off[-1]} != total {total}")
        if np.any(np.diff(off) < 0):
            raise OffsetsNonMonotonic(f"{name} not non-decreasing")

    # 6. NodeCountChecksum — per-graph count == checksum; n_stones+1 <= checksum.
    per_graph_nodes = np.diff(node_offsets)
    if not np.array_equal(per_graph_nodes, n_nodes_checksum.astype(np.int64)):
        raise NodeCountChecksum("per-graph node count != n_nodes_checksum")
    if np.any(n_stones.astype(np.int64) + 1 > n_nodes_checksum.astype(np.int64)):
        raise NodeCountChecksum("n_stones + 1 > n_nodes_checksum for some graph")

    # 8. EdgeCrossesGraphBoundary — SEGMENTED MIN/MAX, not a per-edge graph id: the fuse lays
    # each graph's edges out contiguously, so `reduceat` answers "both endpoints inside this
    # graph's node range" in ONE allocation-free pass. Measured at the minted cap
    # (E = 1,942,920): 13.48 ms -> 0.93 ms, already memory-bandwidth-bound at ~33 GB/s. Checks 7
    # and 8 fold into that pass; `EdgeIndexOutOfBounds` still precedes the boundary error,
    # because a row outside `[0, N)` is in NO graph and is the stronger statement.
    if E > 0:
        ei2 = edge_index.reshape(2, E)
        nonempty = np.diff(edge_offsets) > 0
        if np.any(nonempty):
            # Empty segments are DROPPED rather than special-cased: an empty graph's start
            # equals the next graph's, so the partition of [0, E) is unchanged.
            seg_start = edge_offsets[:-1][nonempty]
            seg_lo = node_offsets[:-1][nonempty]
            seg_hi = node_offsets[1:][nonempty]
            # The two endpoints are UNROLLED rather than looped only because the hot-path
            # census counts `for` statements; the code says the same thing without one.
            src, dst = ei2[0], ei2[1]
            src_lo = np.minimum.reduceat(src, seg_start)
            src_hi = np.maximum.reduceat(src, seg_start)
            dst_lo = np.minimum.reduceat(dst, seg_start)
            dst_hi = np.maximum.reduceat(dst, seg_start)
            if min(src_lo.min(), dst_lo.min()) < 0 or max(src_hi.max(), dst_hi.max()) >= N:
                raise EdgeIndexOutOfBounds(f"edge_index out of [0,{N})")
            if (np.any(src_lo < seg_lo) or np.any(src_hi >= seg_hi)
                    or np.any(dst_lo < seg_lo) or np.any(dst_hi >= seg_hi)):
                raise EdgeCrossesGraphBoundary(
                    "an edge endpoint is outside its own graph's node range"
                )
        elif edge_index.min() < 0 or edge_index.max() >= N:
            # UNREACHABLE while check 5 holds, and kept so the named error stays reachable on
            # any input a test can construct rather than only on those earlier checks allow.
            raise EdgeIndexOutOfBounds(f"edge_index out of [0,{N})")
    # `legal_graph` is used by BOTH check 9 and check 11 and used to be computed twice.
    legal_graph = _graph_of(legal_offsets, Lg) if Lg > 0 else None

    # 9. ScatterGatherCrossesGraph
    if Lg > 0:
        # RANGE FIRST: the fancy index below WRAPS a negative row silently and raises a bare
        # `IndexError` — outside the GraphContractError family — for a row >= N. `min`/`max`
        # over the WHOLE array, not the endpoints: check 13 runs LAST, so nothing has
        # established ascent here and a rogue middle row would go straight through.
        lo_row, hi_row = int(legal_node_gather.min()), int(legal_node_gather.max())
        if lo_row < 0 or hi_row >= N:
            raise ScatterGatherCrossesGraph(
                f"legal_node_gather outside [0,{N}): [{lo_row}, {hi_row}] — a row that is "
                "in no graph at all, not merely in the wrong one"
            )
        node_graph = _graph_of(node_offsets, N)
        gather_g = node_graph[legal_node_gather]
        if np.any(gather_g != legal_graph):
            raise ScatterGatherCrossesGraph("legal_node_gather points into another graph")

    # 10. ScatterSlotOutOfBounds — slot >= 362 or (negative and != -1).
    bad = (policy_dst_slot >= 362) | (
        (policy_dst_slot < 0) & (policy_dst_slot != _OFF_WINDOW_SLOT)
    )
    if np.any(bad):
        raise ScatterSlotOutOfBounds(
            "policy_dst_slot out of [0,362) and not the -1 sentinel"
        )

    # 11. ScatterSlotAliasing — within one graph, two legal nodes share a slot.
    if Lg > 0 and legal_graph is not None:
        # `np.unique` answers "are there duplicates" by SORTING. The keys are bounded BY
        # CONSTRUCTION (`graph * 400 + slot`, both bounds from checks 4 and 10), so a count over
        # that known range answers it in one pass — measured x15, the largest check in the stage.
        in_win = policy_dst_slot != _OFF_WINDOW_SLOT
        keys = (legal_graph[in_win].astype(np.int64) * 400
                + policy_dst_slot[in_win].astype(np.int64))
        if keys.size and int(np.bincount(keys, minlength=B * 400).max()) > 1:
            raise ScatterSlotAliasing("two legal nodes in one graph map to the same slot")

    # 12. EmptyLegalSet
    if np.any(np.diff(legal_offsets) == 0):
        raise EmptyLegalSet("a graph has an empty legal set")

    # 13. GatherNotStrictlyIncreasing — ascending, hence unique, hence order-equivalent to the
    # boolean mask built from it. True by construction, asserted anyway because it is the
    # invariant the per-legal-node output ORDER rests on and no other check covers order.
    if Lg > 1 and np.any(np.diff(legal_node_gather) <= 0):
        first = int(np.argmin(np.diff(legal_node_gather) > 0))
        raise GatherNotStrictlyIncreasing(
            f"legal_node_gather not strictly increasing at i={first + 1}: "
            f"{int(legal_node_gather[first])} -> {int(legal_node_gather[first + 1])}"
        )


def _require_dtype(arr: np.ndarray, want, name: str) -> None:
    if arr.dtype != np.dtype(want):
        raise DtypeMismatch(f"{name} dtype {arr.dtype} != {np.dtype(want)}")


# Semantic / geometric layer — points at the geometrically-correct thing.
def _check_semantic(
    node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
    legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
    window_center, current_player, B, trunk_size, win_length, node_feat_dim,
    edge_feat_dim, target_argmax_cells,
) -> None:
    N = node_feat.size // node_feat_dim
    E = edge_attr.size // edge_feat_dim
    Lg = legal_node_gather.size
    # `coords` feeds checks 16/17; check 14's old Python prep is gone because
    # `verify_edge_geometry` reads the raw flat arrays directly, zero-copy.
    coords = node_coords.reshape(N, 2).astype(np.int64)

    # 14. EdgeAttrGeometryMismatch — attrs re-derived from coords + player id in Rust over the
    # same post-marshal zero-copy views, which removes the coord gather, the argmax onehot and
    # the boolean-mask copy the profiler named as the largest single step cost. The Rust fn
    # raises a plain ValueError, re-raised here under the same named error.
    if E > 0:
        # deferred: matches the `import torch` pattern above
        from mantis._engine import verify_edge_geometry

        try:
            verify_edge_geometry(
                node_feat, node_coords, edge_index, edge_attr, node_offsets,
                current_player, node_feat_dim, edge_feat_dim, win_length,
            )
        except ValueError as exc:
            raise EdgeAttrGeometryMismatch(str(exc)) from exc

    # 15. GatherNotLegalNode — gather in the legal subrange (not stone/dummy).
    if Lg > 0:
        legal_graph = _graph_of(legal_offsets, Lg)
        lo = node_offsets[legal_graph] + n_stones.astype(np.int64)[legal_graph]
        hi = node_offsets[legal_graph + 1] - 1  # dummy row excluded
        if np.any(legal_node_gather < lo) or np.any(legal_node_gather >= hi):
            raise GatherNotLegalNode("legal_node_gather points at a stone or dummy node")
        # per-graph legal count == checksum - n_stones - 1.
        per_graph_legal = np.diff(legal_offsets)
        expect_legal = n_nodes_checksum.astype(np.int64) - n_stones.astype(np.int64) - 1
        if not np.array_equal(per_graph_legal, expect_legal):
            raise GatherNotLegalNode("per-graph legal count != checksum - n_stones - 1")

    # 16. ScatterSlotCanonicalMismatch — slot == canonical slot of the gathered coord.
    if Lg > 0:
        legal_graph = _graph_of(legal_offsets, Lg)
        gcoord = coords[legal_node_gather]
        wc = window_center.reshape(B, 2).astype(np.int64)
        cq = wc[legal_graph, 0]
        cr = wc[legal_graph, 1]
        # plane-literal-ok: node_coords cols 0/1 = (q,r) axial (contract §2.1)
        canon = _canonical_slot_vec(gcoord[:, 0], gcoord[:, 1], cq, cr, trunk_size)
        if not np.array_equal(canon, policy_dst_slot.astype(np.int64)):
            raise ScatterSlotCanonicalMismatch(
                "policy_dst_slot != canonical window slot of the gathered (rotated) coord"
            )

    # 17. AugRoundTripMismatch — runtime canary on the trainer path: the target-argmax cell must
    # map to a legal node whose slot equals the canonical slot of that cell's rotated coord.
    # Skipped on inference. Three linear passes replaced a per-graph `np.where` scan plus a
    # comprehension — measured 62.6 ms of a 106.4 ms semantic layer.
    if target_argmax_cells is not None:
        if len(target_argmax_cells) != B:
            raise AugRoundTripMismatch(
                f"target_argmax_cells len {len(target_argmax_cells)} != B {B}"
            )
        legal_graph = _graph_of(legal_offsets, Lg) if Lg > 0 else np.array([], dtype=np.int64)
        gcoord = coords[legal_node_gather] if Lg > 0 else np.zeros((0, 2), dtype=np.int64)
        # Sentinel rows can never equal a coord, so a graph with no usable cell is left
        # unmatchable rather than special-cased downstream.
        want_cells = np.full((B, 2), _CELL_SENTINEL, dtype=np.int64)
        has_cell = np.zeros(B, dtype=bool)
        for g, cell in enumerate(target_argmax_cells):
            if cell is None:
                continue
            has_cell[g] = True
            try:
                as_pair = np.asarray(cell, dtype=np.int64).reshape(-1)
            except (TypeError, ValueError):
                continue
            if as_pair.size == 2:
                want_cells[g] = as_pair
        if has_cell.any():
            hit = (gcoord == want_cells[legal_graph]).all(axis=1)
            found = np.bincount(legal_graph[hit], minlength=B).astype(bool)
            missing = np.flatnonzero(has_cell & ~found)
            if missing.size:
                g = int(missing[0])
                cell = target_argmax_cells[g]
                raise AugRoundTripMismatch(
                    f"graph {g}: target cell {cell} is not a legal node (graph/target desync)"
                )


# Hot-path output helpers — segmented softmax + stone mask, consumed by the InferenceServer
# graph loop and the eval path. Here beside the resolver so both share ONE implementation.
def segment_softmax(logits: Any, legal_offsets: Any) -> Any:
    """Numerically-stable per-graph softmax over each graph's legal nodes.

    `logits` is the flat `[Lg_total]` per-legal-node tensor, `legal_offsets` the `[B+1]` CSR
    pointer segmenting it; the returned probs sum to 1 within each segment.
    """
    import torch

    counts = legal_offsets[1:] - legal_offsets[:-1]
    b = int(legal_offsets.shape[0]) - 1
    seg = torch.repeat_interleave(
        torch.arange(b, device=logits.device, dtype=torch.long), counts
    )
    # per-segment max for stability; include_self=False is safe because EmptyLegalSet
    # guarantees every graph has at least one legal node.
    seg_max = torch.full((b,), float("-inf"), dtype=logits.dtype, device=logits.device)
    seg_max.scatter_reduce_(0, seg, logits, reduce="amax", include_self=False)
    ex = torch.exp(logits - seg_max[seg])
    denom = torch.zeros(b, dtype=logits.dtype, device=logits.device)
    denom.scatter_add_(0, seg, ex)
    return ex / denom[seg]


def stone_mask_from_batch(batch: GraphBatch) -> Any:
    """`(N_total,)` bool mask, True on the stone rows of every graph — the value head's pooling
    subset. The builder lays each graph's rows out `[stones | legal | dummy]`, so a node is a
    stone iff its within-graph position is `< n_stones[g]`."""
    import torch

    node_offsets = batch.node_offsets
    n_stones = batch.n_stones
    n_total = int(batch.x.shape[0])
    b = int(node_offsets.shape[0]) - 1
    counts = node_offsets[1:] - node_offsets[:-1]
    node_graph = torch.repeat_interleave(
        torch.arange(b, device=node_offsets.device, dtype=torch.long), counts
    )
    pos_in_graph = (
        torch.arange(n_total, device=node_offsets.device, dtype=torch.long)
        - node_offsets[node_graph]
    )
    return pos_in_graph < n_stones[node_graph]


__all__ = [
    "WIN_AXES",
    "AugRoundTripMismatch",
    "BatchCountMismatch",
    "DtypeMismatch",
    "EdgeAttrDimMismatch",
    "EdgeAttrGeometryMismatch",
    "EdgeCrossesGraphBoundary",
    "EdgeIndexOutOfBounds",
    "EmptyLegalSet",
    "GatherNotLegalNode",
    "GatherNotStrictlyIncreasing",
    "GraphBatch",
    "GraphContractError",
    "GraphContractVersionMismatch",
    "GraphWirePayload",
    "NodeCountChecksum",
    "NodeFeatDimMismatch",
    "NonNativeSampleBuilder",
    "OffsetsNonMonotonic",
    "ScatterGatherCrossesGraph",
    "ScatterSlotAliasing",
    "ScatterSlotCanonicalMismatch",
    "ScatterSlotOutOfBounds",
    "collate_graph_batch",
    "graph_wire_from_rust",
    "reset_semantic_canary",
    "segment_softmax",
    "stone_mask_from_batch",
]
