"""The SINGLE wire reader for the GNN ragged-payload contract v1.

>300 justify: the named contract errors, the resolver, the structural + semantic check layers
and the two hot-path output helpers are ONE contract (`docs/contracts/graph_wire.md`).
Splitting them would break the "one place asserts the wire" property the ADV suite gates.

`collate_graph_batch` is the one-and-only consumer of the block-diagonal wire emitted by the
Rust `InferenceBatcher.next_graph_batch`, imported by BOTH the self-play hot path and the
promotion-gate eval path, and import-safe with no module-scope torch. It asserts the contract
version and the native-builder handshake, runs the structural checks (always full; 4-13 in the
Rust pack that fills ONE host block) and the semantic ones (canary on the hot path), then ships
the block. Every mismatch raises a NAMED error; there is no silent fixed-width fallback anywhere.
The OUTPUT is not a dense scatter — the InferenceServer segment-softmaxes and returns ragged probs.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

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


#: The classes the Rust pack names its refusals by: checks 4-13, and check 14's for a coded row outside the vocabulary.
_PACK_ERRORS: dict[str, type[GraphContractError]] = {
    cls.__name__: cls for cls in (
        BatchCountMismatch, OffsetsNonMonotonic, NodeCountChecksum, EdgeIndexOutOfBounds,
        EdgeCrossesGraphBoundary, ScatterGatherCrossesGraph, ScatterSlotOutOfBounds,
        ScatterSlotAliasing, EmptyLegalSet, GatherNotStrictlyIncreasing, EdgeAttrGeometryMismatch,
    )
}


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


@dataclass(frozen=True)
class EdgeGeometryCheck:
    """Check 14's call, captured: run inline or handed to a checker thread, never rebuilt."""

    node_feat: np.ndarray
    node_coords: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_offsets: np.ndarray
    current_player: np.ndarray
    node_feat_dim: int
    edge_feat_dim: int
    win_length: int

    def run(self) -> None:
        """Re-derive every edge attribute in Rust; raises `EdgeAttrGeometryMismatch` by name."""
        from mantis._engine import verify_edge_geometry

        try:
            verify_edge_geometry(
                self.node_feat, self.node_coords, self.edge_index, self.edge_attr,
                self.node_offsets, self.current_player, self.node_feat_dim,
                self.edge_feat_dim, self.win_length,
            )
        except ValueError as exc:
            raise EdgeAttrGeometryMismatch(str(exc)) from exc


@dataclass
class GraphBatch:
    """Collated block-diagonal torch tensors feeding `GnnNet.forward_batch`, plus the fields
    the ragged OUTPUT assemble needs.

    `node_coords` is deliberately absent: the DEVICE tensor had zero reads, so it was an H2D
    transfer per part that nothing read. The WIRE array of the same name is live.
    """

    x: Any  # torch.Tensor (N, 11) float
    edge_index: Any  # (2, E) int64
    edge_attr: Any  # (E, 5) float; None from a coded collate
    legal_offsets: Any  # (B+1,) int64
    legal_node_gather: Any  # (Lg,) int64 (global rows)
    node_offsets: Any  # (B+1,) int64
    n_stones: Any  # (B,) int64
    n_graphs: int = 0
    device: str = "cpu"
    extra: dict = field(default_factory=dict)
    edge_code: Any = None  # (E,) uint8 rows of the edge vocabulary, from a coded collate


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


@dataclass(frozen=True)
class StagedGraphBatch:
    """One wire checked and packed into its host block (pinned for CUDA); `ship_graph_batch` puts it on the device."""

    staged: list[Any]
    n_graphs: int
    n_nodes: int
    n_edges: int
    node_feat_dim: int
    edge_feat_dim: int
    coded_edges: bool


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
    deferred_edge_geometry: list[EdgeGeometryCheck] | None = None,
    coded_edges: bool = False,
    pad_to: tuple[int, int, int, int] | None = None,
    device_out: Sequence[Any] | None = None,
) -> GraphBatch:
    """`stage_graph_batch` then `ship_graph_batch` onto `device` or into `device_out`; Raises: GraphContractError, RuntimeError (as staged)."""
    import torch  # deferred: keeps this module import-safe in torch-free envs

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    staged = stage_graph_batch(
        wire, expected_version, trunk_size=trunk_size, win_length=win_length,
        node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim, device=device, semantic=semantic,
        canary_period=canary_period, allow_oracle_builder=allow_oracle_builder,
        target_argmax_cells=target_argmax_cells, deferred_edge_geometry=deferred_edge_geometry,
        coded_edges=coded_edges, pad_to=pad_to)
    return ship_graph_batch(staged, device, device_out)


def stage_graph_batch(
    wire: Any,
    expected_version: int = 1,
    *,
    trunk_size: int,
    win_length: int,
    node_feat_dim: int,
    edge_feat_dim: int,
    device: str,
    semantic: str = "full",
    canary_period: int = 64,
    allow_oracle_builder: bool = False,
    target_argmax_cells: Sequence[tuple[int, int] | None] | None = None,
    deferred_edge_geometry: list[EdgeGeometryCheck] | None = None,
    coded_edges: bool = False,
    pad_to: tuple[int, int, int, int] | None = None,
) -> StagedGraphBatch:
    """Validate one block-diagonal graph wire and pack it into a host block, touching no device.

    `semantic`: "full" (trainer), "canary" (hot path — first + every Nth) or "off". The
    structural layer always runs full, and any mismatch raises a NAMED `GraphContractError`.
    `deferred_edge_geometry`: a sink for check 14 — when given, the check is appended to it
    instead of run, exactly when it would have run, for the caller's checker thread. `coded_edges`: each edge's vocabulary code, not its row; `pad_to` `(graphs, nodes, edges, legal)`: one padding graph to those sizes; `device`: whose type pins the block.

    THE FOUR GEOMETRY PARAMETERS ARE REQUIRED: they are the EXPECTED geometry the wire is
    checked against, so a default is a silent expectation and a payload re-captured at another
    radius would collate under stale geometry with nothing red.

    Raises:
        GraphContractError: any wire array disagrees with the declared geometry or the structural contract.
        RuntimeError: the staged block and the wire disagree in the Rust pack, a wiring break, never a contract error.
    """
    import torch  # deferred: keeps this module import-safe in torch-free envs

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
    node_feat = _flat(wire.node_feat)
    node_coords = _flat(wire.node_coords)
    edge_index = _flat(wire.edge_index)
    edge_attr = _flat(wire.edge_attr)
    node_offsets = _flat(wire.node_offsets)
    edge_offsets = _flat(wire.edge_offsets)
    legal_offsets = _flat(wire.legal_offsets)
    legal_node_gather = _flat(wire.legal_node_gather)
    policy_dst_slot = _flat(wire.policy_dst_slot)
    n_nodes_checksum = _flat(wire.n_nodes_checksum)
    n_stones = _flat(wire.n_stones)
    window_center = _flat(wire.window_center)
    current_player = _flat(wire.current_player)
    B = int(wire.n_graphs)

    # --- resolver step 3a: STRUCTURAL layer (13) — always full; 4-13 run in the pack ---
    N, E = _check_wire_shape(
        node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
        legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
        window_center, current_player, B, node_feat_dim, edge_feat_dim,
    )
    pinned = torch.device(device).type == "cuda"
    B_out, N_out, E_out, L_out = (B, N, E, legal_node_gather.size) if pad_to is None else pad_to
    staged = _stage_block((
        (N_out * node_feat_dim, torch.float32), (2 * E_out, torch.int64),
        (E_out, torch.uint8) if coded_edges else (E_out * edge_feat_dim, torch.float32),
        (B_out + 1, torch.int64), (L_out, torch.int64), (B_out + 1, torch.int64), (B_out, torch.int64),
    ), pinned)
    out = _numpy_views(staged)
    refusal = _collate_pack(
        B, node_feat, edge_index, edge_attr, node_offsets, edge_offsets, legal_offsets,
        legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones, window_center,
        current_player, out[0], out[1], None if coded_edges else out[2],
        out[2] if coded_edges else None, out[3], out[4], out[5], out[6], node_feat_dim,
        edge_feat_dim, win_length, _PACK_THREADS, pad_to,
    )
    if refusal is not None:
        if refusal[0] == EdgeAttrGeometryMismatch.__name__:
            # Check 14 names the row's geometric fault; the vocabulary only knows the row has no code.
            EdgeGeometryCheck(node_feat, node_coords, edge_index, edge_attr, node_offsets, current_player,
                              node_feat_dim, edge_feat_dim, win_length).run()
        raise _PACK_ERRORS[refusal[0]](refusal[1])

    # --- resolver step 3b: SEMANTIC/GEOMETRIC layer (4) — mode-gated ---
    run_semantic = semantic == "full" or (
        semantic == "canary" and _canary_should_run(canary_period)
    )
    if run_semantic:
        _check_semantic(
            node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
            legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
            window_center, current_player, B, trunk_size, win_length, node_feat_dim,
            edge_feat_dim, target_argmax_cells, deferred_edge_geometry,
        )

    return StagedGraphBatch(staged=staged, n_graphs=B_out, n_nodes=N_out, n_edges=E_out,
                            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim, coded_edges=coded_edges)


def ship_graph_batch(s: StagedGraphBatch, device: str, device_out: Sequence[Any] | None = None) -> GraphBatch:
    """Resolver step 4: the staged arrays on `device` or copied into `device_out`; Raises: RuntimeError, OutOfMemoryError."""
    x, ei, edges, lo, lg, no, ns = _ship(s.staged, device, device_out)
    return GraphBatch(
        x=x.reshape(s.n_nodes, s.node_feat_dim),
        edge_index=ei.reshape(2, s.n_edges),
        edge_attr=None if s.coded_edges else edges.reshape(s.n_edges, s.edge_feat_dim),
        edge_code=edges if s.coded_edges else None,
        legal_offsets=lo,
        legal_node_gather=lg,
        node_offsets=no,
        n_stones=ns,
        n_graphs=s.n_graphs,
        device=device,
    )


#: Every staged segment starts on this boundary, so each typed view is aligned for its dtype and its DMA.
_STAGE_ALIGN = 64
#: The pack's edge-copy threads beside the caller (1 below its edge threshold): best of 1/4/8 measured under load.
_PACK_THREADS = 4


def _flat(arr: Any) -> np.ndarray:
    """`arr` as the contiguous 1-D view the pack reads (no copy when it already is one)."""
    return np.ascontiguousarray(arr).reshape(-1)


def _stage_block(sizes: tuple[tuple[int, Any], ...], pinned: bool) -> list[Any]:
    """Typed views, in order, of ONE fresh host block sized for `(numel, dtype)` each; pinned for a CUDA part."""
    import torch

    offsets: list[int] = []
    total = 0
    for numel, dtype in sizes:
        offsets.append(total)
        total += -(-numel * dtype.itemsize // _STAGE_ALIGN) * _STAGE_ALIGN
    # The caching host allocator holds a pinned block until every DMA queued from it has completed.
    block = torch.empty(total, dtype=torch.uint8, pin_memory=pinned)
    return [block[o:o + n * d.itemsize].view(d) for o, (n, d) in zip(offsets, sizes, strict=True)]


def _numpy_views(staged: list[Any]) -> list[np.ndarray]:
    """The staged views as writable numpy arrays over the same bytes, for the pack to fill."""
    return [t.numpy() for t in staged]


def _ship(staged: list[Any], device: str, out: Sequence[Any] | None = None) -> list[Any]:
    """The staged arrays on `device` (one queued DMA each from the pinned block on CUDA; the views themselves on CPU), or copied into `out`."""
    if out is not None:
        return [dst.copy_(src, non_blocking=True) for src, dst in zip(staged, out, strict=True)]
    return [t.to(device, non_blocking=True) for t in staged]


def ship_host_arrays(arrays: Sequence[np.ndarray], device: str) -> list[Any]:
    """Host arrays on `device` through ONE staged block, pinned for CUDA, each copy queued with no host wait."""
    import torch

    flat = [np.ascontiguousarray(a).reshape(-1) for a in arrays]
    staged = _stage_block(tuple((a.size, torch.from_numpy(a[:0]).dtype) for a in flat),
                          torch.device(device).type == "cuda")
    for view, a in zip(_numpy_views(staged), flat, strict=True):
        view[...] = a
    return [t.reshape(np.shape(a)) for t, a in zip(_ship(staged, device), arrays, strict=True)]


def _collate_pack(*args: Any) -> tuple[str, str] | None:
    """Checks 4-13 and the pack, in Rust with the GIL released; a refusal is `(class name, message)`."""
    from mantis._engine import collate_pack

    return collate_pack(*args)


# Structural layer, checks 1-3: the dims and dtypes every later check and the pack index by.
def _check_wire_shape(
    node_feat: np.ndarray, node_coords: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray,
    node_offsets: np.ndarray, edge_offsets: np.ndarray, legal_offsets: np.ndarray,
    legal_node_gather: np.ndarray, policy_dst_slot: np.ndarray, n_nodes_checksum: np.ndarray,
    n_stones: np.ndarray, window_center: np.ndarray, current_player: np.ndarray, B: int,
    node_feat_dim: int, edge_feat_dim: int,
) -> tuple[int, int]:
    """`(N, E)` once checks 1-3 hold and `B` sizes arrays; each failure raises its named class."""
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
    if B < 0:
        raise BatchCountMismatch(f"n_graphs={B} sizes no array")
    return N, E


def _require_dtype(arr: np.ndarray, want, name: str) -> None:
    if arr.dtype != np.dtype(want):
        raise DtypeMismatch(f"{name} dtype {arr.dtype} != {np.dtype(want)}")


# Semantic / geometric layer — points at the geometrically-correct thing.
def _check_semantic(
    node_feat, node_coords, edge_index, edge_attr, node_offsets, edge_offsets,
    legal_offsets, legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones,
    window_center, current_player, B, trunk_size, win_length, node_feat_dim,
    edge_feat_dim, target_argmax_cells, deferred: list[EdgeGeometryCheck] | None = None,
) -> None:
    N = node_feat.size // node_feat_dim
    E = edge_attr.size // edge_feat_dim
    Lg = legal_node_gather.size
    # `coords` feeds checks 16/17; check 14's old Python prep is gone because
    # `verify_edge_geometry` reads the raw flat arrays directly, zero-copy.
    coords = node_coords.reshape(N, 2).astype(np.int64)

    # 14. EdgeAttrGeometryMismatch — attrs re-derived from coords + player id in Rust over the
    # same post-marshal zero-copy views. A `deferred` sink CAPTURES the call for the caller's
    # checker thread instead of running it here; it still runs on every batch.
    if E > 0:
        check = EdgeGeometryCheck(
            node_feat, node_coords, edge_index, edge_attr, node_offsets, current_player,
            node_feat_dim, edge_feat_dim, win_length,
        )
        if deferred is None:
            check.run()
        else:
            deferred.append(check)

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
def segment_ids(legal_offsets: Any, *, total: int | None = None) -> Any:
    """The `[Lg_total]` graph id of every legal node, from the `[B+1]` CSR `legal_offsets`."""
    import torch

    counts = legal_offsets[1:] - legal_offsets[:-1]
    b = int(legal_offsets.shape[0]) - 1
    return torch.repeat_interleave(
        torch.arange(b, device=legal_offsets.device, dtype=torch.long), counts, output_size=total,
    )


def segment_sum(values: Any, legal_offsets: Any) -> Any:
    """Per-graph sums of flat per-legal-node `values` over the `[B+1]` CSR `legal_offsets`, `[B]`, in a fixed order."""
    from mantis.model.gnn import segment_sums

    return segment_sums(values, legal_offsets)


def segment_softmax(logits: Any, legal_offsets: Any) -> Any:
    """Numerically-stable per-graph softmax over each graph's legal nodes.

    `logits` is the flat `[Lg_total]` per-legal-node tensor, `legal_offsets` the `[B+1]` CSR
    pointer segmenting it; the returned probs sum to 1 within each segment.
    """
    import torch

    b = int(legal_offsets.shape[0]) - 1
    seg = segment_ids(legal_offsets, total=int(logits.shape[0]))
    # per-segment max for stability; include_self=False is safe because EmptyLegalSet
    # guarantees every graph has at least one legal node.
    seg_max = torch.full((b,), float("-inf"), dtype=logits.dtype, device=logits.device)
    seg_max.scatter_reduce_(0, seg, logits, reduce="amax", include_self=False)
    ex = torch.exp(logits - seg_max[seg])
    return ex / segment_sum(ex, legal_offsets)[seg]


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
        torch.arange(b, device=node_offsets.device, dtype=torch.long), counts,
        output_size=n_total,
    )
    pos_in_graph = (
        torch.arange(n_total, device=node_offsets.device, dtype=torch.long)
        - node_offsets[node_graph]
    )
    return pos_in_graph < n_stones[node_graph]


__all__ = [
    "AugRoundTripMismatch",
    "BatchCountMismatch",
    "DtypeMismatch",
    "EdgeAttrDimMismatch",
    "EdgeAttrGeometryMismatch",
    "EdgeCrossesGraphBoundary",
    "EdgeGeometryCheck",
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
    "StagedGraphBatch",
    "collate_graph_batch",
    "graph_wire_from_rust",
    "reset_semantic_canary",
    "segment_ids",
    "segment_softmax",
    "segment_sum",
    "ship_graph_batch",
    "ship_host_arrays",
    "stage_graph_batch",
    "stone_mask_from_batch",
]
