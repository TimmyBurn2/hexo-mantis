# >300 justify (R8). NO LINE COUNT is stated, per G-DFIX-4 and R192(e)'s derive-or-delete.
# ONE rig for the whole F-816-10 fused-inference family: the payload builder, the collate
# stand-in, the identity-keyed stub net, the scripted batcher and the config factory are a
# single apparatus — every fused-forward oracle drives the SAME `_run_graph_loop` over the
# SAME wire shape, and the round-trip claim (split == unsplit, positionally) is only
# meaningful if both sides are produced by one builder. Splitting the builder from the net
# would let the per-graph identity the net keys on drift away from the identity the builder
# stamps, and the transposition oracles would go quiet without changing a number.
"""Shared rig for the F-816-10 memory-bounded graph-inference-fusion oracles.

Written by ORACLE-WRITE **before** the feature exists. This module imports only surfaces
that are LIVE at HEAD (`graph_collate`, `graph_wire_split`'s existing names, `InferenceServer`)
so it collects and runs today; the suites that import the NOT-YET-WRITTEN names
(`plan_fused_forwards`, `FusedGraphOverCap`, `mantis.config.resolve.fused_graph_caps`) are the
ones that go RED, which is the correct oracle-first state.

WHAT IS REAL AND WHAT IS NOT. Real: `InferenceServer._run_graph_loop` (the production loop,
driven end to end), `segment_softmax`, `stone_mask_from_batch`, the finiteness gate, the
D2H copies, the submit call and — once IMPL lands — the plan, the slice and the concat.
Fake: the ARCH (an identity-keyed stub net, so a transposition is visible in the output
instead of being washed out by a trained net's near-uniform policy) and `collate_graph_batch`
(replaced by `collate_from_payload`, a faithful wire->tensor transcription; the real collate's
18-check contract is pinned by its own suites and is not what these rows are about).

THE PER-GRAPH IDENTITY IS CARRIED IN THE NODE FEATURES, NOT IN THE POSITION. `build_payload`
stamps a globally unique id into `node_feat[:, 0]`, and `SentinelGraphNet` derives both the
per-legal-node logits and the per-graph value sentinel from THAT id. A part's `node_offsets`
are re-based by `slice_graph_wire`, so any identity read off a position would differ between
the split and un-split drives for a CORRECT implementation and the round-trip oracle would
red on its own rig. The feature-borne id is invariant under the slice; the position is not.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mantis.encoding import lookup
from mantis.selfplay.graph_collate import GraphBatch, GraphWirePayload
from mantis.selfplay.inference_server import InferenceServer

GRAPH_SPEC = lookup("gnn_axis_v1")
GRID_SPEC = lookup("v6")
SEED = 20260817

#: Node-feature width the collate stand-in reshapes against (registry `node_feat_dim`).
NODE_FEAT_DIM = int(GRAPH_SPEC.node_feat_dim or 11)
EDGE_FEAT_DIM = int(GRAPH_SPEC.edge_feat_dim or 5)


# ── the wire ────────────────────────────────────────────────────────────────────────────
def build_payload(
    legal_counts: list[int],
    edges_per_graph: list[int] | None = None,
    *,
    uid_base: int = 1,
) -> GraphWirePayload:
    """A block-diagonal `GraphWirePayload` over `len(legal_counts)` graphs.

    Node layout per graph is the builder's own `[stones | legal | dummy]`
    (`graph_collate.stone_mask_from_batch` reads exactly that), with ONE stone and ONE dummy,
    so `n_nodes[g] == legal_counts[g] + 2`. `node_feat[:, 0]` carries `uid_base + global row`,
    the identity every oracle keys on.

    UNEQUAL `legal_counts` is the point (D-3): the FFI checks a per-id probs SEGMENT LENGTH
    (`inference.rs`, `policy_dst_slot.len() != leaf_probs.len()`) and nothing at all about
    `values[i]` ordering, so a transposition between two SAME-length graphs is invisible to
    every check downstream of this rig. Callers pass ragged counts on purpose.
    """
    b = len(legal_counts)
    assert b >= 1, "a payload needs at least one graph"
    n_nodes = [int(lc) + 2 for lc in legal_counts]
    if edges_per_graph is None:
        # Two dummy edges per real node is the builder's own floor
        # (`crates/mantis-graph/src/lib.rs`), so this is a shape the production wire reaches.
        edges_per_graph = [2 * (n - 1) for n in n_nodes]
    assert len(edges_per_graph) == b, "one edge count per graph"

    node_offsets = np.concatenate([[0], np.cumsum(n_nodes)]).astype(np.int64)
    edge_offsets = np.concatenate([[0], np.cumsum(edges_per_graph)]).astype(np.int64)
    legal_offsets = np.concatenate([[0], np.cumsum(legal_counts)]).astype(np.int64)
    n_total = int(node_offsets[-1])
    e_total = int(edge_offsets[-1])

    node_feat = np.zeros((n_total, NODE_FEAT_DIM), dtype=np.float32)
    node_feat[:, 0] = np.arange(uid_base, uid_base + n_total, dtype=np.float32)

    src = np.zeros(e_total, dtype=np.int64)
    dst = np.zeros(e_total, dtype=np.int64)
    gather = np.zeros(int(legal_offsets[-1]), dtype=np.int64)
    for g in range(b):
        n0, n1 = int(node_offsets[g]), int(node_offsets[g + 1])
        e0, e1 = int(edge_offsets[g]), int(edge_offsets[g + 1])
        rows = np.arange(n0, n1, dtype=np.int64)
        if e1 > e0:
            src[e0:e1] = rows[np.arange(e1 - e0) % len(rows)]
            dst[e0:e1] = rows[(np.arange(e1 - e0) + 1) % len(rows)]
        l0, l1 = int(legal_offsets[g]), int(legal_offsets[g + 1])
        # Legal rows sit AFTER the single stone row and BEFORE the single dummy row.
        gather[l0:l1] = np.arange(n0 + 1, n0 + 1 + (l1 - l0), dtype=np.int64)

    return GraphWirePayload(
        contract_version=1,
        builder_impl=1,
        n_graphs=b,
        node_feat=node_feat.reshape(-1),
        node_coords=np.zeros(n_total * 2, dtype=np.int64),
        edge_index=np.concatenate([src, dst]).astype(np.int64),
        edge_attr=np.zeros(e_total * EDGE_FEAT_DIM, dtype=np.float32),
        node_offsets=node_offsets,
        edge_offsets=edge_offsets,
        legal_offsets=legal_offsets,
        legal_node_gather=gather,
        policy_dst_slot=np.arange(int(legal_offsets[-1]), dtype=np.int64),
        n_nodes_checksum=np.asarray(n_nodes, dtype=np.int64),
        n_stones=np.ones(b, dtype=np.int64),
        window_center=np.zeros(b * 2, dtype=np.int64),
        current_player=np.ones(b, dtype=np.int64),
    )


def per_graph_counts(payload: GraphWirePayload) -> tuple[np.ndarray, np.ndarray]:
    """`(edge_counts, node_counts)` per graph — the two quantities the caps bound."""
    return (np.diff(np.asarray(payload.edge_offsets, dtype=np.int64)),
            np.diff(np.asarray(payload.node_offsets, dtype=np.int64)))


# ── the collate stand-in ────────────────────────────────────────────────────────────────
def collate_from_payload(wire: Any, *_a: Any, **_kw: Any) -> GraphBatch:
    """Transcribe a `GraphWirePayload` (whole or SLICED) into a `GraphBatch`.

    Accepts every extra kwarg the production call site passes so it is a drop-in for
    `collate_graph_batch` under `monkeypatch.setattr`. It reads only fields the slice
    re-bases, which is what makes it a faithful stand-in for the split path: a slice that
    forgot to re-base `legal_node_gather` or `edge_index` produces a batch this function
    builds WRONG, and the round-trip oracle sees it.
    """
    no = np.asarray(wire.node_offsets, dtype=np.int64)
    eo = np.asarray(wire.edge_offsets, dtype=np.int64)
    lo = np.asarray(wire.legal_offsets, dtype=np.int64)
    n_total, e_total, b = int(no[-1]), int(eo[-1]), int(no.shape[0]) - 1
    gather = np.asarray(wire.legal_node_gather, dtype=np.int64)

    legal_mask = torch.zeros(n_total, dtype=torch.bool)
    legal_mask[torch.from_numpy(gather.copy())] = True
    return GraphBatch(
        x=torch.from_numpy(
            np.ascontiguousarray(wire.node_feat, dtype=np.float32).reshape(n_total, -1)
        ),
        edge_index=torch.from_numpy(
            np.ascontiguousarray(wire.edge_index, dtype=np.int64).reshape(2, e_total)
        ),
        edge_attr=torch.from_numpy(
            np.ascontiguousarray(wire.edge_attr, dtype=np.float32).reshape(e_total, -1)
        ),
        legal_offsets=torch.from_numpy(lo.copy()),
        legal_node_gather=torch.from_numpy(gather.copy()),
        node_offsets=torch.from_numpy(no.copy()),
        n_stones=torch.from_numpy(np.ascontiguousarray(wire.n_stones, dtype=np.int64)),
        n_graphs=b,
        device="cpu",
    )


# ── the identity-keyed stub net ─────────────────────────────────────────────────────────
class SentinelGraphNet(torch.nn.Module):
    """Finite outputs keyed on the node-feature uid, so a transposition is VISIBLE.

    `policy_logits[j]` is a pseudo-random function of legal node `j`'s own uid, so two graphs
    of the SAME legal count still get different (shift-invariant) softmax segments — a
    same-length swap is caught, not only a ragged one. `values[g]` is an affine function of
    graph `g`'s FIRST node uid, giving each graph its own value sentinel; nothing downstream
    of the server checks `values[i]` ordering at all (review Finding 9), so this is the only
    instrument on that axis.
    """

    def __init__(self, *, oom_on_call: int | None = None) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple[int, int]] = []
        self._oom_on_call = oom_on_call

    def forward_batch(self, x, edge_index, edge_attr, legal_index, stone_mask, node_offsets):
        self.calls.append((int(x.shape[0]), int(edge_index.shape[1])))
        if self._oom_on_call is not None and len(self.calls) == self._oom_on_call:
            raise torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 1.72 GiB (simulated)"
            )
        uids = x[:, 0].to(torch.float64)
        legal_uids = uids.index_select(0, legal_index)  # R284 P-MASK: the wire's gather
        logits = 0.31 * ((legal_uids * 13.0 + 5.0) % 17.0)
        b = int(node_offsets.shape[0]) - 1
        first = uids[node_offsets[:-1].to(torch.long)]
        values = (-0.5 + 0.001 * first).reshape(b, 1)
        return logits.to(torch.float32), values.to(torch.float32), torch.zeros(b, 65)


# ── the scripted batcher ────────────────────────────────────────────────────────────────
class ScriptedGraphBatcher:
    """Serve `pops` (each a `GraphWirePayload`) once each, then stop the loop.

    Mirrors `tests/selfplay/test_inference_batch_timing.py::_FakeGraphBatcher`, but the wire
    it hands back is a REAL payload rather than an opaque sentinel, because the split reads
    the wire's own CSR offsets to plan.
    """

    def __init__(self, pops: list[GraphWirePayload]) -> None:
        self._pops = list(pops)
        self.server: InferenceServer | None = None
        self.results: list[tuple] = []
        self.failures: list[tuple[list[int], str]] = []
        self.closed = 0

    def next_graph_batch(self, batch_size: int, max_wait_ms: float):
        if not self._pops:
            assert self.server is not None, "the batcher must be bound to its server"
            self.server._stop_event.set()
            return [], None
        payload = self._pops.pop(0)
        return list(range(1, int(payload.n_graphs) + 1)), payload

    def submit_graph_inference_results(self, ids, probs, offsets, values) -> None:
        self.results.append((list(ids), np.asarray(probs), np.asarray(offsets),
                             np.asarray(values)))

    def submit_graph_inference_failure(self, ids, error_msg: str) -> None:
        self.failures.append((list(ids), error_msg))

    def bump_model_version(self) -> int:
        return 1

    def close(self) -> None:
        self.closed += 1


# ── the config ──────────────────────────────────────────────────────────────────────────
def graph_cfg(
    max_fused_edges: int | None = 10_000_000,
    max_fused_nodes: int | None = 1_000_000,
    *,
    omit_block: bool = False,
    batch_size: int = 64,
    **over: Any,
) -> dict[str, Any]:
    """The `InferenceServer` config dict, with the NEW `inference.fused_graph_caps` block.

    `omit_block=True` returns the HEAD shape — the block absent entirely — which is what the
    LAW-11 absence rows drive. The defaults are deliberately far above anything this rig can
    build, so a caller that does not ask for a split does not get one by accident (MB-24's
    posture applied to the oracle rig).
    """
    inference: dict[str, Any] = {
        "inference_batch_size": batch_size, "inference_max_wait_ms": 20.0,
        "trace_inference": False, "compile_inference": False,
        "compile_inference_mode": "default", "compile_inference_dynamic": True,
        "perf_timing": False, "perf_sync_cuda": False,
    }
    if not omit_block:
        inference["fused_graph_caps"] = {
            "max_fused_edges": max_fused_edges, "max_fused_nodes": max_fused_nodes,
        }
    inference.update(over)
    return {"inference": inference, "encoding": "gnn_axis_v1",
            "train": {"amp_dtype": "bf16"}}


def grid_cfg(*, omit_block: bool = True, **over: Any) -> dict[str, Any]:
    cfg = graph_cfg(omit_block=omit_block, **over)
    cfg["encoding"] = "v6"
    cfg["train"]["amp_dtype"] = "fp16"
    return cfg


# ── the drive ───────────────────────────────────────────────────────────────────────────
def drive_one_pop(
    monkeypatch: Any,
    payload: GraphWirePayload,
    *,
    max_fused_edges: int | None = 10_000_000,
    max_fused_nodes: int | None = 1_000_000,
    net: torch.nn.Module | None = None,
    batch_size: int = 64,
) -> tuple[InferenceServer, ScriptedGraphBatcher, torch.nn.Module]:
    """Run the REAL `_run_graph_loop` over exactly one pop of `payload`.

    Returns `(server, batcher, net)` so a row can read the submitted results, the failures,
    the counters and the per-forward call log off one drive.
    """
    import mantis.selfplay.graph_collate as collate_mod

    monkeypatch.setattr(collate_mod, "collate_graph_batch", collate_from_payload)
    model = net if net is not None else SentinelGraphNet()
    batcher = ScriptedGraphBatcher([payload])
    server = InferenceServer(
        model, torch.device("cpu"),
        graph_cfg(max_fused_edges, max_fused_nodes, batch_size=batch_size),
        batcher=batcher, encoding_spec=GRAPH_SPEC,
    )
    batcher.server = server
    server.run()
    return server, batcher, model
