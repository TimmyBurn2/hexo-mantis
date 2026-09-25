# >300 justify (R8): one rig — the three graph-inference suites share these fakes, and a copy
# per suite is exactly the drift the module exists to prevent.
"""Shared rig for the memory-bounded graph-inference-fusion oracles.

Imports only surfaces LIVE at HEAD, so the rig cannot mask an import regression the suites
would otherwise catch. Real:
`InferenceServer._run_graph_loop` driven end to end, `segment_softmax`,
`stone_mask_from_batch`, the finiteness gate, the D2H copies and the submit call. Fake: the ARCH
(an identity-keyed stub net, so a transposition is visible instead of washed out) and
`collate_graph_batch` (a faithful wire->tensor transcription).

THE PER-GRAPH IDENTITY IS CARRIED IN THE NODE FEATURES, NOT IN THE POSITION: a part's
`node_offsets` are re-based by `slice_graph_wire`, so an identity read off a position would
differ between the split and un-split drives for a CORRECT implementation.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from _fused_caps import CAPS_DICT
from mantis.encoding import lookup
from mantis.selfplay.graph_collate import GraphBatch, GraphWirePayload
from mantis.selfplay.inference_server import InferenceServer
from mantis.selfplay.pool_hooks import batch_fill_pct, inference_batch_timing
from mantis.train.events import emit_iteration_complete_event

GRAPH_SPEC = lookup("gnn_axis_v1")
SEED = 20260817


def graph_dim(spec: Any, field: str) -> int:
    """One graph dim off the registry row. Raises: LookupError when the row does not carry it."""
    value = getattr(spec, field, None)
    if value is None:
        raise LookupError(f"registry row {spec.name!r} carries no {field}: not a graph row")
    return int(value)


NODE_FEAT_DIM = graph_dim(GRAPH_SPEC, "node_feat_dim")
EDGE_FEAT_DIM = graph_dim(GRAPH_SPEC, "edge_feat_dim")


def build_payload(
    legal_counts: list[int],
    edges_per_graph: list[int] | None = None,
    *,
    uid_base: int = 1,
) -> GraphWirePayload:
    """A block-diagonal `GraphWirePayload` over `len(legal_counts)` graphs.

    Node layout per graph is the builder's own `[stones | legal | dummy]` with ONE stone and ONE
    dummy, and `node_feat[:, 0]` carries the identity every oracle keys on. UNEQUAL
    `legal_counts` is the point: the FFI checks a per-id probs SEGMENT LENGTH and nothing about
    `values[i]` ordering, so a transposition between SAME-length graphs is invisible downstream.
    """
    b = len(legal_counts)
    assert b >= 1, "a payload needs at least one graph"
    n_nodes = [int(lc) + 2 for lc in legal_counts]
    if edges_per_graph is None:
        # Two dummy edges per real node is the builder's own floor, so the production wire
        # reaches this shape.
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


def collate_from_payload(wire: Any, *_a: Any, **_kw: Any) -> GraphBatch:
    """Transcribe a `GraphWirePayload` (whole or SLICED) into a `GraphBatch`.

    It reads only fields the slice re-bases, so a slice that forgot to re-base
    `legal_node_gather` or `edge_index` builds a WRONG batch the round-trip sees.
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


class SentinelGraphNet(torch.nn.Module):
    """Finite outputs keyed on the node-feature uid, so a transposition is VISIBLE.

    `policy_logits[j]` is pseudo-random in legal node `j`'s uid, so two graphs of the SAME legal
    count get different softmax segments; `values[g]` is affine in graph `g`'s FIRST node uid,
    and nothing downstream of the server checks `values[i]` ordering at all.
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
        legal_uids = uids.index_select(0, legal_index)  # the wire's own gather
        logits = 0.31 * ((legal_uids * 13.0 + 5.0) % 17.0)
        b = int(node_offsets.shape[0]) - 1
        first = uids[node_offsets[:-1].to(torch.long)]
        values = (-0.5 + 0.001 * first).reshape(b, 1)
        return logits.to(torch.float32), values.to(torch.float32), torch.zeros(b, 65)


class ScriptedGraphBatcher:
    """Serve `pops` once each, then stop the loop. The wire is a REAL payload, because the split
    reads its CSR offsets to plan."""

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


def graph_cfg(
    max_fused_edges: int | None = 10_000_000,
    max_fused_nodes: int | None = 1_000_000,
    *,
    omit_block: bool = False,
    batch_size: int = 64,
    **over: Any,
) -> dict[str, Any]:
    """The `InferenceServer` config dict with the `inference.fused_graph_caps` block;
    `omit_block=True` gives the block-absent shape, and the defaults sit far above anything this
    rig can build.
    """
    inference: dict[str, Any] = {
        "inference_batch_size": batch_size, "inference_max_wait_ms": 20.0,
    }
    if not omit_block:
        inference["fused_graph_caps"] = {
            "max_fused_edges": max_fused_edges, "max_fused_nodes": max_fused_nodes,
        }
    inference.update(over)
    return {"inference": inference, "encoding": "gnn_axis_v1"}


def drive_one_pop(
    monkeypatch: Any,
    payload: GraphWirePayload,
    *,
    max_fused_edges: int | None = 10_000_000,
    max_fused_nodes: int | None = 1_000_000,
    net: torch.nn.Module | None = None,
    batch_size: int = 64,
) -> tuple[InferenceServer, ScriptedGraphBatcher, torch.nn.Module]:
    """Run the REAL `_run_graph_loop` over one pop, returning `(server, batcher, net)`."""
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


def server_cfg(**over: Any) -> dict[str, Any]:
    # `from_config` reads `config["inference"]` and requires an explicit `encoding` key; the
    # pair is non-binding by construction here, so nothing splits on it.
    base = {
        "inference_batch_size": 8, "inference_max_wait_ms": 20.0,
        "fused_graph_caps": CAPS_DICT,
    }
    base.update(over)
    return {"inference": base, "encoding": "gnn_axis_v1"}


def wire_for(n_graphs: int = 2, nodes_per_graph: int = 3, legal_per_graph: int = 2
             ) -> GraphWirePayload:
    """A REAL `GraphWirePayload` whose CSR offsets match `hand_built_batch`'s shape; the loop
    reads those offsets before any collate runs, so only they have to be real."""
    nodes = n_graphs * nodes_per_graph
    return GraphWirePayload(
        contract_version=1, builder_impl=1, n_graphs=n_graphs,
        node_feat=np.zeros(nodes * 11, dtype=np.float32),
        node_coords=np.zeros(nodes * 2, dtype=np.int64),
        edge_index=np.zeros(0, dtype=np.int64),
        edge_attr=np.zeros(0, dtype=np.float32),
        node_offsets=np.arange(0, nodes + 1, nodes_per_graph, dtype=np.int64),
        edge_offsets=np.zeros(n_graphs + 1, dtype=np.int64),
        legal_offsets=np.arange(0, n_graphs * legal_per_graph + 1, legal_per_graph,
                                dtype=np.int64),
        legal_node_gather=np.zeros(n_graphs * legal_per_graph, dtype=np.int64),
        policy_dst_slot=np.zeros(n_graphs * legal_per_graph, dtype=np.int64),
        n_nodes_checksum=np.full(n_graphs, nodes_per_graph, dtype=np.int64),
        n_stones=np.ones(n_graphs, dtype=np.int64),
        window_center=np.zeros(n_graphs * 2, dtype=np.int64),
        current_player=np.ones(n_graphs, dtype=np.int64),
    )


def hand_built_batch(n_graphs: int = 2, nodes_per_graph: int = 3) -> GraphBatch:
    """A minimal VALID collated batch, without a live Rust queue."""
    n = n_graphs * nodes_per_graph
    node_offsets = torch.arange(0, n + 1, nodes_per_graph, dtype=torch.int64)
    legal_offsets = torch.arange(0, 2 * n_graphs + 1, 2, dtype=torch.int64)
    return GraphBatch(
        x=torch.zeros(n, 11, dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.int64),
        edge_attr=torch.zeros((0, 5), dtype=torch.float32),
        legal_offsets=legal_offsets,
        # The legal rows are 1 and 2 of each graph, ascending across the fuse; all zeros
        # would pass only because the stub nets read `.numel()`.
        legal_node_gather=torch.tensor(
            [g * nodes_per_graph + k for g in range(n_graphs) for k in (1, 2)],
            dtype=torch.int64,
        ),
        node_offsets=node_offsets,
        n_stones=torch.ones(n_graphs, dtype=torch.int64),
        n_graphs=n_graphs,
        device="cpu",
    )


class FiniteGraphNet(torch.nn.Module):
    """Stub graph net: finite per-legal-node logits + per-graph values, recording every call
    and able to poison its first logit with NaN."""

    def __init__(self, *, nonfinite: bool = False) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.nonfinite = nonfinite
        self.calls: list[tuple[int, ...]] = []

    def forward_batch(self, x, edge_index, edge_attr, legal_index, stone_mask, node_offsets):
        self.calls.append(tuple(x.shape))
        n_legal = int(legal_index.numel())  # rows, not a dense mask
        b = int(node_offsets.shape[0]) - 1
        logits = torch.zeros(n_legal, dtype=torch.float32)
        value = torch.zeros(b, 1, dtype=torch.float32)
        if self.nonfinite:
            logits[0] = float("nan")
        return logits, value, torch.zeros(b, 65, dtype=torch.float32)


class CountingGraphBatcher:
    """Drives `_run_graph_loop` over scripted per-pop request counts, sleeping `wait_s` inside
    every pop so the measured collector wait has a known lower bound."""

    def __init__(self, wire: Any, counts: list[int], wait_s: float = 0.0) -> None:
        self._wire = wire
        self._counts = list(counts)
        self._wait_s = wait_s
        self.server: InferenceServer | None = None
        self.results: list[tuple] = []
        self.failures: list[tuple[list[int], str]] = []
        self.closed = 0
        self.model_version = 0

    def next_graph_batch(self, batch_size: int, max_wait_ms: float):
        if self._wait_s:
            time.sleep(self._wait_s)
        if not self._counts:
            assert self.server is not None
            self.server._stop_event.set()
            return [], None
        return list(range(1, self._counts.pop(0) + 1)), self._wire

    def submit_graph_inference_results(self, ids, probs, offsets, values) -> None:
        self.results.append((list(ids), probs, offsets, values))

    def submit_graph_inference_failure(self, ids, error_msg: str) -> None:
        self.failures.append((list(ids), error_msg))

    def bump_model_version(self) -> int:
        self.model_version += 1
        return self.model_version

    def close(self) -> None:
        self.closed += 1


class ListSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class TelemetryPool:
    """The narrow `PoolTelemetryLike` surface over a REAL inference server, driving the REAL
    `pool_hooks` rather than a restatement of them."""

    search_kind = "gumbel"  # suppresses the PUCT-only cluster block
    avg_game_length = 12.0
    x_winrate = 0.5
    o_winrate = 0.4
    draw_rate = 0.1  # the third outcome share.
    draws = 1
    sims_per_sec = 100.0
    recent_move_histories: list[list[tuple[int, int]]] = []

    def __init__(self, server: InferenceServer) -> None:
        self._inference_server = server

    @property
    def batch_fill_pct(self) -> float:
        return batch_fill_pct(self)

    @property
    def inference_batch_timing(self) -> dict[str, Any]:
        return inference_batch_timing(self)


class Buffer:
    size = 7
    capacity = 64


class RStats:
    mcts_mean_depth = 3.0
    mcts_mean_root_concentration = 0.1


def emit_iteration(pool: Any) -> dict[str, Any]:
    """One `iteration_complete` for `pool` through the real builder; the event is returned."""
    sink = ListSink()
    emit_iteration_complete_event(
        11, 10, 4, pool, Buffer(),
        lambda: 0.0, None, {}, RStats(), sink, search_levers={},
    )
    assert len(sink.events) == 1
    return sink.events[0]
