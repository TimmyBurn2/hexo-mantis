"""Suite Q3 — the inference batching instrument (collector wait / collate / occupancy).

>300 justify: ONE instrument, one set of fakes. Every assertion binds the same producer chain —
the graph loop's timers -> `pool_hooks` -> the `iteration_complete` builder -> the sink — so
splitting the measurement arms from the arrival arms would duplicate every fake across the seam.
Both batching fields get their producer test here, driving the REAL producer through the real
hook into an injected sink. The occupancy DISTRIBUTION is asserted and not just the ratio,
because a mean cannot separate "one request per forward" from "sometimes 64, sometimes 0".
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import torch

import mantis.selfplay.graph_collate as collate_mod
from mantis.encoding import lookup
from mantis.selfplay.graph_collate import GraphBatch, GraphWirePayload
from mantis.selfplay.inference_server import InferenceServer
from mantis.selfplay.pool_hooks import batch_fill_pct, inference_batch_timing
from mantis.train.events import emit_iteration_complete_event

_GRAPH_SPEC = lookup("gnn_axis_v1")


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


def _cfg(**over: Any) -> dict[str, Any]:
    base = {
        "inference_batch_size": 8, "inference_max_wait_ms": 20.0,
        # The graph arm resolves the fused-forward memory bound at construction. NON-BINDING
        # here: these rows measure WAIT and OCCUPANCY at the POP and are blind to the split.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }
    base.update(over)
    return {"inference": base, "encoding": "gnn_axis_v1"}


def _wire_for(n_graphs: int = 2, nodes_per_graph: int = 3, legal_per_graph: int = 2
              ) -> GraphWirePayload:
    """A REAL `GraphWirePayload` matching `_hand_built_batch`'s shape — `_run_graph_loop` reads
    the wire's own CSR offsets to plan, so only the offsets have to be real."""
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


class _FakeGraphBatcher:
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
        return 1

    def close(self) -> None:
        self.closed += 1


class _FiniteGraphNet(torch.nn.Module):
    """Stub graph net: finite per-legal-node logits + per-graph values."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward_batch(self, x, edge_index, edge_attr, legal_index, stone_mask, node_offsets):
        n_legal = int(legal_index.numel())  # R284 P-MASK: rows, not a dense mask
        b = int(node_offsets.shape[0]) - 1
        return (
            torch.zeros(n_legal, dtype=torch.float32),
            torch.zeros(b, 1, dtype=torch.float32),
            torch.zeros(b, 65, dtype=torch.float32),
        )


def _hand_built_batch(n_graphs: int = 2, nodes_per_graph: int = 3) -> GraphBatch:
    """A minimal VALID collated batch, without a live Rust queue."""
    n = n_graphs * nodes_per_graph
    node_offsets = torch.arange(0, n + 1, nodes_per_graph, dtype=torch.int64)
    legal_mask = torch.zeros(n, dtype=torch.bool)
    for g in range(n_graphs):
        legal_mask[g * nodes_per_graph + 1] = True
        legal_mask[g * nodes_per_graph + 2] = True
    return GraphBatch(
        x=torch.zeros(n, 11, dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.int64),
        edge_attr=torch.zeros((0, 5), dtype=torch.float32),
        legal_offsets=torch.arange(0, 2 * n_graphs + 1, 2, dtype=torch.int64),
        # The REAL gather for the mask above: rows 1 and 2 of each graph, ascending across
        # the fuse. All zeros worked only because the stub reads `.numel()`.
        legal_node_gather=torch.tensor(
            [g * nodes_per_graph + k for g in range(n_graphs) for k in (1, 2)],
            dtype=torch.int64,
        ),
        node_offsets=node_offsets,
        n_stones=torch.ones(n_graphs, dtype=torch.int64),
        n_graphs=n_graphs,
        device="cpu",
    )


def _run_graph_server(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    counts: list[int],
    *,
    wait_s: float = 0.0,
    collate_s: float = 0.0,
    batch_size: int = 8,
) -> InferenceServer:
    """Build a graph `InferenceServer`, drive its loop over `counts`, return the server."""
    batch = _hand_built_batch()

    def _collate(*_a: Any, **_kw: Any) -> GraphBatch:
        if collate_s:
            time.sleep(collate_s)
        return batch

    monkeypatch.setattr(collate_mod, "collate_graph_batch", _collate)
    batcher = _FakeGraphBatcher(_wire_for(), counts, wait_s=wait_s)
    server = InferenceServer(
        _FiniteGraphNet(), device, _cfg(inference_batch_size=batch_size),
        batcher=batcher, encoding_spec=_GRAPH_SPEC,
    )
    batcher.server = server
    server.run()
    return server


def test_the_graph_loop_measures_its_own_collector_wait_and_collate_cost(
    device, monkeypatch
) -> None:
    """The collector wait inside `next_graph_batch` and the collate cost are MEASURED per served
    batch: a wait pegged at `inference_max_wait_ms` says the batch threshold was never reached.
    """
    server = _run_graph_server(
        device, monkeypatch, [2, 2, 2], wait_s=0.005, collate_s=0.002,
    )
    snap = server.batch_timing_snapshot()

    assert snap["representation"] == "graph"
    assert snap["batch_size"] == 8
    assert snap["queue_wait"]["count"] == 3
    assert snap["queue_wait"]["min_ms"] >= 5.0
    assert snap["queue_wait"]["mean_ms"] >= 5.0
    assert snap["collate"]["count"] == 3
    assert snap["collate"]["min_ms"] >= 2.0
    # The stop-pop returned no requests: a deadline that expired empty is counted apart
    # from the served waits, never folded into their mean.
    assert snap["empty_polls"] == 1


def test_an_occupancy_histogram_separates_always_one_from_a_mixed_load(
    device, monkeypatch
) -> None:
    """min/max/histogram resolve what the mean cannot: two loads with the SAME mean occupancy
    are indistinguishable by ratio, so the histogram is asserted as a distribution."""
    server = _run_graph_server(device, monkeypatch, [1, 1, 8], batch_size=8)
    occ = server.batch_timing_snapshot()["occupancy"]

    assert occ["count"] == 3
    assert occ["total"] == 10
    assert occ["min"] == 1
    assert occ["max"] == 8
    assert occ["histogram"] == {"1": 2, "8": 1}

    flat = _run_graph_server(device, monkeypatch, [1, 1, 1], batch_size=8)
    flat_occ = flat.batch_timing_snapshot()["occupancy"]
    assert flat_occ["min"] == 1 and flat_occ["max"] == 1
    assert flat_occ["histogram"] == {"1": 3}
    assert flat_occ["histogram"] != occ["histogram"]


def test_the_instrument_is_defined_before_the_first_forward(device, monkeypatch) -> None:
    """Read before any batch: every derived reading is `None`, never a fabricated zero."""
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: None)
    server = InferenceServer(
        _FiniteGraphNet(), device, _cfg(inference_batch_size=64),
        batcher=_FakeGraphBatcher(_wire_for(), []), encoding_spec=_GRAPH_SPEC,
    )
    snap = server.batch_timing_snapshot()
    assert (snap["queue_wait"], snap["collate"], snap["occupancy"]) == (None, None, None)
    assert snap["batch_size"] == 64
    assert snap["max_wait_ms"] == 20.0
    assert snap["empty_polls"] == 0


class _ListSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class _TelemetryPool:
    """The narrow `PoolTelemetryLike` surface over a REAL inference server, driving the REAL
    `pool_hooks` rather than a restatement of them."""

    search_kind = "gumbel"          # suppresses the PUCT-only cluster block
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


class _Buffer:
    size = 7
    capacity = 64


class _RStats:
    mcts_mean_depth = 3.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


def _emit(pool: Any) -> dict[str, Any]:
    sink = _ListSink()
    emit_iteration_complete_event(
        11, 0.0, 10, 4, pool, _Buffer(), {}, {}, 64,
        lambda: 0.0, None, {}, _RStats(), sink,
    )
    assert len(sink.events) == 1
    return sink.events[0]


def test_the_batching_block_reaches_the_sink_on_iteration_complete(
    device, monkeypatch
) -> None:
    """The manifest producer test for `inference_batching`: a LIVE graph-loop measurement
    travels pool -> hook -> builder -> sink, whole and unfabricated."""
    server = _run_graph_server(device, monkeypatch, [2, 2], wait_s=0.003)
    payload = _emit(_TelemetryPool(server))

    block = payload["inference_batching"]
    assert block["representation"] == "graph"
    assert block["queue_wait"]["count"] == 2
    assert block["queue_wait"]["min_ms"] >= 3.0
    assert block["occupancy"]["histogram"] == {"2": 2}
    assert block == server.batch_timing_snapshot()


def test_batch_fill_pct_reaches_the_sink_from_the_live_inference_counters(
    device, monkeypatch
) -> None:
    """The manifest producer test for `batch_fill_pct`: 2 requests per forward against 8 slots
    is 25%, and the SAME server's occupancy block agrees."""
    server = _run_graph_server(device, monkeypatch, [2, 2, 2], batch_size=8)
    payload = _emit(_TelemetryPool(server))

    assert payload["batch_fill_pct"] == pytest.approx(25.0)
    assert payload["inference_batching"]["occupancy"]["fill_pct_mean"] == pytest.approx(
        25.0
    )


def test_the_occupancy_histogram_leaves_the_one_bucket_under_a_batched_submit(
    device, monkeypatch
) -> None:
    """No forward serves a single graph. The counts are SCRIPTED by the fake batcher, so this
    pins what the instrument reports, not the Rust dispatch (pinned queue-side and cross-FFI).
    """
    server = _run_graph_server(device, monkeypatch, [8, 8, 7, 8, 6], batch_size=8)
    occ = server.batch_timing_snapshot()["occupancy"]

    assert occ["histogram"].get("1", 0) == 0, "no forward may serve a single graph"
    assert occ["max"] == 8
    assert occ["mean"] > 1.0
    # Absolute counts are the primary reading; fill_pct is knob-relative (a mean
    # occupancy of ~6 reads as ~75% at batch_size 8 and ~9.4% at 64, same physics).
    assert occ["fill_pct_mean"] == pytest.approx(occ["mean"] / 8 * 100.0, rel=1e-6)


def test_batch_fill_pct_and_the_occupancy_block_agree_on_the_same_forwards(
    device, monkeypatch
) -> None:
    """`batch_fill_pct` and `inference_batching.occupancy` come from DIFFERENT accumulators over
    the same loop, so a drift between them means one counter stopped being fed."""
    server = _run_graph_server(device, monkeypatch, [8, 4, 8], batch_size=8)
    occ = server.batch_timing_snapshot()["occupancy"]

    assert batch_fill_pct(_TelemetryPool(server)) == pytest.approx(
        occ["fill_pct_mean"], rel=1e-6
    )
    assert occ["total"] == server.total_requests
    assert occ["count"] == server.forward_count


def test_a_telemetry_source_without_the_producer_publishes_none_never_zero() -> None:
    """The key is always present and carries `None`, which a consumer reads as "no producer"."""

    class _NoInstrumentPool:
        """A telemetry source with NO batching producer — no such member at all."""

        search_kind = "gumbel"
        avg_game_length = 12.0
        x_winrate = 0.5
        o_winrate = 0.4
        draw_rate = 0.1  # the third outcome share.
        draws = 1
        sims_per_sec = 100.0
        batch_fill_pct = 0.0
        recent_move_histories: list[list[tuple[int, int]]] = []

    assert not hasattr(_NoInstrumentPool(), "inference_batch_timing")
    payload = _emit(_NoInstrumentPool())
    assert "inference_batching" in payload
    assert payload["inference_batching"] is None
