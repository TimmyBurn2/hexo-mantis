"""Suite Q3 — the inference batching instrument (collector wait / collate / occupancy).

>300 justify: ONE instrument, one set of fakes. Every assertion here binds the same
producer chain — `InferenceServer`'s graph-loop timers -> `pool_hooks` -> the
`iteration_complete` builder -> the sink — and the loop-driving fakes (a scripted graph
batcher, a dense batcher, a stub net, a hand-built collated batch, the telemetry pool)
are shared by all of them. Splitting the measurement arms from the arrival arms would put
the producer in one file and the thing that proves it reaches the channel in another, and
would duplicate every fake across the seam.

The producer tests for the two `iteration_complete` batching fields (LAW-07/LAW-18):
`inference_batching`, authored here, and `batch_fill_pct`, which shipped a live producer
and NO manifest row. Both arms drive the REAL producer — the graph loop's own timers on a
real `InferenceServer` — through the real hook and assert arrival at an injected sink, so
a deleted timer or a dropped payload key reds here instead of going quiet in a run.

WHY the occupancy distribution and not just the ratio: `batch_fill_pct` is a mean, and a
mean cannot separate "one request per forward, every forward" from "sometimes 64,
sometimes 0" — the two agree on the ratio and disagree on everything that matters about
the queue. `test_an_occupancy_histogram_separates_always_one_from_a_mixed_load` is that
separation, asserted.
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

_GRID_SPEC = lookup("v6")
_GRAPH_SPEC = lookup("gnn_axis_v1")


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


def _cfg(**over: Any) -> dict[str, Any]:
    base = {
        "inference_batch_size": 8, "inference_max_wait_ms": 20.0,
        "trace_inference": False, "compile_inference": False,
        "compile_inference_mode": "default", "compile_inference_dynamic": True,
        "perf_timing": False, "perf_sync_cuda": False,
        # F-816-10: the graph arm resolves the fused-forward memory bound at
        # construction. NON-BINDING here on purpose — the rows in this file are
        # about the WAIT and OCCUPANCY instruments, which are measured at the POP
        # and are blind to the split by construction; a cap that bound would put a
        # split into rows that assert nothing about it.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }
    base.update(over)
    return {"inference": base, "encoding": "v6", "train": {"amp_dtype": "fp16"}}


def _wire_for(n_graphs: int = 2, nodes_per_graph: int = 3, legal_per_graph: int = 2
              ) -> GraphWirePayload:
    """A REAL `GraphWirePayload` matching `_hand_built_batch`'s shape.

    F-816-10: `_run_graph_loop` now reads the wire's own CSR offsets ONCE per pop to plan its
    bounded forwards, so an opaque `object()` sentinel no longer reaches the collate — the
    plan is computed BEFORE it. The collate stays monkeypatched (these rows are about the
    timing instrument, not the wire contract); only the offsets have to be real.
    """
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
    """Drives `_run_graph_loop` over a scripted list of per-pop request counts, sleeping
    `wait_s` inside every pop so the measured collector wait has a known lower bound."""

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


class _FakeDenseBatcher:
    """Drives the dense `run()` loop for `n_batches` iterations, then stops it."""

    def __init__(self, feature_len: int, n_batches: int = 2, n_requests: int = 2) -> None:
        self._left = n_batches
        self._ids = list(range(1, n_requests + 1))
        self._batch = np.ascontiguousarray(
            np.zeros((n_requests, feature_len), dtype=np.float32)
        )
        self.server: InferenceServer | None = None
        self.closed = 0

    def next_inference_batch(self, batch_size: int, max_wait_ms: float):
        if self._left <= 0:
            assert self.server is not None
            self.server._stop_event.set()
            return [], self._batch
        self._left -= 1
        return list(self._ids), self._batch

    def submit_inference_results(self, ids, policies, values) -> None:
        return None

    def submit_inference_failure(self, ids, error_msg: str) -> None:
        return None

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


class _DenseStubNet(torch.nn.Module):
    """Stub CNN-shaped net: uniform log-policy + zero value, whatever the input."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = int(x.shape[0])
        pol = torch.zeros(n, _GRID_SPEC.policy_logit_count, dtype=torch.float32)
        return pol, torch.zeros(n, 1, dtype=torch.float32), torch.zeros(n, 1)


def _hand_built_batch(n_graphs: int = 2, nodes_per_graph: int = 3) -> GraphBatch:
    """A minimal VALID collated batch — enough for `stone_mask_from_batch`,
    `segment_softmax` and the finiteness gate, without a live Rust queue."""
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
        # The REAL gather for the mask above: rows 1 and 2 of each graph, ascending
        # across the fuse (wire check 13). It used to be all zeros, which worked only
        # because the stub reads `.numel()` — a stub that counted PER GRAPH would have
        # put every legal node in graph 0 and the fixture would not have said so.
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


# ══ the instrument itself ════════════════════════════════════════════════════════
def test_the_graph_loop_measures_its_own_collector_wait_and_collate_cost(
    device, monkeypatch
) -> None:
    """Q3-01 — the wait spent inside `next_graph_batch` and the cost of
    `collate_graph_batch` are both MEASURED, per served batch, with known lower bounds.

    The collector wait is the load-bearing number: it is exactly the Rust-side
    `batch_size / 2`-or-deadline wait, so a wait pegged at `inference_max_wait_ms` on
    every forward says the threshold was never reached."""
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
    """Q3-02 — min/max/histogram resolve what the mean cannot.

    Two loads with the SAME mean occupancy (1,1,1,1 vs 1,1,1,... plus one full batch) are
    indistinguishable by ratio; the histogram is what tells them apart, so it is asserted
    as a distribution, not as a summary."""
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


def test_a_grid_run_reports_no_producer_rather_than_a_fabricated_zero(device) -> None:
    """Q3-03 — the dense loop is NOT instrumented, and says so.

    Every derived reading is `None` on a grid run, after real dense forwards. A constant
    `0` in the ONE channel would read as a real measurement ("the queue never waits") —
    the F-10 class in miniature (docs/contracts/event_manifest.md)."""
    feature_len = _GRID_SPEC.n_planes * _GRID_SPEC.trunk_size * _GRID_SPEC.trunk_size
    batcher = _FakeDenseBatcher(feature_len, n_batches=2)
    server = InferenceServer(
        _DenseStubNet(), device, _cfg(inference_batch_size=4),
        batcher=batcher, encoding_spec=_GRID_SPEC,
    )
    batcher.server = server
    server.run()

    snap = server.batch_timing_snapshot()
    assert server.forward_count == 2, "the dense loop must actually have run"
    assert snap["representation"] == "grid"
    assert snap["queue_wait"] is None
    assert snap["collate"] is None
    assert snap["occupancy"] is None
    assert snap["empty_polls"] is None


def test_the_instrument_is_defined_before_the_first_forward(device, monkeypatch) -> None:
    """Q3-04 — read before any batch: no division by zero, and no fabricated zero
    either. Every derived reading is `None`; the two config facts are already known."""
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


# ══ arrival at the sink (the manifest's producer tests) ══════════════════════════
class _ListSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class _TelemetryPool:
    """The narrow `PoolTelemetryLike` surface over a REAL inference server — the two
    batching members go through the REAL `pool_hooks` functions, so this drives the
    production producer and not a restatement of it."""

    gumbel_mcts = True          # suppresses the PUCT-only cluster block
    avg_game_length = 12.0
    x_winrate = 0.5
    o_winrate = 0.4
    draw_rate = 0.1  # F-816-2: the third outcome share.
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
    """Q3-05 — the manifest producer test for `inference_batching`: a LIVE graph-loop
    measurement travels pool -> hook -> builder -> sink, whole and unfabricated."""
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
    """Q3-06 — the manifest producer test for `batch_fill_pct`, which had none.

    2 requests per forward against 8 configured slots is 25%, and the SAME server's
    occupancy block agrees — the ratio and the distribution are two readings of one
    producer, so they can never drift apart silently."""
    server = _run_graph_server(device, monkeypatch, [2, 2, 2], batch_size=8)
    payload = _emit(_TelemetryPool(server))

    assert payload["batch_fill_pct"] == pytest.approx(25.0)
    assert payload["inference_batching"]["occupancy"]["fill_pct_mean"] == pytest.approx(
        25.0
    )


def test_the_occupancy_histogram_leaves_the_one_bucket_under_a_batched_submit(
    device, monkeypatch
) -> None:
    """Q-FIND-1 — the post-fix contract on the LIVE instrument: no forward serves a
    single graph.

    The pre-fix run5 shape is `histogram == {"1": N}`, `mean == 1.0`, `max == 1` — the
    reading Q-FIND-1b pre-registered at 1.5625% fill, and the shape
    `test_an_occupancy_histogram_separates_always_one_from_a_mixed_load` above holds as
    the *starved* reference. This is its complement: the shape a whole-leaf-batch submit
    must produce on the same producer chain.

    SCOPE, stated so it is not over-read: the counts are SCRIPTED by the fake batcher, so
    this pins what the instrument reports for a batched arrival — it does not itself
    exercise the Rust dispatch. The dispatch claim is pinned queue-side
    (`crates/mantis-selfplay/tests/queue_roundtrip.rs`) and cross-FFI
    (`tests/bridge/test_graph_batch_dispatch_parity.py`).
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
    """Q-FIND-1 — the two published instruments may not disagree.

    `batch_fill_pct` (`pool_hooks.batch_fill_pct`) and `inference_batching.occupancy`
    (`InferenceServer.batch_timing_snapshot`) are computed from DIFFERENT accumulators
    over the same loop, so a drift between them means one of the two counters stopped
    being fed — which is exactly how a starved-queue reading would go quiet after the
    dispatch change.
    """
    server = _run_graph_server(device, monkeypatch, [8, 4, 8], batch_size=8)
    occ = server.batch_timing_snapshot()["occupancy"]

    assert batch_fill_pct(_TelemetryPool(server)) == pytest.approx(
        occ["fill_pct_mean"], rel=1e-6
    )
    assert occ["total"] == server.total_requests
    assert occ["count"] == server.forward_count


def test_a_telemetry_source_without_the_producer_publishes_none_never_zero() -> None:
    """Q3-07 — the key is always present and carries `None` when the source produces
    nothing for it. A consumer must read `None` as "no producer", never as zero."""

    class _NoInstrumentPool:
        """A telemetry source with NO batching producer — declares no such member at all
        (not a member that raises: `getattr`'s default would swallow that)."""

        gumbel_mcts = True
        avg_game_length = 12.0
        x_winrate = 0.5
        o_winrate = 0.4
        draw_rate = 0.1  # F-816-2: the third outcome share.
        draws = 1
        sims_per_sec = 100.0
        batch_fill_pct = 0.0
        recent_move_histories: list[list[tuple[int, int]]] = []

    assert not hasattr(_NoInstrumentPool(), "inference_batch_timing")
    payload = _emit(_NoInstrumentPool())
    assert "inference_batching" in payload
    assert payload["inference_batching"] is None
