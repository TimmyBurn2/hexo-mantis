"""Suite F — the ONE inference server (`mantis.selfplay.inference_server`).

>300 justify: one server, two loops, one seam. The dense-loop assertions (F-01..F-05),
the representation dispatch (F-08/F-09), the graph-loop failure + heartbeat + collate
call-site pins (F-10/F-11/F-15) and the wire/seam obligations (F-13/F-14) all bind the
SAME class; splitting them by loop would put the dispatch pin in one file and the two
things it dispatches to in others, and the shared fakes would have to be duplicated.

IMPL-written (non-⊕) per DESIGN §b: ports of the old server suites, rewritten
public-surface against `build_net`-built nets. GPU-only paths (compile
`reduce-overhead` + CUDA-graph warmup, pinned-staging H2D) keep loud skips on CPU — a
PASS state for this WP (DESIGN §f-R11), recorded for the cutover battery.

F-15 is the sharpest pin in the file. `gnn_axis_v1` is the ONLY registered graph
encoding and it happens to be exactly trunk 19 / win 6 / node-feat 11 / edge-feat 5, so a
test driven by a real registry spec cannot tell spec-derived dims from hard-coded
literals. The non-default-spec arm therefore swaps `server.encoding_spec` for a stub with
DIFFERENT dims after construction (the loop binds `spec = self.encoding_spec` at loop
entry and reads the four dims inline at the collate call, so the swap is observable).
Dropping or weakening that arm silently reopens the hard-coded-dims escape.
"""
from __future__ import annotations

import math
import threading
import time
import unittest.mock as mock
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

import mantis.selfplay.graph_collate as collate_mod
from mantis._engine import InferenceBatcher
from mantis.encoding import lookup
from mantis.model import CnnArch, RepresentationMismatch, amp_dtype_for, build_net
from mantis.selfplay.graph_collate import (
    GraphBatch,
    GraphWirePayload,
    graph_wire_from_rust,
)
from mantis.selfplay.inference_server import InferenceServer

_GRID_SPEC = lookup("v6")
_GRAPH_SPEC = lookup("gnn_axis_v1")

BOARD_CHANNELS = _GRID_SPEC.n_planes  # spec-derived, never a literal
BOARD_SIZE = _GRID_SPEC.trunk_size
N_ACTIONS = _GRID_SPEC.policy_logit_count

_NO_CUDA = not torch.cuda.is_available()
_GPU_ONLY = pytest.mark.skipif(
    _NO_CUDA,
    reason=(
        "GPU-only path (CUDA-graph capture / pinned-staging H2D). Skip-with-reason on "
        "CPU is the pre-registered PASS state for this WP; recorded for the cutover "
        "GPU battery (DESIGN §f-R11)."
    ),
)


# ── shared helpers ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_cnn(device: torch.device, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    net = build_net(
        CnnArch(
            board_size=BOARD_SIZE,
            in_channels=BOARD_CHANNELS,
            filters=64,
            res_blocks=2,
        )
    ).to(device)
    net.eval()
    return net


@pytest.fixture(scope="module")
def model(device: torch.device) -> torch.nn.Module:
    return _make_cnn(device)


def _random_state() -> np.ndarray:
    return np.random.randn(BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)


def _cfg(**over: Any) -> dict[str, Any]:
    # WPSC Phase 2 SC-A2 reshape: `InferenceHParams.from_config` now reads `config
    # ["inference"]` (a nested schema-shaped section), not `config["selfplay"]`/a flat dict.
    # WPSC Phase 3 SC-B2: `resolve_from_config` no longer defaults an absent 'encoding'
    # key to v6 (R28) — this fixture's model is v6-grid-derived (BOARD_CHANNELS/BOARD_SIZE
    # above), so the encoding is now explicit rather than relying on the retired fallback.
    # WPSC Phase 3 SC-B3: `InferenceServer.__init__` now hard-reads `config["train"]
    # ["amp_dtype"]` unconditionally (R30b, no fallback) — every config needs a `train`
    # section; amp-focused tests below override it explicitly.
    base = {
        "inference_batch_size": 8, "inference_max_wait_ms": 20.0, "trace_inference": True,
        "compile_inference": False, "compile_inference_mode": "default",
        "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
        # F-816-10 (R276(f)): the GRAPH arm resolves `inference.fused_graph_caps`
        # EAGERLY at construction, so every graph-route site built from this base needs
        # it. The pair is the template's NON-BINDING-BY-CONSTRUCTION value — nothing in
        # this file splits, and the split's own coverage lives in the F-816-10 oracles
        # where its M is asserted. Inert on the grid route, which never reads the block.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }
    base.update(over)
    return {"inference": base, "encoding": "v6", "train": {"amp_dtype": "fp16"}}


def _make_server(
    model: torch.nn.Module, device: torch.device, batch_size: int = 8, **kw: Any
) -> InferenceServer:
    return InferenceServer(model, device, _cfg(inference_batch_size=batch_size, **kw))


@dataclass
class _SpecStub:
    """A NON-default graph spec: every dim differs from `gnn_axis_v1`'s 19/6/11/5, so a
    hard-coded literal at the collate call site cannot pass."""

    trunk_size: int = 21
    win_length: int = 7
    node_feat_dim: int = 13
    edge_feat_dim: int = 3
    representation: str = "graph"
    policy_logit_count: int = 442
    name: str = "spec_stub_non_default"


class _FakeGraphBatcher:
    """Drives `_run_graph_loop` for exactly `n_batches` iterations, then stops the loop."""

    def __init__(self, wire: Any, n_batches: int = 1, n_requests: int = 2) -> None:
        self._wire = wire
        self._left = n_batches
        self._ids = list(range(1, n_requests + 1))
        self.server: InferenceServer | None = None
        self.results: list[tuple] = []
        self.failures: list[tuple[list[int], str]] = []
        self.closed = 0
        self.model_version = 0

    def next_graph_batch(self, batch_size: int, max_wait_ms: int):
        if self._left <= 0:
            assert self.server is not None
            self.server._stop_event.set()
            return [], None
        self._left -= 1
        return list(self._ids), self._wire

    def submit_graph_inference_results(self, ids, probs, offsets, values) -> None:
        self.results.append((list(ids), probs, offsets, values))

    def submit_graph_inference_failure(self, ids, error_msg: str) -> None:
        self.failures.append((list(ids), error_msg))

    def bump_model_version(self) -> int:
        self.model_version += 1
        return self.model_version

    def close(self) -> None:
        self.closed += 1


class _FakeDenseBatcher:
    """Drives the dense `run()` loop for exactly `n_batches` iterations, then stops it."""

    def __init__(self, feature_len: int, n_batches: int = 1, n_requests: int = 2) -> None:
        self._left = n_batches
        self._ids = list(range(1, n_requests + 1))
        self._batch = np.ascontiguousarray(
            np.zeros((n_requests, feature_len), dtype=np.float32)
        )
        self.server: InferenceServer | None = None
        self.results: list[tuple] = []
        self.failures: list[tuple[list[int], str]] = []
        self.closed = 0
        self.model_version = 0

    def next_inference_batch(self, batch_size: int, max_wait_ms: int):
        if self._left <= 0:
            assert self.server is not None
            self.server._stop_event.set()
            return [], self._batch
        self._left -= 1
        return list(self._ids), self._batch

    def submit_inference_results(self, ids, policies, values) -> None:
        self.results.append((list(ids), policies, values))

    def submit_inference_failure(self, ids, error_msg: str) -> None:
        self.failures.append((list(ids), error_msg))

    def bump_model_version(self) -> int:
        self.model_version += 1
        return self.model_version

    def close(self) -> None:
        self.closed += 1


def _graph_server(
    device: torch.device,
    batcher: Any,
    model: torch.nn.Module | None = None,
    *,
    heartbeat=None,
    batch_size: int = 8,
) -> InferenceServer:
    net = model if model is not None else _FiniteGraphNet()
    server = InferenceServer(
        net,
        device,
        _cfg(inference_batch_size=batch_size),
        batcher=batcher,
        encoding_spec=_GRAPH_SPEC,
        heartbeat=heartbeat,
    )
    batcher.server = server
    return server


class _FiniteGraphNet(torch.nn.Module):
    """Stub graph net: finite per-legal-node logits + per-graph values."""

    def __init__(self, *, nonfinite: bool = False) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.nonfinite = nonfinite
        self.calls: list[tuple[int, ...]] = []

    def forward_batch(self, x, edge_index, edge_attr, legal_mask, stone_mask, node_offsets):
        self.calls.append(tuple(x.shape))
        n_legal = int(legal_mask.sum().item())
        b = int(node_offsets.shape[0]) - 1
        logits = torch.zeros(n_legal, dtype=torch.float32)
        value = torch.zeros(b, 1, dtype=torch.float32)
        if self.nonfinite:
            logits[0] = float("nan")
        return logits, value, torch.zeros(b, 65, dtype=torch.float32)


def _hand_built_batch(n_graphs: int = 2, nodes_per_graph: int = 3) -> GraphBatch:
    """A minimal, VALID collated batch — enough for `stone_mask_from_batch`,
    `segment_softmax` and the finiteness gate. Built by hand so the loop can be driven
    without a live Rust queue."""
    n = n_graphs * nodes_per_graph
    node_offsets = torch.arange(0, n + 1, nodes_per_graph, dtype=torch.int64)
    legal_mask = torch.zeros(n, dtype=torch.bool)
    for g in range(n_graphs):
        legal_mask[g * nodes_per_graph + 1] = True
        legal_mask[g * nodes_per_graph + 2] = True
    legal_offsets = torch.arange(0, 2 * n_graphs + 1, 2, dtype=torch.int64)
    return GraphBatch(
        x=torch.zeros(n, 11, dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.int64),
        edge_attr=torch.zeros((0, 5), dtype=torch.float32),
        legal_mask=legal_mask,
        legal_offsets=legal_offsets,
        legal_node_gather=torch.zeros(2 * n_graphs, dtype=torch.int64),
        policy_dst_slot=torch.zeros(2 * n_graphs, dtype=torch.int64),
        node_offsets=node_offsets,
        node_coords=torch.zeros((n, 2), dtype=torch.int64),
        window_center=torch.zeros((n_graphs, 2), dtype=torch.int64),
        current_player=torch.ones(n_graphs, dtype=torch.int64),
        n_stones=torch.ones(n_graphs, dtype=torch.int64),
        n_graphs=n_graphs,
        device="cpu",
    )


def _wire_for(n_graphs: int = 2, nodes_per_graph: int = 3, legal_per_graph: int = 2
              ) -> GraphWirePayload:
    """A REAL `GraphWirePayload` whose CSR offsets match `_hand_built_batch`'s shape.

    F-816-10: `_run_graph_loop` reads the wire's own offsets ONCE per pop to PLAN its bounded
    forwards, before any collate runs, so an opaque `object()` sentinel no longer reaches the
    loop. The collate itself stays monkeypatched in the rows below — those rows are about the
    call's kwargs, the finiteness gate and the heartbeat, not about the wire contract, which
    has its own suites. Only the offsets have to be real, and the caps these servers carry are
    non-binding, so every drive here is the M == 1 path.
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


# ══ F-01 — dense submit_and_wait correctness ═════════════════════════════════════
def test_policy_shape_and_sums_to_one(model, device) -> None:
    server = _make_server(model, device, batch_size=4)
    server.start()
    try:
        policy, _value = server.infer(_random_state())
        assert policy.shape == (N_ACTIONS,)
        assert abs(policy.sum() - 1.0) < 1e-4
    finally:
        server.stop()
        server.join(timeout=2.0)


def test_value_in_range(model, device) -> None:
    server = _make_server(model, device, batch_size=4)
    server.start()
    try:
        _policy, value = server.infer(_random_state())
        assert -1.0 <= value <= 1.0
    finally:
        server.stop()
        server.join(timeout=2.0)


def test_policy_is_finite(model, device) -> None:
    server = _make_server(model, device, batch_size=4)
    server.start()
    try:
        policy, value = server.infer(_random_state())
        assert np.all(np.isfinite(policy))
        assert math.isfinite(value)
    finally:
        server.stop()
        server.join(timeout=2.0)


# ══ F-02 — concurrency + request accounting ══════════════════════════════════════
def test_all_results_valid_under_concurrency(model, device) -> None:
    n_requests = 24
    server = _make_server(model, device, batch_size=8)
    server.start()

    errors: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        policy, value = server.infer(_random_state())
        if policy.shape != (N_ACTIONS,):
            with lock:
                errors.append(f"bad policy shape: {policy.shape}")
        if not np.all(np.isfinite(policy)):
            with lock:
                errors.append("policy has non-finite values")
        if not (-1.0 <= value <= 1.0):
            with lock:
                errors.append(f"value out of range: {value}")

    threads = [threading.Thread(target=worker) for _ in range(n_requests)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    server.stop()
    server.join(timeout=2.0)
    assert errors == []


def test_total_requests_counted_correctly(model, device) -> None:
    n_requests = 10
    server = _make_server(model, device, batch_size=4)
    server.start()
    for _ in range(n_requests):
        server.infer(_random_state())
    server.stop()
    server.join(timeout=2.0)
    assert server.total_requests == n_requests


# ══ F-03 — TorchScript trace path ════════════════════════════════════════════════
def _trace_server(
    model: torch.nn.Module, device: torch.device, *, trace: bool, batch_size: int = 4
) -> InferenceServer:
    return InferenceServer(
        model, device,
        _cfg(inference_batch_size=batch_size, trace_inference=trace),
    )


def test_traced_matches_untraced(model, device) -> None:
    np.random.seed(0)
    states = [_random_state() for _ in range(6)]

    s_off = _trace_server(model, device, trace=False)
    s_off.start()
    try:
        ref = [s_off.infer(s) for s in states]
    finally:
        s_off.stop()
        s_off.join(timeout=2.0)

    s_on = _trace_server(model, device, trace=True)
    assert s_on._traced_model is not None, "trace did not compile on the test model"
    s_on.start()
    try:
        traced = [s_on.infer(s) for s in states]
    finally:
        s_on.stop()
        s_on.join(timeout=2.0)

    for i, ((p_ref, v_ref), (p_tr, v_tr)) in enumerate(zip(ref, traced, strict=True)):
        assert p_tr.shape == p_ref.shape, f"state {i}: policy shape mismatch"
        max_p = float(np.abs(p_tr - p_ref).max())
        assert max_p < 5e-3, f"state {i}: policy diverged max={max_p}"
        assert abs(v_tr - v_ref) < 5e-3, f"state {i}: value diverged"


def test_traced_follows_weight_swap(device) -> None:
    net = _make_cnn(device, seed=11)
    server = _trace_server(net, device, trace=True)
    assert server._traced_model is not None
    server.start()
    try:
        np.random.seed(123)
        state = _random_state()
        p_before, v_before = server.infer(state)

        new_sd = {
            k: torch.randn_like(v) if v.dtype.is_floating_point else v
            for k, v in net.state_dict().items()
        }
        server.load_state_dict_safe(new_sd)

        p_after, v_after = server.infer(state)
    finally:
        server.stop()
        server.join(timeout=2.0)

    diff_p = float(np.abs(p_after - p_before).max())
    diff_v = abs(v_after - v_before)
    assert diff_p > 1e-3, "traced model did not pick up the weight swap"
    assert diff_v > 1e-3 or diff_p > 1e-2


def test_trace_disabled_via_config(model, device) -> None:
    server = _trace_server(model, device, trace=False)
    assert server._trace_inference is False
    assert server._traced_model is None
    server.start()
    try:
        policy, _ = server.infer(_random_state())
        assert policy.shape == (N_ACTIONS,)
    finally:
        server.stop()
        server.join(timeout=2.0)


# ══ F-04 — compile / trace mutex + the GPU-only compile arm ══════════════════════
def test_compile_and_trace_mutex(model, device) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        InferenceServer(
            model, device,
            _cfg(trace_inference=True, compile_inference=True),
        )


@_GPU_ONLY
def test_compile_inference_weight_swap_propagates(device) -> None:
    """`torch.compile` wraps the model in an `OptimizedModule`; `load_state_dict_safe`
    must unwrap `_orig_mod` so the swap lands on the underlying parameters."""
    net = _make_cnn(device, seed=7)
    server = InferenceServer(
        net, device,
        _cfg(
            inference_batch_size=4,
            trace_inference=False,
            compile_inference=True,
            compile_inference_mode="default",
            compile_inference_dynamic=True,
        ),
    )
    assert server._compile_inference is True, "compile failed at init — test is meaningless"
    server.start()
    try:
        np.random.seed(7)
        state = _random_state()
        p_before, v_before = server.infer(state)
        new_sd = {
            k: torch.randn_like(v) if v.dtype.is_floating_point else v
            for k, v in net.state_dict().items()
        }
        server.load_state_dict_safe(new_sd)
        p_after, v_after = server.infer(state)
    finally:
        server.stop()
        server.join(timeout=5.0)
    assert float(np.abs(p_after - p_before).max()) > 1e-3 or abs(v_after - v_before) > 1e-3


@_GPU_ONLY
def test_compile_reduce_overhead_padding_and_warmup(device) -> None:
    """CUDA-graph replay requires a fixed input shape: `_padding_active` arms only when
    compile + `reduce-overhead` + pinned staging all hold, and `_warmup_compile_path`
    captures the graph on the dispatcher thread."""
    net = _make_cnn(device, seed=3)
    server = InferenceServer(
        net, device,
        _cfg(
            inference_batch_size=4,
            trace_inference=False,
            compile_inference=True,
            compile_inference_mode="reduce-overhead",
        ),
    )
    assert server._h2d_staging is not None, "pinned staging must exist on CUDA"
    assert server._padding_active() is True
    server._warmup_compile_path()
    server.stop()


def test_padding_is_inert_without_compile(model, device) -> None:
    """The CPU-observable half of the padding contract: with compile off, the padded
    CUDA-graph path is never armed regardless of device."""
    server = _make_server(model, device, batch_size=4)
    assert server._compile_inference is False
    assert server._padding_active() is False


@pytest.mark.skipif(not _NO_CUDA, reason="CPU-only assertion about the staging buffer")
def test_pinned_staging_absent_on_cpu(model, device) -> None:
    server = _make_server(model, device, batch_size=4)
    assert server._h2d_staging is None


# ══ F-05 — failure handling releases waiters ═════════════════════════════════════
def test_batch_prep_error_unblocks_workers(device) -> None:
    """A batch-prep error must translate into a raised error for the caller, never a hang
    (the prep runs INSIDE the guarded region)."""

    class IdentityNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            n = x.shape[0]
            pol = torch.ones(n, N_ACTIONS, device=x.device) / N_ACTIONS
            val = torch.zeros(n, 1, device=x.device)
            return pol.log(), val, val

    net = IdentityNet().to(device)
    net.eval()
    server = InferenceServer(net, device, _cfg(inference_batch_size=4))
    server.start()
    try:
        state = _random_state()
        done = threading.Event()
        error_caught: list[str] = []

        def _call() -> None:
            try:
                server.infer(state)
            except Exception as exc:  # noqa: BLE001 — recorded for the assertion
                error_caught.append(str(exc))
            finally:
                done.set()

        import mantis.selfplay.inference_server as server_mod

        with mock.patch.object(
            server_mod.np, "ascontiguousarray", side_effect=ValueError("bad array")
        ):
            t = threading.Thread(target=_call, daemon=True)
            t.start()
            hung = not done.wait(5.0)

        t.join(timeout=2.0)
        assert not hung, "server.infer() hung — the caller was not unblocked"
        assert len(error_caught) == 1, f"expected one error, got {error_caught}"
    finally:
        server.stop()
        server.join(timeout=2.0)


def test_infer_returns_on_model_forward_exception(device) -> None:
    class FailingNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor):
            raise RuntimeError("boom")

    net = FailingNet().to(device)
    net.eval()
    server = InferenceServer(net, device, _cfg(inference_batch_size=4))
    server.start()
    try:
        state = _random_state()
        done = threading.Event()
        error_caught: list[str] = []

        def _call() -> None:
            try:
                server.infer(state)
            except ValueError as exc:
                error_caught.append(str(exc))
            finally:
                done.set()

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        assert done.wait(5.0), "server.infer() hung waiting for results"
        assert len(error_caught) == 1
        assert "Model inference failed: boom" in error_caught[0]
    finally:
        server.stop()
        server.join(timeout=2.0)


def test_dense_loop_forward_failure_submits_failure_to_waiters(device) -> None:
    """The dispatcher arm of the same contract: a forward exception inside the LOOP is
    reported through `submit_inference_failure` with the pinned message prefix, and the
    loop keeps serving."""

    class FailingNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor):
            raise RuntimeError("boom")

    feature_len = BOARD_CHANNELS * BOARD_SIZE * BOARD_SIZE
    batcher = _FakeDenseBatcher(feature_len, n_batches=1)
    net = FailingNet().to(device)
    net.eval()
    server = InferenceServer(
        net, device, _cfg(inference_batch_size=4, trace_inference=False),
        batcher=batcher, encoding_spec=_GRID_SPEC,
    )
    batcher.server = server
    server.run()

    assert batcher.results == []
    assert len(batcher.failures) == 1
    ids, msg = batcher.failures[0]
    assert ids == [1, 2]
    assert msg.startswith("Model inference failed: ")
    assert "boom" in msg
    assert batcher.closed == 1


# ══ F-08 — representation dispatch ═══════════════════════════════════════════════
def test_representation_dispatch_arms(device, model) -> None:
    grid = InferenceServer(model, device, _cfg(), encoding_spec=_GRID_SPEC)
    assert grid._is_graph is False
    assert grid._shape == (BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    grid.stop()

    batcher = _FakeGraphBatcher(_wire_for(), n_batches=0)
    graph = _graph_server(device, batcher)
    assert graph._is_graph is True
    assert graph._shape is None
    assert graph._feature_len == 0
    graph.stop()


def test_run_dispatches_to_the_graph_loop_for_a_graph_spec(device) -> None:
    batcher = _FakeGraphBatcher(_wire_for(), n_batches=0)
    server = _graph_server(device, batcher)
    called: list[str] = []
    server._run_graph_loop = lambda: called.append("graph")  # type: ignore[method-assign]
    server.run()
    assert called == ["graph"], "a graph spec must route to the graph loop, not the dense one"


def test_unknown_representation_raises_at_construction(device, model) -> None:
    """AM-1 / LAW-11: there is no dense-by-default arm. A spec whose representation is
    unknown must raise, not quietly take the grid path."""

    @dataclass
    class _BadSpec:
        representation: str = "hexcanvas"
        policy_logit_count: int = 362
        trunk_size: int = 19
        n_planes: int = 8
        name: str = "bad"

    with pytest.raises((RepresentationMismatch, TypeError)):
        InferenceServer(model, device, _cfg(), encoding_spec=_BadSpec())


def test_non_spec_encoding_spec_type_rejected(device, model) -> None:
    with pytest.raises(TypeError, match="unrecognised encoding_spec type"):
        InferenceServer(model, device, _cfg(), encoding_spec={"representation": "grid"})


def test_grid_batcher_rejects_graph_methods() -> None:
    b = InferenceBatcher(
        feature_len=BOARD_CHANNELS * BOARD_SIZE * BOARD_SIZE, policy_len=N_ACTIONS
    )
    try:
        with pytest.raises(ValueError, match="RepresentationMismatch"):
            b.next_graph_batch(4, 5)
        with pytest.raises(ValueError, match="RepresentationMismatch"):
            b.spawn_mock_graph_games(1)
    finally:
        b.close()


# ══ F-09 — LAW-06 autocast-dtype wiring ══════════════════════════════════════════
def test_graph_amp_dtype_is_bf16_unconditionally(device) -> None:
    """LAW-06: bf16 on the graph path is pinned in CODE, so no DECLARED config value can
    flip it back to fp16 (fp16 GINE sum-aggregation overflows on production-scale graphs).
    WPSC Phase 3 SC-B3: `train.amp_dtype` is now a REQUIRED schema field (R1, no code
    default) — a config can no longer omit it, even for a graph run that ignores its
    value; the two remaining cases (fp16/bf16 declared) still prove the graph branch
    ignores whatever is declared."""
    for amp_knob in ("fp16", "bf16"):
        cfg = _cfg()
        cfg["train"] = {"amp_dtype": amp_knob}
        batcher = _FakeGraphBatcher(_wire_for(), n_batches=0)
        server = InferenceServer(
            _FiniteGraphNet(), device, cfg,
            batcher=batcher, encoding_spec=_GRAPH_SPEC,
        )
        assert server._amp_dtype is torch.bfloat16, f"amp_dtype={amp_knob!r} flipped graph"
        assert server._amp_dtype is amp_dtype_for("graph", amp_knob)
        server.stop()


@pytest.mark.parametrize(
    ("knob", "expected"), [("fp16", torch.float16), ("bf16", torch.bfloat16)]
)
def test_dense_amp_dtype_follows_the_knob(device, model, knob, expected) -> None:
    cfg = _cfg()
    cfg["train"] = {"amp_dtype": knob}
    server = InferenceServer(model, device, cfg, encoding_spec=_GRID_SPEC)
    assert server._amp_dtype is expected
    assert server._amp_dtype is amp_dtype_for("grid", knob)
    server.stop()


def test_dense_amp_dtype_requires_explicit_knob(device, model) -> None:
    """WPSC Phase 3 SC-B3 (R30b/R1): the grid branch's old implicit 'fp16' default is
    retired — omitting `train.amp_dtype` is now a hard KeyError, not a silent fallback."""
    cfg = _cfg()
    cfg["train"] = {}
    with pytest.raises(KeyError):
        InferenceServer(model, device, cfg, encoding_spec=_GRID_SPEC)


def test_dense_amp_dtype_matches_fixture_declared_value(device, model) -> None:
    # WPSC Phase 3 SC-B3: NOT a code-level default (R1 retires that) — `_cfg()`'s own
    # baseline `train.amp_dtype` happens to be "fp16"; this is a fixture-shape pin, not a
    # production fallback claim (see `test_dense_amp_dtype_requires_explicit_knob` above
    # for the actual no-fallback proof).
    server = InferenceServer(model, device, _cfg(), encoding_spec=_GRID_SPEC)
    assert server._amp_dtype is torch.float16
    server.stop()


# ══ F-10 — NaN/Inf on the graph path dies loud ═══════════════════════════════════
def test_nonfinite_graph_output_submits_failure_and_releases_waiters(
    device, monkeypatch
) -> None:
    batch = _hand_built_batch()
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: batch)

    batcher = _FakeGraphBatcher(_wire_for(), n_batches=1)
    server = _graph_server(device, batcher, model=_FiniteGraphNet(nonfinite=True))
    server.run()

    assert batcher.results == [], "a NaN forward must NOT be submitted as a result"
    assert len(batcher.failures) == 1
    ids, msg = batcher.failures[0]
    assert ids == [1, 2]
    assert msg.startswith("Graph inference failed: ")
    assert "NonFiniteModelOutput" in msg
    assert batcher.closed == 1


def test_finite_graph_output_submits_results(device, monkeypatch) -> None:
    """LAW-07 clean twin for F-10: the same harness with finite outputs submits RESULTS
    and no failure — the gate is not rejecting everything."""
    batch = _hand_built_batch()
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: batch)

    batcher = _FakeGraphBatcher(_wire_for(), n_batches=1)
    server = _graph_server(device, batcher)
    server.run()

    assert batcher.failures == []
    assert len(batcher.results) == 1
    ids, probs, offsets, values = batcher.results[0]
    assert ids == [1, 2]
    assert probs.dtype == np.float32
    assert offsets.dtype == np.int64
    assert values.dtype == np.float32
    # Ragged probs: one segment per graph, each summing to 1.
    assert probs.shape == (4,)
    assert probs[0:2].sum() == pytest.approx(1.0)
    assert probs[2:4].sum() == pytest.approx(1.0)
    assert server.forward_count == 1
    assert server.total_requests == 2


# ══ F-11 — heartbeat emission at dispatch (behaviour-neutral by default) ═════════
def test_graph_loop_emits_one_heartbeat_per_batch(device, monkeypatch) -> None:
    batch = _hand_built_batch()
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: batch)

    beats: list[str] = []
    batcher = _FakeGraphBatcher(_wire_for(), n_batches=3)
    server = _graph_server(device, batcher, heartbeat=beats.append)
    server.run()

    assert beats == ["inference_dispatch"] * 3


def test_graph_loop_default_heartbeat_none_emits_nothing(device, monkeypatch) -> None:
    batch = _hand_built_batch()
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: batch)

    batcher = _FakeGraphBatcher(_wire_for(), n_batches=2)
    server = _graph_server(device, batcher)
    assert server._heartbeat is None
    server.run()  # must not raise — the default sink is a true no-op
    assert len(batcher.results) == 2


def test_dense_loop_emits_one_heartbeat_per_batch(device, model) -> None:
    beats: list[str] = []
    feature_len = BOARD_CHANNELS * BOARD_SIZE * BOARD_SIZE
    batcher = _FakeDenseBatcher(feature_len, n_batches=2)
    server = InferenceServer(
        model, device, _cfg(inference_batch_size=4, trace_inference=False),
        batcher=batcher, encoding_spec=_GRID_SPEC, heartbeat=beats.append,
    )
    batcher.server = server
    server.run()

    assert len(batcher.results) == 2
    assert beats == ["inference_dispatch"] * 2


def test_dense_loop_default_heartbeat_none_emits_nothing(device, model) -> None:
    feature_len = BOARD_CHANNELS * BOARD_SIZE * BOARD_SIZE
    batcher = _FakeDenseBatcher(feature_len, n_batches=2)
    server = InferenceServer(
        model, device, _cfg(inference_batch_size=4, trace_inference=False),
        batcher=batcher, encoding_spec=_GRID_SPEC,
    )
    batcher.server = server
    assert server._heartbeat is None
    server.run()
    assert len(batcher.results) == 2


def test_heartbeat_not_emitted_for_a_failed_batch(device, monkeypatch) -> None:
    """Emission sits after a SUCCESSFUL submit, so a failing batch does not beat — the
    watchdog's liveness signal must track dispatch, not loop spins."""
    def _boom(*_a, **_kw):
        raise RuntimeError("collate exploded")

    monkeypatch.setattr(collate_mod, "collate_graph_batch", _boom)
    beats: list[str] = []
    batcher = _FakeGraphBatcher(_wire_for(), n_batches=1)
    server = _graph_server(device, batcher, heartbeat=beats.append)
    server.run()

    assert len(batcher.failures) == 1
    assert beats == []


# ══ F-12 — model_version bump attribution ════════════════════════════════════════
def test_load_state_dict_safe_bumps_model_version(device) -> None:
    net = _make_cnn(device, seed=5)
    server = InferenceServer(
        net, device, _cfg(trace_inference=False), encoding_spec=_GRID_SPEC
    )
    try:
        before = server.batcher.model_version
        server.load_state_dict_safe(net.state_dict())
        after_one = server.batcher.model_version
        server.load_state_dict_safe(net.state_dict())
        after_two = server.batcher.model_version
    finally:
        server.stop()
    assert after_one == before + 1
    assert after_two == before + 2


# ══ F-13 / F-14 — the wire seam obligations ══════════════════════════════════════
def test_wire_round_trips_to_assemble_and_completes() -> None:
    batcher = InferenceBatcher(encoding_spec=_GRAPH_SPEC)
    try:
        n = 4
        batcher.spawn_mock_graph_games(n)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not batcher.has_pending_graph_requests():
            time.sleep(0.01)
        time.sleep(0.3)
        ids, gw = batcher.next_graph_batch(batch_size=n, max_wait_ms=2000)
        assert len(ids) == n

        payload = graph_wire_from_rust(gw)
        collate_mod.collate_graph_batch(payload, device="cpu", semantic="full")

        lo = payload.legal_offsets.astype(np.int64)
        total_legal = int(lo[-1])
        probs = np.zeros(total_legal, dtype=np.float32)
        for g in range(len(ids)):
            s, e = int(lo[g]), int(lo[g + 1])
            probs[s:e] = 1.0 / float(e - s)
        values = np.zeros(len(ids), dtype=np.float32)
        batcher.submit_graph_inference_results(ids, probs, lo, values)

        done = time.monotonic() + 5.0
        while time.monotonic() < done and batcher.completed_graph_games() < n:
            time.sleep(0.01)
        assert batcher.completed_graph_games() == n
    finally:
        batcher.close()


def test_check_graph_request_seam_obligations() -> None:
    b = InferenceBatcher(encoding_spec=_GRAPH_SPEC)
    try:
        good = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
        b.check_graph_request(good, 1, 2)  # no raise — the clean twin
        with pytest.raises(ValueError, match="current_player"):
            b.check_graph_request(good, 2, 2)
        with pytest.raises(ValueError, match="moves_remaining"):
            b.check_graph_request(good, 1, 256)
        with pytest.raises(ValueError, match="coord"):
            b.check_graph_request([(2**31 - 1, 0, 1)], 1, 2)
        with pytest.raises(ValueError, match="player"):
            b.check_graph_request([(0, 0, 5)], 1, 2)
    finally:
        b.close()


# ══ F-15 — the production collate call site (M5 / A-1) ═══════════════════════════
class _CollateSpy:
    def __init__(self, batch: GraphBatch) -> None:
        self.batch = batch
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.batch


def _run_graph_loop_with_spy(
    device: torch.device,
    monkeypatch,
    spec: Any,
    *,
    batch_size: int,
    n_batches: int,
) -> tuple[_CollateSpy, list[int], InferenceServer]:
    """Drive the REAL graph loop with the collate call spied at its import site.

    The loop takes a function-local `from mantis.selfplay.graph_collate import …`, so the
    spy must replace the attribute on the SOURCE module — which is what makes the call
    site observable at all.
    """
    batch = _hand_built_batch()
    spy = _CollateSpy(batch)
    resets: list[int] = []
    monkeypatch.setattr(collate_mod, "collate_graph_batch", spy)
    monkeypatch.setattr(collate_mod, "reset_semantic_canary", lambda: resets.append(1))

    batcher = _FakeGraphBatcher(_wire_for(), n_batches=n_batches)
    server = _graph_server(device, batcher, batch_size=batch_size)
    # Post-ctor spec swap: the ctor only accepts a real registry spec, and the loop binds
    # `spec = self.encoding_spec` at loop entry, reading the four dims inline at each
    # collate call — so a swapped spec is observable exactly where it matters.
    server.encoding_spec = spec
    server.run()
    return spy, resets, server


def test_graph_loop_collate_call_pinned_production_kwargs(device, monkeypatch) -> None:
    """The production semantic mode is `"canary"`, with `canary_period == batch_size` and
    exactly ONE `reset_semantic_canary()` before the first batch.

    `"off"` would silently remove the geometric checks (ADV-7/8/9) from live self-play —
    the exact silent-corruption class this WP exists to kill — while every ⊕ oracle stayed
    green. `"full"` is the symmetric failure: a permanent per-batch geometry recompute on
    the hot path.
    """
    spy, resets, server = _run_graph_loop_with_spy(
        device, monkeypatch, _GRAPH_SPEC, batch_size=8, n_batches=3
    )

    assert len(spy.calls) == 3
    assert resets == [1], "reset_semantic_canary must run exactly once, before batch 1"
    for _args, kwargs in spy.calls:
        assert kwargs["expected_version"] == 1
        assert kwargs["semantic"] == "canary"
        assert kwargs["canary_period"] == int(server._batch_size) == 8
        assert kwargs["device"] == str(server.device)


def test_graph_loop_collate_dims_flow_from_the_spec(device, monkeypatch) -> None:
    """The anti-hard-coding arm.

    `gnn_axis_v1` is the only registered graph encoding and it is exactly 19/6/11/5 — the
    same values as `collate_graph_batch`'s own defaults — so a call site that ignored the
    spec and passed literals would pass every fixture-driven test. A NON-default
    spec-shaped stub makes the difference observable: the four dims must equal the STUB's
    values, which appear nowhere in the registry or in any default.
    """
    stub = _SpecStub()
    assert (stub.trunk_size, stub.win_length, stub.node_feat_dim, stub.edge_feat_dim) != (
        _GRAPH_SPEC.trunk_size,
        _GRAPH_SPEC.win_length,
        _GRAPH_SPEC.node_feat_dim,
        _GRAPH_SPEC.edge_feat_dim,
    ), "the stub must differ from the registry spec or this arm is vacuous"

    spy, _resets, _server = _run_graph_loop_with_spy(
        device, monkeypatch, stub, batch_size=8, n_batches=1
    )

    assert len(spy.calls) == 1
    _args, kwargs = spy.calls[0]
    assert kwargs["trunk_size"] == stub.trunk_size == 21
    assert kwargs["win_length"] == stub.win_length == 7
    assert kwargs["node_feat_dim"] == stub.node_feat_dim == 13
    assert kwargs["edge_feat_dim"] == stub.edge_feat_dim == 3


def test_graph_loop_canary_period_tracks_batch_size(device, monkeypatch) -> None:
    """`canary_period` is `int(batch_size)`, not a constant: a hard-coded 64 would change
    the geometric-check cadence on every non-64 batch size."""
    spy, _resets, server = _run_graph_loop_with_spy(
        device, monkeypatch, _GRAPH_SPEC, batch_size=13, n_batches=1
    )
    assert server._batch_size == 13
    assert spy.calls[0][1]["canary_period"] == 13
