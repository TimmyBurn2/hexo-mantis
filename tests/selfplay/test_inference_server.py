"""Suite F — the ONE inference server (`mantis.selfplay.inference_server`).

>300 justify: one server, one loop, one seam. The representation dispatch (F-08/F-09), the
graph-loop failure + heartbeat + collate call-site pins (F-10/F-11/F-15) and the wire/seam
obligations (F-13/F-14) all bind the SAME class and share the fakes; splitting them would
duplicate the fakes and separate the dispatch pin from the thing it dispatches to.

IMPL-written (non-⊕) per DESIGN §b: ports of the old server suites, rewritten
public-surface against `build_net`-built nets. The dense-loop assertions (F-01..F-05), the
TorchScript-trace and compile paths, and the dense halves of F-08/F-09/F-11 went with the
grid path (R346(f)) — `run()` now delegates to `_run_graph_loop` and there is no second arm.

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
from mantis.model import RepresentationMismatch, amp_dtype_for
from mantis.selfplay.graph_collate import (
    GraphBatch,
    GraphWirePayload,
    graph_wire_from_rust,
)
from mantis.selfplay.inference_server import InferenceServer
_GRAPH_SPEC = lookup("gnn_axis_v1")

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


def _cfg(**over: Any) -> dict[str, Any]:
    # `InferenceHParams.from_config` reads `config["inference"]`, and `resolve_from_config`
    # requires an explicit `encoding` key (R28, no default).
    base = {
        "inference_batch_size": 8, "inference_max_wait_ms": 20.0,
        # F-816-10 (R276(f)): the GRAPH arm resolves `inference.fused_graph_caps`
        # EAGERLY at construction, so every graph-route site built from this base needs
        # it. The pair is the template's NON-BINDING-BY-CONSTRUCTION value — nothing in
        # this file splits, and the split's own coverage lives in the F-816-10 oracles
        # where its M is asserted. Inert on the grid route, which never reads the block.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }
    base.update(over)
    return {"inference": base, "encoding": "gnn_axis_v1"}


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

    def forward_batch(self, x, edge_index, edge_attr, legal_index, stone_mask, node_offsets):
        self.calls.append(tuple(x.shape))
        n_legal = int(legal_index.numel())  # R284 P-MASK: rows, not a dense mask
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
        legal_offsets=legal_offsets,
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


def test_run_dispatches_to_the_graph_loop_for_a_graph_spec(device) -> None:
    batcher = _FakeGraphBatcher(_wire_for(), n_batches=0)
    server = _graph_server(device, batcher)
    called: list[str] = []
    server._run_graph_loop = lambda: called.append("graph")  # type: ignore[method-assign]
    server.run()
    assert called == ["graph"], "a graph spec must route to the graph loop, not the dense one"


def test_unknown_representation_raises_at_construction(device) -> None:
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
        InferenceServer(_FiniteGraphNet(), device, _cfg(), encoding_spec=_BadSpec())


def test_non_spec_encoding_spec_type_rejected(device) -> None:
    with pytest.raises(TypeError, match="unrecognised encoding_spec type"):
        InferenceServer(_FiniteGraphNet(), device, _cfg(),
                        encoding_spec={"representation": "graph"})


# ══ F-09 — LAW-06 autocast-dtype wiring ══════════════════════════════════════════
def test_graph_amp_dtype_is_bf16_unconditionally(device) -> None:
    """LAW-06: bf16 on the graph path is pinned in CODE (fp16 GINE sum-aggregation overflows
    on production-scale graphs). R346(f) deleted `train.amp_dtype`, so there is no longer a
    declared value that could disagree — the pin is now that the server resolves bf16 from
    the representation alone, with nothing in the config to consult."""
    cfg = _cfg()
    assert "amp_dtype" not in cfg.get("train", {}), (
        "a config row for the autocast dtype is the second authority LAW-06 refuses"
    )
    batcher = _FakeGraphBatcher(_wire_for(), n_batches=0)
    server = InferenceServer(
        _FiniteGraphNet(), device, cfg,
        batcher=batcher, encoding_spec=_GRAPH_SPEC,
    )
    assert server._amp_dtype is torch.bfloat16
    assert server._amp_dtype is amp_dtype_for("graph")
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
    net = _FiniteGraphNet()
    batcher = _FakeGraphBatcher(_wire_for(), n_batches=0)
    server = InferenceServer(
        net, device, _cfg(), batcher=batcher, encoding_spec=_GRAPH_SPEC
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
        collate_mod.collate_graph_batch(
            payload, device="cpu", semantic="full",
            trunk_size=_GRAPH_SPEC.trunk_size, win_length=_GRAPH_SPEC.win_length,
            node_feat_dim=_GRAPH_SPEC.node_feat_dim,
            edge_feat_dim=_GRAPH_SPEC.edge_feat_dim,
        )

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
