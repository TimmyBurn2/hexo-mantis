"""Cover the one inference server: dispatch, the graph loop's failures and heartbeats, the wire
seam obligations and the production collate call site.

>300 justify: one server, one loop, one seam — every pin binds the SAME class and shares the fakes,
so splitting would separate the dispatch pin from what it dispatches to. The sharpest is the
anti-hard-coding arm: the one registered graph encoding carries exactly the collate defaults.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

import _fused_graph_harness as H
import mantis.selfplay.graph_collate as collate_mod
from _fused_graph_harness import (
    CountingGraphBatcher,
    FiniteGraphNet,
    hand_built_batch,
    server_cfg,
    wire_for,
)
from mantis._engine import InferenceBatcher
from mantis.model import RepresentationMismatch, amp_dtype_for
from mantis.selfplay.graph_collate import (
    GraphBatch,
    graph_wire_from_rust,
)
from mantis.selfplay.inference_server import InferenceServer


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class _SpecStub:
    """A non-default graph spec whose every dim differs from the registered one's 19/6/11/5, so
    a hard-coded literal at the collate call site cannot pass."""

    trunk_size: int = 21
    win_length: int = 7
    node_feat_dim: int = 13
    edge_feat_dim: int = 3
    representation: str = "graph"
    policy_logit_count: int = 442
    name: str = "spec_stub_non_default"


def _graph_server(
    device: torch.device,
    batcher: Any,
    model: torch.nn.Module | None = None,
    *,
    heartbeat=None,
    batch_size: int = 8,
) -> InferenceServer:
    net = model if model is not None else FiniteGraphNet()
    server = InferenceServer(
        net,
        device,
        server_cfg(inference_batch_size=batch_size),
        batcher=batcher,
        encoding_spec=H.GRAPH_SPEC,
        heartbeat=heartbeat,
    )
    batcher.server = server
    return server


def test_unknown_representation_raises_at_construction(device) -> None:
    """Prove an unknown representation raises at construction, with no dense-by-default arm."""

    @dataclass
    class _BadSpec:
        representation: str = "hexcanvas"
        policy_logit_count: int = 362
        trunk_size: int = 19
        n_planes: int = 8
        name: str = "bad"

    with pytest.raises((RepresentationMismatch, TypeError)):
        InferenceServer(FiniteGraphNet(), device, server_cfg(), encoding_spec=_BadSpec())


def test_non_spec_encoding_spec_type_rejected(device) -> None:
    with pytest.raises(TypeError, match="unrecognised encoding_spec type"):
        InferenceServer(FiniteGraphNet(), device, server_cfg(),
                        encoding_spec={"representation": "graph"})


def test_graph_amp_dtype_is_bf16_unconditionally(device) -> None:
    """Prove the server resolves bf16 from the representation alone: fp16 GINE sum-aggregation
    overflows on production-scale graphs, and a config row would be a second authority."""
    cfg = server_cfg()
    assert "amp_dtype" not in cfg.get("train", {}), (
        "a config row for the autocast dtype is the second authority LAW-06 refuses"
    )
    batcher = CountingGraphBatcher(wire_for(), [])
    server = InferenceServer(
        FiniteGraphNet(), device, cfg,
        batcher=batcher, encoding_spec=H.GRAPH_SPEC,
    )
    assert server._amp_dtype is torch.bfloat16
    assert server._amp_dtype is amp_dtype_for("graph")
    server.stop()


def test_nonfinite_graph_output_submits_failure_and_releases_waiters(
    device, monkeypatch
) -> None:
    batch = hand_built_batch()
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: batch)

    batcher = CountingGraphBatcher(wire_for(), [2])
    server = _graph_server(device, batcher, model=FiniteGraphNet(nonfinite=True))
    server.run()

    assert batcher.results == [], "a NaN forward must NOT be submitted as a result"
    assert len(batcher.failures) == 1
    ids, msg = batcher.failures[0]
    assert ids == [1, 2]
    assert msg.startswith("Graph inference failed: ")
    assert "NonFiniteModelOutput" in msg
    assert batcher.closed == 1


def test_finite_graph_output_submits_results(device, monkeypatch) -> None:
    """Prove the clean twin: the same harness with finite outputs submits results, no failure."""
    batch = hand_built_batch()
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: batch)

    batcher = CountingGraphBatcher(wire_for(), [2])
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


def test_graph_loop_emits_one_heartbeat_per_batch(device, monkeypatch) -> None:
    batch = hand_built_batch()
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: batch)

    beats: list[str] = []
    batcher = CountingGraphBatcher(wire_for(), [2, 2, 2])
    server = _graph_server(device, batcher, heartbeat=beats.append)
    server.run()

    assert beats == ["inference_dispatch"] * 3


def test_graph_loop_default_heartbeat_none_emits_nothing(device, monkeypatch) -> None:
    batch = hand_built_batch()
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: batch)

    batcher = CountingGraphBatcher(wire_for(), [2, 2])
    server = _graph_server(device, batcher)
    assert server._heartbeat is None
    server.run()  # must not raise — the default sink is a true no-op
    assert len(batcher.results) == 2


def test_heartbeat_not_emitted_for_a_failed_batch(device, monkeypatch) -> None:
    """Prove a failed batch emits no heartbeat: liveness tracks dispatch, not loop spins."""
    def _boom(*_a, **_kw):
        raise RuntimeError("collate exploded")

    monkeypatch.setattr(collate_mod, "collate_graph_batch", _boom)
    beats: list[str] = []
    batcher = CountingGraphBatcher(wire_for(), [2])
    server = _graph_server(device, batcher, heartbeat=beats.append)
    server.run()

    assert len(batcher.failures) == 1
    assert beats == []


def test_load_state_dict_safe_bumps_model_version(device) -> None:
    net = FiniteGraphNet()
    batcher = CountingGraphBatcher(wire_for(), [])
    server = InferenceServer(
        net, device, server_cfg(), batcher=batcher, encoding_spec=H.GRAPH_SPEC
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


def test_wire_round_trips_to_assemble_and_completes() -> None:
    batcher = InferenceBatcher(encoding_spec=H.GRAPH_SPEC)
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
            trunk_size=H.GRAPH_SPEC.trunk_size, win_length=H.GRAPH_SPEC.win_length,
            node_feat_dim=H.GRAPH_SPEC.node_feat_dim,
            edge_feat_dim=H.GRAPH_SPEC.edge_feat_dim,
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
    b = InferenceBatcher(encoding_spec=H.GRAPH_SPEC)
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
    """Drive the real graph loop with the collate spied on its SOURCE module — the loop imports
    it function-locally, so nothing else makes the call site observable."""
    batch = hand_built_batch()
    spy = _CollateSpy(batch)
    resets: list[int] = []
    monkeypatch.setattr(collate_mod, "collate_graph_batch", spy)
    monkeypatch.setattr(collate_mod, "reset_semantic_canary", lambda: resets.append(1))

    batcher = CountingGraphBatcher(wire_for(), [2] * n_batches)
    server = _graph_server(device, batcher, batch_size=batch_size)
    # Post-ctor swap: the ctor accepts only a real registry spec, and the loop reads the four
    # dims inline at each collate call, so the swap is observable exactly where it matters.
    server.encoding_spec = spec
    server.run()
    return spy, resets, server


def test_graph_loop_collate_call_pinned_production_kwargs(device, monkeypatch) -> None:
    """Pin the production collate kwargs: `"canary"`, `canary_period == batch_size`, one reset
    before the first batch. `"off"` drops the geometric checks with every other row green."""
    spy, resets, server = _run_graph_loop_with_spy(
        device, monkeypatch, H.GRAPH_SPEC, batch_size=8, n_batches=3
    )

    assert len(spy.calls) == 3
    assert resets == [1], "reset_semantic_canary must run exactly once, before batch 1"
    for _args, kwargs in spy.calls:
        assert kwargs["expected_version"] == 1
        assert kwargs["semantic"] == "canary"
        assert kwargs["canary_period"] == int(server._batch_size) == 8
        assert kwargs["device"] == str(server.device)


def test_graph_loop_collate_dims_flow_from_the_spec(device, monkeypatch) -> None:
    """Prove the collate dims flow from the spec, using values in no registry row or default."""
    stub = _SpecStub()
    assert (stub.trunk_size, stub.win_length, stub.node_feat_dim, stub.edge_feat_dim) != (
        H.GRAPH_SPEC.trunk_size,
        H.GRAPH_SPEC.win_length,
        H.GRAPH_SPEC.node_feat_dim,
        H.GRAPH_SPEC.edge_feat_dim,
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
    """Prove `canary_period` tracks the batch size rather than a constant."""
    spy, _resets, server = _run_graph_loop_with_spy(
        device, monkeypatch, H.GRAPH_SPEC, batch_size=13, n_batches=1
    )
    assert server._batch_size == 13
    assert spy.calls[0][1]["canary_period"] == 13
