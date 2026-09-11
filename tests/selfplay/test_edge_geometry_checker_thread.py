"""R347(e): check 14 off the critical path, still 1-in-1, still detect-and-halt — driven over
the real loop, the real collate and the engine's verifier, with ADV-8's flip as the defect."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch

import _fused_graph_harness as H
from _wire_geometry import geometry_kwargs

from mantis.config.loader import load_config
from mantis.config.resolve.edge_geometry_check import (
    EDGE_GEOMETRY_CHECK_MODES,
    MissingEdgeGeometryCheckError,
    resolve_edge_geometry_check,
)
from mantis.selfplay.graph_collate import (
    EdgeAttrGeometryMismatch,
    EdgeGeometryCheck,
    GraphWirePayload,
)
from mantis.selfplay.inference_server import (
    _EDGE_GEOMETRY_QUEUE_DEPTH,
    InferenceServer,
    _EdgeGeometryChecker,
)
from mantis.selfplay.pool import WorkerPool

_REPO = Path(__file__).resolve().parents[2]
_GEOMETRY = geometry_kwargs()
_DEADLINE_SEC = 20.0



def test_the_schema_defaults_to_inline_through_the_one_loader() -> None:
    config = load_config(_REPO / "configs" / "run6.yaml").model_dump()
    assert config["inference"]["edge_geometry_check"] == "inline"
    assert resolve_edge_geometry_check(config) == "inline"


@pytest.mark.parametrize("bad", [{}, {"inference": {}}, {"inference": {"edge_geometry_check": "async"}}])
def test_an_absent_or_unknown_posture_is_a_named_refusal(bad: dict[str, Any]) -> None:
    with pytest.raises(MissingEdgeGeometryCheckError):
        resolve_edge_geometry_check(bad)


def test_the_server_refuses_a_posture_outside_the_closed_set() -> None:
    with pytest.raises(ValueError, match="edge_geometry_check"):
        InferenceServer(H.SentinelGraphNet(), torch.device("cpu"), H.graph_cfg(),
                        batcher=H.ScriptedGraphBatcher([]), encoding_spec=H.GRAPH_SPEC,
                        edge_geometry_check="async")
    assert set(EDGE_GEOMETRY_CHECK_MODES) == {"inline", "checker_thread"}



class _PacedBatcher(H.ScriptedGraphBatcher):
    """Serve the next pop only once every earlier pop is DISPATCHED and checked: no race."""

    def __init__(self, pops) -> None:
        super().__init__(pops)
        self.handed_out = 0

    def next_graph_batch(self, batch_size: int, max_wait_ms: float):
        assert self.server is not None
        checker = self.server._edge_geometry_checker
        deadline = time.monotonic() + _DEADLINE_SEC
        while len(self.results) + len(self.failures) < self.handed_out and time.monotonic() < deadline:
            time.sleep(0.005)
        served = len(self.results)  # only a SERVED pop hands the thread a check
        if checker is not None and served:
            while checker.processed < served and time.monotonic() < deadline:
                time.sleep(0.005)
        ids, payload = super().next_graph_batch(batch_size, max_wait_ms)
        if ids:
            self.handed_out += 1
        return ids, payload


def _payload(payload_fields, *, corrupt: bool) -> GraphWirePayload:
    fields = payload_fields("b6")
    if corrupt:
        fields["edge_attr"][3] = -fields["edge_attr"][3]  # ADV-8: a flipped signed distance
    return GraphWirePayload(**fields)


def _drive(tmp_path: Path, pops: list[GraphWirePayload], mode: str):
    batcher = _PacedBatcher(pops)
    server = InferenceServer(
        H.SentinelGraphNet(), torch.device("cpu"), H.graph_cfg(),
        batcher=batcher, encoding_spec=H.GRAPH_SPEC, collate_check_period=1,
        collate_dump=(str(tmp_path), lambda: {"round_id": "lever", "step": 3,
                                                "concurrency": 1, "phase": "selfplay"}),
        edge_geometry_check=mode,
    )
    batcher.server = server
    server.run()
    server.stop()
    return server, batcher


def _sidecars(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(tmp_path.glob("collate_dump_*.json"))]


def _lever(server: InferenceServer) -> dict[str, Any]:
    block = server.batch_timing_snapshot()["edge_geometry_check"]
    assert block is not None, "the lever publishes no block on the graph path (LAW-18)"
    return block


def test_clean_pops_are_served_and_every_check_runs_on_the_thread(tmp_path, payload_fields) -> None:
    pops = [_payload(payload_fields, corrupt=False) for _ in range(3)]
    server, batcher = _drive(tmp_path, pops, "checker_thread")
    assert len(batcher.results) == 3 and batcher.failures == []
    lever = _lever(server)
    assert lever == {"mode": "checker_thread", "deferred": 3, "inline_fallback": 0, "failures": 0}
    assert _sidecars(tmp_path) == [] and server.deferred_contract_failure is None


def test_a_planted_defect_is_served_once_then_halts_the_next_pop_with_the_dump(
    tmp_path, payload_fields,
) -> None:
    """One batch later: the defective pop is served, the dump lands, the NEXT pop is refused."""
    pops = [_payload(payload_fields, corrupt=True), _payload(payload_fields, corrupt=False)]
    server, batcher = _drive(tmp_path, pops, "checker_thread")
    assert len(batcher.results) == 1, "the defective pop is served before its check runs"
    assert len(batcher.failures) == 1, "the pop after the finding must be refused"
    _ids, message = batcher.failures[0]
    assert message.startswith("Graph inference failed: deferred edge-geometry"), message
    assert isinstance(server.deferred_contract_failure, EdgeAttrGeometryMismatch)
    sidecars = _sidecars(tmp_path)
    assert len(sidecars) == 1 and sidecars[0]["error_type"] == "EdgeAttrGeometryMismatch"
    assert sidecars[0]["finding"] == "F-816-37" and sidecars[0]["round_id"] == "lever"
    lever = _lever(server)
    assert lever["failures"] == 1 and lever["deferred"] == 1


def test_inline_refuses_the_same_defect_before_serving_it(tmp_path, payload_fields) -> None:
    """The control: under `inline` the defective pop is NOT served and the counters stay at 0."""
    pops = [_payload(payload_fields, corrupt=True), _payload(payload_fields, corrupt=False)]
    server, batcher = _drive(tmp_path, pops, "inline")
    assert len(batcher.failures) == 1 and len(batcher.results) == 1
    assert "signed" in batcher.failures[0][1].lower() or "geometry" in batcher.failures[0][1].lower()
    assert server.deferred_contract_failure is None
    assert _lever(server) == {"mode": "inline", "deferred": 0, "inline_fallback": 0, "failures": 0}
    assert len(_sidecars(tmp_path)) == 1



class _Blocking:
    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    def run(self) -> None:
        self._gate.wait(_DEADLINE_SEC)


class _Refusing:
    def run(self) -> None:
        raise EdgeAttrGeometryMismatch("planted: refused on the caller's thread")


def test_a_full_queue_runs_the_check_inline_and_a_dropped_check_would_be_caught(tmp_path) -> None:
    """Mutation: a checker that dropped on `queue.Full` leaves the fifth defect unlatched."""
    server = InferenceServer(
        H.SentinelGraphNet(), torch.device("cpu"), H.graph_cfg(),
        batcher=H.ScriptedGraphBatcher([]), encoding_spec=H.GRAPH_SPEC,
        collate_dump=(str(tmp_path), lambda: {}), edge_geometry_check="checker_thread",
    )
    checker = _EdgeGeometryChecker(server)  # never started: the queue only fills
    gate = threading.Event()
    for _ in range(_EDGE_GEOMETRY_QUEUE_DEPTH):
        checker.submit(_Blocking(gate), None, (0, 1))
    assert server._edge_geometry_deferred == _EDGE_GEOMETRY_QUEUE_DEPTH
    checker.submit(_Refusing(), None, (0, 1))
    assert server._edge_geometry_inline_fallback == 1
    assert isinstance(server.deferred_contract_failure, EdgeAttrGeometryMismatch)
    assert server._edge_geometry_failures == 1
    gate.set()


def test_stop_drains_queued_checks_on_the_callers_thread(tmp_path) -> None:
    server = InferenceServer(
        H.SentinelGraphNet(), torch.device("cpu"), H.graph_cfg(),
        batcher=H.ScriptedGraphBatcher([]), encoding_spec=H.GRAPH_SPEC,
        collate_dump=(str(tmp_path), lambda: {}), edge_geometry_check="checker_thread",
    )
    checker = _EdgeGeometryChecker(server)
    checker.submit(_Refusing(), None, (0, 1))
    assert server.deferred_contract_failure is None, "queued, not yet run"
    checker.drain_and_stop()
    assert isinstance(server.deferred_contract_failure, EdgeAttrGeometryMismatch)
    assert checker.processed == 1



class _Runner:
    def worker_panics(self) -> int:
        return 0


class _Server:
    def __init__(self, failure: BaseException | None) -> None:
        self.deferred_contract_failure = failure


class _Pool:
    check_producer_health = WorkerPool.check_producer_health

    def __init__(self, server: Any) -> None:
        self._runner = _Runner()
        self._producer_exc = None
        self._inference_server = server


def test_the_pools_health_check_halts_on_a_deferred_failure_and_passes_without_one() -> None:
    _Pool(_Server(None)).check_producer_health()
    with pytest.raises(RuntimeError, match="served batch") as excinfo:
        _Pool(_Server(EdgeAttrGeometryMismatch("planted"))).check_producer_health()
    assert isinstance(excinfo.value.__cause__, EdgeAttrGeometryMismatch)



def test_the_captured_check_is_the_same_call_the_inline_path_makes(payload_fields) -> None:
    """`EdgeGeometryCheck.run` IS check 14: clean passes, the ADV-8 flip raises by name."""
    from mantis.selfplay.graph_collate import collate_graph_batch

    clean = _payload(payload_fields, corrupt=False)
    sink: list[EdgeGeometryCheck] = []
    collate_graph_batch(clean, device="cpu", semantic="full", deferred_edge_geometry=sink,
                        **_GEOMETRY)
    assert len(sink) == 1, "one capture per collate, exactly when the inline check would run"
    sink[0].run()
    bad_sink: list[EdgeGeometryCheck] = []
    collate_graph_batch(_payload(payload_fields, corrupt=True), device="cpu", semantic="full",
                        deferred_edge_geometry=bad_sink, **_GEOMETRY)
    with pytest.raises(EdgeAttrGeometryMismatch):
        bad_sink[0].run()
    off: list[EdgeGeometryCheck] = []
    collate_graph_batch(clean, device="cpu", semantic="off", deferred_edge_geometry=off,
                        **_GEOMETRY)
    assert off == [], "no semantic layer, no capture: the sink never widens the check"
