"""The builder's own verify is skipped exactly where the serving collate runs check 14 on every batch before the launch."""
from __future__ import annotations

import threading
from typing import Any

import _fused_graph_harness as H
import pytest
import torch
from _fused_caps import CAPS

from mantis import _engine
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.selfplay.inference_local import LocalInferenceEngine
from mantis.selfplay.inference_server import InferenceServer

_POSITIONS: list[tuple[list[tuple[int, int, int]], int, int]] = [
    ([(0, 0, 1)], -1, 2),
    ([(0, 0, 1), (1, 0, -1)], 1, 1),
    ([(0, 0, 1), (1, 0, -1), (0, 1, 1), (2, 0, -1)], 1, 2),
]


def _engine_at(period: int | None) -> LocalInferenceEngine:
    """A tiny net served on the CPU by an engine whose collate checks at `period`."""
    spec = lookup("gnn_axis_v1")
    arch = GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, hidden=16, num_layers=2)
    net = build_net(arch)
    net.eval()
    return LocalInferenceEngine(
        net, torch.device("cpu"), encoding_spec=spec, fused_graph_caps=CAPS,
        inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10),
        max_in_flight=len(_POSITIONS), submitters=1, collate_check_period=period,
    )


@pytest.mark.parametrize(("period", "posture", "every_batch"), [
    (1, "inline", True), (1, "checker_thread", False), (64, "inline", False), (None, "inline", False),
])
def test_only_a_period_one_inline_server_checks_every_batch_before_launch(
    period: int | None, posture: str, every_batch: bool,
) -> None:
    """A checker thread runs check 14 after the launch and a longer period skips batches, so neither licenses the skip."""
    server = InferenceServer(H.SentinelGraphNet(), torch.device("cpu"), H.graph_cfg(),
                             batcher=H.ScriptedGraphBatcher([]), encoding_spec=H.GRAPH_SPEC,
                             collate_check_period=period, edge_geometry_check=posture)
    assert server.checks_every_batch_before_launch is every_batch


@pytest.mark.parametrize(("period", "skipped"), [(1, True), (64, False), (None, False)])
def test_only_a_period_one_engine_builds_its_leaves_without_the_producer_verify(
    period: int | None, skipped: bool,
) -> None:
    """Read off the builder's own count of skipped verifies, so a break anywhere from the engine to the builder reds."""
    engine = _engine_at(period)
    try:
        before = _engine.unverified_graph_builds()
        row_before = engine.batch_timing_snapshot()["unverified_graph_builds"]
        dense, _overflow, values, _centers = engine.infer_positions_ls(_POSITIONS)
        delta = _engine.unverified_graph_builds() - before
        row_delta = engine.batch_timing_snapshot()["unverified_graph_builds"] - row_before
    finally:
        engine.close()
    assert len(dense) == len(values) == len(_POSITIONS)
    assert delta == (len(_POSITIONS) if skipped else 0), f"period {period}: {delta} unverified builds"
    assert row_delta == delta, "the serving snapshot carries the builder's own count"


def test_a_bare_batcher_call_keeps_the_producer_verify() -> None:
    """The skip is a keyword the caller names; a call that does not name it verifies every leaf."""
    batcher = _engine.InferenceBatcher(encoding_spec=_engine.RegistrySpec.from_registry("gnn_axis_v1"))
    served: dict[str, Any] = {}
    before = _engine.unverified_graph_builds()

    def serve() -> None:
        ids, _wire = batcher.next_graph_batch(len(_POSITIONS), 5_000)
        batcher.submit_graph_inference_failure(list(ids), "refused by the posture test")
        served["ids"] = list(ids)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    with pytest.raises(ValueError, match="refused by the posture test"):
        batcher.submit_graphs_and_wait_ls(_POSITIONS, 1)
    thread.join(timeout=30.0)
    batcher.close()
    assert len(served["ids"]) == len(_POSITIONS)
    assert _engine.unverified_graph_builds() == before
