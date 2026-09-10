"""Suite I (engine half) — `mantis.selfplay.inference_local.LocalInferenceEngine`.

IMPL-written (non-⊕). The four dense-decode invariants this suite was built around —
min-over-windows value, max-over-windows policy, the out-of-action-space skip and the
uniform fallback — were properties of the K-cluster decode, and went with the dense path
(R346(f)) together with the no-graph-server dense arm and the two model/spec-disagreement
arms. What remains is the graph branch: shape, legality of the argmax, the empty-batch
no-op and the close/idempotence contract on the server thread the engine owns.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis._engine import Board
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.selfplay.inference_local import LocalInferenceEngine
_GRAPH_SPEC = lookup("gnn_axis_v1")
_CPU = torch.device("cpu")
#: F-816-10 D-1: `fused_graph_caps` is REQUIRED and keyword-only on this class — it
#: hand-builds its `InferenceServer` config with no `RunConfig`, so the bound is
#: THREADED from a parent resolver and never hardcoded at the site. The engine below
#: gets a NON-BINDING pair (nothing here splits).
_CAPS = FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)


def _graph_engine() -> LocalInferenceEngine:
    torch.manual_seed(20260723)
    net = build_net(
        GnnArch(
            in_dim=_GRAPH_SPEC.node_feat_dim,
            edge_dim=_GRAPH_SPEC.edge_feat_dim,
            hidden=16,
            num_layers=1,
            policy_hidden=16,
            value_hidden=16,
        )
    ).to(_CPU)
    net.eval()
    return LocalInferenceEngine(net, _CPU, encoding_spec=_GRAPH_SPEC,
                                fused_graph_caps=_CAPS,
                                inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8, )


# I-02 — graph branch
def test_graph_infer_batch_no_attributeerror_and_correct_shape() -> None:
    engine = _graph_engine()
    try:
        policies, values = engine.infer_batch([Board()])
        assert len(policies) == 1
        assert len(values) == 1
        assert len(policies[0]) == _GRAPH_SPEC.policy_logit_count
        assert all(np.isfinite(policies[0]))
        assert np.isfinite(values[0])
        # Whole-board encoding at ply 0: no off-window drop, so the dense half alone is
        # already a distribution.
        assert abs(sum(policies[0]) - 1.0) < 1e-3
    finally:
        engine.close()


def test_graph_infer_batch_produces_a_legal_argmax_move() -> None:
    engine = _graph_engine()
    try:
        board = Board()
        policies, _values = engine.infer_batch([board])
        best_idx = int(np.argmax(policies[0]))
        legal_flat = {board.to_flat(q, r) for q, r in board.legal_moves()}
        assert best_idx in legal_flat
    finally:
        engine.close()


def test_graph_infer_batch_empty_boards_no_op() -> None:
    engine = _graph_engine()
    try:
        assert engine.infer_batch([]) == ([], [])
    finally:
        engine.close()


def test_graph_engine_close_stops_server_thread_and_is_idempotent() -> None:
    engine = _graph_engine()
    assert engine._graph_server is not None
    server_thread = engine._graph_server
    engine.infer_batch([Board()])  # warm the thread up first
    engine.close()
    assert engine._graph_server is None
    assert engine._graph_batcher is None
    assert not server_thread.is_alive()
    engine.close()  # idempotent — a second close must not raise
