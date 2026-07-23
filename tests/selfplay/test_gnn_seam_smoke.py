"""Suite F-07 — GNN inference-seam end-to-end smoke (INTEGRATION tier).

IMPL-written (non-⊕). Drives the LIVE graph seam through every production component the
self-play worker's graph leaf evaluation rides:

    engine builds axis graphs (native builder, via the seam guards)
      → `next_graph_batch` block-diagonal fuse
      → `collate_graph_batch` — the ONE resolver, full 18-assertion contract
      → `GnnNet.forward_batch`
      → per-graph segment softmax
      → `submit_graph_inference_results`
      → the Rust legal-set assemble wakes the blocked mock games

Exit criterion (DESIGN §b F-07): `completed_graph_games == n` within the timeout, with no
thread left running. Multi-thread end-to-end, so it lives on the integration tier
(`make test.integration`), not the default tier.

The old suite's variant of this test loaded a banked BC checkpoint from disk; that
checkpoint is not a repo artifact (R7 — no tracked weights), so the port drives the same
seam with mock graph games and a small randomly-initialised `GnnNet`. The seam, not the
checkpoint, is what this smoke gates.
"""
from __future__ import annotations

import time

import pytest
import torch

from mantis._engine import InferenceBatcher
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.selfplay.inference_server import InferenceServer

pytestmark = pytest.mark.integration

_SPEC = lookup("gnn_axis_v1")


def _tiny_gnn(device: torch.device) -> torch.nn.Module:
    torch.manual_seed(20260723)
    net = build_net(
        GnnArch(
            in_dim=_SPEC.node_feat_dim,
            edge_dim=_SPEC.edge_feat_dim,
            hidden=16,
            num_layers=1,
            policy_hidden=16,
            value_hidden=16,
        )
    ).to(device)
    net.eval()
    return net


def test_gnn_inference_seam_end_to_end_smoke() -> None:
    """CPU is the reference device (fp32, deterministic); the identical seam runs on CUDA
    under bf16 autocast when one is present."""
    device = torch.device("cpu")
    net = _tiny_gnn(device)

    batcher = InferenceBatcher(encoding_spec=_SPEC)
    assert batcher.representation_py == "graph", "the batcher must be a graph batcher"

    server = InferenceServer(
        net, device,
        {"selfplay": {"inference_batch_size": 8, "inference_max_wait_ms": 10}},
        batcher=batcher, encoding_spec=_SPEC,
    )
    assert server._is_graph is True
    server.start()
    try:
        n = 8
        batcher.spawn_mock_graph_games(n)

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and batcher.completed_graph_games() < n:
            time.sleep(0.02)

        completed = batcher.completed_graph_games()
        assert completed == n, (
            f"graph seam completed {completed}/{n} mock games before the timeout — the "
            "dispatch loop did not carry every batch through collate → forward → submit"
        )
        # The loop actually ran (rather than the games completing some other way).
        assert server.forward_count > 0
        assert server.total_requests >= n
    finally:
        server.stop()
        server.join(timeout=10.0)

    assert not server.is_alive(), "the dispatcher thread leaked past stop()/join()"
