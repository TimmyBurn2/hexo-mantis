"""A4-3: `inference.compile_trunk` defaults to eager; the compiled trunk is server-private."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import _fused_graph_harness as H
import pytest
import torch
from _wire_geometry import geometry_kwargs

from mantis.config.loader import load_config
from mantis.config.resolve.compile_trunk import (
    MissingCompileTrunkError,
    resolve_compile_trunk,
)
from mantis.model.arch import GnnArchV2
from mantis.model.build import build_net
from mantis.selfplay.graph_collate import (
    GraphWirePayload,
    collate_graph_batch,
    stone_mask_from_batch,
)
from mantis.selfplay.inference_server import InferenceServer

_REPO = Path(__file__).resolve().parents[2]
_GEOMETRY = geometry_kwargs()


def test_the_schema_defaults_to_eager_through_the_one_loader() -> None:
    """The DEFAULT is read off a config that omits the row; run6 MINTS `true` (PERF-A4)."""
    config = load_config(_REPO / "configs" / "smoke_preflight_armed.yaml").model_dump()
    assert config["inference"]["compile_trunk"] is False
    assert resolve_compile_trunk(config) is False
    assert resolve_compile_trunk(load_config(_REPO / "configs" / "run6.yaml").model_dump()) is True


@pytest.mark.parametrize("bad", [{}, {"inference": {}}, {"inference": {"compile_trunk": "yes"}}])
def test_an_absent_or_non_bool_value_is_a_named_refusal(bad: dict[str, Any]) -> None:
    with pytest.raises(MissingCompileTrunkError):
        resolve_compile_trunk(bad)


def test_the_eager_server_hands_forward_batch_no_trunk() -> None:
    server = InferenceServer(H.SentinelGraphNet(), torch.device("cpu"), H.graph_cfg(),
                             batcher=H.ScriptedGraphBatcher([]), encoding_spec=H.GRAPH_SPEC)
    assert server._trunk is None
    block = server.batch_timing_snapshot()["compile"]
    assert block["enabled"] is False and block["unique_graphs"] == 0
    assert block["frames_total"] == 0 and block["frames_ok"] == 0


def _net_and_batch(payload_fields, device: str):
    torch.manual_seed(0)
    net = build_net(GnnArchV2(in_dim=_GEOMETRY["node_feat_dim"], hidden=32, num_layers=2,
                              edge_dim=_GEOMETRY["edge_feat_dim"], policy_hidden=16,
                              value_hidden=8, n_value_bins=65)).to(device).eval()
    batch = collate_graph_batch(GraphWirePayload(**payload_fields("b6")), device=device,
                                semantic="off", **_GEOMETRY)
    return net, batch


def _forward(net, batch, trunk=None):
    with torch.inference_mode():
        return net.forward_batch(batch.x, batch.edge_index, batch.edge_attr,
                                 batch.legal_node_gather, stone_mask_from_batch(batch),
                                 batch.node_offsets, trunk=trunk)


def test_a_trunk_override_leaves_the_shared_module_eager_and_untouched(payload_fields) -> None:
    net, batch = _net_and_batch(payload_fields, "cpu")
    before = _forward(net, batch)
    calls = []

    def spy(*args):
        calls.append(len(args))
        return net.representation(*args)

    via_spy = _forward(net, batch, trunk=spy)
    assert calls == [4], "the V2 forward hands the trunk (x, edge_index, edge_attr, normalize_mask)"
    after = _forward(net, batch)
    for a, b in zip(before, via_spy, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(before, after, strict=True):
        assert torch.equal(a, b), "the module's own forward must not change when a trunk was used"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LOUD SKIP — the compiled trunk is a CUDA lever")
def test_compiled_trunk_matches_eager_to_bf16_noise_and_leaves_the_module_eager(payload_fields) -> None:
    net, batch = _net_and_batch(payload_fields, "cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # The device's identical-code null: exactly 0 on the RTX 5080, bf16-noise on sm_86.
        eager = _forward(net, batch)
        null = max(float((a - b).abs().max()) for _ in range(3)
                   for a, b in zip(eager, _forward(net, batch), strict=True))
        compiled = torch.compile(net.representation, dynamic=True)
        fast = _forward(net, batch, trunk=compiled)
        eager_again = _forward(net, batch)
    assert not isinstance(net.representation, torch._dynamo.eval_frame.OptimizedModule)
    drift = max(float((a - b).abs().max()) for a, b in zip(eager, eager_again, strict=True))
    assert drift <= 2 * null, (
        f"compiling a server-private wrapper changed the module's forward: drift {drift:.3e} "
        f"against the device's own eager null {null:.3e}")
    probs_e = torch.softmax(eager[0].float(), dim=0)
    probs_c = torch.softmax(fast[0].float(), dim=0)
    d_probs = float((probs_e - probs_c).abs().max())
    d_value = float((eager[1].float() - fast[1].float()).abs().max())
    print(f"COMPILE-PARITY max|Δprobs| {d_probs:.3e} max|Δvalue| {d_value:.3e}")
    assert d_probs < 1e-2 and d_value < 1e-2, (d_probs, d_value)


def test_the_compiling_server_makes_the_recompile_limit_a_loud_failure() -> None:
    """Past `recompile_limit` Dynamo runs eager silently; the server makes it raise (red team 1)."""
    import torch._dynamo
    import torch._dynamo.config as dynamo_config

    class _TrunkedSentinel(H.SentinelGraphNet):
        def __init__(self) -> None:
            super().__init__()
            self.representation = torch.nn.Linear(4, 4)

    was = dynamo_config.fail_on_recompile_limit_hit, dynamo_config.recompile_limit
    try:
        dynamo_config.fail_on_recompile_limit_hit = False
        server = InferenceServer(_TrunkedSentinel(), torch.device("cpu"), H.graph_cfg(),
                                 batcher=H.ScriptedGraphBatcher([]), encoding_spec=H.GRAPH_SPEC,
                                 compile_trunk=True)
        assert server._trunk is not None
        assert dynamo_config.fail_on_recompile_limit_hit is True
        block = server.batch_timing_snapshot()["compile"]
        assert block["enabled"] is True and block["fail_on_recompile_limit_hit"] is True
        assert {"frames_total", "frames_ok", "unique_graphs", "recompile_limit"} <= set(block)
        # The mechanism itself: a limit of 1 and a second dtype must RAISE, not fall back.
        dynamo_config.recompile_limit = 1
        torch._dynamo.reset()
        trunk = torch.compile(torch.nn.Linear(4, 4).eval(), dynamic=True)
        with torch.inference_mode():
            trunk(torch.ones(3, 4))
            with pytest.raises(Exception, match="recompile_limit"):
                trunk(torch.ones(3, 4, dtype=torch.float64))
    finally:
        dynamo_config.fail_on_recompile_limit_hit, dynamo_config.recompile_limit = was
        torch._dynamo.reset()
