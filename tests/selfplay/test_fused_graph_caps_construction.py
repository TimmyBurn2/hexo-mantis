"""Prove no graph construction site can omit the fused-forward caps.

`LocalInferenceEngine` builds its graph server from a hand-built dict with no `RunConfig`, so
the caps are THREADED there rather than hardcoded: a hardcoded cap would be a second authority
over one byte budget, on the arm that runs with its own allocator on the eval device. The caps'
survival across the eval process seam is pinned by test_inference_batching_threaded.py's
round-spec round trip.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

import _fused_graph_harness as H
from mantis.config.resolve.fused_graph_caps import (
    FusedGraphCapsSpec,
    MissingFusedGraphCapsError,
)
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.model import GnnArch, build_net
from mantis.selfplay.inference_local import LocalInferenceEngine
from mantis.selfplay.inference_server import InferenceServer

_CPU = torch.device("cpu")
_REPO = Path(__file__).resolve().parents[2]
_INFERENCE_LOCAL = _REPO / "src" / "mantis" / "selfplay" / "inference_local.py"


class _DummyBatcher:
    def close(self) -> None:
        return None


def test_fg6_01_a_graph_server_cannot_be_built_without_a_caps_value() -> None:
    """Prove a graph server with neither a config block nor an explicit value raises by name.

    After this holds an unbounded fused graph forward is unconstructible, not just discouraged.
    """
    with pytest.raises(MissingFusedGraphCapsError) as exc:
        InferenceServer(
            H.SentinelGraphNet(), _CPU, H.graph_cfg(omit_block=True),
            batcher=_DummyBatcher(), encoding_spec=H.GRAPH_SPEC,
        )
    assert "inference.fused_graph_caps" in str(exc.value)


def test_fg6_01_an_explicit_caps_argument_is_honoured_over_an_absent_block() -> None:
    """Prove an explicit caps argument is honoured when the config block is absent."""
    caps = FusedGraphCapsSpec(max_fused_edges=4_500_000, max_fused_nodes=170_000)
    server = InferenceServer(
        H.SentinelGraphNet(), _CPU, H.graph_cfg(omit_block=True),
        batcher=_DummyBatcher(), encoding_spec=H.GRAPH_SPEC, fused_graph_caps=caps,
    )
    assert server.batch_timing_snapshot()["fusion"]["caps"] == {
        "max_fused_edges": 4_500_000, "max_fused_nodes": 170_000}


def test_fg6_03_the_local_engine_takes_a_required_keyword_only_caps_parameter() -> None:
    """Prove the caps parameter is required and keyword-only, with no default.

    A default here is a value nobody minted, on the one construction path with no config to mint
    it from; required makes absent unconstructible, which pyright catches before a worker spawns.
    """
    params = inspect.signature(LocalInferenceEngine.__init__).parameters
    assert "fused_graph_caps" in params, (
        "`LocalInferenceEngine` takes no `fused_graph_caps` parameter — the eval-side graph "
        "server is still built from a hand-made dict with no bound (D-1 BLOCKER)")
    caps = params["fused_graph_caps"]
    assert caps.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"`fused_graph_caps` must be keyword-only; it is {caps.kind}")
    assert caps.default is inspect.Parameter.empty, (
        f"`fused_graph_caps` carries a default ({caps.default!r}). A default is a SECOND "
        "authority over one byte budget, on the path that has no config to be the first")


def test_fg6_06_the_threaded_caps_reach_the_engines_own_server() -> None:
    """Prove the threaded caps reach the engine's own server, not merely get stored."""
    net = build_net(GnnArch(in_dim=H.GRAPH_SPEC.node_feat_dim,
                            edge_dim=H.GRAPH_SPEC.edge_feat_dim, hidden=16, num_layers=1,
                            policy_hidden=16, value_hidden=16)).to(_CPU)
    net.eval()
    caps = FusedGraphCapsSpec(max_fused_edges=1_234_567, max_fused_nodes=89_012)
    engine = LocalInferenceEngine(net, _CPU, encoding_spec=H.GRAPH_SPEC,
                                  fused_graph_caps=caps,
                                  inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8,
                                  )
    try:
        assert engine._graph_server is not None
        assert engine._graph_server.batch_timing_snapshot()["fusion"]["caps"] == {
            "max_fused_edges": 1_234_567, "max_fused_nodes": 89_012}, (
            "the threaded caps did not reach the engine's own server — the eval arm is still "
            "unbounded, which is the arm design §2.2 shows OOM'd")
    finally:
        engine.close()


def test_fg6_07_no_cap_value_is_hardcoded_at_the_standalone_construction_site() -> None:
    """Prove no integer literal is assigned to either caps member at the standalone site.

    A hardcoded cap is invisible to every behavioural row: the run would be bounded, at a number
    nobody measured, on the arm with its own allocator.
    """
    tree = ast.parse(_INFERENCE_LOCAL.read_text(encoding="utf-8"))
    members = {"max_fused_edges", "max_fused_nodes"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if (isinstance(key, ast.Constant) and key.value in members
                    and isinstance(value, ast.Constant)):
                raise AssertionError(
                    f"{key.value} is hardcoded to {value.value!r} in "
                    f"{_INFERENCE_LOCAL.relative_to(_REPO)}. D-1 refused this: the caps are "
                    "THREADED from the parent's resolver, never written at this site.")
        for key, value in zip(node.keys, node.values, strict=False):
            if isinstance(key, ast.Constant) and key.value == "fused_graph_caps":
                raise AssertionError(
                    "`fused_graph_caps` is written into the hand-built config dict at "
                    f"{_INFERENCE_LOCAL.relative_to(_REPO)}. The spec is threaded as a "
                    "resolver-produced dataclass (the `RoundSpec` precedent), not smuggled "
                    "back through a config-shaped literal.")
