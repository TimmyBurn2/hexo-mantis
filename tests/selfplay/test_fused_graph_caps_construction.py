"""Prove no graph construction site can omit the fused-forward caps.

`LocalInferenceEngine` builds its graph server from a hand-built dict with no `RunConfig`, so
the caps are THREADED there rather than hardcoded: a hardcoded cap would be a second authority
over one byte budget, on the arm that runs with its own allocator on the eval device. The caps
must also survive the eval process seam, or the child runs unbounded and the parent cannot tell.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

import _fused_graph_harness as H
from mantis.config.resolve.fused_graph_caps import (
    FusedGraphCapsSpec,
    MissingFusedGraphCapsError,
)
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.eval.rounds import GateSpec, RoundSpec
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


def test_fg6_04_a_graph_engine_cannot_be_built_without_the_caps() -> None:
    """Prove omitting the caps is a TypeError at the call, not a surprise inside an eval round."""
    net = build_net(GnnArch(in_dim=H.GRAPH_SPEC.node_feat_dim,
                            edge_dim=H.GRAPH_SPEC.edge_feat_dim, hidden=16, num_layers=1,
                            policy_hidden=16, value_hidden=16)).to(_CPU)
    net.eval()
    with pytest.raises(TypeError):
        LocalInferenceEngine(net, _CPU, encoding_spec=H.GRAPH_SPEC)  # type: ignore[call-arg]


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


def _round_spec_base() -> dict:
    """Build the `RoundSpec` field set minus the posture members and the caps member."""
    return dict(
        round_index=0, round_id="r1", step=1, candidate_snapshot="c.pt", best_snapshot=None, best_step=None,
        encoding="gnn_axis_v1", worker_device="cpu",
        gate=GateSpec(stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
                      screen_confirm_lo=0.44, deploy_sims=1, opening_book="b",
                      bootstrap_resamples=1, min_distinct_per_pair=1, seed_base=1,
                      run_gate=False),
        rung_jobs=[], random_floor_games=0, random_model_sims=1, sealbot_model_sims=1,
        seed_base=1, round_timeout_sec=1.0,
        result_path="r.json", progress_path="p.txt", ladder_bootstrap_resamples=1,
        ladder_bootstrap_ci_level=0.95, ladder_bootstrap_seed=1,
        game_record=None,
        ply_cap_adjudication=None, strength_floor=None,
    )


def test_fg6_08_the_round_spec_carries_the_caps_across_the_process_seam() -> None:
    """Prove the round spec rehydrates the caps to their dataclass across the process seam.

    A raw mapping would give the child an attribute error at the moment it bounds a forward.
    """
    assert "fused_graph_caps" in RoundSpec.__dataclass_fields__, (
        "`RoundSpec` carries no `fused_graph_caps` field — the resolved caps stop at the "
        "process boundary and the eval child (its OWN allocator, `eval.worker_device: cuda`) "
        "runs unbounded")
    caps = FusedGraphCapsSpec(max_fused_edges=4_500_000, max_fused_nodes=170_000)
    spec = RoundSpec(**_round_spec_base(), fused_graph_caps=caps, leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, max_plies=128, leaf_build_threads=1, concurrency=1,
                     inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10))
    back = RoundSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert isinstance(back.fused_graph_caps, FusedGraphCapsSpec), (
        f"the caps came back as {type(back.fused_graph_caps).__name__}, not the dataclass — "
        "the child would raise on the first attribute read")
    assert back.fused_graph_caps == caps
    assert back == spec


def test_fg6_08_a_grid_round_carries_none_across_the_same_seam() -> None:
    """Prove a grid round's `None` round-trips as `None`, not as a rehydration failure."""
    spec = RoundSpec(**_round_spec_base(), fused_graph_caps=None, leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, max_plies=128, leaf_build_threads=1, concurrency=1,
                     inference_batching=None)
    back = RoundSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert back.fused_graph_caps is None
    assert back == spec
