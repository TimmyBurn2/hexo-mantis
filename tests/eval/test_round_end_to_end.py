"""Full headless eval round, end to end (integration tier).

Unlike the other `tests/eval/*.py` suites, which fake the subprocess boundary to stay fast,
this one runs the REAL out-of-process worker on CPU: no `multiprocessing.get_context` patch.

The net is built at dims READ OFF `_ENC`'s registry spec, because the worker runs inference
bound to the encoding the ROUND declared — the wire carries that geometry whatever the net was
built at. Since R362(c) a production round carries no rung job: the round is the floor probe,
the gate block (skipped here — no anchor) and the random floor, whose games are the ones played.

The routed result carries an ADDITIONAL `"worker_pid"` key beyond the superset-stable shape; it
is how this suite asserts eval inference is out-of-process without reaching into internals.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.config.schema import (
    EvalConfig,
    GateConfig,
    PlyCapAdjudicationConfig,
)
from mantis.config.loader import load_config
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import resolve_inference_batching
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net

pytestmark = pytest.mark.integration


#: A DENSE encoding at radius 8, not radius-5 `v6`: the round replays real openings from
#: `book_v1_s20260625_p4`, most of which need radius >= 6. Under `v6` the round dies in the
#: eval CHILD with `IllegalOpeningError`, surfacing only as `EXIT_NONZERO`.
_ENC = "gnn_axis_v1"
_REPO = Path(__file__).resolve().parents[2]
_SMOKE = load_config(_REPO / "configs" / "smoke_preflight_armed.yaml").model_dump()


def _tiny_model(*, weight_seed: int) -> torch.nn.Module:
    # Registry-TRUE dims, DERIVED from the spec rather than written as literals, so the wire and
    # the net cannot drift apart when the encoding moves. `weight_seed` is deterministic-but
    # -different per round because at `n_sims=4` a decisive game is a high-variance event.
    torch.manual_seed(weight_seed)
    spec = lookup(_ENC)
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch
    return net


def _eval_cfg(*, adjudicate: bool = False) -> EvalConfig:
    gate = GateConfig(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625, sequential=None,
    )
    return EvalConfig(
        random_model_sims=4, max_plies=128, random_floor_games=2, worker_device="cpu",
        round_timeout_sec=600.0, worker_kill_grace_sec=5.0, gate=gate,
        ply_cap_adjudication=(
            PlyCapAdjudicationConfig(criterion="longest_run_margin", min_margin=1)
            if adjudicate else None
        ),
        strength_floor=None,
    )


def _promotion_hooks(tmp_path: Path) -> DeployTagHooks:
    from types import SimpleNamespace

    return DeployTagHooks(
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        best_model_path=tmp_path / "best_model.pt",
        run_id="oracle_e2e_run",
        encoding=_ENC,
        save_anchor=lambda *a, **k: None,
        guarded_load=lambda *a, **k: None,
    )


def _build_pipeline(tmp_path: Path, *, adjudicate: bool = False):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(exist_ok=True)
    return build_eval_pipeline(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16, max_plies=128,
        eval_cfg=_eval_cfg(adjudicate=adjudicate),
        coordinator_cfg_caps=DrainCaps(
            final_eval_drain_timeout_sec=600.0,
            eval_final_drain_safety_factor=1.0,
            eval_final_drain_hard_cap_sec=600.0,
            terminal_eval_hard_cap_sec=600.0,
        ),
        encoding=_ENC,
        run_id="oracle_e2e_run",
        spool_dir=spool_dir, game_record_dir=str(spool_dir) + "_games",
        promotion=_promotion_hooks(tmp_path),
        # Both graph specs are REQUIRED since the grid arm's `None` was deleted; the fused bound
        # is the smoke's own minted caps as a spec, since the tiny oracle net is not its arch.
        fused_graph_caps=FusedGraphCapsSpec(**_SMOKE["inference"]["fused_graph_caps"]),
        inference_batching=resolve_inference_batching(_SMOKE),
    )


def _poll_until_complete(pipeline, *, timeout: float) -> dict:
    """Poll for the round result. The ceilings here are RUNAWAY bounds, never timing claims —
    a real subprocess needs real seconds, and 600 s bounds runaway while no plausible load
    contamination reaches it. Healthy rounds complete in seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = pipeline.poll_completed()
        if result is not None:
            return result
        time.sleep(0.1)
    pytest.fail(f"eval round did not complete within {timeout}s")


def test_full_headless_round_end_to_end(tmp_path) -> None:
    pipeline = _build_pipeline(tmp_path)
    try:
        ack = pipeline.run_evaluation(_tiny_model(weight_seed=20260625), 1000, None,
                                       full_config={}, best_model_step=None)
        assert ack["kicked"] is True

        result = _poll_until_complete(pipeline, timeout=600.0)
        assert result["eval_broken_reason"] is None
        assert isinstance(result["promoted"], bool)   # gate decision present in the routed shape
        assert result["wr_random"] is not None and 0.0 <= result["wr_random"] <= 1.0, (
            "the random floor played its games and its win rate reached the routed result"
        )
        assert "worker_pid" in result
        assert result["worker_pid"] != os.getpid(), "eval inference must run out-of-process"
    finally:
        pipeline.stop()


def test_a_second_round_with_an_armed_adjudicator_completes_on_the_same_pipeline(tmp_path) -> None:
    """Two rounds on one pipeline, the ply-cap adjudicator ARMED so capped games resolve: the
    round counter advances and each round routes its own result."""
    pipeline = _build_pipeline(tmp_path, adjudicate=True)
    try:
        pipeline.run_evaluation(_tiny_model(weight_seed=42), 1000, None,
                                 full_config={}, best_model_step=None)
        result1 = _poll_until_complete(pipeline, timeout=600.0)
        pipeline.run_evaluation(_tiny_model(weight_seed=1337), 2000, None,
                                 full_config={}, best_model_step=None)
        result2 = _poll_until_complete(pipeline, timeout=600.0)
        assert (result1["step"], result2["step"]) == (1000, 2000)
        assert result1["round_id"] != result2["round_id"]
        assert result1["eval_broken_reason"] is None and result2["eval_broken_reason"] is None
    finally:
        pipeline.stop()
