"""Only the `'spawn'` multiprocessing context is accepted, and the refusal carries the reason.

Two independent structural reasons: the eval worker needs its OWN CUDA context, which a forked
child inherits poisoned; and `_worker_entry` arms `PR_SET_PDEATHSIG`, which the kernel signals on
the death of the thread that CREATED the child — under `'forkserver'` that thread belongs to the
forkserver process, so the arming would track the wrong process and fire early or never.

THE REFUSAL KEYS OFF THE NAME STRING, never the context object: frozen eval suites monkeypatch
`multiprocessing.get_context`, so a check that inspected the returned context would red them all.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.config.schema import EvalConfig, GateConfig
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks


def _eval_cfg() -> EvalConfig:
    gate = GateConfig(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625, sequential=None,
    )
    return EvalConfig(
        random_model_sims=96, max_plies=128, random_floor_games=4, worker_device="cpu",
        round_timeout_sec=5.0, worker_kill_grace_sec=1.0, gate=gate,
        ply_cap_adjudication=None, strength_floor=None,
    )


def _pipeline_kwargs(tmp_path: Path, **overrides: Any) -> dict:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(exist_ok=True)
    kwargs: dict[str, Any] = dict(
        eval_cfg=_eval_cfg(),
        coordinator_cfg_caps=DrainCaps(
            final_eval_drain_timeout_sec=5.0, eval_final_drain_safety_factor=1.0,
            eval_final_drain_hard_cap_sec=5.0, terminal_eval_hard_cap_sec=5.0,
        ),
        encoding="gnn_axis_v1",
        max_plies=128,
        c_visit=50.0, c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16,
        run_id="q3_mp_ctx_whitelist",
        spool_dir=spool_dir, game_record_dir=str(spool_dir) + "_games",
        promotion=DeployTagHooks(
            anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
            best_model_path=tmp_path / "best_model.pt",
            run_id="q3_mp_ctx_whitelist",
            encoding="gnn_axis_v1",
            save_anchor=lambda *a, **k: None,
            guarded_load=lambda *a, **k: None,
        ),
        fused_graph_caps=FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921),
        inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10),
    )
    kwargs.update(overrides)
    return kwargs


def test_a_forkserver_context_is_refused_at_construction(tmp_path) -> None:
    """`'forkserver'` is refused at CONSTRUCTION, not at first round, and with `ValueError` — a bad
    argument VALUE, deliberately distinct from the `RuntimeError` that reports a violated invariant."""
    with pytest.raises(ValueError, match="forkserver"):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx="forkserver"), leaf_batch_size=1)


def test_a_fork_context_is_refused_at_construction(tmp_path) -> None:
    """A mutant that special-cases only `'forkserver'` passes the row above and dies here, which
    forces the refusal to be a WHITELIST equality rather than a blacklist of known-bad names."""
    with pytest.raises(ValueError, match="fork"):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx="fork"), leaf_batch_size=1)


@pytest.mark.parametrize("bad", ["", "SPAWN", "spawn ", "threads"])
def test_any_other_context_name_is_refused_too(tmp_path, bad: str) -> None:
    """Empty string, wrong case, stray whitespace and an invented name are refused by the same
    equality, with no per-value arm to forget."""
    with pytest.raises(ValueError):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx=bad), leaf_batch_size=1)


def test_the_default_context_constructs(tmp_path) -> None:
    """The positive control, asserting past "no exception raised": a constructor that returned
    early would satisfy a bare does-not-raise check while breaking every round."""
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    try:
        assert pipeline._mp_ctx_name == "spawn"
        assert pipeline._work_dir.is_dir()
        assert pipeline._poller.is_alive(), "the constructed pipeline never started its poller"
    finally:
        pipeline.stop()


def test_the_refusal_names_the_only_supported_value(tmp_path) -> None:
    """The message must say what IS supported and why, not merely that the value was rejected."""
    with pytest.raises(ValueError) as caught:
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx="fork"), leaf_batch_size=1)
    message = str(caught.value)
    assert "'spawn'" in message
    assert "PR_SET_PDEATHSIG" in message
    assert "CUDA" in message
