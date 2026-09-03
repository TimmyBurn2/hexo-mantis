"""F-816-20 item 2 — only the `'spawn'` multiprocessing context is accepted, and the
refusal is where the reason now lives.

TWO independent structural reasons, both previously written only in a module docstring where
no test could cross them:

  1. the eval worker needs its OWN CUDA context, and a forked child inherits a poisoned one;
  2. `_worker_entry` arms `PR_SET_PDEATHSIG` (F-816-14) and the kernel signals on the death of
     the thread that CREATED the child — under `'forkserver'` that is a thread of the
     forkserver process, not the trainer, so the arming would track the wrong process and
     either fire early or never.

`mp_ctx_name` had NO caller anywhere in the tree (`grep -rn "mp_ctx"` outside `pipeline.py`:
nothing), which made deleting it the obvious move under LAW-08. It is retained and CHECKED
instead: the refusal is the parameter's live consumer, it is testable where a comment is not,
and it puts the explanation exactly where a future caller will be standing.

THE REFUSAL KEYS OFF THE NAME STRING, never the context object, and that is a binding
constraint rather than a preference: five frozen eval suites monkeypatch
`multiprocessing.get_context` and take the `'spawn'` default, so a check that inspected the
returned context would red all five.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.config.schema import EvalConfig, GateConfig, LadderConfig, LadderRung
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks


# ── fixtures, self-contained (R5 bars importing another test module) ─────────────────────
def _eval_cfg() -> EvalConfig:
    rungs = [
        LadderRung(name="sealbot_d5", bot="sealbot", variant="d5", depth=5, opponent_sims=None,
                   opening_book="book_v1_s20260625_p4", deploy_matched=True, games_max=32),
    ]
    gate = GateConfig(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625,
    )
    ladder = LadderConfig(
        rungs=rungs, round_games=64, min_games_per_active_rung=4,
        graduation_wr_lower_ci=0.75, graduation_consec_rounds=3, activation_wr_lower_ci=0.65,
        calibration_every_k_rounds=4, calibration_games=8, bootstrap_resamples=1000,
        bootstrap_ci_level=0.95, bt_prior_games=1.0, bootstrap_seed=1234,
    )
    return EvalConfig(
        random_model_sims=96, sealbot_model_sims=128, kraken_model_sims=128,
        strix_model_sims=128, random_floor_games=4, worker_device="cpu",
        round_timeout_sec=5.0, worker_kill_grace_sec=1.0, gate=gate, ladder=ladder,
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
        encoding="v6_live2_ls",
        amp_dtype="bf16",
        max_plies=128,
        c_visit=50.0, c_scale=1.0,
        run_id="q3_mp_ctx_whitelist",
        spool_dir=spool_dir,
        ladder_state_path=tmp_path / "ladder_state.json",
        promotion=DeployTagHooks(
            anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
            best_model_path=tmp_path / "best_model.pt",
            run_id="q3_mp_ctx_whitelist",
            encoding="v6_live2_ls",
            save_anchor=lambda *a, **k: None,
            guarded_load=lambda *a, **k: None,
        ),
        fused_graph_caps=None,
        inference_batching=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_a_forkserver_context_is_refused_at_construction(tmp_path) -> None:
    """`'forkserver'` is refused, at CONSTRUCTION and not at first round: a round is a long way
    into a run, and a refusal that arrives thirty minutes in is a worse instrument than one
    that arrives at boot.

    `ValueError` — a bad argument VALUE — deliberately distinct from item 1's `RuntimeError`,
    which reports a violated invariant."""
    with pytest.raises(ValueError, match="forkserver"):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx="forkserver"), leaf_batch_size=1)


def test_a_fork_context_is_refused_at_construction(tmp_path) -> None:
    """THE FLIP-SET BOUNDARY ROW (R71/R72). A mutant that special-cases only `'forkserver'`
    passes the row above and dies here — which is what forces the refusal to be a WHITELIST
    equality (`!= "spawn"`) rather than a blacklist of the two known-bad names."""
    with pytest.raises(ValueError, match="fork"):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx="fork"), leaf_batch_size=1)


@pytest.mark.parametrize("bad", ["", "SPAWN", "spawn ", "threads"])
def test_any_other_context_name_is_refused_too(tmp_path, bad: str) -> None:
    """The whitelist's whole point: it is not a list of the bad names, it is the ONE good one.
    Empty string, wrong case, stray whitespace and an invented name are all refused by the same
    equality, with no per-value arm to forget."""
    with pytest.raises(ValueError):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx=bad), leaf_batch_size=1)


def test_the_default_context_constructs(tmp_path) -> None:
    """THE POSITIVE CONTROL, with an assertion past "no exception raised": the pipeline must
    actually be built and carry `'spawn'` forward to the spawn site, and its work dir must
    exist — a refusal that swallowed the legal value, or a constructor that returned early,
    would satisfy a bare does-not-raise check while breaking every round."""
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    try:
        assert pipeline._mp_ctx_name == "spawn"
        assert pipeline._work_dir.is_dir()
        assert pipeline._poller.is_alive(), "the constructed pipeline never started its poller"
    finally:
        pipeline.stop()


def test_the_refusal_names_the_only_supported_value(tmp_path) -> None:
    """R73/R71 — the message carries the reason, because a refusal a caller cannot act on just
    moves the puzzle. It must say what IS supported and why the arming makes the alternatives
    wrong, not merely that the value was rejected."""
    with pytest.raises(ValueError) as caught:
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx="fork"), leaf_batch_size=1)
    message = str(caught.value)
    assert "'spawn'" in message
    assert "PR_SET_PDEATHSIG" in message
    assert "CUDA" in message
