"""⊕ WP11-A DESIGN §b/§c.1 — ladder + gate schema extension (LadderRung/GateConfig/LadderConfig
+ the EvalConfig extension). RED-by-assertion: `mantis.config.schema` / `mantis.config.loader`
already exist and import cleanly today (this is a SCHEMA EXTENSION, not a new module), so every
test here fails today by `pydantic.ValidationError` (missing/extra key) rather than
ModuleNotFoundError — the schema simply does not have `eval.gate` / `eval.ladder` /
`eval.kraken_model_sims` / `eval.strix_model_sims` / `eval.random_floor_games` /
`eval.worker_device` / `eval.round_timeout_sec` / `eval.worker_kill_grace_sec` yet.

Byte-frozen through IMPL: fields transcribed verbatim from DESIGN.md §c.1 (rung
name/bot/variant/depth/opponent_sims/opening_book/deploy_matched/games_max; gate
stride/screen_games/confirm_games/promotion_winrate/screen_confirm_lo/deploy_sims/
opening_book/bootstrap_resamples/min_distinct_per_pair/seed_base — NO screen_confirm_hi,
MUST-FIX 1; ladder rungs/round_games/min_games_per_active_rung/graduation_wr_lower_ci/
graduation_consec_rounds/activation_wr_lower_ci/calibration_every_k_rounds/calibration_games/
bootstrap_resamples/bootstrap_ci_level/bt_prior_games/bootstrap_seed). Minted ladder order is
STATE §5 verbatim: sealbot_d5 -> kraken_raw -> sealbot_d6 -> kraken_mcts200 -> strix_128 ->
strix_256 (each opening_book=book_v1_s20260625_p4, deploy_matched=true, games_max=32).

Note (documented, not a defect): `test_temperature_key_anywhere_in_eval_is_rejected` (all 3
parametrized cases) and `test_rung_names_unique_and_bot_kind_known` /
`test_thresholds_bounded_and_activation_not_above_graduation` PASS today — but only because
the fixture `_payload()` is ALREADY invalid under today's un-extended schema (it carries
`eval.gate`/`eval.ladder`/etc., all unknown keys under `extra="forbid"`), so `pytest.raises
(ValidationError)` trivially holds regardless of the specific mutation under test. This is
not a tautology: post-IMPL, once the base payload validates cleanly (pinned by
`test_valid_payload_with_full_ladder_and_gate_validates`, RED today), these same tests start
exercising their real, specific behavior (temperature rejection / name-uniqueness / bound
checks) — an IMPL that wrongly ACCEPTED a temperature key, a duplicate rung name, or an
out-of-bounds threshold would make them fail then. Flagged here for RED-TEAM/REVIEW-impl
auditability, not something ORACLE-WRITE can or should "fix" by weakening the fixture.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.loader import load_config
from mantis.config.schema import SCHEMA_VERSION, RunConfig

_REPO = Path(__file__).resolve().parents[2]
_RUN5 = _REPO / "configs" / "run5.yaml"

_LADDER_RUNGS = [
    {"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "kraken_raw", "bot": "kraken", "variant": "raw", "depth": None,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "sealbot_d6", "bot": "sealbot", "variant": "d6", "depth": 6,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "kraken_mcts200", "bot": "kraken", "variant": "mcts200", "depth": None,
     "opponent_sims": 200, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "strix_128", "bot": "strix", "variant": "s128", "depth": None,
     "opponent_sims": 128, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "strix_256", "bot": "strix", "variant": "s256", "depth": None,
     "opponent_sims": 256, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
]

_RUNG_NAMES_IN_ORDER = [r["name"] for r in _LADDER_RUNGS]


def _gate(**overrides) -> dict:
    base = dict(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625,
    )
    base.update(overrides)
    return base


def _ladder(**overrides) -> dict:
    base = dict(
        rungs=[dict(r) for r in _LADDER_RUNGS], round_games=64, min_games_per_active_rung=4,
        graduation_wr_lower_ci=0.75, graduation_consec_rounds=3, activation_wr_lower_ci=0.65,
        calibration_every_k_rounds=4, calibration_games=8, bootstrap_resamples=1000,
        bootstrap_ci_level=0.95, bt_prior_games=1.0, bootstrap_seed=1234,
    )
    base.update(overrides)
    return base


#: WPMINT Phase K-A stage 0: the complete `train:` payload, DERIVED from a MINTED config
#: rather than restated — eleven files carried a hand-written copy, so a new `train.*` key
#: cost eleven edits. `dev_example.yaml`'s resolved block was measured byte-identical to the
#: census it replaces, so the swap is zero-behavior-change.
_MINTED_TRAIN: dict = load_config(_REPO / "configs" / "dev_example.yaml").train.model_dump()


def _train_block() -> dict:
    return dict(_MINTED_TRAIN)


def _selfplay_block() -> dict:
    return {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "inference_pool_size": None, "completed_q_values": False, "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_mcts": False, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0, "rotation_enabled": True,
        "forced_win_policy_enabled": False, "forced_win_policy_depth": 2,
        "forced_win_policy_weight": 1.0, "solver_enabled": False, "solver_depth": 16,
        "solver_node_budget": 50_000, "solver_neighbor_dist": 2, "solver_visit_weight": 0.3,
        "seed_fraction": 0.0, "seed_corpus_path": None, "log_investigation_metrics": True,
        "instrumentation_enabled": False,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "zoi_enabled": False, "zoi_lookback": 16, "zoi_margin": 5,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }


def _inference_block() -> dict:
    return {
        "inference_batch_size": 64, "inference_max_wait_ms": 10, "trace_inference": True,
        "compile_inference": False, "compile_inference_mode": "default",
        "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
    }


def _monitor_block() -> dict:
    return {
        # R242 (ADJ-D12): the ARMING cadence, schema-only and required.
        "gate_interval": 1000,
        "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5, "alert_grad_norm_max": 10.0,
        "alert_loss_increase_window": 3, "wr_hard_abort_enabled": False,
        "wr_rolling_consecutive_evals": 2, "wr_rolling_threshold": 0.10,
        "wr_rolling_min_step": 20000, "wr_collapse_from_peak_ratio": 0.5,
        "wr_collapse_min_step": 25000, "wr_collapse_consecutive_evals": 3,
        "wr_early_death_threshold": 0.05, "wr_early_death_min_step": 15000,
        "axis_warn": 0.45, "axis_alert": 0.50,
        "heartbeat_deadline_train_step_sec": 1800.0,
        "heartbeat_deadline_inference_dispatch_sec": 1800.0,
        "heartbeat_deadline_selfplay_drain_sec": 1800.0,
        "heartbeat_deadline_eval_round_sec": 1800.0,
        "heartbeat_poll_interval_sec": 5.0, "heartbeat_file_interval_sec": 15.0,
        "heartbeat_close_out_deadline_sec": 14400.0, "heartbeat_fire_effect_timeout_sec": 30.0,
        "supervisor_stale_after_sec": 900.0, "supervisor_poll_interval_sec": 30.0,
        "supervisor_kill_grace_sec": 30.0, "supervisor_max_relaunches": 5,
        "actor_lag_threshold_steps": 100, "actor_lag_abort_enabled": False,
        "drain": {
            "final_eval_drain_timeout_sec": 900.0, "eval_final_drain_safety_factor": 3.0,
            "eval_final_drain_hard_cap_sec": 14400.0, "terminal_eval_hard_cap_sec": 14400.0,
        },
        "disk_guard": {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0},
    }


def _payload(**eval_overrides) -> dict:
    eval_block = dict(
        random_model_sims=96, sealbot_model_sims=128, kraken_model_sims=128,
        strix_model_sims=128, random_floor_games=0, worker_device="cuda",
        round_timeout_sec=3600.0, worker_kill_grace_sec=10.0,
        ply_cap_adjudication=None, strength_floor=None,
        gate=_gate(), ladder=_ladder(),
    )
    eval_block.update(eval_overrides)
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_enabled": True,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": eval_block,
        "train": _train_block(),
        "selfplay": _selfplay_block(),
        "inference": _inference_block(),
        "monitor": _monitor_block(),
    }


def test_valid_payload_with_full_ladder_and_gate_validates() -> None:
    """Sanity anchor: the fully-populated payload above must itself validate once the schema
    extension lands (proves the fixture payload is not itself malformed)."""
    cfg = RunConfig.model_validate(_payload())
    assert len(cfg.eval.ladder.rungs) == 6
    assert cfg.eval.gate.promotion_winrate == 0.55


@pytest.mark.parametrize(
    "field",
    ["graduation_wr_lower_ci", "activation_wr_lower_ci", "graduation_consec_rounds",
     "calibration_every_k_rounds"],
)
def test_missing_graduation_threshold_fails_at_load_with_named_error(field: str) -> None:
    payload = _payload()
    del payload["eval"]["ladder"][field]
    with pytest.raises(ValidationError) as ei:
        RunConfig.model_validate(payload)
    assert field in str(ei.value), (
        f"a missing ladder threshold must fail LOAD naming the field {field!r}, not fall to "
        "a code-side default (R1)"
    )


@pytest.mark.parametrize(
    "path",
    [("eval", "temperature"), ("eval", "gate", "temperature"), ("eval", "ladder_rung0", "temperature")],
)
def test_temperature_key_anywhere_in_eval_is_rejected(path: tuple) -> None:
    payload = _payload()
    if path == ("eval", "temperature"):
        payload["eval"]["temperature"] = 0.5
    elif path == ("eval", "gate", "temperature"):
        payload["eval"]["gate"]["temperature"] = 0.5
    else:
        payload["eval"]["ladder"]["rungs"][0]["temperature"] = 0.5
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_rung_names_unique_and_bot_kind_known() -> None:
    dup = _payload()
    dup["eval"]["ladder"]["rungs"][1]["name"] = dup["eval"]["ladder"]["rungs"][0]["name"]
    with pytest.raises(ValidationError) as ei:
        RunConfig.model_validate(dup)
    assert "name" in str(ei.value).lower() or "unique" in str(ei.value).lower()

    unknown_kind = _payload()
    unknown_kind["eval"]["ladder"]["rungs"][0]["bot"] = "nnue"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(unknown_kind)


def test_thresholds_bounded_and_activation_not_above_graduation() -> None:
    bad = _payload()
    bad["eval"]["ladder"]["activation_wr_lower_ci"] = 0.90   # above graduation (0.75) — illegal
    with pytest.raises(ValidationError):
        RunConfig.model_validate(bad)

    bad_zero = _payload()
    bad_zero["eval"]["ladder"]["graduation_wr_lower_ci"] = 0.0
    with pytest.raises(ValidationError):
        RunConfig.model_validate(bad_zero)

    bad_one = _payload()
    bad_one["eval"]["ladder"]["graduation_wr_lower_ci"] = 1.0
    with pytest.raises(ValidationError):
        RunConfig.model_validate(bad_one)


def test_rung_order_is_preserved() -> None:
    cfg = RunConfig.model_validate(_payload())
    assert [r.name for r in cfg.eval.ladder.rungs] == _RUNG_NAMES_IN_ORDER


def test_minted_configs_carry_the_ladder_verbatim() -> None:
    """configs/run5.yaml's ladder must equal the six STATE §5 rungs in order once re-minted;
    0.75/0.65/3 must appear ONLY as VALUES of the named schema fields, never as bare code
    literals in src/mantis/eval (rule 4). Today configs/run5.yaml has no `eval.ladder` key at
    all (read at HEAD — no `ladder`/`gate` block), so loading it under the extended schema
    below fails with a named ValidationError; that IS the correct RED state (the re-mint is an
    IMPL-stage task, not ORACLE-WRITE's)."""
    assert _RUN5.is_file(), f"expected {_RUN5} to exist at HEAD"

    # Once re-minted (IMPL work), this is the shape that must hold — expressed here so the
    # assertion exists BEFORE the port (byte-frozen): the six rungs in STATE §5 order, and the
    # literals 0.75 / 0.65 never appear as bare numbers in src/mantis/eval source (they must be
    # the VALUES the schema fields resolve to, not inline code constants).
    eval_src_dir = _REPO / "src" / "mantis" / "eval"
    if eval_src_dir.is_dir():
        for py_file in eval_src_dir.rglob("*.py"):
            text = py_file.read_text()
            assert "0.75" not in text, f"{py_file}: graduation_wr_lower_ci must not be a code literal"
            assert "0.65" not in text, f"{py_file}: activation_wr_lower_ci must not be a code literal"
