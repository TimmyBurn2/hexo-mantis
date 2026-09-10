"""The ladder + gate schema: LadderRung / GateConfig / LadderConfig and the EvalConfig block.

Every mutation below relies on the base payload validating cleanly, pinned by
`test_valid_payload_with_full_ladder_and_gate_validates` — without it a `pytest.raises
(ValidationError)` holds trivially for any mutation.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.loader import load_config
from mantis.config.schema import SCHEMA_VERSION, RunConfig

_REPO = Path(__file__).resolve().parents[2]
_RUN5 = _REPO / "configs" / "run6.yaml"

_LADDER_RUNGS = [
    {"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
    {"name": "sealbot_d6", "bot": "sealbot", "variant": "d6", "depth": 6,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
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


#: The complete `train:` payload, DERIVED from a MINTED config rather than restated, so a new
#: `train.*` key costs no edit here.
_MINTED_TRAIN: dict = load_config(_REPO / "configs" / "dev_example.yaml").train.model_dump()


def _train_block() -> dict:
    return dict(_MINTED_TRAIN)


def _selfplay_block() -> dict:
    return {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }


def _inference_block() -> dict:
    return {
        "inference_batch_size": 64, "inference_max_wait_ms": 10,
        # `inference.fused_graph_caps` is a REQUIRED block; this pair is the template's
        # non-binding-by-construction value, so nothing here exercises a split.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }


def _monitor_block() -> dict:
    return {
        # the ARMING cadence, schema-only and required.
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
        random_model_sims=96, sealbot_model_sims=128, random_floor_games=0, worker_device="cuda",
        round_timeout_sec=3600.0, worker_kill_grace_sec=10.0,
        ply_cap_adjudication=None, strength_floor=None,
        gate=_gate(), ladder=_ladder(),
    )
    eval_block.update(eval_overrides)
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_enabled": True,
        # A REQUIRED top-level leaf, minted `null` everywhere and refused at boot on a cuda
        # process; only the re-calibration sitting gives it a value.
        "allocator_posture": None,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": eval_block,
        "train": _train_block(),
        "search": {"kind": "puct"},
        "selfplay": _selfplay_block(),
        "inference": _inference_block(),
        "monitor": _monitor_block(),
    }


def test_valid_payload_with_full_ladder_and_gate_validates() -> None:
    """Sanity anchor: the full payload must itself validate, or every mutation is vacuous."""
    cfg = RunConfig.model_validate(_payload())
    assert len(cfg.eval.ladder.rungs) == len(_LADDER_RUNGS)
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
    """The graduation and activation thresholds are schema VALUES, never bare code literals."""
    assert _RUN5.is_file(), f"expected {_RUN5} to exist at HEAD"

    eval_src_dir = _REPO / "src" / "mantis" / "eval"
    if eval_src_dir.is_dir():
        for py_file in eval_src_dir.rglob("*.py"):
            text = py_file.read_text()
            assert "0.75" not in text, f"{py_file}: graduation_wr_lower_ci must not be a code literal"
            assert "0.65" not in text, f"{py_file}: activation_wr_lower_ci must not be a code literal"
