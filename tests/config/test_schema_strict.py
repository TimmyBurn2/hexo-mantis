"""O14 — strict-coercion rejection (StrictModel with strict=True).

strict=True rejects silent scalar coercions (str->int, float->int, bool->int). yaml-native
scalar types satisfy strict, so every committed + reminted config parses clean.
"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.loader import load_config
from mantis.config.schema import SCHEMA_VERSION, RunConfig


# WP11-A schema extension: eval.gate/eval.ladder are now required fields (design §c.1).
def _valid_eval_block() -> dict:
    return {
        "random_model_sims": 96, "sealbot_model_sims": 128, "kraken_model_sims": 128,
        "strix_model_sims": 128, "random_floor_games": 0, "worker_device": "cuda",
        "round_timeout_sec": 3600.0, "worker_kill_grace_sec": 10.0,
        "ply_cap_adjudication": None, "strength_floor": None,
        "gate": {
            "stride": 1, "screen_games": 80, "confirm_games": 128, "promotion_winrate": 0.55,
            "screen_confirm_lo": 0.44, "deploy_sims": 150, "opening_book": "book_v1_s20260625_p4",
            "bootstrap_resamples": 1000, "min_distinct_per_pair": 10, "seed_base": 20260625,
        },
        "ladder": {
            "rungs": [{"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
                      "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
                      "deploy_matched": True, "games_max": 32}],
            "round_games": 64, "min_games_per_active_rung": 4, "graduation_wr_lower_ci": 0.75,
            "graduation_consec_rounds": 3, "activation_wr_lower_ci": 0.65,
            "calibration_every_k_rounds": 4, "calibration_games": 8,
            "bootstrap_resamples": 1000, "bootstrap_ci_level": 0.95,
            "bt_prior_games": 1.0, "bootstrap_seed": 1234,
        },
    }


#: WPMINT Phase K-A stage 0: the complete `train:` payload, DERIVED from a MINTED config
#: rather than restated — eleven files carried a hand-written copy, so a new `train.*` key
#: cost eleven edits. `dev_example.yaml`'s resolved block was measured byte-identical to the
#: census it replaces, so the swap is zero-behavior-change.
_MINTED_TRAIN: dict = load_config(
    Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train.model_dump()


def _valid_train_block() -> dict:
    return dict(_MINTED_TRAIN)


def _valid_selfplay_block() -> dict:
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


def _valid_inference_block() -> dict:
    return {
        "inference_batch_size": 64, "inference_max_wait_ms": 10, "trace_inference": True,
        "compile_inference": False, "compile_inference_mode": "default",
        "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
    }


def _valid_monitor_block() -> dict:
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


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_enabled": True,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": _valid_eval_block(),
        "train": _valid_train_block(),
        "selfplay": _valid_selfplay_block(),
        "inference": _valid_inference_block(),
        "monitor": _valid_monitor_block(),
    }


def test_str_to_int_rejected():
    payload = _valid_payload()
    payload["seed"] = "20260716"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_float_to_int_rejected():
    payload = _valid_payload()
    payload["eval"]["random_model_sims"] = 96.5
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_bool_to_int_rejected():
    payload = _valid_payload()
    payload["eval"]["random_model_sims"] = True
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_selfplay_nested_float_to_int_rejected():
    # O14 strict-coercion pin ported off the retired RadiusStage float-step case: a nested
    # sub-model field (`selfplay.mcts.n_simulations`) still rejects a silent float->int
    # coercion (WPSC Phase 2 SC-A2: `selfplay.legal_move_radius_schedule`/`RadiusStage` are
    # gone from the schema, DESIGN_P2.md §5).
    payload = _valid_payload()
    payload["selfplay"]["mcts"]["n_simulations"] = 50.0
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_correctly_typed_config_validates_clean():
    cfg = RunConfig.model_validate(_valid_payload())
    assert cfg.seed == 1
    assert cfg.eval.random_model_sims == 96
