"""Schema contract tests (run-config-schema v1): unknown/missing keys hard-fail,
representation is the closed set {grid, graph}, and O16 schema round-trip + every-config-
validates + no-code-side-defaults across the full model tree."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.loader import discover_configs, load_config
from mantis.config.schema import (
    SCHEMA_VERSION,
    DrainCapsConfig,
    EvalConfig,
    IdentityConfig,
    MonitorSchemaConfig,
    RunConfig,
    SelfplayConfig,
    TrainConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# WP11-A schema extension: eval.gate/eval.ladder are now required fields (design §c.1).
# This mirrors tests/eval/test_ladder_config_schema.py's fixture verbatim (kept in one
# place there; duplicated here only because this file predates the extension and must
# still construct a schema-complete payload for its own, unrelated assertions).
_LADDER_RUNGS = [
    {"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
]


def _valid_eval_block() -> dict:
    return {
        "random_model_sims": 96, "sealbot_model_sims": 128, "kraken_model_sims": 128,
        "strix_model_sims": 128, "random_floor_games": 0, "worker_device": "cuda",
        "round_timeout_sec": 3600.0, "worker_kill_grace_sec": 10.0,
        "gate": {
            "stride": 1, "screen_games": 80, "confirm_games": 128, "promotion_winrate": 0.55,
            "screen_confirm_lo": 0.44, "deploy_sims": 150, "opening_book": "book_v1_s20260625_p4",
            "bootstrap_resamples": 1000, "min_distinct_per_pair": 10, "seed_base": 20260625,
        },
        "ladder": {
            "rungs": [dict(r) for r in _LADDER_RUNGS], "round_games": 64,
            "min_games_per_active_rung": 4, "graduation_wr_lower_ci": 0.75,
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
_MINTED_TRAIN: dict = load_config(REPO_ROOT / "configs" / "dev_example.yaml").train.model_dump()


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
    }


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": _valid_eval_block(),
        "train": _valid_train_block(),
        "selfplay": _valid_selfplay_block(),
        "inference": _valid_inference_block(),
        "monitor": _valid_monitor_block(),
    }


def test_example_config_validates():
    cfg = load_config(REPO_ROOT / "configs" / "dev_example.yaml")
    assert cfg.run_id == "dev_example"
    assert cfg.identity.representation == "graph"


def test_top_level_unknown_key_rejected():
    payload = _valid_payload()
    payload["bogus_knob"] = 1
    with pytest.raises(ValidationError, match="bogus_knob"):
        RunConfig.model_validate(payload)


def test_nested_unknown_key_rejected():
    payload = _valid_payload()
    payload["identity"]["bogus_nested"] = "x"
    with pytest.raises(ValidationError, match="bogus_nested"):
        RunConfig.model_validate(payload)


def test_missing_top_level_key_rejected():
    payload = _valid_payload()
    del payload["seed"]
    with pytest.raises(ValidationError, match="seed"):
        RunConfig.model_validate(payload)


def test_missing_identity_key_rejected():
    payload = _valid_payload()
    del payload["identity"]["representation"]
    with pytest.raises(ValidationError, match="representation"):
        RunConfig.model_validate(payload)


def test_missing_eval_key_rejected():
    payload = _valid_payload()
    del payload["eval"]["sealbot_model_sims"]
    with pytest.raises(ValidationError, match="sealbot_model_sims"):
        RunConfig.model_validate(payload)


def test_missing_selfplay_key_rejected():
    payload = _valid_payload()
    del payload["selfplay"]["n_workers"]
    with pytest.raises(ValidationError, match="n_workers"):
        RunConfig.model_validate(payload)


def test_wrong_schema_version_rejected():
    payload = _valid_payload()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValidationError, match="schema_version"):
        RunConfig.model_validate(payload)


def test_representation_closed_set_rejects_dense():
    # dense->grid correction (judgment #4): "dense" is now OUTSIDE the closed set.
    payload = _valid_payload()
    payload["identity"]["representation"] = "dense"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_representation_grid_now_accepted():
    # "grid" is in the closed set — accepted for a GRID encoding (v6w25).
    payload = _valid_payload()
    payload["identity"] = {"encoding": "v6w25", "representation": "grid"}
    cfg = RunConfig.model_validate(payload)
    assert cfg.identity.representation == "grid"


# ── F1 — representation↔encoding consistency is a RUNTIME guard (not test-only) ──
def test_f1_graph_encoding_declared_grid_rejected_at_validate():
    # gnn_axis_v1 is a GRAPH encoding; declaring representation=grid must RAISE (LAW-06 pin guard).
    payload = _valid_payload()
    payload["identity"] = {"encoding": "gnn_axis_v1", "representation": "grid"}
    with pytest.raises(ValidationError, match="disagrees with the registry"):
        RunConfig.model_validate(payload)


def test_f1_grid_encoding_declared_graph_rejected_at_validate():
    payload = _valid_payload()
    payload["identity"] = {"encoding": "v6w25", "representation": "graph"}
    with pytest.raises(ValidationError, match="disagrees with the registry"):
        RunConfig.model_validate(payload)


def test_f1_unknown_encoding_rejected_at_validate():
    payload = _valid_payload()
    payload["identity"] = {"encoding": "no_such_encoding", "representation": "graph"}
    with pytest.raises(ValidationError, match="no_such_encoding"):
        RunConfig.model_validate(payload)


# ── O16 — schema round-trip + every-config-validates + no code-side defaults ──
def test_o16_every_committed_config_validates():
    # ADJ-13 F-1 corrective pass (recheck R-5): the ONE discovery authority, not a
    # sixth flat glob. A flat `*.yaml` census is blind to `configs/prod/run6.yaml`,
    # which gate 7 and gate 12 both now make legal.
    configs = discover_configs(REPO_ROOT / "configs")
    assert configs, "no committed configs found (gate 7 must never be vacuous)"
    for cfg_path in configs:
        load_config(cfg_path)  # raises on any failure


def test_o16_schema_round_trip():
    cfg = load_config(REPO_ROOT / "configs" / "run5.yaml")
    again = RunConfig.model_validate(cfg.model_dump())
    assert again == cfg


def test_o16_all_fields_required_no_code_side_defaults():
    for model in (RunConfig, IdentityConfig, EvalConfig, SelfplayConfig, TrainConfig,
                  MonitorSchemaConfig, DrainCapsConfig):
        for name, field in model.model_fields.items():
            assert field.is_required(), f"{model.__name__}.{name} has a code-side default"
