"""⊕ WPUF Phase U ORACLE — O-U5: the three knobs and nothing else (DESIGN_U §5/§9).

RED-at-import until IMPL lands `mantis.config.resolve.actor_sync.resolve_actor_sync_cadence`
(K1's ONE read path) + the three schema fields.

R1/LAW-08: missing key = named error at load, never a fallback; `ge=1` on the cadence
means NO representable "off" value exists (R49 at the type level); the cross-field
validator (`RunConfig`-level, since it spans sections) rejects a threshold at or below
the cadence with a NAMED message. Payload builders mirror
tests/config/test_train_policy_value_target_consistency.py's full-RunConfig shape.

>300 justify (R8): the full RunConfig payload builder (every field explicit per R1) is the
price of testing a RunConfig-level validator; the assertions themselves are compact.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mantis.config.loader import load_config
from mantis.config.resolve.actor_sync import resolve_actor_sync_cadence  # RED-at-import anchor
from mantis.config.resolve import resolve_monitor_config
from mantis.config.schema import RunConfig, SCHEMA_VERSION, TrainConfig, MonitorSchemaConfig
from mantis.monitor.config import MonitorConfig

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = ("dev_example.yaml", "run5.yaml", "smoke_gnn.yaml",
            "smoke_radius_curriculum.yaml", "sustained_kcluster.yaml")

_NEW_KEYS = (
    ("train", "actor_sync_cadence_steps"),
    ("monitor", "actor_lag_threshold_steps"),
    ("monitor", "actor_lag_abort_enabled"),
)


def _eval_block() -> dict:
    return {
        "random_model_sims": 96, "sealbot_model_sims": 128, "kraken_model_sims": 128,
        "strix_model_sims": 128, "random_floor_games": 0, "worker_device": "cuda",
        "round_timeout_sec": 3600.0, "worker_kill_grace_sec": 10.0,
        "gate": {
            "stride": 1, "screen_games": 80, "confirm_games": 128,
            "promotion_winrate": 0.55, "screen_confirm_lo": 0.44, "deploy_sims": 150,
            "opening_book": "book_v1_s20260625_p4", "bootstrap_resamples": 1000,
            "min_distinct_per_pair": 10, "seed_base": 20260625,
        },
        "ladder": {
            "rungs": [{"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
                       "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
                       "deploy_matched": True, "games_max": 32}],
            "round_games": 64, "min_games_per_active_rung": 4,
            "graduation_wr_lower_ci": 0.75, "graduation_consec_rounds": 3,
            "activation_wr_lower_ci": 0.65, "calibration_every_k_rounds": 4,
            "calibration_games": 8, "bootstrap_resamples": 1000,
            "bootstrap_ci_level": 0.95, "bt_prior_games": 1.0, "bootstrap_seed": 1234,
        },
    }


#: WPMINT Phase K-A stage 0 — the complete `train:` payload, DERIVED from a MINTED config
#: instead of restated. Eleven test files carried a hand-written copy of this block, so every
#: new `train.*` key cost eleven edits and gave eleven chances to disagree with the schema;
#: derived, they cost none. `dev_example.yaml` is the base because its RESOLVED train block
#: was measured BYTE-IDENTICAL to the census this replaces, which is what makes the swap
#: zero-behavior-change rather than a re-baselining.
_MINTED_TRAIN: dict = load_config(_REPO / "configs" / "dev_example.yaml").train.model_dump()


def _train_block(**over: object) -> dict:
    return dict(_MINTED_TRAIN, **over)


def _selfplay_block() -> dict:
    return {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "inference_pool_size": None, "completed_q_values": False,
        "c_visit": 50.0, "c_scale": 1.0, "gumbel_mcts": False, "gumbel_m": 16,
        "gumbel_explore_moves": 10, "results_queue_cap": 10_000,
        "random_opening_plies": 0, "rotation_enabled": True,
        "forced_win_policy_enabled": False, "forced_win_policy_depth": 2,
        "forced_win_policy_weight": 1.0, "solver_enabled": False, "solver_depth": 16,
        "solver_node_budget": 50_000, "solver_neighbor_dist": 2,
        "solver_visit_weight": 0.3, "seed_fraction": 0.0, "seed_corpus_path": None,
        "log_investigation_metrics": True, "instrumentation_enabled": False,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
                 "dirichlet_enabled": True},
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


def _monitor_block(**over: object) -> dict:
    base = {
        # R242 (ADJ-D12): the ARMING cadence, schema-only and required.
        "gate_interval": 1000,
        "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5,
        "alert_grad_norm_max": 10.0, "alert_loss_increase_window": 3,
        "wr_hard_abort_enabled": False, "wr_rolling_consecutive_evals": 2,
        "wr_rolling_threshold": 0.10, "wr_rolling_min_step": 20000,
        "wr_collapse_from_peak_ratio": 0.5, "wr_collapse_min_step": 25000,
        "wr_collapse_consecutive_evals": 3, "wr_early_death_threshold": 0.05,
        "wr_early_death_min_step": 15000, "axis_warn": 0.45, "axis_alert": 0.50,
        "heartbeat_deadline_train_step_sec": 1800.0,
        "heartbeat_deadline_inference_dispatch_sec": 1800.0,
        "heartbeat_deadline_selfplay_drain_sec": 1800.0,
        "heartbeat_deadline_eval_round_sec": 1800.0,
        "heartbeat_poll_interval_sec": 5.0, "heartbeat_file_interval_sec": 15.0,
        "heartbeat_close_out_deadline_sec": 14400.0,
        "heartbeat_fire_effect_timeout_sec": 30.0,
        "supervisor_stale_after_sec": 900.0, "supervisor_poll_interval_sec": 30.0,
        "supervisor_kill_grace_sec": 30.0, "supervisor_max_relaunches": 5,
        "drain": {"final_eval_drain_timeout_sec": 900.0,
                  "eval_final_drain_safety_factor": 3.0,
                  "eval_final_drain_hard_cap_sec": 14400.0,
                  "terminal_eval_hard_cap_sec": 14400.0},
        "disk_guard": {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0},
        "actor_lag_threshold_steps": 100,   # K2 — minted inert value (DESIGN §5)
        "actor_lag_abort_enabled": False,   # K3 — the config arms it (run5, not this WP)
    }
    base.update(over)
    return base


def _payload(*, train_over: dict | None = None, monitor_over: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "run_id": "unit_test", "seed": 1,
        "eval_enabled": True,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": _eval_block(), "train": _train_block(**(train_over or {})),
        "selfplay": _selfplay_block(), "inference": _inference_block(),
        "monitor": _monitor_block(**(monitor_over or {})),
    }


# ── construction + named absence errors ───────────────────────────────────────────────
def test_valid_payload_with_the_three_knobs_constructs_clean() -> None:
    cfg = RunConfig.model_validate(_payload())
    assert cfg.train.actor_sync_cadence_steps == 1
    assert cfg.monitor.actor_lag_threshold_steps == 100
    assert cfg.monitor.actor_lag_abort_enabled is False


@pytest.mark.parametrize(("section", "key"), _NEW_KEYS)
def test_missing_knob_is_a_named_error_at_load(section: str, key: str) -> None:
    """R1: no code-side default — a missing key is a ValidationError NAMING the key."""
    payload = _payload()
    del payload[section][key]
    with pytest.raises(ValidationError, match=key):
        RunConfig.model_validate(payload)


def test_schema_fields_are_required_with_no_pydantic_level_default() -> None:
    assert TrainConfig.model_fields["actor_sync_cadence_steps"].is_required()
    assert MonitorSchemaConfig.model_fields["actor_lag_threshold_steps"].is_required()
    assert MonitorSchemaConfig.model_fields["actor_lag_abort_enabled"].is_required()


# ── bounds: no representable "off" (R49) ──────────────────────────────────────────────
@pytest.mark.parametrize("bad_cadence", [0, -1])
def test_cadence_has_no_representable_off_value(bad_cadence: int) -> None:
    """`ge=1`: the schema CANNOT express "don't sync" — R49 enforced at the type level."""
    with pytest.raises(ValidationError, match="actor_sync_cadence_steps"):
        RunConfig.model_validate(
            _payload(train_over={"actor_sync_cadence_steps": bad_cadence}))


@pytest.mark.parametrize("bad_threshold", [0, -1])
def test_lag_threshold_rejects_nonpositive(bad_threshold: int) -> None:
    """`ge=1`; disablement is the arming flag's job — one authority, no zero-sentinel."""
    with pytest.raises(ValidationError, match="actor_lag_threshold_steps"):
        RunConfig.model_validate(
            _payload(monitor_over={"actor_lag_threshold_steps": bad_threshold}))


# ── the cross-field validator (RunConfig-level; DESIGN §5's named message) ────────────
@pytest.mark.parametrize("threshold", [8, 4])
def test_threshold_at_or_below_cadence_rejected_with_named_message(threshold: int) -> None:
    with pytest.raises(ValidationError,
                       match="must exceed train.actor_sync_cadence_steps"):
        RunConfig.model_validate(_payload(
            train_over={"actor_sync_cadence_steps": 8},
            monitor_over={"actor_lag_threshold_steps": threshold}))


def test_threshold_just_above_cadence_accepted() -> None:
    cfg = RunConfig.model_validate(_payload(
        train_over={"actor_sync_cadence_steps": 8},
        monitor_over={"actor_lag_threshold_steps": 9}))
    assert cfg.monitor.actor_lag_threshold_steps == 9


# ── resolvers: the ONE read path per knob ─────────────────────────────────────────────
def test_resolver_returns_the_configured_cadence() -> None:
    cfg = RunConfig.model_validate(
        _payload(train_over={"actor_sync_cadence_steps": 7}))
    assert resolve_actor_sync_cadence(cfg.train) == 7


def test_resolve_monitor_config_copies_the_lag_fields() -> None:
    section = MonitorSchemaConfig.model_validate(
        _monitor_block(actor_lag_threshold_steps=77, actor_lag_abort_enabled=True))
    resolved = resolve_monitor_config(section)
    assert resolved.actor_lag_threshold_steps == 77
    assert resolved.actor_lag_abort_enabled is True


def test_runtime_monitor_config_carries_the_smoke_posture() -> None:
    """The established monitor pattern: schema REQUIRED, runtime dataclass carries the
    smoke value — threshold 100 (inert at cadence 1), abort False (config arms it)."""
    runtime = MonitorConfig()
    assert runtime.actor_lag_threshold_steps == 100
    assert runtime.actor_lag_abort_enabled is False


# ── the minted configs carry all three keys (a hand-revert fails LOCALLY) ─────────────
@pytest.mark.parametrize("name", _CONFIGS)
def test_minted_config_carries_all_three_keys(name: str) -> None:
    data = yaml.safe_load((_REPO / "configs" / name).read_text(encoding="utf-8"))
    for section, key in _NEW_KEYS:
        assert key in data.get(section, {}), (
            f"configs/{name}: missing {section}.{key} — configs are minted complete (R1); "
            "a hand-reverted file must fail here, not only in CI gate 7"
        )
