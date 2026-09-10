# >300 justify (R8): a per-FIELD bounds census over the `eval` section's always-present leaves,
# each contributing one out-of-domain rejection and one in-domain boundary acceptance, so the
# length tracks the field count rather than any logic. NOT covered: the two `Block | None`
# postures, whose bounds need an ARMED fixture (tests/config/test_eval_posture_schema.py).
"""Numeric-bounds validation on the eval/gate/ladder schema fields.

Pre-fix, `random_model_sims=-5`, a `bootstrap_ci_level` outside `(0,1)` and
`gate.promotion_winrate=2.0` all loaded SILENTLY — crashing `np.quantile` inside a worker,
inverting a CI, or disabling promotion forever. Every case is parametrized: one out-of-domain
value raises a `ValidationError` naming the field, one in-domain boundary value loads clean.

The two timeout fields were floor-only bounds that admitted a REAL `.inf` YAML literal end to
end, which reproduced a silent poller death through `Process.join(float("inf"))`. They and
`ladder.bt_prior_games` now carry `allow_inf_nan=False`, and the timeouts a join depends on
carry a finite ceiling.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from mantis.config.loader import load_config
from mantis.config.schema import SCHEMA_VERSION, RunConfig, _EVAL_TIMEOUT_CEILING_SEC


def _gate(**overrides: Any) -> dict:
    base = dict(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625,
    )
    base.update(overrides)
    return base


def _rung(**overrides: Any) -> dict:
    base = dict(
        name="sealbot_d5", bot="sealbot", variant="d5", depth=5, opponent_sims=None,
        opening_book="book_v1_s20260625_p4", deploy_matched=True, games_max=32,
    )
    base.update(overrides)
    return base


def _ladder(**overrides: Any) -> dict:
    base = dict(
        rungs=[_rung()], round_games=64, min_games_per_active_rung=4,
        graduation_wr_lower_ci=0.75, graduation_consec_rounds=3, activation_wr_lower_ci=0.65,
        calibration_every_k_rounds=4, calibration_games=8, bootstrap_resamples=1000,
        bootstrap_ci_level=0.95, bt_prior_games=1.0, bootstrap_seed=1234,
    )
    base.update(overrides)
    return base


#: The complete `train:` payload, DERIVED from a MINTED config rather than restated — eleven
#: hand-written copies meant a new `train.*` key cost eleven edits.
_MINTED_TRAIN: dict = load_config(
    Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train.model_dump()


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
        # A REQUIRED block; this pair is the NON-BINDING-BY-CONSTRUCTION template value.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }


def _monitor_block() -> dict:
    return {
        # the ARMING cadence, schema-only and required
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


def _payload(**eval_overrides: Any) -> dict:
    eval_block = dict(
        random_model_sims=96, sealbot_model_sims=128, random_floor_games=4, worker_device="cuda",
        round_timeout_sec=3600.0, worker_kill_grace_sec=10.0, gate=_gate(), ladder=_ladder(),
        ply_cap_adjudication=None, strength_floor=None,
    )
    eval_block.update(eval_overrides)
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_enabled": True,
        # A REQUIRED top-level leaf; `null` is the placeholder, refused at boot on a cuda process.
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


def _set_path(payload: dict, path: "tuple[str, ...]", value: Any) -> dict:
    """Deep-set a value at `path`, which is relative to `eval`."""
    payload = copy.deepcopy(payload)
    node = payload["eval"]
    for key in path[:-1]:
        if key == "rungs0":
            node = node["ladder"]["rungs"][0]
        else:
            node = node[key]
    last = path[-1]
    node[last] = value
    return payload


def _validate(payload: dict) -> RunConfig:
    return RunConfig.model_validate(payload)


def test_random_model_sims_negative_is_rejected_not_silently_loaded() -> None:
    """`eval.random_model_sims = -5` previously loaded with zero error."""
    payload = _payload(random_model_sims=-5)
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert "random_model_sims" in str(ei.value)


def test_bootstrap_ci_level_out_of_unit_interval_is_rejected_not_silently_loaded() -> None:
    """`bootstrap_ci_level = 1.5` crashes `np.quantile` in a worker; `-0.1` silently inverts."""
    for bad in (1.5, -0.1, 0.0, 1.0):
        payload = _payload()
        payload["eval"]["ladder"]["bootstrap_ci_level"] = bad
        with pytest.raises(ValidationError) as ei:
            _validate(payload)
        assert "bootstrap_ci_level" in str(ei.value), f"bootstrap_ci_level={bad} must be named"


def test_promotion_winrate_above_one_is_rejected_not_silently_loaded() -> None:
    """`promotion_winrate = 2.0` loaded clean and disabled promotion forever."""
    payload = _payload()
    payload["eval"]["gate"]["promotion_winrate"] = 2.0
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert "promotion_winrate" in str(ei.value)


# Every bounded numeric field, out-of-domain + in-domain. Paths are relative to `eval`; "rungs0"
# addresses eval.ladder.rungs[0].
_OUT_OF_DOMAIN_CASES = [
    # EvalConfig
    (("random_model_sims",), 0, "eval.random_model_sims"),
    (("random_model_sims",), -5, "eval.random_model_sims"),
    (("sealbot_model_sims",), 0, "eval.sealbot_model_sims"),
    (("random_floor_games",), -1, "eval.random_floor_games"),
    (("round_timeout_sec",), 0.0, "eval.round_timeout_sec"),
    (("round_timeout_sec",), -1.0, "eval.round_timeout_sec"),
    (("worker_kill_grace_sec",), -1.0, "eval.worker_kill_grace_sec"),
    # Non-finite and above-ceiling on the two timeout fields, plus the third floor-only float.
    (("round_timeout_sec",), float("inf"), "eval.round_timeout_sec"),
    (("round_timeout_sec",), float("-inf"), "eval.round_timeout_sec"),
    (("round_timeout_sec",), float("nan"), "eval.round_timeout_sec"),
    (("round_timeout_sec",), _EVAL_TIMEOUT_CEILING_SEC + 1.0, "eval.round_timeout_sec"),
    (("worker_kill_grace_sec",), float("inf"), "eval.worker_kill_grace_sec"),
    (("worker_kill_grace_sec",), float("-inf"), "eval.worker_kill_grace_sec"),
    (("worker_kill_grace_sec",), float("nan"), "eval.worker_kill_grace_sec"),
    (("worker_kill_grace_sec",), _EVAL_TIMEOUT_CEILING_SEC + 1.0, "eval.worker_kill_grace_sec"),
    # GateConfig
    (("gate", "stride"), 0, "eval.gate.stride"),
    (("gate", "screen_games"), 0, "eval.gate.screen_games"),
    (("gate", "confirm_games"), 0, "eval.gate.confirm_games"),
    (("gate", "promotion_winrate"), 2.0, "eval.gate.promotion_winrate"),
    (("gate", "promotion_winrate"), -0.1, "eval.gate.promotion_winrate"),
    (("gate", "screen_confirm_lo"), 1.1, "eval.gate.screen_confirm_lo"),
    (("gate", "screen_confirm_lo"), -0.1, "eval.gate.screen_confirm_lo"),
    (("gate", "deploy_sims"), 0, "eval.gate.deploy_sims"),
    (("gate", "bootstrap_resamples"), 0, "eval.gate.bootstrap_resamples"),
    (("gate", "min_distinct_per_pair"), 0, "eval.gate.min_distinct_per_pair"),
    # LadderConfig
    (("ladder", "round_games"), 0, "eval.ladder.round_games"),
    (("ladder", "min_games_per_active_rung"), -1, "eval.ladder.min_games_per_active_rung"),
    (("ladder", "calibration_games"), 0, "eval.ladder.calibration_games"),
    (("ladder", "bootstrap_resamples"), 0, "eval.ladder.bootstrap_resamples"),
    (("ladder", "bootstrap_ci_level"), 1.5, "eval.ladder.bootstrap_ci_level"),
    (("ladder", "bootstrap_ci_level"), -0.1, "eval.ladder.bootstrap_ci_level"),
    (("ladder", "bt_prior_games"), -1.0, "eval.ladder.bt_prior_games"),
    (("ladder", "bt_prior_games"), float("inf"), "eval.ladder.bt_prior_games"),
    (("ladder", "bt_prior_games"), float("-inf"), "eval.ladder.bt_prior_games"),
    (("ladder", "bt_prior_games"), float("nan"), "eval.ladder.bt_prior_games"),
    # LadderRung (rungs[0])
    (("rungs0", "depth"), 0, "eval.ladder.rungs.0.depth"),
    (("rungs0", "games_max"), 0, "eval.ladder.rungs.0.games_max"),
]


@pytest.mark.parametrize("path,bad_value,field_hint", _OUT_OF_DOMAIN_CASES,
                        ids=[f"{'.'.join(p)}={v!r}" for p, v, _ in _OUT_OF_DOMAIN_CASES])
def test_out_of_domain_value_raises_named_validation_error(
    path: "tuple[str, ...]", bad_value: Any, field_hint: str,
) -> None:
    payload = _set_path(_payload(), path, bad_value)
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    field_name = path[-1]
    assert field_name in str(ei.value), (
        f"expected a named ValidationError mentioning {field_name!r} for {field_hint}="
        f"{bad_value!r}, got: {ei.value}"
    )


_IN_DOMAIN_BOUNDARY_CASES = [
    (("random_model_sims",), 1),
    (("sealbot_model_sims",), 1),
    (("random_floor_games",), 0),
    (("round_timeout_sec",), 0.001),
    (("worker_kill_grace_sec",), 0.0),
    # The exact ceiling must still load: the fix rejects only non-finite/above-ceiling values.
    (("round_timeout_sec",), _EVAL_TIMEOUT_CEILING_SEC),
    (("worker_kill_grace_sec",), _EVAL_TIMEOUT_CEILING_SEC),
    (("ladder", "bt_prior_games"), 1e18),
    (("gate", "stride"), 1),
    (("gate", "screen_games"), 1),
    (("gate", "confirm_games"), 1),
    (("gate", "promotion_winrate"), 0.0),
    (("gate", "promotion_winrate"), 1.0),
    (("gate", "screen_confirm_lo"), 0.0),
    (("gate", "screen_confirm_lo"), 1.0),
    (("gate", "deploy_sims"), 1),
    (("gate", "bootstrap_resamples"), 1),
    (("gate", "min_distinct_per_pair"), 1),
    (("ladder", "round_games"), 1),
    (("ladder", "min_games_per_active_rung"), 0),
    (("ladder", "calibration_games"), 1),
    (("ladder", "bootstrap_resamples"), 1),
    (("ladder", "bootstrap_ci_level"), 0.001),
    (("ladder", "bootstrap_ci_level"), 0.999),
    (("ladder", "bt_prior_games"), 0.0),
    (("rungs0", "depth"), 1),
    (("rungs0", "games_max"), 1),
]


@pytest.mark.parametrize("path,value", _IN_DOMAIN_BOUNDARY_CASES,
                        ids=[f"{'.'.join(p)}={v!r}" for p, v in _IN_DOMAIN_BOUNDARY_CASES])
def test_in_domain_boundary_value_loads_clean(path: "tuple[str, ...]", value: Any) -> None:
    payload = _set_path(_payload(), path, value)
    _validate(payload)  # must not raise


def test_valid_payload_still_loads_after_bounds_added() -> None:
    """Sanity anchor: the bounds must never reject a legitimate, already-shipped config shape."""
    cfg = RunConfig.model_validate(_payload())
    assert cfg.eval.gate.promotion_winrate == 0.55
    assert cfg.eval.ladder.bootstrap_ci_level == 0.95


    # The ORIGINAL repro shape: a genuine YAML document, not a hand-constructed Python float.
def _yaml_doc_with_eval_override(field: str, yaml_literal: str) -> dict:
    """A full payload whose `eval.<field>` is parsed from a REAL YAML literal, not `float(...)`."""
    doc_text = f"eval_override_value: {yaml_literal}\n"
    parsed_value = yaml.safe_load(doc_text)["eval_override_value"]
    payload = _payload()
    payload["eval"][field] = parsed_value
    return payload


@pytest.mark.parametrize(
    "field,yaml_literal",
    [
        ("worker_kill_grace_sec", ".inf"),
        ("worker_kill_grace_sec", "-.inf"),
        ("worker_kill_grace_sec", ".nan"),
        ("round_timeout_sec", ".inf"),
        ("round_timeout_sec", "-.inf"),
        ("round_timeout_sec", ".nan"),
    ],
    ids=["worker_kill_grace_sec=.inf", "worker_kill_grace_sec=-.inf", "worker_kill_grace_sec=.nan",
         "round_timeout_sec=.inf", "round_timeout_sec=-.inf", "round_timeout_sec=.nan"],
)
def test_original_f_rt2_1_repro_real_yaml_document_now_rejected(field: str, yaml_literal: str) -> None:
    """The exact reproduction: `worker_kill_grace_sec: .inf` as a genuine YAML document loaded
    SILENTLY and killed a poller through `Process.join(float('inf'))`."""
    payload = _yaml_doc_with_eval_override(field, yaml_literal)
    assert not math.isfinite(payload["eval"][field])  # confirm the injected value IS non-finite
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert field in str(ei.value), (
        f"expected a named ValidationError mentioning {field!r} for the real YAML literal "
        f"{yaml_literal!r}, got: {ei.value}"
    )


def test_original_f_rt2_1_repro_ceiling_boundary_still_loads() -> None:
    """The non-`.inf` half of the repro: a grace at the finite ceiling is legitimate."""
    payload = _payload()
    payload["eval"]["worker_kill_grace_sec"] = _EVAL_TIMEOUT_CEILING_SEC
    cfg = _validate(payload)
    assert cfg.eval.worker_kill_grace_sec == _EVAL_TIMEOUT_CEILING_SEC


def test_bt_prior_games_rejects_non_finite_via_real_yaml_document() -> None:
    """An `inf` `bt_prior_games` degrades every rating and `p_hat` in `fit_bt` to NaN."""
    payload = _yaml_doc_with_eval_override("bt_prior_games_probe", ".inf")
    bad_value = payload["eval"].pop("bt_prior_games_probe")
    payload["eval"]["ladder"]["bt_prior_games"] = bad_value
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert "bt_prior_games" in str(ei.value)


def test_minted_configs_still_load_after_f_rt2_1_bounds() -> None:
    """The fix must never require re-minting a shipped config."""
    payload = _payload()  # mirrors the minted-config values verbatim (see docstring)
    assert payload["eval"]["round_timeout_sec"] == 3600.0
    assert payload["eval"]["worker_kill_grace_sec"] == 10.0
    assert payload["eval"]["ladder"]["bt_prior_games"] == 1.0
    _validate(payload)  # must not raise
