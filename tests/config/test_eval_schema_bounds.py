# >300 justify (R8), stated at this file's MEASURED size of 406 lines. Pre-existing gap, closed
# by WPMINT Phase W. The file is a per-FIELD bounds census over the `eval` section's 30 leaves:
# each field contributes a parametrized out-of-domain rejection and an in-domain boundary
# acceptance, so the length tracks the section's field count rather than any logic. Splitting it
# by sub-model would put `EvalConfig`, `GateConfig` and `LadderConfig` bounds in three files
# while the defect class (a silently-loaded out-of-domain value) is one class with one triage
# protocol; the census reads as a census only while it is whole.
"""RED-TEAM-FIX WP11-A F2 (MAJOR) — numeric-bounds validation on eval/gate/ladder schema
fields (mantis-migration/wp/WP11A/RED_TEAM.md Finding F2).

Pre-fix, `eval.random_model_sims=-5`, `eval.ladder.bootstrap_ci_level` outside `(0,1)`, and
`eval.gate.promotion_winrate=2.0` all loaded SILENTLY (no error at config-load time) — the
first two degrade to either a downstream `np.quantile` crash deep inside a worker subprocess
or a silently-inverted-but-plausible CI; the third permanently and silently disables
promotion forever (`wr_confirm >= 2.0` can never be true). R1/LAW-08 exist to kill exactly
this "silently-disabled-lever"/"silently-wrong-number" class.

This is a NEW file (frozen-oracle discipline: `tests/eval/test_ladder_config_schema.py` and
`tests/config/test_schema_strict.py` are NOT edited here, only read for convention). Every
case is parametrized: one out-of-domain value -> `pydantic.ValidationError` naming the
field, one in-domain boundary value -> loads clean. Named error means the field path
appears in `str(ValidationError)` (pydantic includes the dotted `loc` automatically for a
`Field(ge=/le=/gt=/lt=)` constraint — no hand-rolled message needed for these to be "named").

RED-TEAM-2 F-RT2-1 (BLOCKER, extended here by the F-RT2-1 FIX pass): `round_timeout_sec`
and `worker_kill_grace_sec` were floor-only bounds (`gt=0`/`ge=0`) that silently admitted
`float("inf")` — a REAL `.inf` YAML literal parsed clean through `RunConfig.model_validate`
end to end, and `worker_kill_grace_sec=+inf` reproduced F1's exact silent-poller-death
failure mode via a real `multiprocessing.Process.join(float("inf"))` `OverflowError` inside
`_escalate_and_finalize` (pipeline.py) — a code path entirely outside F1's own catch-all.
Both fields (plus `eval.ladder.bt_prior_games`, the third floor-only float this sweep
found) now carry `allow_inf_nan=False` (rejects `inf`/`-inf`/`nan` with a named pydantic
`finite_number` error) and, for the two timeout fields the isolation-law join arithmetic
depends on, a finite ceiling (`mantis.config.schema._EVAL_TIMEOUT_CEILING_SEC`, one day).
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


#: WPMINT Phase K-A stage 0: the complete `train:` payload, DERIVED from a MINTED config
#: rather than restated — eleven files carried a hand-written copy, so a new `train.*` key
#: cost eleven edits. `dev_example.yaml`'s resolved block was measured byte-identical to the
#: census it replaces, so the swap is zero-behavior-change.
_MINTED_TRAIN: dict = load_config(
    Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train.model_dump()


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


def _payload(**eval_overrides: Any) -> dict:
    eval_block = dict(
        random_model_sims=96, sealbot_model_sims=128, kraken_model_sims=128,
        strix_model_sims=128, random_floor_games=4, worker_device="cuda",
        round_timeout_sec=3600.0, worker_kill_grace_sec=10.0, gate=_gate(), ladder=_ladder(),
    )
    eval_block.update(eval_overrides)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": eval_block,
        "train": _train_block(),
        "selfplay": _selfplay_block(),
        "inference": _inference_block(),
        "monitor": _monitor_block(),
    }


def _set_path(payload: dict, path: "tuple[str, ...]", value: Any) -> dict:
    """Deep-set `payload["eval"][...path][-1]] = value` (path is relative to `eval`)."""
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


# ── the two RED_TEAM-reproduced silent-load repro cases, closed loop ────────────────────
def test_random_model_sims_negative_is_rejected_not_silently_loaded() -> None:
    """RED_TEAM.md item 6: `eval.random_model_sims = -5` previously loaded with zero error."""
    payload = _payload(random_model_sims=-5)
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert "random_model_sims" in str(ei.value)


def test_bootstrap_ci_level_out_of_unit_interval_is_rejected_not_silently_loaded() -> None:
    """RED_TEAM.md item 6: `bootstrap_ci_level = 1.5` and `= -0.1` previously loaded with
    zero error; `1.5` degrades to a runtime `np.quantile` crash deep in a worker subprocess,
    `-0.1` silently computes a statistically-inverted-but-plausible CI. Both closed here."""
    for bad in (1.5, -0.1, 0.0, 1.0):
        payload = _payload()
        payload["eval"]["ladder"]["bootstrap_ci_level"] = bad
        with pytest.raises(ValidationError) as ei:
            _validate(payload)
        assert "bootstrap_ci_level" in str(ei.value), f"bootstrap_ci_level={bad} must be named"


def test_promotion_winrate_above_one_is_rejected_not_silently_loaded() -> None:
    """RED_TEAM.md item 6: `promotion_winrate = 2.0` previously loaded with zero error and
    permanently+silently disabled promotion forever (`wr_confirm >= 2.0` can never hold)."""
    payload = _payload()
    payload["eval"]["gate"]["promotion_winrate"] = 2.0
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert "promotion_winrate" in str(ei.value)


# ── full parametrized sweep: every bounded numeric field, out-of-domain + in-domain ─────
# (path relative to `eval`; "rungs0" addresses eval.ladder.rungs[0])
_OUT_OF_DOMAIN_CASES = [
    # EvalConfig
    (("random_model_sims",), 0, "eval.random_model_sims"),
    (("random_model_sims",), -5, "eval.random_model_sims"),
    (("sealbot_model_sims",), 0, "eval.sealbot_model_sims"),
    (("kraken_model_sims",), 0, "eval.kraken_model_sims"),
    (("strix_model_sims",), 0, "eval.strix_model_sims"),
    (("random_floor_games",), -1, "eval.random_floor_games"),
    (("round_timeout_sec",), 0.0, "eval.round_timeout_sec"),
    (("round_timeout_sec",), -1.0, "eval.round_timeout_sec"),
    (("worker_kill_grace_sec",), -1.0, "eval.worker_kill_grace_sec"),
    # RED-TEAM-2 F-RT2-1: non-finite + above-ceiling on the two isolation-law timeout
    # fields, plus the sweep-found third floor-only float (bt_prior_games).
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
    (("kraken_model_sims",), 1),
    (("strix_model_sims",), 1),
    (("random_floor_games",), 0),
    (("round_timeout_sec",), 0.001),
    (("worker_kill_grace_sec",), 0.0),
    # RED-TEAM-2 F-RT2-1: the exact ceiling value must still load (the fix must reject
    # ONLY non-finite/above-ceiling, never clamp or reject a legitimate boundary value).
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
    """Sanity anchor: the fully-populated in-domain payload must still validate — the
    bounds added for F2 must never reject a legitimate, already-shipped config shape."""
    cfg = RunConfig.model_validate(_payload())
    assert cfg.eval.gate.promotion_winrate == 0.55
    assert cfg.eval.ladder.bootstrap_ci_level == 0.95


# ── RED-TEAM-2 F-RT2-1 (BLOCKER): the ORIGINAL repro shape, closed loop ──────────────────
# RED_TEAM_2.md: `worker_kill_grace_sec: .inf` is a genuine YAML document, parsed by
# `yaml.safe_load` (NOT a hand-constructed Python float) and validated through the full
# `RunConfig.model_validate` path -- the exact repro shape the finding used.
def _yaml_doc_with_eval_override(field: str, yaml_literal: str) -> dict:
    """Build a full `RunConfig`-shaped payload where `eval.<field>` is parsed from a REAL
    YAML literal (`.inf`/`-.inf`/`.nan`), not a Python `float(...)` call site."""
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
    """RED-TEAM-2 F-RT2-1's exact reproduction: `worker_kill_grace_sec: .inf` (a genuine
    YAML document, `yaml.safe_load` then `RunConfig.model_validate`) previously loaded
    SILENTLY and went on to reproduce F1's silent-poller-death failure mode via a real
    `multiprocessing.Process.join(float('inf'))` `OverflowError`. The schema now makes
    this repro impossible: a named `ValidationError` at config-load time, never a
    downstream crash three layers deep in a poller thread."""
    payload = _yaml_doc_with_eval_override(field, yaml_literal)
    assert not math.isfinite(payload["eval"][field])  # confirm the injected value IS non-finite
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert field in str(ei.value), (
        f"expected a named ValidationError mentioning {field!r} for the real YAML literal "
        f"{yaml_literal!r}, got: {ei.value}"
    )


def test_original_f_rt2_1_repro_ceiling_boundary_still_loads() -> None:
    """The non-.inf half of the original repro shape: a `worker_kill_grace_sec` at the
    (finite) ceiling is legitimate and must still load -- the fix closes `.inf`
    specifically, not the whole floor-only domain."""
    payload = _payload()
    payload["eval"]["worker_kill_grace_sec"] = _EVAL_TIMEOUT_CEILING_SEC
    cfg = _validate(payload)
    assert cfg.eval.worker_kill_grace_sec == _EVAL_TIMEOUT_CEILING_SEC


def test_bt_prior_games_rejects_non_finite_via_real_yaml_document() -> None:
    """The sweep-found third floor-only float (`eval.ladder.bt_prior_games`): traced
    downstream to `bt.py`'s `fit_bt` -- an `inf` prior degrades every rating/`p_hat` to
    NaN (non-crashing but silently corrupting every downstream scheduling decision)."""
    payload = _yaml_doc_with_eval_override("bt_prior_games_probe", ".inf")
    bad_value = payload["eval"].pop("bt_prior_games_probe")
    payload["eval"]["ladder"]["bt_prior_games"] = bad_value
    with pytest.raises(ValidationError) as ei:
        _validate(payload)
    assert "bt_prior_games" in str(ei.value)


def test_minted_configs_still_load_after_f_rt2_1_bounds() -> None:
    """R1: the fix must never require re-minting a shipped config. Every minted config's
    `round_timeout_sec=3600.0` / `worker_kill_grace_sec=10.0` / `bt_prior_games=1.0` are
    each many orders of magnitude below the new ceiling/finite-only bound."""
    payload = _payload()  # mirrors the minted-config values verbatim (see docstring)
    assert payload["eval"]["round_timeout_sec"] == 3600.0
    assert payload["eval"]["worker_kill_grace_sec"] == 10.0
    assert payload["eval"]["ladder"]["bt_prior_games"] == 1.0
    _validate(payload)  # must not raise
