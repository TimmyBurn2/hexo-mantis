"""O14 — strict-coercion rejection (StrictModel with strict=True).

strict=True rejects silent scalar coercions (str->int, float->int, bool->int). yaml-native
scalar types satisfy strict, so every committed + reminted config parses clean.
"""
import pytest
from pydantic import ValidationError

from mantis.config.schema import SCHEMA_VERSION, RunConfig


# WP11-A schema extension: eval.gate/eval.ladder are now required fields (design §c.1).
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


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": _valid_eval_block(),
        "selfplay": {"legal_move_radius_schedule": None},
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


def test_radius_stage_float_step_rejected():
    payload = _valid_payload()
    payload["selfplay"]["legal_move_radius_schedule"] = [{"step": 0.0, "radius": 5}]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_correctly_typed_config_validates_clean():
    cfg = RunConfig.model_validate(_valid_payload())
    assert cfg.seed == 1
    assert cfg.eval.random_model_sims == 96
