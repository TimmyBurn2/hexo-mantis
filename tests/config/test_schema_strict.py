"""O14 — strict-coercion rejection (StrictModel with strict=True).

strict=True rejects silent scalar coercions (str->int, float->int, bool->int). yaml-native
scalar types satisfy strict, so every committed + reminted config parses clean.
"""
import pytest
from pydantic import ValidationError

from mantis.config.schema import SCHEMA_VERSION, RunConfig


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": {"random_model_sims": 96, "sealbot_model_sims": 128},
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
