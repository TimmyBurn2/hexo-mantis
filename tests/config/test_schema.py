"""Schema contract tests (run-config-schema v1): unknown/missing keys hard-fail."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.loader import load_config
from mantis.config.schema import SCHEMA_VERSION, IdentityConfig, RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
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


def test_wrong_schema_version_rejected():
    payload = _valid_payload()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValidationError, match="schema_version"):
        RunConfig.model_validate(payload)


def test_representation_closed_set_rejects_grid():
    payload = _valid_payload()
    payload["identity"]["representation"] = "grid"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_all_fields_required_no_code_side_defaults():
    for model in (RunConfig, IdentityConfig):
        for name, field in model.model_fields.items():
            assert field.is_required(), f"{model.__name__}.{name} has a code-side default"
