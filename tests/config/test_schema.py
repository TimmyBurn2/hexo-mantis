"""Schema contract tests (run-config-schema v1): unknown/missing keys hard-fail,
representation is the closed set {grid, graph}, and O16 schema round-trip + every-config-
validates + no-code-side-defaults across the full model tree."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.loader import load_config
from mantis.config.schema import (
    SCHEMA_VERSION,
    EvalConfig,
    IdentityConfig,
    RadiusStage,
    RunConfig,
    SelfplayConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": {"random_model_sims": 96, "sealbot_model_sims": 128},
        "selfplay": {"legal_move_radius_schedule": None},
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
    del payload["selfplay"]["legal_move_radius_schedule"]
    with pytest.raises(ValidationError, match="legal_move_radius_schedule"):
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
    configs = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    assert configs, "no committed configs found (gate 7 must never be vacuous)"
    for cfg_path in configs:
        load_config(cfg_path)  # raises on any failure


def test_o16_schema_round_trip():
    cfg = load_config(REPO_ROOT / "configs" / "run5.yaml")
    again = RunConfig.model_validate(cfg.model_dump())
    assert again == cfg


def test_o16_all_fields_required_no_code_side_defaults():
    for model in (RunConfig, IdentityConfig, EvalConfig, SelfplayConfig, RadiusStage):
        for name, field in model.model_fields.items():
            assert field.is_required(), f"{model.__name__}.{name} has a code-side default"
