"""O14 — strict-coercion rejection (StrictModel with strict=True).

strict=True rejects silent scalar coercions (str->int, float->int, bool->int). yaml-native
scalar types satisfy strict, so every committed + reminted config parses clean.
"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from mantis.config.schema import SCHEMA_VERSION, RunConfig
from _schema_blocks import eval_block, inference_block, monitor_block, selfplay_block, train_block


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_enabled": True,
        # RECAL-PREP (R308(g)(i)): a REQUIRED top-level leaf. `null` is R119's
        # placeholder — refused at boot on a cuda process, valued only by the
        # re-calibration sitting under R282(b).
        "allocator_posture": None,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "model": {"gnn": {"hidden": 128, "num_layers": 4}, "aux_soft_policy": None},
        "eval": eval_block(),
        "train": train_block(),
        "deploy": {"search": {"kind": "puct"}},
        "selfplay": selfplay_block(),
        "inference": inference_block(),
        "monitor": monitor_block(),
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
