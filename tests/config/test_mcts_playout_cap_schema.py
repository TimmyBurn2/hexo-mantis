"""SC-A2 oracle — `MctsConfig` / `PlayoutCapConfig` census (DESIGN_P2.md §3 /
PREREG_P2.md suite #4, split from test_selfplay_schema.py per R8's 300-line soft cap).

RED-at-import until IMPL lands `mantis.config.schema.selfplay.MctsConfig` /
`PlayoutCapConfig`. The `PlayoutCapConfig` mutual-exclusion `model_validator` (the three
named-error checks, including the Phase-2 "PCR quick>full" REV1 addition) is pinned
separately in test_selfplay_playout_cap_mutual_exclusion.py — not duplicated here; every
payload below stays inert w.r.t. that validator (all playout_cap defaults are 0/0.0).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.schema import MctsConfig, PlayoutCapConfig

VALID_MCTS: dict = {
    "n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25, "quiescence_enabled": True,
    "quiescence_blend_2": 0.3, "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
    "dirichlet_enabled": True,
}
VALID_PLAYOUT_CAP: dict = {
    "fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0, "full_search_prob": 0.0,
    "n_sims_quick": 0, "n_sims_full": 0, "zoi_enabled": False, "zoi_lookback": 16,
    "zoi_margin": 5, "temperature_threshold_compound_moves": 0, "temp_min": 0.5,
}
MCTS_FIELDS = sorted(VALID_MCTS)
PLAYOUT_CAP_FIELDS = sorted(VALID_PLAYOUT_CAP)

MCTS_BOUND_VIOLATIONS: list[tuple[str, object]] = [
    ("n_simulations", 0), ("c_puct", 0.0), ("quiescence_blend_2", 1.1),
    ("quiescence_blend_2", -0.1), ("dirichlet_alpha", 0.0), ("dirichlet_epsilon", 1.1),
    ("dirichlet_epsilon", -0.1),
]
PLAYOUT_CAP_BOUND_VIOLATIONS: list[tuple[str, object]] = [
    ("fast_sims", 0), ("fast_prob", 1.1), ("fast_prob", -0.1), ("standard_sims", -1),
    ("full_search_prob", 1.1), ("n_sims_quick", -1), ("n_sims_full", -1),
    ("zoi_lookback", -1), ("zoi_margin", -1), ("temperature_threshold_compound_moves", -1),
    ("temp_min", -0.1),
]


def _mcts(**over: object) -> dict:
    out = dict(VALID_MCTS)
    out.update(over)
    return out


def _playout_cap(**over: object) -> dict:
    out = dict(VALID_PLAYOUT_CAP)
    out.update(over)
    return out


# ── MctsConfig ────────────────────────────────────────────────────────────────────────
def test_mcts_valid_payload_constructs_clean():
    cfg = MctsConfig.model_validate(VALID_MCTS)
    assert cfg.n_simulations == 50
    assert cfg.dirichlet_epsilon == 0.25


@pytest.mark.parametrize("field", MCTS_FIELDS)
def test_mcts_missing_field_rejected(field: str):
    payload = _mcts()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        MctsConfig.model_validate(payload)


def test_mcts_extra_key_rejected():
    with pytest.raises(ValidationError, match="bogus_mcts_knob"):
        MctsConfig.model_validate(_mcts(bogus_mcts_knob=1))


@pytest.mark.parametrize("field,bad_value", MCTS_BOUND_VIOLATIONS,
                         ids=[f"{f}={v}" for f, v in MCTS_BOUND_VIOLATIONS])
def test_mcts_bound_violation_rejected(field: str, bad_value: object):
    with pytest.raises(ValidationError):
        MctsConfig.model_validate(_mcts(**{field: bad_value}))


def test_mcts_has_no_pydantic_level_default():
    for name, field in MctsConfig.model_fields.items():
        assert field.is_required(), f"MctsConfig.{name} has a code-side default"


# ── PlayoutCapConfig ──────────────────────────────────────────────────────────────────
def test_playout_cap_valid_payload_constructs_clean():
    cfg = PlayoutCapConfig.model_validate(VALID_PLAYOUT_CAP)
    assert cfg.fast_sims == 50


@pytest.mark.parametrize("field", PLAYOUT_CAP_FIELDS)
def test_playout_cap_missing_field_rejected(field: str):
    payload = _playout_cap()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        PlayoutCapConfig.model_validate(payload)


def test_playout_cap_extra_key_rejected():
    with pytest.raises(ValidationError, match="bogus_playout_cap_knob"):
        PlayoutCapConfig.model_validate(_playout_cap(bogus_playout_cap_knob=1))


@pytest.mark.parametrize("field,bad_value", PLAYOUT_CAP_BOUND_VIOLATIONS,
                         ids=[f"{f}={v}" for f, v in PLAYOUT_CAP_BOUND_VIOLATIONS])
def test_playout_cap_bound_violation_rejected(field: str, bad_value: object):
    with pytest.raises(ValidationError):
        PlayoutCapConfig.model_validate(_playout_cap(**{field: bad_value}))


def test_playout_cap_field_name_matches_config_key_for_temperature_threshold():
    # DESIGN_P2.md §3: `temperature_threshold_compound_moves` is now BOTH the field name
    # and the config key — retiring hparams.py's `_resolve_playout_cap_temperature` key/
    # field-spelling mismatch shim by construction.
    cfg = PlayoutCapConfig.model_validate(_playout_cap(temperature_threshold_compound_moves=9))
    assert cfg.temperature_threshold_compound_moves == 9


def test_playout_cap_has_no_pydantic_level_default():
    for name, field in PlayoutCapConfig.model_fields.items():
        assert field.is_required(), f"PlayoutCapConfig.{name} has a code-side default"
