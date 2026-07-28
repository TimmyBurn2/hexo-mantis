"""SC-A1 oracle — `TrainConfig` (WPSC Phase 2, DESIGN_P2.md §2 / PREREG_P2.md suite #1).

RED-at-import until IMPL lands `mantis.config.schema.train.TrainConfig` (re-exported from
`mantis.config.schema`). Pins: every field's type/bound round-trips, `extra="forbid"`
rejects an unknown key, every field is REQUIRED (a census loop — one sub-test per field,
missing any single field always raises), and no field carries a pydantic-level default
(R1: the default lives in the minted config, never in the schema).

`entropy_reg_weight`'s negative-value NAMED-error behavior is pinned separately in
`test_train_entropy.py` (R37) — not duplicated here.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.schema import TrainConfig

# Zero-behavior-change mint values (DESIGN_P2.md §1.1/§2): every value is the CURRENT
# `TrainHParams` dataclass default, carried over verbatim.
VALID_TRAIN_PAYLOAD: dict = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "fp16": True,
    "amp_dtype": "fp16",
    "lr_schedule": "cosine",
    "total_steps": 1_000_000,
    "scheduler_t_max": None,
    "eta_min": 5e-4,
    "min_lr": None,
    "checkpoint_interval": 0,
    "actor_sync_cadence_steps": 1,
    "max_train_steps": 1_000_000,
    # WPAX Phase D (R65/R80): REQUIRED key, no code-side default; `None` is the
    # EXPLICIT disarmed posture (R79(1)).
    "draw_rate_abort": None,
    "completed_q_values": False,
    "value_target": "pure_outcome_z",
    "policy_target": "raw_visit_distribution",
    "draw_reward": -0.5,
    "ply_cap_value": -0.5,
    "policy_prune_frac": 0.0,
    "entropy_reg_weight": 0.0,
    "aux_opp_reply_weight": 0.0,
    "uncertainty_weight": 0.0,
    "ownership_weight": 0.0,
    "threat_weight": 0.0,
    "aux_chain_weight": 0.0,
    "ply_index_weight": 0.0,
    "threat_pos_weight": 1.0,
}

FIELD_NAMES = sorted(VALID_TRAIN_PAYLOAD)

# (field, invalid-value) pairs that must violate the field's own `Field(...)` bound —
# `entropy_reg_weight` is deliberately excluded (covered, with its NAMED error, by
# test_train_entropy.py).
BOUND_VIOLATIONS: list[tuple[str, object]] = [
    ("lr", 0.0),
    ("lr", -1e-3),
    ("weight_decay", -1e-4),
    ("grad_clip", 0.0),
    ("total_steps", 0),
    ("scheduler_t_max", 0),
    ("eta_min", -1e-9),
    ("min_lr", -1e-9),
    ("checkpoint_interval", -1),
    ("policy_prune_frac", -0.01),
    ("policy_prune_frac", 1.0),
    ("aux_opp_reply_weight", -0.1),
    ("uncertainty_weight", -0.1),
    ("ownership_weight", -0.1),
    ("threat_weight", -0.1),
    ("aux_chain_weight", -0.1),
    ("ply_index_weight", -0.1),
    ("threat_pos_weight", 0.0),
]

LITERAL_VIOLATIONS: list[tuple[str, object]] = [
    ("amp_dtype", "fp8"),
    ("lr_schedule", "step"),
    ("value_target", "raw_z"),
    ("policy_target", "completed_q"),
]


def _payload(**over: object) -> dict:
    out = dict(VALID_TRAIN_PAYLOAD)
    out.update(over)
    return out


def test_valid_payload_constructs_clean():
    cfg = TrainConfig.model_validate(VALID_TRAIN_PAYLOAD)
    assert cfg.lr == 1e-3
    assert cfg.value_target == "pure_outcome_z"
    assert cfg.scheduler_t_max is None


@pytest.mark.parametrize("field", FIELD_NAMES)
def test_missing_field_rejected(field: str):
    payload = _payload()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        TrainConfig.model_validate(payload)


def test_extra_key_rejected():
    payload = _payload(bogus_train_knob=1)
    with pytest.raises(ValidationError, match="bogus_train_knob"):
        TrainConfig.model_validate(payload)


@pytest.mark.parametrize("field,bad_value", BOUND_VIOLATIONS,
                         ids=[f"{f}={v}" for f, v in BOUND_VIOLATIONS])
def test_bound_violation_rejected(field: str, bad_value: object):
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(_payload(**{field: bad_value}))


@pytest.mark.parametrize("field,bad_value", LITERAL_VIOLATIONS,
                         ids=[f"{f}={v}" for f, v in LITERAL_VIOLATIONS])
def test_literal_out_of_enum_rejected(field: str, bad_value: object):
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(_payload(**{field: bad_value}))


def test_no_field_has_a_pydantic_level_default():
    # R1: a default lives ONLY in the minted config, never in the schema field itself —
    # every TrainConfig field must be required (is_required()==True).
    for name, field in TrainConfig.model_fields.items():
        assert field.is_required(), f"TrainConfig.{name} has a code-side default"


def test_scheduler_t_max_and_min_lr_none_is_a_real_value_not_a_missing_key():
    # `None` satisfies the `int | None` / `float | None` union — but the KEY itself is
    # still required (DESIGN_P2.md §2: "no terminal default; None is a real value here").
    cfg = TrainConfig.model_validate(_payload(scheduler_t_max=None, min_lr=None))
    assert cfg.scheduler_t_max is None
    assert cfg.min_lr is None
