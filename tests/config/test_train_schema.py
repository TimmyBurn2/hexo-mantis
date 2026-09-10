"""Oracle for `TrainConfig`: types/bounds, `extra="forbid"`, required-ness, no schema defaults.

`entropy_reg_weight`'s negative-value named error is pinned in `test_train_entropy.py`.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.schema import ARCH_SCOPED_KEYS, TrainConfig, operational_default_fields

# Deliberately hand-written, not derived from a minted config: `FIELD_NAMES` comes from this
# census and drives `test_missing_field_rejected`, so deriving it would make the claim circular.
VALID_TRAIN_PAYLOAD: dict = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "device": "cpu",
    "lr_schedule": "cosine",
    "total_steps": 1_000_000,
    "scheduler_t_max": None,
    "eta_min": 5e-4,
    "checkpoint_interval": 0,
    "actor_sync_cadence_steps": 1,
    "max_train_steps": 1_000_000,
    # `None` is the explicit disarmed posture, not an absent key.
    "draw_rate_abort": None,
    # Step-coordinator knobs, at the values `mantis.run._step_coordinator_config` used.
    # `batch_size` is 256 because the run was measured using that literal, not the dead field's 8.
    "eval_interval": 1000,
    "log_interval": 1000,
    "min_buf_size": 1,
    "replay_capacity": 100_000,
    "replay_capacity_schedule": [],
    "training_steps_per_game": 1.0,
    "max_train_burst": 1,
    "batch_size": 256,
    # Two inseparable members: `batch_size` bounds graphs, not E and N, which drive memory.
    "microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
    "augment": False,
    "recency_weight": 0.0,
    "hard_gn_threshold": 1e9,
    "hard_gn_min_steps": 3,
    "terminal_eval_enabled": True,
    "selfplay_stall_timeout_sec": 1800.0,
    "value_target": "pure_outcome_z",
    "policy_target": "raw_visit_distribution",
    "draw_reward": -0.5,
    "ply_cap_value": -0.5,
    "fast_policy_weight": 0.0,
    "ema": {"enabled": False, "decay": 0.999, "update_every": 10},
}

FIELD_NAMES = sorted(VALID_TRAIN_PAYLOAD)

# (field, invalid-value) pairs violating the field's own bound; `entropy_reg_weight` is
# covered by test_train_entropy.py instead.
BOUND_VIOLATIONS: list[tuple[str, object]] = [
    ("lr", 0.0),
    ("lr", -1e-3),
    ("weight_decay", -1e-4),
    ("grad_clip", 0.0),
    ("total_steps", 0),
    ("scheduler_t_max", 0),
    ("eta_min", -1e-9),
    ("checkpoint_interval", -1),
    ("fast_policy_weight", -0.1),
    # One violation per knob whose bound makes a real defect inexpressible.
    ("eval_interval", 0),               # the entire eval/promotion pipeline, silently off
    ("log_interval", 0),                # the whole hard-abort family AND monitor_gates
    ("min_buf_size", 0),                # "train on an empty buffer"
    ("replay_capacity", 0),
    ("training_steps_per_game", 0.0),   # reads as off; `_steps_budget`'s max(1, ...) is not
    ("max_train_burst", 0),             # here it really does stop the learner forever
    ("batch_size", 0),
    # `ge=1` on both members: there is no off value, because an uncapped graph step is the
    # defect the block exists to make unconstructible.
    ("microbatch_caps", {"max_edges": 0, "max_nodes": 1}),
    ("microbatch_caps", {"max_edges": 1, "max_nodes": 0}),
    ("microbatch_caps", {"max_edges": -1, "max_nodes": 1}),
    # both members arrive together or not at all — one alone bounds only one term of
    # `peak ~ a + b*E + c*N`
    ("microbatch_caps", {"max_edges": 1}),
    ("microbatch_caps", {"max_nodes": 1}),
    # `extra="forbid"` reaches INTO the block, so a third member cannot be smuggled in
    ("microbatch_caps", {"max_edges": 1, "max_nodes": 1, "max_bytes": 1}),
    ("recency_weight", -0.1),
    ("recency_weight", 1.1),            # the sampler clamps, so above 1 is a difference the
                                        # config can express and the run cannot have
    ("hard_gn_threshold", 0.0),         # fires on every finite step
    ("hard_gn_threshold", float("inf")),  # accepted, reads ARMED, can never be met
    ("hard_gn_min_steps", 0),           # fires on the FIRST breach — not "sustained"
    ("selfplay_stall_timeout_sec", 0.0),   # LAW-16's always-armed guard, silently disarmed
    ("selfplay_stall_timeout_sec", -1.0),
]

LITERAL_VIOLATIONS: list[tuple[str, object]] = [
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


#: Arch-scoped blocks are omittable at this level: their required-ness depends on
#: `identity.representation`, which `TrainConfig` cannot see. Derived, not listed.
_ARCH_SCOPED_TRAIN_FIELDS = frozenset(
    key.field for key in ARCH_SCOPED_KEYS if key.section == "train"
)
_OPERATIONAL_TRAIN_FIELDS = operational_default_fields("train")
REQUIRED_FIELD_NAMES = [f for f in FIELD_NAMES
                        if f not in _ARCH_SCOPED_TRAIN_FIELDS | _OPERATIONAL_TRAIN_FIELDS]


@pytest.mark.parametrize("field", REQUIRED_FIELD_NAMES)
def test_missing_field_rejected(field: str):
    payload = _payload()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        TrainConfig.model_validate(payload)


@pytest.mark.parametrize("field", sorted(_OPERATIONAL_TRAIN_FIELDS))
def test_an_operational_field_is_OMITTABLE_and_lands_on_its_declared_default(field: str):
    """Omitting an operational field is legal AND lands on the schema's own declared value."""
    payload = _payload()
    del payload[field]
    cfg = TrainConfig.model_validate(payload)
    assert getattr(cfg, field) == TrainConfig.model_fields[field].get_default(
        call_default_factory=True), f"train.{field} did not land on its schema default"


@pytest.mark.parametrize("field", sorted(_ARCH_SCOPED_TRAIN_FIELDS))
def test_an_arch_scoped_field_is_OMITTABLE_at_the_section_level(field: str):
    """An arch-scoped block is omittable at the section level."""
    payload = _payload()
    del payload[field]
    assert getattr(TrainConfig.model_validate(payload), field) is None


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


def test_no_field_has_a_pydantic_level_default_EXCEPT_the_arch_scoped_ones():
    # A default lives only in the minted config, so every field must be required. Both exempt
    # families are read off their registries, never typed here, so a hand-added default is red.
    exempt = {key.field for key in ARCH_SCOPED_KEYS if key.section == "train"}
    assert exempt, "no train key is arch-scoped, so this exemption is unused and should go"
    # Kept separate from the arch-scoped family: an arch-scoped block is REFUSED on the wrong
    # arch, while an operational default is simply inherited.
    operational = operational_default_fields("train")
    assert not (exempt & operational), "a key cannot be both arch-scoped and operational"
    for name, field in TrainConfig.model_fields.items():
        if name in exempt:
            assert not field.is_required(), (
                f"TrainConfig.{name} is arch-scoped, so it must be omittable — a required "
                "arch-scoped block would force every arch to mint it, which is the defect"
            )
            continue
        if name in operational:
            assert not field.is_required(), (
                f"TrainConfig.{name} is declared in OPERATIONAL_DEFAULT_KEYS but is still "
                "required — the declaration is stale"
            )
            continue
        assert field.is_required(), (
            f"TrainConfig.{name} has a code-side default and is declared in neither registry"
        )



def test_scheduler_t_max_none_is_a_real_value_not_a_missing_key():
    # `None` satisfies the union, but the key itself is still required.
    cfg = TrainConfig.model_validate(_payload(scheduler_t_max=None))
    assert cfg.scheduler_t_max is None
