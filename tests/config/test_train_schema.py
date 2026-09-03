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

from mantis.config.schema import ARCH_SCOPED_KEYS, TrainConfig

# Zero-behavior-change mint values (DESIGN_P2.md §1.1/§2): every value is the CURRENT
# `TrainHParams` dataclass default, carried over verbatim.
#
# WPMINT Phase K-A stage 0 consolidated eleven other copies of this block onto a MINTED
# config and DELIBERATELY LEFT THIS ONE HAND-WRITTEN. This census is not payload scaffolding
# here, it is the SUBJECT: `FIELD_NAMES` is derived from it and drives
# `test_missing_field_rejected`, so the file's claim is "the schema requires exactly these
# fields, independently written down". Deriving it from a config the schema itself validated
# makes that claim circular — the enumeration and the thing enumerated would come from the
# same source, and a field the schema stopped requiring would silently leave both. The
# maintenance cost (one line per new `train.*` key) is the price of the independence, and it
# is the only place in the suite that pays it.
VALID_TRAIN_PAYLOAD: dict = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "fp16": True,
    "amp_dtype": "fp16",
    # WPMAIN / R126: `train.device` is the run device, a CONFIG FACT with a CLOSED
    # vocabulary and no code-side default (the retired `--device` flag on both callers).
    "device": "cpu",
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
    # WPMINT Phase K-B (CARD-COORD-KNOBS, R78 as clarified by R80): the step-coordinator
    # knobs — 19 at K-B, 18 since R178(a) deleted `buffer_save_interval` as a dead knob
    # (R116/LAW-08). Every value is the one `mantis.run._step_coordinator_config` used, so
    # this census records a change of AUTHOR and not of behaviour — except `batch_size`,
    # which is 256 because K-A MEASURED that the production path's dict lookups both missed
    # and the run really used the literal 256, never the dead field's 8.
    "eval_interval": 1000,
    "log_interval": 1000,
    "min_buf_size": 1,
    "replay_capacity": 100_000,
    "replay_capacity_schedule": [],
    "training_steps_per_game": 1.0,
    "max_train_burst": 1,
    "batch_size": 256,
    # WP12-R F2 (CARD-RUN5-GPU-OOM, R179): ONE block, TWO inseparable members. The minted
    # value here is the TEMPLATE's non-binding pair; run5 overrides both with the operator's
    # sized values at mint. `batch_size` above bounds the number of GRAPHS and bounds neither
    # quantity that drives memory — E and N are sums over the sampled graphs, and nothing
    # bounded either before this block existed.
    "microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
    "augment": False,
    "recency_weight": 0.0,
    "mixing_initial_w": 0.0,
    "mixing_min_w": 0.0,
    "mixing_decay_steps": 1.0,
    "hard_gn_threshold": 1e9,
    "hard_gn_min_steps": 3,
    "terminal_eval_enabled": True,
    "bot_batch_share": 0.0,
    "selfplay_stall_timeout_sec": 1800.0,
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
    # AUDIT-1 F-06 / R332(d): `train.ema` is a REQUIRED block. `enabled: false` is what every
    # committed config mints — the posture stated, not inherited from a code-side default.
    "ema": {"enabled": False, "decay": 0.999, "update_every": 10},
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
    # WPMINT Phase K-B — one violation per knob whose bound makes a real defect
    # inexpressible, named at the value that defect is actually written as.
    ("eval_interval", 0),               # the entire eval/promotion pipeline, silently off
    ("log_interval", 0),                # DR-7: the whole hard-abort family AND monitor_gates
    ("min_buf_size", 0),                # "train on an empty buffer"
    ("replay_capacity", 0),
    ("training_steps_per_game", 0.0),   # reads as off; `_steps_budget`'s max(1, ...) is not
    ("max_train_burst", 0),             # here it really does stop the learner forever
    ("batch_size", 0),
    # WP12-R F2: `ge=1` on BOTH members, and the bound is the MECHANISM's own range, not
    # policy — a micro-batch of zero edges (or zero nodes) is not a micro-batch. There is no
    # off value and no disable sentinel: an uncapped graph step is the defect the block exists
    # to make unconstructible, so a sentinel would be a switch for turning the fix off (R79).
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
    ("mixing_initial_w", 1.5),
    ("mixing_min_w", -0.1),
    ("mixing_decay_steps", 0.0),        # a divisor: ZeroDivisionError on the first mixed step
    ("hard_gn_threshold", 0.0),         # fires on every finite step
    ("hard_gn_threshold", float("inf")),  # accepted, reads ARMED, can never be met
    ("hard_gn_min_steps", 0),           # fires on the FIRST breach — not "sustained"
    ("bot_batch_share", 1.5),
    ("selfplay_stall_timeout_sec", 0.0),   # LAW-16's always-armed guard, silently disarmed
    ("selfplay_stall_timeout_sec", -1.0),
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


#: The section-level missing-key oracle skips the ARCH-SCOPED blocks (R322(d)): they are
#: omittable AT THIS LEVEL by design, and their required-ness is a `RunConfig` fact because it
#: depends on `identity.representation`, which `TrainConfig` cannot see. Derived, not listed.
_ARCH_SCOPED_TRAIN_FIELDS = frozenset(
    key.field for key in ARCH_SCOPED_KEYS if key.section == "train"
)
REQUIRED_FIELD_NAMES = [f for f in FIELD_NAMES if f not in _ARCH_SCOPED_TRAIN_FIELDS]


@pytest.mark.parametrize("field", REQUIRED_FIELD_NAMES)
def test_missing_field_rejected(field: str):
    payload = _payload()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        TrainConfig.model_validate(payload)


@pytest.mark.parametrize("field", sorted(_ARCH_SCOPED_TRAIN_FIELDS))
def test_an_arch_scoped_field_is_OMITTABLE_at_the_section_level(field: str):
    """The complement, asserted rather than left as a gap in the parametrize list: the block
    must actually be omittable here, or a grid `RunConfig` could not be built at all."""
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
    # R1: a default lives ONLY in the minted config, never in the schema field itself —
    # every TrainConfig field must be required (is_required()==True).
    #
    # THE ONE EXCEPTION IS DERIVED, NOT LISTED (R322(d)). An arch-scoped block carries
    # `= None` so a config of another representation may OMIT it, and that `None` is not a
    # fallback: `RunConfig._arch_scoped_keys_are_present_iff_their_arch` REFUSES a config of
    # the owning arch that omits the block, and refuses one of any other arch that carries it.
    # So R1's force is intact — there is still no key whose absence silently yields a value —
    # and the exempt set is read off `ARCH_SCOPED_KEYS` rather than typed here, so a
    # hand-added default on any other field is still a red. The other side of the rule —
    # that omitting the block on its OWN arch is an error — is executed by the conformance
    # suite's T9 section, against a real minted file rather than a payload built here.
    exempt = {key.field for key in ARCH_SCOPED_KEYS if key.section == "train"}
    assert exempt, "no train key is arch-scoped, so this exemption is unused and should go"
    for name, field in TrainConfig.model_fields.items():
        if name in exempt:
            assert not field.is_required(), (
                f"TrainConfig.{name} is arch-scoped, so it must be omittable — a required "
                "arch-scoped block would force every arch to mint it, which is the defect"
            )
            continue
        assert field.is_required(), f"TrainConfig.{name} has a code-side default"



def test_scheduler_t_max_and_min_lr_none_is_a_real_value_not_a_missing_key():
    # `None` satisfies the `int | None` / `float | None` union — but the KEY itself is
    # still required (DESIGN_P2.md §2: "no terminal default; None is a real value here").
    cfg = TrainConfig.model_validate(_payload(scheduler_t_max=None, min_lr=None))
    assert cfg.scheduler_t_max is None
    assert cfg.min_lr is None
