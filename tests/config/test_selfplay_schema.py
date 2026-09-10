"""Oracle for `SelfplayConfig` and `InferenceConfig`: required fields, `extra="forbid"`, bounds.

The nested `MctsConfig`/`PlayoutCapConfig` census and the playout-cap mutual-exclusion
validator live in `test_mcts_playout_cap_schema.py` and
`test_selfplay_playout_cap_mutual_exclusion.py`.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.schema import (
    ARCH_SCOPED_KEYS,
    InferenceConfig,
    SelfplayConfig,
    operational_default_fields,
)
from mantis.config.schema.selfplay import MAX_ARMED_SIMS, MAX_ARMED_SIMS_GUMBEL



VALID_MCTS: dict = {
    "n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25, "quiescence_enabled": True,
    "quiescence_blend_2": 0.3, "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
    "dirichlet_enabled": True,
}
VALID_PLAYOUT_CAP: dict = {
    "fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0, "full_search_prob": 0.0,
    "n_sims_quick": 0, "n_sims_full": 0,
    "temperature_threshold_compound_moves": 0, "temp_min": 0.5,
}
VALID_SELFPLAY: dict = {
    "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
    "c_visit": 50.0,
    "c_scale": 1.0, "gumbel_m": 16, "gumbel_explore_moves": 10,
    "results_queue_cap": 10_000, "random_opening_plies": 0,
    "log_investigation_metrics": True,
    "mcts": dict(VALID_MCTS),
    "playout_cap": dict(VALID_PLAYOUT_CAP),
}
VALID_INFERENCE: dict = {
    "inference_batch_size": 64, "inference_max_wait_ms": 10,
    # A non-binding-by-construction pair: no split is exercised here, and the `null`
    # placeholder is pinned against the real configs in test_fused_graph_caps_authority.py.
    "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
}

SELFPLAY_FIELDS = sorted(VALID_SELFPLAY)
INFERENCE_FIELDS = sorted(VALID_INFERENCE)

SELFPLAY_BOUND_VIOLATIONS: list[tuple[str, object]] = [
    ("n_workers", 0), ("leaf_batch_size", 0), ("max_game_moves", 0),
    ("c_visit", 0.0), ("c_scale", 0.0), ("gumbel_m", 0),
    ("gumbel_explore_moves", -1), ("results_queue_cap", 0), ("random_opening_plies", -1),
]
INFERENCE_BOUND_VIOLATIONS: list[tuple[str, object]] = [
    ("inference_batch_size", 0), ("inference_max_wait_ms", -1),
]


def _selfplay(**over: object) -> dict:
    out = dict(VALID_SELFPLAY)
    out.update(over)
    return out


def _inference(**over: object) -> dict:
    out = dict(VALID_INFERENCE)
    out.update(over)
    return out


def test_selfplay_valid_payload_constructs_clean():
    cfg = SelfplayConfig.model_validate(VALID_SELFPLAY)
    assert cfg.n_workers == 1
    assert cfg.mcts.n_simulations == 50
    assert cfg.playout_cap.fast_sims == 50


@pytest.mark.parametrize("field",
                         sorted(set(SELFPLAY_FIELDS) - operational_default_fields("selfplay")))
def test_selfplay_missing_field_rejected(field: str):
    payload = _selfplay()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        SelfplayConfig.model_validate(payload)


def test_selfplay_extra_key_rejected():
    with pytest.raises(ValidationError, match="bogus_selfplay_knob"):
        SelfplayConfig.model_validate(_selfplay(bogus_selfplay_knob=1))


def test_selfplay_nested_extra_key_rejected():
    payload = _selfplay()
    payload["mcts"] = dict(VALID_MCTS, bogus_mcts_knob=1)
    with pytest.raises(ValidationError, match="bogus_mcts_knob"):
        SelfplayConfig.model_validate(payload)


@pytest.mark.parametrize("field,bad_value", SELFPLAY_BOUND_VIOLATIONS,
                         ids=[f"{f}={v}" for f, v in SELFPLAY_BOUND_VIOLATIONS])
def test_selfplay_bound_violation_rejected(field: str, bad_value: object):
    with pytest.raises(ValidationError):
        SelfplayConfig.model_validate(_selfplay(**{field: bad_value}))


def test_selfplay_has_no_pydantic_level_default_EXCEPT_the_declared_operational_ones():
    """Check both ways: an undeclared default is a red, a declared-but-required key is stale."""
    declared = operational_default_fields("selfplay")
    for name, field in SelfplayConfig.model_fields.items():
        if name in declared:
            assert not field.is_required(), (
                f"SelfplayConfig.{name} is declared in OPERATIONAL_DEFAULT_KEYS but is still "
                "required — the declaration is stale"
            )
            continue
        assert field.is_required(), (
            f"SelfplayConfig.{name} has a code-side default and is not declared operational"
        )


@pytest.mark.parametrize("field", sorted(operational_default_fields("selfplay")))
def test_an_omitted_selfplay_operational_key_lands_on_its_declared_default(field: str):
    """Omitting a declared operational key is legal and lands on the schema's own value."""
    payload = _selfplay()
    del payload[field]
    cfg = SelfplayConfig.model_validate(payload)
    assert getattr(cfg, field) == SelfplayConfig.model_fields[field].get_default(
        call_default_factory=True), f"selfplay.{field} did not land on its schema default"


def test_selfplay_has_no_legal_move_radius_field():
    # The registry alone is the radius authority.
    assert "legal_move_radius" not in SelfplayConfig.model_fields
    assert "legal_move_radius_schedule" not in SelfplayConfig.model_fields


def test_dirichlet_epsilon_field_name_equals_config_key():
    # Field name and config key match one-to-one, so a wrong-spelling silent no-op is
    # structurally impossible.
    payload = _selfplay()
    payload["mcts"] = dict(VALID_MCTS, dirichlet_epsilon=0.9)
    cfg = SelfplayConfig.model_validate(payload)
    assert cfg.mcts.dirichlet_epsilon == 0.9


def test_inference_valid_payload_constructs_clean():
    cfg = InferenceConfig.model_validate(VALID_INFERENCE)
    assert cfg.inference_batch_size == 64


#: Arch-scoped blocks are omittable at the section level: their required-ness depends on
#: `identity.representation`, which `InferenceConfig` cannot see. Derived, not listed.
_ARCH_SCOPED_INFERENCE_FIELDS = frozenset(
    key.field for key in ARCH_SCOPED_KEYS if key.section == "inference"
)
REQUIRED_INFERENCE_FIELDS = [f for f in INFERENCE_FIELDS
                             if f not in _ARCH_SCOPED_INFERENCE_FIELDS]


@pytest.mark.parametrize("field", REQUIRED_INFERENCE_FIELDS)
def test_inference_missing_field_rejected(field: str):
    payload = _inference()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        InferenceConfig.model_validate(payload)


@pytest.mark.parametrize("field", sorted(_ARCH_SCOPED_INFERENCE_FIELDS))
def test_an_arch_scoped_inference_field_is_OMITTABLE_at_the_section_level(field: str):
    payload = _inference()
    del payload[field]
    assert getattr(InferenceConfig.model_validate(payload), field) is None


def test_inference_extra_key_rejected():
    with pytest.raises(ValidationError, match="bogus_inference_knob"):
        InferenceConfig.model_validate(_inference(bogus_inference_knob=1))


@pytest.mark.parametrize("field,bad_value", INFERENCE_BOUND_VIOLATIONS,
                         ids=[f"{f}={v}" for f, v in INFERENCE_BOUND_VIOLATIONS])
def test_inference_bound_violation_rejected(field: str, bad_value: object):
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(_inference(**{field: bad_value}))


def test_inference_has_no_pydantic_level_default_EXCEPT_the_arch_scoped_ones():
    """No code-side defaults, with the arch-scoped exception derived from `ARCH_SCOPED_KEYS`.

    The `= None` on an arch-scoped block is not a fallback: `RunConfig` refuses a graph config
    that omits it and any other config that carries it.
    """
    assert _ARCH_SCOPED_INFERENCE_FIELDS, (
        "no inference key is arch-scoped, so this exemption is unused and should go"
    )
    for name, field in InferenceConfig.model_fields.items():
        if name in _ARCH_SCOPED_INFERENCE_FIELDS:
            assert not field.is_required(), (
                f"InferenceConfig.{name} is arch-scoped, so it must be omittable"
            )
            continue
        assert field.is_required(), f"InferenceConfig.{name} has a code-side default"


def test_the_gumbel_kind_lowers_the_sim_ceiling_at_mint(smoke_run_config):
    """The Gumbel kind's lower pool ceiling is a mint refusal, not a boot refusal.

    That kind reaches the root's full legal set, so its ceiling is `MAX_ARMED_SIMS_GUMBEL`,
    below the `MAX_ARMED_SIMS` the field bounds carry. The validator lives on `RunConfig`
    because the applicable ceiling depends on `search.kind`, a different section. Validator
    ORDER is load-bearing: the pool-ceiling check is declared before the HEXG record-format
    one, which would otherwise refuse `gumbel` first and pass this on the wrong ground.
    """
    assert MAX_ARMED_SIMS_GUMBEL < MAX_ARMED_SIMS, (
        "the two ceilings must differ or the refusal below has no domain to fire in"
    )
    in_gap = MAX_ARMED_SIMS_GUMBEL + 1
    assert in_gap <= MAX_ARMED_SIMS, "the probe value must still satisfy the field bound"

    # PUCT at the same budget is accepted: the control arm.
    ok = smoke_run_config(
        "dev_example.yaml", selfplay={"mcts": {"n_simulations": in_gap}}
    )
    assert ok.selfplay.mcts.n_simulations == in_gap

    with pytest.raises(ValidationError, match="MAX_ROOT_CHILDREN"):
        smoke_run_config(
            "dev_example.yaml",
            search={"kind": "gumbel"},
            train={"policy_target": "completed_improved_policy"},
            selfplay={"mcts": {"n_simulations": in_gap}},
        )


def test_every_armed_sims_knob_is_checked_against_the_kinds_ceiling(smoke_run_config):
    """Every armed sims knob, not only `n_simulations`, is checked against the kind's ceiling."""
    over = MAX_ARMED_SIMS_GUMBEL + 1
    for key in ("fast_sims", "n_sims_quick", "n_sims_full"):
        with pytest.raises(ValidationError, match=key):
            smoke_run_config(
                "dev_example.yaml",
                search={"kind": "gumbel"},
                train={"policy_target": "completed_improved_policy"},
                selfplay={"playout_cap": {key: over}},
            )
