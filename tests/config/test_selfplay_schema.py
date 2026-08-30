"""SC-A2 oracle — `SelfplayConfig` (top-level) + `InferenceConfig` (WPSC Phase 2,
DESIGN_P2.md §3 / PREREG_P2.md suite #4). `MctsConfig`/`PlayoutCapConfig` census (the two
nested sub-models) + the playout-cap mutual-exclusion validator live in the sibling files
`test_mcts_playout_cap_schema.py` / `test_selfplay_playout_cap_mutual_exclusion.py`
(R8: keeps every file under the 300-line soft cap).

RED-at-import until IMPL lands `mantis.config.schema.selfplay.SelfplayConfig` /
`InferenceConfig`. Census pattern (as test_train_schema.py): every field required
(missing-key census, one sub-test per field), `extra="forbid"` at this nesting level,
bounds round-trip.

`legal_move_radius`/`legal_move_radius_schedule` are DELIBERATELY ABSENT from both models
(DESIGN_P2.md §5, shape (ii) — the registry alone is the radius authority) — their absence
is pinned separately in test_radius_removed.py, not re-asserted here.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.schema import ARCH_SCOPED_KEYS, InferenceConfig, SelfplayConfig

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
VALID_SELFPLAY: dict = {
    "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
    "inference_pool_size": None, "completed_q_values": False, "c_visit": 50.0,
    "c_scale": 1.0, "gumbel_mcts": False, "gumbel_m": 16, "gumbel_explore_moves": 10,
    "results_queue_cap": 10_000, "random_opening_plies": 0, "rotation_enabled": True,
    "forced_win_policy_enabled": False, "forced_win_policy_depth": 2,
    "forced_win_policy_weight": 1.0, "solver_enabled": False, "solver_depth": 16,
    "solver_node_budget": 50_000, "solver_neighbor_dist": 2, "solver_visit_weight": 0.3,
    "seed_fraction": 0.0, "seed_corpus_path": None, "log_investigation_metrics": True,
    "instrumentation_enabled": False, "mcts": dict(VALID_MCTS),
    "playout_cap": dict(VALID_PLAYOUT_CAP),
}
VALID_INFERENCE: dict = {
    "inference_batch_size": 64, "inference_max_wait_ms": 10, "trace_inference": True,
    "compile_inference": False, "compile_inference_mode": "default",
    "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
    # F-816-10: `inference.fused_graph_caps` is a REQUIRED block. The pair here is
    # the template's NON-BINDING-BY-CONSTRUCTION value, so nothing in this file
    # exercises a split; the R119 `null` placeholder is pinned by
    # tests/config/test_fused_graph_caps_authority.py against the real configs.
    "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
}

SELFPLAY_FIELDS = sorted(VALID_SELFPLAY)
INFERENCE_FIELDS = sorted(VALID_INFERENCE)

SELFPLAY_BOUND_VIOLATIONS: list[tuple[str, object]] = [
    ("n_workers", 0), ("leaf_batch_size", 0), ("max_game_moves", 0),
    ("inference_pool_size", 0), ("c_visit", 0.0), ("c_scale", 0.0), ("gumbel_m", 0),
    ("gumbel_explore_moves", -1), ("results_queue_cap", 0), ("random_opening_plies", -1),
    ("forced_win_policy_depth", 0), ("forced_win_policy_weight", -0.1),
    ("solver_depth", 0), ("solver_node_budget", 0), ("solver_neighbor_dist", -1),
    ("solver_visit_weight", 1.1), ("seed_fraction", 1.1),
]
INFERENCE_BOUND_VIOLATIONS: list[tuple[str, object]] = [
    ("inference_batch_size", 0), ("inference_max_wait_ms", -1), ("compile_inference_mode", ""),
]


def _selfplay(**over: object) -> dict:
    out = dict(VALID_SELFPLAY)
    out.update(over)
    return out


def _inference(**over: object) -> dict:
    out = dict(VALID_INFERENCE)
    out.update(over)
    return out


# ── SelfplayConfig ────────────────────────────────────────────────────────────────────
def test_selfplay_valid_payload_constructs_clean():
    cfg = SelfplayConfig.model_validate(VALID_SELFPLAY)
    assert cfg.n_workers == 1
    assert cfg.mcts.n_simulations == 50
    assert cfg.playout_cap.fast_sims == 50


@pytest.mark.parametrize("field", SELFPLAY_FIELDS)
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


def test_selfplay_has_no_pydantic_level_default():
    for name, field in SelfplayConfig.model_fields.items():
        assert field.is_required(), f"SelfplayConfig.{name} has a code-side default"


def test_selfplay_has_no_legal_move_radius_field():
    # DESIGN_P2.md §5 shape (ii): the registry alone is the radius authority.
    assert "legal_move_radius" not in SelfplayConfig.model_fields
    assert "legal_move_radius_schedule" not in SelfplayConfig.model_fields


def test_dirichlet_epsilon_field_name_equals_config_key():
    # The historical key/field mismatch (config key `mcts.epsilon`, field
    # `dirichlet_epsilon`) is retired: the schema field IS `dirichlet_epsilon`, matching
    # the config key one-to-one — a wrong-spelling silent no-op is now structurally
    # impossible (replaces the retired test_pool_hparams_arms.py wrong-spelling test).
    payload = _selfplay()
    payload["mcts"] = dict(VALID_MCTS, dirichlet_epsilon=0.9)
    cfg = SelfplayConfig.model_validate(payload)
    assert cfg.mcts.dirichlet_epsilon == 0.9


# ── InferenceConfig ───────────────────────────────────────────────────────────────────
def test_inference_valid_payload_constructs_clean():
    cfg = InferenceConfig.model_validate(VALID_INFERENCE)
    assert cfg.inference_batch_size == 64


#: ARCH-SCOPED blocks are omittable at the SECTION level by design (R322(d)); their
#: required-ness depends on `identity.representation`, which `InferenceConfig` cannot see, so
#: it is a `RunConfig` fact and is executed by the conformance suite's T9 section. Derived.
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
    """R1, with the one exception DERIVED from `ARCH_SCOPED_KEYS` rather than typed (R322(d)).

    The `= None` on an arch-scoped block is not a fallback: `RunConfig` refuses a graph config
    that omits it and any other config that carries it, so no key's absence silently yields a
    value. A hand-added default on any other field is still a red.
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
