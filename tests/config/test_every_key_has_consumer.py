"""O15 — every-key-has-consumer bijection (LAW-08).

SCHEMA KEYS ONLY (not registered encodings — that is gate-8's disjoint concern). Enumerate
leaf key-paths of RunConfig.model_fields and assert the set equals an explicit 8-entry
CONSUMER_REGISTRY. Enumeration STOPS at a list[SubModel] field (NIT-3): RadiusStage.step /
.radius are covered transitively by resolve_radius_from_schedule, not separate registry rows.
"""
from pydantic import BaseModel

from mantis.config.schema import RunConfig

# Each value names a REAL WP8 reader; every "emit" reader genuinely appears in the O6 payload.
CONSUMER_REGISTRY = {
    "schema_version": "loader version-pin + emit",
    "run_id": "mint header stamp + emit",
    "seed": "emit source-tag (acting RNG consumer lands WP-train)",
    "identity.encoding": "reconcile_encoding + encoding regime-parity (O11) + emit",
    "identity.representation": "resolve_amp_dtype + IdentityConfig runtime consistency guard (F1) + O11 + emit",
    "eval.random_model_sims": "resolve_eval_model_sims + sims regime-parity (O9) + emit",
    "eval.sealbot_model_sims": "resolve_eval_model_sims + sims regime-parity (O9) + emit",
    "selfplay.legal_move_radius_schedule": "resolve_radius_from_schedule + radius parity (O12) + emit",
}


def _leaf_paths(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Leaf key-paths of a StrictModel; recurse into nested models but STOP at any field
    whose annotation is not itself a BaseModel subclass (NIT-3 — list[SubModel] is one leaf)."""
    out: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            out.extend(_leaf_paths(ann, path))
        else:
            out.append(path)
    return out


def test_schema_leaves_equal_consumer_registry_bijection():
    leaves = set(_leaf_paths(RunConfig))
    registered = set(CONSUMER_REGISTRY)
    assert leaves == registered, (
        f"schema-only (unregistered): {leaves - registered}; "
        f"registry-only (no schema field): {registered - leaves}"
    )


def test_registry_has_exactly_eight_entries():
    assert len(CONSUMER_REGISTRY) == 8
    assert len(_leaf_paths(RunConfig)) == 8


def test_enumeration_stops_at_radius_stage():
    # NIT-3: the schedule is ONE leaf; RadiusStage.step/.radius are not enumerated.
    leaves = _leaf_paths(RunConfig)
    assert "selfplay.legal_move_radius_schedule" in leaves
    assert not any(p.startswith("selfplay.legal_move_radius_schedule.") for p in leaves)


def test_bijection_bites_on_a_real_schema_mutation():
    # F5: a genuine schema mutation (a throwaway subclass adding a leaf field with no registry
    # entry) must make the bijection fail — enumeration picks up the new leaf, not set-algebra.
    class _MutatedRunConfig(RunConfig):
        phantom_leaf: int  # new schema field, no CONSUMER_REGISTRY entry

    leaves = set(_leaf_paths(_MutatedRunConfig))
    assert "phantom_leaf" in leaves
    assert leaves != set(CONSUMER_REGISTRY), "bijection must break when a new leaf is unregistered"
