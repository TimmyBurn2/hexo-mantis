"""O15 — every-key-has-consumer bijection (LAW-08).

SCHEMA KEYS ONLY (not registered encodings — that is gate-8's disjoint concern). Enumerate
leaf key-paths of RunConfig.model_fields and assert the set equals an explicit 8-entry
CONSUMER_REGISTRY. Enumeration STOPS at a list[SubModel] field (NIT-3): RadiusStage.step /
.radius are covered transitively by resolve_radius_from_schedule, not separate registry rows.
"""
from pydantic import BaseModel

from mantis.config.schema import RunConfig

# Each value names a REAL WP8/WP11-A reader; every "emit" reader genuinely appears in the
# O6 payload. WP11-A extends this registry with every eval.gate/eval.ladder leaf (design
# §c.1) — none of these are yet threaded into config/emit.py's resolved payload (that
# would break the pre-existing, non-oracle O6 9-knob pin in
# tests/config/test_resolved_config_emit.py, which this WP does not touch); their live
# consumer is the eval/bots/arena machinery cited below, not the emit payload.
CONSUMER_REGISTRY = {
    "schema_version": "loader version-pin + emit",
    "run_id": "mint header stamp + emit",
    "seed": "emit source-tag (acting RNG consumer lands WP-train)",
    "identity.encoding": "reconcile_encoding + encoding regime-parity (O11) + emit",
    "identity.representation": "resolve_amp_dtype + IdentityConfig runtime consistency guard (F1) + O11 + emit",
    "eval.random_model_sims": "resolve_eval_model_sims (random floor) + sims regime-parity (O9) + emit",
    "eval.sealbot_model_sims": "resolve_eval_model_sims (sealbot rungs) + sims regime-parity (O9) + emit",
    "eval.kraken_model_sims": "resolve_eval_model_sims (kraken rungs)",
    "eval.strix_model_sims": "resolve_eval_model_sims (strix rungs)",
    "eval.random_floor_games": "worker.py random-floor block game count",
    "eval.worker_device": "build_eval_pipeline child-process device",
    "eval.round_timeout_sec": "pipeline.py mid-round subprocess join bound",
    "eval.worker_kill_grace_sec": "pipeline.py terminate->kill grace",
    "eval.gate.stride": "pipeline.py promotion-capable round stride",
    "eval.gate.screen_games": "worker.py gate screen block",
    "eval.gate.confirm_games": "worker.py gate confirm block",
    "eval.gate.promotion_winrate": "aggregate.py gate truth table",
    "eval.gate.screen_confirm_lo": "aggregate.py escalation test",
    "eval.gate.deploy_sims": "arena/deploy_head.py sims",
    "eval.gate.opening_book": "arena/books.py gate openings",
    "eval.gate.bootstrap_resamples": "aggregate.py bootstrap CI",
    "eval.gate.min_distinct_per_pair": "aggregate.py low-power guard",
    "eval.gate.seed_base": "aggregate.py bootstrap seed + worker.py opening seeds",
    "eval.ladder.rungs": "ladder.py LadderState rungs",
    "eval.ladder.round_games": "ladder.py allocate_games budget",
    "eval.ladder.min_games_per_active_rung": "ladder.py allocate_games floor",
    "eval.ladder.graduation_wr_lower_ci": "ladder.py graduation transition",
    "eval.ladder.graduation_consec_rounds": "ladder.py graduation streak",
    "eval.ladder.activation_wr_lower_ci": "ladder.py activation transition",
    "eval.ladder.calibration_every_k_rounds": "ladder.py calibration cadence",
    "eval.ladder.calibration_games": "ladder.py calibration allocation",
    "eval.ladder.bootstrap_resamples": (
        "pipeline.py RoundSpec.ladder_bootstrap_resamples -> worker.py aggregate_rung (M-2)"
    ),
    "eval.ladder.bootstrap_ci_level": (
        "pipeline.py RoundSpec.ladder_bootstrap_ci_level -> worker.py aggregate_rung (M-2)"
    ),
    "eval.ladder.bt_prior_games": "bt.py fit_bt prior",
    "eval.ladder.bootstrap_seed": (
        "pipeline.py RoundSpec.ladder_bootstrap_seed -> worker.py aggregate_rung (M-2)"
    ),
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
    # WP11-A extends the O15 registry 8 -> 36 leaves (design §c.1: eval.gate.* + eval.
    # ladder.* + 6 new eval.* scalars). The name is historical (O15's original 8-entry
    # WP8 count); the bijection test above is the live invariant this file exists to hold.
    assert len(CONSUMER_REGISTRY) == 36
    assert len(_leaf_paths(RunConfig)) == 36


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
