"""Schema contract tests: unknown/missing keys hard-fail, representation is the closed set
{grid, graph}, and the census covers round-trip, every-config-validates, no code-side defaults
and strictness across the full model tree.

>300 justify (R8): the census and the payload builders are ONE unit. Every rejection test mutates
a builder by a single key, so a reviewer judging whether one still probes what it claims must read
the builder on the same screen; the census then proves the SAME property for every field of every
model the walk reaches. Split, a builder drifts out from under the rejection tests.
"""
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from mantis.config.loader import discover_configs, load_config
from mantis.config.schema import (
    ARCH_SCOPED_KEYS,
    OPERATIONAL_DEFAULT_KEYS,
    SCHEMA_VERSION,
    DiskGuardConfig,
    DrainCapsConfig,
    EvalConfig,
    IdentityConfig,
    MonitorSchemaConfig,
    RunConfig,
    SelfplayConfig,
    StrictModel,
    TrainConfig,
    nested_block,
)
from mantis.eval.rounds import EVAL_CONCURRENCY_ROW
from mantis.model import ARCH_KIND_ROW
from mantis.train.warmstart import WARM_START_ROW

REPO_ROOT = Path(__file__).resolve().parents[2]

# eval.gate/eval.ladder are required fields; this mirrors the ladder-schema fixture verbatim,
# duplicated here only because this file predates the extension and must still construct a
# schema-complete payload for its own, unrelated assertions.
_LADDER_RUNGS = [
    {"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
]


def _valid_eval_block() -> dict:
    return {
        "random_model_sims": 96, "sealbot_model_sims": 128, "random_floor_games": 0, "worker_device": "cuda",
        "round_timeout_sec": 3600.0, "worker_kill_grace_sec": 10.0,
        "ply_cap_adjudication": None, "strength_floor": None,
        "gate": {
            "stride": 1, "screen_games": 80, "confirm_games": 128, "promotion_winrate": 0.55,
            "screen_confirm_lo": 0.44, "deploy_sims": 150, "opening_book": "book_v1_s20260625_p4",
            "bootstrap_resamples": 1000, "min_distinct_per_pair": 10, "seed_base": 20260625,
        },
        "ladder": {
            "rungs": [dict(r) for r in _LADDER_RUNGS], "round_games": 64,
            "min_games_per_active_rung": 4, "graduation_wr_lower_ci": 0.75,
            "graduation_consec_rounds": 3, "activation_wr_lower_ci": 0.65,
            "calibration_every_k_rounds": 4, "calibration_games": 8,
            "bootstrap_resamples": 1000, "bootstrap_ci_level": 0.95,
            "bt_prior_games": 1.0, "bootstrap_seed": 1234,
        },
    }


#: The complete `train:` payload, DERIVED from a MINTED config rather than restated: eleven files
#: carried a hand-written copy, so a new `train.*` key cost eleven edits. The resolved block was
#: measured byte-identical to the census it replaces.
_MINTED_TRAIN: dict = load_config(REPO_ROOT / "configs" / "dev_example.yaml").train.model_dump()


def _valid_train_block() -> dict:
    return dict(_MINTED_TRAIN)


def _valid_selfplay_block() -> dict:
    return {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }


def _valid_inference_block() -> dict:
    return {
        "inference_batch_size": 64, "inference_max_wait_ms": 10,
        # `inference.fused_graph_caps` is a REQUIRED block, and the pair here is the template's
        # NON-BINDING-BY-CONSTRUCTION value, so nothing in this file exercises a split.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }


def _valid_monitor_block() -> dict:
    return {
        # R242 (ADJ-D12): the ARMING cadence, schema-only and required.
        "gate_interval": 1000,
        "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5, "alert_grad_norm_max": 10.0,
        "alert_loss_increase_window": 3, "wr_hard_abort_enabled": False,
        "wr_rolling_consecutive_evals": 2, "wr_rolling_threshold": 0.10,
        "wr_rolling_min_step": 20000, "wr_collapse_from_peak_ratio": 0.5,
        "wr_collapse_min_step": 25000, "wr_collapse_consecutive_evals": 3,
        "wr_early_death_threshold": 0.05, "wr_early_death_min_step": 15000,
        "axis_warn": 0.45, "axis_alert": 0.50,
        "heartbeat_deadline_train_step_sec": 1800.0,
        "heartbeat_deadline_inference_dispatch_sec": 1800.0,
        "heartbeat_deadline_selfplay_drain_sec": 1800.0,
        "heartbeat_deadline_eval_round_sec": 1800.0,
        "heartbeat_poll_interval_sec": 5.0, "heartbeat_file_interval_sec": 15.0,
        "heartbeat_close_out_deadline_sec": 14400.0, "heartbeat_fire_effect_timeout_sec": 30.0,
        "supervisor_stale_after_sec": 900.0, "supervisor_poll_interval_sec": 30.0,
        "supervisor_kill_grace_sec": 30.0, "supervisor_max_relaunches": 5,
        "actor_lag_threshold_steps": 100, "actor_lag_abort_enabled": False,
        "drain": {
            "final_eval_drain_timeout_sec": 900.0, "eval_final_drain_safety_factor": 3.0,
            "eval_final_drain_hard_cap_sec": 14400.0, "terminal_eval_hard_cap_sec": 14400.0,
        },
        "disk_guard": {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0},
    }


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_enabled": True,
        # A REQUIRED top-level leaf whose `null` is the placeholder: refused at boot on a cuda
        # process, and valued only by the re-calibration sitting.
        "allocator_posture": None,
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": _valid_eval_block(),
        "train": _valid_train_block(),
        "search": {"kind": "puct"},
        "selfplay": _valid_selfplay_block(),
        "inference": _valid_inference_block(),
        "monitor": _valid_monitor_block(),
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
    del payload["selfplay"]["n_workers"]
    with pytest.raises(ValidationError, match="n_workers"):
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


def test_f1_graph_encoding_declared_grid_rejected_at_validate():
    # gnn_axis_v1 is a GRAPH encoding; declaring representation=grid must RAISE (LAW-06 pin guard).
    payload = _valid_payload()
    payload["identity"] = {"encoding": "gnn_axis_v1", "representation": "grid"}
    with pytest.raises(ValidationError, match="disagrees with the registry"):
        RunConfig.model_validate(payload)


def test_f1_unknown_encoding_rejected_at_validate():
    payload = _valid_payload()
    payload["identity"] = {"encoding": "no_such_encoding", "representation": "graph"}
    with pytest.raises(ValidationError, match="no_such_encoding"):
        RunConfig.model_validate(payload)


def test_o16_every_committed_config_validates():
    # The ONE discovery authority, not a sixth flat glob: a flat `*.yaml` census is blind to
    # `configs/prod/run6.yaml`, which both gates now make legal.
    configs = discover_configs(REPO_ROOT / "configs")
    assert configs, "no committed configs found (gate 7 must never be vacuous)"
    for cfg_path in configs:
        load_config(cfg_path)  # raises on any failure


def test_o16_schema_round_trip():
    cfg = load_config(REPO_ROOT / "configs" / "run6.yaml")
    again = RunConfig.model_validate(cfg.model_dump())
    assert again == cfg


# The census is DERIVED from `RunConfig.model_fields`, never enumerated. The predecessor walked a
# HAND-WRITTEN tuple, and every schema block added after that tuple was written never joined it,
# because nothing made adding a model to the schema also add it here.


#: THE PREDICATE IS THE AUTHORITY'S: `mantis.config.schema.nested_block` is the one predicate,
#: in `src/` where a test may import it, replacing a fourth local copy. The census parts company
#: with the leaf-path walk deliberately — keeping `list[SubModel]` as ONE leaf is a statement
#: about KEY PATHS, and `eval.ladder.rungs` and `train.replay_capacity_schedule` are reachable
#: ONLY through the container arm, so the census asks the SAME predicate its second question.
#: the census asks the SAME predicate its second question instead of holding a second walker.


def _schema_census(root: type[BaseModel]) -> dict[type[BaseModel], str]:
    """Every model reachable from `root`, mapped to the key path it was first reached by."""
    found: dict[type[BaseModel], str] = {root: ""}

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            path = f"{prefix}{name}"
            block = nested_block(field.annotation)
            if block is None:
                block = nested_block(field.annotation, descend_containers=True)
                path = f"{path}[]"
            if block is not None and block not in found:
                found[block] = path
                walk(block, f"{path}.")

    walk(root, "")
    return found


SCHEMA_CENSUS = _schema_census(RunConfig)
#: Non-vacuity floor, measured at adoption: a derived census that quietly discovers ZERO models
#: reports green having asserted nothing. A FLOOR that may only ratchet up; the exact-set claim is
#: `test_o16_census_reaches_every_schema_block`, derived on both sides.
MIN_SCHEMA_MODELS = 17


def test_o16_census_is_not_vacuous():
    assert len(SCHEMA_CENSUS) >= MIN_SCHEMA_MODELS, (
        f"the derived census found {len(SCHEMA_CENSUS)} model(s), floor {MIN_SCHEMA_MODELS}. "
        "A walk that discovers nothing asserts nothing and must never read as green."
    )


def test_o16_census_covers_every_model_the_hand_written_tuple_named():
    """The predecessor's eight, as a regression pin: derived must be a SUPERSET of enumerated."""
    legacy = (RunConfig, IdentityConfig, EvalConfig, SelfplayConfig, TrainConfig,
              MonitorSchemaConfig, DrainCapsConfig, DiskGuardConfig)
    missing = [m.__name__ for m in legacy if m not in SCHEMA_CENSUS]
    assert not missing, f"the derived census lost model(s) the enumerated one had: {missing}"


def test_o16_census_reaches_every_schema_block():
    """Exactness, derived on BOTH sides: reachable-from-RunConfig == defined-in-the-package."""
    import pkgutil
    from importlib import import_module

    import mantis.config.schema as pkg

    defined: set[type[BaseModel]] = set()
    for info in pkgutil.iter_modules(pkg.__path__):
        module = import_module(f"{pkg.__name__}.{info.name}")
        for obj in vars(module).values():
            if (isinstance(obj, type) and issubclass(obj, StrictModel) and obj is not StrictModel
                    and obj.__module__.startswith(pkg.__name__)):
                defined.add(obj)

    unreachable = sorted(m.__name__ for m in defined - set(SCHEMA_CENSUS))
    unknown = sorted(m.__name__ for m in set(SCHEMA_CENSUS) - defined)
    assert not unreachable, (
        f"schema block(s) defined but not reachable from RunConfig: {unreachable}. Either wire "
        "them in or delete them — an unreachable block is a key no config can ever set."
    )
    assert not unknown, f"census reached model(s) not defined in the schema package: {unknown}"


def test_o16_all_fields_required_no_code_side_defaults():
    # `DiskGuardConfig` is in the census, or its family would be the one schema block with no
    # no-pydantic-default census. This test is ALSO the structural holder of "the code-side default
    # True dies": a re-added SCHEMA default on `eval_enabled` or `run_id` reds it.
    # THE ONE EXEMPT CLASS IS DERIVED: an ARCH-SCOPED block carries `= None` so another
    # representation may OMIT it, and that `None` is not a fallback, because the presence validator
    # refuses both omission on the owning arch and presence on any other. Read off
    # `ARCH_SCOPED_KEYS` and asserted BOTH ways.
    exempt = {f"{key.section}.{key.field}" for key in ARCH_SCOPED_KEYS}
    assert exempt, "no key is arch-scoped, so this exemption is unused and should go"
    # THE SECOND, THIRD AND FOURTH EXEMPT ROWS, each enumerated by name so a fifth is still a red:
    # `identity.arch_kind`, whose absent row resolves to the representation's INCUMBENT kind;
    # `identity.warm_start`, a BLOCK whose exemption is on the PARENT only, since `checkpoint` and
    # `net_hash` are REQUIRED inside it; and `eval.concurrency`, whose default is not a placeholder
    # but the BEHAVIOUR ITSELF — `1` is the serial loop that ran before the parameter existed.
    exempt |= {ARCH_KIND_ROW, WARM_START_ROW, EVAL_CONCURRENCY_ROW}
    # THE FIFTH CLASS IS A REGISTRY, not a row: an OPERATIONAL CONSTANT carries a schema default and
    # leaves the YAML, and `OPERATIONAL_DEFAULT_KEYS` is the ONE authority. A default on anything
    # NOT in a registry is still a red, and a registered key that is still required is a stale
    # declaration. Every arming key stays required, deliberately.
    operational = {key for key, _grounds in OPERATIONAL_DEFAULT_KEYS}
    assert operational, "the operational registry is empty; this exemption should go"
    assert not (operational & exempt), (
        "a key is in BOTH registries — arch-scoped and operational are different rules "
        f"(a scoped block is REFUSED off its arch; a default is inherited): {operational & exempt}"
    )
    assert all(grounds.strip() for _key, grounds in OPERATIONAL_DEFAULT_KEYS), (
        "an operational exemption with no written grounds is a default nobody can justify later"
    )
    exempt |= operational
    seen: set[str] = set()
    for model, path in SCHEMA_CENSUS.items():
        for name, field in model.model_fields.items():
            key = f"{path}.{name}" if path else name
            if key in exempt:
                seen.add(key)
                assert not field.is_required(), (
                    f"{model.__name__}.{name} (config key `{key}`) is registered as "
                    "arch-scoped or operational, so it must be omittable — a required "
                    "arch-scoped block forces every arch to mint it, and a required "
                    "operational key is a declaration nobody honoured"
                )
                continue
            assert field.is_required(), (
                f"{model.__name__}.{name} (config key `{key}`) has a code-side default and is "
                "in neither registry; R1 puts a default in the schema field or nowhere, and a "
                "registry row with written grounds is what says the key earned one"
            )
    assert seen == exempt, (
        f"a registry names {sorted(exempt - seen)} which the census never reached — an "
        "exemption for a key nobody walks is an exemption nobody can see go stale"
    )


def test_o16_every_schema_block_is_strict():
    """R1's "unknown key = error" is carried by `StrictModel`, so it must hold block by block."""
    for model, path in SCHEMA_CENSUS.items():
        assert issubclass(model, StrictModel), (
            f"{model.__name__} (config key `{path or '<root>'}`) is not a StrictModel"
        )
        assert model.model_config.get("extra") == "forbid", (
            f"{model.__name__} (config key `{path or '<root>'}`) overrides extra= to "
            f"{model.model_config.get('extra')!r}; an unknown key inside it would be accepted"
        )


# The census's own mutation self-test: it must BITE. Built from throwaway models that are never
# registered anywhere, each planting exactly one of the two defects the census exists to catch, in
# the two shapes that are easy to miss — nested one level down, and reachable only through the
# `list[SubModel]` arm.


class _PlantedDefault(StrictModel):
    knob: int = 3  # the defect: a code-side default (R1)


class _PlantedLoose(BaseModel):
    model_config = ConfigDict(extra="forbid")
    knob: int


class _PlantedPermissive(BaseModel):
    model_config = ConfigDict(extra="allow")  # the defect: unknown keys accepted
    knob: int


class _PlantedRun(StrictModel):
    nested_with_a_default: _PlantedDefault
    nested_not_strict: _PlantedPermissive
    optional_block: _PlantedLoose | None
    element_only: list[_PlantedDefault]
    scalar: int


def test_the_census_walk_reaches_optional_and_list_element_blocks():
    census = _schema_census(_PlantedRun)
    assert census == {
        _PlantedRun: "",
        _PlantedDefault: "nested_with_a_default",
        _PlantedPermissive: "nested_not_strict",
        _PlantedLoose: "optional_block",
    }, f"the walk missed a shape: {sorted(m.__name__ for m in census)}"


def test_the_required_arm_bites_on_a_planted_default():
    census = _schema_census(_PlantedRun)
    offenders = [
        f"{m.__name__}.{n}" for m in census for n, f in m.model_fields.items()
        if not f.is_required()
    ]
    assert offenders == ["_PlantedDefault.knob"], (
        f"the no-code-side-defaults arm did not bite exactly once: {offenders}"
    )


def test_the_strictness_arm_bites_on_a_planted_permissive_block():
    census = _schema_census(_PlantedRun)
    offenders = sorted(
        m.__name__ for m in census
        if not issubclass(m, StrictModel) or m.model_config.get("extra") != "forbid"
    )
    assert offenders == ["_PlantedLoose", "_PlantedPermissive"], (
        f"the strictness arm did not bite on both shapes: {offenders}"
    )
    # `_PlantedLoose` is the second shape and the reason the assertion is two-limbed: it DOES
    # forbid extras, so the `extra=` check alone passes it and only the `issubclass` limb catches
    # it. A block that reimplements strictness by hand is outside the one base every section shares.
    assert _PlantedLoose.model_config.get("extra") == "forbid"


def test_the_non_vacuity_floor_would_fire_on_a_walk_that_found_nothing():
    """`walked nothing, found nothing` must never read as green (gate 15's own lesson)."""
    class _Leaf(StrictModel):
        scalar: int

    assert len(_schema_census(_Leaf)) < MIN_SCHEMA_MODELS
