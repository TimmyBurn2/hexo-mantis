"""Schema contract tests (run-config-schema v1): unknown/missing keys hard-fail,
representation is the closed set {grid, graph}, and O16 schema round-trip + every-config-
validates + no-code-side-defaults + strictness across the full model tree.

>300 justify (R8): the O16 census and the payload builders are ONE unit and cannot be split.
The builders below are the only schema-complete payload in this file, and every rejection test
(`test_missing_*`, `test_nested_unknown_key_rejected`) mutates one of them by a single key --
so a reviewer judging whether a rejection test still probes what it claims has to read the
builder on the same screen. The census at the foot then closes the loop from the other side:
the rejection tests prove hand-picked keys are enforced, the census proves the SAME property
holds for every field of every model the walk reaches, and its mutation self-test is written
against the same imports. Splitting them lets a builder drift out from under the rejection
tests, or a census exemption appear with its counter-example in another file.
"""
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from mantis.config.loader import discover_configs, load_config
from mantis.config.schema import (
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
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# WP11-A schema extension: eval.gate/eval.ladder are now required fields (design §c.1).
# This mirrors tests/eval/test_ladder_config_schema.py's fixture verbatim (kept in one
# place there; duplicated here only because this file predates the extension and must
# still construct a schema-complete payload for its own, unrelated assertions).
_LADDER_RUNGS = [
    {"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
]


def _valid_eval_block() -> dict:
    return {
        "random_model_sims": 96, "sealbot_model_sims": 128, "kraken_model_sims": 128,
        "strix_model_sims": 128, "random_floor_games": 0, "worker_device": "cuda",
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


#: WPMINT Phase K-A stage 0: the complete `train:` payload, DERIVED from a MINTED config
#: rather than restated — eleven files carried a hand-written copy, so a new `train.*` key
#: cost eleven edits. `dev_example.yaml`'s resolved block was measured byte-identical to the
#: census it replaces, so the swap is zero-behavior-change.
_MINTED_TRAIN: dict = load_config(REPO_ROOT / "configs" / "dev_example.yaml").train.model_dump()


def _valid_train_block() -> dict:
    return dict(_MINTED_TRAIN)


def _valid_selfplay_block() -> dict:
    return {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "inference_pool_size": None, "completed_q_values": False, "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_mcts": False, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0, "rotation_enabled": True,
        "forced_win_policy_enabled": False, "forced_win_policy_depth": 2,
        "forced_win_policy_weight": 1.0, "solver_enabled": False, "solver_depth": 16,
        "solver_node_budget": 50_000, "solver_neighbor_dist": 2, "solver_visit_weight": 0.3,
        "seed_fraction": 0.0, "seed_corpus_path": None, "log_investigation_metrics": True,
        "instrumentation_enabled": False,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "zoi_enabled": False, "zoi_lookback": 16, "zoi_margin": 5,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }


def _valid_inference_block() -> dict:
    return {
        "inference_batch_size": 64, "inference_max_wait_ms": 10, "trace_inference": True,
        "compile_inference": False, "compile_inference_mode": "default",
        "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
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
        "run_id": "unit_test",
        "seed": 1,
        "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
        "eval": _valid_eval_block(),
        "train": _valid_train_block(),
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


def test_representation_grid_now_accepted():
    # "grid" is in the closed set — accepted for a GRID encoding (v6w25).
    payload = _valid_payload()
    payload["identity"] = {"encoding": "v6w25", "representation": "grid"}
    cfg = RunConfig.model_validate(payload)
    assert cfg.identity.representation == "grid"


# ── F1 — representation↔encoding consistency is a RUNTIME guard (not test-only) ──
def test_f1_graph_encoding_declared_grid_rejected_at_validate():
    # gnn_axis_v1 is a GRAPH encoding; declaring representation=grid must RAISE (LAW-06 pin guard).
    payload = _valid_payload()
    payload["identity"] = {"encoding": "gnn_axis_v1", "representation": "grid"}
    with pytest.raises(ValidationError, match="disagrees with the registry"):
        RunConfig.model_validate(payload)


def test_f1_grid_encoding_declared_graph_rejected_at_validate():
    payload = _valid_payload()
    payload["identity"] = {"encoding": "v6w25", "representation": "graph"}
    with pytest.raises(ValidationError, match="disagrees with the registry"):
        RunConfig.model_validate(payload)


def test_f1_unknown_encoding_rejected_at_validate():
    payload = _valid_payload()
    payload["identity"] = {"encoding": "no_such_encoding", "representation": "graph"}
    with pytest.raises(ValidationError, match="no_such_encoding"):
        RunConfig.model_validate(payload)


# ── O16 — schema round-trip + every-config-validates + no code-side defaults ──
def test_o16_every_committed_config_validates():
    # ADJ-13 F-1 corrective pass (recheck R-5): the ONE discovery authority, not a
    # sixth flat glob. A flat `*.yaml` census is blind to `configs/prod/run6.yaml`,
    # which gate 7 and gate 12 both now make legal.
    configs = discover_configs(REPO_ROOT / "configs")
    assert configs, "no committed configs found (gate 7 must never be vacuous)"
    for cfg_path in configs:
        load_config(cfg_path)  # raises on any failure


def test_o16_schema_round_trip():
    cfg = load_config(REPO_ROOT / "configs" / "run5.yaml")
    again = RunConfig.model_validate(cfg.model_dump())
    assert again == cfg


# ── O16 census — DERIVED from RunConfig.model_fields, never enumerated ──
#
# The predecessor walked a HAND-WRITTEN tuple: RunConfig, IdentityConfig, EvalConfig,
# SelfplayConfig, TrainConfig, MonitorSchemaConfig, DrainCapsConfig, DiskGuardConfig. Every
# other block in the schema — `GateConfig`, `LadderConfig`, `LadderRung`, `MctsConfig`,
# `PlayoutCapConfig`, `InferenceConfig`, `DrawRateAbortConfig`, `ReplayCapacityStage`,
# `MicrobatchCapsConfig` — was added AFTER that tuple was written and never joined it, because
# nothing made adding a model to the schema also add it here. An enumerated census is a census
# of what someone remembered; R1's "NO code-side defaults" is a claim about the schema itself.


def _nested_block(ann: Any) -> type[BaseModel] | None:
    """The single config BLOCK an annotation names, seen THROUGH `Optional` / `Block | None`.

    Traversal semantics are the ones `tools/ci_gates/contract_doc_gate.py::_leaf_paths` and
    its two `tests/config/test_every_key_has_consumer*.py` twins already hold, so the four
    walkers agree about what a nested block is: an OPTIONAL block is descended into (WPMINT
    DR-6 / R93 — `Block | None` is the house arming idiom and treating it as one opaque leaf
    hid an entirely unconsumed key), and a union naming more than one BaseModel is NOT, since
    it has no single block to hand out. R5/LAW-17 bar importing any of the three, which is why
    this is a fourth copy; it is self-defending in the same way theirs are — a walker that
    diverged would discover a different model set and red the reachability test below, which
    derives its expected set from the package rather than from this walk.
    """
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    if get_origin(ann) in (Union, UnionType):
        blocks = [a for a in get_args(ann) if isinstance(a, type) and issubclass(a, BaseModel)]
        if len(blocks) == 1:
            return blocks[0]
    return None


def _element_block(ann: Any) -> type[BaseModel] | None:
    """The BLOCK inside a `list[SubModel]`, which `_leaf_paths` deliberately does NOT descend.

    This is the one place the census parts company with the leaf-path walk, and the divergence
    is required rather than incidental. NIT-3 keeps `list[SubModel]` as ONE leaf because a list
    element has no single key-path to name — that is a statement about KEY PATHS. It says
    nothing about the element MODEL, which is still a schema block whose fields must be
    required and whose unknown keys must be refused. `eval.ladder.rungs` (`LadderRung`) and
    `train.replay_capacity_schedule` (`ReplayCapacityStage`) are reachable ONLY through this
    arm — measured: without it the census misses both, and both are real config blocks that
    ship in every minted config.
    """
    if get_origin(ann) is list:
        blocks = [a for a in get_args(ann) if isinstance(a, type) and issubclass(a, BaseModel)]
        if len(blocks) == 1:
            return blocks[0]
    return None


def _schema_census(root: type[BaseModel]) -> dict[type[BaseModel], str]:
    """Every model reachable from `root`, mapped to the key path it was first reached by."""
    found: dict[type[BaseModel], str] = {root: ""}

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            path = f"{prefix}{name}"
            block = _nested_block(field.annotation)
            if block is None:
                block = _element_block(field.annotation)
                path = f"{path}[]"
            if block is not None and block not in found:
                found[block] = path
                walk(block, f"{path}.")

    walk(root, "")
    return found


SCHEMA_CENSUS = _schema_census(RunConfig)
#: Non-vacuity floor, measured at adoption. A derived census that quietly discovers ZERO models
#: is the vacuous-pass class gate 15 was hardened against in 35f0bfe: it reports green having
#: asserted nothing. This is a FLOOR and may only ratchet up; the exact-set claim is
#: `test_o16_census_reaches_every_schema_block`, which is derived on both sides.
MIN_SCHEMA_MODELS = 17


def test_o16_census_is_not_vacuous():
    assert len(SCHEMA_CENSUS) >= MIN_SCHEMA_MODELS, (
        f"the derived census found {len(SCHEMA_CENSUS)} model(s), floor {MIN_SCHEMA_MODELS}. "
        "A walk that discovers nothing asserts nothing and must never read as green."
    )


def test_o16_census_covers_every_model_the_hand_written_tuple_named():
    """The predecessor's eight, as a regression pin: derived must be a SUPERSET of enumerated.

    A walker refactor that lost `monitor.disk_guard` would still clear the floor above; this is
    what makes that specific loss fatal, and it is why these eight names stay imported.
    """
    legacy = (RunConfig, IdentityConfig, EvalConfig, SelfplayConfig, TrainConfig,
              MonitorSchemaConfig, DrainCapsConfig, DiskGuardConfig)
    missing = [m.__name__ for m in legacy if m not in SCHEMA_CENSUS]
    assert not missing, f"the derived census lost model(s) the enumerated one had: {missing}"


def test_o16_census_reaches_every_schema_block():
    """Exactness, derived on BOTH sides: reachable-from-RunConfig == defined-in-the-package.

    A `StrictModel` subclass that no field points at is dead schema — nothing can supply it, so
    nothing consumes it (LAW-08's shape) — and a block that exists but is unreachable is also
    the one thing a reachability census can never check. Either direction failing is a real
    finding, so both are reported.
    """
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
    # WPMAIN / §3.1 MISS-9: `DiskGuardConfig` is in the census. Without it R122's granted
    # family would be the one schema block in the tree with no no-pydantic-default census,
    # and a re-added `interval_sec: float = 60.0` is invisible to every liveness drive (they
    # supply a value on each path). This test is ALSO the structural holder of R120's "the
    # code-side default True dies" and R123(c): a re-added SCHEMA default on `eval_enabled`
    # or `run_id` reds it, which a signature census cannot see (wrong instrument).
    # `TrainConfig` is reached too, so `train.device` is covered — and now with zero edits for
    # every FUTURE block as well, which is the point of deriving the set instead of listing it.
    for model, path in SCHEMA_CENSUS.items():
        for name, field in model.model_fields.items():
            key = f"{path}.{name}" if path else name
            assert field.is_required(), (
                f"{model.__name__}.{name} (config key `{key}`) has a code-side default; "
                "R1 puts a default in the schema field or nowhere"
            )


def test_o16_every_schema_block_is_strict():
    """R1's "unknown key = error" is carried by `StrictModel`, so it must hold block by block.

    `extra="forbid"` is not inherited by a block merely because its PARENT forbids extras —
    pydantic resolves `model_config` per model — so one block declared `BaseModel` would accept
    any typo'd key inside it while every rejection test above stayed green.
    """
    for model, path in SCHEMA_CENSUS.items():
        assert issubclass(model, StrictModel), (
            f"{model.__name__} (config key `{path or '<root>'}`) is not a StrictModel"
        )
        assert model.model_config.get("extra") == "forbid", (
            f"{model.__name__} (config key `{path or '<root>'}`) overrides extra= to "
            f"{model.model_config.get('extra')!r}; an unknown key inside it would be accepted"
        )


# ── the census's own mutation self-test (LAW-07): it must BITE ──
#
# Built from throwaway models that are never registered anywhere, so the real schema is not
# mutated and no other test can see them. Each plants exactly one of the two defects the census
# exists to catch, in the two shapes that are easy to miss: nested one level down, and reachable
# only through the `list[SubModel]` arm.


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
    # forbid extras, so the `extra=` check alone passes it, and only the `issubclass` limb
    # catches it. A block that reimplements strictness by hand is outside the one base every
    # section is supposed to share, and the next `model_config` edit to it is unguarded.
    assert _PlantedLoose.model_config.get("extra") == "forbid"


def test_the_non_vacuity_floor_would_fire_on_a_walk_that_found_nothing():
    """`walked nothing, found nothing` must never read as green (gate 15's own lesson)."""
    class _Leaf(StrictModel):
        scalar: int

    assert len(_schema_census(_Leaf)) < MIN_SCHEMA_MODELS
