"""Suite D — the self-play hparams and the runner wire they feed.

>300 justify: ONE surface. Every row binds `SelfPlayHParams.from_config` or the
`build_runner_config` wire it feeds, sharing the recording proxy over the Rust config (which has
no ctor getters, so the recorded kwarg dict is the only observable) and the base config dicts;
splitting the hard-error arms from the wire arms would duplicate both. The old hard-error arms
are now schema bounds covered in tests/config/, and the spelling traps are gone because the
schema field IS the config key.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from mantis import _engine
from mantis.encoding import lookup
from mantis.config.resolve.search import MissingSearchKindError
from mantis.selfplay import hparams as hparams_mod
from mantis.selfplay.pool import WorkerPool
from mantis.selfplay.hparams import (
    PoolDims,
    SelfPlayHParams,
    build_runner_config,
    resolve_pool_encoding,
)

BASE_SELFPLAY: dict[str, Any] = {
    "n_workers": 7, "leaf_batch_size": 12, "max_game_moves": 200,
    "c_visit": 40.0, "c_scale": 2.0,
    "gumbel_m": 24, "gumbel_explore_moves": 14,
    "results_queue_cap": 5000, "random_opening_plies": 3,
    "log_investigation_metrics": False,
}
BASE_MCTS: dict[str, Any] = {
    "n_simulations": 111, "c_puct": 1.75, "fpu_reduction": 0.4, "quiescence_enabled": False,
    "quiescence_blend_2": 0.55, "dirichlet_alpha": 0.25, "dirichlet_epsilon": 0.3,
    "dirichlet_enabled": False,
}
BASE_PLAYOUT_CAP: dict[str, Any] = {
    "fast_sims": 40, "fast_prob": 0.0, "standard_sims": 160, "full_search_prob": 0.0,
    "n_sims_quick": 0, "n_sims_full": 0,
    "temperature_threshold_compound_moves": 0, "temp_min": 0.5,
}
# NOT a `train:` schema payload — `cfg()` builds the LEGACY flat hparams dict the pool reads, so
# this block is deliberately INCOMPLETE and never reaches `RunConfig.model_validate`; a new
# `train.*` schema key costs this file nothing. `draw_reward: -0.4` and `ply_cap_value: -0.7` are
# DISTINGUISHABLE from the minted defaults so the oracles can prove they reached the Rust runner.
BASE_TRAIN: dict[str, Any] = {
    "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0,
    "lr_schedule": "cosine", "total_steps": 1_000_000, "scheduler_t_max": None,
    "eta_min": 5e-4, "checkpoint_interval": 0,
    "value_target": "pure_outcome_z", "policy_target": "raw_visit_distribution",
    "draw_reward": -0.4, "ply_cap_value": -0.7,
    "fast_policy_weight": 0.0,
}


def cfg(
    *, encoding: str = "gnn_axis_v1", selfplay: dict | None = None, mcts: dict | None = None,
    playout_cap: dict | None = None, train: dict | None = None,
    search: dict | None = None,
) -> dict[str, Any]:
    """A nested, schema-shaped config: BASE_* plus per-section overrides. `encoding` stays a
    top-level flat key, read independently of `identity.encoding`."""
    sp = dict(BASE_SELFPLAY)
    sp.update(selfplay or {})
    sp["mcts"] = dict(BASE_MCTS, **(mcts or {}))
    sp["playout_cap"] = dict(BASE_PLAYOUT_CAP, **(playout_cap or {}))
    return {
        "encoding": encoding,
        "search": dict({"kind": "puct"}, **(search or {})),
        "selfplay": sp,
        "train": dict(BASE_TRAIN, **(train or {})),
    }


class _RecordingRunnerConfig:
    """Proxy over the REAL Rust config, recording ctor kwargs + post-ctor attribute sets: the
    Rust config exposes getters for post-ctor attributes only, so the ctor-kwarg dict is the
    ONLY observable of the config->runner wire.
    """

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "recorded_kwargs", dict(kwargs))
        object.__setattr__(self, "recorded_attrs", {})
        object.__setattr__(self, "real", _engine.SelfPlayRunnerConfig(**kwargs))

    def __setattr__(self, name: str, value: Any) -> None:
        self.recorded_attrs[name] = value
        setattr(self.real, name, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "real"), name)


@pytest.fixture
def assemble(monkeypatch):
    """Factory → the `_RecordingRunnerConfig` produced by assembling one config dict."""
    built: list[_RecordingRunnerConfig] = []

    class _Factory(_RecordingRunnerConfig):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            built.append(self)

    monkeypatch.setattr(hparams_mod, "SelfPlayRunnerConfig", _Factory)

    def build(config: dict[str, Any]) -> _RecordingRunnerConfig:
        hp = SelfPlayHParams.from_config(config)
        enc = resolve_pool_encoding(config, arch=None)
        build_runner_config(hp, spec_dims=enc, encoding_name=enc.encoding_name)
        return built[-1]

    return build


# effective-sims resolution + the one hard error with no schema equivalent
def test_effective_sims_zero_is_a_hard_error() -> None:
    """A config resolving to zero effective per-move sims raises, naming both escape routes; the
    check spans two sections, so it has no single-model schema equivalent."""
    with pytest.raises(ValueError) as exc:
        SelfPlayHParams.from_config(cfg(mcts={"n_simulations": 0}))
    message = str(exc.value)
    assert "could not resolve effective per-move sim count" in message
    assert "mcts.n_simulations > 0" in message and "playout_cap.n_sims_full > 0" in message


@pytest.mark.parametrize(
    "playout_cap,expected",
    [
        ({}, 111),
        ({"full_search_prob": 0.3, "n_sims_quick": 40, "n_sims_full": 250}, 250),
    ],
    ids=["flat_regime", "move_level_cap_regime"],
)
def test_effective_sims_per_move_resolution(playout_cap: dict, expected: int) -> None:
    """Effective sims are flat `mcts.n_simulations`, or `n_sims_full` under a move-level cap."""
    hp = SelfPlayHParams.from_config(cfg(playout_cap=playout_cap))
    assert hp.effective_sims_per_move == expected


# the temperature schedule reaches the hparams AND the wire
def test_playout_cap_temperature_threshold_reaches_hparams_and_wire(assemble) -> None:
    """A schedule-ON `playout_cap` reaches BOTH the hparams field and the runner ctor kwarg."""
    config = cfg(playout_cap={"temperature_threshold_compound_moves": 12, "temp_min": 0.35})
    hp = SelfPlayHParams.from_config(config)
    assert (hp.temp_threshold_compound_moves, hp.temp_min) == (12, 0.35)

    recorded = assemble(config)
    assert recorded.recorded_kwargs["temp_threshold_compound_moves"] == 12
    assert recorded.recorded_kwargs["temp_min"] == 0.35


def test_dirichlet_alpha_field_name_equals_its_key(assemble) -> None:
    """`mcts.dirichlet_alpha` reaches the hparams field and the runner kwarg."""
    config = cfg(mcts={"dirichlet_alpha": 0.6})
    assert SelfPlayHParams.from_config(config).dirichlet_alpha == 0.6
    assert assemble(config).recorded_kwargs["dirichlet_alpha"] == 0.6


def test_dirichlet_epsilon_reaches_hparams_and_wire(assemble) -> None:
    """`mcts.dirichlet_epsilon` reaches the hparams field and the runner kwarg."""
    config = cfg(mcts={"dirichlet_epsilon": 0.9})
    assert SelfPlayHParams.from_config(config).dirichlet_epsilon == 0.9
    assert assemble(config).recorded_kwargs["dirichlet_epsilon"] == 0.9


# the ply-cap value chain and its wire site
@pytest.mark.parametrize(
    "train_over,expected_draw,expected_ply",
    [
        ({"draw_reward": -0.5, "ply_cap_value": -0.9}, -0.5, -0.9),
        ({"draw_reward": -0.3, "ply_cap_value": -0.3}, -0.3, -0.3),
    ],
    ids=["explicit_split", "explicit_equal"],
)
def test_ply_cap_value_wire(assemble, train_over, expected_draw, expected_ply) -> None:
    """`train.draw_reward`/`train.ply_cap_value` land on the runner kwargs of the same name."""
    config = cfg(train=train_over)

    hp = SelfPlayHParams.from_config(config)
    assert hp.draw_value == expected_draw
    assert hp.ply_cap_value == expected_ply

    recorded = assemble(config)
    assert recorded.recorded_kwargs["draw_reward"] == expected_draw
    assert recorded.recorded_kwargs["ply_cap_value"] == expected_ply


# `search_kind` re-reads the LIVE config
def test_search_kind_property_reads_live_config() -> None:
    """`search_kind` reflects a config mutated AFTER construction and REFUSES an absent one."""
    holder = type("H", (), {"search_kind": WorkerPool.search_kind})()
    holder.config = {"search": {"kind": "puct"}}
    assert holder.search_kind == "puct"

    holder.config["search"]["kind"] = "gumbel"
    assert holder.search_kind == "gumbel", "the property must re-read the live config"

    # NO FALLBACK: a pool that cannot say which search it ran must raise rather than answer
    # "puct", because the emitter gates PUCT-only diagnostics on this. The refusal is the
    # RESOLVER's, so the pool cannot grow a fallback of its own.
    holder.config = {}
    with pytest.raises(MissingSearchKindError):
        _ = holder.search_kind

    hp = SelfPlayHParams.from_config(cfg(search={"kind": "gumbel"}))
    assert hp.search_kind == "gumbel", "the frozen ctor-time snapshot still records the kind"



# the derived dense dims, and the FFI agreement
@pytest.mark.parametrize(
    "encoding,expected",
    [("gnn_axis_v1", PoolDims(0, 0, 362)),
     ("gnn_axis_r8", PoolDims(0, 0, 362))],
)
def test_pool_dims_derivation_golden(assemble, encoding: str, expected: PoolDims) -> None:
    """The derived dims stay ZERO on a graph encoding with the policy length off the spec; a
    non-zero feat_len would mean a dense geometry no wire carries is being re-derived."""
    config = cfg(encoding=encoding, playout_cap={"fast_sims": 100})
    hp = SelfPlayHParams.from_config(config)
    enc = resolve_pool_encoding(config, arch=None)
    _, dims = build_runner_config(hp, spec_dims=enc, encoding_name=enc.encoding_name)
    assert dims == expected


def test_killed_knobs_are_never_read(assemble) -> None:
    """Killed self-play knobs assemble cleanly and neither name reaches the Rust config."""
    config = cfg(selfplay={"legal_move_radius_jitter": True})
    recorded = assemble(config)
    assert "legal_move_radius_jitter" not in recorded.recorded_kwargs
    assert "interior_selector" not in recorded.recorded_attrs


def test_hparams_round_trip_is_json_stable() -> None:
    """Every resolved hparam is a plain scalar or None, so a run manifest can record them."""
    hp = SelfPlayHParams.from_config(cfg())
    payload = {f: getattr(hp, f) for f in hp.__dataclass_fields__}
    json.dumps(payload)  # raises TypeError on any non-JSON scalar
    for name, value in payload.items():
        assert value is None or isinstance(value, (int, float, str, bool)), (
            f"{name} resolved to {type(value).__name__}, not a plain scalar"
        )
