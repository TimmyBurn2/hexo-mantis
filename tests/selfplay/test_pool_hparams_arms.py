"""Suite D (hparams + wire arms) — D-06 … D-14, D-18, and the `mcts.epsilon` pin.

>300 justify: ONE surface. Every row here binds `SelfPlayHParams.from_config` or the
`build_runner_config` wire it feeds, and they share the recording proxy over the Rust
config (which has no ctor getters, so the recorded kwarg dict is the only observable) plus
the base config dicts the old-side capture used. Splitting the hard-error arms from the
wire arms would duplicate both.

WPSC Phase 2 SC-A2 (R-SELFPLAYCONFIG-SCHEMA closure) reshape: `SelfPlayHParams.from_config`
no longer reads a flat legacy dict with top-level-namespace fallback — every config literal
below is nested (`selfplay: {..., mcts: {...}, playout_cap: {...}}`, `train: {...}`) matching
the schema shape. The four old hard-error arms (D-06/D-07/D-08 + the temperature-resolver
key/spelling cases) are now `PlayoutCapConfig`/`MctsConfig` schema bounds/validators — their
coverage moved to `tests/config/test_mcts_playout_cap_schema.py` and
`tests/config/test_selfplay_playout_cap_mutual_exclusion.py` (DESIGN_P2.md §12); this file
keeps only the arms that exercise the ACTUAL `from_config`/wire behavior. The
`mcts.epsilon`-vs-`dirichlet_epsilon` spelling trap (`test_mcts_epsilon_key_wins`) and the
`max_game_moves`/`max_moves_per_game` dual-alias chain (`test_max_moves_alias_chain`) are
DELETED outright: the schema field IS the config key now, so there is no wrong spelling or
alias fallback left to test (replaces `test_selfplay_schema.py::
test_dirichlet_epsilon_field_name_equals_config_key`).
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from mantis import _engine
from mantis.encoding import lookup
from mantis.selfplay import hparams as hparams_mod
from mantis.selfplay.hparams import (
    PoolDims,
    SelfPlayHParams,
    _load_seed_corpus,
    build_runner_config,
    resolve_pool_encoding,
)

BASE_SELFPLAY: dict[str, Any] = {
    "n_workers": 7, "leaf_batch_size": 12, "max_game_moves": 200,
    "inference_pool_size": 1536, "completed_q_values": True, "c_visit": 40.0, "c_scale": 2.0,
    "gumbel_mcts": True, "gumbel_m": 24, "gumbel_explore_moves": 14,
    "results_queue_cap": 5000, "random_opening_plies": 3, "rotation_enabled": False,
    "forced_win_policy_enabled": True, "forced_win_policy_depth": 4,
    "forced_win_policy_weight": 0.75, "solver_enabled": True, "solver_depth": 20,
    "solver_node_budget": 77000, "solver_neighbor_dist": 3, "solver_visit_weight": 0.45,
    "seed_fraction": 0.0, "seed_corpus_path": None,
    "log_investigation_metrics": False, "instrumentation_enabled": True,
}
BASE_MCTS: dict[str, Any] = {
    "n_simulations": 111, "c_puct": 1.75, "fpu_reduction": 0.4, "quiescence_enabled": False,
    "quiescence_blend_2": 0.55, "dirichlet_alpha": 0.25, "dirichlet_epsilon": 0.3,
    "dirichlet_enabled": False,
}
BASE_PLAYOUT_CAP: dict[str, Any] = {
    "fast_sims": 40, "fast_prob": 0.0, "standard_sims": 160, "full_search_prob": 0.0,
    "n_sims_quick": 0, "n_sims_full": 0, "zoi_enabled": True, "zoi_lookback": 20,
    "zoi_margin": 7, "temperature_threshold_compound_moves": 0, "temp_min": 0.5,
}
BASE_TRAIN: dict[str, Any] = {
    "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0, "fp16": True, "amp_dtype": "fp16",
    "lr_schedule": "cosine", "total_steps": 1_000_000, "scheduler_t_max": None,
    "eta_min": 5e-4, "min_lr": None, "checkpoint_interval": 0, "completed_q_values": False,
    "value_target": "pure_outcome_z", "policy_target": "raw_visit_distribution",
    "draw_reward": -0.4, "ply_cap_value": -0.7, "policy_prune_frac": 0.0,
    "entropy_reg_weight": 0.0, "aux_opp_reply_weight": 0.0, "uncertainty_weight": 0.0,
    "ownership_weight": 0.0, "threat_weight": 0.0, "aux_chain_weight": 0.0,
    "ply_index_weight": 0.0, "threat_pos_weight": 1.0,
}


def cfg(
    *, encoding: str = "v6", selfplay: dict | None = None, mcts: dict | None = None,
    playout_cap: dict | None = None, train: dict | None = None,
) -> dict[str, Any]:
    """A nested, schema-shaped config: BASE_* plus per-section overrides. `encoding` stays a
    top-level flat key — `resolve_pool_encoding`/`resolve_from_config` read it independently
    of `identity.encoding` (a separate, pre-existing pool.py/hparams.py convention untouched
    by SC-A2)."""
    sp = dict(BASE_SELFPLAY)
    sp.update(selfplay or {})
    sp["mcts"] = dict(BASE_MCTS, **(mcts or {}))
    sp["playout_cap"] = dict(BASE_PLAYOUT_CAP, **(playout_cap or {}))
    return {
        "encoding": encoding,
        "selfplay": sp,
        "train": dict(BASE_TRAIN, **(train or {})),
    }


class _RecordingRunnerConfig:
    """Proxy over the REAL Rust config, recording ctor kwargs + post-ctor attribute sets.

    Same instrument as the ⊕ D-15 golden uses, and for the same reason: the Rust config
    exposes getters for its post-ctor attributes only, so the ctor-kwarg dict is the ONLY
    observable of the config→runner wire. Constructing the real object underneath keeps
    Rust-side validation live.
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

    def build(config: dict[str, Any], *, seed_prefixes=None) -> _RecordingRunnerConfig:
        hp = SelfPlayHParams.from_config(config)
        enc = resolve_pool_encoding(config, arch=None)
        build_runner_config(hp, spec_dims=enc, encoding_name=enc.encoding_name,
                            seed_prefixes=seed_prefixes)
        return built[-1]

    return build


# ═══ D-09 — effective-sims resolution + the one hard error with no schema equivalent ═════
def test_effective_sims_zero_is_a_hard_error() -> None:
    """D-09 — PASS iff a config resolving to zero effective per-move sims raises, naming
    both escape routes. This check spans `mcts.n_simulations` AND `playout_cap.*`, so it has
    no single-model schema equivalent and stays a `from_config` runtime check."""
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
    """D-09 (resolution arm) — PASS iff the effective per-move sim count equals the flat
    `mcts.n_simulations` with no cap, the full-search ceiling `n_sims_full` under a
    move-level cap."""
    hp = SelfPlayHParams.from_config(cfg(playout_cap=playout_cap))
    assert hp.effective_sims_per_move == expected


# ═══ D-10 — the temperature schedule reaches the hparams AND the wire ═════════════════════
def test_playout_cap_temperature_threshold_reaches_hparams_and_wire(assemble) -> None:
    """D-10 (wire arm; RENAMED per R38 — ADJ-02/R38 disposition) — PASS iff a schedule-ON
    `playout_cap` arrives at BOTH the hparams field and the runner ctor kwarg. The schema
    field IS the config key now (`temperature_threshold_compound_moves`), so there is no
    silently-disabled-schedule trap left to test — this asserts the wire, not a spelling."""
    config = cfg(playout_cap={"temperature_threshold_compound_moves": 12, "temp_min": 0.35})
    hp = SelfPlayHParams.from_config(config)
    assert (hp.temp_threshold_compound_moves, hp.temp_min) == (12, 0.35)

    recorded = assemble(config)
    assert recorded.recorded_kwargs["temp_threshold_compound_moves"] == 12
    assert recorded.recorded_kwargs["temp_min"] == 0.35


def test_dirichlet_alpha_field_name_equals_its_key(assemble) -> None:
    """PASS iff `mcts.dirichlet_alpha` reaches `dirichlet_alpha` on both the hparams and the
    runner kwarg — a basic wiring-through check (the spelling-mismatch control this used to
    pair with, `test_mcts_epsilon_key_wins`, is deleted: the schema retires the mismatch)."""
    config = cfg(mcts={"dirichlet_alpha": 0.6})
    assert SelfPlayHParams.from_config(config).dirichlet_alpha == 0.6
    assert assemble(config).recorded_kwargs["dirichlet_alpha"] == 0.6


def test_dirichlet_epsilon_reaches_hparams_and_wire(assemble) -> None:
    """PASS iff `mcts.dirichlet_epsilon` reaches `dirichlet_epsilon` on both the hparams and
    the runner kwarg (replaces the retired `mcts.epsilon`-spelling pin — the schema field IS
    the config key now, `test_selfplay_schema.py::
    test_dirichlet_epsilon_field_name_equals_config_key` pins the schema side)."""
    config = cfg(mcts={"dirichlet_epsilon": 0.9})
    assert SelfPlayHParams.from_config(config).dirichlet_epsilon == 0.9
    assert assemble(config).recorded_kwargs["dirichlet_epsilon"] == 0.9


# ═══ D-11 — the ply-cap value chain and its wire site ════════════════════════════════════
@pytest.mark.parametrize(
    "train_over,expected_draw,expected_ply",
    [
        ({"draw_reward": -0.5, "ply_cap_value": -0.9}, -0.5, -0.9),
        ({"draw_reward": -0.3, "ply_cap_value": -0.3}, -0.3, -0.3),
    ],
    ids=["explicit_split", "explicit_equal"],
)
def test_ply_cap_value_wire(assemble, train_over, expected_draw, expected_ply) -> None:
    """D-11 — PASS iff `train.draw_reward`/`train.ply_cap_value` land on the runner's
    `draw_reward`/`ply_cap_value` kwargs (cross-section read, DESIGN_P2.md §2 note — no
    fallback-to-sibling once both are required schema fields)."""
    config = cfg(train=train_over)

    hp = SelfPlayHParams.from_config(config)
    assert hp.draw_value == expected_draw
    assert hp.ply_cap_value == expected_ply

    recorded = assemble(config)
    assert recorded.recorded_kwargs["draw_reward"] == expected_draw
    assert recorded.recorded_kwargs["ply_cap_value"] == expected_ply


# ═══ D-12 — seed corpus: five captured cases + the assembly wire ═════════════════
def test_seed_corpus_none_path_frac_zero() -> None:
    """D-12(a) — PASS iff no path and no seeding yields `None` (feature simply off)."""
    assert _load_seed_corpus(None, 0.0) is None


def test_seed_corpus_none_path_frac_positive() -> None:
    """D-12(b) — PASS iff asking for seeding with no corpus raises. FAIL = a seeded run
    that silently seeds nothing, which looks identical to a healthy run in every metric."""
    with pytest.raises(ValueError) as exc:
        _load_seed_corpus(None, 0.5)
    assert "seed_fraction > 0 requires" in str(exc.value)


def test_seed_corpus_malformed_raises(tmp_path) -> None:
    """D-12(c) — PASS iff a JSONL line without `seed_moves` raises a ValueError that names
    the path and the expected shape, chaining the underlying error."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"not_seed_moves": []}\n')
    with pytest.raises(ValueError) as exc:
        _load_seed_corpus(str(bad), 0.5)
    assert "is malformed" in str(exc.value)
    assert "seed_moves" in str(exc.value)


def test_seed_corpus_zero_prefixes_raises(tmp_path) -> None:
    """D-12(d) — PASS iff a syntactically fine but EMPTY corpus raises when seeding is on.
    FAIL = seeding is a silent no-op, the exact failure this check exists for."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(ValueError) as exc:
        _load_seed_corpus(str(empty), 0.5)
    assert "yielded ZERO prefixes" in str(exc.value)


@pytest.mark.parametrize("fraction", [0.5, 0.0], ids=["active", "inert_but_validated"])
def test_seed_corpus_valid_parse(tmp_path, fraction: float) -> None:
    """D-12(e) — PASS iff a valid corpus parses to the captured prefixes, and does so even
    at `seed_fraction == 0` (parsed for validation, never fired). FAIL = a corpus that
    only gets validated once seeding is switched on, i.e. the error surfaces in the run
    that matters instead of the dry run before it."""
    good = tmp_path / "good.jsonl"
    good.write_text('{"seed_moves": [[0, 0], [1, -1]]}\n{"seed_moves": [[2, 3]]}\n')
    assert _load_seed_corpus(str(good), fraction) == [[(0, 0), (1, -1)], [(2, 3)]]


def test_seed_prefixes_land_on_the_runner_config(assemble, tmp_path) -> None:
    """D-12 (assembly-wire arm) — PASS iff parsed prefixes handed to `build_runner_config`
    land on the runner config's seed-corpus attribute. The ⊕ D-15 golden always passes
    `seed_prefixes=None`, so this wire is otherwise completely untested. FAIL = a parsed
    trap corpus that never reaches Rust — seeding configured, validated, and then
    dropped."""
    good = tmp_path / "good.jsonl"
    good.write_text('{"seed_moves": [[0, 0], [1, -1]]}\n{"seed_moves": [[2, 3]]}\n')
    prefixes = _load_seed_corpus(str(good), 0.5)

    recorded = assemble(cfg(), seed_prefixes=prefixes)
    assert recorded.recorded_attrs["seed_corpus"] == prefixes
    assert recorded.real.seed_corpus == [[(0, 0), (1, -1)], [(2, 3)]]


def test_absent_seed_prefixes_leave_the_attribute_unset(assemble) -> None:
    """D-12 (negative arm) — PASS iff `seed_prefixes=None` never touches `seed_corpus`, so
    the Rust default (no seeding) stands. FAIL = an explicit empty corpus is written,
    which is a different thing from "no corpus"."""
    recorded = assemble(cfg(), seed_prefixes=None)
    assert "seed_corpus" not in recorded.recorded_attrs
    assert recorded.real.seed_corpus is None


# ═══ D-13 — inference_pool_size threading ════════════════════════════════════════
@pytest.mark.parametrize(
    "selfplay_over,expected",
    [({"inference_pool_size": None}, None), ({"inference_pool_size": 999}, 999)],
    ids=["absent", "int"],
)
def test_inference_pool_size_threading(assemble, selfplay_over, expected) -> None:
    """D-13 — PASS iff the opt-in pool size threads through as declared: `None` stays
    `None` (the engine's own prefill sizing), an int passes. FAIL = `None` turns into a
    hard-coded size that pins the working set for every encoding."""
    config = cfg(selfplay=selfplay_over)

    hp = SelfPlayHParams.from_config(config)
    assert hp.inference_pool_size == expected
    assert assemble(config).recorded_kwargs["inference_pool_size"] == expected


# ═══ D-14 — `gumbel_mcts` re-reads the LIVE config ═══════════════════════════════
def test_gumbel_mcts_property_reads_live_config() -> None:
    """D-14 — PASS iff the `gumbel_mcts` property reflects a config mutated AFTER
    construction, in both directions, while the frozen ctor-time hparams do not move.

    Deliberate asymmetry: nearly every knob is resolved once at construction, but this one
    re-reads because the event emitter uses it to decide whether the PUCT-only diagnostics
    are meaningful, and that decision must follow the live config. Exercised through a
    bare object carrying only `config` so no runner is needed. `WorkerPool.gumbel_mcts`
    itself is untouched by SC-A2 (its own flat-fallback read is a separate, pre-existing
    mechanism DESIGN_P2.md does not scope in). FAIL = the regime guard freezes at
    construction and the emitter reports PUCT diagnostics for a Gumbel run."""
    from mantis.selfplay.pool import WorkerPool

    holder = object.__new__(WorkerPool)
    holder.config = {"selfplay": {"gumbel_mcts": False}}
    assert holder.gumbel_mcts is False

    holder.config["selfplay"]["gumbel_mcts"] = True
    assert holder.gumbel_mcts is True, "the property must re-read the live config"

    holder.config = {}
    assert holder.gumbel_mcts is False, "absent key ⇒ False, with the namespace fallback"

    hp = SelfPlayHParams.from_config(cfg(selfplay={"gumbel_mcts": True}))
    assert hp.gumbel_mcts is True, "the frozen ctor-time snapshot still records the knob"


# ═══ D-18 — the derived dense dims, and the FFI agreement ════════════════════════
@pytest.mark.parametrize(
    "encoding,expected",
    [("v6w25", PoolDims(5000, 3750, 626)),
     ("v6", PoolDims(2888, 2166, 362)),
     ("gnn_axis_v1", PoolDims(0, 0, 362))],
)
def test_pool_dims_derivation_golden(assemble, encoding: str, expected: PoolDims) -> None:
    """D-18 — PASS iff the derived dims equal the captured `pool_derived` block for all
    three configs, including the degenerate graph case (0/0/policy).

    These dims size the reshape the drain applies to every dense row before it reaches the
    replay buffer. FAIL = a wrong `feat_len` reshapes the batch into a differently-shaped
    tensor of the same total size, which is silent corruption rather than an exception."""
    config = cfg(encoding=encoding, playout_cap={"fast_sims": 100})
    hp = SelfPlayHParams.from_config(config)
    enc = resolve_pool_encoding(config, arch=None)
    _, dims = build_runner_config(hp, spec_dims=enc, encoding_name=enc.encoding_name,
                                  seed_prefixes=None)
    assert dims == expected


@pytest.mark.parametrize("encoding", ["v6", "v6w25"])
def test_pool_dims_agree_with_the_rust_derivation(encoding: str) -> None:
    """D-18 (FFI-agreement arm) — PASS iff the Python-side dims equal the lengths the Rust
    batcher derives from the SAME spec.

    The two derivations live on opposite sides of the FFI: Python computes
    `n_kept_planes · trunk²` and reads `policy_logit_count`; Rust derives both from the
    encoding name. Nothing else in the tree checks that they agree, and a disagreement is
    a buffer/rows shape mismatch that surfaces as garbage rather than an error. The graph
    arm is excluded on purpose — its dense dims are degenerate zeros, so the comparison
    would be vacuous."""
    from mantis._engine import InferenceBatcher

    config = cfg(encoding=encoding)
    hp = SelfPlayHParams.from_config(config)
    enc = resolve_pool_encoding(config, arch=None)
    _, dims = build_runner_config(hp, spec_dims=enc, encoding_name=enc.encoding_name,
                                  seed_prefixes=None)

    batcher = InferenceBatcher(encoding_spec=lookup(encoding))
    assert dims.feat_len == int(batcher.feature_len_py)
    assert dims.pol_len == int(batcher.policy_len_py)


def test_killed_knobs_are_never_read(assemble) -> None:
    """DV-6 pin — PASS iff a config carrying WP-KILLed self-play knobs assembles cleanly
    and neither name reaches the Rust config. `from_config` only reads its own known keys
    off the nested schema sections, so an extra key alongside them is simply never
    consulted (schema-level rejection of unknown keys is a DIFFERENT, already-covered
    concern — `test_selfplay_schema.py::test_selfplay_extra_key_rejected`)."""
    config = cfg(selfplay={"legal_move_radius_jitter": True})
    recorded = assemble(config)
    assert "legal_move_radius_jitter" not in recorded.recorded_kwargs
    assert "interior_selector" not in recorded.recorded_attrs


def test_hparams_round_trip_is_json_stable() -> None:
    """Guard arm — PASS iff every resolved hparam is a plain Python scalar (or None), so
    the whole knob set can be recorded into a run manifest. FAIL = a numpy scalar or a
    config sub-dict leaked into the frozen snapshot and the manifest write dies mid-run."""
    hp = SelfPlayHParams.from_config(cfg())
    payload = {f: getattr(hp, f) for f in hp.__dataclass_fields__}
    json.dumps(payload)  # raises TypeError on any non-JSON scalar
    for name, value in payload.items():
        assert value is None or isinstance(value, (int, float, str, bool)), (
            f"{name} resolved to {type(value).__name__}, not a plain scalar"
        )
