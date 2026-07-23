"""Suite D (hparams + wire arms) — D-06 … D-14, D-18, and the `mcts.epsilon` pin.

>300 justify: ONE surface. Every row here binds `SelfPlayHParams.from_config` or the
`build_runner_config` wire it feeds, and they share the recording proxy over the Rust
config (which has no ctor getters, so the recorded kwarg dict is the only observable) plus
the base config dicts the old-side capture used. Splitting the hard-error arms from the
wire arms would duplicate both.

IMPL-written (non-⊕) per DESIGN §b; D-15 is the ⊕ assembly golden in `test_pool_hparams.py`
and this file carries the arms ORACLE_NOTES §6 left to IMPL. Every expected value is the
dispatcher's old-side capture (`wp/WPSP/oldside/data/c3a_c3d_report.json`, sections
`C3d_hard_error_and_wire_arms`, `C3d_seed_corpus_cases`, `C3d_temperature_resolver`) or the
PREREG §3 D-18 literals; the config dicts are the capture script's own inputs.

The sharpest row in this file is `test_mcts_epsilon_key_wins`. `mcts.epsilon` is read into
a field spelled `dirichlet_epsilon` — the config KEY and the field name differ. Reading the
field spelling returns None on every config, the 0.25 code-side default fires, and an
operator's Dirichlet exploration setting is discarded with no error. Both directions are
asserted here, because only the pair distinguishes "reads the right key" from "happens to
agree with the default".
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
    _resolve_playout_cap_temperature,
    build_runner_config,
    resolve_pool_encoding,
)

# The capture's base config: `mcts.interior_selector` is present in the captured dicts but
# is a WP6-KILLed knob on this side and is neither read nor set (DV-6).
BASE: dict[str, Any] = {"encoding": "v6", "mcts": {"interior_selector": "puct"}}


def cfg(**over: Any) -> dict[str, Any]:
    """A capture-shaped config: BASE plus the overrides a single arm varies."""
    out = dict(BASE)
    out.update(over)
    return out


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


# ═══ D-06 … D-09 — the four hard errors, on the captured inputs ═══════════════════
def test_fast_sims_missing_is_a_hard_error() -> None:
    """D-06 — PASS iff an empty `playout_cap` raises `ValueError` naming the missing key
    and refusing a default. FAIL = a silent `fast_sims` default, which sets the game-level
    playout cap for a whole run to a number nobody chose."""
    with pytest.raises(ValueError) as exc:
        SelfPlayHParams.from_config(cfg(playout_cap={}))
    assert "playout_cap.fast_sims must be set" in str(exc.value)
    assert "no silent defaults" in str(exc.value)


def test_fast_prob_and_full_search_prob_are_mutually_exclusive() -> None:
    """D-07 — PASS iff setting BOTH caps raises, quoting both values. The move-level cap
    overrides the game-level one inside the worker loop, so running both silently ignores
    the latter. FAIL = an operator's game-level cap is discarded with no error."""
    with pytest.raises(ValueError) as exc:
        SelfPlayHParams.from_config(cfg(playout_cap={
            "fast_sims": 40, "fast_prob": 0.5, "full_search_prob": 0.5,
            "n_sims_quick": 40, "n_sims_full": 200}))
    message = str(exc.value)
    assert "mutually exclusive" in message
    assert "fast_prob=0.5" in message and "full_search_prob=0.5" in message


@pytest.mark.parametrize(
    "quick,full", [(0, 200), (40, 0)],
    ids=["missing_quick", "missing_full"],
)
def test_full_search_prob_requires_both_sim_counts(quick: int, full: int) -> None:
    """D-08 — PASS iff `full_search_prob > 0` without BOTH sim counts raises, quoting all
    three numbers. FAIL = a move-level cap regime configured with a zero sim count, which
    the runner would interpret as "search nothing" for that arm."""
    with pytest.raises(ValueError) as exc:
        SelfPlayHParams.from_config(cfg(playout_cap={
            "fast_sims": 40, "full_search_prob": 0.5,
            "n_sims_quick": quick, "n_sims_full": full}))
    message = str(exc.value)
    assert "requires n_sims_quick > 0 AND n_sims_full > 0" in message
    assert f"n_sims_quick={quick}" in message and f"n_sims_full={full}" in message


def test_effective_sims_zero_is_a_hard_error() -> None:
    """D-09 — PASS iff a config resolving to zero effective per-move sims raises, naming
    both escape routes. FAIL = a sims/sec bill of zero for a whole run, i.e. a throughput
    metric that reads 0 while self-play is healthy."""
    with pytest.raises(ValueError) as exc:
        SelfPlayHParams.from_config(cfg(
            mcts={"interior_selector": "puct", "n_simulations": 0},
            playout_cap={"fast_sims": 40}))
    message = str(exc.value)
    assert "could not resolve effective per-move sim count" in message
    assert "mcts.n_simulations > 0" in message and "playout_cap.n_sims_full > 0" in message


@pytest.mark.parametrize(
    "playout_cap,expected",
    [
        ({"fast_sims": 40}, 50),
        ({"fast_sims": 40, "full_search_prob": 0.3,
          "n_sims_quick": 40, "n_sims_full": 250}, 250),
    ],
    ids=["flat_regime", "move_level_cap_regime"],
)
def test_effective_sims_per_move_resolution(playout_cap: dict, expected: int) -> None:
    """D-09 (resolution arm) — PASS iff the effective per-move sim count equals the
    capture: the flat `mcts.n_simulations` with no cap, the full-search ceiling
    `n_sims_full` under a move-level cap. FAIL = the sims/sec bill regresses toward the
    falsified per-GAME undercount."""
    hp = SelfPlayHParams.from_config(cfg(playout_cap=playout_cap))
    assert hp.effective_sims_per_move == expected


# ═══ D-10 — the temperature resolver + the key it actually reads ══════════════════
@pytest.mark.parametrize(
    "playout_cap,expected",
    [
        ({}, (0, 0.5)),
        ({"temperature_threshold_compound_moves": 15, "temp_min": 0.05}, (15, 0.05)),
        ({"temperature_threshold_compound_moves": 0, "temp_min": 0.5}, (0, 0.5)),
        ({"temperature_threshold_compound_moves": 9}, (9, 0.5)),
        ({"temp_min": 0.25}, (0, 0.25)),
        ({"temperature_threshold_compound_moves": None}, (0, 0.5)),
        ({"temp_threshold_compound_moves": 15}, (0, 0.5)),
    ],
    ids=["absent", "explicit_15_005", "explicit_0_05", "threshold_only",
         "temp_min_only", "explicit_null_threshold", "WRONG_SPELLING_ignored"],
)
def test_temperature_resolver_cases(playout_cap: dict, expected: tuple) -> None:
    """D-10 — PASS iff all seven captured resolver cases reproduce exactly, including the
    cosine-OFF fallback `(0, 0.5)` for absent or explicitly-null keys.

    The last case is the trap made visible: `temp_threshold_compound_moves` is the RUNNER
    KWARG spelling, not the config key, and a config carrying it is silently ignored. That
    is old behaviour and it is pinned, not fixed — the fix belongs to the schema extension
    that will retire the code-side default. FAIL on the second case = the schedule is
    silently disabled; FAIL on the first = the toxic legacy 15/0.05 fallback is back."""
    assert _resolve_playout_cap_temperature(playout_cap) == expected


def test_temperature_schedule_reaches_the_hparams_and_the_wire(assemble) -> None:
    """D-10 (wire arm) — PASS iff a schedule-ON `playout_cap` arrives at BOTH the hparams
    field and the runner ctor kwarg. FAIL = the schedule is resolved and then dropped
    between the two, which no resolver-level test can see."""
    config = cfg(playout_cap={"fast_sims": 40,
                              "temperature_threshold_compound_moves": 12,
                              "temp_min": 0.35})
    hp = SelfPlayHParams.from_config(config)
    assert (hp.temp_threshold_compound_moves, hp.temp_min) == (12, 0.35)

    recorded = assemble(config)
    assert recorded.recorded_kwargs["temp_threshold_compound_moves"] == 12
    assert recorded.recorded_kwargs["temp_min"] == 0.35


# ═══ the OWED pin — `mcts.epsilon` is the key; `dirichlet_epsilon` is the field ═══
def test_mcts_epsilon_key_wins(assemble) -> None:
    """CORRECTION-3b pin — PASS iff `mcts.epsilon` reaches `dirichlet_epsilon` on both the
    hparams and the runner kwarg, AND the field spelling in the config is IGNORED.

    Both directions are required. Asserting only that `mcts.epsilon = 0.3` arrives would
    also pass if the port read some third key that happened to be absent — no, it would
    not, but asserting only that the FIELD spelling is ignored would pass on a port that
    reads nothing at all and always ships 0.25. The pair distinguishes "reads the right
    key" from "agrees with the default by accident".

    FAIL = Dirichlet root exploration silently runs at the code-side 0.25 for an entire
    run while the operator's config says otherwise. That knob controls how much noise is
    injected at the search root; substituting it changes what the run explores."""
    with_key = cfg(playout_cap={"fast_sims": 40}, mcts={"epsilon": 0.3})
    hp = SelfPlayHParams.from_config(with_key)
    assert hp.dirichlet_epsilon == 0.3, (
        "the CONFIG KEY is `mcts.epsilon`; reading the field spelling "
        "`dirichlet_epsilon` off the config returns None on every config"
    )
    assert assemble(with_key).recorded_kwargs["dirichlet_epsilon"] == 0.3

    wrong_spelling = cfg(playout_cap={"fast_sims": 40},
                         mcts={"dirichlet_epsilon": 0.9})
    hp_wrong = SelfPlayHParams.from_config(wrong_spelling)
    assert hp_wrong.dirichlet_epsilon == 0.25, (
        "a config carrying the FIELD spelling must be ignored (old truth): the key is "
        "`epsilon`, so this config gets the default"
    )
    assert assemble(wrong_spelling).recorded_kwargs["dirichlet_epsilon"] == 0.25


def test_dirichlet_alpha_field_name_equals_its_key(assemble) -> None:
    """CORRECTION-3b pin (adjacent-line control) — PASS iff `mcts.dirichlet_alpha` DOES
    reach `dirichlet_alpha`. The two knobs are read on adjacent frozen lines, one by its
    own name and one not; without this control the epsilon pin could be "fixed" by
    reading `dirichlet_*` for both and breaking alpha instead."""
    config = cfg(playout_cap={"fast_sims": 40}, mcts={"dirichlet_alpha": 0.25})
    assert SelfPlayHParams.from_config(config).dirichlet_alpha == 0.25
    assert assemble(config).recorded_kwargs["dirichlet_alpha"] == 0.25


# ═══ D-11 — the ply-cap value chain and its wire site ════════════════════════════
@pytest.mark.parametrize(
    "training,expected_draw,expected_ply",
    [
        ({"draw_value": -0.5, "ply_cap_value": -0.9}, -0.5, -0.9),
        ({"draw_value": -0.3}, -0.3, -0.3),
        (None, -0.5, -0.5),
    ],
    ids=["explicit", "falls_back_to_draw_value", "both_absent"],
)
def test_ply_cap_value_wire(assemble, training, expected_draw, expected_ply) -> None:
    """D-11 — PASS iff the three captured ply-cap cases resolve as captured AND land on
    the runner's `ply_cap_value` / `draw_reward` kwargs.

    Note the kwarg names differ from the config keys on BOTH sides of this pair:
    `training.draw_value` → `draw_reward`, `training.ply_cap_value` → `ply_cap_value`.
    FAIL = ply-capped truncations and organic draws collapse onto one value target, which
    is the split this key exists to make."""
    config = cfg(playout_cap={"fast_sims": 40})
    if training is not None:
        config["training"] = training

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

    recorded = assemble(cfg(playout_cap={"fast_sims": 40}), seed_prefixes=prefixes)
    assert recorded.recorded_attrs["seed_corpus"] == prefixes
    assert recorded.real.seed_corpus == [[(0, 0), (1, -1)], [(2, 3)]]


def test_absent_seed_prefixes_leave_the_attribute_unset(assemble) -> None:
    """D-12 (negative arm) — PASS iff `seed_prefixes=None` never touches `seed_corpus`, so
    the Rust default (no seeding) stands. FAIL = an explicit empty corpus is written,
    which is a different thing from "no corpus"."""
    recorded = assemble(cfg(playout_cap={"fast_sims": 40}), seed_prefixes=None)
    assert "seed_corpus" not in recorded.recorded_attrs
    assert recorded.real.seed_corpus is None


# ═══ D-13 — inference_pool_size threading ════════════════════════════════════════
@pytest.mark.parametrize(
    "selfplay_over,expected",
    [({}, None), ({"inference_pool_size": 999}, 999),
     ({"inference_pool_size": "777"}, 777)],
    ids=["absent", "int", "string_coerced"],
)
def test_inference_pool_size_threading(assemble, selfplay_over, expected) -> None:
    """D-13 — PASS iff the opt-in pool size threads through as captured: absent stays
    `None` (the engine's own prefill sizing), an int passes, a string coerces. FAIL = a
    YAML-quoted number silently becomes a `TypeError` at the FFI boundary, or `None`
    turns into a hard-coded size that pins the working set for every encoding."""
    selfplay = {"playout_cap": {"fast_sims": 40}}
    selfplay.update(selfplay_over)
    config = cfg(selfplay=selfplay)

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
    bare object carrying only `config` so no runner is needed. FAIL = the regime guard
    freezes at construction and the emitter reports PUCT diagnostics for a Gumbel run."""
    from mantis.selfplay.pool import WorkerPool

    holder = object.__new__(WorkerPool)
    holder.config = {"selfplay": {"gumbel_mcts": False}}
    assert holder.gumbel_mcts is False

    holder.config["selfplay"]["gumbel_mcts"] = True
    assert holder.gumbel_mcts is True, "the property must re-read the live config"

    holder.config = {}
    assert holder.gumbel_mcts is False, "absent key ⇒ False, with the namespace fallback"

    hp = SelfPlayHParams.from_config(cfg(playout_cap={"fast_sims": 40},
                                         selfplay={"gumbel_mcts": True,
                                                   "playout_cap": {"fast_sims": 40}}))
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

    config = cfg(encoding=encoding, playout_cap={"fast_sims": 40})
    hp = SelfPlayHParams.from_config(config)
    enc = resolve_pool_encoding(config, arch=None)
    _, dims = build_runner_config(hp, spec_dims=enc, encoding_name=enc.encoding_name,
                                  seed_prefixes=None)

    batcher = InferenceBatcher(encoding_spec=lookup(encoding))
    assert dims.feat_len == int(batcher.feature_len_py)
    assert dims.pol_len == int(batcher.policy_len_py)


def test_max_moves_alias_chain(assemble) -> None:
    """D-06 (alias arm; captured as MAXMOVES_*) — PASS iff BOTH spellings reach
    `max_moves_per_game`, with the primary key winning.

    The FIELD is named after the SECONDARY key: `max_moves_per_game` is the fallback and
    `max_game_moves` is what the config actually sets first. A port that read only the
    field's own name would silently ignore every config using the primary spelling and cap
    games at the code-side 128."""
    primary = cfg(selfplay={"max_game_moves": 77, "playout_cap": {"fast_sims": 40}})
    secondary = cfg(selfplay={"max_moves_per_game": 88, "playout_cap": {"fast_sims": 40}})
    both = cfg(selfplay={"max_game_moves": 77, "max_moves_per_game": 88,
                         "playout_cap": {"fast_sims": 40}})

    assert SelfPlayHParams.from_config(primary).max_moves_per_game == 77
    assert SelfPlayHParams.from_config(secondary).max_moves_per_game == 88
    assert SelfPlayHParams.from_config(both).max_moves_per_game == 77, (
        "`max_game_moves` is the primary key and must win the alias chain"
    )
    assert assemble(primary).recorded_kwargs["max_moves_per_game"] == 77


def test_killed_knobs_are_never_read(assemble) -> None:
    """DV-6 pin — PASS iff a config carrying the two WP-KILLed self-play knobs assembles
    cleanly and neither name reaches the Rust config.

    `legal_move_radius_jitter` was a ctor kwarg and `interior_selector` a post-ctor attr
    that the frozen code HARD-read (a missing key was a `KeyError`). Removing that read
    removes an error path: a config omitting `interior_selector` now constructs instead of
    raising. Declared, and asserted here so it is not rediscovered as a surprise."""
    config = cfg(
        selfplay={"legal_move_radius_jitter": True, "playout_cap": {"fast_sims": 40}},
        mcts={},                       # no interior_selector at all — used to be fatal
    )
    recorded = assemble(config)
    assert "legal_move_radius_jitter" not in recorded.recorded_kwargs
    assert "interior_selector" not in recorded.recorded_attrs


def test_hparams_round_trip_is_json_stable() -> None:
    """Guard arm — PASS iff every resolved hparam is a plain Python scalar (or None), so
    the whole knob set can be recorded into a run manifest. FAIL = a numpy scalar or a
    config sub-dict leaked into the frozen snapshot and the manifest write dies mid-run."""
    hp = SelfPlayHParams.from_config(cfg(playout_cap={"fast_sims": 40}))
    payload = {f: getattr(hp, f) for f in hp.__dataclass_fields__}
    json.dumps(payload)  # raises TypeError on any non-JSON scalar
    for name, value in payload.items():
        assert value is None or isinstance(value, (int, float, str, bool)), (
            f"{name} resolved to {type(value).__name__}, not a plain scalar"
        )
