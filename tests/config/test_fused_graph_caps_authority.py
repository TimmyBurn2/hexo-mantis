# >300 justify (R8): the rows here are ONE claim — `inference.fused_graph_caps` has exactly ONE
# authority, is minted in every config, and cannot be silently absent — over one apparatus: the
# real loader, `discover_configs`, and one `ast` parse of the read path. Splitting the config
# sweep from the no-`.get` census separates the pair that must hold together for R1 to mean anything.
"""The fused-graph cap's config authority.

THE MINT POSTURE, so the rows below read correctly: `int | None` with `ge=1` on the int arm and
NO "uncapped" sentinel, because a disable sentinel is a switch for turning the fix off. `null`
is a placeholder, not an off state — schema-VALID so gate 7 stays green, runtime-REFUSED so a
graph run on an uncalibrated production config cannot construct its inference server.

The rows are the only witnesses to: a key minted into some configs and not others; a GUESSED
production value; `null` silently meaning "uncapped"; an absence that defaults instead of
raising; a refusal not REACHED at construction; a `.get(...)` on the read path; "non-binding by
construction" becoming false; a stale sweep premise; and an expressible "uncapped" sentinel.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from mantis.config.loader import discover_configs, load_config
from mantis.config.resolve.fused_graph_caps import (
    FusedGraphCapsSpec,
    MissingFusedGraphCapsError,
    UncalibratedFusedGraphCapsError,
    resolve_fused_graph_caps,
)
from mantis.config.schema.selfplay import FusedGraphCapsConfig
from mantis.encoding import lookup
from mantis.selfplay.graph_wire_split import plan_fused_forwards
from mantis.selfplay.inference_server import InferenceServer

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"
_READ_PATH = _REPO / "src" / "mantis" / "config" / "resolve" / "fused_graph_caps.py"

#: The two configs whose value is the OPERATOR'S, minted at the box sitting from the calibration
#: tool's output. `run6.yaml` joins run5 on run5's own grounds: a box-class config that already
#: mints run5's `microbatch_caps` and is already excluded beside it from the train-side sweep.
_PRODUCTION = ("run6.yaml",)
_NON_PRODUCTION = ("dev_example.yaml", "smoke_preflight_armed.yaml")


def _all_config_names() -> list[str]:
    return sorted(p.relative_to(_CONFIGS).as_posix() for p in discover_configs(_CONFIGS))


def _names_by_arch(arch: str) -> list[str]:
    """Return the shipped configs that SELECT `arch`, read off each file through the one loader.
    Derived rather than listed, so re-minting a config to the other representation moves it
    between these sets instead of leaving a stale name behind."""
    return [name for name in _all_config_names()
            if load_config(_CONFIGS / name).identity.representation == arch]


_GRAPH_CONFIGS = _names_by_arch("graph")
_GRID_CONFIGS = _names_by_arch("grid")


@pytest.mark.parametrize("name", _GRAPH_CONFIGS)
def test_fg5_01_every_GRAPH_config_mints_the_block_through_the_real_loader(name: str) -> None:
    """The block is present, complete and typed in every shipped GRAPH config, read back through
    the REAL loader. The complement — a grid config carrying it is refused — is asserted below."""
    cfg = load_config(_CONFIGS / name)
    block = cfg.inference.fused_graph_caps
    assert block is not None, f"{name}: `inference.fused_graph_caps` is absent"
    for member in ("max_fused_edges", "max_fused_nodes"):
        value = getattr(block, member)
        assert value is None or isinstance(value, int), (
            f"{name}: {member} is {value!r} ({type(value).__name__}); the schema admits "
            "`int >= 1` or the `null` placeholder and nothing else")
        assert value is None or value >= 1, f"{name}: {member}={value} is below the range"


# FG5-01b — RETIRED with the grid representation: it parametrized over an empty `_GRID_CONFIGS`,
# so pytest collected a permanent empty parameter set, a SKIP that reads like coverage.


def test_fg5_01c_the_arch_split_covers_every_shipped_config(name=None) -> None:
    """Vacuity guard: both parametrize lists are DERIVED, so one going empty is a live way for a
    row above to assert nothing."""
    assert _GRAPH_CONFIGS, "no shipped config selects graph; FG5-01 asserts nothing"
    # A config selecting neither arch would drop out of both sweeps unnoticed.
    assert _GRID_CONFIGS == [], (
        "a grid config is shipped again; FG5-01b's arm was retired with the representation "
        "(R346(f)) and would now assert nothing over it")
    assert sorted(_GRAPH_CONFIGS + _GRID_CONFIGS) == _all_config_names(), (
        "the two arch lists do not partition the shipped configs")


@pytest.mark.parametrize("name", _PRODUCTION)
def test_fg5_02_the_production_configs_ship_the_minted_pair(name: str) -> None:
    """The production configs ship the MINTED pair, as of the box sitting.

    This row used to assert both members were `null`, and the flip is recorded rather than
    quietly swapped. What it still guards: both members are VALUED TOGETHER, sized from ONE fit
    against ONE budget so a half-minted block is unreachable; and a dispatcher-chosen number is
    still forbidden, since what licenses this one is the recorded acceptance.
    """
    block = load_config(_CONFIGS / name).inference.fused_graph_caps
    assert block.max_fused_edges is not None and block.max_fused_nodes is not None, (
        f"{name} ships an UNCALIBRATED fused-graph cap ({block.max_fused_edges}, "
        f"{block.max_fused_nodes}). Since the 2026-08-18 mint both members are valued; a "
        "`null` here now means a re-mint dropped the measured pair, and the run will refuse "
        "at inference-server construction (UncalibratedFusedGraphCapsError).")
    assert block.max_fused_edges >= 1 and block.max_fused_nodes >= 1


#: WHAT ONE FIT IS DENOMINATED IN, derived from the config rather than listed: `fusion_calibrate`
#: solves at the operating E/N implied by the encoding, `selfplay.max_game_moves` and
#: `inference.inference_batch_size`, and the arch decides the forward. All four shared, one fit.
def _fit_identity(name: str) -> tuple:
    cfg = load_config(_CONFIGS / name)
    return (cfg.identity.encoding, cfg.identity.arch_kind,
            int(cfg.selfplay.max_game_moves), int(cfg.inference.inference_batch_size))


def test_fg5_02_production_configs_SHARING_A_FIT_carry_the_SAME_minted_pair() -> None:
    """ONE fit, ONE card, ONE partition — so one pair per FIT, in every file that shares it.

    THE PREMISE MOVED AT THE RUN6 MINT: it read "both production configs carry the SAME pair",
    true while every one was `gnn_axis_v1` at ply cap 128. run6 mints `gnn_axis_r8` + `GnnArchV2`
    at ply cap 256, so its fit differs legitimately and demanding one pair would demand a number
    sized for neither. MUTATION THAT REDS IT: re-minting one config of a fit group and leaving
    its twin on the old pair — only the comparison sees it.
    """
    groups: dict[tuple, dict[str, tuple]] = {}
    for name in _PRODUCTION:
        block = load_config(_CONFIGS / name).inference.fused_graph_caps
        groups.setdefault(_fit_identity(name), {})[name] = (
            block.max_fused_edges, block.max_fused_nodes)
    assert groups, "no production config was read, so this comparison asserts nothing"
    for fit, pairs in groups.items():
        assert len(set(pairs.values())) == 1, (
            f"production configs sharing the fit {fit} disagree about the fused-graph bound: "
            f"{pairs}. They partition the SAME card from the SAME sweep; a divergence means "
            "one was minted without the other, which R281(d) rules is not a legal posture.")
    # THE CROSS-FILE COMPARISON HAS NO SUBJECT, stated rather than papered over: `configs/` holds
    # ONE production config, so every fit group is a group of one. The vacuity is asserted in the
    # direction that survives, so a SECOND production config reds this until the row is re-armed.
    assert sum(len(pairs) for pairs in groups.values()) == 1, (
        f"more than one production config is shipped ({ {k: sorted(v) for k, v in groups.items()} }); "
        "the cross-file fit comparison above is live again and this vacuity note must be "
        "replaced by the `any(len(pairs) > 1)` arm it stands in for")


def test_fg5_02_the_placeholder_is_schema_valid_so_gate_7_stays_green() -> None:
    """`null` VALIDATES: an unrepresentable placeholder would leave gate 7 red on `dev`."""
    assert FusedGraphCapsConfig(max_fused_edges=None, max_fused_nodes=None) is not None
    assert FusedGraphCapsConfig(max_fused_edges=1, max_fused_nodes=1) is not None


@pytest.mark.parametrize("member", ["max_fused_edges", "max_fused_nodes"])
def test_fg5_03_a_null_member_refuses_at_read_naming_the_way_out(member: str) -> None:
    """`null` is REFUSED AT READ by a named subclass carrying the remedy — which member, the
    calibration entry point, and the `--set` line — because "never minted" and "malformed" send
    an operator to different places."""
    block = {"max_fused_edges": 4_500_000, "max_fused_nodes": 170_000}
    block[member] = None
    with pytest.raises(UncalibratedFusedGraphCapsError) as exc:
        resolve_fused_graph_caps({"inference": {"fused_graph_caps": block}})
    msg = str(exc.value)
    assert member in msg, f"the null member is not named: {msg!r}"
    assert "fusion_calibrate" in msg, (
        f"the refusal does not name the calibration entry point that produces the value: "
        f"{msg!r}")
    assert "mint_config" in msg, (
        f"the refusal does not carry the mint line that fixes it: {msg!r}")
    assert issubclass(UncalibratedFusedGraphCapsError, MissingFusedGraphCapsError), (
        "an uncalibrated cap is a special case of an unusable one; a caller that handles the "
        "general absence must not miss the placeholder")


_ABSENCE_CASES = [
    ("not a mapping", "banana", "not a mapping"),
    ("no inference section", {"train": {}}, "inference"),
    ("inference not a mapping", {"inference": 7}, "inference"),
    ("no block", {"inference": {"inference_batch_size": 64}}, "fused_graph_caps"),
    ("block not a mapping", {"inference": {"fused_graph_caps": 7}}, "fused_graph_caps"),
    ("edges member absent", {"inference": {"fused_graph_caps": {"max_fused_nodes": 1}}},
     "max_fused_edges"),
    ("nodes member absent", {"inference": {"fused_graph_caps": {"max_fused_edges": 1}}},
     "max_fused_nodes"),
]


@pytest.mark.parametrize(("label", "config", "needle"), _ABSENCE_CASES,
                         ids=[c[0] for c in _ABSENCE_CASES])
def test_fg5_04_absence_raises_and_names_the_level(label: str, config, needle: str) -> None:
    """Absence raises at all seven levels, each naming what is missing: the seven are seven edits."""
    with pytest.raises(MissingFusedGraphCapsError) as exc:
        resolve_fused_graph_caps(config)
    msg = str(exc.value)
    assert "inference.fused_graph_caps" in msg, (
        f"[{label}] the refusal does not name the key path: {msg!r}")
    assert needle in msg, f"[{label}] the refusal does not name the missing level: {msg!r}"


def test_fg5_04_a_complete_block_resolves_to_the_frozen_pair() -> None:
    """The clean twin: a complete block resolves to a FROZEN spec, frozen because a rebindable
    run-scoped constant is a second authority."""
    spec = resolve_fused_graph_caps(
        {"inference": {"fused_graph_caps": {"max_fused_edges": 42, "max_fused_nodes": 7}}})
    assert isinstance(spec, FusedGraphCapsSpec)
    assert (spec.max_fused_edges, spec.max_fused_nodes) == (42, 7)
    with pytest.raises(FrozenInstanceError):
        spec.max_fused_edges = 43  # type: ignore[misc]


class _DummyBatcher:
    def close(self) -> None:
        return None


def test_fg5_05_an_uncalibrated_production_config_cannot_build_its_graph_server() -> None:
    """An uncalibrated production config cannot build its graph server, through the REAL
    `InferenceServer.__init__`.

    EAGER, in the graph branch — `__init__` already branches on `self._is_graph`, so failing a
    mis-minted run in the first second rather than three hours in costs nothing. The caps are
    NULLED IN THE DUMP rather than read as null from the file, which tests the REFUSAL rather
    than the current mint state and so survives every future re-mint.
    """
    cfg = load_config(_CONFIGS / "run6.yaml")
    assert cfg.identity.representation == "graph", (
        "run5 no longer declares the graph representation — this row's premise is gone")
    dump = cfg.model_dump()
    dump["inference"]["fused_graph_caps"] = {"max_fused_edges": None, "max_fused_nodes": None}
    with pytest.raises(UncalibratedFusedGraphCapsError):
        InferenceServer(
            torch.nn.Linear(1, 1), torch.device("cpu"), dump,
            batcher=_DummyBatcher(), encoding_spec=lookup(cfg.identity.encoding),
        )


def test_fg5_05b_the_minted_production_config_DOES_build_its_graph_server() -> None:
    """The other direction: run5's config AS COMMITTED constructs an `InferenceServer`. Without
    it the refusal row could pass forever on a config that had quietly regressed to `null`."""
    cfg = load_config(_CONFIGS / "run6.yaml")
    server = InferenceServer(
        torch.nn.Linear(1, 1), torch.device("cpu"), cfg.model_dump(),
        batcher=_DummyBatcher(), encoding_spec=lookup(cfg.identity.encoding),
    )
    assert server is not None


def test_fg5_06_there_is_no_defaulting_read_anywhere_on_the_read_path() -> None:
    """No `.get(...)`, no `or`-default, no `except KeyError` in the resolver: a defaulting read
    on the input to a memory-safety cap is the silent-fallback class. An `ast` census and not a
    grep, because a grep cannot tell a `.get` call from the string `".get"` in a docstring."""
    assert _READ_PATH.exists(), (
        f"{_READ_PATH.relative_to(_REPO)} does not exist — there is no ONE read path to "
        "census, so the key has no single authority (design §3.3)")
    tree = ast.parse(_READ_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "get", (
                "a `.get(...)` on the cap read path smuggles the default the schema is "
                f"supposed to own (R1/LAW-11): {ast.unparse(node)[:120]}")
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            raise AssertionError(
                "an `or` on the cap read path is a code-side default for a config fact "
                f"(R1): {ast.unparse(node)[:120]}")
        if isinstance(node, ast.ExceptHandler):
            raise AssertionError(
                "the cap read path catches an exception; absence must PROPAGATE as a named "
                f"raise, never be recovered from: {ast.unparse(node.type or node)[:120]}")


def test_fg5_06_exactly_one_module_reads_the_two_member_names() -> None:
    """Exactly ONE authority reads `max_fused_edges`/`max_fused_nodes` off a config mapping.
    Restricted to SUBSCRIPT reads with a constant string index, so consumers of the RESOLVED
    dataclass are correctly not counted."""
    src_root = _REPO / "src" / "mantis"
    readers: dict[str, int] = {}
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
            and node.slice.value in ("max_fused_edges", "max_fused_nodes",
                                     "fused_graph_caps")
        )
        if hits:
            readers[path.relative_to(_REPO).as_posix()] = hits
    assert set(readers) == {"src/mantis/config/resolve/fused_graph_caps.py"}, (
        f"more than one module reads the cap block out of a config mapping: {readers}. Two "
        "authorities over one byte budget agree right up until they diverge, and no "
        "behavioural oracle can see the second one.")


def _n_ceiling(cfg) -> int:
    """The config's OWN geometric ceiling on one graph's node count: `max_game_moves` stones,
    each a radius-`r` hex ball of legal cells, plus the stones and the dummy node."""
    r = int(lookup(cfg.identity.encoding).legal_move_radius)
    moves = int(cfg.selfplay.max_game_moves)
    return moves * (3 * r * (r + 1) + 1) + moves + 1


#: The non-production sweep's GRAPH half — the only configs that carry the block. Intersected
#: rather than re-listed, so `_NON_PRODUCTION` stays the one place the split is stated.
_NON_PRODUCTION_GRAPH = [n for n in _NON_PRODUCTION if n in _GRAPH_CONFIGS]


@pytest.mark.parametrize("name", _NON_PRODUCTION_GRAPH)
def test_fg5_07_the_non_production_caps_are_non_binding_by_construction(name: str) -> None:
    """"Non-binding by construction" is a DERIVATION, re-run here: a smoke config whose cap BOUND
    would make CI exercise a split by accident with no count changing to say so.

    `E <= 32 N` is the builder's pre-dedup ceiling (3 axes x 2 signs x 5 depths x 2 directions =
    60 directed edges per node, halved by the `(src, dst, axis)` dedup, plus 2 dummy edges), and
    `N_ceiling` is the config's own geometry, so the cap must exceed
    `inference_batch_size x 32 x N_ceiling` edges and `inference_batch_size x N_ceiling` nodes.
    """
    cfg = load_config(_CONFIGS / name)
    block = cfg.inference.fused_graph_caps
    assert block.max_fused_edges is not None and block.max_fused_nodes is not None, (
        f"{name} is a non-production config and must mint a real value, not the R119 "
        "placeholder — CI and preflight have to be able to BOOT")
    n_ceiling = _n_ceiling(cfg)
    batch = int(cfg.inference.inference_batch_size)
    assert block.max_fused_edges > batch * 32 * n_ceiling, (
        f"{name}: max_fused_edges={block.max_fused_edges} does not exceed the config's own "
        f"worst-case pop of {batch * 32 * n_ceiling} edges — it can bind, and CI would then "
        "be exercising a split by accident (MB-24)")
    assert block.max_fused_nodes > batch * n_ceiling, (
        f"{name}: max_fused_nodes={block.max_fused_nodes} does not exceed the config's own "
        f"worst-case pop of {batch * n_ceiling} nodes")
    assert cfg.identity.representation == "graph"   # the sweep's own premise, executed


@pytest.mark.parametrize("name", _NON_PRODUCTION_GRAPH)
def test_fg5_07_the_non_production_caps_never_split_their_own_worst_case_pop(
    name: str
) -> None:
    """The arithmetic above through the REAL planner: asserting only the arithmetic would let a
    planner that mis-reads its caps split anyway."""
    cfg = load_config(_CONFIGS / name)
    block = cfg.inference.fused_graph_caps
    batch = int(cfg.inference.inference_batch_size)
    n_ceiling = _n_ceiling(cfg)
    node_counts = np.full(batch, n_ceiling, dtype=np.int64)
    edge_counts = node_counts * 32
    parts = plan_fused_forwards(
        np.concatenate([[0], np.cumsum(edge_counts)]).astype(np.int64),
        np.concatenate([[0], np.cumsum(node_counts)]).astype(np.int64),
        FusedGraphCapsSpec(int(block.max_fused_edges), int(block.max_fused_nodes)),
    )
    assert len(parts) == 1, (
        f"{name}: the minted caps ({block.max_fused_edges}, {block.max_fused_nodes}) split "
        f"its own saturated pop into {len(parts)} forwards")


def test_fg5_08_production_is_excluded_deliberately_and_the_set_is_the_directory() -> None:
    """The two sweeps partition ALL the configs, enumerated by `discover_configs` — a second flat
    glob here would let a subdirectory or `.yml` shape both gates make legal slip out silently."""
    assert _all_config_names() == sorted(_PRODUCTION + _NON_PRODUCTION), (
        "a config was added or renamed; both sweeps above now have a stale premise")
    assert _NON_PRODUCTION_GRAPH, (
        "no non-production config selects graph, so FG5-07's two limbs assert nothing")
    assert set(_NON_PRODUCTION) - set(_NON_PRODUCTION_GRAPH) == set(_GRID_CONFIGS), (
        "the configs FG5-07 skips must be EXACTLY the grid ones — a graph config dropping out "
        "of that sweep for any other reason is a coverage hole, not a scoping. With the grid "
        "representation deleted (R346(f)) that set is EMPTY, so FG5-07 must skip nothing")


@pytest.mark.parametrize("bad", [0, -1, -1_000_000])
@pytest.mark.parametrize("member", ["max_fused_edges", "max_fused_nodes"])
def test_fg5_09_the_schema_cannot_express_uncapped(member: str, bad: int) -> None:
    """`ge=1` and NO sentinel: a disable sentinel is a switch for turning the fix off."""
    kwargs = {"max_fused_edges": 1, "max_fused_nodes": 1}
    kwargs[member] = bad
    with pytest.raises(ValidationError):
        FusedGraphCapsConfig(**kwargs)


def test_fg5_09_the_block_forbids_unknown_members() -> None:
    """`extra="forbid"`: a third member added to a two-member fact is a third authority."""
    with pytest.raises(ValidationError):
        FusedGraphCapsConfig(max_fused_edges=1, max_fused_nodes=1, max_fused_bytes=1)
