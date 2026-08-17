# >300 justify (R8). NO LINE COUNT is stated, per G-DFIX-4 and R192(e)'s derive-or-delete:
# R8 asks for a one-line justification, not a tally, and a number that must be re-edited
# whenever a row is added will eventually be wrong and then be read as evidence.
# The rows here are ONE claim — "`inference.fused_graph_caps` has exactly ONE authority, is
# minted in every config, and cannot be silently absent" — and they share one apparatus: the
# real loader, `discover_configs` as the ONE enumeration authority, and one `ast` parse of the
# read path. Splitting the seven-config sweep from the no-`.get` census would put the "every
# config carries it" claim in one file and the "and it can never quietly default" claim in
# another, which is precisely the pair that has to hold together for R1 to mean anything.
"""⊕ F-816-10 F5 — the cap's config authority (R1, LAW-08, LAW-11, R119).

Written by ORACLE-WRITE **before** the feature exists.

THE MINT POSTURE, RESTATED SO THE ROWS BELOW ARE READ CORRECTLY. `int | None` with `ge=1` on
the int arm and NO "uncapped" sentinel — the off state is deliberately UNREPRESENTABLE,
because a disable sentinel is a switch for turning the fix off
(`MicrobatchCapsConfig`'s own recorded refusal, transferred). `null` is NOT an off state: it
is R119's placeholder — schema-VALID, so gate 7 stays green and the repo ships a complete
config, and runtime-REFUSED, so a graph run on an uncalibrated production config CANNOT
CONSTRUCT ITS INFERENCE SERVER. The in-repo precedent for a schema-valid, production-illegal
placeholder awaiting an operator mint is `checkpoint_interval: 0` (R137) and
`random_floor_games: 0` (R147/R272(d)); the difference — and it is an improvement on both —
is that this one RAISES instead of running.

The defect each row is the ONLY witness to:

- **FG5-01** — a key minted into some configs and not others, which R1 forbids and which
  gate 7 alone cannot see (gate 7 validates each file against the schema; a REQUIRED field
  makes absence a schema error, and this row is what proves the field is required rather than
  optional-with-a-default).
- **FG5-02** — the two production configs shipping a GUESSED value. R119 makes the value the
  operator's act at the box sitting; a number here would be a cap nobody measured, minted by
  a dispatcher, on a mint-critical card.
- **FG5-03** — `null` silently meaning "uncapped". That is the F2-ABORT-5(i) refusal exactly:
  *a cap that silently becomes absent-and-unbounded is worse than no cap, because it reports
  as present*.
- **FG5-04** — an absence that defaults instead of raising, at ANY of the six levels. LAW-11's
  shape: absent is an ERROR, and the error NAMES THE LEVEL so the operator knows which line to
  add.
- **FG5-05** — the refusal existing but not being REACHED at construction. Resolving lazily
  would fail a mis-minted run three hours in instead of in the first second.
- **FG5-06** — a `.get(...)` or an `or`-default appearing anywhere on the read path. A grep
  cannot tell a call from a string (R93/DR-11), so this is an `ast` census in the shape of the
  existing `tests/test_run_one_authority.py` authority census.
- **FG5-07** — DESIGN §3.4's "non-binding by construction" silently becoming false, with CI
  exercising a split BY ACCIDENT and no count changing to say so (the MB-24 shape). The
  split's coverage must come from the oracles, where its M is deliberate and asserted.
- **FG5-08** — the sweep's own premise going stale when a config is added. Enumeration is
  `discover_configs` (R71/R75), the ONE authority gates 7 and 12 consume; a second flat glob
  here would be exactly the divergence ADJ-13 F-1 was.
- **FG5-09** — an "uncapped" sentinel (`0`, `-1`) becoming expressible. The bound is the
  mechanism's own range: a fused forward of zero edges is not a fused forward.
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

#: The two configs whose value is the OPERATOR'S, minted at the box sitting from the
#: calibration tool's output (R119). `shakedown_20260807.yaml` joins run5 on run5's own
#: grounds: it is a box-class config that already mints run5's `microbatch_caps`, and it is
#: already excluded beside run5 from the train-side non-binding sweep (F-P2B/R259).
_PRODUCTION = ("run5.yaml", "shakedown_20260807.yaml")
_NON_PRODUCTION = ("dev_example.yaml", "smoke_gnn.yaml", "smoke_preflight_armed.yaml",
                   "smoke_radius_curriculum.yaml", "sustained_kcluster.yaml")


def _all_config_names() -> list[str]:
    return sorted(p.relative_to(_CONFIGS).as_posix() for p in discover_configs(_CONFIGS))


# ═══ FG5-01/02 — every config carries it; production carries the placeholder ═════════════
@pytest.mark.parametrize("name", sorted(_PRODUCTION + _NON_PRODUCTION))
def test_fg5_01_every_config_mints_the_block_through_the_real_loader(name: str) -> None:
    """FG5-01 — the block is present, complete and typed in every shipped config, read back
    through the REAL loader rather than by parsing YAML here."""
    cfg = load_config(_CONFIGS / name)
    block = cfg.inference.fused_graph_caps
    assert block is not None, f"{name}: `inference.fused_graph_caps` is absent"
    for member in ("max_fused_edges", "max_fused_nodes"):
        value = getattr(block, member)
        assert value is None or isinstance(value, int), (
            f"{name}: {member} is {value!r} ({type(value).__name__}); the schema admits "
            "`int >= 1` or the `null` placeholder and nothing else")
        assert value is None or value >= 1, f"{name}: {member}={value} is below the range"


@pytest.mark.parametrize("name", _PRODUCTION)
def test_fg5_02_the_production_configs_ship_the_uncalibrated_placeholder(name: str) -> None:
    """FG5-02 — R119: this packet ships the KEY, the MECHANISM, the TOOL and the PROCEDURE,
    and NEVER the value. A number in either of these two files would be a cap nobody measured.

    Both members are `null` TOGETHER: they are sized from ONE fit against ONE budget, so a
    half-minted block is a state the operator's own act cannot produce."""
    block = load_config(_CONFIGS / name).inference.fused_graph_caps
    assert block.max_fused_edges is None and block.max_fused_nodes is None, (
        f"{name} ships a MINTED fused-graph cap ({block.max_fused_edges}, "
        f"{block.max_fused_nodes}). The value is the operator's, from the calibration tool "
        "at the box sitting (R119) — a dispatcher-chosen number on a mint-critical card is "
        "exactly what this packet may not do.")


def test_fg5_02_the_placeholder_is_schema_valid_so_gate_7_stays_green() -> None:
    """FG5-02 second limb — `null` VALIDATES. The whole posture depends on it: an
    unrepresentable placeholder would leave the repo shipping an incomplete config, and gate 7
    (every `configs/` file schema-validates, empty = fail) would be red on `dev` until the box
    sitting happened."""
    assert FusedGraphCapsConfig(max_fused_edges=None, max_fused_nodes=None) is not None
    assert FusedGraphCapsConfig(max_fused_edges=1, max_fused_nodes=1) is not None


# ═══ FG5-03/04 — the read path refuses ═══════════════════════════════════════════════════
@pytest.mark.parametrize("member", ["max_fused_edges", "max_fused_nodes"])
def test_fg5_03_a_null_member_refuses_at_read_naming_the_way_out(member: str) -> None:
    """FG5-03 — `null` is REFUSED AT READ, by a named subclass, with the remedy in the
    message: which member, the calibration entry point, and the `--set` line that fixes it.

    A distinct subclass (not a bare `MissingFusedGraphCapsError`) because "you never minted
    this" and "your config is malformed" send an operator to two different places."""
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
    """FG5-04 — LAW-11 at all seven levels, each naming what is missing.

    A single "the caps are absent" message would be a refusal an operator cannot act on: the
    seven cases are seven different edits."""
    with pytest.raises(MissingFusedGraphCapsError) as exc:
        resolve_fused_graph_caps(config)
    msg = str(exc.value)
    assert "inference.fused_graph_caps" in msg, (
        f"[{label}] the refusal does not name the key path: {msg!r}")
    assert needle in msg, f"[{label}] the refusal does not name the missing level: {msg!r}"


def test_fg5_04_a_complete_block_resolves_to_the_frozen_pair() -> None:
    """FG5-04's clean twin (LAW-07): the resolver is not refusing everything. A complete block
    resolves to a FROZEN spec — frozen because a resolved run-scoped constant that a consumer
    could rebind is a second authority with extra steps."""
    spec = resolve_fused_graph_caps(
        {"inference": {"fused_graph_caps": {"max_fused_edges": 42, "max_fused_nodes": 7}}})
    assert isinstance(spec, FusedGraphCapsSpec)
    assert (spec.max_fused_edges, spec.max_fused_nodes) == (42, 7)
    with pytest.raises(FrozenInstanceError):
        spec.max_fused_edges = 43  # type: ignore[misc]


# ═══ FG5-05 — the refusal is REACHED, at construction ════════════════════════════════════
class _DummyBatcher:
    def close(self) -> None:
        return None


def test_fg5_05_an_uncalibrated_production_config_cannot_build_its_graph_server() -> None:
    """FG5-05 — run5's OWN config dump, through the REAL `InferenceServer.__init__`.

    EAGER, in the graph branch, at construction — not lazy. The difference from the
    `caps_provider` precedent is deliberate: the microbatch resolver is lazy because eager
    resolution would read `train` on BOTH routes and a grid `full_config` may carry no `train`
    section; here `__init__` ALREADY branches on `self._is_graph`, so the resolution is
    naturally route-scoped and there is nothing to buy by deferring it. Failing a mis-minted
    run in the first second instead of three hours in is the whole value of the placeholder."""
    cfg = load_config(_CONFIGS / "run5.yaml")
    assert cfg.identity.representation == "graph", (
        "run5 no longer declares the graph representation — this row's premise is gone")
    with pytest.raises(UncalibratedFusedGraphCapsError):
        InferenceServer(
            torch.nn.Linear(1, 1), torch.device("cpu"), cfg.model_dump(),
            batcher=_DummyBatcher(), encoding_spec=lookup(cfg.identity.encoding),
        )


# ═══ FG5-06 — one read path, no defaulting read ══════════════════════════════════════════
def test_fg5_06_there_is_no_defaulting_read_anywhere_on_the_read_path() -> None:
    """FG5-06 — no `.get(...)`, no `or`-default, no `except KeyError` in the resolver.

    Ruled rather than stylistic (F2-ABORT-5(i), transferred verbatim from
    `resolve/microbatch.py`): a defaulting read on the input to a memory-safety cap is the
    silent-fallback class — the phantom-gate shape R4/LAW-07 exist to kill. An `ast` census
    and not a grep, because a grep cannot tell a `.get` call from the string `".get"` in a
    docstring (R93/DR-11)."""
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
    """FG5-06 second limb — the OF2-9 census shape applied to the new members: exactly ONE
    authority reads `max_fused_edges`/`max_fused_nodes` off a config mapping.

    Restricted to SUBSCRIPT reads with a constant string index (the config-dict shape), so
    consumers that read the RESOLVED dataclass's attributes are correctly not counted — those
    are consumers of one authority, not second authorities."""
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


# ═══ FG5-07/08 — non-binding by construction ═════════════════════════════════════════════
def _n_ceiling(cfg) -> int:
    """The config's OWN geometric ceiling on one graph's node count, derived from the config
    and the registry rather than chosen: `max_game_moves` stones, each contributing a
    hex ball of radius `r` worth of legal cells, plus the stones and the dummy node."""
    r = int(lookup(cfg.identity.encoding).legal_move_radius)
    moves = int(cfg.selfplay.max_game_moves)
    return moves * (3 * r * (r + 1) + 1) + moves + 1


@pytest.mark.parametrize("name", _NON_PRODUCTION)
def test_fg5_07_the_non_production_caps_are_non_binding_by_construction(name: str) -> None:
    """FG5-07 — DESIGN §3.4's "non-binding by construction" claim is a DERIVATION, and this
    row re-runs it. A smoke config whose cap BOUND would make CI exercise a split by accident,
    with no count changing to say so (MB-24).

    `E <= 32 N` is the builder's own pre-dedup ceiling (3 axes x 2 signs x 5 depths x 2
    directions = 60 directed edges per node, halved by the `(src, dst, axis)` dedup, plus 2
    dummy edges per node); `N_ceiling` is the config's own geometry. So the cap must exceed
    `inference_batch_size x 32 x N_ceiling` in edges and `inference_batch_size x N_ceiling` in
    nodes for the config's own worst-case saturated pop.

    For the two GRID configs the block has NO live consumer at all — the grid branch never
    reads it (design §3.3) — so the check there is an UPPER BOUND on a route those configs
    never take, and it is labelled as one rather than dressed up as a route drive."""
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
    if cfg.identity.representation != "graph":
        assert cfg.identity.representation == "grid"   # never read on this route


@pytest.mark.parametrize("name", _NON_PRODUCTION)
def test_fg5_07_the_non_production_caps_never_split_their_own_worst_case_pop(
    name: str
) -> None:
    """FG5-07 second limb — the arithmetic above, run through the REAL planner.

    The derivation and the planner are two statements of one claim; asserting only the
    arithmetic would let a planner that mis-reads its own caps split anyway."""
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
    """FG5-08 — the two sweeps above partition ALL the configs, enumerated by
    `discover_configs` (R71/R75), the ONE discovery authority gates 7 and 12 consume.

    A second flat `configs/*.yaml` glob here would be exactly the divergence ADJ-13 F-1 was: a
    subdirectory or `.yml` shape both gates make legal would slip out of this sweep silently
    while staying visible to everyone else."""
    assert _all_config_names() == sorted(_PRODUCTION + _NON_PRODUCTION), (
        "a config was added or renamed; both sweeps above now have a stale premise")


# ═══ FG5-09 — the off state is unrepresentable ═══════════════════════════════════════════
@pytest.mark.parametrize("bad", [0, -1, -1_000_000])
@pytest.mark.parametrize("member", ["max_fused_edges", "max_fused_nodes"])
def test_fg5_09_the_schema_cannot_express_uncapped(member: str, bad: int) -> None:
    """FG5-09 — `ge=1` and NO sentinel. The off state is deliberately unrepresentable, because
    a disable sentinel is a switch for turning the fix off — `MicrobatchCapsConfig`'s recorded
    refusal, transferred. The bound is the mechanism's own range: a fused forward of zero
    edges is not a fused forward."""
    kwargs = {"max_fused_edges": 1, "max_fused_nodes": 1}
    kwargs[member] = bad
    with pytest.raises(ValidationError):
        FusedGraphCapsConfig(**kwargs)


def test_fg5_09_the_block_forbids_unknown_members() -> None:
    """FG5-09 second limb — `extra="forbid"` (R1). A third member added to a two-member fact
    would be a third authority over one byte budget."""
    with pytest.raises(ValidationError):
        FusedGraphCapsConfig(max_fused_edges=1, max_fused_nodes=1, max_fused_bytes=1)
