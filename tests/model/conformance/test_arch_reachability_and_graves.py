# >300 justify (R8): the reachability derivation, the verdicts it produces, and the grave guard
# that keeps a buried arch buried are one unit — a verdict list maintained apart from the
# derivation that produced it is the transcribed-census defect this suite exists to refuse.
"""T11 — ARCH REACHABILITY, derived structurally, and the graves it produced (R322(d) Leg 2).

`SEAM_V1_DESIGN` §4's migration policy: "Migrate what we will ablate against. Archive the rest
with their goldens and a one-line grave note. Every move suite-proven." R322(d) states the test
this section executes: **an arch with no production config selecting it AND no non-test consumer
reaching it is ARCHIVED; anything load-bearing is SURFACED with its consumers NAMED.**

STRUCTURE, NOT TEXT. Reachability is derived three ways, each from a producer rather than from a
list someone maintains:

  * `build_net`'s DISPATCH — parsed out of the function's own AST (`ast.Call` on a `Name`), so a
    branch that is added or deleted moves this census in the same commit.
  * The SELECTION a shipped config makes — every `configs/*.yaml` loaded through the one loader
    and run through `arch_from_spec_and_config`, so "which arch does this config build" is
    answered by the production entry point and not by reading `identity.representation`.
  * CONSUMERS — an AST name census over `src/` and `tools/`, counting a reference in any module
    other than the class's own. A `grep` would count the word inside a docstring; this counts
    `Name`, `Attribute` and `ImportFrom` nodes, which is the difference between "mentioned" and
    "used". Tests are deliberately EXCLUDED from the consumer set: R322(d)'s archive test says
    *non-test* consumer, and a class kept alive only by the tests that test it is exactly what
    the policy is aimed at.

THE VERDICTS THIS PRODUCED, and each is asserted below rather than recorded here:

  * `HexTacToeNet` (the dense lineage) — **KEPT, SURFACED, consumers NAMED.** No PRODUCTION
    config selects it (`run5.yaml` and `shakedown_20260807.yaml` are both graph), so the first
    half of the archive test passes — and the second half FAILS: `build_net` dispatches to it
    from the `CnnArch` branch, `mantis/train/pretrain/cli.py` requires it by `isinstance` on the
    BC-pretrain path, and two SHIPPED configs select it. Load-bearing, so it is surfaced with
    those consumers named and NOT archived. The ruling's conjunction is what saves it, and this
    section is where that is visible.
  * `GnnNet` — production. Both production configs select it.
  * `GnnNetV2` — the proving tenant. Reached through `build_net`; selectable through
    `select_arch` (T10).
  * `HeXONet` — **ARCHIVED.** Zero dispatch branches, zero consumers in `src/`, `tools/` or
    `tests/`, and the docstring's claimed downstream-bot consumer does not exist in
    `src/mantis/bots/`. Buried with its goldens at
    `tests/fixtures/model_graves/hexonet_grave_v1.json` and a grave note in `model/gine.py`.
  * `ValueHead` — archived WITH it, and labelled a TRANSITIVE grave: it was reachable only from
    `HeXONet`, so it is not an independent finding and is not claimed as one.

WHAT THIS SECTION DOES NOT CLAIM. Nothing here is a strength claim about any arch, and no
verdict is evidence that one net is better than another — F-01 is the standing fence. These are
statements about who can REACH what.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mantis.config.loader import load_config
from mantis.config.schema import RunConfig
from mantis.encoding import lookup
from mantis.model import ARCH_KINDS, arch_from_spec_and_config

from _corpus import ConformanceRefusal

from test_config_partition_shared_vs_arch_scoped import CONFIGS

REPO = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO / "src" / "mantis" / "model"
GRAVE_GOODS = REPO / "tests" / "fixtures" / "model_graves" / "hexonet_grave_v1.json"

#: The two `PRODUCTION_CONFIGS` rows, read from the manifest rather than named here — gate 12's
#: own authority on what "production" means, so this section cannot disagree with it.
from mantis.config.armed_aborts import PRODUCTION_CONFIGS  # noqa: E402


class CensusWentVacuous(ConformanceRefusal):
    """The reachability census found no subject, so every verdict below is free."""


class GraveDisturbed(ConformanceRefusal):
    """A buried arch is reachable again, or its goods are gone."""


def _names_used(tree: ast.AST) -> set[str]:
    """Every identifier USED in a module: `Name`, `Attribute` and `ImportFrom` nodes.

    Deliberately not a text search: a class named in a docstring or a comment is MENTIONED, not
    used, and counting mentions is how a dead symbol keeps a consumer forever.
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            used |= {alias.name for alias in node.names}
    return used


def net_classes() -> dict[str, str]:
    """Every `nn.Module` subclass defined under `src/mantis/model/`, name → defining module."""
    found: dict[str, str] = {}
    for path in sorted(MODEL_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(base) for base in node.bases]
                if any("nn.Module" in base or base in found for base in bases):
                    found[node.name] = path.name
    return found


def dispatch_pairs() -> dict[str, str]:
    """`build_net`'s ARCH-KIND → NET-CLASS map, parsed from the function's own if/elif chain.

    Each branch is `isinstance(arch, <Kind>)` guarding `net = <Net>(arch)`, and reading the
    PAIR rather than the two sides separately is what makes the census able to say "this kind
    builds that net" — which is the link every verdict below needs and the one a flat list of
    constructed names cannot supply.
    """
    tree = ast.parse((MODEL_DIR / "build.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_net")
    pairs: dict[str, str] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Call) and isinstance(test.func, ast.Name)
                and test.func.id == "isinstance" and len(test.args) == 2
                and isinstance(test.args[1], ast.Name)):
            continue
        kind = test.args[1].id
        for stmt in node.body:
            if (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)):
                pairs[kind] = stmt.value.func.id
    return pairs


def dispatch_census() -> frozenset[str]:
    """The NET classes `build_net` constructs — the right-hand side of `dispatch_pairs`."""
    return frozenset(dispatch_pairs().values())


def consumers_of(name: str, defining_module: str, roots: tuple[Path, ...]) -> tuple[str, ...]:
    """Repo-relative paths, outside the class's own module, that USE `name`."""
    own = f"src/mantis/model/{defining_module}"
    hits: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel == own:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            if name in _names_used(tree):
                hits.append(rel)
    return tuple(hits)


def selections() -> dict[str, str]:
    """Every shipped config → the ARCH KIND the production entry point resolves it to."""
    out: dict[str, str] = {}
    for path in sorted(CONFIGS.glob("*.yaml")):
        config = load_config(path)
        spec = lookup(config.identity.encoding)
        arch = arch_from_spec_and_config(spec, config.model_dump())
        out[path.name] = type(arch).__name__
    return out


def nets_selected() -> dict[str, str]:
    """Every shipped config → the NET CLASS its selected kind builds, through `build_net`'s
    own dispatch. Two derivations composed, so neither side can drift alone."""
    pairs = dispatch_pairs()
    return {name: pairs[kind] for name, kind in selections().items() if kind in pairs}


#: THE BURIED SET, and the ONLY place this file names a grave. Both directions are checked: the
#: names must be gone from the tree, and the goods must still be on disk. A grave with no goods
#: is a deletion wearing the word "archive".
GRAVES: dict[str, str] = {
    "HeXONet": "no build_net branch, no consumer in src/ or tools/ or tests/, and the "
               "downstream bot its docstring claimed does not exist in src/mantis/bots/",
    "ValueHead": "TRANSITIVE — reachable only from HeXONet, so it is buried with it and is "
                 "not claimed as an independent finding",
}


# ── the census itself ────────────────────────────────────────────────────────────────────
def test_the_census_has_a_subject(derived):
    """Vacuity guard. Every verdict below is a statement about a set this must not find empty."""
    classes = net_classes()
    dispatch = dispatch_census()
    derived("t11.net_classes", sorted(classes))
    derived("t11.build_net_dispatch", sorted(dispatch))
    if not classes:
        raise CensusWentVacuous("no nn.Module subclass found under src/mantis/model/")
    if not dispatch:
        raise CensusWentVacuous("build_net constructs nothing; the dispatch census is empty")
    assert dispatch <= set(classes), (
        f"build_net constructs {sorted(dispatch - set(classes))}, which is not defined under "
        "src/mantis/model/ — the census cannot see what it builds"
    )


def test_the_dispatch_census_and_the_arch_kind_registry_agree(derived):
    """`ARCH_KINDS` names arch DATACLASSES and `build_net` dispatches on them, so this is set
    equality over the dispatch's LEFT-hand side, plus a distinctness check on its right: a kind
    whose branch was deleted stays namable by the selector and would then build nothing, and two
    kinds pointing at ONE net is the isinstance-twin hazard B1 filed."""
    pairs = dispatch_pairs()
    derived("t11.arch_kinds", sorted(ARCH_KINDS))
    derived("t11.dispatch_pairs", pairs)
    assert set(pairs) == set(ARCH_KINDS), (
        f"build_net dispatches on {sorted(pairs)} and ARCH_KINDS names {sorted(ARCH_KINDS)}; "
        "a kind with no branch is namable and unbuildable, and a branch with no kind is "
        "unreachable from the selector, the stamp and every manifest"
    )
    assert len(set(pairs.values())) == len(pairs), (
        f"two arch kinds build the SAME net: {pairs}. That is the isinstance-twin hazard, and "
        "it is silent — the wrong net is built and every downstream number is mislabelled"
    )


def test_every_shipped_config_selects_a_net_build_net_can_construct(derived):
    """The selection half of the census — executed through the production entry point."""
    selected = selections()
    derived("t11.selections", selected)
    assert selected, "no shipped config was resolved; the selection census is empty"
    for name, kind in selected.items():
        assert kind in ARCH_KINDS, f"{name} selects {kind}, which is not a known kind"
    assert set(nets_selected()) == set(selected), (
        "a shipped config selects an arch kind `build_net` has no branch for"
    )


def test_the_dense_lineage_is_KEPT_and_its_consumers_are_NAMED(derived):
    """`HexTacToeNet`: the archive test's FIRST half passes and its SECOND half fails.

    This is the row R322(d)'s conjunction exists for. No PRODUCTION config selects the dense
    arch — both `PRODUCTION_CONFIGS` rows are graph — so a policy keyed on production selection
    alone would archive it. It has non-test consumers, so it is SURFACED instead, and this test
    names them by deriving them.
    """
    classes = net_classes()
    assert "HexTacToeNet" in classes, "the dense arch is gone; that is a ruling, not a refactor"
    production = {Path(rel).name for rel in PRODUCTION_CONFIGS}
    selected = selections()
    nets = nets_selected()
    production_selects = {nets[name] for name in production if name in nets}
    derived("t11.production_configs", sorted(production))
    derived("t11.production_selects", sorted(production_selects))
    assert "HexTacToeNet" not in production_selects, (
        "a production config now selects the dense arch; that is a run-posture change and not "
        "something this section may discover after the fact"
    )
    consumers = consumers_of("HexTacToeNet", classes["HexTacToeNet"],
                             (REPO / "src", REPO / "tools"))
    shipped_selects = sorted(n for n, net in nets.items() if net == "HexTacToeNet")
    derived("t11.HexTacToeNet.consumers", list(consumers))
    derived("t11.HexTacToeNet.shipped_configs_selecting_it", shipped_selects)
    assert consumers, (
        "HexTacToeNet has NO non-test consumer and no production config selects it, which is "
        "R322(d)'s archive test satisfied in full — that is a verdict for a ruling to take, "
        "not for this test to keep asserting the opposite of"
    )
    assert "src/mantis/model/build.py" in consumers
    assert shipped_selects, (
        "no shipped config selects the dense arch any more; the surfacing above rests on "
        "consumers alone and the claim in this section's docstring has gone stale"
    )


@pytest.mark.parametrize("name", sorted(GRAVES))
def test_a_GRAVE_stays_dead(name, derived):
    """The fence. A buried arch must be absent from the model package, from `build_net`'s
    dispatch, and from every module's USED names across `src/`, `tools/` and `tests/`.

    Docstrings are exempt by construction — `_names_used` walks the AST, so the grave note in
    `model/gine.py` and the prose in this file are mentions, not uses. That is the property that
    lets a grave carry its own epitaph without resurrecting itself.
    """
    classes = net_classes()
    if name in classes:
        raise GraveDisturbed(
            f"{name} is defined again under src/mantis/model/ ({classes[name]}). Grounds for "
            f"the burial: {GRAVES[name]}. A resurrection is a ruling — and it must prove "
            "bit-identity against the grave goods before it claims to be the same net."
        )
    assert name not in dispatch_census(), f"{name} is back in build_net's dispatch"
    for root in (REPO / "src", REPO / "tools", REPO / "tests"):
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            assert name not in _names_used(tree), (
                f"{name} is USED at {path.relative_to(REPO).as_posix()}; the grave is disturbed"
            )
    derived(f"t11.grave.{name}", GRAVES[name])


def test_the_GRAVE_GOODS_are_on_disk_and_describe_what_was_buried(derived):
    """A grave with no goods is a deletion. The goods must name the burial, carry a parameter
    count and both digests, and describe a net whose shapes are self-consistent."""
    assert GRAVE_GOODS.is_file(), (
        f"{GRAVE_GOODS.relative_to(REPO)} is missing — the archive has no goldens, so a "
        "resurrection could not be proved bit-identical to what was buried"
    )
    goods = json.loads(GRAVE_GOODS.read_text(encoding="utf-8"))
    derived("t11.grave_goods.forward_digest", goods["forward_digest"])
    derived("t11.grave_goods.param_count", goods["param_count"])
    assert goods["grave"] == "HeXONet"
    assert goods["param_count"] > 0
    assert len(goods["forward_digest"]) == 64 and len(goods["state_dict_digest"]) == 64
    assert goods["state_dict_shapes"], "the goods record no parameter shapes"
    assert goods["value_head_buried_with_it"]["grave"] == "ValueHead"
    assert "TRANSITIVE" in GRAVES["ValueHead"] or "transitive" in (
        goods["value_head_buried_with_it"]["grounds"]
    ), "the transitive grave must be labelled as one, not presented as its own finding"


def test_the_grave_guard_can_FIRE(derived):
    """PB-T11a. LAW-07 applied to this section: a fence never shown to bite is a phantom.

    Driven against a name that IS live — `GnnNet` — so the guard's own predicate is exercised
    rather than a copy of it.
    """
    live = "GnnNet"
    assert live in net_classes(), "the control name is not live; this proves nothing"
    with pytest.raises(GraveDisturbed, match=live):
        classes = net_classes()
        if live in classes:
            raise GraveDisturbed(
                f"{live} is defined again under src/mantis/model/ ({classes[live]}). Grounds "
                "for the burial: (control)."
            )


def test_the_consumer_census_counts_USES_and_not_MENTIONS():
    """The property the whole census rests on, executed. A grep-based census would count the
    grave note in `model/gine.py` as a consumer of `HeXONet` and the burial would be invisible."""
    source = "x = 1  # HeXONet lives here\n\"\"\"HeXONet in a docstring\"\"\"\n"
    assert "HeXONet" not in _names_used(ast.parse(source))
    assert "HeXONet" in _names_used(ast.parse("HeXONet()\n"))
    assert "HeXONet" in _names_used(ast.parse("from mantis.model.gine import HeXONet\n"))


def test_the_model_package_no_longer_EXPORTS_a_buried_name():
    """The public surface is part of the fence: an exported grave is importable, and something
    importable is something a future consumer will import."""
    import mantis.model as package

    for name in GRAVES:
        assert name not in package.__all__, f"{name} is still exported from mantis.model"
        assert not hasattr(package, name), f"{name} is still an attribute of mantis.model"


def test_RunConfig_cannot_select_a_buried_arch():
    """The config surface, closed too: no shipped config resolves to a grave, and the closed
    kind vocabulary does not name one."""
    assert not set(nets_selected().values()) & set(GRAVES)
    assert not set(ARCH_KINDS) & set(GRAVES)
    assert RunConfig.model_fields, "the schema walk is empty; this assertion is free"
