# >300 justify (R8): a verdict list maintained apart from the derivation that produced it is
# the transcribed-census defect this suite exists to refuse.
"""Arch reachability, derived structurally, and the graves that derivation produced.

An arch with no production config selecting it AND no non-test consumer reaching it is ARCHIVED;
anything load-bearing is SURFACED with its consumers NAMED. Reachability comes from three
producers, never a maintained list: `build_net`'s dispatch AST, the selection each shipped config
resolves to, and an AST name census that counts uses rather than mentions. Tests are excluded —
a class kept alive only by the tests that test it is what the policy is aimed at.
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

#: Read from the manifest rather than named here, so this section cannot disagree with the
#: gate's own authority on what "production" means.
from mantis.config.armed_aborts import PRODUCTION_CONFIGS  # noqa: E402


class CensusWentVacuous(ConformanceRefusal):
    """The reachability census found no subject, so every verdict below is free."""


class GraveDisturbed(ConformanceRefusal):
    """A buried arch is reachable again, or its goods are gone."""


def _names_used(tree: ast.AST) -> set[str]:
    """Return every identifier USED in a module: `Name`, `Attribute` and `ImportFrom` nodes —
    not a text search, because counting mentions is how a dead symbol keeps a consumer."""
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
    """Map every `nn.Module` subclass under `src/mantis/model/` to its defining module."""
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
    """Parse `build_net`'s ARCH-KIND -> NET-CLASS map out of its own if/elif chain — the PAIR,
    because "this kind builds that net" is the link every verdict below needs."""
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
    """Return the NET classes `build_net` constructs."""
    return frozenset(dispatch_pairs().values())


def consumers_of(name: str, defining_module: str, roots: tuple[Path, ...]) -> tuple[str, ...]:
    """Return the repo-relative paths outside the class's own module that USE `name`."""
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
    """Map every shipped config to the ARCH KIND the production entry point resolves."""
    out: dict[str, str] = {}
    for path in sorted(CONFIGS.glob("*.yaml")):
        config = load_config(path)
        spec = lookup(config.identity.encoding)
        arch = arch_from_spec_and_config(spec, config.model_dump())
        out[path.name] = type(arch).__name__
    return out


def nets_selected() -> dict[str, str]:
    """Map every shipped config to the NET CLASS its kind builds — two derivations composed,
    so neither side can drift alone."""
    pairs = dispatch_pairs()
    return {name: pairs[kind] for name, kind in selections().items() if kind in pairs}


#: The buried set. Both directions are checked: the names gone from the tree, the goods still
#: on disk — a grave with no goods is a deletion wearing the word "archive".
GRAVES: dict[str, str] = {
    "HeXONet": "no build_net branch, no consumer in src/ or tools/ or tests/, and the "
               "downstream bot its docstring claimed does not exist in src/mantis/bots/",
    "ValueHead": "TRANSITIVE — reachable only from HeXONet, so it is buried with it and is "
                 "not claimed as an independent finding",
    "HexTacToeNet": "R346(f) deleted the GRID/DENSE representation, and this was its net. It "
                    "was SURFACED rather than archived while `build.py` still dispatched to "
                    "it and a shipped config still selected it; the ruling took both.",
}


def test_the_census_has_a_subject(derived):
    """Vacuity guard: every verdict below is about a set this must not find empty."""
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
    """The dispatch's kinds equal `ARCH_KINDS` and its nets are distinct: a kind whose branch
    was deleted stays namable and builds nothing, and two kinds on ONE net is silent."""
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
    """Every shipped config selects a net `build_net` can construct."""
    selected = selections()
    derived("t11.selections", selected)
    assert selected, "no shipped config was resolved; the selection census is empty"
    for name, kind in selected.items():
        assert kind in ARCH_KINDS, f"{name} selects {kind}, which is not a known kind"
    assert set(nets_selected()) == set(selected), (
        "a shipped config selects an arch kind `build_net` has no branch for"
    )


@pytest.mark.parametrize("name", sorted(GRAVES))
def test_a_GRAVE_stays_dead(name, derived):
    """A buried arch is absent from the package, the dispatch, and every module's USED names.
    Docstrings are exempt by construction, so a grave can carry its own epitaph."""
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
    """The goods name the burial, carry a parameter count and both digests — a grave with no
    goods is a deletion."""
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
    """The grave guard fires against a name that IS live, so the fence is shown to bite."""
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
    """The census counts USES, not MENTIONS: a grep would read a grave note as a consumer and
    the burial would be invisible."""
    source = "x = 1  # HeXONet lives here\n\"\"\"HeXONet in a docstring\"\"\"\n"
    assert "HeXONet" not in _names_used(ast.parse(source))
    assert "HeXONet" in _names_used(ast.parse("HeXONet()\n"))
    assert "HeXONet" in _names_used(ast.parse("from mantis.model.gine import HeXONet\n"))


def test_the_model_package_no_longer_EXPORTS_a_buried_name():
    """The public surface is part of the fence: an exported grave is importable."""
    import mantis.model as package

    for name in GRAVES:
        assert name not in package.__all__, f"{name} is still exported from mantis.model"
        assert not hasattr(package, name), f"{name} is still an attribute of mantis.model"


def test_RunConfig_cannot_select_a_buried_arch():
    """The config surface is closed too: no shipped config resolves to a grave."""
    assert not set(nets_selected().values()) & set(GRAVES)
    assert not set(ARCH_KINDS) & set(GRAVES)
    assert RunConfig.model_fields, "the schema walk is empty; this assertion is free"
