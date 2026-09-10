"""A collaborator Protocol with no injection point is a seam nobody can enter.

The seam layer declared a `runtime_checkable` Protocol and its concrete, exported both through
the package facade, and had NO parameter anywhere that accepts either: the probe they were
written for never landed, and the conformance suite polices declared members against CALL SITES,
so a Protocol with zero call sites is exactly what it cannot see. The claim is not that every
Protocol must be injected, but that one in this seam layer must be REACHABLE — in an injection
signature, as an annotation, or as a deliberate exception with grounds, a table asserted for
EQUALITY so a row whose subject is gone reds too.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import mantis.train.coordinator.dispatch as dispatch_mod
from mantis.train.coordinator import config as config_mod
from mantis.train.coordinator import drain as drain_mod
from mantis.train.coordinator import step as step_mod

#: The modules that would NAME an injected seam Protocol — the coordinator package's own four.
SEAM_CONSUMERS = (step_mod, config_mod, drain_mod, dispatch_mod)

#: Protocols in the seam layer that are deliberately NOT injected, each with its ground. EMPTY
#: today, and that is the point: a row added here is a claim someone has to defend at review.
DECLARED_UNINJECTED: dict[str, str] = {}


def _protocol_names(module: object) -> set[str]:
    """Every `Protocol` subclass declared in the module's own source."""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Protocol":
                    names.add(node.name)
    return names


def _policed_protocols() -> set[str]:
    """Every Protocol the conformance suite's `SEAM_MATRIX` polices, by name — read off the
    suite's SOURCE rather than its imported table, because what is asserted is that the
    declaration appears in the matrix."""
    suite = Path(__file__).with_name("test_trainer_seam_conformance.py")
    tree = ast.parse(suite.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "SEAM_MATRIX"):
            continue
        for row in getattr(node.value, "elts", []):
            elts = getattr(row, "elts", [])
            if len(elts) >= 3:
                names |= {e.id for e in getattr(elts[2], "elts", []) if isinstance(e, ast.Name)}
    return names


def _names_outside_own_class(module: object, class_name: str) -> int:
    """How many times `class_name` appears as a NAME outside its own `class` statement.
    Annotations count and a runtime census misses them: `from __future__ import annotations`
    makes `clock: ClockLike | None` a string, invisible to a `getattr` sweep but a real use."""
    tree = ast.parse(inspect.getsource(module))
    own = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name]
    inside = {id(sub) for body in own for sub in ast.walk(body)}
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Name) and n.id == class_name and id(n) not in inside)


def test_every_seam_protocol_is_either_POLICED_or_INJECTED():
    """A declared seam Protocol must be enforced somewhere: POLICED, appearing in the conformance
    suite's `SEAM_MATRIX` so its members are checked against real call sites, or INJECTED,
    meaning something accepts it. Neither is a contract with no party to it."""
    declared = _protocol_names(config_mod)
    assert len(declared) > 3, (
        f"the seam layer declares only {sorted(declared)} — this census would be near-vacuous"
    )
    policed = _policed_protocols()
    assert len(policed) > 3, (
        f"only {sorted(policed)} parsed out of SEAM_MATRIX — the parse broke and this census "
        "would call every Protocol an orphan"
    )
    orphans = {
        name for name in declared
        if name not in policed
        and sum(_names_outside_own_class(m, name) for m in SEAM_CONSUMERS) == 0
    } - set(DECLARED_UNINJECTED)
    assert orphans == set(), (
        f"seam Protocol(s) neither policed by SEAM_MATRIX nor named by the coordinator: "
        f"{sorted(orphans)}. A Protocol nothing accepts and nothing checks is a seam nobody can "
        "enter — the conformance suite reads declared members against CALL SITES, so it is "
        "blind to a member with zero call sites (AUDIT-1 F-47). Wire it, delete it, or declare "
        "it in DECLARED_UNINJECTED with grounds."
    )
    stale = set(DECLARED_UNINJECTED) - declared
    assert stale == set(), (
        f"DECLARED_UNINJECTED names {sorted(stale)}, which the seam layer no longer declares — "
        "an exemption that outlived its subject"
    )


def test_the_census_counts_an_ANNOTATION_as_a_use():
    """The control that makes the census right rather than merely green: `ClockLike` is INJECTED
    and appears only as an annotation, so a census blind to annotations would call the one
    genuinely-injected Protocol an orphan."""
    assert sum(_names_outside_own_class(m, "ClockLike") for m in SEAM_CONSUMERS) > 0, (
        "the census cannot see an annotation, so it cannot tell an injected Protocol from a "
        "phantom one"
    )
    assert sum(_names_outside_own_class(m, "AProtocolThatDoesNotExist")
               for m in SEAM_CONSUMERS) == 0


def test_a_protocol_that_is_policed_but_not_injected_is_NOT_an_orphan():
    """The other half of the disjunction on a real member: `TrainerLike` is accepted as
    `trainer: Any` — the seam is duck-typed at runtime — and enforced entirely by SEAM_MATRIX,
    so a census that demanded injection would delete the whole seam layer."""
    assert "TrainerLike" in _policed_protocols()
    assert sum(_names_outside_own_class(m, "TrainerLike") for m in SEAM_CONSUMERS) == 0


def test_the_retired_seam_members_stay_retired():
    """The planted break, inverted: re-adding a retired member must be visible. Module AND
    package facade are both asserted, so a re-add must bring an injection point or a matrix row
    rather than slide back in as an export."""
    import mantis.train.coordinator as pkg

    for name in ("TracemallocLike", "RealTracemalloc", "GpuMonitorLike"):
        assert not hasattr(config_mod, name), (
            f"{name} is back in the seam layer. It is a Protocol/concrete pair with no "
            "injection point; if the deferred perf probe has landed, wire it — do not re-export it"
        )
        assert name not in getattr(pkg, "__all__", ()), f"{name} is back in the facade's __all__"


def test_the_census_is_not_satisfiable_by_an_empty_module(tmp_path: Path):
    """Vacuity control: a module declaring no Protocol yields an empty set, which would satisfy
    the orphan assertion for free — hence the FLOOR on the declared count."""
    module = tmp_path / "empty_seam.py"
    module.write_text("x = 1\n", encoding="utf-8")
    tree = ast.parse(module.read_text(encoding="utf-8"))
    declared = {n.name for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef)
                and any(isinstance(b, ast.Name) and b.id == "Protocol" for b in n.bases)}
    assert declared == set()
    with pytest.raises(AssertionError):
        assert len(declared) > 3
