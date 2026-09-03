"""A collaborator Protocol with no injection point is a seam nobody can enter.

AUDIT-1 F-47, the seam half. `coordinator/config.py` declared `TracemallocLike` (a
`runtime_checkable` Protocol) and `RealTracemalloc` (its concrete), exported both through the
package facade, and had NO parameter anywhere that accepts either — `StepCoordinator.__init__`
takes `clock: ClockLike | None` and nothing else of that shape. `step.py`'s own severance note
says the perf/tracemalloc probes "stay DEFER/ARCH"; the probe never landed and the seam member
outlived it. The conformance suite polices declared members against call sites, so a Protocol
with zero call sites is exactly what it cannot see.

WHAT THIS PINS, AND WHAT IT DOES NOT. It does not say every Protocol must be injected — it says
a Protocol in the coordinator's seam layer must be REACHABLE: named in an injection signature,
or used as an annotation, or declared here as a deliberate exception with grounds. The
deliberate-exception table is asserted for EQUALITY, so a row whose subject is gone reds too.
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

#: Protocols in the seam layer that are deliberately NOT injected, each with its ground.
#: EMPTY today, and that is the point: the one member that was here has been deleted rather than
#: excused. A row added here is a claim someone has to defend at review.
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
    """Every Protocol the conformance suite's `SEAM_MATRIX` actually polices, by name.

    Read off the suite's own source rather than by importing its module-level table, because
    what is being asserted is that the DECLARATION appears in the matrix — a table entry, not a
    runtime value. Structure, not text: the matrix rows are parsed as tuples and the Protocol
    slot is read positionally.
    """
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

    Annotations matter and a runtime census misses them: `from __future__ import annotations`
    makes an annotation a string at runtime, so `clock: ClockLike | None` is invisible to a
    `getattr` sweep while being a real use. The AST sees it either way — which is also how this
    session learned that `bufs: BatchBuffers` keeps `BatchBuffers` alive against a `Name`-load
    census that called it dead.
    """
    tree = ast.parse(inspect.getsource(module))
    own = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name]
    inside = {id(sub) for body in own for sub in ast.walk(body)}
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, ast.Name) and n.id == class_name and id(n) not in inside)


def test_every_seam_protocol_is_either_POLICED_or_INJECTED():
    """The property: a declared seam Protocol is enforced somewhere.

    Two ways to be enforced, and `TracemallocLike` had neither. POLICED = it appears in the
    conformance suite's `SEAM_MATRIX`, so its declared members are checked against real call
    sites. INJECTED = `step.py` names it, i.e. something accepts it. A Protocol that is neither
    is a contract with no party to it.
    """
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
    """The control that makes the census right rather than merely green.

    `ClockLike` is INJECTED and appears only as an annotation (`clock: ClockLike | None`). If
    the census missed annotations it would call the one genuinely-injected Protocol an orphan,
    and the test would be measuring the opposite of what it claims.
    """
    assert sum(_names_outside_own_class(m, "ClockLike") for m in SEAM_CONSUMERS) > 0, (
        "the census cannot see an annotation, so it cannot tell an injected Protocol from a "
        "phantom one"
    )
    assert sum(_names_outside_own_class(m, "AProtocolThatDoesNotExist")
               for m in SEAM_CONSUMERS) == 0


def test_a_protocol_that_is_policed_but_not_injected_is_NOT_an_orphan():
    """The other half of the disjunction, driven on a real member: `TrainerLike` is accepted as
    `trainer: Any` (deliberately — the seam is duck-typed at runtime) and is enforced entirely
    by SEAM_MATRIX. A census that demanded injection would delete the whole seam layer."""
    assert "TrainerLike" in _policed_protocols()
    assert sum(_names_outside_own_class(m, "TrainerLike") for m in SEAM_CONSUMERS) == 0


def test_the_retired_seam_members_stay_retired():
    """The planted break, inverted: re-adding a member must be visible.

    THREE members go: `TracemallocLike`/`RealTracemalloc` (a Protocol/concrete pair for a probe
    `step.py`'s own severance note says "stays DEFER/ARCH", which never landed) and
    `GpuMonitorLike` — which the audit did NOT name and this census found. `gpu_monitor` IS
    threaded through the tree, but `events.py`'s header records the deliberate decision that
    the probe collaborators are duck-typed `Any`; a Protocol for one of them is a second,
    unenforced declaration of the same shape. This asserts all three are gone from the module
    AND the package facade, so a re-add has to come with an injection point or a matrix row
    rather than sliding back in as an export.
    """
    import mantis.train.coordinator as pkg

    for name in ("TracemallocLike", "RealTracemalloc", "GpuMonitorLike"):
        assert not hasattr(config_mod, name), (
            f"{name} is back in the seam layer. It is a Protocol/concrete pair with no "
            "injection point; if the deferred perf probe has landed, wire it — do not re-export it"
        )
        assert name not in getattr(pkg, "__all__", ()), f"{name} is back in the facade's __all__"


def test_the_census_is_not_satisfiable_by_an_empty_module(tmp_path: Path):
    """Vacuity control, driven on a stand-in: a module declaring no Protocol yields an empty
    set, which would satisfy the orphan assertion for free. The real test asserts a FLOOR on
    the declared count, and this records why."""
    module = tmp_path / "empty_seam.py"
    module.write_text("x = 1\n", encoding="utf-8")
    tree = ast.parse(module.read_text(encoding="utf-8"))
    declared = {n.name for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef)
                and any(isinstance(b, ast.Name) and b.id == "Protocol" for b in n.bases)}
    assert declared == set()
    with pytest.raises(AssertionError):
        assert len(declared) > 3
