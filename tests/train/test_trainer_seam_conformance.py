"""WPTS Phase T CLASS-KILL — the coordinator↔trainer seam conformance gate (R102 clause ii).

THE CLASS this gate kills: a coordinator-side call site invoking a trainer member the
protocol never declared. TD-1 lived for four WPs exactly because nothing owned this check —
nine test fakes defined `train_step`, production defined none, and no surface existed on
which the difference could red anything (CENSUS_C C-17).

THE SCAN: an AST walk over the coordinator seam sources (`coordinator/step.py`,
`coordinator/dispatch.py`, `train/loop.py`) collecting every attribute access on the trainer
object — `self.trainer.<name>`, `trainer.<name>`, and the `getattr(trainer, "<name>", …)`
string form — asserted ⊆ the members `TrainerLike` declares.

THE SEPARATION (the dispatch's mutation matrix, R86 "alone"): this gate reads ONLY the
Protocol and the caller SOURCES — it never imports or instantiates a trainer. Removing a
protocol declaration therefore reds THIS gate alone (runtime never consults a Protocol);
removing a trainer-side implementation reds the step oracles alone
(`test_train_step_dispatch.py`), because execution — not this scan — is what proves a
member exists.

Scope fence (R102): the coordinator↔trainer seam ONLY. Eval/pool protocol drift is measured
and QUEUED (WPTS ADJ-26) — deliberately not folded in here.
"""
from __future__ import annotations

import ast
import inspect

import mantis.train.coordinator.dispatch as dispatch_mod
import mantis.train.coordinator.step as step_mod
import mantis.train.loop as loop_mod
from mantis.train.coordinator.config import TrainerLike

SEAM_MODULES = (step_mod, dispatch_mod, loop_mod)


# ── the scanner ──────────────────────────────────────────────────────────────────────────
def declared_members(proto: type) -> set[str]:
    """The Protocol's declared surface: annotated attributes + public methods."""
    members = {n for n in getattr(proto, "__annotations__", {}) if not n.startswith("_")}
    members |= {n for n, v in vars(proto).items()
                if callable(v) and not n.startswith("_")}
    return members


def _is_trainer_object(node: ast.expr) -> bool:
    """True for the expressions the seam sources use to hold the trainer:
    `self.trainer` and a bare `trainer` name (ctor/local/parameter)."""
    if isinstance(node, ast.Attribute) and node.attr == "trainer":
        return True
    return isinstance(node, ast.Name) and node.id == "trainer"


def trainer_accesses(source: str) -> set[str]:
    """Every member name the source accesses ON the trainer object."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and _is_trainer_object(node.value):
            names.add(node.attr)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and node.args
              and _is_trainer_object(node.args[0])
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)):
            names.add(node.args[1].value)
    return names


def _all_seam_accesses() -> set[str]:
    out: set[str] = set()
    for mod in SEAM_MODULES:
        out |= trainer_accesses(inspect.getsource(mod))
    return out


# ── the gate ─────────────────────────────────────────────────────────────────────────────
def test_every_coordinator_trainer_call_site_is_declared_on_the_protocol() -> None:
    accessed = _all_seam_accesses()
    declared = declared_members(TrainerLike)
    undeclared = accessed - declared
    assert not undeclared, (
        f"coordinator seam sources access trainer members TrainerLike does not declare: "
        f"{sorted(undeclared)} — declare them on the protocol or remove the call site "
        f"(WPTS R102 class-kill; TD-1 was exactly this class)"
    )


def test_the_seam_is_actually_exercised_by_the_scan() -> None:
    """The scan must SEE the seam it guards — an empty access set would mean the scanner
    lost the sources (a self-satisfying gate, the LAW-07 failure mode)."""
    accessed = _all_seam_accesses()
    for load_bearing in ("train_step_from_tensors", "train_step_from_graph_batch",
                         "save_checkpoint", "step", "device"):
        assert load_bearing in accessed, (
            f"scanner no longer sees {load_bearing!r} — the seam moved or the scanner broke"
        )


def test_the_dead_name_stays_dead() -> None:
    """`train_step` died with TD-1: neither declared on the protocol nor called anywhere on
    the seam. Its reappearance on either side is the card's regression."""
    assert "train_step" not in declared_members(TrainerLike)
    assert "train_step" not in _all_seam_accesses()


# ── mutation self-tests (LAW-07: the gate reds when it must, and is not self-satisfying) ─
def test_scanner_flags_an_undeclared_call_site_constructed_in_a_fixture() -> None:
    fixture_src = (
        "class C:\n"
        "    def f(self):\n"
        "        self.trainer.not_on_the_protocol()\n"
        "        return getattr(trainer, 'also_not_declared', None)\n"
    )
    hits = trainer_accesses(fixture_src)
    assert "not_on_the_protocol" in hits and "also_not_declared" in hits
    assert not hits <= declared_members(TrainerLike), (
        "the gate assertion must go RED on an undeclared call site"
    )


def test_declaration_removal_reds_the_gate() -> None:
    """Doctoring `train_step_from_graph_batch` out of the declared set makes the live seam
    fail the gate — a protocol narrowed under live call sites cannot pass silently."""
    doctored = declared_members(TrainerLike) - {"train_step_from_graph_batch"}
    assert not _all_seam_accesses() <= doctored


def test_the_gate_never_imports_a_trainer_implementation() -> None:
    """The separation arm of the mutation matrix, pinned structurally: this module reads the
    Protocol + sources only, so implementation removal CANNOT red it (the step oracles own
    that direction). Pinned by AST over this module's own imports."""
    with open(__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            names = [a.name for a in node.names]
            assert mod != "mantis.train.trainer.core", "the gate must stay implementation-blind"
            assert "Trainer" not in names, "the gate must stay implementation-blind"
