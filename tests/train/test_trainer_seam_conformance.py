"""The coordinator collaborator-seam conformance gate (WPTS Phase T, widened WPCLEAN Phase PC).

THE CLASS this gate kills: a seam-side call site invoking a collaborator member its protocol
never declared. TD-1 lived for four WPs exactly because nothing owned this check — nine test
fakes defined `train_step`, production defined none, and no surface existed on which the
difference could red anything (CENSUS_C C-17). Phase T built the check for the trainer seam
(R102 clause ii); R106 ruled the remaining called-and-undeclared families (CENSUS_C
C-6/C-7/C-10/C-11/C-16 + the Phase-PC re-census rows) the same class at zero runtime effect,
and this widening is its card: the SEAM_MATRIX below asserts, per (module, holder), that
every attribute access and getattr-string access on that holder is ⊆ the union of its
protocols' declared members.

THE SEPARATION (the mutation matrix, R86 "alone"): this gate reads ONLY Protocols and caller
SOURCES — it never imports or instantiates a concrete collaborator. Removing a protocol
declaration reds THIS gate alone (runtime never consults a Protocol); removing an
implementation member reds the behavioural oracles alone, because execution — not this
scan — is what proves a member exists.

Non-protocol holders (actor_sync, heartbeat_watchdog, subsystems) are deliberately absent:
CENSUS_C ruled them no-gap with their own pinned postures (C-4, C-12).
"""
from __future__ import annotations

import ast
import inspect

import mantis.train.buffer_persist as persist_mod
import mantis.train.coordinator.dispatch as dispatch_mod
import mantis.train.coordinator.step as step_mod
import mantis.train.events as events_mod
import mantis.train.loop as loop_mod
from mantis.train.coordinator import drain as drain_mod
from mantis.train.coordinator.config import (
    ClockLike,
    EvalPipelineLike,
    GraphRouteBufferLike,
    GridRouteBufferLike,
    RecentBufferLike,
    ReplayBufferLike,
    TrainerLike,
    WorkerPoolLike,
)
from mantis.train.events import PoolTelemetryLike


# ── the scanner ──────────────────────────────────────────────────────────────────────────
def declared_members(*protos: type) -> set[str]:
    """The declared surface of a protocol union: annotated attributes + public methods."""
    members: set[str] = set()
    for proto in protos:
        members |= {n for n in getattr(proto, "__annotations__", {}) if not n.startswith("_")}
        members |= {n for n, v in vars(proto).items()
                    if callable(v) and not n.startswith("_")}
    return members


def _holds(node: ast.expr, aliases: tuple[str, ...]) -> bool:
    """True when the expression denotes one of the holder aliases: a bare name (`pool`) or
    the attribute form any owner spells it with (`self.pool`, `coord.eval_pipeline`)."""
    if isinstance(node, ast.Attribute) and node.attr in aliases:
        return True
    return isinstance(node, ast.Name) and node.id in aliases


def holder_accesses(source: str, aliases: tuple[str, ...]) -> set[str]:
    """Every member name the source reaches ON the holder — attribute form AND the
    getattr-string form (drain.py reaches `drain_pending`/`apply_gate_decision` ONLY via
    getattr, so an attribute-only scan would self-satisfy there)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and _holds(node.value, aliases):
            names.add(node.attr)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and node.args
              and _holds(node.args[0], aliases)
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)):
            names.add(node.args[1].value)
    return names


def trainer_accesses(source: str) -> set[str]:
    return holder_accesses(source, ("trainer",))


# ── the seam matrix ──────────────────────────────────────────────────────────────────────
# (module, holder aliases, protocol union, sentinels the scanner MUST see there).
# Sentinels are the anti-self-satisfying arm (LAW-07): an empty or shrunken access set
# means the scanner lost the seam, and that must red, not pass.
SEAM_MATRIX: tuple[tuple[object, tuple[str, ...], tuple[type, ...], tuple[str, ...]], ...] = (
    (step_mod, ("trainer",), (TrainerLike,),
     ("train_step_from_tensors", "save_checkpoint", "step")),
    (dispatch_mod, ("trainer",), (TrainerLike,),
     ("train_step_from_graph_batch", "train_step_from_tensors", "device")),
    (loop_mod, ("trainer",), (TrainerLike,), ("save_checkpoint",)),
    (step_mod, ("pool",), (WorkerPoolLike,),
     ("games_completed", "check_producer_health", "pooled_draw_counts")),
    (step_mod, ("eval_pipeline",), (EvalPipelineLike,), ("poll_completed", "run_evaluation")),
    (drain_mod, ("eval_pipeline", "pipeline"), (EvalPipelineLike,),
     ("drain_pending", "apply_gate_decision", "run_evaluation")),
    (step_mod, ("buffer", "pretrained_buffer", "bot_buffer"), (ReplayBufferLike,),
     ("resize", "save_to_path", "size")),
    (dispatch_mod, ("buffer",), (ReplayBufferLike, GraphRouteBufferLike, GridRouteBufferLike),
     ("sample_graph_batch", "sample_batch_with_pos")),
    (dispatch_mod, ("recent_buffer",), (RecentBufferLike,), ("sample", "size")),
    (persist_mod, ("buffer",), (ReplayBufferLike,), ("save_to_path",)),
    (persist_mod, ("recent_buffer",), (RecentBufferLike,), ("save_to_path", "size")),
    (step_mod, ("_clock",), (ClockLike,), ("now", "sleep")),
    (events_mod, ("pool",), (PoolTelemetryLike,),
     ("recent_move_histories", "runner_stats", "x_winrate", "batch_fill_pct")),
    (events_mod, ("buffer",), (ReplayBufferLike,), ("size", "capacity")),
)


def _row_accesses(mod: object, aliases: tuple[str, ...]) -> set[str]:
    return holder_accesses(inspect.getsource(mod), aliases)


# ── the gate ─────────────────────────────────────────────────────────────────────────────
def test_every_seam_call_site_is_declared_on_its_protocol() -> None:
    failures: list[str] = []
    for mod, aliases, protos, _ in SEAM_MATRIX:
        undeclared = _row_accesses(mod, aliases) - declared_members(*protos)
        if undeclared:
            failures.append(
                f"{mod.__name__} holder {aliases}: accesses undeclared on "
                f"{'/'.join(p.__name__ for p in protos)}: {sorted(undeclared)}"
            )
    assert not failures, (
        "seam sources access collaborator members no protocol declares — declare them or "
        "remove the call site (R106 / WPTS R102 class-kill; TD-1 was exactly this class):\n"
        + "\n".join(failures)
    )


def test_every_seam_is_actually_exercised_by_the_scan() -> None:
    """The scan must SEE each seam it guards (the LAW-07 self-satisfying failure mode)."""
    for mod, aliases, _, sentinels in SEAM_MATRIX:
        accessed = _row_accesses(mod, aliases)
        for load_bearing in sentinels:
            assert load_bearing in accessed, (
                f"scanner no longer sees {load_bearing!r} on {aliases} in {mod.__name__} — "
                "the seam moved or the scanner broke"
            )


def test_the_dead_name_stays_dead() -> None:
    """`train_step` died with TD-1: neither declared on the protocol nor called anywhere on
    the seam. Its reappearance on either side is the card's regression."""
    assert "train_step" not in declared_members(TrainerLike)
    for mod in (step_mod, dispatch_mod, loop_mod):
        assert "train_step" not in _row_accesses(mod, ("trainer",))


# ── mutation self-tests (LAW-07: the gate reds when it must, and is not self-satisfying) ─
def test_scanner_flags_an_undeclared_call_site_constructed_in_a_fixture() -> None:
    fixture_src = (
        "class C:\n"
        "    def f(self):\n"
        "        self.trainer.not_on_the_protocol()\n"
        "        self.pool.invented_stat\n"
        "        return getattr(pipeline, 'also_not_declared', None)\n"
    )
    assert not holder_accesses(fixture_src, ("trainer",)) <= declared_members(TrainerLike)
    assert not holder_accesses(fixture_src, ("pool",)) <= declared_members(PoolTelemetryLike)
    assert not holder_accesses(fixture_src, ("pipeline",)) <= declared_members(EvalPipelineLike)


def test_declaration_removal_reds_the_gate() -> None:
    """Doctoring a declaration out of each seam's declared set makes the LIVE sources fail —
    a protocol narrowed under live call sites cannot pass silently. One arm per widened
    family (trainer / eval / telemetry / recent-buffer)."""
    assert not _row_accesses(step_mod, ("trainer",)) <= (
        declared_members(TrainerLike) - {"train_step_from_tensors"})
    assert not _row_accesses(step_mod, ("eval_pipeline",)) <= (
        declared_members(EvalPipelineLike) - {"poll_completed"})
    assert not _row_accesses(drain_mod, ("eval_pipeline", "pipeline")) <= (
        declared_members(EvalPipelineLike) - {"drain_pending"})
    assert not _row_accesses(events_mod, ("pool",)) <= (
        declared_members(PoolTelemetryLike) - {"recent_move_histories"})
    assert not _row_accesses(dispatch_mod, ("recent_buffer",)) <= (
        declared_members(RecentBufferLike) - {"size"})


def test_the_gate_never_imports_a_collaborator_implementation() -> None:
    """The separation arm of the mutation matrix, pinned structurally: this module reads
    Protocols + sources only, so implementation removal CANNOT red it (the behavioural
    oracles own that direction). Pinned by AST over this module's own imports."""
    forbidden_modules = {
        "mantis.train.trainer.core", "mantis.selfplay.pool",
        "mantis.eval.pipeline", "mantis.train.recency_buffer",
    }
    with open(__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            names = [a.name for a in node.names]
            assert mod not in forbidden_modules, "the gate must stay implementation-blind"
            assert not (set(names) & forbidden_modules), "the gate must stay implementation-blind"
            assert "Trainer" not in names, "the gate must stay implementation-blind"
