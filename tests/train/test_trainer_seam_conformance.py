"""Gate the coordinator's collaborator seams: per (module, holder), every attribute and
getattr-string access on that holder is a subset of its protocols' declared members.

The gate reads ONLY Protocols and caller SOURCES, never a concrete collaborator, so removing a
declaration reds this gate alone and removing an implementation reds the behavioural oracles.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

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


def declared_members(*protos: type) -> set[str]:
    """Return a protocol union's declared surface: annotated attributes and public methods."""
    members: set[str] = set()
    for proto in protos:
        members |= {n for n in getattr(proto, "__annotations__", {}) if not n.startswith("_")}
        members |= {n for n, v in vars(proto).items()
                    if callable(v) and not n.startswith("_")}
    return members


def _holds(node: ast.expr, aliases: tuple[str, ...]) -> bool:
    """Report whether the expression denotes a holder alias, bare or attribute-form."""
    if isinstance(node, ast.Attribute) and node.attr in aliases:
        return True
    return isinstance(node, ast.Name) and node.id in aliases


def holder_accesses(source: str, aliases: tuple[str, ...]) -> set[str]:
    """Return every member name the source reaches on the holder, in both forms: one seam reaches
    its members ONLY via `getattr`, so an attribute-only scan would self-satisfy there."""
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


# (module, holder aliases, protocol union, sentinels the scanner MUST see there). The sentinels
# are the anti-self-satisfying arm: a shrunken access set means the scanner lost the seam.
SEAM_MATRIX: tuple[tuple[object, tuple[str, ...], tuple[type, ...], tuple[str, ...]], ...] = (
    # `step.py` reaches the dispatcher and names no typed entry point of its own.
    (step_mod, ("trainer",), (TrainerLike,), ("save_checkpoint", "step")),
    (dispatch_mod, ("trainer",), (TrainerLike,),
     ("train_step_from_graph_batch", "device")),
    (loop_mod, ("trainer",), (TrainerLike,), ("save_checkpoint",)),
    (step_mod, ("pool",), (WorkerPoolLike,),
     ("games_completed", "check_producer_health", "pooled_draw_counts")),
    (step_mod, ("eval_pipeline",), (EvalPipelineLike,), ("poll_completed", "run_evaluation")),
    (drain_mod, ("eval_pipeline", "pipeline"), (EvalPipelineLike,),
     ("drain_pending", "apply_gate_decision", "run_evaluation")),
    (step_mod, ("buffer", "pretrained_buffer", "bot_buffer"), (ReplayBufferLike,),
     ("resize", "save_to_path", "size")),
    # The graph arm flows recency in-engine, so the only `recent_buffer` access left in
    # `dispatch` is the refusal that names it.
    (dispatch_mod, ("buffer",), (ReplayBufferLike, GraphRouteBufferLike, GridRouteBufferLike),
     ("sample_graph_batch",)),
    (persist_mod, ("buffer",), (ReplayBufferLike,), ("save_to_path",)),
    (persist_mod, ("recent_buffer",), (RecentBufferLike,), ("save_to_path", "size")),
    (step_mod, ("_clock",), (ClockLike,), ("now", "sleep")),
    # `runner_stats` is deliberately NOT a sentinel here: the snapshot is passed INTO the builder
    # so it makes no call of its own. The inverse assertion is the last test in this file.
    (events_mod, ("pool",), (PoolTelemetryLike,),
     ("recent_move_histories", "x_winrate", "batch_fill_pct")),
    (events_mod, ("buffer",), (ReplayBufferLike,), ("size", "capacity")),
)


def _row_accesses(mod: object, aliases: tuple[str, ...]) -> set[str]:
    return holder_accesses(inspect.getsource(mod), aliases)


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
    """Prove the scan actually sees each seam it guards."""
    for mod, aliases, _, sentinels in SEAM_MATRIX:
        accessed = _row_accesses(mod, aliases)
        for load_bearing in sentinels:
            assert load_bearing in accessed, (
                f"scanner no longer sees {load_bearing!r} on {aliases} in {mod.__name__} — "
                "the seam moved or the scanner broke"
            )


def test_the_dead_name_stays_dead() -> None:
    """Prove the dead name is neither declared on the protocol nor called on the seam."""
    assert "train_step" not in declared_members(TrainerLike)
    for mod in (step_mod, dispatch_mod, loop_mod):
        assert "train_step" not in _row_accesses(mod, ("trainer",))


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
    """Prove removing a declaration reds the gate against the LIVE sources, one arm per family."""
    assert not _row_accesses(step_mod, ("trainer",)) <= (
        declared_members(TrainerLike) - {"save_checkpoint"})
    assert not _row_accesses(step_mod, ("eval_pipeline",)) <= (
        declared_members(EvalPipelineLike) - {"poll_completed"})
    assert not _row_accesses(drain_mod, ("eval_pipeline", "pipeline")) <= (
        declared_members(EvalPipelineLike) - {"drain_pending"})
    assert not _row_accesses(events_mod, ("pool",)) <= (
        declared_members(PoolTelemetryLike) - {"recent_move_histories"})
    assert not _row_accesses(persist_mod, ("recent_buffer",)) <= (
        declared_members(RecentBufferLike) - {"size"})


def test_the_gate_never_imports_a_collaborator_implementation() -> None:
    """Prove the gate imports no collaborator implementation, so removing one cannot red it."""
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


def test_the_iteration_complete_builder_makes_NO_pool_runner_stats_call() -> None:
    """Prove the iteration-complete builder makes no `pool.runner_stats()` call of its own: the
    snapshot is passed in so the reads cannot straddle a game boundary."""
    # Over the AST's CALL nodes, not the raw source: the builder's own docstring names
    # `pool.runner_stats()` to say it does not call it, and a text census cannot tell them apart.
    body = ast.parse(textwrap.dedent(
        inspect.getsource(events_mod.emit_iteration_complete_event)))
    calls = [n for n in ast.walk(body)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "runner_stats"]
    assert calls == [], (
        "`emit_iteration_complete_event` calls `pool.runner_stats()` again. R218 rider 1: the "
        "snapshot is PASSED IN (`rstats`) precisely so the two reads cannot straddle a game "
        "boundary; a second read inside the builder re-introduces the straddle with every test "
        "still green"
    )
    assert "rstats" in inspect.signature(events_mod.emit_iteration_complete_event).parameters, (
        "the builder no longer takes `rstats`, so the collapse it is asserted against is gone"
    )
