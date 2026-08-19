"""LAW-18 (Q3 red-team A6) — the parent-death arming lever announces itself IN-RUN.

THE DEFECT. The arming decision existed only as a `logging` line on the run's INHERITED stderr
— which, under a supervisor, IS the supervisor's stderr: the stream that dies with the
supervisor. So after the exact event the arming exists to handle, no artifact anywhere said
whether the run had ever been armed, or why it had not been. LAW-18 requires a lever under test
to log its own fire-rate in-run, and the two sibling watchdogs (`heartbeat_watchdog_armed`,
`selfplay_stall_watchdog_armed`) already set the shape: the arm-log is UNCONDITIONAL, so that a
DISABLED lever is visible rather than silent.

THE CARRIER, and why it is a module latch rather than a signature change. The gate runs at
`main`'s first statement — before the sink, the out-dir and the run id exist — so the decision
has to be carried, not returned. Threading it through `launch_run` would touch the signature
`tests/test_run_launcher.py` drives in-process five times, and that file is FROZEN and HELD.
A process has exactly ONE arming decision and module scope is exactly that lifetime.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
from dataclasses import fields
from pathlib import Path

import pytest

import mantis.run as mantis_run
from mantis.monitor.heartbeat import PARENT_DEATH_PPID_ENV
from mantis.run import _parent_death_event
from mantis.train.lifecycle import signals as signals_mod
from mantis.train.lifecycle.signals import (
    ParentDeathDecision,
    arm_parent_death_if_supervised,
    last_parent_death_decision,
)


def _synthetic(**over) -> ParentDeathDecision:
    base = {
        "armed": False, "reason": "wrapper_chain_too_deep", "supervisor_pid": 4242,
        "ppid": 4243, "chain_depth": 3, "signal_name": None,
    }
    return ParentDeathDecision(**{**base, **over})


def test_an_unsupervised_boot_still_emits_the_arming_event_with_armed_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE LAW-18 ROW. An unsupervised launch takes the `not_supervised` arm and DOES NOT arm —
    and that must be readable, because "this run is orphanable" is exactly the fact an operator
    needs after the run has been orphaned. Driven through the REAL gate: the decision is
    produced by `arm_parent_death_if_supervised`, latched, and read back out."""
    monkeypatch.setattr(signals_mod, "_LAST_DECISION", None, raising=False)
    monkeypatch.delenv(PARENT_DEATH_PPID_ENV, raising=False)

    assert arm_parent_death_if_supervised() is False
    event = _parent_death_event(last_parent_death_decision())
    assert event is not None, (
        "an unsupervised boot emitted NOTHING — a lever that announces itself only when it "
        "fires leaves the orphanable case, the one that matters, invisible"
    )
    assert event["event"] == "parent_death_signal_armed"
    assert event["armed"] is False and event["enabled"] is False
    assert event["reason"] == "not_supervised"
    assert event["ppid"] == os.getppid()


def test_the_wrapper_case_names_its_reason_in_the_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The A4b decision reaches the run's OWN stream, not only stderr. The two refusals are
    different residuals — `wrapper_chain_too_deep` means "this run WILL survive its supervisor",
    `ancestry_unreadable` means "the stamp named nothing we are descended from" — and an event
    that flattened them to `armed=false` would be no better than the log line it replaces.

    (The gate's own production of these reasons is driven as real process chains in
    `tests/monitor/test_arm_exec_trampoline.py` and `tests/test_run_pdeathsig.py`; what this
    row pins is that the reason SURVIVES into the payload.)"""
    for reason, depth in (("wrapper_chain_too_deep", 3), ("ancestry_unreadable", None)):
        monkeypatch.setattr(
            signals_mod, "_LAST_DECISION", _synthetic(reason=reason, chain_depth=depth),
            raising=False,
        )
        event = _parent_death_event(last_parent_death_decision())
        assert event is not None and event["reason"] == reason, event
        assert event["chain_depth"] == depth
        assert event["signal"] is None, "an unarmed decision must name no signal"


def test_the_event_carries_the_same_decision_the_gate_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE ANTI-DRIFT ROW. Every field of the decision must appear in the payload, and the
    field list is DERIVED from the dataclass rather than transcribed here — a transcribed list
    would silently stop covering a field the day one is added (R192(e), derive-or-delete)."""
    decision = _synthetic(armed=True, reason="wrapper_armed_by_trampoline", chain_depth=2,
                          signal_name="SIGKILL")
    monkeypatch.setattr(signals_mod, "_LAST_DECISION", decision, raising=False)
    event = _parent_death_event(last_parent_death_decision())
    assert event is not None

    renamed = {"signal_name": "signal"}
    for field in fields(ParentDeathDecision):
        key = renamed.get(field.name, field.name)
        assert key in event, (
            f"decision field {field.name!r} reaches no event key — the payload is a "
            "hand-assembled copy that has drifted from the record it claims to publish"
        )
        assert event[key] == getattr(decision, field.name), (
            f"event[{key!r}]={event[key]!r} but the gate recorded "
            f"{getattr(decision, field.name)!r}: the emitter is a SECOND authority"
        )
    assert event["enabled"] is decision.armed, (
        "`enabled` is the sibling watchdogs' field name for 'this lever CAN fire' and must "
        "mirror `armed`, never be independently computed"
    )


def test_a_gate_that_records_nothing_leaves_the_event_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE LAW-07 MUTATION SELF-TEST. With no recorded decision the composition must emit
    NOTHING rather than a comfortable `armed=false` — proving the producer is the GATE and the
    event is not manufactured by the emitter.

    A None latch is a real production state, not a contrived one: `launch_run` is entered
    directly (never through `main`) by the five in-process boots in the frozen
    `tests/test_run_launcher.py`, and those runs genuinely have no arming decision."""
    monkeypatch.setattr(signals_mod, "_LAST_DECISION", None, raising=False)
    assert last_parent_death_decision() is None
    assert _parent_death_event(last_parent_death_decision()) is None, (
        "the emitter fabricated an event with no producer behind it — the gate input would "
        "then be a value nothing measured"
    )


def test_the_event_is_emitted_exactly_once_per_segment() -> None:
    """One emit site, inside `compose_run`, fed by the ONE reader. A duplicate emitter would
    turn a decision into a counter, and a second `last_parent_death_decision()` call site would
    be a second authority for reading it."""
    source = inspect.getsource(mantis_run)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "_parent_death_event"]
    assert len(calls) == 1, f"exactly one emit site is allowed; found {len(calls)}"

    compose = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "compose_run")
    inside = [n for n in ast.walk(compose)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "_parent_death_event"]
    assert len(inside) == 1, "the emit must live in the composition root, not somewhere else"


def test_the_arming_event_is_not_a_config_key_and_cannot_be_disabled() -> None:
    """The sibling law, stated as a row: both watchdogs' arm-logs are unconditional BY
    CONSTRUCTION, and so is this one. A mutant that gated the emit on a config flag would put
    the one lever that reports orphanability behind a knob nobody re-checks."""
    fn = ast.parse(inspect.getsource(_parent_death_event)).body[0]
    assert isinstance(fn, ast.FunctionDef)
    names = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(fn) if isinstance(node, ast.Attribute)
    }
    assert not any("config" in name for name in names), (
        f"the arming event must not read config; it is unconditional. Saw {sorted(names)}"
    )
    returns = [node for node in ast.walk(fn) if isinstance(node, ast.Return)]
    assert len(returns) == 2, (
        "exactly two returns — the None-latch guard and the payload. A third would be a "
        f"silent branch where the lever declines to announce itself; found {len(returns)}"
    )


@pytest.mark.integration
def test_a_real_boot_writes_the_arming_event_into_the_runs_own_jsonl(
    tmp_path, smoke_run_config,
) -> None:
    """END TO END, and the row that makes every fast row above production-relevant: a REAL
    composed boot must leave `parent_death_signal_armed` in the run's own event segment on
    disk. That file is the artifact an operator reads after an orphan; a payload that is
    correct in-process and never reaches the segment answers nobody.

    `integration`-marked for the same reason the launcher's own real-boot rows are: it composes
    the whole run. The gate is driven FIRST so the latch is populated, exactly as `main` does."""
    signals_mod._LAST_DECISION = None
    os.environ.pop(PARENT_DEATH_PPID_ENV, None)
    assert arm_parent_death_if_supervised() is False

    config = smoke_run_config("smoke_preflight_armed.yaml", train={"max_train_steps": 16})
    mantis_run.launch_run(config=config, out_dir=tmp_path)

    rows = [json.loads(line)
            for segment in sorted(Path(tmp_path / "logs").glob("events_*.jsonl"))
            for line in segment.read_text(encoding="utf-8").splitlines() if line.strip()]
    armed = [row for row in rows if row["event"] == "parent_death_signal_armed"]
    assert len(armed) == 1, (
        f"exactly one arming record per segment; got {armed} out of "
        f"{sorted({row['event'] for row in rows})}"
    )
    assert armed[0]["armed"] is False and armed[0]["reason"] == "not_supervised", armed[0]
