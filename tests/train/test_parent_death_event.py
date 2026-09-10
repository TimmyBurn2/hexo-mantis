"""The parent-death arming lever announces itself in the run's own event stream.

The decision used to live only on the run's inherited stderr, which under a supervisor is the
supervisor's stderr — gone after exactly the event the arming exists to handle. Like the two
sibling watchdogs, the arm-log is UNCONDITIONAL, so a disabled lever is visible not silent.

The decision is carried by a module latch because the gate runs at `main`'s first statement,
before the sink, the out-dir and the run id exist. A process has exactly one arming decision
and module scope is exactly that lifetime.
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
    """Prove an unsupervised boot still emits the arming event, with armed false.

    "This run is orphanable" is the fact an operator needs after the run has been orphaned.
    """
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
    """Prove each refusal reason survives into the payload rather than flattening to armed=false.

    `wrapper_chain_too_deep` means the run will survive its supervisor; `ancestry_unreadable`
    means the stamp named nothing we descend from. Real process chains drive the gate itself in
    tests/monitor/test_arm_exec_trampoline.py and tests/test_run_pdeathsig.py.
    """
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
    """Prove every decision field reaches the payload, over a field list derived from the dataclass."""
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
    """Prove no recorded decision emits nothing, not a comfortable armed=false.

    A None latch is a real production state: the in-process boots enter `launch_run` directly,
    never through `main`, and genuinely have no arming decision.
    """
    monkeypatch.setattr(signals_mod, "_LAST_DECISION", None, raising=False)
    assert last_parent_death_decision() is None
    assert _parent_death_event(last_parent_death_decision()) is None, (
        "the emitter fabricated an event with no producer behind it — the gate input would "
        "then be a value nothing measured"
    )


def test_the_event_is_emitted_exactly_once_per_segment() -> None:
    """Prove there is exactly one emit site and it lives in the composition root.

    A duplicate emitter would turn a decision into a counter.
    """
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
    """Prove the arming event reads no config and has exactly two returns, so it cannot be disabled."""
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
    """Prove a real composed boot leaves the arming event in the run's own event segment on disk.

    The gate is driven first so the latch is populated, exactly as `main` does.
    """
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
