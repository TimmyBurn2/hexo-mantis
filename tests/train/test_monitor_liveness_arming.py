"""The arming half of the disk-guard audit — SHAPE A plus the `poll_once` last-emit age.

Gate 12 reads a config number, so a guard whose `check_once` raised on every tick still audited
ARMED: a threshold nobody reads is indistinguishable from one being read. SHAPE A gives the
arming predicate a PRODUCER-LIVENESS operand while leaving gate 12 untouched, since with no
probe answer the new mechanism is byte-for-byte the old one. SHAPE B was REJECTED in its form,
and the rejection is asserted rather than described: a guard made a fifth heartbeat source sits
on an instrument whose stall code is 42, the class the supervisor RELAUNCHES on.

THE PLANTED DEAD-GUARD BREAK is a real `DiskGuard` whose every tick raises, and the contrast is
the finding: config-only says ARMED, live says DISARMED, same config, same process.

R8 >300 justify: ONE unit, because the finding IS that comparison — split, a reader could see
either verdict without the one that makes it mean something.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from mantis.config.armed_aborts import (
    DISK_GUARD_LIVENESS_PROBE,
    DISK_SPACE_ABORT_RULE,
    MANIFEST,
    ArmedAbort,
    Cadence,
    Mechanism,
    ProducerProbeMissingError,
    Status,
    audit_arming,
    audit_arming_live,
)
from mantis.config.loader import load_config
from mantis.monitor.heartbeat import HeartbeatRegistry
from mantis.train.lifecycle.disk_guard import DiskGuard
from mantis.train.lifecycle.heartbeat_watchdog import (
    MONITOR_STALL_INTERVALS,
    HeartbeatWatchdog,
    MonitorLivenessSpec,
    MonitorSample,
)

_PRODUCTION = "configs/run6.yaml"


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


def _disk_row() -> ArmedAbort:
    rows = [r for r in MANIFEST if r.name == DISK_SPACE_ABORT_RULE]
    assert len(rows) == 1, "the disk row must appear exactly once in the manifest"
    return rows[0]




def test_the_disk_row_carries_the_live_producer_mechanism_and_the_exported_probe_name() -> None:
    """Structure, not text: the row's probe IS the exported constant object, so a rename of
    either side cannot leave the two agreeing by coincidence."""
    row = _disk_row()
    assert row.mechanism is Mechanism.CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER
    assert row.producer_probe == DISK_GUARD_LIVENESS_PROBE
    assert row.status is Status.REQUIRED
    assert row.cadence is Cadence.WALL_CLOCK_POLL


@pytest.mark.parametrize(
    "value",
    [5.0, 1, 0.0, 0, -1.0, None, True, False, "5.0", float("inf"), float("nan")],
)
def test_GATE_12_IS_UNTOUCHED_the_config_only_verdict_is_identical_to_the_old_mechanism(
    value: Any,
) -> None:
    """The load-bearing claim of shape A, proven over a value battery rather than asserted: with
    NO probe answer the new mechanism must answer exactly what `CONFIG_THRESHOLD_GT_ZERO`
    answers, which is what keeps gate 12 byte-unchanged. A single divergence would move a
    verdict on a committed config without anyone deciding to."""
    old = Mechanism.CONFIG_THRESHOLD_GT_ZERO.is_armed(value)
    new = Mechanism.CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER.is_armed(value)
    assert old == new, f"the two mechanisms disagree on {value!r} with no probe supplied"


def test_the_live_operand_only_ever_TIGHTENS_the_verdict() -> None:
    """A probe cannot arm a row the config disarms. Otherwise a live producer would excuse an
    unminted threshold, which is the arming question pointing backwards."""
    mech = Mechanism.CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER
    assert mech.is_armed(5.0, producer_live=True) is True
    assert mech.is_armed(5.0, producer_live=False) is False
    assert mech.is_armed(0.0, producer_live=True) is False
    assert mech.is_armed(0.0, producer_live=False) is False


def test_a_live_producer_row_with_no_probe_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="names no `producer_probe`"):
        ArmedAbort(
            name="x", config_path="monitor.disk_guard.fail_gb",
            mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO_WITH_LIVE_PRODUCER,
            status=Status.REQUIRED, exit_code=None, owner=None, source_pin=None, note="",
        )


def test_a_probe_on_a_mechanism_that_ignores_it_is_refused_at_construction() -> None:
    """The other direction, and it is the phantom-input class: a probe nothing reads is a
    claim the audit does not make."""
    with pytest.raises(ValueError, match="ignores"):
        ArmedAbort(
            name="x", config_path="monitor.disk_guard.fail_gb",
            mechanism=Mechanism.CONFIG_THRESHOLD_GT_ZERO,
            status=Status.REQUIRED, exit_code=None, owner=None, source_pin=None, note="",
            producer_probe="whatever",
        )


def test_an_unanswerable_probe_RAISES_rather_than_assuming_either_answer() -> None:
    config = load_config(_PRODUCTION)
    with pytest.raises(ProducerProbeMissingError) as exc:
        audit_arming_live(config, probes={})
    assert DISK_GUARD_LIVENESS_PROBE in str(exc.value)
    assert DISK_SPACE_ABORT_RULE in str(exc.value)


def test_the_live_audit_agrees_with_the_config_audit_when_the_producer_is_live() -> None:
    config = load_config(_PRODUCTION)
    static = audit_arming(config)
    live = audit_arming_live(config, probes={DISK_GUARD_LIVENESS_PROBE: lambda: True})
    assert [r.name for r in live.disarmed] == [r.name for r in static.disarmed]
    assert [r.name for r in live.required] == [r.name for r in static.required]


def test_the_live_audit_DIVERGES_from_the_config_audit_when_the_producer_is_dead() -> None:
    """The whole point of the mechanism, on a real committed config."""
    config = load_config(_PRODUCTION)
    static = audit_arming(config)
    live = audit_arming_live(config, probes={DISK_GUARD_LIVENESS_PROBE: lambda: False})
    assert DISK_SPACE_ABORT_RULE not in [r.name for r in static.disarmed]
    assert DISK_SPACE_ABORT_RULE in [r.name for r in live.disarmed]




def _guard(tmp_path: Path, sink: _Sink, *, interval: float = 0.01) -> DiskGuard:
    return DiskGuard(watch_path=tmp_path, interval_sec=interval, warn_gb=0.0, fail_gb=0.0,
                     keep_all=True, sink=sink)


def _drive_loop(guard: DiskGuard, *, until: int, attr: str, timeout: float = 5.0) -> None:
    """Run the guard's REAL loop thread until `attr` reaches `until`, then stop it."""
    guard.start()
    deadline = threading.Event()
    waited = 0.0
    while getattr(guard, attr) < until and waited < timeout:
        deadline.wait(0.01)
        waited += 0.01
    guard.stop()


def test_THE_PLANTED_BREAK_a_guard_whose_every_tick_raises_reads_ARMED_statically_and_DISARMED_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE producer test (LAW-07), with the break planted in the producer rather than the
    predicate: a real `DiskGuard`, its real loop and counters, against the real manifest and a
    real committed config. The config-only audit reports ARMED and is not wrong to — the
    threshold is minted and positive — while the live audit reports DISARMED, because nothing
    is reading it."""
    import shutil as _shutil

    def _boom(_path: Any) -> Any:
        raise OSError("planted break: the volume is unreadable")

    monkeypatch.setattr(_shutil, "disk_usage", _boom)
    sink = _Sink()
    guard = _guard(tmp_path, sink)
    _drive_loop(guard, until=2, attr="errors_total")

    assert guard.errors_total >= 2, "the planted break must actually raise on every tick"
    assert guard.checks_total == 0, "no tick completed, so nothing was measured"
    assert sink.named("disk_guard_error"), "REPAIR-1's channel must still carry the failure"
    assert not sink.named("disk_free"), "a dead guard publishes no free-space reading"

    config = load_config(_PRODUCTION)
    probes = {DISK_GUARD_LIVENESS_PROBE: lambda: guard.checks_total > 0}
    assert DISK_SPACE_ABORT_RULE not in [r.name for r in audit_arming(config).disarmed]
    assert DISK_SPACE_ABORT_RULE in [
        r.name for r in audit_arming_live(config, probes=probes).disarmed
    ]


def test_THE_CONTROL_a_healthy_guard_reads_ARMED_on_BOTH_audits(tmp_path: Path) -> None:
    """The break must be removable. Without this row the test above is satisfied by an audit
    that reports DISARMED unconditionally."""
    sink = _Sink()
    guard = _guard(tmp_path, sink)
    _drive_loop(guard, until=1, attr="checks_total")

    assert guard.checks_total >= 1 and guard.errors_total == 0
    config = load_config(_PRODUCTION)
    probes = {DISK_GUARD_LIVENESS_PROBE: lambda: guard.checks_total > 0}
    assert DISK_SPACE_ABORT_RULE not in [r.name for r in audit_arming(config).disarmed]
    assert DISK_SPACE_ABORT_RULE not in [
        r.name for r in audit_arming_live(config, probes=probes).disarmed
    ]


def test_every_committed_production_config_still_passes_the_live_audit_with_a_live_producer(
) -> None:
    """Shape A must not newly refuse a mint. Every REQUIRED row stays armed on every config
    the manifest lists, given a producer that has run."""
    from mantis.config.armed_aborts import PRODUCTION_CONFIGS

    for rel in PRODUCTION_CONFIGS:
        config = load_config(rel)
        live = audit_arming_live(config, probes={DISK_GUARD_LIVENESS_PROBE: lambda: True})
        assert not live.disarmed, f"{rel} newly reports {[r.name for r in live.disarmed]}"




class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _watchdog(tmp_path: Path, sink: _Sink, clock: _Clock,
              sample_fn: Any, exits: list[int]) -> HeartbeatWatchdog:
    """A REAL `HeartbeatRegistry` on the same fake clock, one source, deadline `0.0`. Not a stub,
    because the staleness branch runs on every poll beside the liveness check; `0.0` is the
    watchdog's own "this source cannot fire" spelling, isolating the axis under test without
    disabling the one beside it."""
    return HeartbeatWatchdog(
        registry=HeartbeatRegistry(sources=("train_step",), clock=clock),
        deadlines={"train_step": 0.0}, sink=sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / "hb.json", file_interval_sec=1.0, poll_interval_sec=0.1,
        clock=clock, save_snapshot=lambda: None, exit_fn=exits.append,
        monitor_liveness=(MonitorLivenessSpec(name="disk_guard", sample_fn=sample_fn),),
    )


def test_a_monitor_that_does_not_exist_yet_is_SILENT_not_stalled(
    tmp_path: Path,
) -> None:
    """The composition root starts the watchdog BEFORE it builds the guard, so `None` is a
    real state. Treating it as a stall would make every run report one at boot."""
    sink, clock, exits = _Sink(), _Clock(), []
    wd = _watchdog(tmp_path, sink, clock, lambda: None, exits)
    for _ in range(5):
        clock.t += 100.0
        wd.poll_once()
    assert not sink.named("monitor_stalled")
    assert not sink.named("monitor_liveness_sample")


def test_a_LIVE_monitor_logs_its_own_reading_in_run_and_never_stalls(tmp_path: Path) -> None:
    """LAW-18: a lever under test logs its reading on a HEALTHY run too, or no observer can
    tell a live reading from a frozen one."""
    sink, clock, exits = _Sink(), _Clock(), []
    checks = [0]

    def sample() -> MonitorSample:
        checks[0] += 1
        return MonitorSample(checks_total=checks[0], errors_total=0, interval_sec=60.0)

    wd = _watchdog(tmp_path, sink, clock, sample, exits)
    for _ in range(4):
        clock.t += 60.0
        wd.poll_once()
    assert not sink.named("monitor_stalled")
    samples = sink.named("monitor_liveness_sample")
    assert samples, "a healthy monitor must still publish its reading"
    assert samples[-1]["monitors"][0]["monitor"] == "disk_guard"
    assert samples[-1]["monitors"][0]["checks_total"] == checks[0]


def test_a_FROZEN_counter_stalls_after_its_own_intervals_and_the_event_is_LATCHED(
    tmp_path: Path,
) -> None:
    sink, clock, exits = _Sink(), _Clock(), []
    frozen = MonitorSample(checks_total=7, errors_total=3, interval_sec=60.0)
    wd = _watchdog(tmp_path, sink, clock, lambda: frozen, exits)

    wd.poll_once()                       # first sighting: the age clock starts here
    clock.t += MONITOR_STALL_INTERVALS * 60.0 - 1.0
    wd.poll_once()
    assert not sink.named("monitor_stalled"), "inside its own deadline it is slow, not dead"

    clock.t += 2.0
    wd.poll_once()
    fired = sink.named("monitor_stalled")
    assert len(fired) == 1
    assert fired[0]["monitor"] == "disk_guard"
    assert fired[0]["errors_total"] == 3
    assert fired[0]["stall_intervals"] == MONITOR_STALL_INTERVALS

    for _ in range(5):
        clock.t += 600.0
        wd.poll_once()
    assert len(sink.named("monitor_stalled")) == 1, "one outage is ONE event, not a flood"


def test_a_RECOVERED_monitor_says_so_and_can_stall_again(tmp_path: Path) -> None:
    """Without the recovery arm the latch would silence a second, real outage."""
    sink, clock, exits = _Sink(), _Clock(), []
    state = {"checks": 7}

    def sample() -> MonitorSample:
        return MonitorSample(checks_total=state["checks"], errors_total=0, interval_sec=60.0)

    wd = _watchdog(tmp_path, sink, clock, sample, exits)
    wd.poll_once()
    clock.t += MONITOR_STALL_INTERVALS * 60.0 + 1.0
    wd.poll_once()
    assert len(sink.named("monitor_stalled")) == 1

    state["checks"] = 8
    clock.t += 1.0
    wd.poll_once()
    assert len(sink.named("monitor_recovered")) == 1

    clock.t += MONITOR_STALL_INTERVALS * 60.0 + 1.0
    wd.poll_once()
    assert len(sink.named("monitor_stalled")) == 2, "a second outage must be visible"


def test_a_stalled_monitor_NEVER_EXITS_THE_PROCESS(tmp_path: Path) -> None:
    """The rejection of shape B, held as an assertion rather than a comment. The stall
    watchdog's own code is 42 — the TRANSIENT class the supervisor RELAUNCHES on — so a disk
    guard raising every tick that could reach `exit_fn` would stall-abort and be relaunched into
    the same broken state: a crash loop into a filling volume."""
    sink, clock, exits = _Sink(), _Clock(), []
    frozen = MonitorSample(checks_total=1, errors_total=999, interval_sec=1.0)
    wd = _watchdog(tmp_path, sink, clock, lambda: frozen, exits)
    for _ in range(20):
        clock.t += 100.0
        wd.poll_once()
    assert sink.named("monitor_stalled"), "the observable must have fired"
    assert exits == [], "a monitor stall must never reach exit_fn"


def test_the_liveness_wiring_is_named_at_ARM_TIME_in_both_directions(tmp_path: Path) -> None:
    """An unwired monitor is a gap somebody must be able to see, exactly as an unwired
    heartbeat source is."""
    sink, clock, exits = _Sink(), _Clock(), []
    _watchdog(tmp_path, sink, clock, lambda: None, exits).arm()
    assert sink.named("heartbeat_watchdog_armed")[0]["monitor_liveness"] == ["disk_guard"]

    bare_sink = _Sink()
    HeartbeatWatchdog(
        registry=HeartbeatRegistry(sources=("train_step",), clock=_Clock()),
        deadlines={"train_step": 0.0}, sink=bare_sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / "hb2.json", file_interval_sec=1.0, poll_interval_sec=0.1,
        clock=_Clock(), save_snapshot=lambda: None, exit_fn=exits.append,
    ).arm()
    armed = bare_sink.named("heartbeat_watchdog_armed")[0]
    assert armed["monitor_liveness"] == "monitor_liveness_unwired"




def _compose_run_ast() -> Any:
    """`mantis.run.compose_run`'s AST. Raises rather than returning None if it moves."""
    import ast
    import inspect

    import mantis.run as run_mod

    tree = ast.parse(inspect.getsource(run_mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compose_run":
            return node
    raise AssertionError(
        "compose_run was not found in mantis.run — this census has no subject, and a census "
        "that cannot find its subject must fail rather than pass vacuously (LAW-07)"
    )


def _calls_named(node: Any, name: str) -> list[Any]:
    import ast

    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == name)
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]


def test_the_composition_root_DECLARES_the_disk_guard_as_a_watched_monitor() -> None:
    """Without this the mechanism is a phantom: an audit nothing calls and a watchdog told about
    no monitors. Structural, not textual — the keyword must be present on the real
    `build_run_safety` call and its value must construct a `MonitorLivenessSpec`."""
    import ast

    compose = _compose_run_ast()
    calls = _calls_named(compose, "build_run_safety")
    assert len(calls) == 1, f"expected one build_run_safety call in compose_run, got {len(calls)}"
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "monitor_liveness" in kw, "compose_run must DECLARE the monitors it wants watched"
    specs = _calls_named(kw["monitor_liveness"], "MonitorLivenessSpec")
    assert specs, "the declaration must construct at least one MonitorLivenessSpec"
    names = [k.value.value for s in specs for k in s.keywords
             if k.arg == "name" and isinstance(k.value, ast.Constant)]
    assert "disk_guard" in names, f"the disk guard must be among the watched monitors: {names}"


def test_the_composition_root_RUNS_the_live_arming_audit_off_the_exported_probe_name() -> None:
    """The probe key must be the imported CONSTANT, never a string literal: two literals is the
    duplicated-authority shape, and a rename would then silently produce a
    `ProducerProbeMissingError` at teardown instead of failing here."""
    import ast

    compose = _compose_run_ast()
    calls = _calls_named(compose, "_emit_live_arming_audit")
    assert len(calls) == 1, "compose_run must run the live arming audit exactly once"
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "probes" in kw and isinstance(kw["probes"], ast.Dict)
    keys = kw["probes"].keys
    assert keys and all(isinstance(k, ast.Name) for k in keys), (
        "every probe key must be an imported name, not a literal: "
        f"{[type(k).__name__ for k in keys]}"
    )
    assert [k.id for k in keys] == ["DISK_GUARD_LIVENESS_PROBE"]  # pyright: ignore[reportAttributeAccessIssue]


def test_THE_VACUITY_CONTROL_the_census_fires_on_a_stripped_function() -> None:
    """The two censuses above are only worth their green if they can go red. Drive them
    against a `compose_run` with the wiring removed."""
    import ast

    stripped = ast.parse(
        "def compose_run():\n"
        "    run_safety = build_run_safety(log_dir=1)\n"
        "    return run_safety\n"
    ).body[0]
    calls = _calls_named(stripped, "build_run_safety")
    assert calls and "monitor_liveness" not in {k.arg for k in calls[0].keywords}
    assert not _calls_named(stripped, "_emit_live_arming_audit")




def test_THE_LIVE_AUDIT_EMITS_WHERE_THE_SINK_IS_STILL_OPEN_ON_BOTH_PATHS() -> None:
    """A REGRESSION ROW for a defect this leg introduced and the gate set caught.

    The audit was first written beside the disk guard's own teardown, one line too late:
    `sink.close()` runs on the PARTIAL path and `disk_guard.stop()` after it, so the emit hit a
    closed file, the sink COUNTED the failed write, and the watchdog turned a non-zero persist
    counter into `os._exit(43)`. The only point where the sink is open on both paths is the top
    of the `finally`, so the audit call must precede the `coordinator is None` branch.
    """
    import ast

    compose = _compose_run_ast()
    finallys = [n for n in ast.walk(compose) if isinstance(n, ast.Try) and n.finalbody]
    assert finallys, "compose_run must still have a teardown ladder"
    ladder = max(finallys, key=lambda n: len(n.finalbody))

    audit_line = next(
        (n.lineno for n in ast.walk(ladder)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
         and n.func.id == "_emit_live_arming_audit"), None)
    close_line = next(
        (n.lineno for n in ast.walk(ladder)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "close"
         and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "sink"), None)
    assert audit_line is not None, "the live arming audit left the teardown ladder"
    assert close_line is not None, (
        "no `sink.close()` in the ladder — this row has no subject and must fail rather than "
        "pass vacuously (LAW-07)"
    )
    assert audit_line < close_line, (
        f"the live arming audit ({audit_line}) must run BEFORE sink.close() ({close_line}): "
        "emitting into a closed sink counts a persist error and the watchdog exits 43"
    )


def test_the_audit_survives_a_composition_that_never_BUILT_a_disk_guard() -> None:
    """The partial path's other half. A run that failed before the guard existed has no producer
    to ask, and the truthful answer is DISARMED — never a `ProducerProbeMissingError` emitted as
    the audit's own failure on every failed boot."""
    config = load_config(_PRODUCTION)
    live = audit_arming_live(config,
                             probes={DISK_GUARD_LIVENESS_PROBE: lambda: None is not None})
    assert DISK_SPACE_ABORT_RULE in [r.name for r in live.disarmed]


def test_THE_LIVENESS_SAMPLE_IS_SILENT_DURING_CLOSE_OUT_because_it_is_self_fatal_otherwise(
    tmp_path: Path,
) -> None:
    """The second defect this leg introduced, and the reason the gate is structural. The check
    first ran on EVERY poll, on the argument that a monitor's death is not a pipeline stall —
    an argument about WHEN the reading is interesting that ignored WHO takes it. The watchdog's
    thread emits through the sink it polices, so an emit after `sink.close()` is a counted
    failed write and `counters_fn` answers a non-zero count with `os._exit(43)`."""
    sink, clock, exits = _Sink(), _Clock(), []
    live = MonitorSample(checks_total=1, errors_total=0, interval_sec=60.0)
    wd = _watchdog(tmp_path, sink, clock, lambda: live, exits)

    clock.t += 60.0
    wd.poll_once()
    assert sink.named("monitor_liveness_sample"), "armed, it must publish its reading"

    wd.disarm_staleness()
    sink.events.clear()
    for _ in range(10):
        clock.t += 600.0
        wd.poll_once()
    assert not sink.named("monitor_liveness_sample"), (
        "close-out must be silent: an emit here can only be a failed write, and a failed "
        "write is the counter this watchdog exits 43 on"
    )
    assert not sink.named("monitor_stalled"), "and so must the stall event, for the same reason"
    assert exits == []
