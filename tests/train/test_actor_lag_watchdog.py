"""⊕ WPUF Phase U ORACLE — O-U4: the actor-lag watchdog (DESIGN_U §4/§8).

RED-at-import until IMPL lands `ActorLagSpec` (train/lifecycle/heartbeat_watchdog.py) and
`ACTOR_LAG_EXIT_CODE` (monitor/heartbeat.py).

Invariant: `learner_step − actor_ckpt_step > N` → armed = fire exit 45 (fail-fast family,
R19); disarmed = ONE loud event per exceedance episode, never an abort. Every drive is a
direct `poll_once()` under an injected clock and a spy `exit_fn` — the watchdog thread is
NEVER started; the only threads are the fire path's own bounded effect workers (join
timeout = the ctor's snapshot_timeout_sec). Zero sleeps, zero unbounded joins.

O-U4g (the E32-extension signature pin) lives HERE, not in test_run_safety_wiring.py —
DESIGN row E32 says ORACLE-WRITE picks ONE home; this is it.

>300 justify (R8): nine oracles (a–i) over one rig; the rig/spy set is the file's shared
spine and duplicating it per-file would hide the family.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from mantis.train.lifecycle.heartbeat_watchdog import (  # RED-at-import anchor
    ActorLagSpec,
    HeartbeatWatchdog,
)
from mantis.monitor.heartbeat import (
    ACTOR_LAG_EXIT_CODE,
    HEARTBEAT_SOURCES,
    PERSIST_FATAL_EXIT_CODE,
    WATCHDOG_STALL_EXIT_CODE,
)
from mantis.monitor.config import MonitorConfig
from mantis.train.subsystems import build_run_safety


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _registry():
    return SimpleNamespace(
        sources=("train_step",), ages=lambda: {"train_step": 0.0},
        beaten_sources=lambda: frozenset({"train_step"}), arm=lambda: None,
    )


def _watchdog(tmp_path, *, spec, sink=None, exit_codes=None):
    """Staleness structurally silenced (deadline <= 0 disables that source's fire), so
    every fire observed below is the lag check's own. Fire-effect budget kept small —
    the fire path's internal workers are the ONLY threads and each join is bounded."""
    return HeartbeatWatchdog(
        registry=_registry(), deadlines={"train_step": 0.0},
        sink=sink if sink is not None else _SpySink(),
        counters_fn=lambda: 0, heartbeat_file=tmp_path / "hb.json",
        file_interval_sec=0.0, poll_interval_sec=0.0, clock=lambda: 0.0,
        save_snapshot=lambda: None,
        exit_fn=(exit_codes.append if exit_codes is not None else (lambda code: None)),
        snapshot_timeout_sec=2.0, wired_sources=["train_step"],
        actor_lag=spec,
    )


def _spec(*, learner, actor, threshold, armed):
    return ActorLagSpec(
        learner_step_fn=learner, actor_ckpt_step_fn=actor,
        threshold_steps=threshold, abort_enabled=armed,
    )


# ── (a) armed fire ────────────────────────────────────────────────────────────────────
def test_rigged_lag_over_threshold_fires_named_escalation_when_armed(tmp_path) -> None:
    sink, codes = _SpySink(), []
    wd = _watchdog(tmp_path, sink=sink, exit_codes=codes,
                   spec=_spec(learner=lambda: 1000, actor=lambda: 100,
                              threshold=500, armed=True))
    wd.poll_once()

    assert codes == [ACTOR_LAG_EXIT_CODE] == [45], (
        f"an armed lag breach must exit with the named code 45; got {codes}"
    )
    assert ACTOR_LAG_EXIT_CODE not in {WATCHDOG_STALL_EXIT_CODE, PERSIST_FATAL_EXIT_CODE, 44}, (
        "45 must be a NEW code — 42/43/44 are taken"
    )
    fired = sink.named("heartbeat_watchdog_fired")
    assert len(fired) == 1 and fired[0]["reason"] == "actor_lag_exceeded"
    assert fired[0]["code"] == 45
    for key, value in (("learner_step", 1000), ("actor_ckpt_step", 100),
                       ("lag_steps", 900), ("threshold_steps", 500)):
        assert fired[0].get(key) == value, (
            f"the fire detail must carry {key}={value}; got {fired[0]!r}"
        )


# ── (b) disarmed = one loud event per episode, never an abort ─────────────────────────
def test_rigged_lag_over_threshold_disarmed_emits_event_and_never_aborts(tmp_path) -> None:
    sink, codes = _SpySink(), []
    actor = {"v": 100}
    wd = _watchdog(tmp_path, sink=sink, exit_codes=codes,
                   spec=_spec(learner=lambda: 1000, actor=lambda: actor["v"],
                              threshold=500, armed=False))
    for _ in range(3):
        wd.poll_once()

    assert codes == [], "a disarmed lag breach must NEVER abort"
    exceeded = sink.named("actor_lag_exceeded")
    assert len(exceeded) == 1, (
        f"one event per exceedance EPISODE, not per poll (latch pin); got {len(exceeded)}"
    )
    assert exceeded[0].get("armed") is False

    actor["v"] = 1000          # lag drops to 0 — the latch must reset…
    wd.poll_once()
    assert len(sink.named("actor_lag_exceeded")) == 1

    actor["v"] = 100           # …so a NEW episode is loud again
    wd.poll_once()
    assert len(sink.named("actor_lag_exceeded")) == 2, (
        "a new exceedance episode after recovery must emit again (latch resets on "
        "lag <= threshold)"
    )
    assert codes == []


# ── (c) close-out suppression ─────────────────────────────────────────────────────────
def test_lag_check_suppressed_during_close_out(tmp_path) -> None:
    """During close-out both counters freeze; a teardown must never die to a stale lag
    reading — the check runs iff staleness is armed (DESIGN §4.2)."""
    sink, codes = _SpySink(), []
    wd = _watchdog(tmp_path, sink=sink, exit_codes=codes,
                   spec=_spec(learner=lambda: 1000, actor=lambda: 0,
                              threshold=10, armed=True))
    wd.disarm_staleness()
    wd.poll_once()

    assert codes == [], "no fire during close-out, however large the rigged lag"
    assert sink.named("actor_lag_exceeded") == [], "no lag event during close-out either"


# ── (d) NOT a fifth heartbeat source ──────────────────────────────────────────────────
def test_actor_lag_is_not_a_heartbeat_source() -> None:
    """A step-delta threshold in a seconds-deadline dict is a type lie (DESIGN §4.1)."""
    assert HEARTBEAT_SOURCES == (
        "train_step", "inference_dispatch", "selfplay_drain", "eval_round",
    ), f"HEARTBEAT_SOURCES must stay exactly the 4-tuple at HEAD: {HEARTBEAT_SOURCES}"
    assert all("actor" not in s for s in HEARTBEAT_SOURCES)


# ── (e) arm-time visibility ───────────────────────────────────────────────────────────
def test_arm_event_names_actor_lag_posture(tmp_path) -> None:
    """WP10 visibility law: a disabled or unwired lag check is loud at arm time."""
    sink = _SpySink()
    wd = _watchdog(tmp_path, sink=sink,
                   spec=_spec(learner=lambda: 0, actor=lambda: 0,
                              threshold=500, armed=True))
    wd.arm()
    armed = sink.named("heartbeat_watchdog_armed")
    assert armed and armed[-1].get("actor_lag") == {"armed": True, "threshold_steps": 500}

    sink2 = _SpySink()
    wd2 = _watchdog(tmp_path, sink=sink2,
                    spec=_spec(learner=lambda: 0, actor=lambda: 0,
                               threshold=500, armed=False))
    wd2.arm()
    assert sink2.named("heartbeat_watchdog_armed")[-1].get("actor_lag") == {
        "armed": False, "threshold_steps": 500}

    sink3 = _SpySink()
    wd3 = _watchdog(tmp_path, sink=sink3, spec=None)
    wd3.arm()
    assert sink3.named("heartbeat_watchdog_armed")[-1].get("actor_lag") == "absent", (
        "an unwired lag check must read 'absent' at arm time, never be silent"
    )


# ── (f) build_run_safety wiring ───────────────────────────────────────────────────────
def test_build_run_safety_wires_actor_lag_from_monitor_config(tmp_path, monkeypatch) -> None:
    """E31's sibling: the watchdog built by `build_run_safety` carries the monitor
    config's threshold/arming AND the two injected callables LIVE — proven by a fire
    whose detail echoes the injected values, then read back from the real JSONL sink."""
    import mantis.train.checkpoints as _ckpt
    monkeypatch.setattr(_ckpt, "persist_errors_total", 0)  # keep 43 out of the way
    codes: list[int] = []
    run_safety = build_run_safety(
        log_dir=tmp_path, run_id="oracle_u4f",
        buffer=SimpleNamespace(save_to_path=lambda p: None),
        buffer_persist_path=tmp_path / "replay_buffer.bin",
        wired_sources=list(HEARTBEAT_SOURCES),
        monitor_cfg=MonitorConfig(actor_lag_threshold_steps=77,
                                  actor_lag_abort_enabled=True),
        exit_fn=codes.append,
        actor_ckpt_step_fn=lambda: 123,
        learner_step_fn=lambda: 300,
    )
    run_safety.watchdog.poll_once()

    assert codes == [45], (
        "the wired watchdog must fire 45 on the injected lag (300-123=177 > 77) — the "
        f"callables and threshold must be LIVE end to end; got {codes}"
    )
    lines = [json.loads(l) for l in
             run_safety.sink.path.read_text().splitlines() if l.strip()]
    fired = [e for e in lines if e.get("event") == "heartbeat_watchdog_fired"]
    assert fired and fired[0]["reason"] == "actor_lag_exceeded"
    assert fired[0]["learner_step"] == 300 and fired[0]["actor_ckpt_step"] == 123
    assert fired[0]["threshold_steps"] == 77, "monitor_cfg's threshold must reach the spec"


# ── (g) ALL THREE ActorLagSpec inputs carry NO defaults (E32 extension; R1 posture) ───
# WPAX R67 (bounded R43 event). Three changes, and the reason for each:
#   1. `monitor_cfg` JOINS the census. It was found by RED-TEAM F-2 and parked in
#      `tests/train/test_actor_lag_wiring_live.py` with an authority note, because this file
#      was byte-frozen and that fix pass held no R43 event. R67 is that event, so the second
#      site is deleted and the rule lives HERE, once (LAW-08). RED-TEAM-2's MUT-10 measured
#      what the two-site arrangement cost: deleting the parked test AND restoring the F-2
#      defect together produced **0 failures**, caught only by the collected-count floor,
#      which any co-added test masks.
#   2. PARAMETRIZED, one item per name, so each name DIES ALONE — the discipline R-P2/R72
#      make standing law, applied to the census that is itself a census.
#   3. RENAMED from `test_build_run_safety_lag_fns_have_no_defaults`: `monitor_cfg` is not a
#      fn, so the old name would have become false. A test name asserting something untrue is
#      the same class R67's other half is fixing in `tests/test_run_strict_composition.py`.
#      Old name recorded here because REDTEAM_S/REDTEAM_S2 cite it.
@pytest.mark.parametrize(
    ("name", "silently_unwires"),
    [
        ("actor_ckpt_step_fn", "the lag READING (the E32/RED-TEAM F3 class)"),
        ("learner_step_fn", "the lag READING (the E32/RED-TEAM F3 class)"),
        # F-2's own evidence, moved rather than discarded: measured against an ARMED run5,
        # `build_run_safety(monitor_cfg=None)` fell back to a bare `MonitorConfig()` whose
        # `actor_lag_abort_enabled` is False, so a lag of 10 000 over a threshold of 100
        # produced `exit codes []`. Note the failure mode is NOT the siblings': they unwire
        # the reading, this unwires the ABORT while the reading stays perfectly live.
        ("monitor_cfg",
         "the actor-lag ABORT — a bare MonitorConfig() carries actor_lag_abort_enabled=False "
         "(the ADJ-07 / RED-TEAM F-2 class)"),
    ],
)
def test_build_run_safety_actor_lag_inputs_have_no_defaults(
    name: str, silently_unwires: str
) -> None:
    params = inspect.signature(build_run_safety).parameters
    assert name in params, f"build_run_safety must take {name} explicitly"
    assert params[name].default is inspect.Parameter.empty, (
        f"{name} must be REQUIRED — a default here silently unwires {silently_unwires}"
    )


# ── (h) negative lag is a reported wiring bug, never a fire ───────────────────────────
def test_negative_lag_reports_wiring_bug_event_once(tmp_path) -> None:
    sink, codes = _SpySink(), []
    wd = _watchdog(tmp_path, sink=sink, exit_codes=codes,
                   spec=_spec(learner=lambda: 10, actor=lambda: 50,
                              threshold=5, armed=True))
    for _ in range(3):
        wd.poll_once()

    assert codes == [], "a negative lag is a wiring bug being reported, never a fire"
    assert len(sink.named("actor_lag_negative")) == 1, (
        "the wiring-bug report must be loud exactly once, never spam"
    )


# ── (i) LAW-07 mutation self-test: the check reads the callables LIVE ─────────────────
def test_lag_check_reads_callables_live_not_captured_values(tmp_path) -> None:
    """The O-28 pattern: freeze the actor callable, advance the learner between polls —
    the fire MUST happen once the delta crosses N, proving poll-time reads (a value
    captured at ctor/arm would read a frozen delta forever and never fire)."""
    codes: list[int] = []
    learner = {"v": 3}
    wd = _watchdog(tmp_path, exit_codes=codes,
                   spec=_spec(learner=lambda: learner["v"], actor=lambda: 0,
                              threshold=5, armed=True))
    wd.poll_once()
    assert codes == [], "lag 3 <= threshold 5: no fire yet"

    learner["v"] = 10
    wd.poll_once()
    assert codes == [45], (
        "the watchdog must observe the ADVANCED learner step on the next poll — the "
        "callables are read live, never captured"
    )
