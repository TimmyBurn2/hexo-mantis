"""The INDEPENDENT heartbeat watchdog.

The crux: a `tick()`-driven stall watchdog is driven from the MAIN loop, so a tick from the wedged
thread can never fire. This one is an INDEPENDENT thread reading heartbeat STALENESS, so a wedge
INSIDE an eval call still trips it. Real-thread tests are bounded by
`threading.Event.wait(timeout=...)`, so a bug fails within ~5 s and never hangs CI.

>300 justify (R8): ONE unit under test with one shared harness — the fake clock, the `_make_wd`
factory and the `_TimedExit` spy are used by every row, so splitting by fire class would duplicate
the harness and let the deterministic and real-thread halves drift.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import pytest

from mantis.monitor.heartbeat import (
    HEARTBEAT_SOURCES,
    HeartbeatRegistry,
    read_heartbeat_file,
)
from mantis.monitor.supervise import LivenessTracker
from mantis.train.lifecycle.heartbeat_watchdog import HeartbeatWatchdog


def _wait_until(pred, timeout: float) -> bool:
    """Poll `pred` until true or `timeout` elapses (bounded — never hangs CI)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False

_ALL = {s: 0.5 for s in HEARTBEAT_SOURCES}


def _make_wd(*, registry, deadlines, sink, clock, exit_fn, save_snapshot, hb_file,
            counters_fn=lambda: 0, poll=0.1, file_interval=0.0):
    return HeartbeatWatchdog(
        registry=registry, deadlines=deadlines, sink=sink, counters_fn=counters_fn,
        heartbeat_file=hb_file, file_interval_sec=file_interval, poll_interval_sec=poll,
        clock=clock, save_snapshot=save_snapshot, exit_fn=exit_fn,
    )


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_no_fire_below_deadline_fire_at_first_poll_past_deadline(tmp_path, spy_sink):
    """No fire at age D−ε; fire at the first poll where age ≥ D, naming the stale source, exit 42."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    wd = _make_wd(registry=reg, deadlines=dict(_ALL), sink=spy_sink, clock=clock,
                 exit_fn=exits.append, save_snapshot=lambda: None, hb_file=tmp_path / "hb.json")
    wd.arm()
    clock.t = 0.49
    wd.poll_once()
    assert not exits, "must NOT fire below the deadline (age 0.49 < 0.5)"
    clock.t = 0.5
    wd.poll_once()
    assert exits and exits[0] == 42, "must fire 42 at the first poll with age ≥ deadline"
    fired = spy_sink.named("heartbeat_watchdog_fired")
    assert fired and "train_step" in str(fired[-1].get("reason", ""))


def test_per_source_deadlines_are_independent(tmp_path, spy_sink):
    """A short deadline on one source fires while the others stay silent, and the fire names the
    short-deadline source only."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    deadlines = {"train_step": 100.0, "inference_dispatch": 0.5, "selfplay_drain": 100.0,
                "eval_round": 100.0}
    wd = _make_wd(registry=reg, deadlines=deadlines, sink=spy_sink, clock=clock,
                 exit_fn=exits.append, save_snapshot=lambda: None, hb_file=tmp_path / "hb.json")
    wd.arm()
    clock.t = 0.6
    wd.poll_once()
    assert exits == [42]
    assert "inference_dispatch" in str(spy_sink.named("heartbeat_watchdog_fired")[-1]["reason"])


def test_arm_log_emitted_even_when_a_deadline_disables_a_source(tmp_path, spy_sink):
    """A source with deadline ≤ 0 is disabled from firing, but the arm-log STILL names it. The
    clock stops short of the positive deadlines, then crosses them to prove the OTHER sources are
    armed: asserting zero fires past every deadline would force a GLOBAL staleness disable."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    deadlines = {"train_step": 0.0, "inference_dispatch": 100.0, "selfplay_drain": 100.0,
                "eval_round": 100.0}
    wd = _make_wd(registry=reg, deadlines=deadlines, sink=spy_sink, clock=clock,
                 exit_fn=exits.append, save_snapshot=lambda: None, hb_file=tmp_path / "hb.json")
    wd.arm()
    armed = spy_sink.named("heartbeat_watchdog_armed")
    assert armed, "arm() must emit heartbeat_watchdog_armed"
    assert "train_step" in str(armed[-1]), "the arm-log must name every source, even a disabled one"

    clock.t = 50.0                       # past NOTHING: train_step disabled, others still fresh
    wd.poll_once()
    assert not exits, "a deadline ≤ 0 disables THAT source's fire (age 50 ≫ 0, no fire)"

    clock.t = 150.0                      # now the two POSITIVE deadlines (100 s) are crossed
    wd.poll_once()
    assert exits == [42], (
        "a zeroed deadline must NOT disable the other sources — that would kill both levels "
        "of livelock protection (in-process fire AND the supervisor's frozen-seq backstop)"
    )
    reason = str(spy_sink.named("heartbeat_watchdog_fired")[-1]["reason"])
    assert "train_step" not in reason, f"the disabled source must never be blamed: {reason!r}"
    assert "inference_dispatch" in reason or "selfplay_drain" in reason


def test_missing_deadline_for_a_registry_source_is_a_loud_wiring_error(tmp_path, spy_sink):
    """A registry source with NO entry in `deadlines` must raise at CONSTRUCTION: reading it as an
    implicit 0.0 would silently blind the watchdog to a whole pipeline stage, and the supervisor
    too, since the thread keeps mirroring a fresh `seq`."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    with pytest.raises(ValueError) as ei:
        _make_wd(registry=reg, deadlines={"train_step": 1.0}, sink=spy_sink, clock=clock,
                 exit_fn=lambda code: None, save_snapshot=lambda: None,
                 hb_file=tmp_path / "hb.json")
    assert "inference_dispatch" in str(ei.value) and "selfplay_drain" in str(ei.value)


def test_disarm_staleness_stops_stall_fire_but_persist_and_file_stay_live(tmp_path, spy_sink):
    """After `disarm_staleness()` per-source staleness never fires, but persist-fatal still works
    and `seq` keeps advancing. The disarm SWAPS the per-source deadlines for one bounded close-out
    budget, so this drives far past the per-source deadline while staying inside that budget."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    counter = {"n": 0}
    hb_file = tmp_path / "hb.json"
    wd = _make_wd(registry=reg, deadlines=dict(_ALL), sink=spy_sink, clock=clock,
                 exit_fn=exits.append, save_snapshot=lambda: None, hb_file=hb_file,
                 counters_fn=lambda: counter["n"], poll=0.1, file_interval=0.0)
    wd.arm()
    wd.disarm_staleness()

    clock.t = 100.0                       # 200× the per-source deadline, ≪ the close-out budget
    wd.poll_once()
    seq_a = read_heartbeat_file(hb_file).seq
    clock.t = 200.0
    wd.poll_once()
    seq_b = read_heartbeat_file(hb_file).seq
    assert not exits, "per-source staleness must NOT fire after disarm, far past the deadline"
    assert seq_b > seq_a, "the heartbeat file seq must keep advancing through a disarmed close-out"

    counter["n"] = 1                      # a persist failure during the disarmed window
    wd.poll_once()
    assert exits and exits[0] == 43, "persist-fatal is NEVER disarmed"


def test_close_out_overrun_fires_after_the_teardown_budget(tmp_path, spy_sink):
    """A teardown that overruns the close-out budget STILL fires 42. Switching staleness off
    permanently while the file mirror kept advancing `seq` made a close-out wedge invisible to
    BOTH levels for an unbounded window."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    wd = HeartbeatWatchdog(
        registry=reg, deadlines=dict(_ALL), sink=spy_sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / "hb.json", file_interval_sec=0.0, poll_interval_sec=0.1,
        clock=clock, save_snapshot=lambda: None, exit_fn=exits.append,
        close_out_deadline_sec=50.0,
    )
    wd.arm()
    wd.disarm_staleness()

    clock.t = 49.0
    wd.poll_once()
    assert not exits, "a close-out inside its budget must stay quiet (O-27 by construction)"

    clock.t = 51.0
    wd.poll_once()
    assert exits == [42], "a teardown wedge past the close-out budget must fire 42"
    fired = spy_sink.named("heartbeat_watchdog_fired")[-1]
    assert fired["reason"] == "close_out_timeout", fired
    assert fired["elapsed_sec"] >= 50.0


def test_close_out_deadline_zero_keeps_the_old_unbounded_behaviour(tmp_path, spy_sink):
    """`close_out_deadline_sec <= 0` is the documented off switch for the teardown budget: an
    operator who genuinely wants an unbounded close-out must say so explicitly, and it is never
    the default."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    wd = HeartbeatWatchdog(
        registry=reg, deadlines=dict(_ALL), sink=spy_sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / "hb.json", file_interval_sec=0.0, poll_interval_sec=0.1,
        clock=clock, save_snapshot=lambda: None, exit_fn=exits.append,
        close_out_deadline_sec=0.0,
    )
    wd.arm()
    wd.disarm_staleness()
    clock.t = 1_000_000.0
    wd.poll_once()
    assert not exits


def test_checkpoint_source_live_attribute_increment_fires_43(tmp_path, spy_sink, monkeypatch):
    """O-28 / P-28 — an increment of `mantis.train.checkpoints.persist_errors_total` made AFTER
    the watchdog's `counters_fn` is constructed is observed (live module-attribute read) → fire
    43. LAW-14 requires the checkpoint source fatal via the watchdog, not just the sink."""
    import mantis.train.checkpoints as checkpoints

    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    wd = _make_wd(registry=reg, deadlines=dict(_ALL), sink=spy_sink, clock=clock,
                 exit_fn=exits.append, save_snapshot=lambda: None, hb_file=tmp_path / "hb.json",
                 counters_fn=lambda: checkpoints.persist_errors_total)  # LIVE module-attr read
    wd.arm()
    wd.poll_once()
    assert not exits, "clean checkpoint state must not fire"

    monkeypatch.setattr(checkpoints, "persist_errors_total",
                        checkpoints.persist_errors_total + 1)  # a checkpoint persist failure
    wd.poll_once()
    assert exits and exits[0] == 43, "a checkpoint-source persist failure must fire 43"


def test_frozen_int_binding_mutant_is_rejected(tmp_path, spy_sink, monkeypatch):
    """A `counters_fn` that VALUE-BINDS the counter at construction reads 0 forever after a
    `global … += 1`, so it never fires. This proves the oracle DISTINGUISHES the footgun: the
    frozen-int mutant does NOT fire, so the implementation must read the module ATTRIBUTE live."""
    import mantis.train.checkpoints as checkpoints

    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    frozen = checkpoints.persist_errors_total          # the footgun: value bound at construction
    wd = _make_wd(registry=reg, deadlines=dict(_ALL), sink=spy_sink, clock=clock,
                 exit_fn=exits.append, save_snapshot=lambda: None, hb_file=tmp_path / "hb.json",
                 counters_fn=lambda: frozen)
    wd.arm()
    monkeypatch.setattr(checkpoints, "persist_errors_total",
                        checkpoints.persist_errors_total + 1)
    wd.poll_once()
    assert not exits, "the frozen-int binding mutant is blind to the increment (correctly rejected)"


class _TimedExit:
    """Injected exit_fn recording (code, monotonic-timestamp) and setting a fire Event."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []
        self.evt = threading.Event()

    def __call__(self, code: int) -> None:
        self.calls.append((int(code), time.monotonic()))
        self.evt.set()


def test_livelock_fires_within_envelope_while_main_thread_wedged(tmp_path, spy_sink):
    """Deadline 0.5 s, poll 0.1 s; the "main thread" is wedged forever in a mock eval so
    `train_step` never beats. The INDEPENDENT watchdog fires inside the envelope, writes the
    snapshot, exits 42, and the wedge is STILL set — the fire needed no main-thread cooperation."""
    wedge = threading.Event()            # the wedged eval: never released during the test
    snap_path = tmp_path / "buffer.bin.watchdog"

    def _snapshot() -> None:
        snap_path.write_text("snap")

    exit_fn = _TimedExit()
    reg = HeartbeatRegistry()            # real time.monotonic clock
    wd = _make_wd(registry=reg, deadlines={s: 0.5 for s in HEARTBEAT_SOURCES}, sink=spy_sink,
                 clock=time.monotonic, exit_fn=exit_fn, save_snapshot=_snapshot,
                 hb_file=tmp_path / "hb.json", poll=0.1)

    # A daemon thread stuck in the mock eval — the watchdog does not need it to un-wedge.
    threading.Thread(target=wedge.wait, daemon=True).start()

    t0 = time.monotonic()
    wd.start()                           # arm + spawn the independent daemon thread
    try:
        assert exit_fn.evt.wait(timeout=5.0), "the watchdog must fire while the main thread wedges"
        code, t_fire = exit_fn.calls[0]
        elapsed = t_fire - t0
        assert code == 42
        assert 0.5 <= elapsed <= 2.0, f"fire must land in the deadline envelope, got {elapsed:.3f}s"
        assert snap_path.exists(), "the fire must write the distinct .watchdog snapshot"
        assert not wedge.is_set(), "the wedge must still be unset — the fire needed no main thread"
    finally:
        wd.stop()
        wedge.set()


@pytest.mark.parametrize("wedged", list(HEARTBEAT_SOURCES))
def test_wedge_matrix_fire_names_exactly_the_wedged_source(tmp_path, spy_sink, wedged):
    """Wedge exactly one source while a beater keeps the others fresh; the fire names precisely the
    wedged source and never a healthy one."""
    healthy = [s for s in HEARTBEAT_SOURCES if s != wedged]
    reg = HeartbeatRegistry()
    exit_fn = _TimedExit()
    wd = _make_wd(registry=reg, deadlines={s: 0.5 for s in HEARTBEAT_SOURCES}, sink=spy_sink,
                 clock=time.monotonic, exit_fn=exit_fn, save_snapshot=lambda: None,
                 hb_file=tmp_path / "hb.json", poll=0.05)

    stop_beater = threading.Event()

    def _beat_healthy() -> None:
        while not stop_beater.is_set():
            for s in healthy:
                reg.beat(s)
            time.sleep(0.05)

    beater = threading.Thread(target=_beat_healthy, daemon=True)
    wd.start()
    beater.start()
    try:
        assert exit_fn.evt.wait(timeout=5.0), f"wedging {wedged} must fire the watchdog"
        assert exit_fn.calls[0][0] == 42
        reasons = " ".join(str(e.get("reason", "")) for e in spy_sink.named("heartbeat_watchdog_fired"))
        assert wedged in reasons, f"the fire must name the wedged source {wedged}: {reasons!r}"
        for s in healthy:
            assert s not in reasons, f"a healthy source {s} must never be blamed: {reasons!r}"
    finally:
        stop_beater.set()
        wd.stop()
        beater.join(timeout=2.0)


def test_fire_path_documents_gil_limit_and_supervisor_backstop():
    """O-14 / P-14 — the native-GIL limit is DOCUMENTED: the watchdog module docstring names the
    GIL starvation case and the supervisor as the backstop (missing documentation = FAIL)."""
    import mantis.train.lifecycle.heartbeat_watchdog as mod

    doc = (mod.__doc__ or "").lower()
    assert "gil" in doc, "the module doc must name the native-GIL starvation limit"
    assert "supervisor" in doc, "the module doc must name the supervisor as the backstop"


def test_fire_path_runs_while_main_thread_holds_a_lock(tmp_path, spy_sink):
    """O-14 / P-14 — the fire path completes entirely on the watchdog thread while the main
    thread is blocked (a lock proxy is held the whole time). Bites the assumption that the fire
    needs main-thread cooperation."""
    held = threading.Lock()
    held.acquire()                       # the main thread holds a lock the watchdog never touches
    exit_fn = _TimedExit()
    reg = HeartbeatRegistry()
    wd = _make_wd(registry=reg, deadlines={s: 0.5 for s in HEARTBEAT_SOURCES}, sink=spy_sink,
                 clock=time.monotonic, exit_fn=exit_fn, save_snapshot=lambda: None,
                 hb_file=tmp_path / "hb.json", poll=0.1)
    wd.start()
    try:
        assert exit_fn.evt.wait(timeout=5.0), "the watchdog must fire on its own thread"
        assert held.locked(), "the main-thread lock is still held — the fire needed no main thread"
    finally:
        wd.stop()
        held.release()


def test_undeclared_never_beaten_source_warns_instead_of_firing(tmp_path, spy_sink):
    """An UNDECLARED source that has never beaten must NOT age into a 42; it gets a loud
    `heartbeat_source_unwired` instead. One omitted `heartbeat=` kwarg made a healthy run fire 42
    and the supervisor relaunch into the same missing wiring until the budget was gone."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    wd = HeartbeatWatchdog(
        registry=reg, deadlines=dict(_ALL), sink=spy_sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / "hb.json", file_interval_sec=0.0, poll_interval_sec=0.1,
        clock=clock, save_snapshot=lambda: None, exit_fn=exits.append,
        wired_sources=["train_step", "selfplay_drain"],      # inference_dispatch NOT wired
    )
    wd.arm()
    armed = spy_sink.named("heartbeat_watchdog_armed")[-1]
    assert armed["unwired_sources"] == ["inference_dispatch", "eval_round"], (
        "eval_round joins inference_dispatch as unwired: WP11-A extends HEARTBEAT_SOURCES "
        "and this fixture declares neither wired"
    )
    assert armed["enabled"]["inference_dispatch"] is False, (
        "an unwired source must not read enabled:True in the arm log"
    )

    reg.beat("train_step")
    reg.beat("selfplay_drain")
    clock.t = 10.0                                   # 20× every deadline
    reg.beat("train_step")
    reg.beat("selfplay_drain")
    wd.poll_once()
    assert not exits, "an unwired source must never fire a stall abort"
    unwired = spy_sink.named("heartbeat_source_unwired")
    assert [e["source"] for e in unwired] == ["inference_dispatch", "eval_round"]

    clock.t = 20.0                                   # emitted ONCE per source, not once per poll
    reg.beat("train_step")
    reg.beat("selfplay_drain")
    wd.poll_once()
    assert len(spy_sink.named("heartbeat_source_unwired")) == 2
    assert not exits


def test_declared_source_that_never_beats_still_fires(tmp_path, spy_sink):
    """The carve-out is narrow: a source the root DECLARED as wired is watched from arm time, so a
    stage that dies before its very FIRST beat is still caught. Bites an over-broad "never beaten
    ⇒ never fire" rule that would silently drop wedge coverage."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    wd = HeartbeatWatchdog(
        registry=reg, deadlines=dict(_ALL), sink=spy_sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / "hb.json", file_interval_sec=0.0, poll_interval_sec=0.1,
        clock=clock, save_snapshot=lambda: None, exit_fn=exits.append,
        wired_sources=list(HEARTBEAT_SOURCES),
    )
    wd.arm()
    clock.t = 1.0
    wd.poll_once()
    assert exits == [42], "a DECLARED source that never beats is a wedge, not a wiring gap"


def test_wired_sources_rejects_an_unknown_source_name(tmp_path, spy_sink):
    """A typo'd declaration must fail LOUD at construction, never silently widen or narrow what is
    watched."""
    reg = HeartbeatRegistry(clock=_Clock())
    with pytest.raises(ValueError) as ei:
        HeartbeatWatchdog(
            registry=reg, deadlines=dict(_ALL), sink=spy_sink, counters_fn=lambda: 0,
            heartbeat_file=tmp_path / "hb.json", file_interval_sec=0.0, poll_interval_sec=0.1,
            clock=_Clock(), save_snapshot=lambda: None, exit_fn=lambda code: None,
            wired_sources=["train_step", "typo_source"],
        )
    assert "typo_source" in str(ei.value)


def test_hung_snapshot_still_exits_within_a_bounded_time(tmp_path, spy_sink):
    """A `save_snapshot` that NEVER returns must not suppress `exit_fn`: `best_effort` catches
    exceptions, not hangs, so each optional effect runs on its own thread under a hard budget."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []
    release = threading.Event()

    def _hung_snapshot() -> None:
        release.wait(timeout=30.0)               # never released during the assertions

    wd = HeartbeatWatchdog(
        registry=reg, deadlines=dict(_ALL), sink=spy_sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / "hb.json", file_interval_sec=0.0, poll_interval_sec=0.1,
        clock=clock, save_snapshot=_hung_snapshot, exit_fn=exits.append,
        snapshot_timeout_sec=0.3,
    )
    wd.arm()
    clock.t = 1.0
    started = time.monotonic()
    wd.poll_once()
    elapsed = time.monotonic() - started
    release.set()

    assert exits == [42], "a hung snapshot must NOT swallow the exit"
    assert elapsed < 5.0, f"the fire must be bounded by the effect budget, took {elapsed:.2f}s"
    complete = spy_sink.named("heartbeat_watchdog_fire_complete")
    assert complete and complete[-1]["snapshot_ok"] is False
    assert complete[-1]["best_effort_counters"].get("watchdog_snapshot_timeout") == 1, (
        "the abandoned effect must be COUNTED, not silent"
    )


def test_fire_complete_publishes_the_best_effort_counters(tmp_path, spy_sink):
    """The fire's outcome reaches the ONE channel: a snapshot that FAILS is recorded in
    `heartbeat_watchdog_fire_complete` with the counter registry, not only in a stderr WARN
    moments before `os._exit`."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    exits: list[int] = []

    def _boom() -> None:
        raise OSError("no such directory")

    wd = _make_wd(registry=reg, deadlines=dict(_ALL), sink=spy_sink, clock=clock,
                 exit_fn=exits.append, save_snapshot=_boom, hb_file=tmp_path / "hb.json")
    wd.arm()
    clock.t = 1.0
    wd.poll_once()
    assert exits == [42]
    complete = spy_sink.named("heartbeat_watchdog_fire_complete")[-1]
    assert complete["snapshot_ok"] is False
    assert complete["best_effort_counters"].get("watchdog_snapshot") == 1
    assert wd.counters.get("watchdog_snapshot") == 1


def test_gil_starvation_freezes_seq_and_the_supervisor_declares_it_stale(tmp_path, spy_sink):
    """GIL starvation, asserted rather than documented: one non-yielding C call starves the
    watchdog thread, `seq` FREEZES (level 1 is genuinely blind — the honest limit), and the
    supervisor's own staleness core then declares that frozen seq stale."""
    reg = HeartbeatRegistry()
    wd = _make_wd(registry=reg, deadlines={s: 1e9 for s in HEARTBEAT_SOURCES}, sink=spy_sink,
                 clock=time.monotonic, exit_fn=lambda code: None, save_snapshot=lambda: None,
                 hb_file=tmp_path / "hb.json", poll=0.01, file_interval=0.0)
    hb = tmp_path / "hb.json"
    wd.start()
    try:
        assert _wait_until(lambda: read_heartbeat_file(hb) is not None, 5.0), "no first mirror"
        before = read_heartbeat_file(hb).seq
        t0 = time.monotonic()
        math.factorial(700_000)              # one non-yielding C call: the GIL is held
        elapsed = time.monotonic() - t0
        frozen_state = read_heartbeat_file(hb)
        frozen = frozen_state.seq
        # A free-running watchdog mirrors once per 0.01 s poll, so this window should have produced
        # ~elapsed/0.01 increments. At most ONE boundary tick may land on either side of the read;
        # anything more means the thread was NOT starved.
        free_running = elapsed / 0.01
        assert elapsed > 0.2, f"the GIL window was too short to be meaningful ({elapsed:.3f}s)"
        assert frozen - before <= 2 and free_running > 10 * 2, (
            f"the watchdog thread must be STARVED while the GIL is held: seq {before} → "
            f"{frozen} over {elapsed:.2f}s (a free-running thread would add ~{free_running:.0f}); "
            "this is the documented limit the supervisor exists to cover"
        )
        # Level 2 keys on seq PROGRESSION on its own clock — a frozen seq is stale no matter how
        # healthy the child looks.
        tracker = LivenessTracker(stale_after_sec=1.0)
        tracker.reset(now=0.0)
        # The SAME captured state, twice. Re-READING the live file here raced the watchdog thread:
        # the GIL window has ended, so a tick landing between the two reads made `observe` see
        # PROGRESS and `is_stale` go False at a measured ~10%. The property under test is "an
        # UNCHANGED seq observed 2 s apart is stale", so the input must be unchanged by
        # construction, not by luck.
        tracker.observe(frozen_state, now=0.0)
        tracker.observe(frozen_state, now=2.0)      # the SAME frozen state, 2 s later
        assert tracker.is_stale(2.0), "the supervisor must declare a frozen seq stale"
        assert _wait_until(lambda: read_heartbeat_file(hb).seq > frozen, 5.0), (
            "the watchdog thread must resume mirroring once the GIL is released"
        )
    finally:
        wd.stop()
