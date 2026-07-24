"""⊕ O-12 / O-17 / O-13(ii,iii) — the out-of-process supervisor (L-C).

RED-at-import until IMPL writes `mantis.monitor.supervise`. ORACLE-FIRST (⊕): the top-level
`import mantis.monitor.supervise` raises ModuleNotFoundError before any port code exists.
Torch-free host-neutral babysitter (§c.5).

`Supervisor` takes injected `spawn_fn/kill_fn/clock` (+ `sleep_fn` advancing the fake clock)
so the loop is driven deterministically; `LivenessTracker` is the seq/pid staleness core.

PASS bars:
  * O-12 / P-12: a frozen `seq` ⇒ exactly one SIGTERM→(grace)→SIGKILL→relaunch; a child
    exit 42 ⇒ one relaunch with ZERO kill calls — the two paths are DISTINCT by construction.
  * O-17 / P-17: 0→stop rc 0; 42→relaunch; 43→stop rc 43 (persist faults are not transient);
    other rc→propagate no relaunch; budget exceeded→loud nonzero. Equality pin
    `SELFPLAY_STALL_EXIT_CODE == WATCHDOG_STALL_EXIT_CODE == 42`.
  * O-13(ii): an mtime touch WITHOUT a `seq` change still counts stale (seq-keyed, never mtime).
  * O-13(iii): a `pid` change resets the seq baseline (a child restart is not forgery).
"""
from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from mantis.monitor.heartbeat import (
    PERSIST_FATAL_EXIT_CODE,
    WATCHDOG_STALL_EXIT_CODE,
    write_heartbeat_file,
)
from mantis.monitor.supervise import LivenessTracker, Supervisor
from mantis.train.lifecycle.watchdog import SELFPLAY_STALL_EXIT_CODE


# ── scripted child + clock harness ────────────────────────────────────────────────────
class FakeChild:
    """A scripted child process handle: `.poll()` walks `poll_returns` (the last value
    repeats forever); `.pid` is fixed."""

    def __init__(self, pid: int, poll_returns: list[int | None]) -> None:
        self.pid = pid
        self._seq = list(poll_returns)

    def poll(self) -> int | None:
        if len(self._seq) > 1:
            return self._seq.pop(0)
        return self._seq[0]


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _make_supervisor(children, *, hb_file, stale_after=10.0, poll=1.0, grace=1.0,
                     max_relaunches=5):
    """Wire a Supervisor over a scripted spawn sequence with a fake clock advanced by sleep."""
    clock = _Clock()
    spawns: list = []
    kills: list[tuple[int, int]] = []
    seq = list(children)

    def spawn_fn(argv):
        child = seq.pop(0)
        spawns.append(child)
        return child

    def kill_fn(child, sig):
        kills.append((child.pid, sig))

    sup = Supervisor(
        child_argv=["python", "-m", "mantis.train"],
        heartbeat_file=hb_file,
        stale_after_sec=stale_after,
        poll_interval_sec=poll,
        kill_grace_sec=grace,
        max_relaunches=max_relaunches,
        spawn_fn=spawn_fn,
        kill_fn=kill_fn,
        clock=clock,
        sleep_fn=lambda s: clock.advance(s),
    )
    return sup, spawns, kills, clock


# ── O-17 exit-code contract ───────────────────────────────────────────────────────────
def test_exit_code_equality_pin() -> None:
    """O-17 — a single restart-wrapper key: the two authorities agree at 42."""
    assert SELFPLAY_STALL_EXIT_CODE == WATCHDOG_STALL_EXIT_CODE == 42


def test_child_exit_zero_stops_rc_zero(tmp_path: Path) -> None:
    """O-17 — child rc 0 ⇒ supervisor exits 0, no relaunch."""
    sup, spawns, kills, _ = _make_supervisor([FakeChild(1, [0])], hb_file=tmp_path / "hb.json")
    assert sup.run() == 0
    assert len(spawns) == 1 and kills == []


def test_child_exit_43_stops_no_relaunch(tmp_path: Path) -> None:
    """O-17 — persist-fatal (43) is NOT transient: relaunching would loop the storage fault, so
    the supervisor stops loud with rc 43 and does not respawn."""
    sup, spawns, kills, _ = _make_supervisor([FakeChild(1, [PERSIST_FATAL_EXIT_CODE])],
                                             hb_file=tmp_path / "hb.json")
    assert sup.run() == 43
    assert len(spawns) == 1 and kills == []


def test_child_exit_other_rc_propagates_no_relaunch(tmp_path: Path) -> None:
    """O-17 — an arbitrary rc (config/code error, e.g. 7) is propagated with NO relaunch (a
    crash-loop is worse than a loud stop)."""
    sup, spawns, kills, _ = _make_supervisor([FakeChild(1, [7])], hb_file=tmp_path / "hb.json")
    assert sup.run() == 7
    assert len(spawns) == 1 and kills == []


def test_child_exit_42_relaunches_without_kill(tmp_path: Path) -> None:
    """O-12 / O-17 — child rc 42 (stall/livelock, the transient class) ⇒ exactly one relaunch
    with ZERO kill calls (distinct from the stale-seq kill path)."""
    sup, spawns, kills, _ = _make_supervisor(
        [FakeChild(1, [42]), FakeChild(2, [0])], hb_file=tmp_path / "hb.json")
    assert sup.run() == 0
    assert len(spawns) == 2, "one respawn after the 42 exit"
    assert kills == [], "the child-exit-42 path performs ZERO kill calls"


def test_relaunch_budget_exceeded_exits_loud(tmp_path: Path) -> None:
    """O-17 — a child that keeps exiting 42 past `max_relaunches` ⇒ a loud NONZERO exit (a
    crash-loop must not be papered over forever)."""
    children = [FakeChild(i, [42]) for i in range(1, 6)]  # spawns until budget blows
    sup, spawns, kills, _ = _make_supervisor(children, hb_file=tmp_path / "hb.json",
                                             max_relaunches=2)
    rc = sup.run()
    assert rc != 0, "budget exhaustion must be a loud nonzero exit"
    assert len(spawns) == 3, "initial spawn + max_relaunches(2) respawns, then stop"


# ── O-12 stale-seq liveness (distinct from child exit) ────────────────────────────────
def test_frozen_seq_triggers_sigterm_then_sigkill_then_relaunch(tmp_path: Path) -> None:
    """O-12 / P-12 — a running child whose heartbeat-file `seq` is frozen past stale_after ⇒
    exactly one SIGTERM, then SIGKILL after the grace window, then one respawn. Bites a
    supervisor that only reacts to child EXIT (the GIL-starvation hole)."""
    hb = tmp_path / "hb.json"
    write_heartbeat_file(hb, seq=5, pid=100, ages={"train_step": 0.0}, wall_ts=0.0)  # frozen
    wedged = FakeChild(100, [None])        # never exits, never responds to SIGTERM
    recovered = FakeChild(200, [0])        # the relaunched child exits cleanly
    sup, spawns, kills, _ = _make_supervisor([wedged, recovered], hb_file=hb,
                                             stale_after=10.0, poll=1.0, grace=1.0)
    assert sup.run() == 0
    assert [sig for _, sig in kills] == [signal.SIGTERM, signal.SIGKILL], (
        f"stale seq must escalate SIGTERM→SIGKILL, got {kills}"
    )
    assert all(pid == 100 for pid, _ in kills), "only the wedged child is killed"
    assert len(spawns) == 2, "exactly one relaunch after the stale kill"


# ── O-13(ii,iii) LivenessTracker seq/pid semantics ────────────────────────────────────
def test_tracker_mtime_touch_without_seq_change_counts_stale() -> None:
    """O-13(ii) — the tracker keys on `seq` progression only; observing the SAME seq at a
    later time (an mtime touch with no real progress) accrues staleness. Bites an
    mtime/wall-clock-derived liveness decision."""
    from mantis.monitor.heartbeat import read_heartbeat_file  # noqa: F401 — API presence pin

    tracker = LivenessTracker(stale_after_sec=900.0)
    state = _State(seq=5, pid=1)
    tracker.observe(state, now=0.0)
    tracker.observe(state, now=1000.0)     # same seq (mtime touched only)
    assert tracker.is_stale(1000.0) is True


def test_tracker_seq_progress_resets_staleness() -> None:
    """O-13 — a real `seq` advance resets the staleness clock (a live child is never killed)."""
    tracker = LivenessTracker(stale_after_sec=900.0)
    tracker.observe(_State(seq=5, pid=1), now=0.0)
    tracker.observe(_State(seq=6, pid=1), now=800.0)   # progress
    assert tracker.is_stale(1000.0) is False           # 1000 - 800 = 200 < 900


def test_tracker_pid_change_resets_seq_baseline() -> None:
    """O-13(iii) — a `pid` change (child restart) resets the seq baseline; a seq that DROPS to
    a lower value under a new pid is NOT an instant staleness. Bites a supervisor that treats a
    legitimate restart as forgery/stall."""
    tracker = LivenessTracker(stale_after_sec=900.0)
    tracker.observe(_State(seq=50, pid=100), now=0.0)
    tracker.observe(_State(seq=1, pid=200), now=950.0)  # new child: pid changed, seq dropped
    assert tracker.is_stale(960.0) is False, "a pid change must reset the baseline, no instant kill"


class _State:
    """A minimal HeartbeatFileState stand-in (`.seq`, `.pid`) for LivenessTracker unit tests."""

    def __init__(self, *, seq: int, pid: int) -> None:
        self.seq = seq
        self.pid = pid


# ══ RED-TEAM F6 — a pid FLIP must not manufacture freshness ═══════════════════════════
def test_tracker_pid_flip_between_two_writers_does_not_stay_fresh_forever() -> None:
    """RED-TEAM F6 — the pid rebase is ONE-SHOT PER PID. Two writers alternating pids on one
    heartbeat path (`111, 112, 111, 112 …`) with a FROZEN seq kept `is_stale` False forever,
    because every flip re-based the window — a dead child that the supervisor never kills.
    Only the FIRST sighting of a genuinely new pid re-bases."""
    tracker = LivenessTracker(stale_after_sec=10.0)
    tracker.observe(_State(seq=7, pid=111), now=0.0)
    now = 0.0
    for i in range(1, 41):                       # 200 s of alternating pids, seq frozen at 7
        now = i * 5.0
        tracker.observe(_State(seq=7, pid=111 if i % 2 else 112), now=now)
    assert tracker.is_stale(now) is True, (
        "a frozen seq under alternating pids must go STALE — a pid flip is not progress"
    )


def test_tracker_first_sighting_of_each_new_pid_still_rebases() -> None:
    """F6 companion — the fix must not break O-13(iii): a genuine restart chain
    (pid 100 → 200 → 300, each seen once) still re-bases and is never an instant kill."""
    tracker = LivenessTracker(stale_after_sec=10.0)
    tracker.observe(_State(seq=50, pid=100), now=0.0)
    tracker.observe(_State(seq=1, pid=200), now=100.0)
    assert tracker.is_stale(105.0) is False
    tracker.observe(_State(seq=1, pid=300), now=200.0)
    assert tracker.is_stale(205.0) is False


# ══ RED-TEAM F3′ — "never written at all" is a CONFIG fault, not a stall ══════════════
def test_never_written_heartbeat_file_is_reported_distinctly(tmp_path: Path) -> None:
    """RED-TEAM F3′ — a `--heartbeat-file` that does not match the child's makes the
    supervisor kill a perfectly HEALTHY child until the budget is gone. It still kills (a
    child that cannot be observed cannot be trusted), but it must say WHICH fault it is:
    `child_heartbeat_file_never_written` names the path, instead of a generic
    `child_heartbeat_stale` that sends the operator hunting a wedge that does not exist."""
    import io

    child = FakeChild(pid=501, poll_returns=[None] * 50)
    stream = io.StringIO()
    sup, spawns, kills, clock = _make_supervisor(
        [child, FakeChild(pid=502, poll_returns=[0])], hb_file=tmp_path / "absent.json",
        stale_after=5.0, max_relaunches=1,
    )
    sup._stream = stream                          # capture the supervisor's own JSON lines
    sup.run()
    events = [json.loads(ln)["event"] for ln in stream.getvalue().splitlines()]
    assert "child_heartbeat_file_never_written" in events, events
    assert "child_heartbeat_stale" not in events, (
        "a file that was never written is a configuration fault, not a stale heartbeat"
    )


def test_supervisor_loop_survives_a_reader_that_raises(tmp_path: Path) -> None:
    """RED-TEAM F4 (level-2 half) — an exception out of the heartbeat READ must not kill the
    supervisor loop and leave the child unsupervised. The failure is recorded
    (`heartbeat_read_failed`) and treated as "no progress observable" (the safe side)."""
    import io

    def _exploding_reader(_path):
        raise OverflowError("cannot convert float infinity to integer")

    child = FakeChild(pid=601, poll_returns=[None] * 20 + [0])
    stream = io.StringIO()
    sup, _spawns, _kills, clock = _make_supervisor([child], hb_file=tmp_path / "hb.json",
                                                   stale_after=1e9, max_relaunches=1)
    sup._read_heartbeat = _exploding_reader
    sup._stream = stream
    rc = sup.run()                                 # must terminate normally, not explode
    assert rc == 0
    events = [json.loads(ln)["event"] for ln in stream.getvalue().splitlines()]
    assert "heartbeat_read_failed" in events
