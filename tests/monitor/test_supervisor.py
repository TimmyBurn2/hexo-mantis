# >300 justify (R8): ONE subject — the supervisor's relaunch decision — through ONE scripted
# harness. The exit-code table and the stale-`seq` kill ladder are two routes to that SAME
# decision, and `LivenessTracker`'s seq/pid rows are the input it is computed FROM; cross-test
# imports are barred, so any split also forks the harness into drifting copies.
"""The out-of-process supervisor: a torch-free, host-neutral babysitter.

`Supervisor` takes injected `spawn_fn`/`kill_fn`/`clock` and a `sleep_fn` that advances the fake
clock, so the loop is driven deterministically; `LivenessTracker` is the seq/pid staleness core.
A frozen `seq` and a child exit 42 are DISTINCT paths by construction, and staleness is
seq-keyed, never mtime."""
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


class FakeChild:
    """A scripted child handle: `.poll()` walks `poll_returns`, the last value repeating."""

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


def test_exit_code_equality_pin() -> None:
    """A single restart-wrapper key: the two authorities agree at 42."""
    assert SELFPLAY_STALL_EXIT_CODE == WATCHDOG_STALL_EXIT_CODE == 42


def test_child_exit_zero_stops_rc_zero(tmp_path: Path) -> None:
    """O-17 — child rc 0 ⇒ supervisor exits 0, no relaunch."""
    sup, spawns, kills, _ = _make_supervisor([FakeChild(1, [0])], hb_file=tmp_path / "hb.json")
    assert sup.run() == 0
    assert len(spawns) == 1 and kills == []


def test_child_exit_43_stops_no_relaunch(tmp_path: Path) -> None:
    """Persist-fatal (43) is NOT transient — relaunching would loop the storage fault."""
    sup, spawns, kills, _ = _make_supervisor([FakeChild(1, [PERSIST_FATAL_EXIT_CODE])],
                                             hb_file=tmp_path / "hb.json")
    assert sup.run() == 43
    assert len(spawns) == 1 and kills == []


def test_child_exit_other_rc_propagates_no_relaunch(tmp_path: Path) -> None:
    """An arbitrary rc is propagated with NO relaunch: a crash-loop is worse than a loud stop."""
    sup, spawns, kills, _ = _make_supervisor([FakeChild(1, [7])], hb_file=tmp_path / "hb.json")
    assert sup.run() == 7
    assert len(spawns) == 1 and kills == []


def test_child_exit_42_relaunches_without_kill(tmp_path: Path) -> None:
    """Child rc 42 gives one relaunch with ZERO kill calls, distinct from the stale-seq path."""
    sup, spawns, kills, _ = _make_supervisor(
        [FakeChild(1, [42]), FakeChild(2, [0])], hb_file=tmp_path / "hb.json")
    assert sup.run() == 0
    assert len(spawns) == 2, "one respawn after the 42 exit"
    assert kills == [], "the child-exit-42 path performs ZERO kill calls"


def test_relaunch_budget_exceeded_exits_loud(tmp_path: Path) -> None:
    """A child that keeps exiting 42 past `max_relaunches` exits loud and NONZERO."""
    children = [FakeChild(i, [42]) for i in range(1, 6)]
    sup, spawns, kills, _ = _make_supervisor(children, hb_file=tmp_path / "hb.json",
                                             max_relaunches=2)
    rc = sup.run()
    assert rc != 0, "budget exhaustion must be a loud nonzero exit"
    assert len(spawns) == 3, "initial spawn + max_relaunches(2) respawns, then stop"


def test_frozen_seq_triggers_sigterm_then_sigkill_then_relaunch(tmp_path: Path) -> None:
    """A frozen `seq` past stale_after gets one SIGTERM, then SIGKILL after the grace, then one
    respawn — biting a supervisor that only reacts to child EXIT."""
    hb = tmp_path / "hb.json"
    write_heartbeat_file(hb, seq=5, pid=100, ages={"train_step": 0.0}, wall_ts=0.0)  # frozen
    wedged = FakeChild(100, [None])        # never exits, never responds to SIGTERM
    recovered = FakeChild(200, [0])
    sup, spawns, kills, _ = _make_supervisor([wedged, recovered], hb_file=hb,
                                             stale_after=10.0, poll=1.0, grace=1.0)
    assert sup.run() == 0
    assert [sig for _, sig in kills] == [signal.SIGTERM, signal.SIGKILL], (
        f"stale seq must escalate SIGTERM→SIGKILL, got {kills}"
    )
    assert all(pid == 100 for pid, _ in kills), "only the wedged child is killed"
    assert len(spawns) == 2, "exactly one relaunch after the stale kill"


def test_tracker_mtime_touch_without_seq_change_counts_stale() -> None:
    """The tracker keys on `seq` progression only, so an mtime touch with no real progress still
    accrues staleness."""
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
    tracker.observe(_State(seq=6, pid=1), now=800.0)
    assert tracker.is_stale(1000.0) is False           # 1000 - 800 = 200 < 900


def test_tracker_pid_change_resets_seq_baseline() -> None:
    """A `pid` change resets the seq baseline: a legitimate restart is not forgery."""
    tracker = LivenessTracker(stale_after_sec=900.0)
    tracker.observe(_State(seq=50, pid=100), now=0.0)
    tracker.observe(_State(seq=1, pid=200), now=950.0)  # new child: pid changed, seq dropped
    assert tracker.is_stale(960.0) is False, "a pid change must reset the baseline, no instant kill"


class _State:
    """A minimal HeartbeatFileState stand-in (`.seq`, `.pid`) for LivenessTracker unit tests."""

    def __init__(self, *, seq: int, pid: int) -> None:
        self.seq = seq
        self.pid = pid


def test_tracker_pid_flip_between_two_writers_does_not_stay_fresh_forever() -> None:
    """The pid rebase is ONE-SHOT PER PID: alternating pids under a FROZEN seq kept `is_stale`
    False forever, because every flip re-based the window."""
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
    """A genuine restart chain (pid 100 → 200 → 300, each seen once) still re-bases."""
    tracker = LivenessTracker(stale_after_sec=10.0)
    tracker.observe(_State(seq=50, pid=100), now=0.0)
    tracker.observe(_State(seq=1, pid=200), now=100.0)
    assert tracker.is_stale(105.0) is False
    tracker.observe(_State(seq=1, pid=300), now=200.0)
    assert tracker.is_stale(205.0) is False


def test_never_written_heartbeat_file_is_reported_distinctly(tmp_path: Path) -> None:
    """A `--heartbeat-file` that does not match the child's kills a HEALTHY child until the
    budget is gone; it must NAME that fault rather than report a generic stale heartbeat."""
    import io

    child = FakeChild(pid=501, poll_returns=[None] * 50)
    stream = io.StringIO()
    sup, spawns, kills, clock = _make_supervisor(
        [child, FakeChild(pid=502, poll_returns=[0])], hb_file=tmp_path / "absent.json",
        stale_after=5.0, max_relaunches=1,
    )
    sup._stream = stream
    sup.run()
    events = [json.loads(ln)["event"] for ln in stream.getvalue().splitlines()]
    assert "child_heartbeat_file_never_written" in events, events
    assert "child_heartbeat_stale" not in events, (
        "a file that was never written is a configuration fault, not a stale heartbeat"
    )


def test_supervisor_loop_survives_a_reader_that_raises(tmp_path: Path) -> None:
    """An exception out of the heartbeat READ is recorded and treated as "no progress
    observable", never allowed to kill the loop."""
    import io

    def _exploding_reader(_path):
        raise OverflowError("cannot convert float infinity to integer")

    child = FakeChild(pid=601, poll_returns=[None] * 20 + [0])
    stream = io.StringIO()
    sup, _spawns, _kills, clock = _make_supervisor([child], hb_file=tmp_path / "hb.json",
                                                   stale_after=1e9, max_relaunches=1)
    sup._read_heartbeat = _exploding_reader
    sup._stream = stream
    rc = sup.run()
    assert rc == 0
    events = [json.loads(ln)["event"] for ln in stream.getvalue().splitlines()]
    assert "heartbeat_read_failed" in events


def test_child_exit_48_stops_loud_and_never_relaunches(tmp_path: Path) -> None:
    """Child rc 48 — a terminal eval that produced no promotion decision — is a loud stop, never
    a relaunch, on the existing "any code that is not 0/43/42 propagates" arm. GREEN at HEAD by
    design: relaunching a FINISHED run repeats the whole burst. MUTATION THAT REDS IT: add 48 to
    the relaunch arm — `spawns` grows to 2."""
    import io
    import json as _json

    stream = io.StringIO()
    sup, spawns, kills, _ = _make_supervisor(
        [FakeChild(1, [48]), FakeChild(2, [0])], hb_file=tmp_path / "hb.json")
    sup._stream = stream

    assert sup.run() == 48, (
        "a cooperative armed-abort code PROPAGATES unchanged — rewriting it to a supervisor "
        "code would destroy the very signal the run went to the trouble of authoring"
    )
    assert len(spawns) == 1, (
        f"…and the run is NOT relaunched: it completed, degraded. Spawns: {len(spawns)}"
    )
    assert kills == [], (
        f"a child that exited on its own is never killed; got {kills}"
    )
    stops = [_json.loads(line) for line in stream.getvalue().splitlines()
             if _json.loads(line)["event"] == "supervisor_stop"]
    assert stops and stops[-1]["reason"] == "child_error", (
        "the stop must be NAMED in the supervisor's own stream so an operator can tell it "
        f"from a budget exhaustion or a persist fault; got {stops}"
    )
    assert stops[-1]["code"] == 48, f"…and it must carry the child's code; got {stops}"
