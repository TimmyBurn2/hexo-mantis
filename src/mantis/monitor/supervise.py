"""Out-of-process supervisor (WP13-A §c.5, L-C) — `python -m mantis.monitor.supervise`.

The backstop for the one failure the in-process watchdog cannot cover: a watchdog thread
STARVED by the GIL held in non-yielding native code. The child's watchdog thread mirrors
its heartbeats to a file with a monotonic ``seq``; a frozen ``seq`` means "the watchdog
thread itself can no longer run", and only a separate process can act on that.

Two liveness inputs, deliberately distinct:
  * the child EXIT code — 42 relaunch (stall/livelock is the transient class), 0 stop,
    43 stop (a persistent-storage fault relaunches straight back into itself), any other
    code propagated with NO relaunch (a crash-loop is worse than a loud stop);
  * heartbeat-file ``seq`` PROGRESSION, measured on the supervisor's OWN monotonic clock —
    never file mtime, never wall clock (mtime-forgery and NTP-skew immune, O-13); a ``pid``
    change resets the baseline, because a legitimate restart is not a stall.

Host-neutral by construction: the child command is the verbatim argv after ``--``; no
default paths, no provider names, no baked launcher (§7). Torch-free (O-18) — a liveness
babysitter must not need seconds and gigabytes to load on a box whose GPU just wedged.

>300 justify (R8): ONE subject — the out-of-process babysitter — and its three inseparable
halves: the staleness core it decides FROM, the exit-code table it decides BY, and the real
`Popen`/signal collaborators `main` binds INTO it. Splitting would put the decision in one
file and its premise in another, and the whole unit has a second, file-scoped property that a
split would make unverifiable: it must stay torch-free (O-18), which is a claim about this
file as a unit, not about any one of its parts.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from mantis.monitor.config import MonitorConfig
from mantis.monitor.heartbeat import (
    PARENT_DEATH_PPID_ENV,
    PERSIST_FATAL_EXIT_CODE,
    WATCHDOG_STALL_EXIT_CODE,
    read_heartbeat_file,
)

# Relaunch budget exhausted: a loud, distinct nonzero code (never confused with the child's).
RELAUNCH_BUDGET_EXIT_CODE: int = 44


class LivenessTracker:
    """Staleness core: keys on heartbeat ``seq`` progression, on the caller's clock.

    `observe` records progress when the seq ADVANCES (same pid) or when a NEW pid is seen
    for the first time (a restart re-bases the baseline — not forgery, not an instant
    stall). An unchanged or regressed seq under an already-seen pid accrues staleness, so
    neither an mtime touch nor a pid FLIP buys freshness: two writers alternating pids on
    one heartbeat path used to keep a frozen `seq` looking healthy forever (RED-TEAM F6),
    because every flip re-based the window.
    """

    def __init__(self, *, stale_after_sec: float) -> None:
        self.stale_after_sec = float(stale_after_sec)
        self._seq: int | None = None
        self._pid: int | None = None
        self._seen_pids: set[int] = set()
        self._last_progress: float | None = None
        self.observations = 0

    def reset(self, now: float) -> None:
        """Re-base after a (re)spawn: the new child gets a full staleness window."""
        self._seq = None
        self._pid = None
        self._seen_pids.clear()
        self._last_progress = float(now)
        self.observations = 0

    def observe(self, state: Any, now: float) -> None:
        """Fold one heartbeat-file state in at time ``now`` (a None state is ignored)."""
        if state is None:
            return
        seq = int(state.seq)
        pid = int(state.pid)
        self.observations += 1
        # A pid rebase is ONE-SHOT per pid: the first sighting of a genuinely new child
        # re-bases, a flip back to a pid we have already watched does not.
        new_pid = pid not in self._seen_pids
        progressed = self._seq is None or new_pid or (pid == self._pid and seq > self._seq)
        self._seen_pids.add(pid)
        self._pid = pid
        self._seq = seq
        if progressed:
            self._last_progress = float(now)

    @property
    def ever_observed(self) -> bool:
        """True once ANY heartbeat state has been read for the current child.

        False means the file was never written at all — a configuration fault (a
        `--heartbeat-file` that does not match the child's) rather than a stall, and the
        supervisor says so distinctly instead of reporting a stale heartbeat (RED-TEAM F3′).
        """
        return self.observations > 0

    def is_stale(self, now: float) -> bool:
        """True once ``stale_after_sec`` has elapsed with no seq progression."""
        if self._last_progress is None:
            return False
        return (float(now) - self._last_progress) >= self.stale_after_sec


class Supervisor:
    """Spawn the child, watch its exit code and its heartbeat ``seq``, relaunch or stop.

    Collaborators are injected (``spawn_fn``/``kill_fn``/``clock``/``sleep_fn``/
    ``read_heartbeat``) so the whole loop is drivable deterministically in tests; `main()`
    binds the real ones.
    """

    def __init__(
        self,
        *,
        child_argv: Sequence[str],
        heartbeat_file: Path | str,
        stale_after_sec: float,
        poll_interval_sec: float,
        kill_grace_sec: float,
        max_relaunches: int,
        spawn_fn: Callable[[Sequence[str]], Any],
        kill_fn: Callable[[Any, int], None],
        clock: Callable[[], float],
        sleep_fn: Callable[[float], Any] = time.sleep,
        read_heartbeat: Callable[[Path], Any] = read_heartbeat_file,
        stream: TextIO | None = None,
    ) -> None:
        self._child_argv = list(child_argv)
        self._heartbeat_file = Path(heartbeat_file)
        self._poll_interval = float(poll_interval_sec)
        self._kill_grace = float(kill_grace_sec)
        self._max_relaunches = int(max_relaunches)
        self._spawn_fn = spawn_fn
        self._kill_fn = kill_fn
        self._clock = clock
        self._sleep_fn = sleep_fn
        self._read_heartbeat = read_heartbeat
        self._stream = stream
        self._tracker = LivenessTracker(stale_after_sec=stale_after_sec)
        self.relaunches = 0
        self.spawns = 0

    # ── the loop ─────────────────────────────────────────────────────────────────────
    def run(self) -> int:
        """Supervise until a terminal decision; returns the supervisor's exit code."""
        child = self._spawn()
        while True:
            code = child.poll()
            if code is not None:
                decision = self._on_child_exit(int(code))
                if decision is not None:
                    return decision
                child = self._spawn()
                continue

            state = self._read_state()
            now = self._clock()
            self._tracker.observe(state, now=now)
            if self._tracker.is_stale(now):
                self._emit(
                    "child_heartbeat_stale" if self._tracker.ever_observed
                    else "child_heartbeat_file_never_written",
                    pid=getattr(child, "pid", None),
                    stale_after_sec=self._tracker.stale_after_sec,
                    heartbeat_file=str(self._heartbeat_file),
                )
                self._kill(child)
                if not self._claim_relaunch("heartbeat_stale"):
                    return RELAUNCH_BUDGET_EXIT_CODE
                child = self._spawn()
                continue
            self._sleep_fn(self._poll_interval)

    def _on_child_exit(self, code: int) -> int | None:
        """Apply the exit-code contract; ``None`` means "relaunch" (the caller respawns)."""
        self._emit("child_exited", code=code)
        if code == 0:
            return 0
        if code == PERSIST_FATAL_EXIT_CODE:
            # A persistence fault is NOT transient — relaunching just loops the failure.
            self._emit("supervisor_stop", reason="persist_fatal", code=code)
            return PERSIST_FATAL_EXIT_CODE
        if code != WATCHDOG_STALL_EXIT_CODE:
            self._emit("supervisor_stop", reason="child_error", code=code)
            return code
        if not self._claim_relaunch("child_exit_stall"):
            return RELAUNCH_BUDGET_EXIT_CODE
        return None

    # ── collaborators ────────────────────────────────────────────────────────────────
    def _spawn(self) -> Any:
        child = self._spawn_fn(list(self._child_argv))
        self.spawns += 1
        self._tracker.reset(now=self._clock())
        self._emit("child_spawned", pid=getattr(child, "pid", None), spawns=self.spawns)
        return child

    def _kill(self, child: Any) -> None:
        """SIGTERM → grace → SIGKILL (a GIL-starved child may never run its handler)."""
        self._kill_fn(child, signal.SIGTERM)
        self._sleep_fn(self._kill_grace)
        if child.poll() is None:
            self._kill_fn(child, signal.SIGKILL)
            self._emit("child_sigkilled", pid=getattr(child, "pid", None))

    def _claim_relaunch(self, reason: str) -> bool:
        """Consume one relaunch from the budget; False ⇒ stop loud."""
        if self.relaunches >= self._max_relaunches:
            self._emit(
                "relaunch_budget_exhausted", reason=reason, max_relaunches=self._max_relaunches
            )
            return False
        self.relaunches += 1
        self._emit("child_relaunching", reason=reason, relaunches=self.relaunches)
        return True

    def _read_state(self) -> Any:
        """Read the heartbeat file. `read_heartbeat_file` contracts never to raise, but this
        is LEVEL 2 of the livelock protection and an injected reader is a duck-typed seam:
        one exception here would kill the supervisor loop and leave the child unsupervised
        (RED-TEAM F4 did exactly that with an `Infinity` seq). A failed read is counted as
        "no progress observable" — the safe side, which errs toward a relaunch."""
        try:
            return self._read_heartbeat(self._heartbeat_file)
        except Exception as exc:  # noqa: BLE001 — level 2 must outlive a corrupt file
            self._emit("heartbeat_read_failed", error=repr(exc),
                       heartbeat_file=str(self._heartbeat_file))
            return None

    def _emit(self, event: str, **fields: Any) -> None:
        """One JSON line per action on the supervisor's OWN stream (never the child's
        event sink — separate process, separate file identity)."""
        stream = self._stream if self._stream is not None else sys.stderr
        line = json.dumps({"event": event, "ts": time.time(), **fields}, default=str)
        stream.write(line + "\n")
        stream.flush()


# ── real collaborators + CLI ─────────────────────────────────────────────────────────
def spawn_child(child_argv: Sequence[str]) -> subprocess.Popen[bytes]:
    """Launch the child verbatim — no shell, no injected flags, no baked path.

    ONE thing is injected and it is NOT argv: the child's ENVIRONMENT carries this
    supervisor's pid under `PARENT_DEATH_PPID_ENV`, which is how a mantis run learns it is
    supervised and may arm `PR_SET_PDEATHSIG` (F-816-19). Until this existed, a supervisor
    killed outright — `kill -9`, an OOM kill, a dropped session — ORPHANED the run: the
    process holding the GPU, the trainer, the worker pool and the buffer kept running with
    nobody watching it.

    A COPY of the environment is built and passed; `os.environ` is NEVER mutated, and that is
    not tidiness. A mutation would leak the stamp to every LATER child of this process — in a
    test session, to children that are not runs at all — and each of them would then read a
    stamp naming a parent that is merely "some ancestor", which is the exact confusion the
    child's gate exists to refuse.

    MAIN-THREAD CALL, and it is a real precondition rather than a nicety: `PR_SET_PDEATHSIG`
    fires when the parent THREAD dies, so a child spawned from a worker thread here would be
    SIGKILLed the moment that thread returned — a premature kill of a healthy run, strictly
    worse than the orphan the arming prevents. The supervisor is single-threaded today; this
    refusal is what makes that fact fail LOUD instead of silently arming a premature kill.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "spawn_child was called from thread "
            f"{threading.current_thread().name!r}, not the main thread. The child is stamped "
            f"with {PARENT_DEATH_PPID_ENV} so it may arm PR_SET_PDEATHSIG, and the kernel "
            "signals on the death of the CREATING THREAD — a child spawned from a worker "
            "thread is SIGKILLed as soon as that thread returns, killing a healthy run. If "
            "the supervisor ever needs to spawn off its main thread, the child's arming gate "
            "must be re-derived against a thread identity FIRST."
        )
    env = {**os.environ, PARENT_DEATH_PPID_ENV: str(os.getpid())}
    return subprocess.Popen(list(child_argv), env=env)


def signal_child(child: Any, sig: int) -> None:
    child.send_signal(sig)


def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split ``[flags…] -- CHILD_ARGV…``; the child argv is taken VERBATIM."""
    args = list(argv)
    if "--" not in args:
        raise SystemExit(
            "usage: python -m mantis.monitor.supervise [flags] -- CHILD_ARGV...  "
            "(the child command after `--` is required)"
        )
    cut = args.index("--")
    child = args[cut + 1:]
    if not child:
        raise SystemExit("no child argv given after `--`")
    return args[:cut], child


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: flag defaults are the `MonitorConfig` fields (single authority)."""
    defaults = MonitorConfig()
    flags, child_argv = _split_argv(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m mantis.monitor.supervise",
        description="Host-neutral liveness supervisor for a mantis run.",
    )
    parser.add_argument("--heartbeat-file", required=True)
    parser.add_argument("--stale-after-sec", type=float,
                        default=defaults.supervisor_stale_after_sec)
    parser.add_argument("--poll-interval-sec", type=float,
                        default=defaults.supervisor_poll_interval_sec)
    parser.add_argument("--kill-grace-sec", type=float,
                        default=defaults.supervisor_kill_grace_sec)
    parser.add_argument("--max-relaunches", type=int,
                        default=defaults.supervisor_max_relaunches)
    args = parser.parse_args(flags)
    supervisor = Supervisor(
        child_argv=child_argv,
        heartbeat_file=Path(args.heartbeat_file),
        stale_after_sec=args.stale_after_sec,
        poll_interval_sec=args.poll_interval_sec,
        kill_grace_sec=args.kill_grace_sec,
        max_relaunches=args.max_relaunches,
        spawn_fn=spawn_child,
        kill_fn=signal_child,
        clock=time.monotonic,
    )
    return supervisor.run()


if __name__ == "__main__":  # pragma: no cover — process entry point
    raise SystemExit(main())
