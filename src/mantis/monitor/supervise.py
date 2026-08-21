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

Host-neutral by construction: the child PROGRAM sees the verbatim argv after ``--`` as its own
``sys.argv``; no default paths, no provider names, no baked launcher (§7). The ``Popen``-level
argv now carries one prefix — the arming trampoline, which `execvp`s into the given command and
so is gone by the time that command runs (see `spawn_child`). Torch-free (O-18) — a liveness
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
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from mantis.monitor.best_effort import BestEffortCounters, best_effort
from mantis.monitor.heartbeat import (
    PARENT_DEATH_ARM_EXEC_MODULE,
    PARENT_DEATH_PPID_ENV,
    PARENT_VANISHED_EXIT_CODE,
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
        # The LIVE child handle, published so the module-level stop ladder can reach it when a
        # signal or an escaping exception unwinds `run()`. Additive and externally inert: the
        # loop never reads it, and nothing in the frozen fake-driven oracle knows it exists.
        self.child: Any = None

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
        if code == PARENT_VANISHED_EXIT_CODE:
            # The child's own arming gate found the pid that stamped it already gone — this
            # supervisor died between its `Popen` and the child's entry point, and what is
            # reading this is a RELAUNCHED or otherwise later supervisor. Named rather than
            # swallowed into the catch-all below: it is the one rc that says the child never
            # began, so there is nothing to relaunch it INTO and a crash-loop would be the only
            # possible outcome. This arm is also the only artifact the exit-71 path can leave
            # anywhere — the run had no sink, no out-dir and no run id when it took it.
            self._emit("supervisor_stop", reason="child_parent_vanished", code=code)
            return PARENT_VANISHED_EXIT_CODE
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
        self.child = child
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
    """Launch the child through the arming trampoline — no shell, no baked path.

    THE ARGV CONTRACT, AT THE LEVEL IT ACTUALLY HOLDS. The child PROGRAM's own `sys.argv` is
    byte-for-byte the argv given after `--`; the `Popen`-level argv carries exactly one prefix,
    `python -m PARENT_DEATH_ARM_EXEC_MODULE --`, and that prefix `execvp`s itself out of
    existence before the child program's first instruction. Nothing the run records — argv in
    provenance included — sees it. Saying "verbatim" without naming the level would be the kind
    of stale claim the R8 header rules exist to prevent.

    WHY THE TRAMPOLINE (Q3 A4b). `PR_SET_PDEATHSIG` is cleared across `fork` and preserved
    across `execve`. `uv run` — this repo's own idiom — does NOT exec, so under a plain `Popen`
    the supervisor's DIRECT child was the wrapper and the run was a grandchild nobody had
    promised to kill: measured, `kill -9` on the supervisor left a running GPU holder behind.
    The trampoline arms and then becomes the wrapper, so the wrapper dies with the supervisor;
    the run's own gate arms against its direct parent when that parent is the trampoline's
    process, and the two together take the whole chain down. Either half ALONE is inert, which
    is why they land together.

    ONE thing is injected and it is NOT the child's argv: the child's ENVIRONMENT carries this
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
    The MIRROR case is the same hazard read from the other end and is why this guard has to
    stay: a supervisor that grew a NON-DAEMON thread outliving its main thread would arm the
    child against a thread that returns while the supervisor is still alive and working, and
    the kernel would SIGKILL a healthy run under a supervisor that never asked for it.

    NEW SESSION, and it is the second half of the signal posture rather than a spawn detail.
    Without it the child shares the terminal's process group, so a `Ctrl-C` is delivered to the
    RUN by the tty AND forwarded to it by this supervisor's own stop ladder — two deliveries
    the run cannot tell apart from an operator's deliberate second press, which is LAW-16's
    force-exit: `os._exit(1)` mid-save. With it, every signal the child receives comes from
    this supervisor, exactly once, and `stop_count` counts presses instead of routes. In-tree
    precedent for the same reasoning: `tools/ci_gates/preflight_mint.py`'s `--_boot` child.
    The cost is disclosed: `kill -INT -<pgid>` no longer reaches the run directly (a directed
    `kill <run-pid>` still does), and the run has no controlling terminal — nothing in
    `src/mantis/` reads one. `PR_SET_PDEATHSIG` is session-independent, so the arming is
    unaffected.
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
    # THE REFUSALS ARE MANDATORY, not tidiness. With the trampoline in front, a bad child
    # command would make `Popen` SUCCEED and move the failure into the child — turning a
    # launcher typo into a `child_error` rc the supervisor reads as the run's own diagnosis.
    # Resolving the program here keeps both failures exactly where they were before the
    # trampoline existed: loud, in the supervisor, before anything is spawned.
    argv = list(child_argv)
    if not argv or not str(argv[0]):
        raise ValueError(
            "spawn_child requires a non-empty program as the first element of the child argv"
        )
    if shutil.which(argv[0]) is None:
        raise FileNotFoundError(
            f"the child program {argv[0]!r} is not resolvable on this host; the supervisor "
            "refuses to spawn it rather than let the arming trampoline fail as the child"
        )
    env = {**os.environ, PARENT_DEATH_PPID_ENV: str(os.getpid())}
    return subprocess.Popen(
        [sys.executable, "-m", PARENT_DEATH_ARM_EXEC_MODULE, "--", *argv],
        env=env,
        start_new_session=True,
    )


def signal_child(child: Any, sig: int) -> None:
    child.send_signal(sig)


# ── the supervisor's own signal posture ──────────────────────────────────────────────
class _SupervisorStop(BaseException):
    """Raised FROM the stop handler to unwind `Supervisor.run`'s poll sleep.

    `BaseException` and not `Exception` on purpose: it must not be swallowed by an
    `except Exception` anywhere it passes through. It is a control-flow token for a stop the
    operator asked for — the same family as `KeyboardInterrupt`, which is exactly what CPython
    raises for the identical event when the default handler is left in place.
    """

    def __init__(self, signum: int, press: int) -> None:
        super().__init__(signum)
        self.signum = int(signum)
        self.press = int(press)


#: Press counter for LAW-16's second-signal affordance, mirrored at the supervisor. A list and
#: not an `int` because the handler must mutate it without a `global`; process-scoped, which is
#: exactly the lifetime of "how many times has the operator asked this supervisor to stop".
_PRESSES: list[int] = [0]

#: Failures of the emergency-stop path itself. `monitor/**` allows NO `except …: pass` (O-20,
#: censused): an optional effect is either counted through `best_effort` or fails loud, and the
#: stop path's own emits are the textbook optional effect — the commonest reason to be on that
#: path at all is that the stream they write to has died.
_STOP_COUNTERS = BestEffortCounters()


def _on_stop_signal(signum: int, _frame: Any) -> None:
    _PRESSES[0] += 1
    raise _SupervisorStop(signum, _PRESSES[0])


def _install_stop_handlers() -> None:
    """Install the stop handler for SIGINT, SIGTERM and SIGHUP. Called by `main`, NEVER at
    import: `signal.signal` at module scope would install handlers in every process that so
    much as imports this module, pytest included.

    SIGHUP is in the set deliberately and is not scope creep. `spawn_child` puts the child in
    its own session, so on a terminal close the SUPERVISOR is the only process that receives
    SIGHUP, and its default disposition is to terminate — after which the run's armed
    `PR_SET_PDEATHSIG` SIGKILLs it, unsaved. Omitting SIGHUP would have this module CREATE that
    hole on the "close the terminal" gesture; with the handler it is a cooperative save.
    """
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_stop_signal)


def stop_child_cooperatively(
    child: Any,
    *,
    grace_sec: float,
    emit: Callable[..., None],
    send: Callable[[Any, int], None] = signal_child,
) -> int | None:
    """SIGTERM → bounded wait → SIGKILL → bounded reap, on a REAL child handle.

    THIS IS THE LADDER `Supervisor._kill` CANNOT BECOME, and the reason is mechanical rather
    than stylistic: `_kill` is driven by a frozen oracle whose `FakeChild` exposes `.pid` and
    `.poll()` and nothing else, so a `wait()` there is an `AttributeError` in a HELD test. Kept
    as a module-level function over the real `Popen`, it can use `wait(timeout=…)` — which is
    what makes the wait BOUNDED AND EARLY-RETURNING instead of `_kill`'s unconditional
    `sleep(grace)`, i.e. what lets a healthy child actually finish its save.

    ALWAYS SIGTERM OUTBOUND, whatever signal this supervisor received: the run registers one
    handler for SIGINT and SIGTERM alike, the child has no controlling terminal, and one
    outbound vocabulary is one thing to reason about.

    THE BOUND IS `grace_sec` AND THERE IS NO SECOND AUTHORITY FOR IT — it is the MINTED
    `monitor.supervisor_kill_grace_sec`, loaded from the run's own config by `main` and resolved
    through `resolve_monitor_config`. (This sentence used to name `main`'s `--kill-grace-sec`
    flag as the arrival route, and until F-816-24 that route began at a bare dataclass default,
    so the claim of a single authority was the thing it denied: the minted key reached nothing.
    The flag survives as an OVERRIDE, and one that is published in `supervisor_boot_identity`
    rather than applied silently.) Whether the value is ADEQUATE against a measured >320 s
    cooperative drain is a prereg/mint question (RQ-7, prereg row 19) and is deliberately NOT
    answered here; this function is written so that answering it is a change to one number in
    one config.

    LAW-16 mirrored, not reimplemented: a further signal arriving while we wait lands as
    `_SupervisorStop` INSIDE the wait. The second re-forwards SIGTERM — which is the run's OWN
    second press, so the run's `force_teardown_all` still tears down its registered mp children
    rather than leaving them to the kernel — and the third stops waiting and SIGKILLs.

    Returns the child's exit code, or None if it could not be reaped inside the bound. The
    supervisor NEVER blocks forever: an unreapable child is a `D`-state pathology and the
    correct posture is to say so and leave.
    """
    pid = getattr(child, "pid", None)
    grace = float(grace_sec)
    emit("supervisor_forwarding_stop", signum=int(signal.SIGTERM), child_pid=pid,
         grace_sec=grace)
    send(child, signal.SIGTERM)
    presses = 1
    while True:
        try:
            code = child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            emit("supervisor_child_stop_timeout", child_pid=pid, grace_sec=grace)
            break
        except _SupervisorStop as stop:
            presses += 1
            if presses >= 3:
                emit("supervisor_force_kill", signum=stop.signum, press=presses, child_pid=pid)
                break
            emit("supervisor_force_stop", signum=stop.signum, press=presses, child_pid=pid)
            send(child, signal.SIGTERM)
            continue
        else:
            emit("supervisor_child_stopped", code=int(code), child_pid=pid)
            return int(code)

    send(child, signal.SIGKILL)
    # The same event name `_kill` already emits, so an existing reader needs no new vocabulary.
    emit("child_sigkilled", pid=pid)
    try:
        code = child.wait(timeout=grace)
    except (subprocess.TimeoutExpired, _SupervisorStop):
        emit("supervisor_child_unreaped", child_pid=pid)
        return None
    emit("supervisor_child_stopped", code=int(code), child_pid=pid)
    return int(code)


def _die_of(signum: int) -> None:
    """Die OF the signal we were asked to die of: restore the default disposition and re-raise
    it at ourselves. NO NUMBER IS MINTED — the waiter (a shell, `systemd`, a job scheduler) sees
    "terminated by SIGTERM", which is the truth and what `systemctl stop` expects, and
    `repo_design.md`'s rule that a signal-caused clean stop resolves to 0 at the RUN stays true
    end to end.

    The trailing `os._exit` is reached only if the signal was blocked or ignored upstream of
    this process, in which case 128+n is the shell's own long-standing encoding of "died of
    signal n" rather than a code this repo authored.
    """
    sys.stderr.flush()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
    os._exit(128 + int(signum))   # pragma: no cover — only if the signal is blocked


#: The two signals `stop_child_cooperatively`'s own ladder can put in the child's `wait()`
#: code (as `-signum`) — it sends nothing else, ever (see that function). A negative code
#: outside this set was NOT caused by our escalation and is a genuine child diagnosis.
_LADDER_SIGNALS: frozenset[int] = frozenset({int(signal.SIGTERM), int(signal.SIGKILL)})


def _stop_and_exit(supervisor: Supervisor, stop: _SupervisorStop) -> int:
    """The stop path: cooperative ladder, then the rc decision (fix-design §2.4, RED-TEAM
    addendum 2 fix).

    THE CHILD'S DIAGNOSIS OUTRANKS THE STOP GESTURE — but only a genuine one. A disk-guard 47
    or a terminal-eval-broken 48 recorded during the drain must not be erased by the fact that
    an operator also pressed Ctrl-C, so a positive child code is propagated exactly as
    `_on_child_exit` would propagate it, with the relaunch suppressed (a stop is a stop).

    A NEGATIVE code is `Popen.wait()`'s "died of signal N" shape, and `stop_child_cooperatively`
    ONLY EVER sends SIGTERM then, on escalation, SIGKILL — so `-SIGTERM`/`-SIGKILL` here is our
    OWN stop gesture landing back on the child, not a diagnosis. Propagating it as `child_error`
    made `main` return e.g. `-9`, and `SystemExit(-9)` is exit status 247 — on the grace-timeout
    and third-press paths, the common ones under the shipped 30 s grace, defeating `_die_of`'s
    own contract that the waiter sees "died of SIGTERM" (RED-TEAM addendum 2, BROKE-IT). Those
    two codes fall through to the die-of-signal path below instead.

    A negative code OUTSIDE `_LADDER_SIGNALS` (e.g. `-11` for a child that SIGSEGVs on its own,
    mid-drain, independent of anything this ladder sent) is still a genuine diagnosis and stays
    `child_error` with its real code named — never relabelled as "died of the stop gesture",
    and never routed through `_die_of` to mint an unrelated 128+n. A 0 or an unreapable child
    (`code is None`) leaves nothing to report but the gesture itself, and we die of it.
    """
    child = supervisor.child
    code: int | None = None
    if child is not None:
        code = stop_child_cooperatively(
            child, grace_sec=supervisor._kill_grace, emit=supervisor._emit,
        )
    if code is not None and (code > 0 or (code < 0 and -code not in _LADDER_SIGNALS)):
        supervisor._emit("supervisor_stop", reason="child_error", code=int(code),
                         signum=stop.signum)
        return int(code)
    supervisor._emit("supervisor_stop", reason="signal", signum=stop.signum, child_code=code)
    _die_of(stop.signum)
    return 0   # pragma: no cover — `_die_of` does not return


def _best_effort_stop(supervisor: Supervisor) -> None:
    """Stop the child cooperatively on the way out of an escaping exception, then let that
    exception continue unchanged.

    `_emit` writes to stderr on every spawn/exit/stale event, so a `BrokenPipeError` (the log
    consumer went away) or a `MemoryError` here used to unwind the supervisor's main thread in
    milliseconds — and with the run now armed against this process, the KERNEL would SIGKILL a
    healthy run mid-save. Nothing is swallowed but a failure of the stop itself: the original
    exception is re-raised by the caller, so the supervisor still dies loud with its own
    traceback.

    THE LADDER GETS A TOLERANT EMIT HERE, and that is the whole reason this wrapper exists
    rather than a direct call. The commonest way to reach this path is that the SINK ITSELF
    died, so a ladder whose first act is an `_emit` would raise again before it ever sent the
    SIGTERM — the child would then be killed by the kernel exactly as before the fix, and the
    row that proves otherwise would be green for the wrong reason. Losing the stop-path events
    on a dead stream is the correct trade: they had no reader anyway.
    """
    child = getattr(supervisor, "child", None)
    if child is None:
        return

    def _tolerant(event: str, **fields: Any) -> None:
        best_effort("supervisor_stop_emit", lambda: supervisor._emit(event, **fields),
                    counters=_STOP_COUNTERS)

    try:
        best_effort(
            "supervisor_stop_child",
            lambda: stop_child_cooperatively(
                child, grace_sec=supervisor._kill_grace, emit=_tolerant,
            ),
            counters=_STOP_COUNTERS,
        )
    except _SupervisorStop:
        # An operator signalled us DURING the emergency stop. The child already has its
        # SIGTERM; counting this is the honest record, and re-raising would replace the
        # original failure — the one the operator needs to read — with the gesture.
        _STOP_COUNTERS.increment("supervisor_stop_interrupted")


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


#: The named refusal for a missing `--config`. R1/LAW-11: absent is an error, never a default.
#: It names the missing input AND where a minted one comes from — argparse's own "the following
#: arguments are required: --config" names the flag and no remedy, which tells an operator
#: nothing about what a config is.
_CONFIG_REFUSAL = (
    "usage: python -m mantis.monitor.supervise --config PATH --heartbeat-file PATH [flags] "
    "-- CHILD_ARGV...\n"
    "--config is REQUIRED and has no default. The supervisor's thresholds ARE the minted "
    "monitor.supervisor_* keys; a supervisor that invented them would wait a kill grace no "
    "config records, which is the defect this flag exists to close (F-816-24). Pass the same "
    "minted config the run is launched with — mint one with tools/mint_config.py, or use one "
    "of the committed configs under configs/."
)

#: `(argparse dest, MonitorConfig field)` for every threshold a flag may override. The flags
#: carry NO code-side default (R1: a default lives only in the schema field) — `None` means
#: "not supplied", the config supplies the value, and anything actually supplied is PUBLISHED
#: in the boot event rather than silently replacing a minted number.
_OVERRIDABLE: tuple[tuple[str, str], ...] = (
    ("stale_after_sec", "supervisor_stale_after_sec"),
    ("poll_interval_sec", "supervisor_poll_interval_sec"),
    ("kill_grace_sec", "supervisor_kill_grace_sec"),
    ("max_relaunches", "supervisor_max_relaunches"),
)


def _require_config(flags: Sequence[str]) -> None:
    """Refuse a missing `--config` BY NAME, before argparse can pre-empt the message.

    The ordering is the whole point. With `required=True` set, `parse_args` exits with
    argparse's generic line the instant the flag is absent, so a named check written after it is
    unreachable. This pre-scans raw flags exactly as `_split_argv` pre-scans raw argv, and
    `required=True` stays below as a backstop that in practice never fires.
    """
    for index, flag in enumerate(flags):
        if flag.startswith("--config="):
            return
        if flag == "--config":
            # PRESENT IS NOT THE SAME AS SUPPLIED. A trailing `--config` with nothing after it
            # satisfies a presence test and then falls through to argparse's "expected one
            # argument" — a stock message with no remedy in it, which is the half this refusal
            # exists to add (REVIEW(impl) #3).
            if index + 1 < len(flags) and not flags[index + 1].startswith("-"):
                return
            raise SystemExit(_CONFIG_REFUSAL)
    raise SystemExit(_CONFIG_REFUSAL)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: the MINTED config is the threshold authority; a flag overrides and SAYS SO.

    F-816-24. Until this existed, `main` built a bare `MonitorConfig()` and used its dataclass
    literals as the argparse defaults, so the four minted `monitor.supervisor_*` keys reached no
    process: a `supervisor_kill_grace_sec` written into `configs/` changed nothing about the
    grace this program actually waits. The config is now REQUIRED, read through the ONE loader,
    and resolved through the ONE resolver.

    THE SIGNAL POSTURE IS INSTALLED HERE AND NOWHERE ELSE. Until this existed the supervisor
    had NO handlers at all, so its own catchable death — an operator's `kill`, a terminal
    close, a `BrokenPipeError` out of `_emit` — killed it instantly and, with the child now
    armed against it, had the KERNEL SIGKILL a healthy run mid-save. The run's own
    save-then-exit path (LAW-16) was reachable only by signalling the run directly, which is
    not what an operator supervising a run does.

    Three exits, and none of them invents a number:
      * a caught stop  -> the cooperative ladder, then die OF that signal (`_stop_and_exit`);
      * an escaping exception -> stop the child cooperatively FIRST, then re-raise unchanged,
        so the supervisor still dies loud with its own traceback;
      * everything else -> `Supervisor.run`'s existing exit-code contract, untouched.
    """
    # LAZY BY NECESSITY, NOT BY TASTE — and gate 9's own rule is "top-level imports only; lazy
    # imports need a stated reason", so here is the reason. A top-level `mantis.config` import
    # in this module closes a condensed-subpackage cycle, because `config/resolve/monitor.py`
    # imports `mantis.monitor.config`. Measured, by moving the import up and running the gate:
    # `CYCLE: mantis.config -> mantis.monitor -> mantis.config`. Deferring it here also keeps
    # `import mantis.monitor.supervise` as cheap as it was — but note that makes the IMPORT-time
    # O-18 check trivially true, so the load-bearing torch check is the RUN-time one, taken from
    # inside a process that has actually executed this function.
    from mantis.config.loader import config_identity_sha256, load_config
    from mantis.config.resolve.monitor import resolve_monitor_config

    flags, child_argv = _split_argv(sys.argv[1:] if argv is None else argv)
    _require_config(flags)
    parser = argparse.ArgumentParser(
        prog="python -m mantis.monitor.supervise",
        description="Host-neutral liveness supervisor for a mantis run.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--heartbeat-file", required=True)
    parser.add_argument("--stale-after-sec", type=float, default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=None)
    parser.add_argument("--kill-grace-sec", type=float, default=None)
    parser.add_argument("--max-relaunches", type=int, default=None)
    args = parser.parse_args(flags)

    # HOST-NEUTRALITY IS INTACT (§7): the operator supplies the path, exactly as they already
    # supply --heartbeat-file and the whole child argv. Nothing is discovered, nothing is baked,
    # no path is defaulted — a *defaulted* config path is what would breach it, not a required one.
    config = load_config(args.config)
    thresholds = resolve_monitor_config(config.monitor)
    supplied = {dest: getattr(args, dest) for dest, _ in _OVERRIDABLE}
    overrides = {dest: value for dest, value in supplied.items() if value is not None}
    # A NON-FINITE OR NEGATIVE BOUND IS MALFORMED, AND REFUSING IT IS NOT AUTHORING A VALUE.
    #
    # WHAT THIS DOES NOT DO, STATED BECAUSE AN EARLIER COMMENT HERE OVERCLAIMED IT. This refuses
    # MALFORMED values. It does NOT bound the grace. `--kill-grace-sec 1e308` is finite and
    # non-negative, passes this check, and then produces exactly the NaN failure — driven:
    # `wait(timeout=1e308)` never fires, so the automatic escalation never runs and only the
    # operator's second and third signal recover the child. And the hazard is NOT specific to
    # the override path: `MonitorSchemaConfig.supervisor_kill_grace_sec` carries `Field(ge=0)`
    # and nothing else, so the same value is MINTABLE. Closing the class needs an upper bound,
    # i.e. a number — which is an operator/architect value under R119, on a key whose derivation
    # is already prereg row 19's. Filed as `F-816-27`; deliberately not decided here.
    # D2's rule is that this mechanism publishes and does not JUDGE — it takes no view on
    # whether 5 s or 500 s is the right grace, because that is an operator's to mint (R119).
    # Well-formedness is a different question. `--kill-grace-sec=nan` was driven end to end by
    # RED-TEAM: `Popen.wait(timeout=nan)` never raises `TimeoutExpired`, so the ladder's
    # automatic SIGTERM -> grace -> SIGKILL escalation silently stops escalating and only the
    # operator's second and third signal recover the child. That converts a LAW-16 bounded stop
    # into an unbounded one through an input nothing validated. The schema already refuses these
    # values for the minted keys (`Field(ge=0)`); the flags had no such guard, which is the
    # duplicate-authority asymmetry this packet exists to close, pointing the other way.
    malformed = sorted(
        dest for dest, value in overrides.items()
        if not math.isfinite(float(value)) or float(value) < 0
    )
    if malformed:
        raise SystemExit(
            f"{', '.join(f'--{d.replace(chr(95), chr(45))}' for d in malformed)}: a supervisor "
            "bound must be finite and non-negative. The minted keys are schema-bounded the same "
            "way (ge=0); a NaN or negative grace does not loosen the stop ladder, it disables "
            "its automatic escalation entirely."
        )
    effective = {
        dest: (supplied[dest] if supplied[dest] is not None else getattr(thresholds, field))
        for dest, field in _OVERRIDABLE
    }

    supervisor = Supervisor(
        child_argv=child_argv,
        heartbeat_file=Path(args.heartbeat_file),
        spawn_fn=spawn_child,
        kill_fn=signal_child,
        clock=time.monotonic,
        **effective,
    )
    _install_stop_handlers()
    try:
        # THE PARENT-SIDE IDENTITY WITNESS (F-B1 class), published as the first EVENT —
        # the same one authority (`config_identity_sha256`) as the run's own `run_boot_identity`.
        #
        # HANDLERS ARE INSTALLED BEFORE THIS EMIT, matching `run.py`, whose signal handlers are
        # hoisted above its own identity publication. An earlier draft published first and called
        # itself "the parent-side twin" while inverting the twin's order (REVIEW(impl) #1); the
        # window was benign — no child exists until `run()` — but it had grown to cover `load_config`
        # and schema validation, which is real file I/O the old in-memory path did not do, and a
        # deviation nobody wrote down is one nobody can weigh later.
        #
        # IT IS ALSO INSIDE THE `try`, and that is the second half of the same argument. `main`'s
        # docstring promises three exits, and all three are scoped to this block; an earlier draft
        # emitted ABOVE it, so a signal landing during the emit — the statement immediately after
        # the handlers arm, i.e. reachable — escaped as a raw `_SupervisorStop` traceback, taking
        # none of the three documented paths and bypassing `_die_of`'s "died of signal N"
        # convention (RED-TEAM #4, driven). Nothing is orphaned either way, because no child
        # exists until `run()`; what was wrong was that the contract had a hole in it. Once this program reads a config, parent and child read config files
        # INDEPENDENTLY and nothing makes them the same file; publishing the parent's identity is what
        # makes a divergence visible instead of invisible. It is PUBLISH, not COMPARE: learning the
        # child's config would mean parsing the verbatim child argv, which `spawn_child`'s contract
        # forbids (an env-channel handshake that breaches nothing is possible and is filed as
        # F-816-26, but its half lives in the child).
        #
        # The effective bounds are read back OFF THE LIVE OBJECT, never off the args namespace: the
        # defect being closed here is precisely that a config and a process disagreed, so the record
        # names what this supervisor will actually do. `overrides` names any flag an operator
        # supplied, so a hand-variation of a minted safety bound is an event in the record rather
        # than a silent substitution.
        supervisor._emit(
            "supervisor_boot_identity",
            config=str(args.config),
            config_sha256=config_identity_sha256(config),
            effective={
                "stale_after_sec": supervisor._tracker.stale_after_sec,
                "poll_interval_sec": supervisor._poll_interval,
                "kill_grace_sec": supervisor._kill_grace,
                "max_relaunches": supervisor._max_relaunches,
            },
            overrides=overrides,
        )
        return supervisor.run()
    except _SupervisorStop as stop:
        return _stop_and_exit(supervisor, stop)
    except BaseException:
        _best_effort_stop(supervisor)
        raise


if __name__ == "__main__":  # pragma: no cover — process entry point
    raise SystemExit(main())
