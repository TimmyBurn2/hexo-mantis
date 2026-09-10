"""Out-of-process supervisor — `python -m mantis.monitor.supervise`.

The backstop for the one failure the in-process watchdog cannot cover: a watchdog thread
STARVED by the GIL held in non-yielding native code. A frozen heartbeat ``seq`` means that
thread can no longer run, and only a separate process can act on it.

Two liveness inputs: the child EXIT code (42 relaunch, 0 stop, 43 stop, anything else
propagated with NO relaunch) and ``seq`` progression on this process's OWN monotonic clock,
never mtime and never wall clock. Host-neutral and torch-free by construction.

>300 justify (R8): ONE subject — the out-of-process babysitter — and its three inseparable
halves: the staleness core it decides FROM, the exit-code table it decides BY, and the real
`Popen`/signal collaborators `main` binds INTO it. A split would put the decision in one file
and its premise in another, and would make the file-scoped torch-free property unverifiable.
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
from mantis.monitor.logging_setup import configure_logging

# Relaunch budget exhausted: a loud, distinct nonzero code (never confused with the child's).
RELAUNCH_BUDGET_EXIT_CODE: int = 44


class LivenessTracker:
    """Staleness core: keys on heartbeat ``seq`` progression, on the caller's clock.

    A regressed or unchanged seq under an already-seen pid accrues staleness, so a pid FLIP
    buys no freshness: two writers alternating pids kept a frozen `seq` looking healthy.
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
        """True once ANY heartbeat state has been read for the current child; False means a
        `--heartbeat-file` that does not match the child's, which is a config fault rather
        than a stall."""
        return self.observations > 0

    def is_stale(self, now: float) -> bool:
        """True once ``stale_after_sec`` has elapsed with no seq progression."""
        if self._last_progress is None:
            return False
        return (float(now) - self._last_progress) >= self.stale_after_sec


class Supervisor:
    """Spawn the child, watch its exit code and its heartbeat ``seq``, relaunch or stop.

    Collaborators are injected so the loop is deterministically drivable; `main()` binds real ones.
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
        # The LIVE child handle, so the module-level stop ladder can reach it when a signal or
        # an escaping exception unwinds `run()`. The loop itself never reads it.
        self.child: Any = None

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
            # The child's arming gate found the pid that stamped it already gone: nothing to
            # relaunch it INTO, so a crash-loop would be the only possible outcome.
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
        """Read the heartbeat file; a failed read counts as "no progress observable", because
        one exception on this duck-typed seam would kill the loop and unsupervise the child."""
        try:
            return self._read_heartbeat(self._heartbeat_file)
        except Exception as exc:  # noqa: BLE001 — level 2 must outlive a corrupt file
            self._emit("heartbeat_read_failed", error=repr(exc),
                       heartbeat_file=str(self._heartbeat_file))
            return None

    def _emit(self, event: str, **fields: Any) -> None:
        """Write one JSON line per action to the supervisor's OWN stream, never the child's."""
        stream = self._stream if self._stream is not None else sys.stderr
        line = json.dumps({"event": event, "ts": time.time(), **fields}, default=str)
        stream.write(line + "\n")
        stream.flush()


def spawn_child(child_argv: Sequence[str]) -> subprocess.Popen[bytes]:
    """Launch the child through the arming trampoline — no shell, no baked path.

    The child PROGRAM's `sys.argv` is byte-for-byte the argv after `--`; the one `Popen`-level
    prefix `execvp`s itself away before the child's first instruction.

    WHY THE TRAMPOLINE: `PR_SET_PDEATHSIG` is cleared across `fork` and preserved across
    `execve`, and `uv run` does not exec — so a plain `Popen` made the run a grandchild nobody
    had promised to kill (measured: `kill -9` here left a running GPU holder behind).

    A COPY of the environment carries this supervisor's pid; mutating `os.environ` would stamp
    every later child of this process with a parent that is merely "some ancestor".

    MAIN-THREAD CALL is a precondition: `PR_SET_PDEATHSIG` fires on the parent THREAD's death,
    so a child spawned off a worker thread dies the moment that thread returns. NEW SESSION is
    the other half: otherwise a `Ctrl-C` reaches the run twice and reads as a force-exit second
    press. Disclosed cost: `kill -INT -<pgid>` no longer reaches the run, which also has no
    controlling terminal.
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
    # THE REFUSALS ARE MANDATORY: with the trampoline in front, a bad child command would make
    # `Popen` SUCCEED and turn a launcher typo into a `child_error` rc read as a run diagnosis.
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


class _SupervisorStop(BaseException):
    """Raised FROM the stop handler to unwind `Supervisor.run`'s poll sleep.

    `BaseException` so it cannot be swallowed by an `except Exception` on the way through —
    the same family as `KeyboardInterrupt`.
    """

    def __init__(self, signum: int, press: int) -> None:
        super().__init__(signum)
        self.signum = int(signum)
        self.press = int(press)


#: Press counter for the second-signal affordance, mirrored at the supervisor. A list and not
#: an `int` because the handler must mutate it without a `global`.
_PRESSES: list[int] = [0]

#: Failures of the emergency-stop path itself. `monitor/**` allows no `except ...: pass`, and
#: the commonest reason to be on this path is that the stream its emits write to has died.
_STOP_COUNTERS = BestEffortCounters()


def _on_stop_signal(signum: int, _frame: Any) -> None:
    _PRESSES[0] += 1
    raise _SupervisorStop(signum, _PRESSES[0])


def _install_stop_handlers() -> None:
    """Install the stop handler for SIGINT, SIGTERM and SIGHUP. Called by `main`, NEVER at
    import, which would install handlers in every process that imports this module.

    SIGHUP is deliberate: the child has its own session, so a terminal close reaches only the
    supervisor, and dying by default would have the run's armed PDEATHSIG kill it unsaved.
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

    THE LADDER `Supervisor._kill` CANNOT BECOME: its frozen oracle's `FakeChild` has no
    `wait()`, so only here can the wait be bounded AND early-returning rather than an
    unconditional `sleep(grace)`. Always SIGTERM outbound whatever signal arrived — the run
    registers one handler for both, and one outbound vocabulary is one thing to reason about.

    The bound is the minted `monitor.supervisor_kill_grace_sec` and has no second authority;
    `--kill-grace-sec` is an override published in `supervisor_boot_identity`.

    A further signal lands as `_SupervisorStop` INSIDE the wait: the second re-forwards SIGTERM
    so the run's own `force_teardown_all` runs, and the third SIGKILLs. Returns the exit code,
    or None for a child unreapable inside the bound — a `D`-state pathology worth saying rather
    than blocking on forever.
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
    it at ourselves, so the waiter sees "terminated by SIGTERM" and no number is minted. The
    trailing `os._exit` is reached only if the signal was blocked upstream."""
    sys.stderr.flush()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
    os._exit(128 + int(signum))   # pragma: no cover — only if the signal is blocked


#: The two signals this module's ladder can put in a child's `wait()` code (as `-signum`); it
#: sends nothing else. A negative code outside this set is a genuine child diagnosis.
_LADDER_SIGNALS: frozenset[int] = frozenset({int(signal.SIGTERM), int(signal.SIGKILL)})


def _stop_and_exit(supervisor: Supervisor, stop: _SupervisorStop) -> int:
    """The stop path: the cooperative ladder, then the rc decision.

    THE CHILD'S DIAGNOSIS OUTRANKS THE STOP GESTURE, but only a genuine one. A positive child
    code propagates as `_on_child_exit` would, with the relaunch suppressed. `-SIGTERM` and
    `-SIGKILL` are this ladder's OWN gesture landing back on the child, not a diagnosis, and
    propagating them made `main` return `-9`, i.e. exit status 247 — they fall through to the
    die-of-signal path. Any other negative code keeps its own code.
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
    """Stop the child cooperatively while an exception escapes, then let it continue unchanged.

    `_emit` writes to stderr on every event, so a `BrokenPipeError` here used to unwind the
    main thread in milliseconds and have the KERNEL SIGKILL a healthy run mid-save. The ladder
    gets a TOLERANT emit, which is why this wrapper exists: the commonest way to reach this
    path is that the sink itself died, and a ladder that emits first would raise before ever
    sending the SIGTERM.
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
        # An operator signalled us DURING the emergency stop. Counting it is the honest record;
        # re-raising would replace the original failure with the gesture.
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


#: The named refusal for a missing `--config`: absent is an error, never a default. It names
#: the missing input AND where a minted one comes from, which argparse's own line does not.
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
#: carry NO code-side default, and anything supplied is PUBLISHED in the boot event.
_OVERRIDABLE: tuple[tuple[str, str], ...] = (
    ("stale_after_sec", "supervisor_stale_after_sec"),
    ("poll_interval_sec", "supervisor_poll_interval_sec"),
    ("kill_grace_sec", "supervisor_kill_grace_sec"),
    ("max_relaunches", "supervisor_max_relaunches"),
)


def _require_config(flags: Sequence[str]) -> None:
    """Refuse a missing `--config` BY NAME, before argparse can pre-empt the message.

    With `required=True`, `parse_args` exits on the generic line the instant the flag is
    absent, so a named check written after it is unreachable.
    """
    for index, flag in enumerate(flags):
        if flag.startswith("--config="):
            return
        if flag == "--config":
            # PRESENT IS NOT SUPPLIED: a trailing `--config` satisfies a presence test and
            # falls through to argparse's stock "expected one argument", which has no remedy.
            if index + 1 < len(flags) and not flags[index + 1].startswith("-"):
                return
            raise SystemExit(_CONFIG_REFUSAL)
    raise SystemExit(_CONFIG_REFUSAL)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: the MINTED config is the threshold authority; a flag overrides and SAYS SO.

    `main` used to use a bare `MonitorConfig()`'s dataclass literals as the argparse defaults,
    so the four minted `monitor.supervisor_*` keys reached no process. The signal posture is
    installed here and nowhere else: without handlers the supervisor's own catchable death had
    the KERNEL SIGKILL a healthy run mid-save. Three exits, none inventing a number — a caught
    stop takes the ladder then dies OF that signal, an escaping exception stops the child then
    re-raises unchanged, everything else keeps `Supervisor.run`'s exit-code contract.
    """
    # The one mantis stderr handler, at this process entry too: a supervisor whose own INFO
    # lines are dropped cannot report why it relaunched.
    configure_logging()
    # LAZY BY NECESSITY: a top-level `mantis.config` import here closes the cycle
    # `mantis.config -> mantis.monitor -> mantis.config`.
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

    # Host-neutrality is intact: the operator supplies the path, as they already supply
    # --heartbeat-file and the child argv. A *defaulted* config path would breach it.
    config = load_config(args.config)
    thresholds = resolve_monitor_config(config.monitor)
    supplied = {dest: getattr(args, dest) for dest, _ in _OVERRIDABLE}
    overrides = {dest: value for dest, value in supplied.items() if value is not None}
    # A NON-FINITE OR NEGATIVE BOUND IS MALFORMED: `Popen.wait(timeout=nan)` never raises
    # `TimeoutExpired`, so the ladder silently stops escalating. This does NOT bound the grace —
    # `1e308` passes and fails the same way; an upper bound is an operator value.
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
        # THE PARENT-SIDE IDENTITY WITNESS, first EVENT out, through the same authority as the
        # run's own `run_boot_identity`. Handlers arm BEFORE it and it is INSIDE the `try`, so a
        # signal landing during it takes one of the three documented exits. PUBLISH, not
        # COMPARE: the child's config is not readable without parsing its verbatim argv.
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
