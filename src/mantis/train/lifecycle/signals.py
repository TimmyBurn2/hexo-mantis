"""Signal handler registration for the self-play training loop (repo_design §11).

Wraps SIGINT/SIGTERM → cooperative shutdown state. Two presses force-exit; one press
flips ``running=False`` and ``shutdown_save=True`` so the loop saves a checkpoint before
returning (save-then-exit). Ported verbatim from the old `training/signals.py`
(structlog → stdlib logging is the only change — no behaviour change).

CARD-ORPHAN-WORKERS (R230): the second-signal path now force-tears-down all registered
child processes (terminate → bounded join → kill) before ``os._exit(1)`` — the old
``sys.exit(1)`` raised ``SystemExit`` which could propagate out of ``finally`` blocks and
leave the eval pipeline's spawn-child orphaned (PPID=1, CPU-pinned). The eval pipeline
registers its spawn child via ``register_child``/``unregister_child``.

F-816-14 (R284(f)): R230's teardown, and every other teardown in this tree, requires THE PARENT
TO RUN. ``arm_parent_death_signal`` is the half that does not — the child asks the kernel to
signal it when its parent dies, so a parent that is killed outright still takes its children
with it. It is a belt beside R230's braces, not a replacement for them: the cooperative path
remains the one that saves state.

F-816-19 (R285(h)): ``arm_parent_death_if_supervised`` applies that same mechanism ONE LAYER
UP, at a run's own entry point, gated on an env-carried supervisor pid. The gate is what makes
it safe there — an unconditional arm on a top-level process kills every unattended run whose
launching shell exits, which this repo has already measured from the pytest side.

>300 justify (R8): ONE subject — how this process is asked to STOP — and its three halves are
inseparable by construction: the cooperative handler pair writes the shutdown state, the child
registry is what the SECOND signal tears down before ``os._exit``, and the parent-death arming
is the same teardown expressed as a kernel promise for the case where this process runs
nothing at all. The three share ``ShutdownState``, the registry lock and one exit-code
vocabulary; splitting them would put a teardown in one file and the thing it tears down in
another, and the handler would have to import the registry back across the split.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Any

from mantis.monitor.heartbeat import PARENT_DEATH_PPID_ENV

_LOG = logging.getLogger(__name__)

#: Bounded join timeout (s) for the second-signal force-teardown. Short by design: the
#: first signal already requested cooperative shutdown; the second means "exit NOW."
_FORCE_TEARDOWN_GRACE_SEC = 3.0

_child_lock = threading.Lock()
_children: set[Any] = set()


#: `PR_SET_PDEATHSIG` (linux/prctl.h). Asks the KERNEL to signal this process when its parent
#: THREAD dies — the one teardown that does not require the parent to execute anything.
_PR_SET_PDEATHSIG = 1

#: Exit code for "my parent vanished while I was arming". NOT 0: an exit that did none of the
#: work it was started for is not a success, and a waiter reading 0 would record a completed
#: child. Distinct from the run's own abort codes so it cannot be confused with one.
PARENT_VANISHED_EXIT_CODE = 71


def arm_parent_death_signal(sig: int = signal.SIGKILL) -> bool:
    """Called IN A CHILD at startup: ask the kernel to `sig` us when our parent dies.

    THE DEFECT THIS CLOSES (F-816-14, R284(f)). Every teardown this module already has —
    `force_teardown_all`, the eval pipeline's `terminate → join → kill`, the preflight tool's
    `os.killpg` — has one thing in common: **the parent must run code**. When the parent is
    killed outright (a harness timeout, an OOM kill, `kill -9`, an interrupted session) none of
    them execute, and a child in its own session receives nothing, because a new session is
    precisely what puts it out of reach of signals aimed at the parent's group. The kernel then
    reparents it to init and it runs without bound.

    That is not a hypothesis. MEASURED on the local host 2026-08-18: a `preflight_mint.py`
    `--_boot` child spawned by a test with `--timeout-sec 45.0` was found at **PPID 1, 4 h 06 m
    old, 682% CPU, `VmHWM` 13.8 GB** — and it reproduced immediately when a pytest tier carrying
    a preflight row was killed. On the migration box the same class held **458 MiB of a GPU**
    whose partition has 0.514 GiB of headroom, which is what R284(f) calls partition-threatening.
    `PR_SET_PDEATHSIG` is the only one of the three candidate shapes that survives its parent
    being killed, and that is the whole reason it is the one chosen.

    THE DEFAULT IS `SIGKILL`, AND THAT IS A MEASURED CHOICE, NOT A BLUNT ONE. The first version
    of this function defaulted to `SIGTERM`, on the reasoning that LAW-16 says signals
    save-then-exit. Driven end to end against a real `preflight_mint.py --_boot` child whose
    parent was SIGKILLed, it **did not work**: the signal was delivered, but that child installs
    the cooperative handler, so `SIGTERM` means *"finish the step and save"* — it flipped
    `running=False` and PARKED (measured: `%CPU` decaying 408 → 133 over two minutes in state
    `Ssl`, still alive). A death signal a process can convert into a park is not a death signal.

    `SIGKILL` is correct for THIS path specifically, because of what the path's premise is: the
    parent is ALREADY DEAD. The child's stdout/stderr pipes are closed, nobody will `wait()` for
    it, its report has no reader, and its result cannot be routed anywhere. LAW-16's
    save-then-exit governs a signal sent to a run whose parent is alive and wants a checkpoint;
    it is untouched, and that path still runs through `install_signal_handlers` exactly as
    before. `sig` stays a parameter so a caller with a genuine save obligation can choose
    otherwise, but it must then answer the question this default already answers: save to whom?

    Returns True iff the signal is armed. Best-effort by construction and NEVER fatal: this is a
    belt beside existing braces, and a child that refuses to start because a defence-in-depth
    teardown is unavailable would be a worse failure than the one it prevents. Non-Linux hosts
    and kernels without prctl simply return False.
    """
    if not sys.platform.startswith("linux"):
        return False
    # CAPTURED BEFORE THE PRCTL, and this is the whole of the race check. Comparing against the
    # captured value is the ONLY correct form; two earlier shapes were both wrong and both were
    # caught by review rather than by running:
    #
    #   * `getppid() == 1` is a FALSE NEGATIVE under a subreaper. On this very development host
    #     an orphan reparents to `systemd --user` (PID 1237, `PR_SET_CHILD_SUBREAPER`), never to
    #     PID 1 — so the check would have been inert on the exact machine where F-816-14 was
    #     found. Docker `--init` / tini behave the same way.
    #   * `getppid() == 1` is also a FALSE POSITIVE inside a PID namespace whose shell IS PID 1
    #     (`docker run img sh -c ...`, many CI images, `kubectl exec`). There the parent is alive
    #     and well, and the earlier version exited the process anyway — with rc 0.
    ppid_before = os.getppid()
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int] * 5
        libc.prctl.restype = ctypes.c_int
        if libc.prctl(_PR_SET_PDEATHSIG, int(sig), 0, 0, 0) != 0:
            _LOG.debug("arm_parent_death_signal: prctl failed errno=%d", ctypes.get_errno())
            return False
    except Exception:  # noqa: BLE001 — see the docstring: never fatal, never re-raised
        _LOG.debug("arm_parent_death_signal: prctl unavailable", exc_info=True)
        return False
    # THE RACE, closed. If the parent died between our fork and the prctl above, the death signal
    # was already delivered-and-missed and we would inherit the immortality this function exists
    # to prevent. A CHANGED ppid says exactly that happened, on every host and in every namespace.
    if os.getppid() != ppid_before:
        _LOG.warning(
            "arm_parent_death_signal: parent %d vanished during arming (now %d); exiting %d",
            ppid_before, os.getppid(), PARENT_VANISHED_EXIT_CODE,
        )
        os._exit(PARENT_VANISHED_EXIT_CODE)
    return True


#: Hop ceiling for the ancestry walk below. A bound and not a `while True`: `/proc` is a live
#: filesystem and a walk over it is not atomic, so the loop must terminate on a pathological
#: tree as surely as on a normal one. 64 is far past any real launcher chain.
_MAX_ANCESTRY_HOPS = 64


def _ppid_of(pid: int) -> int | None:
    """The `PPid` line of `/proc/<pid>/status`; None if unreadable (exited, or not Linux).

    None is "do not know", never "no parent" — the caller must not arm on a guess.
    """
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _ancestry_depth_of(stamped: int) -> int | None:
    """Hops from THIS process up to `stamped`: 1 = direct parent, 2 = one wrapper between us.

    None means "not found, or the walk could not be completed" — an ancestor that exits
    mid-walk, a `/proc` that cannot be read, or a chain that reaches init without matching.
    Both cases must resolve to "do not arm", which is today's status quo rather than a new
    kill; a WRONG arm at depth ≥ 3 is the outcome this function exists to make impossible.
    """
    pid = os.getpid()
    for depth in range(1, _MAX_ANCESTRY_HOPS + 1):
        parent = _ppid_of(pid)
        if parent is None:
            return None
        if parent == stamped:
            return depth
        if parent <= 1:
            return None
        pid = parent
    return None


def arm_parent_death_if_supervised() -> bool:
    """Called at the TOP of a run's own entry point: arm `PR_SET_PDEATHSIG` iff a mantis
    supervisor spawned us. Returns True iff armed. Never raises.

    THE DEFECT (F-816-19, R285(h)). `mantis.monitor.supervise` launched the run with a plain
    `Popen` and no arming at all, so a supervisor killed outright left the run — the process
    that holds the GPU, the trainer, the worker pool and the replay buffer — running with
    nobody watching it. `arm_parent_death_signal` above is the mechanism that closes it; this
    is the GATE that makes applying the mechanism HERE safe.

    THE GATE IS THE LOAD-BEARING HALF, and it is not caution — it is a measured constraint.
    `PR_SET_PDEATHSIG` on a TOP-LEVEL process ties that process's life to whatever launched
    it, so an unconditional arm at a launcher's entry ends every unattended burn whose
    launching shell exits after handing off. This repo has measured that exact failure from
    the other side: arming the pytest process SIGKILLed the whole tier the instant its
    launcher exited (`tests/train/test_parent_death_signal.py`, the subprocess-probe docstring
    added by eae0fc4). A run launched DIRECTLY is therefore still orphanable, deliberately:
    that is the operator's own choice of detachment and nothing here overrides it.

    The predicate, in order:

      * no stamp                     -> not supervised; no arming, no side effect. This is the
                                        arm pytest and every direct launch take.
      * a stamp that is not an int   -> malformed input is never silently read as "supervised";
                                        WARN and do not arm.
      * stamp == `os.getppid()`      -> the supervised case; delegate to
                                        `arm_parent_death_signal`, which performs the prctl AND
                                        its own captured-ppid re-check.
      * stamp != ppid, stamp GONE    -> the supervisor died between its `Popen` and this line.
                                        Nothing has been built and nothing can be saved, and no
                                        supervisor is left to relaunch or to watch — exit
                                        `PARENT_VANISHED_EXIT_CODE`, the eae0fc4 posture one
                                        layer up.
      * stamp != ppid, stamp ALIVE   -> a wrapper is in the chain; the DEPTH decides, by a
                                        bounded `/proc` ancestry walk (Q3 A4b):
                                          depth 2  -> ARM against the DIRECT parent. Our parent
                                                      IS the supervisor's direct child, i.e.
                                                      the process the supervisor's arming
                                                      trampoline armed, so tying our life to
                                                      it is exactly the cascade that takes the
                                                      whole chain down.
                                          depth ≥3 -> DO NOT ARM (`wrapper_chain_too_deep`).
                                                      Two stacked non-exec wrappers defeat the
                                                      chain: measured, and the disclosed
                                                      residual. Escalating this to a boot-time
                                                      REFUSAL needs an rc, and minting one
                                                      inside a fix packet is the drift the
                                                      red-team is raising — queued as RQ-9.
                                          unknown  -> DO NOT ARM (`ancestry_unreadable`).

    WHY DEPTH 2 IS SAFE TO ARM AND DEPTH 3 IS NOT. The predicate is exactly "is my parent the
    process the trampoline armed?", and it is derivable from `/proc` alone — which matters,
    because `PR_GET_PDEATHSIG` is self-only and a process cannot ask whether its parent is
    armed. At depth 2 the answer is yes by construction of the supervisor's own spawn; at
    depth ≥ 3 there is an unarmed process between us and the armed one, and arming would tie
    the run to a wrapper whose death nobody has promised.

    THE HAZARD THAT IS NOT NEW, disclosed: a wrapper that ABANDONS its child (exits while the
    run continues) would now take the run down with it. Such a wrapper already breaks the
    supervisor's exit-code contract — the supervisor reads the wrapper's exit as the child's
    and may relaunch a SECOND run into the same out-dir — so killing the abandoned run is the
    correct disposition, not a premature kill.

    PID REUSE, disclosed rather than defended: a pid recycled in the microseconds between the
    supervisor's `Popen` and this check would downgrade an exit-`PARENT_VANISHED_EXIT_CODE`
    into a warning. The consequence is a log line and one unarmed boot — today's status quo —
    never a wrong kill.

    Never fatal, inheriting `arm_parent_death_signal`'s contract unchanged: off Linux, or on a
    kernel without prctl, it returns False and the run proceeds.
    """
    raw = os.environ.get(PARENT_DEATH_PPID_ENV)
    if raw is None:
        _LOG.debug(
            "parent_death_signal armed=false reason=not_supervised (%s unset)",
            PARENT_DEATH_PPID_ENV,
        )
        return False
    try:
        stamped = int(raw)
    except ValueError:
        _LOG.warning(
            "parent_death_signal armed=false reason=malformed_stamp %s=%r",
            PARENT_DEATH_PPID_ENV, raw,
        )
        return False

    actual = os.getppid()
    if stamped == actual:
        armed = arm_parent_death_signal()
        _LOG.info(
            "parent_death_signal armed=%s reason=%s supervisor_pid=%d",
            str(armed).lower(),
            "supervised" if armed else "unsupported_platform",
            stamped,
        )
        return armed

    try:
        os.kill(stamped, 0)
    except ProcessLookupError:
        _LOG.warning(
            "parent_death_signal armed=false reason=supervisor_vanished supervisor_pid=%d "
            "ppid=%d; exiting %d — the supervisor died before this run reached its entry "
            "point, so nothing is built, nothing can be saved and nobody is left to watch",
            stamped, actual, PARENT_VANISHED_EXIT_CODE,
        )
        os._exit(PARENT_VANISHED_EXIT_CODE)
    except (PermissionError, OSError):
        # The pid exists but is not ours to signal — alive for the purpose of this check.
        pass

    depth = _ancestry_depth_of(stamped)
    if depth == 2:
        armed = arm_parent_death_signal()
        _LOG.info(
            "parent_death_signal armed=%s reason=%s supervisor_pid=%d ppid=%d chain_depth=2 — "
            "the stamping supervisor is our GRANDparent, so our direct parent is the process "
            "its arming trampoline armed; tying our life to that parent is the cascade",
            str(armed).lower(),
            "wrapper_armed_by_trampoline" if armed else "unsupported_platform",
            stamped, actual,
        )
        return armed

    if depth is None:
        _LOG.warning(
            "parent_death_signal armed=false reason=ancestry_unreadable supervisor_pid=%d "
            "ppid=%d — the stamped supervisor is alive but could not be located in this "
            "process's ancestry, so there is no parent whose death this run may be tied to",
            stamped, actual,
        )
        return False

    _LOG.error(
        "parent_death_signal armed=false reason=wrapper_chain_too_deep supervisor_pid=%d "
        "ppid=%d chain_depth=%d — two or more non-exec wrappers stand between the supervisor "
        "and this run, so the death cascade cannot reach it: THIS RUN WILL SURVIVE ITS "
        "SUPERVISOR and must be reaped by hand if the supervisor is killed outright",
        stamped, actual, depth,
    )
    return False


def register_child(proc: Any) -> None:
    """Track a live child process for force-teardown on second signal."""
    with _child_lock:
        _children.add(proc)


def unregister_child(proc: Any) -> None:
    """Drop a child that exited or was torn down through the normal path."""
    with _child_lock:
        _children.discard(proc)


def force_teardown_all(*, grace_sec: float = _FORCE_TEARDOWN_GRACE_SEC) -> None:
    """Terminate → bounded join → kill for every registered child (CARD-ORPHAN-WORKERS).

    Idempotent and best-effort: called from the second-signal handler before
    ``os._exit(1)``. Each child gets ``terminate()`` → ``join(grace)`` → (if still alive)
    ``kill()`` → ``join(grace)``. Errors are swallowed — this is the force-exit path.
    """
    with _child_lock:
        procs = list(_children)
    for proc in procs:
        try:
            if not proc.is_alive():
                continue
            proc.terminate()
            proc.join(grace_sec)
            if proc.is_alive():
                proc.kill()
                proc.join(grace_sec)
        except Exception:  # noqa: BLE001 — best-effort during force-exit; never re-raise
            pass
        finally:
            with _child_lock:
                _children.discard(proc)


@dataclass
class ShutdownState:
    """The cooperative stop flag, and WHY it was flipped.

    `running=False` is written by four sites in `train/coordinator/step.py` — `stop()`, the
    O2 iteration-limit (clean completion), the O3 signal shutdown-save, and
    `_fire_hard_abort`. Three of the four are CLEAN stops, and until WPMINT Phase X
    (CARD-ABORT-EXIT / R84) nothing on this object distinguished the fourth: a run that
    collapsed on the draw-rate abort was indistinguishable, in state and in exit status, from
    a run that finished its last step.

    `abort_rule` is that distinction, and it carries the RULE NAME rather than an exit code —
    deliberately, on three grounds:

    * the manifest (`mantis.config.armed_aborts.MANIFEST`) already carries `exit_code` per
      row, so a number here would be a SECOND place recording which code an abort uses;
    * it keeps `mantis.train` free of a `mantis.config.armed_aborts` import — the coordinator
      records WHAT FIRED and the rule -> code mapping is resolved at the process boundary,
      where the manifest already lives (gate 9 verifiable);
    * `_fire_hard_abort` is shared by `grad_norm_hard_abort` and `sealbot_wr_abort`, neither
      of which is pre-registered with an authored code. Under a rule-name carrier the
      resolver truthfully returns `None` for them; under a code carrier this object would
      have had to answer "which code for grad_norm?" with a number that does not exist.

    It is set ONCE, by the fire that stops the run, and never cleared: a stopped run is never
    re-decided. Until WPMAIN's RT-2 pass that invariant was PROSE, held up by a single write
    site and by `_fire_hard_abort`'s `hard_abort_after_stop` arm returning before it. `record_abort`
    below is that invariant expressed as the ONE writer, because RT-2 added a second fire path
    (the disk guard's, recorded by the composition root) and two assignments would have been two
    authorities for "which rule stopped this run" the first time they disagreed.
    """

    running: bool = True
    stop_count: int = 0
    shutdown_save: bool = False
    abort_rule: str | None = None

    def record_abort(self, rule: str) -> bool:
        """THE writer of ``abort_rule``. Records ``rule`` iff none is recorded yet.

        Returns True iff this call is the one that recorded — FIRST FIRE WINS, and a later
        fire is a no-op rather than an overwrite. That direction is deliberate: the rule that
        stopped the run is the one that stopped it, and a second gate resolving afterwards
        (the teardown-routed eval result, a guard tick racing the handler) must not re-label
        a stop that already happened. `_fire_hard_abort` already refuses to re-decide a
        stopped run; this is the same rule where the two fire paths meet.

        It stays a RULE NAME and never a code (the class docstring's three grounds are
        unchanged), and this method imports nothing: `mantis.train` must not reach
        `mantis.config.armed_aborts`, and the rule -> code resolution stays at the process
        boundary where the manifest already lives.
        """
        if self.abort_rule is not None:
            return False
        self.abort_rule = rule
        return True


def install_signal_handlers(state: ShutdownState) -> None:
    """Install SIGINT/SIGTERM handlers that flip ``state``.

    Two consecutive signals force-teardown all registered child processes then
    ``os._exit(1)`` (not ``sys.exit`` — ``SystemExit`` can re-enter ``finally`` blocks
    and leave children orphaned); one signal sets ``running=False`` and
    ``shutdown_save=True``. The training loop is responsible for polling ``state``
    between iterations.
    """

    def _stop(sig: int, frame: Any) -> None:
        state.stop_count += 1
        if state.stop_count >= 2:
            force_teardown_all()
            os._exit(1)
        _LOG.info(
            "shutdown_requested: finishing current step… press Ctrl+C again to force",
        )
        state.shutdown_save = True
        state.running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
