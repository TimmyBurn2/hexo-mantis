"""Signal handler registration for the self-play training loop.

One SIGINT/SIGTERM flips ``running=False`` and ``shutdown_save=True`` so the loop saves a
checkpoint before returning; two force-tear-down every registered child and then
``os._exit(1)`` — never ``sys.exit``, whose ``SystemExit`` can propagate out of ``finally``
blocks and leave a spawn-child orphaned at PPID 1.

That teardown, like every other in this tree, requires THE PARENT TO RUN.
``arm_parent_death_signal`` is the half that does not, and
``arm_parent_death_if_supervised`` applies it one layer up at a run's own entry point, gated
on an env-carried supervisor pid — the gate is what makes it safe, since an unconditional arm
on a top-level process kills every unattended run whose launching shell exits.

>300 justify (R8): ONE subject — how this process is asked to STOP — whose three halves share
``ShutdownState``, the registry lock and one exit-code vocabulary. The handler pair writes the
state, the child registry is what the SECOND signal tears down, and the parent-death arming is
that same teardown as a kernel promise; splitting them would put a teardown in one file and
the thing it tears down in another.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Any

from mantis.monitor.heartbeat import PARENT_DEATH_PPID_ENV, PARENT_VANISHED_EXIT_CODE

_LOG = logging.getLogger(__name__)

#: Bounded join timeout (s) for the second-signal force-teardown. Short by design: the
#: first signal already requested cooperative shutdown; the second means "exit NOW."
_FORCE_TEARDOWN_GRACE_SEC = 3.0

_child_lock = threading.Lock()
_children: set[Any] = set()


#: `PR_SET_PDEATHSIG` (linux/prctl.h). Asks the KERNEL to signal this process when its parent
#: THREAD dies — the one teardown that does not require the parent to execute anything.
_PR_SET_PDEATHSIG = 1

#: `PARENT_VANISHED_EXIT_CODE` is defined in `monitor/heartbeat.py` and imported above by
#: design: it became a code the SUPERVISOR reads, so it needs ONE spelling that `monitor/**`
#: can name without importing `train/**`, which is an illegal edge.


@dataclass(frozen=True)
class ParentDeathDecision:
    """WHAT the arming gate decided, and WHY — the record the run publishes.

    A process has exactly ONE arming decision, taken at its entry point before any sink, config
    or run id exists, so it has to be CARRIED to the composition root rather than returned.
    `reason` is a closed vocabulary, one member per branch of the gate.
    """

    armed: bool
    reason: str
    supervisor_pid: int | None
    ppid: int
    chain_depth: int | None
    signal_name: str | None


#: The carrier. Set ONCE by the gate and never cleared: module scope IS the lifetime of "this
#: process's arming decision". A latch and not a signature change, because the alternative
#: threads the decision through `launch_run`, whose caller is a frozen oracle.
_LAST_DECISION: ParentDeathDecision | None = None


def last_parent_death_decision() -> ParentDeathDecision | None:
    """The arming decision this process took, or None if the gate never ran.

    None is meaningful and must stay reachable: `launch_run` is called in-process by tests that
    never go through `main`, and manufacturing an `armed=false` record for those would be a
    SECOND authority for a decision only the gate can take.
    """
    return _LAST_DECISION


def _record(decision: ParentDeathDecision) -> bool:
    """Latch the gate's decision (set-once) and return `decision.armed`."""
    global _LAST_DECISION
    if _LAST_DECISION is None:
        _LAST_DECISION = decision
    return decision.armed


def arm_parent_death_signal(sig: int = signal.SIGKILL) -> bool:
    """Called IN A CHILD at startup: ask the kernel to `sig` us when our parent dies.

    Every other teardown here needs the parent to run code, and none of them execute when it is
    killed outright; a child in its own session receives nothing either, because a new session
    is what puts it out of reach of signals aimed at the parent's group. Measured 2026-08-18: a
    `preflight_mint.py --_boot` child at PPID 1, 4 h 06 m old, 682% CPU, `VmHWM` 13.8 GB, and on
    the migration box the same class holding 458 MiB of a GPU with 0.514 GiB of headroom.

    THE DEFAULT IS `SIGKILL`, AND THAT IS MEASURED: under `SIGTERM` the child's cooperative
    handler reads it as "finish the step and save", so it flipped `running=False` and PARKED
    (%CPU decaying 408 to 133 over two minutes, still alive). Correct here because the path's
    premise is that the parent is ALREADY DEAD — pipes closed, nobody to `wait()`, no route for
    a result. Returns True iff armed; best-effort and NEVER fatal.
    """
    if not sys.platform.startswith("linux"):
        return False
    # CAPTURED BEFORE THE PRCTL, and comparing against the captured value is the ONLY correct
    # form. `getppid() == 1` is a FALSE NEGATIVE under a subreaper (on this host an orphan
    # reparents to `systemd --user`, never to PID 1) and a FALSE POSITIVE inside a PID namespace
    # whose shell IS PID 1, where the parent is alive and well.
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
    # THE RACE, closed: if the parent died between our fork and the prctl above, the death
    # signal was already delivered-and-missed. A CHANGED ppid says exactly that happened, on
    # every host and in every namespace.
    if os.getppid() != ppid_before:
        _LOG.warning(
            "arm_parent_death_signal: parent %d vanished during arming (now %d); exiting %d",
            ppid_before, os.getppid(), PARENT_VANISHED_EXIT_CODE,
        )
        os._exit(PARENT_VANISHED_EXIT_CODE)
    return True


#: Hop ceiling for the ancestry walk. A bound and not a `while True`: `/proc` is live and the
#: walk is not atomic, so it must terminate on a pathological tree too.
_MAX_ANCESTRY_HOPS = 64


def _ppid_of(pid: int) -> int | None:
    """The `PPid` line of `/proc/<pid>/status`; None if unreadable (exited, or not Linux).
    None is "do not know", never "no parent" — the caller must not arm on a guess."""
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

    None means "not found, or the walk could not be completed"; both resolve to "do not arm",
    which is the status quo rather than a new kill. A WRONG arm at depth >= 3 is the outcome
    this function exists to make impossible.
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

    THE GATE IS THE LOAD-BEARING HALF: `PR_SET_PDEATHSIG` on a top-level process ties its life
    to whatever launched it, so an unconditional arm ends every unattended burn whose launching
    shell exits — measured from the other side, where arming the pytest process SIGKILLed the
    whole tier the instant its launcher exited. A run launched DIRECTLY stays orphanable.

    The predicate, in order: no stamp is not-supervised; a non-integer stamp is malformed and
    never read as supervised; `stamp == getppid()` delegates to `arm_parent_death_signal`; a
    stamp that is GONE means the supervisor died between its `Popen` and this line, so nothing
    can be saved and nobody is left to relaunch — exit `PARENT_VANISHED_EXIT_CODE`; a stamp
    alive but not our parent means a wrapper is in the chain and a bounded `/proc` walk decides.

    Depth 2 arms and depth >= 3 does not, because the real question is "is my parent the process
    the trampoline armed?" — derivable from `/proc` alone, since `PR_GET_PDEATHSIG` is self-only.
    At depth 2 the answer is yes by construction; deeper, an unarmed process stands between us
    and the armed one. An unreadable ancestry does not arm either.

    Disclosed: a wrapper that ABANDONS its child now takes the run down, which is the correct
    disposition since such a wrapper already breaks the supervisor's exit-code contract; a
    recycled pid costs one unarmed boot, never a wrong kill; two stacked non-exec wrappers
    defeat the cascade entirely. EVERY BRANCH RECORDS ITS DECISION through `_record`, because
    the only prior record was a log line on the stream that dies with the supervisor.
    """
    raw = os.environ.get(PARENT_DEATH_PPID_ENV)
    if raw is None:
        _LOG.debug(
            "parent_death_signal armed=false reason=not_supervised (%s unset)",
            PARENT_DEATH_PPID_ENV,
        )
        return _record(ParentDeathDecision(
            armed=False, reason="not_supervised", supervisor_pid=None, ppid=os.getppid(),
            chain_depth=None, signal_name=None,
        ))
    try:
        stamped = int(raw)
    except ValueError:
        _LOG.warning(
            "parent_death_signal armed=false reason=malformed_stamp %s=%r",
            PARENT_DEATH_PPID_ENV, raw,
        )
        return _record(ParentDeathDecision(
            armed=False, reason="malformed_stamp", supervisor_pid=None, ppid=os.getppid(),
            chain_depth=None, signal_name=None,
        ))

    actual = os.getppid()
    if stamped == actual:
        armed = arm_parent_death_signal()
        _LOG.info(
            "parent_death_signal armed=%s reason=%s supervisor_pid=%d",
            str(armed).lower(),
            "supervised" if armed else "unsupported_platform",
            stamped,
        )
        return _record(ParentDeathDecision(
            armed=armed, reason="supervised" if armed else "unsupported_platform",
            supervisor_pid=stamped, ppid=actual, chain_depth=1,
            signal_name="SIGKILL" if armed else None,
        ))

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
        return _record(ParentDeathDecision(
            armed=armed,
            reason="wrapper_armed_by_trampoline" if armed else "unsupported_platform",
            supervisor_pid=stamped, ppid=actual, chain_depth=2,
            signal_name="SIGKILL" if armed else None,
        ))

    if depth is None:
        _LOG.warning(
            "parent_death_signal armed=false reason=ancestry_unreadable supervisor_pid=%d "
            "ppid=%d — the stamped supervisor is alive but could not be located in this "
            "process's ancestry, so there is no parent whose death this run may be tied to",
            stamped, actual,
        )
        return _record(ParentDeathDecision(
            armed=False, reason="ancestry_unreadable", supervisor_pid=stamped, ppid=actual,
            chain_depth=None, signal_name=None,
        ))

    _LOG.error(
        "parent_death_signal armed=false reason=wrapper_chain_too_deep supervisor_pid=%d "
        "ppid=%d chain_depth=%d — two or more non-exec wrappers stand between the supervisor "
        "and this run, so the death cascade cannot reach it: THIS RUN WILL SURVIVE ITS "
        "SUPERVISOR and must be reaped by hand if the supervisor is killed outright",
        stamped, actual, depth,
    )
    return _record(ParentDeathDecision(
        armed=False, reason="wrapper_chain_too_deep", supervisor_pid=stamped, ppid=actual,
        chain_depth=depth, signal_name=None,
    ))


def register_child(proc: Any) -> None:
    """Track a live child process for force-teardown on second signal."""
    with _child_lock:
        _children.add(proc)


def unregister_child(proc: Any) -> None:
    """Drop a child that exited or was torn down through the normal path."""
    with _child_lock:
        _children.discard(proc)


def force_teardown_all(*, grace_sec: float = _FORCE_TEARDOWN_GRACE_SEC) -> None:
    """Terminate, bounded join, kill for every registered child.

    Idempotent and best-effort: called from the second-signal handler before ``os._exit(1)``.
    Errors are swallowed, because this is the force-exit path.
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

    Three of the four sites that write `running=False` are CLEAN stops, and nothing on this
    object used to distinguish the fourth: a run that collapsed on a hard abort looked exactly
    like one that finished. `abort_rule` is that distinction, and it carries the RULE NAME
    rather than an exit code so the armed-abort manifest stays the one record of codes, so
    `mantis.train` needs no import of it, and so a rule with no authored code can truthfully
    resolve to `None`.

    Set ONCE and never cleared — a stopped run is never re-decided — with `record_abort` as the
    ONE writer, because a second fire path exists and two assignments would be two authorities
    the first time they disagreed.
    """

    running: bool = True
    stop_count: int = 0
    shutdown_save: bool = False
    abort_rule: str | None = None

    def record_abort(self, rule: str) -> bool:
        """THE writer of ``abort_rule``. Records ``rule`` iff none is recorded yet.

        Returns True iff this call is the one that recorded — FIRST FIRE WINS, and a later fire
        is a no-op rather than an overwrite: the rule that stopped the run is the one that
        stopped it, and a gate resolving afterwards must not re-label a stop that happened.

        It stays a RULE NAME and never a code, and this method imports nothing: `mantis.train`
        must not reach `mantis.config.armed_aborts`.
        """
        if self.abort_rule is not None:
            return False
        self.abort_rule = rule
        return True


def install_signal_handlers(state: ShutdownState) -> None:
    """Install SIGINT/SIGTERM handlers that flip ``state``.

    Two consecutive signals force-tear-down all registered children then ``os._exit(1)``, not
    ``sys.exit`` — ``SystemExit`` can re-enter ``finally`` blocks and leave children orphaned.
    One signal sets ``running=False`` and ``shutdown_save=True``; the training loop polls
    ``state`` between iterations.
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
