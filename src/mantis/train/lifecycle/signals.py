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
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Any

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
