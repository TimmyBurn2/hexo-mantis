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
"""
from __future__ import annotations

import logging
import os
import signal
import threading
from dataclasses import dataclass
from typing import Any

_LOG = logging.getLogger(__name__)

#: Bounded join timeout (s) for the second-signal force-teardown. Short by design: the
#: first signal already requested cooperative shutdown; the second means "exit NOW."
_FORCE_TEARDOWN_GRACE_SEC = 3.0

_child_lock = threading.Lock()
_children: set[Any] = set()


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
