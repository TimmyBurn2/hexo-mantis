"""Signal handler registration for the self-play training loop (repo_design §11).

Wraps SIGINT/SIGTERM → cooperative shutdown state. Two presses force-exit; one press
flips ``running=False`` and ``shutdown_save=True`` so the loop saves a checkpoint before
returning (save-then-exit). Ported verbatim from the old `training/signals.py`
(structlog → stdlib logging is the only change — no behaviour change).
"""
from __future__ import annotations

import logging
import signal
import sys
from dataclasses import dataclass
from typing import Any

_LOG = logging.getLogger(__name__)


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
    re-decided (`_fire_hard_abort`'s `hard_abort_after_stop` arm returns before the write).
    """

    running: bool = True
    stop_count: int = 0
    shutdown_save: bool = False
    abort_rule: str | None = None


def install_signal_handlers(state: ShutdownState) -> None:
    """Install SIGINT/SIGTERM handlers that flip ``state``.

    Two consecutive signals force-exit (``sys.exit(1)``); one signal sets
    ``running=False`` and ``shutdown_save=True``. The training loop is responsible for
    polling ``state`` between iterations.
    """

    def _stop(sig: int, frame: Any) -> None:
        state.stop_count += 1
        if state.stop_count >= 2:
            sys.exit(1)
        _LOG.info(
            "shutdown_requested: finishing current step… press Ctrl+C again to force",
        )
        state.shutdown_save = True
        state.running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
