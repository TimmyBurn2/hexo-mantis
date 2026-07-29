"""Replay-buffer persistence helper (WP10 §a.4 PORT; old training/buffer_persist.py).

Behaviour-exact relocation; the only change is structlog → stdlib logging (the new
repo has no structlog dependency). Optional buffer save is a best-effort effect —
its failure is logged, not run-fatal (the run-fatal persist path is the canonical
checkpoint write in `train/checkpoints.py`, LAW-14).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

# Counted best-effort (WPCLEAN Phase RES, paying R-BUFFER-PERSIST-COUNTER): both swallow
# arms below used to WARN uncounted, which left the LAST save of a run (the
# "shutdown_signal" trigger) invisible to any observer. The count is deliberately NOT
# `checkpoints.persist_errors_total`: that counter is the watchdog's `> 0` FATAL feed, and
# this path is best-effort BY DESIGN (module docstring). Live consumer: the coordinator
# publishes it in the `monitor_gates` LAW-18 payload (`buffer_save_errors_total`).
buffer_save_errors_total = 0


def try_save_buffer(
    buffer: Any,
    mixing_cfg: dict[str, Any],
    trigger: str,
    recent_buffer: Any | None = None,
) -> None:
    """Save the replay buffer (and optionally the recent_buffer) when buffer_persist
    is enabled. A save failure is logged AND COUNTED (best-effort, never uncounted —
    R-BUFFER-PERSIST-COUNTER) — the canonical resume buffer is guarded by the atomic
    checkpoint path, not this helper."""
    global buffer_save_errors_total
    if not mixing_cfg.get("buffer_persist", False):
        return
    bp = Path(mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
    try:
        buffer.save_to_path(str(bp))
        _LOG.info("buffer_saved path=%s positions=%s trigger=%s", bp, buffer.size, trigger)
    except Exception as exc:  # noqa: BLE001 — best-effort snapshot; counted + logged, not fatal
        buffer_save_errors_total += 1
        _LOG.warning("buffer_save_failed path=%s error=%s", bp, exc)
    if recent_buffer is not None and recent_buffer.size > 0:
        rbp = Path(str(bp) + ".recent")
        try:
            n = recent_buffer.save_to_path(str(rbp))
            _LOG.info("recent_buffer_saved path=%s positions=%s trigger=%s", rbp, n, trigger)
        except Exception as exc:  # noqa: BLE001
            buffer_save_errors_total += 1
            _LOG.warning("recent_buffer_save_failed path=%s error=%s", rbp, exc)
