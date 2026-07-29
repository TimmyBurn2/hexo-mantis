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


def try_save_buffer(
    buffer: Any,
    mixing_cfg: dict[str, Any],
    trigger: str,
    recent_buffer: Any | None = None,
) -> None:
    """Save the replay buffer (and optionally the recent_buffer) when buffer_persist
    is enabled. A save failure is logged (best-effort) — the canonical resume buffer
    is guarded by the atomic checkpoint path, not this helper."""
    if not mixing_cfg.get("buffer_persist", False):
        return
    bp = Path(mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
    try:
        buffer.save_to_path(str(bp))
        _LOG.info("buffer_saved path=%s positions=%s trigger=%s", bp, buffer.size, trigger)
    except Exception as exc:  # noqa: BLE001 — best-effort snapshot; logged, not fatal
        _LOG.warning("buffer_save_failed path=%s error=%s", bp, exc)
    if recent_buffer is not None and recent_buffer.size > 0:
        rbp = Path(str(bp) + ".recent")
        try:
            n = recent_buffer.save_to_path(str(rbp))
            _LOG.info("recent_buffer_saved path=%s positions=%s trigger=%s", rbp, n, trigger)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("recent_buffer_save_failed path=%s error=%s", rbp, exc)
