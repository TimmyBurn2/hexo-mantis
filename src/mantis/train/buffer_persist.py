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

#: The canonical replay-buffer filename, in ONE place.
#:
#: It was written out as the literal `"checkpoints/replay_buffer.bin"` at three sites, and
#: the two that were code-side DEFAULTS (`try_save_buffer` below and
#: `coordinator/step.py`'s watchdog wiring) were also CWD-RELATIVE — so a run launched from
#: anywhere but the repo root snapshotted outside its own `--out-dir`, into whatever
#: `./checkpoints/` happened to be. R1 forbids the code-side default; this constant plus
#: `canonical_buffer_path` is what replaces it, so the filename has one authority and the
#: DIRECTORY always comes from the caller that actually knows the run's layout.
CANONICAL_BUFFER_FILENAME = "replay_buffer.bin"


def canonical_buffer_path(checkpoint_dir: str | Path) -> Path:
    """The run's canonical replay-buffer path, derived from ITS checkpoint directory.

    Derived at point of use from a directory the caller owns (R98), never defaulted. The
    watchdog snapshot is a separate path on top of this one — see `watchdog_snapshot_path`,
    which appends `.watchdog` so an abnormal-exit save can never truncate the resume buffer.
    """
    return Path(checkpoint_dir) / CANONICAL_BUFFER_FILENAME


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
    # NO code-side default (R1). Persistence is OFF unless `buffer_persist` is truthy, and a
    # caller that switches it on without saying WHERE has not configured persistence — it has
    # asked this helper to invent a path, which is how the old default came to write a
    # CWD-relative `checkpoints/replay_buffer.bin` outside the run's own out-dir. Loud beats
    # a file in the wrong place: a snapshot nobody can find is indistinguishable from no
    # snapshot at the moment you need it.
    if "buffer_persist_path" not in mixing_cfg:
        raise KeyError(
            "buffer_persist is enabled but 'buffer_persist_path' is absent from mixing_cfg. "
            "There is no default (R1): derive it from the run's checkpoint directory with "
            "`mantis.train.buffer_persist.canonical_buffer_path(checkpoint_dir)`."
        )
    bp = Path(mixing_cfg["buffer_persist_path"])
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
