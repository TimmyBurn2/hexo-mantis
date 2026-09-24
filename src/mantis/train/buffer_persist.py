"""The run's canonical replay-buffer path, derived from the run's own checkpoint directory."""
from __future__ import annotations

from pathlib import Path

#: The canonical replay-buffer filename; the DIRECTORY always comes from the caller (R1: no
#: CWD-relative default).
CANONICAL_BUFFER_FILENAME = "replay_buffer.bin"


def canonical_buffer_path(checkpoint_dir: str | Path) -> Path:
    """The run's canonical replay-buffer path, derived from ITS checkpoint directory.

    Derived at point of use from a directory the caller owns (R98), never defaulted. The
    watchdog snapshot is a separate path on top of this one — see `watchdog_snapshot_path`,
    which appends `.watchdog` so an abnormal-exit save can never truncate the resume buffer.
    """
    return Path(checkpoint_dir) / CANONICAL_BUFFER_FILENAME
