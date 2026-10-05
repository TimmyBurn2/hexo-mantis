"""Liveness from the heartbeat file: its age alone says live, stale or stopped; its format is uncontracted and feeds nothing else."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: A heartbeat younger than this is live, younger than `STALE_SEC` stale, older stopped.
LIVE_SEC = 600.0
STALE_SEC = 3600.0


@dataclass(frozen=True)
class Liveness:
    """The newest evidence of the run's heartbeat (None when there is no heartbeat file) and its source."""

    beat_ts: float | None
    source: str

    def state(self, now: float) -> str:
        """`live`, `stale`, `stopped`, or `unknown` without a heartbeat file."""
        if self.beat_ts is None:
            return "unknown"
        age = now - self.beat_ts
        return "live" if age < LIVE_SEC else "stale" if age < STALE_SEC else "stopped"


def read(logs_dir: Path, run_id: str) -> Liveness:
    """The heartbeat's newest time: the file's mtime or its `wall_ts`, whichever is later."""
    path = logs_dir / f"heartbeat_{run_id}.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return Liveness(None, f"no {path.name}")
    wall = None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        value = raw.get("wall_ts") if isinstance(raw, dict) else None
        wall = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    except (OSError, ValueError):
        wall = None
    if wall is not None and wall > mtime:
        return Liveness(wall, f"{path.name} wall_ts")
    return Liveness(mtime, f"{path.name} mtime")
