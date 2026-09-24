"""The /proc and supervisor-stream twins the process suites share; spawn/reap stay local."""
from __future__ import annotations

import json
from pathlib import Path


def alive(pid: int | None) -> bool:
    """True iff `pid` names a live, non-zombie process."""
    if pid is None:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def supervisor_events(err: Path) -> list[dict]:
    """The supervisor's own stream: one JSON line per action."""
    if not err.exists():
        return []
    rows = []
    for line in err.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows
