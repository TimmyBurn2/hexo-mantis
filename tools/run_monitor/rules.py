"""The monitor's decisions: which halting rows fired, the rates between two saves, and the one signal to a live run."""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

_RATE_KEYS = ("games_total", "positions_produced_total", "step")
_RATE_NAMES = {"games_total": "games_per_h", "positions_produced_total": "positions_per_h", "step": "steps_per_h"}


def halting_rows(exams: dict[str, Any], bands: dict[str, Any]) -> list[str]:
    """Every halting row that fired: an exam's calibrated mean below its floor, a ring band outside (an unmeasured ring is not a miss)."""
    fired = [f"{exam} calibrated {row['calibrated_mean']:.4f} below the floor {row['floor']}"
             for exam, row in exams.items() if not row["holds"]]
    return fired + [f"ring band {miss}" for miss in bands["misses"]]


def counter_row(event: dict[str, Any]) -> dict[str, float] | None:
    """An `iteration_complete` row's counters, or `None` when one is missing: an absent counter is never a zero."""
    keys = ("ts", *_RATE_KEYS)
    if not all(isinstance(event.get(k), int | float) and not isinstance(event.get(k), bool) for k in keys):
        return None
    return {k: float(event[k]) for k in keys}


def rates(counters: list[dict[str, float]], since: float, until: float,
          busy: list[tuple[float, float]]) -> dict[str, Any]:
    """Games, positions and steps per hour between two saves, over the whole span and over the monitor's idle windows."""
    rows = [c for c in counters if since <= c["ts"] <= until]
    out: dict[str, Any] = {"span_h": (until - since) / 3600.0, "rows": len(rows)}
    if len(rows) < 2:
        return {**out, "note": "NOT MEASURED: fewer than two iteration rows in the span"}
    total = {k: rows[-1][k] - rows[0][k] for k in _RATE_KEYS}
    hours = (rows[-1]["ts"] - rows[0]["ts"]) / 3600.0
    idle = dict.fromkeys(_RATE_KEYS, 0.0)
    idle_h = 0.0
    for a, b in zip(rows, rows[1:], strict=False):
        if any(a["ts"] < hi and b["ts"] > lo for lo, hi in busy):
            continue
        idle_h += (b["ts"] - a["ts"]) / 3600.0
        for k in _RATE_KEYS:
            idle[k] += b[k] - a[k]
    out.update({_RATE_NAMES[k]: total[k] / hours if hours > 0 else None for k in _RATE_KEYS})
    out["idle_h"] = idle_h
    out.update({f"idle_{_RATE_NAMES[k]}": idle[k] / idle_h if idle_h > 0 else None for k in _RATE_KEYS})
    return out


def run_pid(run_dir: Path, run_id: str) -> int | None:
    """The run's live process from its heartbeat, `None` when there is none or the pid is no longer the run (exited, or reused)."""
    try:
        beat = json.loads((run_dir / "logs" / f"heartbeat_{run_id}.json").read_text(encoding="utf-8"))
        pid = int(beat["pid"])
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return pid if run_id in cmdline else None


def halt_run(run_dir: Path, run_id: str) -> dict[str, Any]:
    """ONE SIGTERM to the run's live process (it saves, then exits; a second would tear it down); a gone run is recorded, not signalled."""
    pid = run_pid(run_dir, run_id)
    if pid is None:
        return {"sent": False, "reason": "no live run process carries the run id", "ts": time.time()}
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"sent": False, "pid": pid, "reason": "the process exited before the signal", "ts": time.time()}
    return {"sent": True, "pid": pid, "signal": "SIGTERM", "ts": time.time()}


__all__ = ["counter_row", "halt_run", "halting_rows", "rates", "run_pid"]
