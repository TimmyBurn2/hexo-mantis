"""The monitor's decisions: which halting rows fired, the gap rule, the rates between two saves, the one signal to a live run."""
from __future__ import annotations

import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

_RATE_KEYS = ("games_total", "positions_produced_total", "step")
_RATE_NAMES = {"games_total": "games_per_h", "positions_produced_total": "positions_per_h", "step": "steps_per_h"}


def verdict(exams: dict[str, Any], bands: dict[str, Any], *, armed: list[str], floors_live: bool,
            bands_live: bool, two_read_bands: frozenset[str], armed_bands: list[str]) -> dict[str, Any]:
    """Floors and two-read bands decide by `_two_read`; every other band fires on one miss once the bands are live. Raises: KeyError."""
    unread = {exam for exam, row in exams.items() if row["holds"] is None}
    misses = {exam: f"{exam} calibrated {row['calibrated_mean']:.4f} below the floor {row['floor']}"
              for exam, row in exams.items() if exam not in unread and not row["holds"]}
    fired, reported, armed_now = _two_read(misses, armed, unread, floors_live)
    band_misses = {band_key(miss): f"ring band {miss}" for miss in bands["misses"]}
    once = [text for key, text in band_misses.items() if key not in two_read_bands]
    # An audit that read nothing keeps every band's arm, as an unread floor keeps its own.
    band_unread = set(armed_bands) if "not_measured" in bands else set()
    band_fired, band_reported, band_armed = _two_read(
        {key: text for key, text in band_misses.items() if key in two_read_bands}, armed_bands, band_unread, bands_live)
    if bands_live:
        fired, reported = fired + once + band_fired, reported + band_reported
    else:
        reported = reported + once + band_reported
    # Before the floors are live every miss only reports; they go live at the first save that reads and passes them all.
    return {"fired": fired, "reported": reported, "armed": armed_now,
            "floors_live": floors_live or (bool(exams) and not unread and not misses), "armed_bands": band_armed}


def _two_read(misses: dict[str, str], armed: list[str], unread: set[str],
              live: bool) -> tuple[list[str], list[str], list[str]]:
    """(fired, reported, armed): an armed name's miss fires, any other miss reports and arms, a pass disarms, an unread name keeps its arm; nothing arms before live."""
    second = {name for name in misses if live and name in armed}
    fired = [f"{misses[name]}, its second miss in a row" for name in sorted(second)]
    reported = [text for name, text in misses.items() if name not in second]
    return fired, reported, sorted(set(misses) | (set(armed) & unread)) if live else []


def band_key(miss: str) -> str:
    """The band a miss names: the audit writes every miss as `<key>: <reading>`."""
    return miss.split(":", 1)[0]


def band_trends(rows: dict[str, Any], previous: dict[str, Any] | None, previous_step: int | None,
                keys: frozenset[str]) -> dict[str, Any]:
    """Each named band's change since the last read save, `NOT MEASURED` where either reading is missing or not finite."""
    out: dict[str, Any] = {}
    for key in sorted(keys):
        now, before = _finite(rows.get(key)), _finite((previous or {}).get(key))
        if now is None or before is None:
            out[key] = {"value": now, "previous_step": previous_step,
                        "note": "NOT MEASURED: this save or the last read save has no reading"}
            continue
        out[key] = {"value": now, "previous": before, "previous_step": previous_step, "per_save": now - before}
    return out


def _finite(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) else None


def gap_rule(gap: float | None, over: list[int], step: int, line: float) -> dict[str, Any]:
    """The memorisation gap's rule: the saves in a row whose gap is above the line, firing at two; an unread gap leaves it."""
    if gap is not None:
        over = [*over, step] if gap > line else []
    return {"gap": gap, "line": line, "over": over, "fired": len(over) >= 2}


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


def is_run(pid: int, run_id: str) -> bool:
    """Whether `pid` is a live process, not this one, whose command line names the run id (a reused pid is another process)."""
    if pid == os.getpid():
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return run_id in cmdline


def run_pid(run_dir: Path, run_id: str) -> int | None:
    """The run's live process from its heartbeat, `None` when there is none or the pid is no longer the run (exited, or reused)."""
    try:
        beat = json.loads((run_dir / "logs" / f"heartbeat_{run_id}.json").read_text(encoding="utf-8"))
        pid = int(beat["pid"])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return pid if is_run(pid, run_id) else None


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


__all__ = ["band_key", "band_trends", "counter_row", "gap_rule", "halt_run", "is_run", "rates", "run_pid", "verdict"]
