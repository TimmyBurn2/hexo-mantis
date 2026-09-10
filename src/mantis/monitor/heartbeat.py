"""Heartbeat registry + heartbeat-FILE codec + the exit-code authority.

The half of the livelock-proof watchdog that lives BELOW the DAG cut: the out-of-process
supervisor must read the heartbeat file on a box whose GPU is the thing that wedged, and
`monitor -> train` is an illegal edge. The watchdog THREAD stays in
`mantis.train.lifecycle.heartbeat_watchdog` and imports ONLY this module. A wedged pipeline
thread fires that watchdog and exits `WATCHDOG_STALL_EXIT_CODE`; a starved watchdog thread
freezes the file `seq` and the supervisor kills and relaunches. Staleness is measured on an
INJECTED monotonic clock, so a wall-clock jump can neither hide a stall nor invent one.

>300 justify (R8): the supervisor<->child CONTRACT is ONE unit — file codec, source-name pins,
exit codes and the parent-death env key are read from BOTH sides of an illegal-edge boundary.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# The pipeline stages whose liveness the watchdog keys on (name pins, shared by the manifest
# rows, the coordinator beat, the pool/server beats and the arm-log).
HEARTBEAT_SOURCES: tuple[str, ...] = (
    "train_step", "inference_dispatch", "selfplay_drain", "eval_round",
)

# 42 MUST equal `train.lifecycle.watchdog.SELFPLAY_STALL_EXIT_CODE` — one authority for the
# transient stall/livelock class. 43 is the persistence fault: NOT transient, never relaunched.
WATCHDOG_STALL_EXIT_CODE: int = 42
PERSIST_FATAL_EXIT_CODE: int = 43
# 42 MUST equal `train.lifecycle.watchdog.SELFPLAY_STALL_EXIT_CODE`. 43 is the persistence
# fault: NOT transient, never relaunched.
ACTOR_LAG_EXIT_CODE: int = 45
# 44 is taken supervisor-side. 45 is the actor-lag invariant breach, propagated with NO relaunch,
# because relaunching into a broken sync mechanism is a crash loop.
DRAW_RATE_COLLAPSE_EXIT_CODE: int = 46
# 46 is the first COOPERATIVE member: the draw-rate abort sets `shutdown.running = False` and
# RETURNS, so the loop unwinds through `close_out`, the terminal-eval drain and the shutdown
# checkpoint, all of which an `os._exit(46)` would discard (LAW-16). Parity is taken in this
# registry and in the supervisor's READING of the rc, never in the delivery mechanism.
DISK_SPACE_EXHAUSTED_EXIT_CODE: int = 47
# 46 is the first COOPERATIVE member: the draw-rate abort sets `shutdown.running = False` and
# RETURNS, so the loop unwinds through `close_out`, the terminal-eval drain and the shutdown
# checkpoint, all of which an `os._exit(46)` would discard (LAW-16).
TERMINAL_EVAL_BROKEN_EXIT_CODE: int = 48

# 47 is the SECOND cooperative member: the disk guard's critical arm is a SIGTERM to its own pid,
# and an `os._exit(47)` would discard the save the guard exists to protect. Before it was
# registered a run the disk guard killed reported rc 0 and the supervisor relaunched into it.
PARENT_DEATH_PPID_ENV: str = "MANTIS_PARENT_DEATH_PPID"

# 48 is the THIRD and cleanest: the terminal eval is the LAST action of `close_out`, so delivery
# is `main` returning the number. It registers that rc 0 does not certify eval health.
PARENT_DEATH_ARM_EXEC_MODULE: str = "mantis.train.lifecycle.arm_exec"

# ONE number for SEVEN reason classes, deliberately: one number per OUTCOME with the CAUSE in the
# payload, pairwise-distinguishable through `mantis.eval.errors.EvalBrokenReason`. For all three
# the rc is resolved at the process boundary from `ShutdownState.abort_rule`, never a literal.
PARENT_VANISHED_EXIT_CODE: int = 71


@dataclass(frozen=True)
class HeartbeatFileState:
    """One decoded heartbeat-file snapshot. `seq` is the monotonic progression counter the
    supervisor keys liveness on; `pid` identifies the writing child, so a legitimate restart
    resets the baseline instead of reading as forgery."""

    seq: int
    pid: int
    ages: dict[str, float]
    wall_ts: float


class HeartbeatRegistry:
    """Monotonic per-source last-beat store. `beat` IS the injected `HeartbeatFn`.

    An unknown source RAISES rather than silently dropping a beat that would leave the watchdog
    blind to that stage. A source that has NEVER been beaten is tracked distinctly, because
    "nothing wired this stage up" and "this stage has wedged" must not produce the same abort.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sources: Sequence[str] = HEARTBEAT_SOURCES,
    ) -> None:
        self._clock = clock
        self._sources = tuple(sources)
        if not self._sources:
            raise ValueError("HeartbeatRegistry needs at least one source")
        self._lock = threading.Lock()
        now = float(clock())
        self._last: dict[str, float] = {source: now for source in self._sources}
        self._beaten: set[str] = set()

    @property
    def sources(self) -> tuple[str, ...]:
        return self._sources

    def beat(self, source: str) -> None:
        """Record a beat for ``source`` (THE `HeartbeatFn`)."""
        if source not in self._last:
            raise ValueError(
                f"unknown heartbeat source {source!r}; known sources: {self._sources}"
            )
        now = float(self._clock())
        with self._lock:
            self._last[source] = now
            self._beaten.add(source)

    def beaten_sources(self) -> frozenset[str]:
        """Return the sources beaten at least ONCE since construction. `arm()` does NOT populate
        this: arming grants a grace window, it does not prove a producer exists, and a source
        outside this set is a wiring fact rather than a liveness fact."""
        with self._lock:
            return frozenset(self._beaten)

    def arm(self) -> None:
        """Grace-reset every source to age 0 (called when the watchdog arms)."""
        now = float(self._clock())
        with self._lock:
            for source in self._sources:
                self._last[source] = now

    def ages(self) -> dict[str, float]:
        """Per-source seconds since the last beat, on the INJECTED clock."""
        now = float(self._clock())
        with self._lock:
            return {source: now - last for source, last in self._last.items()}


def write_heartbeat_file(
    path: Path | str,
    *,
    seq: int,
    pid: int,
    ages: Mapping[str, float],
    wall_ts: float,
) -> None:
    """Atomically publish one heartbeat snapshot: tmp file -> ``os.replace``.

    A torn file on an exotic FS reads as no-progress, which errs toward a relaunch. The tmp
    sibling name is UNIQUE per writer+call, because a fixed `<name>.tmp` made two writers race
    invisibly under a best-effort catch; its bits come from `os.urandom` and NOT stdlib
    `random`, which would silently advance the PROCESS-GLOBAL stream other code seeds.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(
        f"{target.name}.{os.getpid()}.{int.from_bytes(os.urandom(4), 'big'):08x}.tmp"
    )
    text = json.dumps(
        {
            "seq": int(seq),
            "pid": int(pid),
            "ages": {str(k): float(v) for k, v in dict(ages).items()},
            "wall_ts": float(wall_ts),
            "sources": list(HEARTBEAT_SOURCES),
        },
        ensure_ascii=False,
    )
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


#: Any `seq`/`pid` outside this range is corruption, not a reading (a 64-bit counter cannot
#: reach it, and `int(float('inf'))` used to raise straight out of the "never raises" reader).
_MAX_COUNTER = 2 ** 63 - 1


def _safe_counter(value: Any) -> int | None:
    """Return a finite, non-negative, in-range integer, or ``None`` for the tolerant reading.
    Hostile files reach this: `int(inf)` raises `OverflowError`, which used to escape
    `read_heartbeat_file` and kill the supervisor loop that has no guard around it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        result = int(value)
    except (OverflowError, ValueError):          # belt and braces on exotic numeric types
        return None
    if result < 0 or result > _MAX_COUNTER:
        return None
    return result


def read_heartbeat_file(path: Path | str) -> HeartbeatFileState | None:
    """Decode a heartbeat file; ``None`` when absent, unreadable, torn OR HOSTILE.

    NEVER raises — a hard contract, since the only caller is the supervisor's poll loop and an
    exception here takes level 2 of the livelock protection down with it. `None` accrues
    staleness, which errs toward a relaunch.
    """
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    try:
        data: Any = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    payload = cast("dict[str, Any]", data)
    seq = _safe_counter(payload.get("seq"))
    pid = _safe_counter(payload.get("pid"))
    if seq is None or pid is None:
        return None
    ages: dict[str, float] = {}
    try:
        raw_ages = payload.get("ages")
        if isinstance(raw_ages, dict):
            items = cast("dict[str, Any]", raw_ages)
            ages = {str(key): float(value) for key, value in items.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value)}
        wall = payload.get("wall_ts", 0.0)
        wall_ts = (float(wall) if isinstance(wall, (int, float))
                   and not isinstance(wall, bool) and math.isfinite(wall) else 0.0)
        return HeartbeatFileState(seq=seq, pid=pid, ages=ages, wall_ts=wall_ts)
    except (TypeError, ValueError, OverflowError):
        return None
