"""Heartbeat registry + heartbeat-FILE codec + the exit-code authority (WP13-A §c.4).

This is the half of the livelock-proof watchdog that lives BELOW the DAG cut: the
out-of-process supervisor (`mantis.monitor.supervise`) must read the heartbeat file on a
box whose GPU is the thing that wedged, so the file format and the exit codes cannot live
in `train/lifecycle/` (`monitor → train` is an illegal edge). The watchdog THREAD — the
fire authority (snapshot closure, `os._exit`) — stays in
`mantis.train.lifecycle.heartbeat_watchdog` and imports ONLY this module.

Two levels, one contract:
  * a wedged pipeline thread → the in-process watchdog fires → exit `WATCHDOG_STALL_EXIT_CODE`;
  * a starved watchdog thread → the file `seq` freezes → the supervisor kills + relaunches.

Staleness is measured on an INJECTED monotonic clock only: a wall-clock/NTP jump can
neither hide a stall nor invent one.
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

# The pipeline stages whose liveness the watchdog keys on (name pins — the manifest rows,
# the coordinator beat, the pool/server beats and the arm-log all use these tokens).
# WP11-A adds "eval_round" as the 4th source: the eval pipeline's persistent poller thread
# beats it every tick, with or without an in-flight round.
HEARTBEAT_SOURCES: tuple[str, ...] = (
    "train_step", "inference_dispatch", "selfplay_drain", "eval_round",
)

# The restart-wrapper key. 42 MUST equal `mantis.train.lifecycle.watchdog.
# SELFPLAY_STALL_EXIT_CODE` (one authority for the transient stall/livelock class);
# 43 is the persistence fault — NOT transient, so the supervisor never relaunches on it.
WATCHDOG_STALL_EXIT_CODE: int = 42
PERSIST_FATAL_EXIT_CODE: int = 43
# 44 is taken supervisor-side (`supervise.RELAUNCH_BUDGET_EXIT_CODE`). 45 is the
# actor-lag invariant breach (WP-UNFREEZE §4: `learner_step − actor_ckpt_step > N`):
# the supervisor's existing "any other code propagated with NO relaunch" arm handles it
# with zero supervisor change — relaunching into a broken sync mechanism is a crash loop.
ACTOR_LAG_EXIT_CODE: int = 45
# 46 is the COOPERATIVE member of the family (WPMINT Phase X, CARD-ABORT-EXIT / R84), and
# the deviation is deliberate rather than an oversight. 42/43/45 are delivered by `os._exit`
# from the watchdog thread; the draw-rate collapse abort is delivered by
# `StepCoordinator._fire_hard_abort` setting `shutdown.running = False` and RETURNING, so the
# loop unwinds through `close_out`, the terminal-eval drain and the shutdown checkpoint.
# Delivering 46 by `os._exit` would discard all three and contradict LAW-16 (save-then-exit),
# making the fix strictly worse than the defect it closes. Parity with the family is therefore
# taken in the REGISTRY (this constant + the `draw_rate_collapse` manifest row's `exit_code`)
# and in the supervisor's READING of the process rc; DELIVERY stays cooperative. The rc is
# resolved at a process boundary from the rule name the coordinator records
# (`ShutdownState.abort_rule` -> `mantis.config.armed_aborts.exit_code_for_abort`), never from
# a second literal here or in the coordinator.
DRAW_RATE_COLLAPSE_EXIT_CODE: int = 46


@dataclass(frozen=True)
class HeartbeatFileState:
    """One decoded heartbeat-file snapshot.

    `seq` is the monotonic progression counter the supervisor keys liveness on; `pid`
    identifies the writing child so a legitimate restart resets the baseline instead of
    reading as forgery.
    """

    seq: int
    pid: int
    ages: dict[str, float]
    wall_ts: float


class HeartbeatRegistry:
    """Monotonic per-source last-beat store. `beat` IS the injected `HeartbeatFn`.

    Every collaborator (pool, inference server, step coordinator) receives `registry.beat`
    and calls it with its own source name — an unknown source is a WIRING BUG and raises,
    never a silently-dropped beat that would make the watchdog blind to that stage.

    A source that has NEVER been beaten is tracked distinctly (`beaten_sources()`): "nothing
    ever wired this stage up" and "this stage has wedged" are different facts and must not
    produce the same abort. Ages are still reported for a never-beaten source (measured from
    construction/arm), so the condition is observable — it just may not be read as staleness.
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
        """Sources that have been beaten at least ONCE since construction.

        `arm()` does NOT populate this: arming grants a grace window, it does not prove a
        producer exists. A source outside this set has no live producer yet, which is a
        wiring fact, not a liveness fact — the watchdog must not turn it into a stall abort.
        """
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
    """Atomically publish one heartbeat snapshot: tmp file → ``os.replace``.

    A reader therefore never sees a half-written file, and no tmp sibling survives a
    successful write. (R6: `os.replace` is atomic on POSIX local FS; on an exotic/network
    FS a torn file is possible — the reader tolerates it as no-progress, which only ever
    errs toward a supervisor relaunch, the safe side.)

    The tmp sibling name is UNIQUE per writer+call: a shared fixed `<name>.tmp` made two
    writers on one path race each other's temp file (`os.replace` → `FileNotFoundError` on
    ~30% of writes), which inside the watchdog was `best_effort`-caught and therefore
    invisible (RED-TEAM F17).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{random.getrandbits(32):08x}.tmp")
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
    """A finite, non-negative, in-range integer — or ``None`` (the tolerant reading).

    Hostile/corrupt files reach this: `Infinity`, `1e400` (→ `inf`), `NaN`, `[1]`, `"7"`,
    `True`. `int(inf)` raises `OverflowError`, which used to escape `read_heartbeat_file`
    and kill the supervisor loop that has no guard around it (RED-TEAM F4).
    """
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

    NEVER raises — this is a hard contract, not a best effort: the only caller is the
    out-of-process supervisor's poll loop, and an exception here takes level 2 of the
    livelock protection down with it, leaving the child unsupervised. `None` means "no
    progress observable", which the supervisor treats as staleness accrual (never as
    freshness) — the failure mode errs toward a relaunch, the safe side.
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
