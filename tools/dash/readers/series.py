"""The reducers: every event row becomes arrays, columns or a kept row — never a parsed dict held for later."""
from __future__ import annotations

import math
from array import array
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

#: Round-level and lifecycle events, kept as the dicts they came as (a run emits tens of each).
KEEP_WHOLE: frozenset[str] = frozenset({
    "run_segment_started", "run_boot_identity", "resolved_config", "heartbeat_watchdog_armed",
    "heartbeat_watchdog_fired", "heartbeat_watchdog_fire_complete", "heartbeat_watchdog_staleness_disarmed",
    "selfplay_stall_watchdog", "selfplay_stall_watchdog_armed", "selfplay_stall_watchdog_save_failed",
    "eval_round_started", "eval_round_complete", "eval_round_abandoned", "eval_strength_floor", "eval_broken",
    "actor_lag_exceeded", "actor_lag_negative", "disk_guard_error", "disk_alert", "training_alert", "monitor_gates",
    "hard_abort", "hard_abort_after_stop", "clean_stop_save", "shutdown_save", "resume_state_persisted",
    "periodic_checkpoint_save", "heldout_gap",
})

#: `(event, x key, y key)` → the field the y is read from.
_DIRECT: dict[tuple[str, str, str], str] = {
    **{("trainer_step", "step", k): k
       for k in ("value_loss", "policy_loss", "loss", "grad_norm", "lr", "policy_entropy")},
    **{("iteration_complete", "step", k): k
       for k in ("games_per_hour", "positions_per_hour", "steps_per_hour", "sims_per_sec", "avg_game_length")},
    ("disk_free", "ts", "disk_free_gb"): "disk_free_gb",
}
#: The per-game payloads never stored; the drop is counted.
HEAVY_GAME_FIELDS: tuple[str, ...] = ("moves_list", "moves_detail", "value_trace")
_WINNER_CODE: dict[Any, int] = {0: 0, 1: 1, -1: -1}


def finite(value: Any) -> float | None:
    """`value` as a float when it is a finite real number (never a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


class Series:
    """One `(x, y)` series as two double arrays; only finite pairs are ever appended."""

    __slots__ = ("x", "y")

    def __init__(self) -> None:
        self.x: array[float] = array("d")
        self.y: array[float] = array("d")

    def append(self, x: float, y: float) -> None:
        self.x.append(x)
        self.y.append(y)

    def __len__(self) -> int:
        return len(self.x)

    def pairs(self) -> list[tuple[float, float]]:
        return list(zip(self.x, self.y, strict=True))

    def last(self) -> float | None:
        return self.y[-1] if len(self.y) else None

    def view(self) -> SeriesView:
        """This series bound at its current length: later appends never show through a view."""
        return SeriesView(self.x, self.y, len(self.x))


@dataclass(frozen=True)
class SeriesView:
    """A series as of one snapshot; the arrays only ever grow, so the first `n` pairs never change."""

    x: array[float]
    y: array[float]
    n: int

    def __len__(self) -> int:
        return self.n

    def pairs(self) -> list[tuple[float, float]]:
        return list(zip(self.x[:self.n], self.y[:self.n], strict=True))

    def last(self) -> float | None:
        return self.y[self.n - 1] if self.n else None


class Games:
    """Per-game facts in stream order, one column each; `winner` −2 is an undecodable winner, `step` −1 before any step."""

    __slots__ = ("plies", "winner", "cap", "step", "hashes")

    def __init__(self) -> None:
        self.plies: array[float] = array("d")
        self.winner: array[int] = array("b")
        self.cap: array[int] = array("b")
        self.step: array[int] = array("q")
        self.hashes: list[str] = []

    @property
    def count(self) -> int:
        return len(self.plies)

    def copy(self) -> Games:
        """A copy of every column, so a snapshot never sees a game fed after it."""
        out = Games()
        out.plies, out.winner, out.cap = array("d", self.plies), array("b", self.winner), array("b", self.cap)
        out.step, out.hashes = array("q", self.step), list(self.hashes)
        return out


@dataclass(frozen=True)
class Segment:
    """One life of the run: its number, its start row (None when the segment wrote none), its rows and time span."""

    number: int
    started: Mapping[str, Any] | None
    rows: int
    first_ts: float | None
    last_ts: float | None


@dataclass(frozen=True)
class Snapshot:
    """The frozen read API over a record's reduced state; built only by `Reducers.snapshot`."""

    _rows: Mapping[str, list[dict[str, Any]]]
    _last: Mapping[str, dict[str, Any]]
    _series: Mapping[tuple[str, str, str], SeriesView]
    games: Games
    counts: Mapping[str, int]
    dropped_fields: Mapping[str, int]
    segments: tuple[Segment, ...]
    first_ts: float | None
    last_ts: float | None
    steps_max: int | None

    def rows(self, name: str) -> list[dict[str, Any]]:
        return self._rows.get(name, [])

    def last(self, name: str) -> dict[str, Any] | None:
        return self._last.get(name)

    def series(self, name: str, x_key: str, y_key: str) -> SeriesView:
        """A reduced series; a triple the reducers do not keep is a KeyError, never an empty series. Raises: KeyError."""
        return self._series[(name, x_key, y_key)]

    @property
    def live_segment(self) -> Segment | None:
        """The newest segment: a resumed run's current life, whatever its earlier segments did."""
        return self.segments[-1] if self.segments else None


class Reducers:
    """Feed rows in stream order, segment by segment; `snapshot()` hands out a frozen view no later feed can change."""

    def __init__(self) -> None:
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._last: dict[str, dict[str, Any]] = {}
        self._series: dict[tuple[str, str, str], Series] = {key: Series() for key in _DIRECT}
        self._games = Games()
        self._counts: Counter[str] = Counter()
        self._dropped: Counter[str] = Counter()
        self._segments: dict[int, list[Any]] = {}
        self._segment: int | None = None
        self._first_ts: float | None = None
        self._last_ts: float | None = None
        self._steps_max: int | None = None

    def enter_segment(self, number: int) -> None:
        """Attribute the rows fed from now on to segment `number`."""
        self._segment = number
        self._segments.setdefault(number, [None, 0, None, None])

    def feed(self, row: dict[str, Any]) -> None:
        name = str(row.get("event", "?"))
        self._counts[name] += 1
        ts = finite(row.get("ts"))
        if ts is not None:
            self._first_ts = ts if self._first_ts is None else min(self._first_ts, ts)
            self._last_ts = ts if self._last_ts is None else max(self._last_ts, ts)
        if self._segment is not None:
            seg = self._segments[self._segment]
            seg[1] += 1
            if name == "run_segment_started" and seg[0] is None:
                seg[0] = row
            if ts is not None:
                seg[2] = ts if seg[2] is None else min(seg[2], ts)
                seg[3] = ts if seg[3] is None else max(seg[3], ts)
        step = row.get("step")
        if isinstance(step, int) and not isinstance(step, bool):
            self._steps_max = step if self._steps_max is None else max(self._steps_max, step)
        if name == "game_complete":
            self._feed_game(row)
        elif name in ("trainer_step", "iteration_complete", "disk_free"):
            self._feed_series(name, row)
            self._last[name] = row
        elif name in KEEP_WHOLE:
            self._rows.setdefault(name, []).append(row)
            self._last[name] = row

    def _feed_series(self, name: str, row: dict[str, Any]) -> None:
        for (ev, x_key, y_key), source in _DIRECT.items():
            if ev != name:
                continue
            x, y = finite(row.get(x_key)), finite(row.get(source))
            if x is not None and y is not None:
                self._series[(ev, x_key, y_key)].append(x, y)

    def _feed_game(self, row: dict[str, Any]) -> None:
        for key in HEAVY_GAME_FIELDS:
            if key in row:
                self._dropped["game_complete"] += 1
        plies = finite(row.get("moves"))
        self._games.plies.append(plies if plies is not None else 0.0)
        self._games.winner.append(_WINNER_CODE.get(row.get("winner"), -2))
        self._games.cap.append(1 if row.get("terminal_reason") == "ply_cap" else 0)
        self._games.step.append(self._steps_max if self._steps_max is not None else -1)
        digest = row.get("game_id_byte_hash")
        if digest:
            self._games.hashes.append(str(digest))

    def snapshot(self) -> Snapshot:
        return Snapshot(
            _rows=MappingProxyType({k: list(v) for k, v in self._rows.items()}),
            _last=MappingProxyType(dict(self._last)),
            _series=MappingProxyType({k: v.view() for k, v in self._series.items()}),
            games=self._games.copy(), counts=MappingProxyType(dict(self._counts)),
            dropped_fields=MappingProxyType(dict(self._dropped)),
            segments=tuple(Segment(n, s[0], s[1], s[2], s[3]) for n, s in sorted(self._segments.items())),
            first_ts=self._first_ts, last_ts=self._last_ts, steps_max=self._steps_max,
        )
