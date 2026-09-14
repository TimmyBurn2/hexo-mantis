"""The reducers: every event row becomes arrays, columns or a kept row — never a parsed dict held for later."""
from __future__ import annotations

import math
from array import array
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

#: Round-level and lifecycle events, kept as the dicts they came as (a run emits ~35 of each).
KEEP_WHOLE: frozenset[str] = frozenset({
    "run_segment_started", "run_boot_identity", "resolved_config", "heartbeat_watchdog_armed",
    "heartbeat_watchdog_fired", "heartbeat_watchdog_fire_complete",
    "heartbeat_watchdog_staleness_disarmed", "heartbeat_source_unwired", "selfplay_stall_watchdog",
    "selfplay_stall_watchdog_armed", "selfplay_stall_watchdog_save_failed", "eval_round_started",
    "eval_round_complete", "eval_round_abandoned", "eval_channel_health", "eval_strength_floor",
    "eval_round_device_memory", "eval_rung_activated", "eval_rung_graduated",
    "eval_ladder_zero_game_round", "eval_broken", "eval_round_skipped_busy", "eval_result_unroutable",
    "eval_rung_skipped", "actor_lag_exceeded", "actor_lag_negative", "disk_guard_error", "disk_alert",
    "training_alert", "training_step", "monitor_gates", "hard_abort", "hard_abort_after_stop",
    "clean_stop_save", "shutdown_save", "resume_state_persisted", "periodic_checkpoint_save",
    "positions_counter_reset",
})

#: `(event, x key, y key)` → the field the y is read from; the derived ys are computed by name.
_DIRECT: dict[tuple[str, str, str], str] = {
    **{("trainer_step", "step", k): k for k in ("value_loss", "policy_loss", "loss", "grad_norm", "lr")},
    **{("iteration_complete", "step", k): k
       for k in ("games_per_hour", "positions_per_hour", "steps_per_hour", "sims_per_sec")},
    ("disk_free", "ts", "disk_free_gb"): "disk_free_gb",
}
_DERIVED: tuple[tuple[str, str, str], ...] = (
    ("trainer_step", "step", "edges_share"), ("trainer_step", "step", "nodes_share"),
    ("iteration_complete", "step", "buffer_fill"), ("iteration_complete", "step", "alpha_per_1000"),
)
#: The per-game payloads never stored (the dashboard's `HEAVY_FIELDS`); the drop is counted.
HEAVY_GAME_FIELDS: tuple[str, ...] = ("moves_list", "moves_detail", "value_trace")
_WINNER_CODE: dict[Any, int] = {0: 0, 1: 1, -1: -1}


def _finite(value: Any) -> float | None:
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


class Games:
    """Per-game facts in stream order, one column each; `winner` −2 is an undecodable winner."""

    __slots__ = ("plies", "winner", "cap", "hashes")

    def __init__(self) -> None:
        self.plies: array[float] = array("d")
        self.winner: array[int] = array("b")
        self.cap: array[int] = array("b")
        self.hashes: list[str] = []

    @property
    def count(self) -> int:
        return len(self.plies)

    def cap_windowed(self, window: int) -> list[tuple[float, float]]:
        """`(game ordinal, cap share of the `window` games ending there)` for every full window."""
        flags = self.cap
        if window <= 0 or len(flags) < window:
            return []
        out: list[tuple[float, float]] = []
        running = sum(flags[:window])
        out.append((float(window), running / window))
        for i in range(window, len(flags)):
            running += flags[i] - flags[i - window]
            out.append((float(i + 1), running / window))
        return out


@dataclass(frozen=True)
class Snapshot:
    """The frozen read API over a record's reduced state; built only by `Reducers.snapshot`."""

    _rows: Mapping[str, list[dict[str, Any]]]
    _last: Mapping[str, dict[str, Any]]
    _series: Mapping[tuple[str, str, str], Series]
    games: Games
    counts: Mapping[str, int]
    dropped_fields: Mapping[str, int]
    first_ts: float | None
    last_ts: float | None
    steps_max: int | None

    def rows(self, name: str) -> list[dict[str, Any]]:
        return self._rows.get(name, [])

    def last(self, name: str) -> dict[str, Any] | None:
        return self._last.get(name)

    def series(self, name: str, x_key: str, y_key: str) -> Series:
        """A reduced series; a `(name, x_key, y_key)` triple the reducers do not keep is a KeyError, never an empty series."""
        return self._series[(name, x_key, y_key)]

    def wall_hours(self) -> float | None:
        if self.first_ts is None or self.last_ts is None:
            return None
        return (self.last_ts - self.first_ts) / 3600.0


class Reducers:
    """Feed rows in stream order; `snapshot()` hands out a frozen view sharing the arrays."""

    def __init__(self) -> None:
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._last: dict[str, dict[str, Any]] = {}
        self._series: dict[tuple[str, str, str], Series] = {
            key: Series() for key in (*_DIRECT, *_DERIVED)}
        self._games = Games()
        self._counts: Counter[str] = Counter()
        self._dropped: Counter[str] = Counter()
        self._first_ts: float | None = None
        self._last_ts: float | None = None
        self._steps_max: int | None = None

    def feed(self, row: dict[str, Any]) -> None:
        name = str(row.get("event", "?"))
        self._counts[name] += 1
        ts = _finite(row.get("ts"))
        if ts is not None:
            self._first_ts = ts if self._first_ts is None else min(self._first_ts, ts)
            self._last_ts = ts if self._last_ts is None else max(self._last_ts, ts)
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
            x, y = _finite(row.get(x_key)), _finite(row.get(source))
            if x is not None and y is not None:
                self._series[(ev, x_key, y_key)].append(x, y)
        x = _finite(row.get("step"))
        if x is None:
            return
        if name == "trainer_step":
            mb = _finite(row.get("microbatches")) or 1.0
            e, ce = _finite(row.get("edges")), _finite(row.get("caps_max_edges"))
            n, cn = _finite(row.get("nodes")), _finite(row.get("caps_max_nodes"))
            if e is not None and ce is not None and n is not None and cn is not None and ce > 0 and cn > 0:
                self._series[(name, "step", "edges_share")].append(x, (e / mb) / ce)
                self._series[(name, "step", "nodes_share")].append(x, (n / mb) / cn)
        elif name == "iteration_complete":
            size, capacity = _finite(row.get("buffer_size")), _finite(row.get("buffer_capacity"))
            if size is not None and capacity is not None and capacity > 0:
                self._series[(name, "step", "buffer_fill")].append(x, size / capacity)
            block = row.get("gumbel_alpha_full")
            per = _finite(block.get("per_1000")) if isinstance(block, dict) else None
            if per is not None:
                self._series[(name, "step", "alpha_per_1000")].append(x, per)

    def _feed_game(self, row: dict[str, Any]) -> None:
        for key in HEAVY_GAME_FIELDS:
            if key in row:
                self._dropped["game_complete"] += 1
        plies = _finite(row.get("moves"))
        self._games.plies.append(plies if plies is not None else 0.0)
        self._games.winner.append(_WINNER_CODE.get(row.get("winner"), -2))
        self._games.cap.append(1 if row.get("terminal_reason") == "ply_cap" else 0)
        digest = row.get("game_id_byte_hash")
        if digest:
            self._games.hashes.append(str(digest))

    def snapshot(self) -> Snapshot:
        return Snapshot(
            _rows=MappingProxyType({k: list(v) for k, v in self._rows.items()}),
            _last=MappingProxyType(dict(self._last)),
            _series=MappingProxyType(dict(self._series)),
            games=self._games, counts=MappingProxyType(dict(self._counts)),
            dropped_fields=MappingProxyType(dict(self._dropped)),
            first_ts=self._first_ts, last_ts=self._last_ts, steps_max=self._steps_max,
        )
