"""The game-shard index: a closed shard is read once into columns with byte offsets; the open one is re-tailed."""
from __future__ import annotations

import json
import re
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mantis.monitor.game_record import CHANNELS

SHARD_RE = re.compile(r"^games_(?P<run>.+)_seg(?P<seg>\d+)_(?P<hour>\d{10})\.jsonl$")
INDEX_NAME = "games_{run}_index.jsonl"
_RESULTS = ("p1", "p2", "draw", "unknown")
_PLIES_MAX = 65535
#: Bytes read per call; a shard is ≈ 1 MB, the bound matters only for a shard that grew past it.
CHUNK_BYTES = 8 * 1024 * 1024


class EmptyGameRecord(RuntimeError):
    """No shard of this run holds a game — refuse rather than serve an empty list as a run."""


class ShardRows:
    """One shard's games as columns; a list column holds `None` where the record lacks the field."""

    __slots__ = ("ids", "offsets", "channel", "result", "plies", "termination", "step", "kind",
                 "has_stats", "rung", "phase", "candidate", "sims")

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.offsets: array[int] = array("q")
        self.channel: array[int] = array("b")
        self.result: array[int] = array("b")
        self.plies: array[int] = array("H")
        self.termination: list[str] = []
        self.step: array[int] = array("q")
        self.kind: list[str | None] = []
        self.has_stats: array[int] = array("b")
        self.rung: list[str | None] = []
        self.phase: list[str | None] = []
        self.candidate: array[int] = array("b")
        self.sims: array[int] = array("q")

    def __len__(self) -> int:
        return len(self.ids)

    def append(self, offset: int, game: dict[str, Any]) -> None:
        self.ids.append(str(game.get("game_id")))
        self.offsets.append(offset)
        ch = game.get("channel")
        self.channel.append(CHANNELS.index(ch) if ch in CHANNELS else -1)
        res = game.get("result")
        self.result.append(_RESULTS.index(res) if res in _RESULTS else -1)
        plies = game.get("plies")
        if isinstance(plies, int) and not isinstance(plies, bool) and 0 <= plies <= _PLIES_MAX:
            self.plies.append(plies)
        else:
            self.plies.append(min(len(game.get("moves") or []), _PLIES_MAX))
        self.termination.append(str(game.get("termination")))
        step = game.get("step")
        self.step.append(step if isinstance(step, int) and not isinstance(step, bool) else -1)
        kind = game.get("step_kind")
        self.kind.append(str(kind) if kind is not None else None)
        self.has_stats.append(1 if game.get("search_stats") else 0)
        rung, phase = game.get("rung"), game.get("phase")
        self.rung.append(str(rung) if rung is not None else None)
        self.phase.append(str(phase) if phase is not None else None)
        colors = game.get("colors")
        cand = colors.get("candidate") if isinstance(colors, dict) else None
        self.candidate.append(cand if cand in (1, -1) else 0)
        sims = game.get("served_sims")
        self.sims.append(sims if isinstance(sims, int) and not isinstance(sims, bool) and colors else -1)

    def light(self, i: int, run_id: str, shard: int) -> dict[str, Any]:
        """The viewer's light row for game `i`; absent facts stay absent."""
        row: dict[str, Any] = {
            "id": self.ids[i], "run": run_id,
            "ch": CHANNELS[self.channel[i]] if self.channel[i] >= 0 else None,
            "res": _RESULTS[self.result[i]] if self.result[i] >= 0 else None,
            "pl": int(self.plies[i]), "term": self.termination[i],
            "step": int(self.step[i]) if self.step[i] >= 0 else None, "kind": self.kind[i],
            "shard": shard, "stats": bool(self.has_stats[i]), "rung": self.rung[i],
            "phase": self.phase[i], "cand": int(self.candidate[i]) or None,
            "sims": int(self.sims[i]) if self.sims[i] >= 0 else None,
        }
        return {k: v for k, v in row.items() if v is not None}


@dataclass
class Shard:
    path: Path
    ordinal: int
    closed: bool = False
    rows: ShardRows = field(default_factory=ShardRows)
    offset: int = 0
    held: bytes = b""
    skipped: int = 0


@dataclass(frozen=True)
class Page:
    """One window of light rows; `next_cursor` is `None` at the end; `total` counts the filter's matches."""

    rows: list[dict[str, Any]]
    next_cursor: tuple[int, int] | None
    total: int


@dataclass(frozen=True)
class Filter:
    channel: str | None = None
    result: str | None = None
    termination: str | None = None
    min_plies: int | None = None
    max_plies: int | None = None

    def admits(self, rows: ShardRows, i: int) -> bool:
        if self.channel is not None and (rows.channel[i] < 0 or CHANNELS[rows.channel[i]] != self.channel):
            return False
        if self.result is not None and (rows.result[i] < 0 or _RESULTS[rows.result[i]] != self.result):
            return False
        if self.termination is not None and rows.termination[i] != self.termination:
            return False
        if self.min_plies is not None and rows.plies[i] < self.min_plies:
            return False
        return not (self.max_plies is not None and rows.plies[i] > self.max_plies)


class ShardIndex:
    """Every shard of `run_id` under `games_dir`; call `poll()` to pick up new shards and new rows."""

    def __init__(self, games_dir: Path | str, run_id: str) -> None:
        self.games_dir = Path(games_dir)
        self.run_id = run_id
        self.shards: list[Shard] = []
        self._by_path: dict[Path, Shard] = {}
        self._where: dict[str, tuple[int, int]] = {}

    @property
    def total(self) -> int:
        return len(self._where)

    def _closed_sizes(self) -> dict[str, int]:
        index = self.games_dir / INDEX_NAME.format(run=self.run_id)
        sizes: dict[str, int] = {}
        if not index.is_file():
            return sizes
        with index.open("rb") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(row, dict) and row.get("record") == "shard_closed" \
                        and isinstance(row.get("bytes"), int):
                    sizes[str(row.get("shard"))] = int(row["bytes"])
        return sizes

    def _discover(self) -> None:
        found: list[tuple[int, str, Path]] = []
        if self.games_dir.is_dir():
            for entry in self.games_dir.iterdir():
                match = SHARD_RE.match(entry.name)
                if match is not None and match.group("run") == self.run_id:
                    found.append((int(match.group("seg")), match.group("hour"), entry))
        for _seg, _hour, path in sorted(found):
            if path not in self._by_path:
                shard = Shard(path=path, ordinal=len(self.shards))
                self.shards.append(shard)
                self._by_path[path] = shard

    def poll(self) -> None:
        """Discover shards, read every new row, and mark a shard closed once complete on disk; OSError propagates.

        Raises:
            EmptyGameRecord: no shard of the run holds a game after reading everything present.
        """
        self._discover()
        sizes = self._closed_sizes()
        for shard in self.shards:
            if shard.closed:
                continue
            self._read_new(shard)
            if sizes.get(shard.path.name) == shard.offset and not shard.held:
                shard.closed = True
        if not self._where:
            raise EmptyGameRecord(
                f"{self.games_dir} holds no game of run {self.run_id!r} ({len(self.shards)} shard(s))")

    def _read_new(self, shard: Shard) -> None:
        size = shard.path.stat().st_size
        if size <= shard.offset:
            return
        line_start = shard.offset - len(shard.held)
        held, offset = shard.held, shard.offset
        with shard.path.open("rb") as handle:
            handle.seek(offset)
            while offset < size:
                chunk = handle.read(min(CHUNK_BYTES, size - offset))
                if not chunk:
                    break
                offset += len(chunk)
                lines = (held + chunk).split(b"\n")
                held = lines.pop()
                for raw in lines:
                    self._feed(shard, line_start, raw)
                    line_start += len(raw) + 1
        shard.held, shard.offset = held, offset

    def _feed(self, shard: Shard, offset: int, raw: bytes) -> None:
        text = raw.strip()
        if not text:
            return
        try:
            row = json.loads(text)
        except (ValueError, UnicodeDecodeError):
            shard.skipped += 1
            return
        if not isinstance(row, dict) or row.get("record") in ("shard_opened", "shard_closed"):
            return
        self._where[str(row.get("game_id"))] = (shard.ordinal, len(shard.rows))
        shard.rows.append(offset, row)

    def fetch(self, game_id: str) -> dict[str, Any] | None:
        """The whole record of one game by one seek, or `None` when the run has no such game; OSError propagates."""
        where = self._where.get(game_id)
        if where is None:
            return None
        shard = self.shards[where[0]]
        with shard.path.open("rb") as handle:
            handle.seek(shard.rows.offsets[where[1]])
            raw = handle.readline()
        try:
            row = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return row if isinstance(row, dict) else None

    def _matches(self, newest_first: bool, where: Filter) -> Iterator[tuple[int, int]]:
        ordinals = range(len(self.shards) - 1, -1, -1) if newest_first else range(len(self.shards))
        for ordinal in ordinals:
            rows = self.shards[ordinal].rows
            indices = range(len(rows) - 1, -1, -1) if newest_first else range(len(rows))
            for i in indices:
                if where.admits(rows, i):
                    yield ordinal, i

    def page(self, after: tuple[int, int] | None, n: int, *, newest_first: bool = True,
             channel: str | None = None, result: str | None = None, termination: str | None = None,
             min_plies: int | None = None, max_plies: int | None = None) -> Page:
        """At most `n` light rows past `after` (exclusive) in the chosen order, and the filter's total."""
        where = Filter(channel, result, termination, min_plies, max_plies)
        matches = list(self._matches(newest_first, where))
        start = 0
        if after is not None:
            try:
                start = matches.index(after) + 1
            except ValueError:
                start = len(matches)
        window = matches[start:start + n]
        rows = [self.shards[o].rows.light(i, self.run_id, o) for o, i in window]
        more = start + len(window) < len(matches)
        return Page(rows=rows, next_cursor=window[-1] if window and more else None, total=len(matches))
