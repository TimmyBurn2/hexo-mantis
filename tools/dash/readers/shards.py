"""The game-shard index: a closed shard is read once into columns with byte offsets; the open one is re-tailed."""
from __future__ import annotations

import json
import re
from array import array
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mantis.monitor.game_record import CHANNELS

SHARD_RE = re.compile(r"^games_(?P<run>.+)_seg(?P<seg>\d+)_(?P<hour>\d{10})\.jsonl$")
INDEX_NAME = "games_{run}_index.jsonl"
RESULTS = ("p1", "p2", "draw", "unknown")
SORTS = ("newest", "longest", "shortest")
WINNERS = ("p1", "p2", "cap")
_PLIES_MAX = 65535
#: Bytes read per call; a shard is a few MB, the bound matters only for a shard that grew past it.
CHUNK_BYTES = 8 * 1024 * 1024


def _int(value: Any, default: int = -1) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


class ShardRows:
    """One shard's games as columns; a list column holds `None` where the record lacks the field."""

    __slots__ = ("ids", "offsets", "channel", "result", "plies", "termination", "step", "kind", "has_stats", "rung",
                 "phase", "sims", "worker")

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
        self.sims: array[int] = array("q")
        self.worker: array[int] = array("q")

    def __len__(self) -> int:
        return len(self.ids)

    def append(self, offset: int, game: dict[str, Any]) -> None:
        self.ids.append(str(game.get("game_id")))
        self.offsets.append(offset)
        ch, res = game.get("channel"), game.get("result")
        self.channel.append(CHANNELS.index(ch) if ch in CHANNELS else -1)
        self.result.append(RESULTS.index(res) if res in RESULTS else -1)
        plies = _int(game.get("plies"))
        self.plies.append(plies if 0 <= plies <= _PLIES_MAX else min(len(game.get("moves") or []), _PLIES_MAX))
        self.termination.append(str(game.get("termination")))
        self.step.append(_int(game.get("step"), -2))
        kind, rung, phase = game.get("step_kind"), game.get("rung"), game.get("phase")
        self.kind.append(str(kind) if kind is not None else None)
        self.has_stats.append(1 if game.get("search_stats") else 0)
        self.rung.append(str(rung) if rung is not None else None)
        self.phase.append(str(phase) if phase is not None else None)
        self.sims.append(_int(game.get("served_sims")))
        self.worker.append(_int(game.get("worker_id")))

    def light(self, i: int, ordinal: int) -> dict[str, Any]:
        """The list's light row for game `i`; absent facts stay absent (a `step` of −1 is the record's own value)."""
        row: dict[str, Any] = {
            "id": self.ids[i], "ch": CHANNELS[self.channel[i]] if self.channel[i] >= 0 else None,
            "res": RESULTS[self.result[i]] if self.result[i] >= 0 else None, "pl": int(self.plies[i]),
            "term": self.termination[i], "step": int(self.step[i]) if self.step[i] != -2 else None,
            "kind": self.kind[i], "shard": ordinal, "stats": bool(self.has_stats[i]), "rung": self.rung[i],
            "phase": self.phase[i], "sims": int(self.sims[i]) if self.sims[i] >= 0 else None,
            "w": int(self.worker[i]) if self.worker[i] >= 0 else None,
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
    next_cursor: str | None
    total: int


@dataclass(frozen=True)
class Filter:
    """The list's filters: a channel, games with search stats only, a winner (`p1`, `p2`, or `cap` for the ply cap)."""

    channel: str | None = None
    with_search: bool = False
    winner: str | None = None

    def admits(self, rows: ShardRows, i: int) -> bool:
        if self.channel is not None and (rows.channel[i] < 0 or CHANNELS[rows.channel[i]] != self.channel):
            return False
        if self.with_search and not rows.has_stats[i]:
            return False
        if self.winner == "cap":
            return rows.termination[i] == "ply_cap"
        return self.winner is None or (rows.result[i] >= 0 and RESULTS[rows.result[i]] == self.winner)


class ShardIndex:
    """Every shard of `run_id` under `games_dir`; `poll()` picks up new shards and rows and hands each new game to the observers."""

    def __init__(self, games_dir: Path | str, run_id: str,
                 observers: tuple[Callable[[dict[str, Any]], None], ...] = ()) -> None:
        self.games_dir = Path(games_dir)
        self.run_id = run_id
        self.observers = observers
        self.shards: list[Shard] = []
        self._by_path: dict[Path, Shard] = {}
        self._where: dict[str, tuple[int, int]] = {}

    @property
    def total(self) -> int:
        return len(self._where)

    def channels(self) -> list[str]:
        """The channels present in the record, in contract order."""
        seen = {c for s in self.shards for c in set(s.rows.channel) if c >= 0}
        return [CHANNELS[c] for c in sorted(seen)]

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
                if isinstance(row, dict) and row.get("record") == "shard_closed" and isinstance(row.get("bytes"), int):
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
        """Discover shards, read every new row, and mark a shard closed once it is whole on disk. Raises: OSError."""
        self._discover()
        sizes = self._closed_sizes()
        for shard in self.shards:
            if shard.closed:
                continue
            self._read_new(shard)
            if sizes.get(shard.path.name) == shard.offset and not shard.held:
                shard.closed = True

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
        probe = ShardRows()
        try:
            probe.append(offset, row)
        except (TypeError, ValueError):
            shard.skipped += 1
            return
        self._where[str(row.get("game_id"))] = (shard.ordinal, len(shard.rows))
        shard.rows.append(offset, row)
        for observe in self.observers:
            try:
                observe(row)
            except (TypeError, ValueError):
                shard.skipped += 1

    def locate(self, game_id: str) -> tuple[int, int] | None:
        """`(shard ordinal, row)` of a game, or None."""
        return self._where.get(game_id)

    def fetch(self, game_id: str) -> dict[str, Any] | None:
        """The whole record of one game by one seek, or `None` when the run has no such game. Raises: OSError."""
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

    def _matches(self, where: Filter) -> Iterator[tuple[int, int]]:
        for ordinal in range(len(self.shards) - 1, -1, -1):
            rows = self.shards[ordinal].rows
            for i in range(len(rows) - 1, -1, -1):
                if where.admits(rows, i):
                    yield ordinal, i

    def page(self, after: str | None, n: int, *, sort: str = "newest", where: Filter | None = None) -> Page:
        """At most `n` light rows past the cursor in the chosen order, and the filter's total; an unknown cursor restarts."""
        matches = list(self._matches(where or Filter()))
        if sort != "newest":
            sign = -1 if sort == "longest" else 1
            matches.sort(key=lambda m: sign * self.shards[m[0]].rows.plies[m[1]])
        start = 0
        key = _cursor(after)
        if key is not None:
            start = next((k + 1 for k, m in enumerate(matches) if m == key), 0)
        window = matches[start:start + n]
        rows = [self.shards[o].rows.light(i, o) for o, i in window]
        more = start + len(window) < len(matches)
        return Page(rows=rows, next_cursor=f"{window[-1][0]}.{window[-1][1]}" if more and window else None,
                    total=len(matches))


def _cursor(text: str | None) -> tuple[int, int] | None:
    """`(shard ordinal, row)` from `"o.i"`: a game's place, stable while the open shard grows."""
    parts = (text or "").split(".")
    if len(parts) != 2 or not all(p.isascii() and p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])
