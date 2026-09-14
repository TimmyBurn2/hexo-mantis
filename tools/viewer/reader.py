"""Build one run's viewer data from its GAME-RECORD-1 shards: a light index and per-shard games."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mantis.monitor.game_record import read_shard

from .hexlogic import owner, win_line

_SHARD_RE = re.compile(r"^games_(?P<run>.+)_seg(?P<seg>\d+)_(?P<hour>\d{10})\.jsonl$")


class EmptyGameRecord(RuntimeError):
    """The games directory held no game of this run — refuse rather than render an empty viewer."""


@dataclass
class ShardData:
    """One shard's games keyed by id: `m` moves, `win` the six line or None, `s` stats if present."""

    name: str
    games: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: int = 0


@dataclass
class RunData:
    """One run: its light index (one row per game, in shard order) and its shards' game data."""

    run_id: str
    index: list[dict[str, Any]]
    shards: list[ShardData]
    findings: list[str]
    skipped_lines: int


def _row(game: dict[str, Any], run_id: str, shard: int) -> dict[str, Any]:
    plies = game.get("plies")
    row = {
        "id": str(game.get("game_id")), "run": run_id, "ch": game.get("channel"),
        "res": game.get("result"), "pl": int(plies) if isinstance(plies, int) else len(game.get("moves") or []),
        "term": game.get("termination"), "step": game.get("step"), "kind": game.get("step_kind"),
        "shard": shard, "stats": bool(game.get("search_stats")),
        "rung": game.get("rung"), "phase": game.get("phase"), "w": game.get("worker_id"),
    }
    # Absent keys stay absent: a null `rung` on every self-play row is bytes, not information.
    return {k: v for k, v in row.items() if v is not None}


def _shard_paths(games_dir: Path, run_id: str) -> list[Path]:
    found: list[tuple[int, str, Path]] = []
    if games_dir.is_dir():
        for entry in games_dir.iterdir():
            match = _SHARD_RE.match(entry.name)
            if match is not None and match.group("run") == run_id:
                found.append((int(match.group("seg")), match.group("hour"), entry))
    return [path for _seg, _hour, path in sorted(found)]


def build_run(games_dir: Path | str, run_id: str) -> RunData:
    """Read every shard of `run_id` under `games_dir` into a `RunData`.

    Raises:
        EmptyGameRecord: no shard of this run held a game."""
    index: list[dict[str, Any]] = []
    shards: list[ShardData] = []
    findings: list[str] = []
    skipped_total = 0
    for shard_no, path in enumerate(_shard_paths(Path(games_dir), run_id)):
        records, skipped = read_shard(path)
        skipped_total += skipped
        shard = ShardData(name=path.name, skipped=skipped)
        for game in records:
            moves = [[int(q), int(r)] for q, r in (game.get("moves") or [])]
            line = win_line(moves)
            entry: dict[str, Any] = {"m": moves, "win": line}
            stats = game.get("search_stats")
            if stats:
                entry["s"] = stats
            row = _row(game, run_id, shard_no)
            if line is not None and row["res"] in ("p1", "p2"):
                side = "p1" if owner(len(moves) - 1) == 0 else "p2"
                if side != row["res"]:
                    findings.append(
                        f"{run_id}/{row['id']}: the six-in-a-row through the last stone belongs "
                        f"to {side} but the record says {row['res']}")
            elif line is None and row["term"] in ("six_in_a_row", "win"):
                findings.append(f"{run_id}/{row['id']}: terminated {row['term']} but no line of "
                                f"six passes through the last stone")
            shard.games[row["id"]] = entry
            index.append(row)
        shards.append(shard)
    if not index:
        raise EmptyGameRecord(
            f"{games_dir} holds no game of run {run_id!r} ({len(shards)} shard(s) read, "
            f"{skipped_total} unparseable line(s)). Refusing to render an empty viewer.")
    return RunData(run_id=run_id, index=index, shards=shards, findings=findings,
                   skipped_lines=skipped_total)
