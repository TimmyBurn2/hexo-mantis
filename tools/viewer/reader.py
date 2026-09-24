"""Build one run's viewer data from its GAME-RECORD-1 shards: a light index and per-shard games."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mantis.monitor.game_record import read_shard, run_shard_paths

from .hexlogic import owner, win_line


class EmptyGameRecord(RuntimeError):
    """The games directory held no game of this run — refuse rather than render an empty viewer."""


_ARM_CHAR = {"opening": "o", "full": "f", "fast": "q"}


@dataclass
class ShardData:
    """One shard's games keyed by id: `m` moves, `win` the six line or None, `s` stats, `a` arm chars (`o`/`f`/`q`) + `sims` per arm."""

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
        "sims": game.get("served_sims") if game.get("colors") else None,
        "cand": (game.get("colors") or {}).get("candidate"),
    }
    # Absent keys stay absent: a null `rung` on every self-play row is bytes, not information.
    return {k: v for k, v in row.items() if v is not None}


def _arms(game: dict[str, Any]) -> tuple[str, dict[str, int]] | None:
    """`(arm string, sims per arm)` from `move_arms` / `move_sims`; None when the record has none."""
    arms, sims = game.get("move_arms"), game.get("move_sims")
    if not isinstance(arms, list) or not isinstance(sims, list) or len(arms) != len(sims):
        return None
    per_arm: dict[str, int] = {}
    chars = []
    for arm, n in zip(arms, sims, strict=True):
        char = _ARM_CHAR.get(str(arm))
        if char is None:
            return None
        chars.append(char)
        if char != "o":
            per_arm[char] = int(n)
    return "".join(chars), per_arm


def build_run(games_dir: Path | str, run_id: str) -> RunData:
    """Read every shard of `run_id` under `games_dir` into a `RunData`.

    Raises:
        EmptyGameRecord: no shard of this run held a game."""
    index: list[dict[str, Any]] = []
    shards: list[ShardData] = []
    findings: list[str] = []
    skipped_total = 0
    for shard_no, path in enumerate(run_shard_paths(games_dir, run_id)):
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
            arms = _arms(game)
            if arms is not None:
                entry["a"], entry["sims"] = arms
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
