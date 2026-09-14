"""One game as the pages read it: moves, owners, the six, the arms and sims, the roots by ply, every absence named."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hexlogic import owner, win_line

#: The self-play channel writes no per-position search stats (GAME-RECORD-1); the page states it.
SELFPLAY_STATS_GAP = ("no per-position search stats on this channel — GAME-RECORD-1 records "
                      "self-play moves only (CARD-SELFPLAY-SEARCH-STATS); a stated gap, never an empty board")
NO_ROOT_EXPOSED = "no player exposed a search root on this game"
_ARM_CHAR = {"opening": "o", "full": "f", "fast": "q"}


@dataclass(frozen=True)
class GameView:
    game_id: str
    run_id: str
    channel: str
    result: str
    termination: str
    moves: list[tuple[int, int]]
    owners: list[int]
    win: list[tuple[int, int]] | None
    arms: str | None
    sims_by_arm: dict[str, int]
    stats_by_ply: dict[int, dict[str, Any]]
    stats_absent_reason: str | None
    candidate: int | None
    served_sims: int | None
    step: int | None
    step_kind: str | None
    finding: str | None

    @classmethod
    def from_record(cls, game: dict[str, Any], run_id: str) -> GameView:
        """Build the view of one record; the record's own `result`/`termination` are checked against the six."""
        moves = [(int(q), int(r)) for q, r in (game.get("moves") or [])]
        owners = [owner(i) for i in range(len(moves))]
        line = win_line([list(m) for m in moves])
        win = [(q, r) for q, r in line] if line is not None else None
        arms, sims_by_arm = _arms(game)
        stats_field = game.get("search_stats")
        stats_by_ply: dict[int, dict[str, Any]] = {}
        if isinstance(stats_field, list):
            stats_by_ply = {int(s["ply"]): s for s in stats_field
                            if isinstance(s, dict) and isinstance(s.get("ply"), int)}
        channel = str(game.get("channel"))
        if channel == "selfplay":
            reason: str | None = SELFPLAY_STATS_GAP
        elif stats_field is None:
            reason = NO_ROOT_EXPOSED
        else:
            reason = None
        colors = game.get("colors")
        cand = colors.get("candidate") if isinstance(colors, dict) else None
        result, termination = str(game.get("result")), str(game.get("termination"))
        step, sims = game.get("step"), game.get("served_sims")
        return cls(
            game_id=str(game.get("game_id")), run_id=run_id, channel=channel, result=result,
            termination=termination, moves=moves, owners=owners, win=win, arms=arms,
            sims_by_arm=sims_by_arm, stats_by_ply=stats_by_ply, stats_absent_reason=reason,
            candidate=cand if cand in (1, -1) else None,
            served_sims=sims if isinstance(sims, int) and not isinstance(sims, bool) else None,
            step=step if isinstance(step, int) and not isinstance(step, bool) else None,
            step_kind=str(game["step_kind"]) if game.get("step_kind") is not None else None,
            finding=_finding(game.get("game_id"), run_id, moves, win, result, termination),
        )


def _arms(game: dict[str, Any]) -> tuple[str | None, dict[str, int]]:
    arms, sims = game.get("move_arms"), game.get("move_sims")
    if not isinstance(arms, list) or not isinstance(sims, list) or len(arms) != len(sims):
        return None, {}
    chars: list[str] = []
    per_arm: dict[str, int] = {}
    for arm, n in zip(arms, sims, strict=True):
        char = _ARM_CHAR.get(str(arm))
        if char is None:
            return None, {}
        chars.append(char)
        if char != "o":
            per_arm[char] = int(n)
    return "".join(chars), per_arm


def _finding(game_id: Any, run_id: str, moves: list[tuple[int, int]], win: list[tuple[int, int]] | None,
             result: str, termination: str) -> str | None:
    """The viewer's build-time check: the six through the last stone must belong to the recorded winner."""
    if win is not None and result in ("p1", "p2"):
        side = "p1" if owner(len(moves) - 1) == 0 else "p2"
        if side != result:
            return (f"{run_id}/{game_id}: the six-in-a-row through the last stone belongs to {side} "
                    f"but the record says {result}")
    elif win is None and termination in ("six_in_a_row", "win"):
        return f"{run_id}/{game_id}: terminated {termination} but no line of six passes through the last stone"
    return None
