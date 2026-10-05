"""One game as the pages read it: moves, the six checked against the record, arms and sims as recorded, stats by ply, tactics, chances."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import chances, tactics
from .hexlogic import owner, turn_of, win_line

_ARMS = ("opening", "full", "fast")


@dataclass(frozen=True)
class GameView:
    """One record read: the six derived through the last stone and checked, every absence kept as None."""

    game_id: str
    channel: str
    result: str
    termination: str
    moves: list[tuple[int, int]]
    win: list[tuple[int, int]] | None
    arms: list[str] | None
    sims: list[int] | None
    stats: dict[int, dict[str, Any]]
    stats_field: str
    step: int | None
    step_kind: str | None
    served_sims: int | None
    worker: int | None
    rung: str | None
    phase: str | None
    candidate: int | None
    finding: str | None

    @classmethod
    def from_record(cls, game: dict[str, Any]) -> GameView:
        moves = [(int(q), int(r)) for q, r in (game.get("moves") or [])]
        line = win_line([list(m) for m in moves])
        win = [(q, r) for q, r in line] if line is not None else None
        arms, sims = game.get("move_arms"), game.get("move_sims")
        if not (isinstance(arms, list) and isinstance(sims, list) and len(arms) == len(sims) == len(moves)
                and all(a in _ARMS for a in arms)):
            arms, sims = None, None
        field = game.get("search_stats")
        stats = {int(s["ply"]): s for s in field if isinstance(s, dict) and isinstance(s.get("ply"), int)} \
            if isinstance(field, list) else {}
        state = "absent" if field is None and "search_stats" not in game else "none" if field is None else \
            "empty" if not field else "recorded"
        colors = game.get("colors")
        cand = colors.get("candidate") if isinstance(colors, dict) else None
        result, termination = str(game.get("result")), str(game.get("termination"))
        return cls(
            game_id=str(game.get("game_id")), channel=str(game.get("channel")), result=result,
            termination=termination, moves=moves, win=win, arms=list(arms) if arms else None,
            sims=[int(n) for n in sims] if sims else None, stats=stats, stats_field=state,
            step=_int(game.get("step")), step_kind=_str(game.get("step_kind")), served_sims=_int(game.get("served_sims")),
            worker=_int(game.get("worker_id")), rung=_str(game.get("rung")), phase=_str(game.get("phase")),
            candidate=cand if cand in (1, -1) else None, finding=_finding(moves, win, result, termination))

    @property
    def turns(self) -> int:
        return turn_of(len(self.moves) - 1) if self.moves else 0

    def payload(self) -> dict[str, Any]:
        """The JSON the board steps through: moves, the six, arms, stats as recorded, tactics per position, the strip."""
        readings = tactics.read_game(self.moves)
        pts = chances.points(self.stats)
        turning = chances.turning_point(pts)
        return {
            "id": self.game_id, "channel": self.channel, "result": self.result, "termination": self.termination,
            "moves": [list(m) for m in self.moves], "win": [list(c) for c in self.win] if self.win else None,
            "arms": self.arms, "sims": self.sims, "turns": self.turns, "finding": self.finding,
            "stats": {str(p): _stat(s) for p, s in sorted(self.stats.items())},
            "tactics": [{"cls": t.cls, "cells": [list(c) for c in t.cells], "fours": t.fours} for t in readings],
            "chances": [{"ply": p.ply, "turn": p.turn, "light": round(p.light, 4),
                         "cost": None if p.cost is None else round(p.cost, 4)} for p in pts],
            "turning": None if turning is None else {"turn": turning.turn, "ply": turning.ply,
                                                     "cost": round(turning.cost or 0.0, 4), "mover": owner(turning.ply)},
        }


def _stat(entry: dict[str, Any]) -> dict[str, Any]:
    """One searched ply as recorded: the root value, the visited cells by visits, their total; `by` when the record names it."""
    visits = [[int(v[0]), int(v[1]), int(v[2])] for v in entry.get("visits") or []
              if isinstance(v, list) and len(v) == 3]
    out: dict[str, Any] = {"v": entry.get("root_value"), "top": sorted(visits, key=lambda v: -v[2]),
                           "n": sum(v[2] for v in visits)}
    if entry.get("by") is not None:
        out["by"] = entry["by"]
    return out


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _str(v: Any) -> str | None:
    return str(v) if v is not None else None


def _finding(moves: list[tuple[int, int]], win: list[tuple[int, int]] | None, result: str, termination: str) -> str | None:
    """The six through the last stone must belong to the recorded winner; a disagreement is a finding, never hidden."""
    if win is not None and result in ("p1", "p2"):
        side = "p1" if owner(len(moves) - 1) == 0 else "p2"
        if side != result:
            return f"the six through the last stone belongs to {side} but the record says {result}"
    elif win is None and termination in ("six_in_a_row", "win"):
        return f"terminated {termination} but no line of six passes through the last stone"
    return None
