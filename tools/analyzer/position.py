"""A move list ↔ the engine's Board, built on the analyst thread; the compact text form; refusals by name."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from mantis._engine import Board

#: The engine's side ints ↔ the game record's strings (contract #11): the ONE mapping table.
SIDE_OF_INT: dict[int, str] = {1: "p1", -1: "p2"}
_SQRT3 = math.sqrt(3.0)


class PositionRefused(ValueError):
    """A move list the engine cannot hold; the message names the ply and the rule."""


@dataclass(frozen=True)
class Position:
    """A replayed move list: the Board, the moves, and the winner (`"p1"` / `"p2"`) once a six stands."""

    board: Board
    moves: list[tuple[int, int]]
    winner: str | None


def parse_moves(text: str | list[Any]) -> list[tuple[int, int]]:
    """`q,r;q,r;…` (spaces tolerated, empty = no moves), a JSON list `[[q, r], …]`, or that list; raises PositionRefused."""
    if isinstance(text, list):
        return _as_moves(text)
    if not isinstance(text, str):
        raise PositionRefused(f"moves must be text or a list of [q, r], got {type(text).__name__}")
    s = text.strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            return _as_moves(json.loads(s))
        except json.JSONDecodeError as exc:
            raise PositionRefused(f"move list JSON is malformed: {exc.msg}") from None
    out: list[tuple[int, int]] = []
    for i, item in enumerate(s.split(";")):
        parts = item.split(",")
        try:
            if len(parts) != 2:
                raise ValueError
            out.append((int(parts[0]), int(parts[1])))
        except ValueError:
            raise PositionRefused(f"move {i} is not `q,r`: {item.strip()!r}") from None
    return out


def _as_moves(data: Any) -> list[tuple[int, int]]:
    if not isinstance(data, list):
        raise PositionRefused("moves must be a list of [q, r]")
    out: list[tuple[int, int]] = []
    for i, m in enumerate(data):
        ok = isinstance(m, (list, tuple)) and len(m) == 2 and all(
            isinstance(x, int) and not isinstance(x, bool) for x in m)
        if not ok:
            raise PositionRefused(f"move {i} is not [q, r]: {m!r}")
        out.append((int(m[0]), int(m[1])))
    return out


def format_moves(moves: list[tuple[int, int]]) -> str:
    """The compact text form, `q,r;q,r;…`."""
    return ";".join(f"{q},{r}" for q, r in moves)


def build_board(moves: list[tuple[int, int]], encoding: str) -> Position:
    """Replay `moves` on a fresh Board; raises PositionRefused on an occupied/out-of-radius cell or a move after a six."""
    board = Board.with_encoding_name(encoding)
    winner: str | None = None
    for k, (q, r) in enumerate(moves):
        if winner is not None:
            raise PositionRefused(f"move after the game ended at ply {k}: {winner} already won")
        if not board.is_legal(q, r):
            raise PositionRefused(f"illegal move ({q}, {r}) at ply {k}: occupied or outside the legal radius")
        board.apply_move(q, r)
        if board.check_win():
            winner = SIDE_OF_INT[int(board.winner() or 0)]
    return Position(board=board, moves=list(moves), winner=winner)


def position_record(pos: Position) -> dict[str, Any]:
    """The `position` block of the record: the facts the page states, all from the Board itself."""
    board = pos.board
    legal = board.legal_moves()
    return {
        "moves": [[q, r] for q, r in pos.moves],
        "ply": int(board.ply),
        "to_move": SIDE_OF_INT[int(board.current_player)],
        "moves_remaining": int(board.moves_remaining),
        "legal": len(legal),
        "winner": pos.winner,
        "win_line": [[q, r] for q, r in board.find_winning_line()] if pos.winner else None,
        "legal_window": [[q, r] for q, r in legal],
    }


# The three below are the Python twins of `web/board.js`, tested here so the JS transliteration has an oracle.
def owner_of_ply(ply: int) -> str:
    """The side that placed stone `ply`: ply 0 is p1's single, then pairs alternate (`Ply::turn`)."""
    return "p1" if ((ply + 1) // 2) % 2 == 0 else "p2"


def hex_to_pixel(q: int, r: int) -> tuple[float, float]:
    """Axial → pointy-top pixel at unit size (Red Blob): the viewer's forward map."""
    return _SQRT3 * (q + r / 2.0), 1.5 * r


def pixel_to_hex(x: float, y: float) -> tuple[int, int]:
    """Pixel → the nearest axial cell by `cube_round` (Red Blob)."""
    qf, rf = _SQRT3 / 3.0 * x - y / 3.0, 2.0 / 3.0 * y
    sf = -qf - rf
    q, r, s = round(qf), round(rf), round(sf)
    dq, dr, ds = abs(q - qf), abs(r - rf), abs(s - sf)
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    return int(q), int(r)


__all__ = ["Position", "PositionRefused", "SIDE_OF_INT", "build_board", "format_moves", "hex_to_pixel",
           "owner_of_ply", "parse_moves", "pixel_to_hex", "position_record"]
