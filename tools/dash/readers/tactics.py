"""Per-position tactics along a game, from the engine's tactics reader: what the side to move wins now, what it must block."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mantis._engine import Board
from mantis.diagnostics.tactics import analyze

from .hexlogic import owner

Cell = tuple[int, int]
#: The engine's side ints for the two seats (contract #11: p1 moves first).
SIDE: tuple[int, int] = (1, -1)
#: The legal fence the tactics reader counts within; the windows it reports do not depend on it.
RADIUS = 8


def stones_left(ply: int) -> int:
    """Stones the side to move still places this turn in the position after `ply` stones."""
    return 1 if ply == 0 or ply % 2 == 0 else 2


@dataclass(frozen=True)
class Reading:
    """One position for its mover: the class (`win`, `block`, `lost1`, `quiet`), its cells, the opponent's fours, W1 and the opponent's fives."""

    mover: int
    k: int
    cls: str
    cells: tuple[Cell, ...]
    fours: int
    w1: tuple[Cell, ...]
    opp_fives: tuple[Cell, ...]


def read_position(moves: list[tuple[int, int]], ply: int) -> Reading:
    """The tactics of the position after `moves[:ply]` for its side to move."""
    placed = moves[:ply]
    q = np.array([m[0] for m in placed], dtype=np.int64)
    r = np.array([m[1] for m in placed], dtype=np.int64)
    p = np.array([SIDE[owner(i)] for i in range(len(placed))], dtype=np.int64)
    mover, k = owner(ply), stones_left(ply)
    row = analyze(q, r, p, SIDE[mover], k, radius=RADIUS)
    cls, cells = row.forced(k)
    return Reading(mover=mover, k=k, cls=cls, cells=tuple(sorted(cells)), fours=len(row.fours),
                   w1=tuple(sorted(row.w1)), opp_fives=tuple(sorted(row.opp_fives)))


def read_game(moves: list[tuple[int, int]], last: int | None = None) -> list[Reading]:
    """One reading per position from the empty board to the position before the last stone (or `last` positions)."""
    count = len(moves) if last is None else min(last, len(moves))
    return [read_position(moves, ply) for ply in range(count)]


def oracle_disagreements(moves: list[tuple[int, int]]) -> list[str]:
    """Positions where W1 or the opponent's fives differ from the engine's own `winning_moves`, as sentences."""
    board = Board()
    out: list[str] = []
    for ply in range(len(moves)):
        reading = read_position(moves, ply)
        mine, theirs = SIDE[reading.mover], SIDE[1 - reading.mover]
        engine_w1 = tuple(sorted((int(a), int(b)) for a, b in board.winning_moves(mine)))
        engine_opp = tuple(sorted((int(a), int(b)) for a, b in board.winning_moves(theirs)))
        if engine_w1 != reading.w1:
            out.append(f"ply {ply}: W1 {list(reading.w1)} against the engine's {list(engine_w1)}")
        if engine_opp != reading.opp_fives:
            out.append(f"ply {ply}: the opponent's fives {list(reading.opp_fives)} against the engine's {list(engine_opp)}")
        board.apply_move(*moves[ply])
    return out
