"""The htttx game notation: `version[1];` then `N. [q,r][q,r];` a turn, after the origin stone the server places itself."""
from __future__ import annotations

import re

from mantis._engine import Board

Cell = tuple[int, int]
#: The server places the first player's single stone here before anyone moves; the notation never writes it.
ORIGIN: Cell = (0, 0)
_TURN = re.compile(r"^\s*(\d+)\s*\.\s*((?:\[\s*-?\d+\s*,\s*-?\d+\s*\]\s*){1,2});?\s*$")
_CELL = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_VERSION = re.compile(r"^\s*version\s*\[\s*1\s*\]\s*;?\s*$")


class NotationRefused(ValueError):
    """A text that is not an htttx game this engine replays; the message names the line and the rule."""


def parse(text: str) -> list[Cell]:
    """The game's stones in placement order, the origin first. Raises: NotationRefused (a line, a number or a stone that does not replay)."""
    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if lines and _VERSION.match(lines[0]):
        lines = lines[1:]
    elif lines and lines[0].strip().lower().startswith("version"):
        raise NotationRefused(f"line 1: only version[1] is read, got {lines[0].strip()!r}")
    if not lines:
        raise NotationRefused("no turn to read: the notation is `N. [q,r][q,r];` per line")
    moves: list[Cell] = [ORIGIN]
    board = Board()
    board.apply_move(*ORIGIN)
    for i, line in enumerate(lines, 1):
        m = _TURN.match(line)
        if m is None:
            raise NotationRefused(f"turn line {i} is not `N. [q,r][q,r];`: {line.strip()!r}")
        if int(m.group(1)) != i:
            raise NotationRefused(f"turn line {i} is numbered {m.group(1)}")
        cells = [(int(a), int(b)) for a, b in _CELL.findall(m.group(2))]
        if len(cells) == 1 and i != len(lines):
            raise NotationRefused(f"turn {i} places one stone, which only the last turn may do")
        for q, r in cells:
            if board.winner() is not None:
                raise NotationRefused(f"turn {i}: ({q}, {r}) comes after the game was already won")
            if board.get(q, r) != 0:
                raise NotationRefused(f"turn {i}: ({q}, {r}) is already occupied")
            board.apply_move(q, r)
            moves.append((q, r))
    return moves


def write(moves: list[Cell]) -> str:
    """A move list as htttx: the origin dropped, one line per turn, a last turn of one stone when the list ends mid-turn. Raises: NotationRefused (a list not starting at the origin)."""
    if not moves or tuple(moves[0]) != ORIGIN:
        raise NotationRefused("htttx starts from the origin stone at (0, 0); this position does not")
    rest = [tuple(m) for m in moves[1:]]
    turns = ["".join(f"[{q},{r}]" for q, r in rest[i:i + 2]) for i in range(0, len(rest), 2)]
    return "version[1];\n" + "".join(f"{n}. {t};\n" for n, t in enumerate(turns, 1))
