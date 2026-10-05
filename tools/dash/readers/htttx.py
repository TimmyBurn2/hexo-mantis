"""The htttx game notation: `version[1];` then `N. [q,r][q,r];` a turn, after the origin stone the server places itself."""
from __future__ import annotations

import re

from mantis._engine import Board

Cell = tuple[int, int]
#: The server places the first player's single stone here before anyone moves; the notation never writes it.
ORIGIN: Cell = (0, 0)
#: The encoding whose legal radius is the site's placement rule: a stone within hex distance 8 of a placed one.
SITE_RULES = "gnn_axis_r8"
#: The most stones the site's notation holds; the Analyzer's turn starts reach this far.
MAX_STONES = 2000
_VERSION = re.compile(r"version\s*\[\s*(\d{1,4})\s*\]", re.IGNORECASE)
_TURN = re.compile(r"(\d{1,5})\s*\.\s*((?:\[\s*-?\d{1,7}\s*,\s*-?\d{1,7}\s*\]\s*)+)")
_CELL = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


class NotationRefused(ValueError):
    """A text that is not an htttx game this engine replays; the message names the turn and the rule."""


def _statements(text: str) -> list[str]:
    """The `;`-separated statements, a pasted code fence, backticks and a byte-order mark taken off as the site does."""
    text = text.replace("﻿", "").strip()
    for fence in ("```", "`"):
        if len(text) > 2 * len(fence) and text.startswith(fence) and text.endswith(fence):
            text = text[len(fence):-len(fence)].strip()
            break
    return [s.strip() for s in text.split(";") if s.strip()]


def parse(text: str) -> list[Cell]:
    """The game's stones in placement order, the origin first. Raises: NotationRefused (a statement, a number or a stone that does not replay)."""
    statements = _statements(text)
    version = _VERSION.fullmatch(statements[0]) if statements else None
    if version is not None:
        if version.group(1) != "1":
            raise NotationRefused(f"only htttx version 1 is read, got version {version.group(1)}")
        statements = statements[1:]
    if not statements:
        raise NotationRefused("no turn to read: a turn is `N. [q,r][q,r];`")
    board = Board.with_encoding_name(SITE_RULES)
    board.apply_move(*ORIGIN)
    moves: list[Cell] = [ORIGIN]
    for i, statement in enumerate(statements, 1):
        m = _TURN.fullmatch(statement)
        if m is None:
            raise NotationRefused(f"turn {i} is not `{i}. [q,r][q,r]`: {statement[:60]!r}")
        if int(m.group(1)) != i:
            raise NotationRefused(f"turn {i} is numbered {m.group(1)}")
        cells = [(int(a), int(b)) for a, b in _CELL.findall(m.group(2))]
        if len(cells) > 2:
            raise NotationRefused(f"turn {i} places {len(cells)} stones; a turn places two")
        if len(cells) == 1 and i != len(statements):
            raise NotationRefused(f"turn {i} places one stone, which only the last turn may do")
        for q, r in cells:
            if board.winner() is not None:
                raise NotationRefused(f"turn {i}: ({q}, {r}) comes after the game was already won")
            if board.get(q, r) != 0:
                raise NotationRefused(f"turn {i}: ({q}, {r}) is already occupied")
            if len(moves) >= MAX_STONES:
                raise NotationRefused(f"turn {i}: the game holds more than {MAX_STONES} stones")
            if not board.is_legal(q, r):
                raise NotationRefused(f"turn {i}: ({q}, {r}) is more than {board.legal_move_radius()} cells from every stone")
            board.apply_move(q, r)
            moves.append((q, r))
    return moves


HEADER = "version[1];"


def turns(moves: list[Cell]) -> list[tuple[int, int, str, str]]:
    """Each turn after the first stone as htttx, the game moved so that stone is the origin: first ply, ply after, whole line, first stone's line."""
    if not moves:
        return []
    dq, dr = moves[0]
    out = []
    for n, first in enumerate(range(1, len(moves), 2), 1):
        pair = [(int(q) - dq, int(r) - dr) for q, r in moves[first:first + 2]]
        out.append((first, first + len(pair), f"{n}. " + "".join(f"[{q},{r}]" for q, r in pair) + ";",
                    f"{n}. [{pair[0][0]},{pair[0][1]}];"))
    return out


def moved(moves: list[Cell]) -> bool:
    """Whether writing the game as htttx moves it: its first stone is not already the origin."""
    return bool(moves) and tuple(moves[0]) != ORIGIN


def write(moves: list[Cell]) -> str:
    """A move list as htttx, moved so its first stone is the origin: one line per turn, a last turn of one stone when it ends mid-turn. Raises: NotationRefused (no stone after the first)."""
    lines = turns(moves)
    if not lines:
        raise NotationRefused("htttx needs a turn after the first stone")
    return HEADER + "\n" + "".join(f"{whole}\n" for _first, _end, whole, _half in lines)
