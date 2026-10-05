"""The Analyzer panel composed from engine records: the verdict, win chances, the three lenses and the candidates, all derived here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fmt import cell, esc, short
from .games_text import NAME, cells

Cell = tuple[int, int]
#: Cells listed beside the board: the read net's top cells, then every block or win cell.
TOP = 7


def name_of(record: dict[str, Any]) -> str:
    """A net as the pages name it: its run and step (`run11a2 at 33k`), or the engine's id."""
    card = record.get("engine") or {}
    if card.get("run_id") is not None and card.get("step") is not None:
        return f"{card['run_id']} at {short(card['step'])}"
    return str(card.get("id", "engine"))


def _policy(record: dict[str, Any] | None) -> dict[Cell, float]:
    raw = (record or {}).get("raw") or {}
    return {(int(r[0]), int(r[1])): float(r[2]) for r in raw.get("policy") or []}


def _light(record: dict[str, Any] | None, mover: int) -> float | None:
    """Light's chance from the raw head value (the mover's frame), uncalibrated."""
    v = ((record or {}).get("raw") or {}).get("value")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    mine = (max(-1.0, min(1.0, float(v))) + 1) / 2
    return round(mine if mover == 0 else 1 - mine, 4)


def _search_shares(record: dict[str, Any] | None, game_entry: dict[str, Any] | None) -> tuple[dict[Cell, float], str]:
    """The analyzer's own search when it ran, else the game's recorded search on the game's line; and which it is."""
    search = (record or {}).get("search") or {}
    visited = [c for c in search.get("children") or [] if int(c[3]) > 0]
    if visited:
        total = sum(int(c[3]) for c in visited)
        return {(int(c[0]), int(c[1])): int(c[3]) / total for c in visited}, f"the analyzer's search, {search.get('sims')} sims"
    if game_entry and game_entry.get("top"):
        total = game_entry.get("n") or 1
        return {(int(v[0]), int(v[1])): int(v[2]) / total for v in game_entry["top"]}, "the game's recorded search"
    return {}, ""


@dataclass(frozen=True)
class Turn:
    """A net's stones for the rest of the turn, why its second went unread, whether search chose them, the second read's ms."""

    stones: list[Cell] = field(default_factory=list)
    unread: str | None = None
    searched: bool = False
    ms: float = 0.0


def _plays(name: str, turn: Turn) -> str:
    said = f"{esc(name)} plays {' then '.join(cell(*c) for c in turn.stones)}"
    return said + (f", its second stone not read ({esc(turn.unread)})" if turn.unread else "")


def compose(a: dict[str, Any], b: dict[str, Any] | None, game_entry: dict[str, Any] | None,
            game_next: Cell | None, game_second: Cell | None, *, turn_a: Turn | None = None,
            turn_b: Turn | None = None) -> dict[str, Any]:
    """The panel for one position: `a` is read, `b` compared; `game_entry` is the game's recorded search at this ply, if on its line."""
    pos = a.get("position") or {}
    mover = 0 if pos.get("to_move") == "p1" else 1
    tac = a.get("tactics") or {}
    cls, tcells = tac.get("class"), [(int(c[0]), int(c[1])) for c in tac.get("cells") or []]
    me, them = NAME[mover], NAME[1 - mover]
    pa, pb = _policy(a), _policy(b)
    na, nb = name_of(a), name_of(b) if b is not None else None
    if nb == na:
        na, nb = f"{na} ({(a.get('engine') or {}).get('sha8', 'A')})", f"{nb} ({((b or {}).get('engine') or {}).get('sha8', 'B')})"
    first = {na: max(pa, key=lambda c: pa[c]) if pa else None}
    if b is not None and nb is not None:
        first[nb] = max(pb, key=lambda c: pb[c]) if pb else None
    if cls == "terminal":
        verdict = f"<strong>{NAME[0 if pos.get('winner') == 'p1' else 1]} has six in a row</strong>: the game is over."
    elif cls == "win":
        verdict = f"<strong>{me} can win now</strong>, at {cells(tcells, 'or')}."
    elif cls == "block":
        verdict = f"<strong>{me} must block</strong> {them}'s four this turn, at {cells(tcells, 'or')}."
    elif cls == "lost1":
        verdict = f"<strong>{me} cannot block</strong> every four {them} has this turn."
    else:
        verdict = "Nothing is forced this turn."
    picks = {n: c for n, c in first.items() if c is not None}
    if cls in ("win", "block") and picks:
        act = "win" if cls == "win" else "block"
        chosen = set(picks.values())
        if len(picks) > 1 and len(chosen) == 1 and next(iter(chosen)) in tcells:
            verdict += f" Both nets {act} with their first choice, {cell(*next(iter(chosen)))}."
        else:
            verdict += " " + " ".join(f"{esc(n)} {'does' if c in tcells else 'does not'} {act} with its first choice, {cell(*c)}."
                                      for n, c in picks.items())
    turn_a, turn_b = turn_a or Turn(), turn_b or Turn()
    if turn_a.stones:
        if b is not None and turn_a.stones == turn_b.stones and not (turn_a.unread or turn_b.unread):
            verdict += f" Both nets play {' then '.join(cell(*c) for c in turn_a.stones)}."
        else:
            verdict += " " + "; ".join(_plays(n, t) for n, t in ((na, turn_a), (nb, turn_b)) if n is not None and t.stones) + "."
    game_line = None
    if game_next is not None:
        marks = {c: " (a block)" if cls == "block" and c in tcells else " (a win)" if cls == "win" and c in tcells else ""
                 for c in (game_next, game_second) if c is not None}
        game_line = f"In the game {me} played {cell(*game_next)}{marks[game_next]}"
        game_line += f" and then {cell(*game_second)}{marks[game_second]}." if game_second else "."
    shares, search_source = _search_shares(a, game_entry)
    union = sorted(set(pa) | set(pb)) if b is not None else []
    diff = [[c[0], c[1], round(pa.get(c, 0.0) - pb.get(c, 0.0), 5)] for c in union]
    ranked = sorted(pa, key=lambda c: -pa[c])[:TOP]
    listed = ranked + [c for c in tcells if c not in ranked]
    tag = "wins" if cls == "win" else "blocks"
    rows = [[c[0], c[1], round(pa.get(c, 0.0), 4), None if b is None else round(pb.get(c, 0.0), 4),
             round(shares[c], 4) if c in shares else None, [tag] if c in tcells else []] for c in listed]
    chances = [{"label": na, "cls": "c1", "light": _light(a, mover)}]
    if b is not None:
        chances.append({"label": nb, "cls": "c2", "light": _light(b, mover)})
    if game_entry is not None and isinstance(game_entry.get("v"), (int, float)):
        mine = (max(-1.0, min(1.0, float(game_entry["v"]))) + 1) / 2
        chances.append({"label": "game", "cls": "cf", "light": round(mine if mover == 0 else 1 - mine, 4)})
    return {"verdict": verdict, "game_line": game_line, "chances": chances, "a": na, "b": nb,
            "lens": {"net": [[c[0], c[1], pa[c]] for c in ranked], "search": [[c[0], c[1], s] for c, s in shares.items()],
                     "diff": diff}, "search_source": search_source, "rows": rows,
            "tactics": {"cls": cls, "cells": [list(c) for c in tcells]}, "mover": mover,
            "turn": {"a": [list(c) for c in turn_a.stones], "b": [list(c) for c in turn_b.stones], "second_ms": turn_a.ms,
                     "note": (f"The numbered stones are {na}'s turn, from its {'search' if turn_a.searched else 'policy'}."
                              if len(turn_a.stones) > 1 else "")}}
