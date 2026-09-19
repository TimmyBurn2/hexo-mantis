"""The ONE analysis record: the net's read, the head's search, the tactics verdict; values for the side to move."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from mantis._engine import Board
from mantis.diagnostics.tactics import analyze as tactics_analyze

from .engines import ANALYZER_GUMBEL_SEED, Child, RawRead, Search, raw_argmax
from .instruments import sweep
from .position import Position, build_board, position_record


def _rows(children: list[Child], *, visits: bool) -> list[list[float | int]]:
    """Children for the wire, `[q, r, prior]` by prior or `[q, r, prior, visits, q]` by visits."""
    if visits:
        return [[c.cell[0], c.cell[1], round(c.prior, 4), c.visits, round(c.q, 4)]
                for c in sorted(children, key=lambda c: (-c.visits, -c.prior))]
    return [[c.cell[0], c.cell[1], round(c.prior, 4)] for c in sorted(children, key=lambda c: -c.prior)]


def raw_record(raw: RawRead, derivation: str) -> dict[str, Any]:
    """The `raw` block: the net's value, the max-prior cell, every legal child's prior (never the returned move)."""
    return {"value": round(raw.value, 4), "argmax": list(raw_argmax(raw.children)),
            "policy": _rows(raw.children, visits=False), "ms": round(raw.ms, 1), "derivation": derivation}


def search_record(s: Search, sims: int, derivation: str, seed: int = ANALYZER_GUMBEL_SEED) -> dict[str, Any]:
    """The `search` block: the head's root value, move, children by visits, and the two counters."""
    return {"sims": int(sims), "root_visits": s.root_visits, "seed": seed, "root_value": round(s.root_value, 4),
            "argmax": [s.argmax[0], s.argmax[1]], "children": _rows(s.children, visits=True),
            "quiescence_fires": s.quiescence_fires, "ms": round(s.ms, 1), "derivation": derivation}


def verdict(cls: str, cells: set[tuple[int, int]], argmax: tuple[int, int] | list[int]) -> str:
    """One argmax against the tactics class: WINS / MISSES WIN / BLOCKS / MISSES THE BLOCK / LOST1 / quiet."""
    cell = (int(argmax[0]), int(argmax[1]))
    if cls == "win":
        return "argmax WINS" if cell in cells else f"argmax MISSES WIN at {sorted(cells)}"
    if cls == "block":
        return "argmax BLOCKS" if cell in cells else f"argmax MISSES THE BLOCK — block set {sorted(cells)}"
    if cls == "lost1":
        return "no block exists (LOST1)"
    return "quiet"


def tactics_record(board: Board, radius: int, raw_argmax_cell: list[int] | None,
                   search_argmax: list[int] | None) -> dict[str, Any]:
    """The census reading of the position for the mover (`tactics.analyze`) and the verdict on each argmax."""
    stones = board.get_stones()
    q = np.array([s[0] for s in stones], dtype=np.int64)
    r = np.array([s[1] for s in stones], dtype=np.int64)
    p = np.array([s[2] for s in stones], dtype=np.int64)
    mover, k = int(board.current_player), int(board.moves_remaining)
    row = tactics_analyze(q, r, p, mover, k, radius=radius)
    cls, cells = row.forced(k)
    fwm = board.forced_win_move(2)
    return {
        "class": cls, "k": k, "cells": sorted([int(c[0]), int(c[1])] for c in cells),
        "opp_fours": [sorted([int(c[0]), int(c[1])] for c in four) for four in row.fours],
        "forced_win_move": [int(fwm[0]), int(fwm[1])] if fwm is not None else None,
        "verdict": {"raw": verdict(cls, cells, raw_argmax_cell) if raw_argmax_cell is not None else None,
                    "search": verdict(cls, cells, search_argmax) if search_argmax is not None else None},
        "derivation": f"mantis.diagnostics.tactics.analyze(k={k}, radius={radius})",
    }


def analyze(engine: Any, moves: list[tuple[int, int]], sims: int, *, symmetry: bool = False) -> dict[str, Any]:
    """The record for `moves` at `sims` (0 = raw only) on `engine`; raises PositionRefused from `build_board`."""
    t0 = time.perf_counter()
    pos: Position = build_board(moves, engine.encoding)
    rec: dict[str, Any] = {"engine": engine.card, "position": position_record(pos), "perspective": "to_move"}
    if pos.winner is not None:
        absent = {"absent": f"position is terminal ({pos.winner} wins)"}
        rec.update(raw=absent, search=absent, tactics={"class": "terminal", "winner": pos.winner},
                   symmetry=absent if symmetry else {"absent": "not requested"})
    else:
        raw = raw_record(engine.raw_read(pos.board), engine.raw_derivation)
        search: dict[str, Any] = (search_record(engine.search(pos.board, sims), sims, engine.head_derivation)
                                  if sims >= 1 else {"absent": "sims=0 (raw only)"})
        rec.update(raw=raw, search=search,
                   tactics=tactics_record(pos.board, engine.radius, raw["argmax"], search.get("argmax")),
                   symmetry=sweep(engine, moves) if symmetry else {"absent": "not requested"})
    rec["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return rec


__all__ = ["analyze", "raw_record", "search_record", "tactics_record", "verdict"]
