"""The ONE analysis record: the net's read, the head's search, the tactics verdict; values for the side to move."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from mantis._engine import Board
from mantis.diagnostics.tactics import analyze as tactics_analyze

from .engines import ANALYZER_GUMBEL_SEED, RawRead, Search
from .instruments import sweep
from .position import Position, build_board, position_record

TERMINAL = "position is terminal"
NOT_REQUESTED = {"absent": "not requested"}


def raw_record(raw: RawRead, derivation: str) -> dict[str, Any]:
    """The `raw` block: the net's value, the max-prior cell, every legal child's prior (never the returned move)."""
    ranked = sorted(raw.children, key=lambda c: -c[2])
    best = ranked[0][0]
    return {"value": round(raw.value, 4), "argmax": [int(best[0]), int(best[1])],
            "policy": [[int(c[0][0]), int(c[0][1]), round(float(c[2]), 4)] for c in ranked],
            "ms": round(raw.ms, 1), "derivation": derivation}


def search_record(s: Search, sims: int, derivation: str, seed: int = ANALYZER_GUMBEL_SEED) -> dict[str, Any]:
    """The `search` block: the head's root value, move, children by visits, and the two counters."""
    ranked = sorted(s.children, key=lambda c: (-c[3], -c[2]))
    return {"sims": int(sims), "root_visits": s.root_visits, "seed": seed, "root_value": round(s.root_value, 4),
            "argmax": [s.argmax[0], s.argmax[1]],
            "children": [[int(c[0][0]), int(c[0][1]), round(float(c[2]), 4), int(c[3]), round(float(c[4]), 4)]
                         for c in ranked],
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


def tactics_record(board: Board, radius: int, raw_argmax: list[int] | None,
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
        "verdict": {"raw": verdict(cls, cells, raw_argmax) if raw_argmax is not None else None,
                    "search": verdict(cls, cells, search_argmax) if search_argmax is not None else None},
        "derivation": f"mantis.diagnostics.tactics.analyze(k={k}, radius={radius})",
    }


def analyze(engine: Any, moves: list[tuple[int, int]], sims: int, *, symmetry: bool = False) -> dict[str, Any]:
    """The record for `moves` at `sims` (0 = raw only) on `engine`; raises PositionRefused from `build_board`."""
    t0 = time.perf_counter()
    pos: Position = build_board(moves, engine.encoding)
    rec: dict[str, Any] = {"engine": engine.card, "position": position_record(pos), "perspective": "to_move"}
    if pos.winner is not None:
        absent = {"absent": f"{TERMINAL} ({pos.winner} wins)"}
        rec.update(raw=absent, search=absent, tactics={"class": "terminal", "winner": pos.winner})
    else:
        raw = raw_record(engine.raw_read(pos.board), engine.raw_derivation)
        search: dict[str, Any] = (search_record(engine.search(pos.board, sims), sims, engine.head_derivation)
                                  if sims >= 1 else {"absent": "sims=0 (raw only)"})
        rec.update(raw=raw, search=search,
                   tactics=tactics_record(pos.board, engine.radius, raw["argmax"], search.get("argmax")))
    rec["symmetry"] = sweep(engine, moves) if symmetry and pos.winner is None else dict(NOT_REQUESTED)
    rec["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return rec


__all__ = ["TERMINAL", "analyze", "raw_record", "search_record",
           "tactics_record", "verdict"]
