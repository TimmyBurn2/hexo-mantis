"""The Analyzer's desk: the engines behind the one analyst thread, a position read by one net and compared with another."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .readers import htttx
from .routes import game_payload
from .serve import Hub
from .views import analyzer as analyzer_view
from .views.analyzer_text import Turn, compose
from .views.games import turn_facts

#: How long a request waits on the analyst before it is answered 504, and the deepest search a request may ask for.
TIMEOUT_SEC = 120.0
MAX_SIMS = 4096


def checked(req: dict[str, Any], *keys: str) -> str | None:
    """Why a request is refused: a named field of the wrong type, or `sims` outside 0..`MAX_SIMS`; None when it is sound."""
    for key in keys:
        if req.get(key) is not None and not isinstance(req[key], str):
            return f"{key} must be text, got {type(req[key]).__name__}"
    sims = req.get("sims", 0)
    if sims is not None and (isinstance(sims, bool) or not isinstance(sims, int) or not 0 <= sims <= MAX_SIMS):
        return f"sims must be an integer from 0 to {MAX_SIMS}, got {sims!r}"
    return None


@dataclass(frozen=True)
class Context:
    """Where a position came from: the game, how far along its line, and the game's recorded search when on it."""

    run: str | None
    game_id: str | None
    on_line: bool
    off: int
    entry: dict[str, Any] | None
    nxt: tuple[int, int] | None
    second: tuple[int, int] | None


class Desk:
    """Owns the dispatcher and its analyst; `None` engines means the Analyzer was started without checkpoints."""

    def __init__(self, checkpoints: list[Path], *, strix: bool, device: str, threads: int | None) -> None:
        """Discover the nets and start the analyst. Raises: OSError (an unreadable checkpoint directory)."""
        # The engine layer (torch) is the optional dependency: it loads only when the Analyzer is given nets.
        from .analyst import Analyst
        from .engine.dispatch import Dispatcher
        from .engine.engines import discover
        infos = discover(checkpoints) if checkpoints else []
        self.dispatcher = Dispatcher(infos, device=device, threads=threads, strix=strix)
        self.analyst = Analyst(self.dispatcher.handle, timeout_sec=TIMEOUT_SEC, on_stop=self.dispatcher.close)
        self.analyst.start()

    def rows(self) -> list[dict[str, Any]]:
        """The engine rows; no engine is touched."""
        return self.dispatcher.rows()

    def default_pair(self, run_id: str | None = None) -> tuple[str | None, str | None]:
        """The newest loadable net to read and the one before it to compare, from `run_id`'s nets when it has any."""
        from .engine.engines import MANTIS
        loadable = [r for r in self.rows() if r.get("kind") == MANTIS]
        own = [r["id"] for r in loadable if r.get("run_id") == run_id] or [r["id"] for r in loadable]
        return (own[-1] if own else None), (own[-2] if len(own) > 1 else None)

    def _submit(self, engine: str, moves: list[tuple[int, int]], sims: int, client: str,
                symmetry: bool = False) -> dict[str, Any]:
        text_moves = ";".join(f"{q},{r}" for q, r in moves)
        return self.analyst.submit({"op": "analyze", "engine": engine, "moves": text_moves, "sims": sims,
                                    "symmetry": symmetry, "client": client})

    def turn(self, engine: str, moves: list[tuple[int, int]], rec: dict[str, Any], sims: int, client: str) -> Turn:
        """The stones the engine plays for the rest of the turn: its choice now, and with two to place its choice after it."""
        pos = rec.get("position") or {}
        first, searched = _choice(rec), bool((rec.get("search") or {}).get("argmax"))
        if pos.get("winner") or first is None:
            return Turn([])
        if pos.get("moves_remaining") != 2:
            return Turn([first], searched=searched)
        # Its own key: a follow-up never supersedes the next request's first read, only an older follow-up.
        out = self._submit(engine, [*moves, first], sims, f"{client}:turn")
        body = out["body"]
        if out["status"] != 200 or body.get("superseded"):
            why = "superseded" if body.get("superseded") else str(body.get("refused") or f"status {out['status']}")
            return Turn([first], unread=why, searched=searched)
        rec2 = body.get("record") or {}
        second = _choice(rec2)
        ms = float((rec2.get("search") or {}).get("ms") or 0.0)
        return Turn([first] + ([second] if second is not None else []), searched=searched, ms=ms)

    def read(self, a: str, b: str | None, moves: list[tuple[int, int]], ctx: Context, *, sims: int = 0,
             symmetry: bool = False, client: str = "page", names: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
        """A's record (at `sims`, the symmetry sweep when asked) and B's raw read, each net's whole turn, composed into the panel."""
        first = self._submit(a, moves, sims, client, symmetry)
        if first["status"] != 200 or first["body"].get("superseded"):
            return first["status"], first["body"]
        rec_a, rec_b, turn_b = first["body"]["record"], None, None
        if b:
            second = self._submit(b, moves, 0, client)
            if second["status"] != 200 or second["body"].get("superseded"):
                return second["status"], second["body"]
            rec_b = second["body"].get("record")
            turn_b = self.turn(b, moves, rec_b or {}, 0, client)
        turn_a = self.turn(a, moves, rec_a, sims, client)
        if "superseded" in (turn_a.unread, turn_b.unread if turn_b else None):
            return 200, {"superseded": True}
        try:
            notation: str | None = htttx.write(moves)
        except htttx.NotationRefused:
            notation = None
        where = analyzer_view.where(moves, (rec_a.get("position") or {}).get("winner"))
        panel = {**compose(rec_a, rec_b, ctx.entry, ctx.nxt, ctx.second, turn_a=turn_a, turn_b=turn_b, names=names),
                 "a_id": a, "b_id": b,
                 "where": where, "htttx": notation, "htttx_moved": htttx.moved(moves), **turn_facts(len(moves))}
        return 200, {"ok": True, "panel": panel, "record": rec_a, "context": ctx.__dict__}

    def close(self) -> None:
        """Stop the analyst; the engines close on its thread."""
        self.analyst.stop()


class LazyDesk:
    """The desk built on the first Analyzer request: a server nobody asks to analyse never loads the engine layer."""

    def __init__(self, checkpoints: list[Path], *, strix: bool, device: str, threads: int | None) -> None:
        self._args = (checkpoints, strix, device, threads)
        self._desk: Desk | None = None
        self._lock = threading.Lock()

    def __call__(self) -> Desk:
        """The desk, built once. Raises: OSError (an unreadable checkpoint directory)."""
        with self._lock:
            if self._desk is None:
                checkpoints, strix, device, threads = self._args
                self._desk = Desk(checkpoints, strix=strix, device=device, threads=threads)
            return self._desk

    def close(self) -> None:
        """Close the desk if it was ever built."""
        if self._desk is not None:
            self._desk.close()


DeskOf = Callable[[], Desk] | None


def _choice(rec: dict[str, Any]) -> tuple[int, int] | None:
    """The record's own choice: its search's when it searched, else the net's highest prior."""
    for block in ("search", "raw"):
        best = (rec.get(block) or {}).get("argmax")
        if isinstance(best, list) and len(best) == 2:
            return int(best[0]), int(best[1])
    return None


def context(hub: Hub, run: str | None, game_id: str | None, moves: list[tuple[int, int]],
            line_moves: list[tuple[int, int]] | None = None) -> Context:
    """The position against its game (a recorded one, else an imported line): on it, or N stones off; recorded search only on it."""
    rec = hub.records.get(run or "")
    body = game_payload(hub, rec, game_id) if rec is not None and game_id else None
    if body is None and not line_moves:
        return Context(run, None, False, 0, None, None, None)
    line = [tuple(m) for m in body["moves"]] if body is not None else list(line_moves or [])
    common = next((i for i, (x, y) in enumerate(zip(moves, line, strict=False)) if x != y), min(len(moves), len(line)))
    on = common == len(moves)
    ply = len(moves)
    nxt = line[ply] if on and ply < len(line) else None
    second = line[ply + 1] if nxt is not None and ply % 2 == 1 and ply + 1 < len(line) else None
    entry = body["stats"].get(str(ply)) if on and body is not None else None
    return Context(run, game_id if body is not None else None, on, len(moves) - common, entry, nxt, second)
