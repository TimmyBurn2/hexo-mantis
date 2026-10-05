"""The Analyzer's desk: the engines behind the one analyst thread, a position read by one net and compared with another, the routes."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine.position import PositionRefused, parse_moves
from .routes import game_payload
from .serve import Hub, Reply, as_json, text
from .views import analyzer as analyzer_view
from .views.analyzer_text import compose

#: How long a request waits on the analyst before it is answered 504.
TIMEOUT_SEC = 120.0


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
        return self.dispatcher.rows()

    def default_pair(self) -> tuple[str | None, str | None]:
        """The newest loadable net to read and the one before it to compare."""
        from .engine.engines import MANTIS
        loadable = [r["id"] for r in self.rows() if r.get("kind") == MANTIS]
        return (loadable[-1] if loadable else None), (loadable[-2] if len(loadable) > 1 else None)

    def read(self, a: str, b: str | None, moves: list[tuple[int, int]], ctx: Context, *, sims: int = 0,
             symmetry: bool = False, client: str = "page") -> tuple[int, dict[str, Any]]:
        """A's record (at `sims`, with the symmetry sweep when asked) and B's raw read, composed into the panel."""
        text_moves = ";".join(f"{q},{r}" for q, r in moves)
        first = self.analyst.submit({"op": "analyze", "engine": a, "moves": text_moves, "sims": sims,
                                     "symmetry": symmetry, "client": client})
        if first["status"] != 200 or first["body"].get("superseded"):
            return first["status"], first["body"]
        rec_b = None
        if b:
            second = self.analyst.submit({"op": "analyze", "engine": b, "moves": text_moves, "sims": 0, "client": client})
            if second["status"] != 200:
                return second["status"], second["body"]
            rec_b = second["body"].get("record")
        rec_a = first["body"]["record"]
        panel = {**compose(rec_a, rec_b, ctx.entry, ctx.nxt, ctx.second), "a_id": a, "b_id": b,
                 "where": analyzer_view.where(moves)}
        return 200, {"ok": True, "panel": panel, "record": rec_a, "context": ctx.__dict__}

    def close(self) -> None:
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
        if self._desk is not None:
            self._desk.close()


DeskOf = Callable[[], Desk] | None


def context(hub: Hub, run: str | None, game_id: str | None, moves: list[tuple[int, int]]) -> Context:
    """The position against its game: on the line, or a variation N stones off it; the game's search only on the line."""
    rec = hub.records.get(run or "")
    body = game_payload(hub, rec, game_id) if rec is not None and game_id else None
    if body is None:
        return Context(run, None, False, 0, None, None, None)
    line = [tuple(m) for m in body["moves"]]
    common = next((i for i, (x, y) in enumerate(zip(moves, line, strict=False)) if x != y), min(len(moves), len(line)))
    on = common == len(moves)
    ply = len(moves)
    nxt = line[ply] if on and ply < len(line) else None
    second = line[ply + 1] if nxt is not None and ply % 2 == 1 and ply + 1 < len(line) else None
    return Context(run, game_id, on, len(moves) - common, body["stats"].get(str(ply)) if on else None, nxt, second)


def page(desk_of: DeskOf) -> Any:
    """`GET /analyzer?run=&g=&ply=&a=&b=`: the board and, with engines, the first read rendered server-side."""
    def handle(hub: Hub, parts: list[str], raw: dict[str, list[str]]) -> Reply:
        desk = desk_of() if desk_of is not None else None
        q = {k: v[0] for k, v in raw.items() if v}
        run, gid = q.get("run"), q.get("g")
        rec = hub.records.get(run or "")
        body = game_payload(hub, rec, gid) if rec is not None and gid else None
        if gid and body is None:
            return text(404, f"no game {gid!r} in run {run!r}")
        game_moves = [tuple(m) for m in body["moves"]] if body else []
        ply = min(len(game_moves), int(q["ply"])) if q.get("ply", "").isdigit() else len(game_moves)
        moves = game_moves[:ply]
        panel, refused = None, None
        if desk is not None:
            a, b = q.get("a") or desk.default_pair()[0], q.get("b") or desk.default_pair()[1]
            if a:
                status, out = desk.read(a, b if b != a else None, moves, context(hub, run, gid, moves))
                panel, refused = (out.get("panel"), None) if status == 200 else (None, out.get("refused"))
        html = analyzer_view.page(hub.labels, desk.rows() if desk else None, body, moves, panel, refused, q)
        return Reply(200, "text/html; charset=utf-8", html.encode("utf-8"))
    return handle


def post(hub: Hub, desk_of: DeskOf) -> Any:
    """`POST /api/read` (the composed panel) and `POST /api/analyze` (the analyzer's own request, unchanged)."""
    def handle(path: str, raw: bytes) -> Reply:
        desk = desk_of() if desk_of is not None else None
        if desk is None:
            return as_json(503, {"ok": False, "refused": "no engines: start the server with --checkpoints"})
        try:
            req = json.loads(raw)
            if not isinstance(req, dict):
                raise ValueError
        except ValueError:
            return as_json(400, {"ok": False, "refused": "the body is not a JSON object"})
        if path == "/api/analyze":
            req["op"] = "analyze"
            out = desk.analyst.submit(req)
            return as_json(int(out["status"]), out["body"])
        if path != "/api/read":
            return as_json(404, {"ok": False, "refused": f"no route {path}"})
        try:
            moves = parse_moves(req.get("moves", ""))
            sims = int(req.get("sims") or 0)
        except (PositionRefused, TypeError, ValueError) as exc:
            return as_json(400, {"ok": False, "refused": str(exc) or "bad sims"})
        ctx = context(hub, req.get("run"), req.get("g"), moves)
        status, out = desk.read(str(req.get("a")), req.get("b") or None, moves, ctx, sims=max(0, sims),
                                symmetry=bool(req.get("symmetry")), client=str(req.get("client", "page")))
        return as_json(status, {**out, "seq": req.get("seq")})
    return handle


def engines(desk_of: DeskOf) -> Any:
    """`GET /api/engines`: the desk's rows, or none when the server was started without nets."""
    def handle(hub: Hub, parts: list[str], raw: dict[str, list[str]]) -> Reply:
        desk = desk_of() if desk_of is not None else None
        return as_json(200, {"engines": desk.rows() if desk else [], "started": desk is not None})
    return handle
