"""The Analyzer's routes: the page from a recorded game, an htttx import or nothing; the reads; the engines."""
from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.parse import parse_qs

from .desk import DeskOf, checked, context
from .engine.position import PositionRefused, parse_moves
from .readers import htttx
from .routes import game_payload
from .serve import Hub, Reply, as_json, text
from .views import analyzer as analyzer_view

#: The longest htttx text the import form takes: a 1 000-turn game is well under it.
MAX_IMPORT = 200_000


def _digits(value: str) -> bool:
    return value.isascii() and value.isdigit()


def _render(hub: Hub, desk_of: DeskOf, q: dict[str, str], body: dict[str, Any] | None,
            imported: list[tuple[int, int]] | None = None, import_error: str | None = None) -> Reply:
    """The page at `ply` along the game (a recorded one or an imported line), the first read made here when there are nets."""
    desk = desk_of() if desk_of is not None else None
    line = imported if imported is not None else [tuple(m) for m in body["moves"]] if body else []
    ply = min(len(line), int(q["ply"])) if _digits(q.get("ply", "")) else len(line)
    moves = line[:ply]
    panel, refused = None, None
    if desk is not None:
        a, b = q.get("a") or desk.default_pair()[0], q.get("b") or desk.default_pair()[1]
        if a:
            ctx = context(hub, q.get("run"), q.get("g") if body else None, moves, imported)
            status, out = desk.read(a, b if b != a else None, moves, ctx, client=f"render-{uuid.uuid4().hex}")
            panel, refused = (out.get("panel"), None) if status == 200 else (None, out.get("refused"))
    slim = ({"id": body["id"], "channel": body["channel"], "moves": body["moves"]} if body else
            {"id": "imported", "channel": "imported", "moves": [list(m) for m in imported]} if imported else None)
    html = analyzer_view.page(hub.labels, desk.rows() if desk else None, slim, moves, panel, refused, q, import_error)
    return Reply(200, "text/html; charset=utf-8", html.encode("utf-8"))


def page(desk_of: DeskOf) -> Any:
    """`GET /analyzer?run=&g=&ply=&a=&b=`: the board and, with engines, the first read rendered server-side."""
    def handle(hub: Hub, parts: list[str], raw: dict[str, list[str]]) -> Reply:
        q = {k: v[0] for k, v in raw.items() if v}
        run, gid = q.get("run"), q.get("g")
        rec = hub.records.get(run or "")
        body = game_payload(hub, rec, gid) if rec is not None and gid else None
        if gid and body is None:
            return text(404, f"no game {gid!r} in run {run!r}")
        return _render(hub, desk_of, q, body)
    return handle


def _import(hub: Hub, desk_of: DeskOf, raw: bytes) -> Reply:
    """`POST /analyzer` with an htttx game in the form's `htttx` field: the page at the game's end, or the refusal named."""
    if len(raw) > MAX_IMPORT:
        return _render(hub, desk_of, {}, None, import_error=f"the game is longer than {MAX_IMPORT} bytes")
    form = {k: v[0] for k, v in parse_qs(raw.decode("utf-8", "replace")).items() if v}
    try:
        moves = htttx.parse(form.get("htttx", ""))
    except htttx.NotationRefused as exc:
        return _render(hub, desk_of, {}, None, import_error=str(exc))
    return _render(hub, desk_of, {k: v for k, v in form.items() if k in ("a", "b", "ply")}, None, imported=moves)


def post(hub: Hub, desk_of: DeskOf) -> Any:
    """`POST /analyzer` (an htttx import), `POST /api/read` (the composed panel) and `POST /api/analyze` (unchanged)."""
    def handle(path: str, raw: bytes) -> Reply:
        if path == "/analyzer":
            return _import(hub, desk_of, raw)
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
            why = checked(req, "engine", "client")
            if why:
                return as_json(400, {"seq": req.get("seq"), "ok": False, "refused": why})
            req["op"] = "analyze"
            out = desk.analyst.submit(req)
            return as_json(int(out["status"]), out["body"])
        if path != "/api/read":
            return as_json(404, {"ok": False, "refused": f"no route {path}"})
        why = checked(req, "a", "b", "run", "g", "client", "line") or (None if req.get("a") else "a names no engine")
        if why:
            return as_json(400, {"seq": req.get("seq"), "ok": False, "refused": why})
        try:
            moves = parse_moves(req.get("moves", ""))
            line = parse_moves(req["line"]) if req.get("line") else None
        except PositionRefused as exc:
            return as_json(400, {"seq": req.get("seq"), "ok": False, "refused": str(exc)})
        ctx = context(hub, req.get("run"), req.get("g"), moves, line)
        status, out = desk.read(str(req["a"]), req.get("b") or None, moves, ctx, sims=int(req.get("sims") or 0),
                                symmetry=bool(req.get("symmetry")), client=str(req.get("client") or "page"))
        return as_json(status, {**out, "seq": req.get("seq")})
    return handle


def engines(desk_of: DeskOf) -> Any:
    """`GET /api/engines`: the desk's rows, or none when the server was started without nets."""
    def handle(hub: Hub, parts: list[str], raw: dict[str, list[str]]) -> Reply:
        desk = desk_of() if desk_of is not None else None
        return as_json(200, {"engines": desk.rows() if desk else [], "started": desk is not None})
    return handle
