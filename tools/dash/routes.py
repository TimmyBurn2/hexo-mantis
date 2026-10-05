"""The Games routes: the page, a window of the list as JSON, one game's payload; the shard index is read under its run's lock."""
from __future__ import annotations

from typing import Any

from .readers.games import GameView
from .readers.record import RunRecord
from .readers.shards import SORTS, WINNERS, Filter, Page
from .serve import Hub, Reply, as_json, text
from .views import games as games_view
from .views.games_text import hour_of

#: Game payloads a server keeps, newest use last.
PAYLOADS = 64


def _query(raw: dict[str, list[str]], channels: list[str]) -> dict[str, str]:
    """The list's query, every value checked against its closed set; anything else is dropped, never echoed."""
    q = {k: v[0] for k, v in raw.items() if v}
    out: dict[str, str] = {}
    if q.get("kind") in channels:
        out["kind"] = q["kind"]
    if q.get("search") == "1":
        out["search"] = "1"
    if q.get("winner") in WINNERS:
        out["winner"] = q["winner"]
    if q.get("sort") in SORTS and q.get("sort") != "newest":
        out["sort"] = q["sort"]
    return out


def _listing(rec: RunRecord, query: dict[str, str], after: str | None, n: int) -> tuple[Page, list[str], int]:
    where = Filter(channel=query.get("kind"), with_search=query.get("search") == "1", winner=query.get("winner"))
    with rec.lock:
        return rec.games.page(after, n, sort=query.get("sort", "newest"), where=where), rec.games.channels(), rec.games.total


def game_payload(hub: Hub, rec: RunRecord, game_id: str) -> dict[str, Any] | None:
    """One game's payload by id, computed once and kept; None when the run has no such game. Raises: OSError."""
    key = (rec.label, game_id)
    with hub.payload_lock:
        if key in hub.payloads:
            hub.payloads.move_to_end(key)
            return hub.payloads[key]
    with rec.lock:
        where = rec.games.locate(game_id)
        raw = rec.games.fetch(game_id)
        shard = rec.games.shards[where[0]].path.name if where is not None else ""
    if raw is None:
        return None
    body = games_view.payload(GameView.from_record(raw), rec.label, hour_of(shard))
    with hub.payload_lock:
        hub.payloads[key] = body
        while len(hub.payloads) > PAYLOADS:
            hub.payloads.popitem(last=False)
    return body


def _ply(raw: dict[str, list[str]], body: dict[str, Any] | None) -> int:
    end = len(body["moves"]) if body else 0
    value = (raw.get("ply") or [""])[0]
    return min(end, max(0, int(value))) if value.isdigit() else end


def games_page(hub: Hub, parts: list[str], raw: dict[str, list[str]]) -> Reply:
    """`/run/<label>/games?kind=&search=&winner=&sort=&after=&g=&ply=`."""
    rec = hub.records.get(parts[1])
    if rec is None:
        return text(404, f"no run {parts[1]!r}")
    with rec.lock:
        channels = rec.games.channels()
    query = _query(raw, channels)
    after = (raw.get("after") or [None])[0]
    listing, present, total = _listing(rec, query, after, games_view.WINDOW)
    wanted = (raw.get("g") or [""])[0].strip()
    chosen = wanted or (listing.rows[0]["id"] if listing.rows else "")
    body = game_payload(hub, rec, chosen) if chosen else None
    if wanted and body is None:
        return text(404, f"run {rec.label} has no game {wanted!r}")
    html = games_view.page(rec.label, hub.labels, present, listing, query, body, _ply(raw, body), total)
    return Reply(200, "text/html; charset=utf-8", html.encode("utf-8"))


def api_run(hub: Hub, parts: list[str], raw: dict[str, list[str]]) -> Reply:
    """`/api/run/<label>/games?…` (a window of light rows) and `/api/run/<label>/game/<id>` (one payload)."""
    rec = hub.records.get(parts[2]) if len(parts) >= 4 else None
    if rec is None:
        return as_json(404, {"ok": False, "refused": "no such run"})
    if parts[3] == "games" and len(parts) == 4:
        with rec.lock:
            channels = rec.games.channels()
        n = (raw.get("n") or ["200"])[0]
        listing, _present, total = _listing(rec, _query(raw, channels), (raw.get("after") or [None])[0],
                                            min(games_view.WINDOW, int(n) if n.isdigit() else games_view.WINDOW))
        return as_json(200, {"ok": True, "rows": listing.rows, "next": listing.next_cursor, "total": listing.total,
                             "games": total})
    if parts[3] == "game" and len(parts) == 5:
        body = game_payload(hub, rec, parts[4])
        return as_json(200, {"ok": True, "game": body}) if body else as_json(404, {"ok": False, "refused": "no such game"})
    return as_json(404, {"ok": False, "refused": "no such route"})


GET = {"run/games": games_page, "api/run": api_run}
