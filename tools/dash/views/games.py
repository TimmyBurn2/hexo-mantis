"""The Games view: an app frame with the list window, the board with its strip and transport, the panel in sentences."""
from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

from ..readers import chances, htttx
from ..readers.games import GameView
from ..readers.hexlogic import first_of_turn, owner, turn_of
from ..readers.shards import Page
from . import board, games_text
from .fmt import esc, num, script_json, short
from .page import Shell, render

KINDS = (("selfplay", "Self-play"), ("promotion", "Gate"), ("external", "External"), ("random_floor", "Random"))
WINNER_LABEL = (("", "Any winner"), ("p1", "Light wins"), ("p2", "Dark wins"), ("cap", "Ended at the cap"))
SORT_LABEL = (("newest", "Newest"), ("longest", "Longest"), ("shortest", "Shortest"))
#: The list's window, and its size when a page asks for more (the operator's 2026-09-14 direction: 200, then load more).
WINDOW = 200
_GLYPH = '<svg class="glyph" width="14" height="14" viewBox="-1.1 -1.1 2.2 2.2" aria-hidden="true"><polygon class="{c}" points="{p}"/></svg>'
_ICON = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="{d}" fill="currentColor"/></svg>'
_TRANSPORT = (("first", "First turn", "M2 2h2v10H2zM12 2v10L5 7z"), ("prev", "Previous turn", "M11 2v10L4 7z"),
              ("next", "Next turn", "M3 2v10l7-5z"), ("last", "Final position", "M10 2h2v10h-2zM2 2v10l7-5z"))
_SEARCH = ('<svg class="search-ico" width="14" height="12" viewBox="0 0 14 12" aria-label="search recorded"><rect x="0" y="6" '
           'width="3" height="6" rx="1"/><rect x="5" y="2" width="3" height="10" rx="1"/><rect x="10" y="4" width="3" height="8" rx="1"/></svg>')


def glyph(cls: str) -> str:
    """A small hexagon in a stone or mark class, for lists and legends."""
    return _GLYPH.format(c=cls, p=board.hex_points((0, 0), 1.0).replace("0.000,", "0,"))


def payload(g: GameView, run_label: str, hour: str | None) -> dict[str, Any]:
    """The game as the page steps through it: the reader's payload plus every position's sentences, all derived here."""
    readings = g.readings()
    body = g.payload(readings)
    pos = []
    for ply, t in enumerate(readings):
        think = games_text.thought(g, ply, set(t.cells), t.cls == "win")
        row = {"where": games_text.where(g, ply), "threat": games_text.threat(t), "think": think}
        if first_of_turn(ply):
            row["turn"] = games_text.turn_thought(g, ply, set(t.cells), t.cls == "win")
        pos.append(row)
    pos.append({"where": games_text.where(g, len(g.moves)), "threat": None, "think": None})
    cells = [tuple(m) for m in g.moves]
    notation = {"head": htttx.HEADER, "turns": htttx.turns(cells), "moved": htttx.moved(cells)}
    tp = chances.turning_point(chances.points(g.stats))
    body.update(run=run_label, head=games_text.headline(g), facts=games_text.facts(g, run_label, hour),
                turning=games_text.turning(tp, g.game_id), pos=pos, htttx=notation, **turn_facts(len(g.moves)))
    return body


def turn_facts(plies: int) -> dict[str, list[int]]:
    """Who placed each stone, the turn it belongs to, and each turn's first ply: the browser steps with these, never derives them."""
    return {"owners": [owner(i) for i in range(plies)], "turn_of": [turn_of(i) for i in range(plies)],
            "turn_starts": [i for i in range(plies) if first_of_turn(i)]}


def row_html(row: dict[str, Any], run: str, selected: bool, query: dict[str, str]) -> str:
    """One list row: the winner's stone, the kind, the net's step, the stones, the search glyph; every value escaped."""
    res, term = row.get("res"), row.get("term")
    win = glyph("s1" if res == "p1" else "s2") if res in ("p1", "p2") else '<span class="muted" title="no winner">–</span>'
    sub = {"promotion": "vs anchor", "external": f"vs {row.get('rung', '?')}"}.get(str(row.get("ch")), "")
    sub += " cap" if term == "ply_cap" else ""
    step = row.get("step")
    net = short(step) if isinstance(step, int) and step >= 0 else '<span class="muted">—</span>'
    href = "?" + urlencode({**query, "g": row["id"]})
    label = dict(KINDS).get(str(row.get("ch")), str(row.get("ch")))
    return (f'<a class="lrow" role="option" aria-selected="{"true" if selected else "false"}" href="{esc(href)}" '
            f'data-id="{esc(row["id"])}"><span>{win}</span><span class="kind">{esc(label)}<small>{esc(sub)}</small></span>'
            f'<span class="r">{net}</span><span class="r">{num(row.get("pl"))}</span>'
            f'<span>{_SEARCH if row.get("stats") else ""}</span></a>')


def _filters(run: str, present: list[str], query: dict[str, str], total: int, shown: int) -> str:
    kinds = [("", "All")] + [(k, label) for k, label in KINDS if k in present]
    rest = {k: v for k, v in query.items() if k in ("search", "winner", "sort")}
    chips = "".join(f'<a href="?{esc(urlencode({**rest, **({"kind": k} if k else {})}))}" '
                    f'aria-pressed="{"true" if query.get("kind", "") == k else "false"}">{label}</a>' for k, label in kinds)
    winner = "".join(f'<option value="{k}"{" selected" if query.get("winner", "") == k else ""}>{label}</option>'
                     for k, label in WINNER_LABEL)
    order = "".join(f'<option value="{k}"{" selected" if query.get("sort", "newest") == k else ""}>{label}</option>'
                    for k, label in SORT_LABEL)
    search = " checked" if query.get("search") == "1" else ""
    return (f'<form class="filters" method="get" action="/run/{esc(quote(run, safe=""))}/games"><div class="seg" role="group" '
            f'aria-label="Kind of game">{chips}</div><input type="hidden" name="kind" value="{esc(query.get("kind", ""))}">'
            f'<div class="row"><label><input type="checkbox" name="search" value="1"{search}> With search</label>'
            f'<select class="btn" name="winner" aria-label="Winner">{winner}</select>'
            f'<select class="btn" name="sort" aria-label="Order">{order}</select><noscript><button class="btn">Show</button>'
            f'</noscript></div><div class="row"><input class="btn goto" name="g" placeholder="Go to game id" '
            f'aria-label="Go to game id"></div><div class="count muted">{num(shown)} of {num(total)} games</div></form>')


def _neighbours(starts: list[int], end: int, ply: int) -> tuple[int, int]:
    """The turn starts before and after `ply` (the final position counts as one), for turn-by-turn stepping."""
    stops = starts + [end]
    prev = max((p for p in stops if p < ply), default=0)
    nxt = min((p for p in stops if p > ply), default=end)
    return prev, nxt


def _stage(body: dict[str, Any] | None, ply: int) -> str:
    if body is None:
        return '<section class="stage"><div class="gap"><strong>No game selected.</strong>Pick one from the list.</div></section>'
    moves = [tuple(m) for m in body["moves"]]
    end = ply >= len(moves)
    pos = body["pos"][min(ply, len(moves))]
    turn = pos.get("turn")
    think = (turn or pos.get("think")) or {}
    t = body["tactics"][ply] if not end else None
    ghosts = ([(tuple(c), str(i + 1)) for i, c in enumerate(turn["stones"])] if turn and len(turn["stones"]) > 1
              else [(moves[ply], "")] if not end else [])
    scene = board.Scene(moves=moves, ply=ply, frame=moves, heat=board.heat_of(think.get("cands") or []),
                        win_cells=[tuple(c) for c in t["cells"]] if t and t["cls"] == "win" else [],
                        block_cells=[tuple(c) for c in t["cells"]] if t and t["cls"] == "block" else [],
                        ghosts=ghosts, win_line=[tuple(c) for c in body["win"]] if end and body["win"] else None)
    base = {"g": body["id"]}
    prev, nxt = _neighbours(body["turn_starts"], len(moves), ply)
    targets = {"first": 0, "prev": prev, "next": nxt, "last": len(moves)}
    nav = "".join(f'<a class="btn" href="?{esc(urlencode({**base, "ply": targets[i]}))}" id="{i}" aria-label="{a}">'
                  f"{_ICON.format(d=d)}</a>" for i, a, d in _TRANSPORT)
    return (f'<section class="stage" aria-label="Board"><div class="boardwrap">{board.render(scene)}{board.ZOOM}</div>'
            f'<div class="keys" id="keys"></div><div class="trace" id="trace"></div>'
            f'<div class="transport">{nav}<button class="btn" id="play" type="button" hidden>Play</button>'
            f'<span class="where" id="where">{pos["where"]}</span><div class="layers">'
            '<button class="btn" id="lNum" type="button" aria-pressed="false">Turn numbers</button>'
            '<button class="btn" id="lSearch" type="button" aria-pressed="true">Bot\'s search</button>'
            '<button class="btn" id="lTac" type="button" aria-pressed="true">Threats</button></div></div></section>')


def _panel(body: dict[str, Any] | None, ply: int, run: str) -> str:
    if body is None:
        return '<aside class="panel" id="panel"></aside>'
    facts = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in body["facts"])
    pos = body["pos"][min(ply, len(body["moves"]))]
    parts = [f'<section><h2>Game</h2><p class="verdict">{body["head"]}</p>'
             + (f'<p class="say">{body["turning"]}</p>' if body["turning"] else "") + f'<dl class="facts">{facts}</dl></section>']
    if pos.get("threat"):
        parts.append(f'<section><h2>Turn {turn_of(ply)}</h2><div class="tacline">{pos["threat"]}</div></section>')
    if pos.get("turn"):
        lines = "".join(f'<p class="say"><strong>{esc(label)}.</strong> {text}</p>' for label, text in pos["turn"]["texts"])
        parts.append(f"<section><h2>What the bot played this turn</h2>{lines}</section>")
    elif pos.get("think"):
        parts.append(f'<section><h2>What the bot thought</h2><p class="say">{pos["think"]["text"]}</p></section>')
    href = f"/analyzer?{urlencode({'run': run, 'g': body['id'], 'ply': ply})}"
    parts.append(f'<section><div class="actions"><a class="btn" href="{esc(href)}">Open in Analyzer</a></div></section>')
    return f'<aside class="panel" id="panel" aria-label="Game details">{"".join(parts)}</aside>'


def page(run: str, runs: tuple[str, ...], present: list[str], listing: Page, query: dict[str, str],
         body: dict[str, Any] | None, ply: int, total: int) -> str:
    """The whole Games view; the selected game's payload rides inline so the board steps without a second request."""
    rows = "".join(row_html(r, run, body is not None and r["id"] == body["id"], {k: v for k, v in query.items()
                                                                                if k in ("kind", "search", "winner", "sort")})
                   for r in listing.rows)
    more = ""
    if listing.next_cursor:
        q = {k: v for k, v in query.items() if k in ("kind", "search", "winner", "sort")}
        more = (f'<a class="btn more" href="?{esc(urlencode({**q, "after": listing.next_cursor}))}" data-after="'
                f'{esc(listing.next_cursor)}">Load {WINDOW} more</a>')
    data = script_json({"run": run, "game": body, "ply": ply, "query": query, "next": listing.next_cursor})
    html = (f'<main class="games"><aside class="list" aria-label="Games">{_filters(run, present, query, total, listing.total)}'
            '<div class="lhead"><span title="Winner">W</span><span>Game</span><span>Net</span><span>Stones</span><span></span>'
            f'</div><div class="scroller" id="scroller" role="listbox">{rows}{more}</div></aside>{_stage(body, ply)}'
            f'{_panel(body, ply, run)}</main><script type="application/json" id="state">{data}</script>')
    return render(f"mantis {run} games", Shell("games", runs, run), html, scripts=("board.js", "games.js"), app=True)
