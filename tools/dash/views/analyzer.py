"""The Analyzer view: the board with one lens, the source line, the nets, the verdict, win chances, candidates and the deeper tools."""
from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

from ..readers.hexlogic import first_of_turn, owner, turn_of
from . import board
from .fmt import cell, esc, pct, script_json, short
from .games import ZOOM, turn_facts
from .games_text import NAME
from .page import Shell, render

_ICON = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="{d}" fill="currentColor"/></svg>'


def _source(body: dict[str, Any] | None, moves: list[Any], q: dict[str, str]) -> str:
    if body is None:
        return '<div class="source" id="source">A position typed in: click the board to place stones.</div>'
    back_button = '<button class="btn" id="back" type="button" hidden>Back to the game</button>'
    if body["id"] == "imported":
        return (f'<div class="source" id="source">An imported game, {len(body["moves"])} stones'
                f'<span class="chip" id="line">On the game\'s line</span>{back_button}</div>')
    back = esc(f"/run/{quote(q.get('run', ''), safe='')}/games?" + urlencode({"g": body["id"], "ply": len(moves)}))
    return (f'<div class="source" id="source">From {esc(body["channel"])} game <a href="{back}">{esc(body["id"][:12])}</a>'
            '<span class="chip" id="line">On the game\'s line</span><button class="btn" id="back" type="button" hidden>'
            "Back to the game</button></div>")


def where(moves: list[Any]) -> str:
    """The transport's line for the position: whose turn and how many stones, or which stone of a turn begun."""
    ply = len(moves)
    if first_of_turn(ply):
        return f"<b>Turn {turn_of(ply)}</b>, {NAME[owner(ply)]} to place {'one stone' if ply == 0 else 'two stones'}"
    return f"<b>Turn {turn_of(ply)}</b>, {NAME[owner(ply)]} places stone 2 of 2"


def _import(error: str | None) -> str:
    """The htttx import form: one game pasted in, read at its end; a refusal is stated beside it."""
    note = f'<p class="refused">{esc(error)}</p>' if error else ""
    return (f'<section><details class="import"{" open" if error else ""}><summary>Import a game (htttx)</summary>'
            '<form method="post" action="/analyzer"><textarea name="htttx" rows="6" spellcheck="false" '
            'placeholder="version[1];&#10;1. [1,-2][-1,1];&#10;2. [0,1][-1,0];"></textarea>'
            f'<button class="btn" type="submit">Load</button></form>{note}</details></section>')


def _nets(rows: list[dict[str, Any]] | None, panel: dict[str, Any] | None) -> str:
    if rows is None:
        return ('<section><h2>Nets</h2><div class="gap"><strong>No engines loaded.</strong>Start the server with '
                "--checkpoints DIR (stamped checkpoints; --strix adds the vendored strix pin) to read positions.</div></section>")
    chips = []
    for r in rows:
        role = "reading" if panel and r["id"] == panel.get("a_id") else "compare" if panel and r["id"] == panel.get("b_id") else ""
        cls = " c1" if role == "reading" else " c2" if role == "compare" else ""
        label = f"{r['run_id']} at {short(r['step'])}" if r.get("run_id") and r.get("step") is not None else str(r["id"])
        chips.append(f'<button class="eng{cls}" type="button" data-id="{esc(r["id"])}" aria-pressed="{"true" if role else "false"}" '
                     f'title="{esc(r["id"])} {esc(r.get("note") or "")}"><i></i>{esc(label)} <small>{role}</small></button>')
    return f'<section><h2>Nets</h2><div class="engines" id="engines">{"".join(chips)}</div></section>'


def _read(panel: dict[str, Any] | None, refused: str | None, moves: list[Any]) -> str:
    """The read's skeleton, always present so the script can fill it; filled here when the server made the first read."""
    if panel is None:
        verdict = esc(refused) if refused else "Pick a net to read this position."
        return (f'<section><h2>Turn {turn_of(len(moves))}</h2><p class="verdict" id="verdict">{verdict}</p>'
                '<p class="say" id="gameline"></p></section><section><h2>Win chance from the value head</h2>'
                '<div class="wcs" id="chances"></div></section><section><h2>On the board</h2><div class="seg" id="lens"></div>'
                '<p class="muted small" id="lensnote"></p><table class="cands" id="cands"><thead></thead><tbody></tbody></table>'
                "</section>")
    bars = "".join(
        f'<div class="wc"><span class="who {c["cls"]}"><i></i>{esc(c["label"])}</span>'
        + (f'<span>Light {pct(c["light"])}</span><span class="bar"><i style="width:{c["light"] * 100:.1f}%"></i></span>'
           f'<span>{pct(1 - c["light"])} Dark</span>' if c["light"] is not None else '<span class="muted">not read</span><span></span><span></span>')
        + "</div>" for c in panel["chances"])
    head = ["Cell", esc(panel["a"])] + ([esc(panel["b"])] if panel["b"] else []) + ["Search", ""]
    rows = "".join(
        f'<tr data-c="{r[0]},{r[1]}"><td class="num">{cell(r[0], r[1])}</td><td class="num">{pct(r[2])}</td>'
        + (f'<td class="num">{pct(r[3])}</td>' if panel["b"] else "")
        + f'<td class="num">{pct(r[4]) if r[4] is not None else "—"}</td>'
        + "<td>" + "".join(f'<span class="tag {t}">{t}</span>' for t in r[5]) + "</td></tr>" for r in panel["rows"])
    lenses = [("net", "Net"), ("search", "Search")] + ([("diff", f"{esc(panel['a'])} vs {esc(panel['b'])}")] if panel["b"] else [])
    seg = "".join(f'<button type="button" data-lens="{k}" aria-pressed="{"true" if k == "net" else "false"}">{label}</button>'
                  for k, label in lenses)
    return (f'<section><h2>Turn {turn_of(len(moves))}</h2><p class="verdict" id="verdict">{panel["verdict"]}</p>'
            f'<p class="say" id="gameline">{panel["game_line"] or ""}</p></section>'
            f'<section><h2>Win chance from the value head</h2><div class="wcs" id="chances">{bars}</div>'
            '<p class="muted small">Raw head values mapped to a chance, not calibrated; "game" is the recorded search\'s root value.</p></section>'
            f'<section><h2>On the board</h2><div class="seg" id="lens">{seg}</div><p class="muted small" id="lensnote">'
            f'Where {esc(panel["a"])} wants to play before any search: its policy, bigger is more.</p>'
            f'<table class="cands" id="cands"><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr></thead><tbody>{rows}</tbody></table></section>')


def page(runs: tuple[str, ...], rows: list[dict[str, Any]] | None, body: dict[str, Any] | None, moves: list[Any],
         panel: dict[str, Any] | None, refused: str | None, q: dict[str, str], import_error: str | None = None) -> str:
    """The whole Analyzer view; the first read rides inline so the page draws it without a request."""
    tac = (panel or {}).get("tactics") or {}
    heat = [((int(c[0]), int(c[1])), float(c[2]) / max((x[2] for x in panel["lens"]["net"]), default=1.0) if panel else 0.0, "")
            for c in (panel["lens"]["net"] if panel else [])]
    turn = ((panel or {}).get("turn") or {}).get("a") or []
    scene = board.Scene(moves=moves, ply=len(moves), frame=[tuple(m) for m in body["moves"]] if body else moves,
                        heat=heat, win_cells=[tuple(c) for c in tac.get("cells", [])] if tac.get("cls") == "win" else [],
                        block_cells=[tuple(c) for c in tac.get("cells", [])] if tac.get("cls") == "block" else [],
                        ghosts=[(tuple(c), str(i + 1) if len(turn) > 1 else "") for i, c in enumerate(turn)])
    line = len(body["moves"]) if body else 0
    data = script_json({"runs": runs, "run": q.get("run"), "g": q.get("g") if body and body["id"] != "imported" else None,
                        "game": body, "moves": moves, "panel": panel, "engines": rows, "ply": len(moves),
                        "line_starts": turn_facts(line)["turn_starts"] + [line], **turn_facts(len(moves))})
    stage = (f'<section class="stage" aria-label="Board">{_source(body, moves, q)}<div class="boardwrap">{board.render(scene)}{ZOOM}</div>'
             '<div class="keys" id="keys"></div><div class="transport">'
             f'<button class="btn" id="prev" type="button" aria-label="Previous turn">{_ICON.format(d="M11 2v10L4 7z")}</button>'
             f'<button class="btn" id="next" type="button" aria-label="Next turn">{_ICON.format(d="M3 2v10l7-5z")}</button>'
             f'<button class="btn" id="undo" type="button">Undo</button><span class="where" id="where">{where(moves)}</span>'
             '<div class="layers"><button class="btn" id="lNum" type="button" aria-pressed="false">Turn numbers</button>'
             '<button class="btn" id="lTac" type="button" aria-pressed="true">Threats</button></div></div></section>')
    tools = ('<section><h2>Look deeper</h2><div class="tools"><select class="btn" id="sims" aria-label="Simulations">'
             + "".join(f'<option value="{n}">Search {n} sims</option>' for n in (64, 256, 1024))
             + '</select><button class="btn" id="search" type="button">Search</button><button class="btn" id="sym" type="button">'
             'Symmetry check</button><button class="btn" id="copy" type="button">Copy position</button></div>'
             '<p class="muted small" id="deeper"></p></section>') if rows is not None else ""
    side = (f'<aside class="panel" id="panel" aria-label="Analysis">{_nets(rows, panel)}<div id="read">'
            f'{_read(panel, refused, moves)}</div>{tools}{_import(import_error)}</aside>')
    html = f'<main class="an">{stage}{side}</main><script type="application/json" id="state">{data}</script>'
    return render("mantis analyzer", Shell("analyzer", runs, q.get("run")), html, scripts=("board.js", "analyzer.js"), app=True)
