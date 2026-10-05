"""The Analyzer view: the board with one lens, the source line, the nets, the verdict, win chances, candidates and the deeper tools."""
from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

from ..readers.hexlogic import first_of_turn, owner, turn_of, turn_size
from ..readers.htttx import MAX_STONES
from . import board
from .fmt import at, cell, esc, pct, script_json
from .games import turn_facts
from .games_text import KIND, NAME
from .page import Shell, render

_ICON = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="{d}" fill="currentColor"/></svg>'


def _source(body: dict[str, Any] | None, moves: list[Any], q: dict[str, str]) -> str:
    if body is None:
        return '<div class="source" id="source">Your own position. Click the board to place stones.</div>'
    back_button = '<button class="btn" id="back" type="button" hidden>Back to the game</button>'
    if body["id"] == "imported":
        return (f'<div class="source" id="source">Imported game, {len(body["moves"])} stones'
                f'<span class="chip" id="line">On the game\'s line</span>{back_button}</div>')
    back = esc(f"/run/{quote(q.get('run', ''), safe='')}/games?" + urlencode({"g": body["id"], "ply": len(moves)}))
    kind = str(KIND.get(body["channel"]) or body["channel"]).lower()
    return (f'<div class="source" id="source">From {esc(kind)} game <a href="{back}">{esc(body["id"][:12])}</a>'
            '<span class="chip" id="line">On the game\'s line</span><button class="btn" id="back" type="button" hidden>'
            "Back to the game</button></div>")


def where(moves: list[Any], winner: str | None = None) -> str:
    """The transport's line for the position: who won, whose turn and how many stones, or which stone of a turn begun."""
    ply = len(moves)
    if winner in ("p1", "p2"):
        return f"<b>Final position</b>, {NAME[0 if winner == 'p1' else 1]} has six in a row"
    if first_of_turn(ply):
        return f"<b>Turn {turn_of(ply)}</b>, {NAME[owner(ply)]} to place {'one stone' if turn_size(ply) == 1 else 'two stones'}"
    return f"<b>Turn {turn_of(ply)}</b>, {NAME[owner(ply)]} places stone 2 of 2"


def _import(refusal: tuple[str, str] | None, panel: dict[str, Any] | None) -> str:
    """The htttx import form, read at the game's end with the nets now chosen; a refusal is stated with the text kept."""
    why, text = refusal or ("", "")
    nets = "".join(f'<input type="hidden" name="{k}" value="{esc(panel[f"{k}_id"] or "")}">' for k in ("a", "b")) if panel else ""
    return (f'<section><details class="import"{" open" if refusal else ""}><summary>Import a HeXO game (htttx)</summary>'
            f'<form method="post" action="/analyzer" id="importform">{nets}<textarea name="htttx" rows="6" spellcheck="false" '
            f'placeholder="version[1];&#10;1. [1,-2][-1,1];&#10;2. [0,1][-1,0];">{esc(text)}</textarea>'
            '<button class="btn" type="submit">Load</button></form>'
            + (f'<p class="refused">{esc(why)}</p>' if why else "") + "</details></section>")


def _net_label(run: str, row: dict[str, Any]) -> str:
    return at(run, row["step"]) if row.get("step") is not None else str(row["id"])


def _nets(rows: list[dict[str, Any]] | None, panel: dict[str, Any] | None, names: dict[str, str]) -> str:
    """Two pickers, Reading and Compare, each listing the loadable nets by run, newest first."""
    if rows is None:
        return ('<section><h2>Nets</h2><div class="gap"><strong>No nets loaded.</strong>Start the server with '
                "--checkpoints DIR to read positions (--strix adds the strix pin).</div></section>")
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("kind") != "snapshot_gap":
            group = names.get(str(r.get("run_id")), str(r.get("run_id"))) if r.get("run_id") else str(r.get("kind", "other")).capitalize()
            groups.setdefault(group, []).append(r)
    gaps = sum(1 for r in rows if r.get("kind") == "snapshot_gap")

    def options(chosen: str | None) -> str:
        out = []
        for run, members in groups.items():
            opts = "".join(f'<option value="{esc(r["id"])}"{" selected" if r["id"] == chosen else ""}>{esc(_net_label(run, r))}</option>'
                           for r in reversed(members))
            out.append(f'<optgroup label="{esc(run)}">{opts}</optgroup>')
        return "".join(out)
    a, b = (panel or {}).get("a_id"), (panel or {}).get("b_id")
    pick = "" if a else '<option value="" selected disabled>Pick a net</option>'
    note = f'<p class="muted small">{gaps} unstamped snapshot{"s" if gaps != 1 else ""} cannot be read and are not listed.</p>' if gaps else ""
    return ('<section><h2>Nets</h2><div class="nets">'
            f'<label class="c1"><i></i>Reading<select class="btn" id="netA">{pick}{options(a)}</select></label>'
            f'<label class="c2"><i></i>Compare<select class="btn" id="netB"><option value="">none</option>{options(b)}</select></label>'
            f"</div>{note}</section>")


def _read(panel: dict[str, Any] | None, refused: str | None, moves: list[Any]) -> str:
    """The read's skeleton, always present so the script can fill it; filled here when the server made the first read."""
    if panel is None:
        verdict = esc(refused) if refused else "Pick a net to read this position."
        return (f'<section><h2>Turn {turn_of(len(moves))}</h2><p class="verdict" id="verdict">{verdict}</p>'
                '<p class="say" id="gameline"></p></section><section><h2>Win chance</h2>'
                '<div class="wcs" id="chances"></div></section><section><h2>On the board</h2><div class="seg" id="lens"></div>'
                '<p class="muted small" id="lensnote"></p><table class="cands" id="cands"><thead></thead><tbody></tbody></table>'
                "</section>")
    bars = "".join(
        f'<div class="wc"><span class="who {c["cls"]}"><i></i>{esc(c["label"])}</span>'
        + (f'<span>Light {pct(c["light"])}</span><span class="bar"><i style="width:{c["light"] * 100:.1f}%"></i></span>'
           f'<span>{pct(1 - c["light"])} Dark</span>' if c["light"] is not None else '<span class="muted">not read</span><span></span><span></span>')
        + "</div>" for c in panel["chances"])
    head = ['Cell', f'<span class="c1">{esc(panel["a"])}</span>'] + ([f'<span class="c2">{esc(panel["b"])}</span>'] if panel["b"] else []) + [
        esc(panel["search_head"]), ""]
    rows = "".join(
        f'<tr data-c="{r[0]},{r[1]}"><td class="num">{cell(r[0], r[1])}</td><td class="pct">{pct(r[2])}</td>'
        + (f'<td class="pct">{pct(r[3])}</td>' if panel["b"] else "")
        + f'<td class="pct">{pct(r[4]) if r[4] is not None else ""}</td>'
        + "<td>" + "".join(f'<span class="tag {t}">{t}</span>' for t in r[5]) + "</td></tr>" for r in panel["rows"])
    lenses = [("net", "Net"), ("search", "Search")] + ([("diff", "Difference")] if panel["b"] else [])
    seg = "".join(f'<button type="button" data-lens="{k}" aria-pressed="{"true" if k == "net" else "false"}">{label}</button>'
                  for k, label in lenses)
    return (f'<section><h2>Turn {turn_of(len(moves))}</h2><p class="verdict" id="verdict">{panel["verdict"]}</p>'
            f'<p class="say" id="gameline">{panel["game_line"] or ""}</p></section>'
            f'<section><h2>Win chance</h2><div class="wcs" id="chances">{bars}</div>'
            '<p class="muted small">Value heads are uncalibrated.</p></section>'
            f'<section><h2>On the board</h2><div class="seg" id="lens">{seg}</div><p class="muted small" id="lensnote">'
            f'Policy before search. {esc(panel["turn"]["note"])}</p>'
            f'<table class="cands" id="cands"><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr></thead><tbody>{rows}</tbody></table></section>')


def page(runs: tuple[str, ...], rows: list[dict[str, Any]] | None, body: dict[str, Any] | None, moves: list[Any],
         panel: dict[str, Any] | None, refused: str | None, q: dict[str, str], refusal: tuple[str, str] | None = None,
         names: dict[str, str] | None = None) -> str:
    """The whole Analyzer view; the first read rides inline so the page draws it without a request."""
    tac = (panel or {}).get("tactics") or {}
    heat = [((int(c[0]), int(c[1])), float(c[2]) / max((x[2] for x in panel["lens"]["net"]), default=1.0) if panel else 0.0, "")
            for c in (panel["lens"]["net"] if panel else [])]
    marks = ((panel or {}).get("turn") or {}).get("ghosts") or []
    scene = board.Scene(moves=moves, ply=len(moves), frame=[tuple(m) for m in body["moves"]] if body else moves,
                        heat=heat, win_cells=[tuple(c) for c in tac.get("cells", [])] if tac.get("cls") == "win" else [],
                        block_cells=[tuple(c) for c in tac.get("cells", [])] if tac.get("cls") == "block" else [],
                        ghosts=[((int(m[0]), int(m[1])), str(m[2]), str(m[3])) for m in marks])
    data = script_json({"runs": runs, "run": q.get("run"), "g": q.get("g") if body and body["id"] != "imported" else None,
                        "game": body, "moves": moves, "panel": panel, "engines": rows, "ply": len(moves),
                        "starts": turn_facts(MAX_STONES)["turn_starts"], **turn_facts(len(moves))})
    stage = (f'<section class="stage" aria-label="Board">{_source(body, moves, q)}<div class="boardwrap">{board.render(scene)}{board.ZOOM}</div>'
             '<div class="keys" id="keys"></div><div class="transport">'
             f'<button class="btn" id="first" type="button" aria-label="Start of the game">{_ICON.format(d="M2 2h2v10H2zM12 2v10L5 7z")}</button>'
             f'<button class="btn" id="prev" type="button" aria-label="Previous turn">{_ICON.format(d="M11 2v10L4 7z")}</button>'
             f'<button class="btn" id="next" type="button" aria-label="Next turn">{_ICON.format(d="M3 2v10l7-5z")}</button>'
             f'<button class="btn" id="last" type="button" aria-label="End of the game"{"" if body else " hidden"}>'
             f'{_ICON.format(d="M10 2h2v10h-2zM2 2v10l7-5z")}</button>'
             f'<button class="btn" id="undo" type="button">Undo</button><span class="where" id="where">{panel["where"] if panel else where(moves)}</span>'
             '<div class="layers"><button class="btn" id="lNum" type="button" aria-pressed="false">Turn numbers</button>'
             '<button class="btn" id="lTac" type="button" aria-pressed="true">Threats</button></div></div></section>')
    tools = ('<section><h2>Look deeper</h2><div class="tools"><select class="btn" id="sims" aria-label="Simulations">'
             + "".join(f'<option value="{n}">Search {n} sims</option>' for n in (64, 256, 1024))
             + '</select><button class="btn" id="search" type="button">Search</button><button class="btn" id="sym" type="button">'
             'Symmetry check</button><button class="btn" id="copy" type="button">Copy position</button></div>'
             '<p class="muted small" id="deeper"></p></section>') if rows is not None else ""
    side = (f'<aside class="panel" id="panel" aria-label="Analysis">{_nets(rows, panel, names or {})}<div id="read">'
            f'{_read(panel, refused, moves)}</div>{tools}{_import(refusal, panel)}</aside>')
    html = f'<main class="an">{stage}{side}</main><script type="application/json" id="state">{data}</script>'
    return render("mantis analyzer", Shell("analyzer", runs, q.get("run")), html, scripts=("board.js", "analyzer.js"), app=True)
