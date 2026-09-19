"""The page shell: the viewer's tokens, one grid, the two static files inlined at render → ONE served page."""
from __future__ import annotations

import html as _html
from pathlib import Path

_WEB = Path(__file__).resolve().parent / "web"
_KEYS = ("<kbd>←</kbd><kbd>→</kbd> ply · <kbd>Home</kbd><kbd>End</kbd> · <kbd>u</kbd> undo · <kbd>h</kbd> visits · "
         "<kbd>p</kbd> prior · <kbd>q</kbd> Q · <kbd>t</kbd> tactics · <kbd>s</kbd> symmetry · <kbd>v</kbd> value trace · "
         "<kbd>l</kbd> copy link · <kbd>c</kbd> copy position · <kbd>?</kbd> this sheet")
_CARD = """<div class="card" data-slot="">
<div class="cardhead"><span class="slotno"></span><select class="cardengine"></select><span class="cardmeta mute"></span><button class="drop" title="remove this engine">×</button></div>
<div class="persp mute">values for the side to move</div>
<table class="readout"><tbody>
<tr class="r-raw"><th>net</th><td class="num">—</td><td class="cell">—</td><td class="mute">—</td></tr>
<tr class="r-head"><th>head</th><td class="num">—</td><td class="cell">—</td><td class="mute">—</td></tr>
</tbody></table>
<div class="tac">—</div>
<table class="children"><thead><tr><th>cell</th><th>prior</th><th>visits</th><th>Q</th></tr></thead><tbody></tbody></table>
<div class="more mute"></div>
<div class="panel sym"><b>symmetry</b> <span class="mute">(s): not requested</span></div>
<div class="panel trace"><b>value trace</b> <span class="mute">(v): not requested</span><svg class="tracesvg" hidden></svg></div>
<div class="status"></div>
</div>"""


def render(title: str = "mantis analyzer") -> str:
    """The whole page as a string; the CSS and the two scripts under `web/` are read and inlined at each render."""
    css = (_WEB / "analyzer.css").read_text(encoding="utf-8")
    js = "\n".join((_WEB / name).read_text(encoding="utf-8") for name in ("board.js", "analyzer.js"))
    t = _html.escape(title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<style>{css}</style></head>
<body>
<header><h1>{t}</h1><p id="hdr">no engine yet</p><p class="keys">keys: {_KEYS}</p></header>
<main>
<section id="left">
<svg id="board" viewBox="0 0 10 10" tabindex="0"></svg>
<div id="hover" class="mute">hover a cell</div>
<div id="help" class="mute">Click a cell to place the next stone (ply 0 is p1's single, then two stones per turn). Placing behind the end replaces the tail. The faint cells are the active engine's legal window; the ringed cell is the active card's argmax, small digits are the other cards'.</div>
</section>
<section id="side">
<div class="row"><span class="chips" id="budget"><button data-value="32" title="quick">32</button><button data-value="" id="deploychip" title="the run's eval.gate.deploy_sims">deploy</button><button data-value="1024" title="deep: about 23 s on a CPU host">1024</button><input id="sims" type="number" min="0" step="1" title="sims (0 = the net only)"></span>
<label class="auto"><input id="auto" type="checkbox" checked> auto</label><button id="go">search</button><button id="addslot" title="another engine on this position">+ slot</button></div>
<div class="row" id="pos"></div>
<div class="row"><input id="moves" placeholder="q,r;q,r;… (paste to import)" spellcheck="false"><button id="copy" title="c">copy</button></div>
<div id="cards"></div>
</section>
</main>
<template id="cardtpl">{_CARD}</template>
<div id="sheet" hidden><div>keys — {_KEYS}</div></div>
<script>{js}</script>
</body></html>
"""


__all__ = ["render"]
