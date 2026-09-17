"""The one self-contained page: tier 1 hero strip, tier 2 charts, tier 3 collapsed details."""
from __future__ import annotations

from . import hero, tier2, tier3
from .fmt import esc
from .health import assess
from .model import Gaps, HeroCell, Panel
from .reader import Record
from .strength import rung_series

_CSS = """
:root{color-scheme:light dark;--fg:#1b1b1b;--bg:#fdfdfc;--muted:#5d5d5d;--rule:#d8d5cf;
--absent:#8a6d1f;--ok:#2f7a45;--warn:#b45309;--bad:#b3261e;--s1:#3b6ea5;--s2:#7a3ba5;
--s3:#a5843b;--s4:#a5533b;--s5:#3b8a5a;--s6:#6b6b6b}
@media (prefers-color-scheme:dark){:root{--fg:#e7e5e0;--bg:#141413;--muted:#a3a09a;
--rule:#33322f;--absent:#d9b45a;--ok:#6fcf8a;--warn:#f0a94a;--bad:#f08c84;--s1:#7fa6d4;
--s2:#b48ad4;--s3:#d4b16a;--s4:#d48a6a;--s5:#6ab88a;--s6:#a3a09a}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;
max-width:1280px;margin-inline:auto}
h1{font-size:24px;font-weight:600;margin:0 0 2px}h2{font-size:14px;font-weight:600;margin:0 0 4px}
h3{font-size:11px;font-weight:600;color:var(--muted);margin:8px 0 2px}
h4{font-size:11px;font-weight:600;color:var(--muted);margin:14px 0 4px}
.sub,.note,.reads,figcaption,.x-labels,.y-labels,.legend,.detail,footer,th,.gap,.absent,
.reasons,.dropped{font-size:11px}
.sub,.note,.reads,figcaption,.x-labels,.y-labels,footer{color:var(--muted)}
.sub{margin:0 0 16px}.note{margin:6px 0 0}.reads{margin:0 0 8px}
code,.value,.x-labels,.y-labels{font-family:ui-monospace,monospace}
.hero{display:grid;grid-template-columns:repeat(2,1fr);gap:16px 12px;margin:0 0 24px}
@media (min-width:720px){.hero{grid-template-columns:repeat(7,1fr)}}
.cell{min-width:0}.cell .label{display:block;font-size:11px;color:var(--muted)}
.cell .value{display:block;font-size:24px;font-weight:600;line-height:1.2;overflow-wrap:anywhere}
.cell .detail{display:block;color:var(--muted);margin-top:2px}
.cell.ok .value{color:var(--ok)}.cell.warn .value{color:var(--warn)}.cell.bad .value{color:var(--bad)}
.cell.absent .value,.cell.unmeasured .value{color:var(--absent)}
.badge .value{font-size:14px;text-transform:none}
.reasons{margin:4px 0 0;padding-left:16px}.reasons li.bad{color:var(--bad)}
.reasons li.warn{color:var(--warn)}.reasons li.unmeasured{color:var(--absent)}
.panel,details.tier3{border-top:1px solid var(--rule);padding:14px 0 6px;margin:0}
.panel.tier2:first-of-type .chart svg{max-height:none}
.multiples{display:grid;grid-template-columns:1fr;gap:8px 16px}
@media (min-width:720px){.multiples{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}}
.multiple{min-width:0}
.chart{display:grid;grid-template-columns:auto 1fr;grid-template-rows:auto auto auto auto;
gap:2px 6px;margin:4px 0 8px}
.chart svg{grid-column:2;grid-row:1;width:100%;height:auto;display:block}
.y-labels{grid-column:1;grid-row:1;display:flex;flex-direction:column;justify-content:space-between;
text-align:right}
.x-labels{grid-column:2;grid-row:2;display:flex;justify-content:space-between}
.legend{grid-column:1/-1;grid-row:3;list-style:none;margin:2px 0 0;padding:0}
.legend li{display:inline-block;margin-right:12px}
figcaption{grid-column:1/-1;grid-row:4}
.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;
vertical-align:-1px;background:currentColor}
.swatch.promoted{background:var(--ok)}.swatch.rejected{background:var(--bad)}
.swatch.broken{background:none;border:1.5px solid var(--absent)}
.swatch.idle{background:none;border:1.5px solid currentColor}
.s1{color:var(--s1)}.s2{color:var(--s2)}.s3{color:var(--s3)}.s4{color:var(--s4)}
.s5{color:var(--s5)}.s6{color:var(--s6)}
svg .band{fill:currentColor;fill-opacity:.18;stroke:none}
svg .line{fill:none;stroke:currentColor;stroke-width:1.6;vector-effect:non-scaling-stroke}
svg .bar{fill:currentColor;fill-opacity:.7}
svg .whisker{stroke:currentColor;stroke-width:1;stroke-opacity:.8}
svg .marker{fill:var(--fg);stroke:none}svg .marker.promoted{fill:var(--ok)}
svg .marker.rejected{fill:var(--bad)}svg .marker.broken{fill:none;stroke:var(--absent);stroke-width:2}
svg .marker.idle{fill:none;stroke:currentColor;stroke-width:2}
svg .half{stroke:var(--muted);stroke-dasharray:4 4;stroke-width:1}
svg .rule{stroke:var(--bad);stroke-dasharray:6 3;stroke-width:1}
svg .tick{stroke-width:2}svg .tick.alert{stroke:var(--warn)}svg .tick.fire{stroke:var(--bad)}
svg .tick.abort{stroke:var(--bad)}
.absent,.gap{color:var(--absent);margin:6px 0}.warn{color:var(--warn);margin:6px 0}
.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:11px;min-width:100%}
th,td{text-align:left;padding:3px 12px 3px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
th{color:var(--muted);font-weight:600}
details.tier3>summary{cursor:pointer;font-size:14px;font-weight:600}
.tier3s{display:grid;grid-template-columns:1fr;gap:0 24px}
@media (min-width:720px){.tier3s{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}}
.tier3s details.tier3[open]{grid-column:1/-1}
#expand{font:inherit;font-size:11px;margin:16px 0 0;background:none;color:var(--fg);
border:1px solid var(--rule);padding:4px 10px;border-radius:4px;cursor:pointer}
.readout{position:absolute;pointer-events:none;background:var(--bg);color:var(--fg);
border:1px solid var(--rule);padding:2px 6px;font:11px ui-monospace,monospace}
footer{margin-top:24px;border-top:1px solid var(--rule);padding-top:8px}
"""

_JS = """
(function(){var o=document.createElement('output');o.className='readout';o.hidden=true;
document.body.appendChild(o);
function f(v){return Math.abs(v)>=1000?Math.round(v).toLocaleString():String(+(+v).toPrecision(3))}
document.querySelectorAll('figure.chart svg').forEach(function(s){
s.addEventListener('pointermove',function(e){var r=s.getBoundingClientRect(),vb=s.viewBox.baseVal,
px=(e.clientX-r.left)/r.width*vb.width,ls=s.querySelectorAll('polyline.line'),b=null,bd=1e9;
ls.forEach(function(l){var p=l.points;for(var i=0;i<p.numberOfItems;i++){var q=p.getItem(i),
d=Math.abs(q.x-px);if(d<bd){bd=d;b={x:q.x,y:q.y,l:l.dataset.label||''}}}});if(!b)return;
var x0=+s.dataset.x0,x1=+s.dataset.x1,y0=+s.dataset.y0,y1=+s.dataset.y1;
o.textContent=b.l+' \\u2014 x '+f(x0+(b.x-2)/(vb.width-4)*(x1-x0))+': '+f(y1-(b.y-2)/(vb.height-4)*(y1-y0));
o.style.left=(e.pageX+12)+'px';o.style.top=(e.pageY+12)+'px';o.hidden=false});
s.addEventListener('pointerleave',function(){o.hidden=true})});
var b=document.getElementById('expand');if(b){b.hidden=false;b.addEventListener('click',function(){
var open=b.dataset.open!=='1';document.querySelectorAll('details.tier3').forEach(function(d){d.open=open});
b.dataset.open=open?'1':'0';b.textContent=open?'Collapse all details':'Expand all details'})}})();
"""


def _cell(c: HeroCell) -> str:
    if c.label == "Health":
        return (f'<div class="cell badge {c.state}"><span class="label">{esc(c.label)}</span>'
                f'<span class="value">{esc(c.value)}</span>{c.detail}</div>')
    return (f'<div class="cell {c.state}"><span class="label">{esc(c.label)}</span>'
            f'<span class="value">{esc(c.value)}</span><span class="detail">{c.detail}</span></div>')


def _panel(p: Panel) -> str:
    note = f'<p class="note">{esc(p.note)}</p>' if p.note else ""
    return (f'<section class="panel tier2" id="{esc(p.slug)}"><h2>{esc(p.title)}</h2>'
            f'{p.body}{note}<p class="reads">reads: <code>{esc(p.reads)}</code></p></section>')


def _details(p: Panel) -> str:
    note = f'<p class="note">{esc(p.note)}</p>' if p.note else ""
    return (f'<details class="tier3" id="{esc(p.slug)}"><summary>{esc(p.title)}</summary>'
            f'{p.body}{note}<p class="reads">reads: <code>{esc(p.reads)}</code></p></details>')


def render(rec: Record, title: str) -> str:
    """The whole page as one HTML string; every number read from one row of the record."""
    gaps = Gaps()
    health = assess(rec)
    series = rung_series(rec)
    cells = hero.cells(rec, series, health, gaps)
    charts = tier2.panels(rec, series, gaps)
    details = tier3.panels(rec, health, series, gaps)
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{esc(title)}</h1>"
        '<p class="sub">Regenerated from an existing run record. No server, no producer, no live '
        "connection. A panel with no producer is a stated gap — absent is not zero.</p>"
        f'<section class="hero">{"".join(_cell(c) for c in cells)}</section>'
        f'{"".join(_panel(p) for p in charts)}'
        '<button id="expand" type="button" hidden>Expand all details</button>'
        f'<div class="tier3s">{"".join(_details(p) for p in details)}</div>'
        "<footer>mantis run dashboard — <code>tools/run_dashboard.py</code> (R333(d)). Every "
        "value on this page is read from one row of the record; the axes are steps, game "
        "ordinals or hours since the first event. Nothing is derived across differently "
        f"windowed producers. Stated gaps: {len(gaps.items)}.</footer>"
        f"<script>{_JS}</script></body></html>"
    )
