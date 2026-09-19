"""The viewer page: a light index embedded, shard data loaded on demand, one hex SVG board."""
from __future__ import annotations

import html as _html
import json
from typing import Any

from .reader import RunData

#: The self-play channel writes no per-position search stats (GAME-RECORD-1); the page states it.
SELFPLAY_STATS_GAP = ("no per-position search stats on this channel — GAME-RECORD-1 records "
                      "self-play moves only (CARD-SELFPLAY-SEARCH-STATS); the heatmap is a stated "
                      "gap here, never an empty board")


def data_path(run_id: str, shard_name: str) -> str:
    """The on-demand data file for one shard, relative to the page."""
    return f"data/{run_id}/{shard_name[:-len('.jsonl')]}.js"


def shard_js(run_id: str, shard_name: str, games: dict[str, Any]) -> str:
    """One shard's games as a script that hands them to the page."""
    payload = json.dumps(games, separators=(",", ":"))
    return f"window.MANTIS_SHARD({json.dumps(run_id)},{json.dumps(shard_name)},{payload});\n"


def render(runs: list[RunData], title: str) -> str:
    """The page over every run given: the index inline, shard data by relative path."""
    index = [row for run in runs for row in run.index]
    meta = {run.run_id: {"shards": [data_path(run.run_id, s.name) for s in run.shards],
                         "games": len(run.index), "findings": run.findings,
                         "skipped": run.skipped_lines} for run in runs}
    summary = " · ".join(f"{_html.escape(r.run_id)}: {len(r.index):,} games in {len(r.shards)} shards"
                         for r in runs)
    findings = [f for r in runs for f in r.findings]
    findings_html = ("" if not findings else
                     '<p class="finding">FINDINGS: ' + "; ".join(_html.escape(f) for f in findings)
                     + "</p>")
    return (_HEAD.replace("__TITLE__", _html.escape(title))
            + _BODY.replace("__TITLE__", _html.escape(title)).replace("__SUMMARY__", summary)
                   .replace("__FINDINGS__", findings_html)
            + "<script>window.MANTIS_INDEX=" + json.dumps(index, separators=(",", ":"))
            + ";window.MANTIS_RUNS=" + json.dumps(meta, separators=(",", ":"))
            + ";window.MANTIS_STATS_GAP=" + json.dumps(SELFPLAY_STATS_GAP) + ";</script>\n"
            + "<script>" + _SCRIPT + "</script>\n</body></html>\n")


_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f6f2;--fg:#1d1d1b;--mute:#6b6a66;--line:#d9d6cf;--p1:#1d1d1b;--p2:#f4f1ea;--hi:#d94f2b;--win:#2b8a3e;--heat:#3b6fd6}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe6;--mute:#9a988f;--line:#3a3935;--p1:#0c0c0b;--p2:#efece4}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line)}header h1{font-size:16px;margin:0 0 2px}
header p{margin:0;color:var(--mute);font-size:12px}
.finding{color:var(--hi);font-size:12px;margin:4px 0 0}
main{display:grid;grid-template-columns:minmax(260px,360px) 1fr;min-height:calc(100vh - 56px)}
@media (max-width:760px){main{grid-template-columns:1fr}#list{max-height:38vh}}
#side{border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
#filters{padding:8px 12px;display:flex;flex-wrap:wrap;gap:6px;border-bottom:1px solid var(--line)}
#filters select,#filters input{font:12px system-ui,sans-serif;padding:3px 4px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px;max-width:120px}
#count{padding:4px 12px;color:var(--mute);font-size:12px}
#list{overflow:auto;flex:1}#list table{border-collapse:collapse;width:100%;font-size:12px}
#list td,#list th{padding:3px 6px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:left}
#list tr{cursor:pointer}#list tr.sel{background:rgba(59,111,214,.18)}#list tr:hover{background:rgba(59,111,214,.08)}
#more{margin:8px 12px;font:12px system-ui,sans-serif}
#view{padding:12px 16px;display:flex;flex-direction:column;gap:8px;min-width:0}
#info{font-size:13px;color:var(--mute)}#info b{color:var(--fg)}#info .gap{color:var(--hi);font-size:12px}
#board{width:100%;max-width:820px;aspect-ratio:1/1;max-height:78vh;border:1px solid var(--line);border-radius:6px;background:var(--bg);touch-action:pan-y}
#ctl{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
#ctl button{font:14px system-ui,sans-serif;padding:4px 10px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg);cursor:pointer}
#ctl .ply{font-variant-numeric:tabular-nums;min-width:90px}
#help{font-size:12px;color:var(--mute)}
.cell{fill:none;stroke:var(--line);stroke-width:.06}
.stone.p1{fill:var(--p1)}.stone.p2{fill:var(--p2);stroke:var(--p1);stroke-width:.05}
.stone.fast{stroke:var(--hi);stroke-width:.07;stroke-dasharray:.18 .12}
.num.p1{fill:var(--p2)}.num.p2{fill:var(--p1)}.num{font:.42px system-ui,sans-serif;text-anchor:middle;dominant-baseline:central;pointer-events:none}
.last{fill:none;stroke:var(--hi);stroke-width:.09}.wincell{fill:none;stroke:var(--win);stroke-width:.12}
.heat{stroke:none}.heatnum{font:.34px system-ui,sans-serif;text-anchor:middle;dominant-baseline:central;fill:var(--fg);pointer-events:none}
</style></head>
"""

_BODY = """<body>
<header><h1>__TITLE__</h1><p>__SUMMARY__ — GAME-RECORD-1 shards, one row per game; ← → step, Home/End, space plays, n/p next/previous game, h heatmap, c copies the position to the current ply (for the analyzer); swipe the board on a phone.</p>__FINDINGS__</header>
<main>
<section id="side">
<div id="filters">
<select id="f-run"><option value="">run: all</option></select>
<select id="f-ch"><option value="">channel: all</option></select>
<select id="f-res"><option value="">result: all</option><option>p1</option><option>p2</option><option>draw</option><option>unknown</option></select>
<select id="f-term"><option value="">termination: all</option></select>
<input id="f-min" type="number" placeholder="plies ≥" min="0">
<input id="f-max" type="number" placeholder="plies ≤" min="0">
<select id="f-sort"><option value="order">order: shard</option><option value="step">order: step</option><option value="pl">order: plies ↓</option></select>
</div>
<div id="count"></div>
<div id="list"><table><thead><tr><th>#</th><th>run</th><th>ch</th><th>res</th><th>plies</th><th>term</th><th>step</th></tr></thead><tbody id="rows"></tbody></table>
<button id="more" hidden>show 300 more</button></div>
</section>
<section id="view">
<div id="info">Pick a game from the list.</div>
<svg id="board" viewBox="0 0 10 10"></svg>
<div id="ctl"><button id="b-first" title="Home">⏮</button><button id="b-prev" title="←">◀</button><button id="b-play" title="space">▶ play</button><button id="b-next" title="→">▶</button><button id="b-last" title="End">⏭</button><span class="ply" id="ply"></span><button id="b-heat" title="h" hidden>heatmap</button><span id="heatinfo"></span></div>
<div id="help">Stones carry their ply number (0 = the opening single; then two per turn). The last two stones are ringed; the six-in-a-row is outlined at the final ply. A dashed stone was a FAST-arm search (the quick budget) — three self-play moves in four are; the ply readout names the arm and sims of the stone just placed, or says the record carries no arm. A position's link is its address (<code>?g=run/game_id&amp;ply=N&amp;heat=1</code>).</div>
</section>
</main>
"""

_SCRIPT = r"""
(function(){
const IDX=window.MANTIS_INDEX,RUNS=window.MANTIS_RUNS,SH={},PENDING={};
window.MANTIS_SHARD=function(run,shard,games){SH[run+'/'+shard]=games;const k=run+'/'+shard;(PENDING[k]||[]).forEach(f=>f(games));delete PENDING[k];};
const $=id=>document.getElementById(id);
const owner=p=>(((p+1)/2|0)%2);
const AX=[[1,0],[0,1],[1,-1]];
let view=[],shown=0,cur=null,curRow=null,ply=0,timer=null,heat=false,atPly=null,atHeat=false;
function opts(sel,vals){vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o);});}
opts($('f-run'),Object.keys(RUNS));opts($('f-ch'),[...new Set(IDX.map(r=>r.ch))].filter(Boolean).sort());opts($('f-term'),[...new Set(IDX.map(r=>r.term))].filter(Boolean).sort());
function applyFilters(){const run=$('f-run').value,ch=$('f-ch').value,res=$('f-res').value,term=$('f-term').value,mn=+$('f-min').value||0,mx=+$('f-max').value||Infinity,sort=$('f-sort').value;
view=IDX.filter(r=>(!run||r.run===run)&&(!ch||r.ch===ch)&&(!res||r.res===res)&&(!term||r.term===term)&&r.pl>=mn&&r.pl<=mx);
if(sort==='step')view.sort((a,b)=>(a.step??-1)-(b.step??-1));else if(sort==='pl')view.sort((a,b)=>b.pl-a.pl);
shown=0;$('rows').innerHTML='';$('count').textContent=view.length.toLocaleString()+' of '+IDX.length.toLocaleString()+' games';renderMore();}
function renderMore(){const tb=$('rows'),end=Math.min(view.length,shown+300);for(let i=shown;i<end;i++){const r=view[i],tr=document.createElement('tr');tr.dataset.i=i;tr.innerHTML='<td>'+(i+1)+'</td><td>'+r.run+'</td><td>'+r.ch+'</td><td>'+r.res+'</td><td>'+r.pl+'</td><td>'+r.term+'</td><td>'+(r.step??'')+'</td>';tr.onclick=()=>open(i);tb.appendChild(tr);}
shown=end;$('more').hidden=shown>=view.length;}
document.querySelectorAll('#filters select,#filters input').forEach(e=>e.addEventListener('input',applyFilters));$('more').onclick=renderMore;
function loadShard(run,i,cb){const path=RUNS[run].shards[i],name=path.split('/').pop().replace(/\.js$/,'.jsonl'),k=run+'/'+name;if(SH[k])return cb(SH[k]);if(PENDING[k]){PENDING[k].push(cb);return;}PENDING[k]=[cb];const s=document.createElement('script');s.src=path;s.onerror=()=>{$('info').textContent='could not load '+path+' (serve the viewer directory or open it from disk beside data/)';};document.head.appendChild(s);}
function open(i){curRow=view[i];document.querySelectorAll('#rows tr.sel').forEach(t=>t.classList.remove('sel'));const tr=document.querySelector('#rows tr[data-i="'+i+'"]');if(tr){tr.classList.add('sel');tr.scrollIntoView({block:'nearest'});}
$('info').textContent='loading '+curRow.id+'…';loadShard(curRow.run,curRow.shard,g=>{cur=g[curRow.id];if(!cur){$('info').textContent='game '+curRow.id+' is not in its shard file';return;}ply=(atPly==null)?cur.m.length:Math.max(0,Math.min(cur.m.length,atPly));heat=!!atHeat&&!!cur.s;atPly=null;atHeat=false;$('b-heat').hidden=!cur.s;draw();});}
function link(){if(!curRow)return;const u=new URL(location.href);u.searchParams.set('g',curRow.run+'/'+curRow.id);u.searchParams.set('ply',ply);if(heat)u.searchParams.set('heat','1');else u.searchParams.delete('heat');history.replaceState(null,'',u);}
function stepTo(k){if(!cur)return;ply=Math.max(0,Math.min(cur.m.length,k));draw();}
function hexPts(cx,cy){let p=[];for(let i=0;i<6;i++){const a=Math.PI/180*(60*i-30);p.push((cx+Math.cos(a)).toFixed(3)+','+(cy+Math.sin(a)).toFixed(3));}return p.join(' ');}
const X=(q,r)=>Math.sqrt(3)*(q+r/2),Y=(q,r)=>1.5*r;
function draw(){const m=cur.m,n=m.length,st=m.slice(0,ply);const r=curRow;
$('info').innerHTML='<b>'+r.run+'</b> · '+r.ch+(r.rung?' · '+r.rung:'')+(r.phase?' · '+r.phase:'')+' · result <b>'+r.res+'</b> · '+r.term+' · '+n+' plies · step '+(r.step??'—')+(r.kind?' ('+r.kind+')':'')+(r.w!=null?' · worker '+r.w:'')+' · <span style="font-size:11px">'+r.id+'</span>'+(cur.s?'':'<br><span class="gap">'+window.MANTIS_STATS_GAP+'</span>');
$('ply').textContent='ply '+ply+' / '+n+(ply>0?' · stone '+(ply-1)+': '+armLabel(ply-1):'');
let cells=m.slice();let hs=null;if(heat&&cur.s){hs=cur.s.find(e=>e.ply===ply)||null;if(hs)hs.visits.forEach(v=>cells.push([v[0],v[1]]));}
if(!cells.length)cells=[[0,0]];
let xs=cells.map(c=>X(c[0],c[1])),ys=cells.map(c=>Y(c[0],c[1]));const pad=2;const x0=Math.min(...xs)-pad,x1=Math.max(...xs)+pad,y0=Math.min(...ys)-pad,y1=Math.max(...ys)+pad;const w=x1-x0,h=y1-y0,S=Math.max(w,h);
const svg=$('board');svg.setAttribute('viewBox',(x0-(S-w)/2)+' '+(y0-(S-h)/2)+' '+S+' '+S);
let out=[];const qmin=Math.floor(Math.min(...cells.map(c=>c[0])))-2,qmax=Math.ceil(Math.max(...cells.map(c=>c[0])))+2,rmin=Math.floor(Math.min(...cells.map(c=>c[1])))-2,rmax=Math.ceil(Math.max(...cells.map(c=>c[1])))+2;
for(let q=qmin;q<=qmax;q++)for(let rr=rmin;rr<=rmax;rr++){const x=X(q,rr),y=Y(q,rr);if(x<x0-1||x>x1+1||y<y0-1||y>y1+1)continue;out.push('<polygon class="cell" points="'+hexPts(x,y)+'"/>');}
if(hs){const mx=Math.max(...hs.visits.map(v=>v[2]),1);hs.visits.forEach(v=>{const x=X(v[0],v[1]),y=Y(v[0],v[1]),a=(0.12+0.78*v[2]/mx).toFixed(2);out.push('<polygon class="heat" style="fill:var(--heat);fill-opacity:'+a+'" points="'+hexPts(x,y)+'"/><text class="heatnum" x="'+x.toFixed(3)+'" y="'+y.toFixed(3)+'">'+v[2]+'</text>');});
$('heatinfo').textContent='root before ply '+ply+' by '+hs.by+': value '+(+hs.root_value).toFixed(3)+', '+hs.visits.length+' visited children';}else{$('heatinfo').textContent=heat&&cur.s?'no root recorded before ply '+ply:'';}
if(ply===n&&cur.win){cur.win.forEach(c=>{out.push('<polygon class="wincell" points="'+hexPts(X(c[0],c[1]),Y(c[0],c[1]))+'"/>');});}
st.forEach((c,i)=>{const x=X(c[0],c[1]),y=Y(c[0],c[1]),o=owner(i)?'p2':'p1',fast=(cur.a&&cur.a[i]==='q')?' fast':'';out.push('<circle class="stone '+o+fast+'" cx="'+x.toFixed(3)+'" cy="'+y.toFixed(3)+'" r="0.78"/><text class="num '+o+'" x="'+x.toFixed(3)+'" y="'+y.toFixed(3)+'">'+i+'</text>');});
for(let i=Math.max(0,ply-2);i<ply;i++){const c=m[i];out.push('<circle class="last" cx="'+X(c[0],c[1]).toFixed(3)+'" cy="'+Y(c[0],c[1]).toFixed(3)+'" r="0.9"/>');}
svg.innerHTML=out.join('');link();}
function armLabel(i){if(cur.a){const c=cur.a[i];if(c==='o')return 'opening (random ply, no search)';if(c==='f')return 'full @ '+(cur.sims.f??'?')+' sims';if(c==='q')return 'fast @ '+(cur.sims.q??'?')+' sims';}
if(curRow.cand!=null&&curRow.sims!=null){if((owner(i)===0)===(curRow.cand===1))return 'candidate @ '+curRow.sims+' sims (deploy head)';return 'opponent'+(curRow.rung?' '+curRow.rung:(curRow.ch==='promotion'?' (anchor @ '+curRow.sims+' sims)':''));}
return 'arm not recorded (a record from before R353(d))';}
function play(){if(timer){clearInterval(timer);timer=null;$('b-play').textContent='▶ play';return;}$('b-play').textContent='⏸ pause';timer=setInterval(()=>{if(!cur||ply>=cur.m.length){play();return;}stepTo(ply+1);},350);}
function nav(d){if(!curRow)return;const i=view.indexOf(curRow)+d;if(i>=0&&i<view.length){while(shown<=i)renderMore();open(i);}}
$('b-first').onclick=()=>stepTo(0);$('b-prev').onclick=()=>stepTo(ply-1);$('b-next').onclick=()=>stepTo(ply+1);$('b-last').onclick=()=>stepTo(cur?cur.m.length:0);$('b-play').onclick=play;$('b-heat').onclick=()=>{heat=!heat;draw();};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;const k=e.key;
if(k==='ArrowLeft')stepTo(ply-1);else if(k==='ArrowRight')stepTo(ply+1);else if(k==='Home')stepTo(0);else if(k==='End')stepTo(cur?cur.m.length:0);else if(k===' '){e.preventDefault();play();}else if(k==='n')nav(1);else if(k==='p')nav(-1);else if(k==='h'&&cur&&cur.s){heat=!heat;draw();}else if(k==='c'&&cur){navigator.clipboard.writeText(cur.m.slice(0,ply).map(c=>c[0]+','+c[1]).join(';'));}else return;e.preventDefault();});
let tx=null,ty=null;const b=$('board');b.addEventListener('touchstart',e=>{tx=e.touches[0].clientX;ty=e.touches[0].clientY;},{passive:true});
b.addEventListener('touchend',e=>{if(tx==null)return;const dx=e.changedTouches[0].clientX-tx,dy=e.changedTouches[0].clientY-ty;tx=null;if(Math.abs(dx)>40&&Math.abs(dx)>Math.abs(dy))stepTo(ply+(dx<0?1:-1));});
applyFilters();
const qs=new URL(location.href).searchParams,want=qs.get('g')||'';const at=want?view.findIndex(r=>r.run+'/'+r.id===want):-1;
if(qs.get('ply')!=null)atPly=+qs.get('ply');atHeat=qs.get('heat')==='1';
if(at>=0){while(shown<=at)renderMore();open(at);}else if(view.length)open(0);
})();
"""
