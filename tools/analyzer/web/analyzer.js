(function(){
'use strict';
const $=id=>document.getElementById(id);
// The position is shared; everything an engine says lives on its slot.
const S={moves:[],ply:0,pos:null,engines:[],slots:[],active:0,sims:0,simsChosen:false,ov:'h',tac:false,auto:true,seq:0,
  client:Math.random().toString(36).slice(2),timer:null,tick:null};
const SQ3=Math.sqrt(3),X=(q,r)=>SQ3*(q+r/2),Y=(q,r)=>1.5*r;
function pixelToHex(x,y){const qf=SQ3/3*x-y/3,rf=2/3*y,sf=-qf-rf;let q=Math.round(qf),r=Math.round(rf),s=Math.round(sf);
  const dq=Math.abs(q-qf),dr=Math.abs(r-rf),ds=Math.abs(s-sf);if(dq>dr&&dq>ds)q=-r-s;else if(dr>ds)r=-q-s;return [q,r];}
const owner=p=>(((p+1)/2|0)%2);
const fmt=m=>m.map(c=>c[0]+','+c[1]).join(';');
const key=c=>c[0]+','+c[1];
const f3=v=>(v>=0?'+':'')+v.toFixed(3);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function hexPts(cx,cy){const p=[];for(let i=0;i<6;i++){const a=Math.PI/180*(60*i-30);p.push((cx+Math.cos(a)).toFixed(3)+','+(cy+Math.sin(a)).toFixed(3));}return p.join(' ');}
const cur=()=>S.moves.slice(0,S.ply);
const posCurrent=()=>S.pos&&fmt(S.pos.moves)===fmt(cur());
const activeSlot=()=>S.slots[S.active];
const argmaxOf=sl=>(sl.search&&!sl.search.absent&&sl.search.argmax)||(sl.raw&&!sl.raw.absent&&sl.raw.argmax)||null;

// ---- slots ------------------------------------------------------------------------
function addSlot(engine){
  const el=$('cardtpl').content.firstElementChild.cloneNode(true);
  const sl={engine:engine||null,card:null,raw:null,search:null,tactics:null,sym:null,trace:null,searching:null,el:el};
  const sel=el.querySelector('.cardengine');
  for(const r of S.engines){const o=document.createElement('option');o.value=(r.kind==='mantis'||r.kind==='strix')?r.id:'';o.disabled=!o.value;
    o.textContent=r.kind==='mantis'?`${r.run_id} · ${r.step.toLocaleString()} · ${r.sha8}`:r.kind==='strix'?`strix${r.note?' — '+r.note:''}`:`${r.id} — ${r.note}`;sel.appendChild(o);}
  sel.value=sl.engine||'';
  sel.addEventListener('change',e=>{sl.engine=e.target.value||null;sl.card=null;setDeployChip();requestSlot(sl);});
  el.querySelector('.cardhead').addEventListener('click',e=>{if(e.target.tagName==='SELECT'||e.target.tagName==='BUTTON')return;S.active=S.slots.indexOf(sl);drawAll();});
  el.querySelector('.drop').onclick=()=>{if(S.slots.length===1)return;S.slots.splice(S.slots.indexOf(sl),1);el.remove();S.active=Math.min(S.active,S.slots.length-1);link();drawAll();};
  S.slots.push(sl);$('cards').appendChild(el);
  return sl;
}
function status(sl,t,err){const e=sl.el.querySelector('.status');e.textContent=t;e.className='status'+(err?' err':'');}

// ---- requests ---------------------------------------------------------------------
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return {status:r.status,body:await r.json()};}
async function analyze(sl,sims,symmetry){
  if(!sl.engine)return;
  const seq=++S.seq,moves=fmt(cur());
  if(sims>0){sl.searching={sims:sims,t0:performance.now(),seq:seq};tickStart();}
  let out;try{out=await post('/analyze',{engine:sl.engine,moves:moves,sims:sims,symmetry:!!symmetry,client:S.client,seq:seq});}
  catch(e){status(sl,'network: '+e,true);return;}
  const b=out.body;
  if(b.superseded)return;
  if(sl.searching&&sl.searching.seq===seq){sl.searching=null;if(!S.slots.some(s=>s.searching))tickStop();}
  if(!b.ok){
    if(out.status===503||out.status===404)sl.card=null;
    if(out.status===400&&moves===fmt(cur())&&S.ply>0&&b.refused.includes('at ply '+(S.ply-1))){S.moves=cur();S.moves.pop();S.ply=S.moves.length;status(sl,b.refused+' — undone',true);edited();return;}
    status(sl,out.status+' '+b.refused,true);drawCard(sl);return;}
  const rec=b.record;if(fmt(rec.position.moves)!==fmt(cur()))return;
  sl.card=rec.engine;S.pos=rec.position;sl.tactics=rec.tactics;sl.raw=rec.raw;
  if(sims>0||(rec.search.absent&&!rec.search.absent.startsWith('sims=0')))sl.search=rec.search;
  if(rec.symmetry&&!rec.symmetry.absent)sl.sym=rec.symmetry;
  status(sl,'');setDeployChip();drawAll();
}
function requestSlot(sl){sl.raw=null;sl.search=null;sl.tactics=null;sl.sym=null;sl.trace=null;sl.searching=null;
  analyze(sl,0,false);drawCard(sl);}
function tickStart(){if(!S.tick)S.tick=setInterval(()=>S.slots.forEach(drawReadout),100);}
function tickStop(){if(S.tick){clearInterval(S.tick);S.tick=null;}}
function edited(){tickStop();link();S.slots.forEach(requestSlot);
  clearTimeout(S.timer);if(S.auto&&S.sims>0)S.timer=setTimeout(()=>S.slots.forEach(sl=>analyze(sl,S.sims,false)),300);drawAll();}
function setDeployChip(){const card=activeSlot().card,b=$('budget').querySelector('[data-sims="deploy"]');
  if(card){b.textContent='deploy '+card.deploy_sims;b.disabled=false;
    if(!S.simsChosen&&S.sims===0){S.sims=card.deploy_sims;$('sims').value=S.sims;link();if(S.auto)S.slots.forEach(sl=>analyze(sl,S.sims,false));}}
  else{b.textContent='deploy';b.disabled=true;}
  $('budget').querySelectorAll('button').forEach(x=>x.classList.toggle('on',(x.dataset.sims==='deploy'?(card&&card.deploy_sims):+x.dataset.sims)===S.sims));}

// ---- drawing ----------------------------------------------------------------------
function drawAll(){drawBoard();S.slots.forEach(drawCard);drawHeader();}
function drawCard(sl){sl.el.classList.toggle('active',sl===activeSlot());sl.el.querySelector('.slotno').textContent=String(S.slots.indexOf(sl));
  drawReadout(sl);drawChildren(sl);drawTactics(sl);drawPanels(sl);
  const c=sl.card;sl.el.querySelector('.cardmeta').textContent=c?`${c.search_kind} · ${c.device} · ${c.threads} threads · radius ${c.radius}`:'';}
function drawHeader(){const c=activeSlot().card;
  $('hdr').textContent=c?`${c.run_id} · ${c.step.toLocaleString()} · ${c.sha8||''} · ${c.search_kind} · ${c.device} · ${c.threads} threads · radius ${c.radius}`:'no engine loaded';
  const p=posCurrent()?S.pos:null;
  $('pos').textContent=p?`ply ${p.ply} · ${p.winner?p.winner+' won':p.to_move+' to move · '+p.moves_remaining+' left'} · ${p.legal} legal`:`ply ${S.ply}`;
  if(document.activeElement!==$('moves'))$('moves').value=fmt(cur());}
function overlayCells(sl){
  if(S.ov==='p'&&sl.raw&&!sl.raw.absent)return {rows:sl.raw.policy.map(r=>({c:[r[0],r[1]],v:r[2]})),max:Math.max(...sl.raw.policy.map(r=>r[2]),1e-9),floor:0.02,label:v=>(v*100).toFixed(0)};
  if(sl.search&&!sl.search.absent){const ch=sl.search.children;
    if(S.ov==='h')return {rows:ch.map(r=>({c:[r[0],r[1]],v:r[3]})),max:Math.max(...ch.map(r=>r[3]),1),floor:2,label:v=>String(v)};
    if(S.ov==='q')return {rows:ch.filter(r=>r[3]>=2).map(r=>({c:[r[0],r[1]],v:r[4]})),max:1,floor:-2,label:v=>f3(v),div:true};}
  return null;}
function drawBoard(){
  const sl=activeSlot(),m=S.moves,st=cur(),win=(posCurrent()&&S.pos.legal_window)||[],cells=[...m,...win];if(!cells.length)cells.push([0,0]);
  const xs=cells.map(c=>X(c[0],c[1])),ys=cells.map(c=>Y(c[0],c[1])),pad=2;
  const x0=Math.min(...xs)-pad,x1=Math.max(...xs)+pad,y0=Math.min(...ys)-pad,y1=Math.max(...ys)+pad,w=x1-x0,h=y1-y0,Sz=Math.max(w,h);
  const svg=$('board');svg.setAttribute('viewBox',(x0-(Sz-w)/2)+' '+(y0-(Sz-h)/2)+' '+Sz+' '+Sz);
  const out=[],winSet=new Set(win.map(key)),played=new Set(st.map(key));
  const qmin=Math.floor(Math.min(...cells.map(c=>c[0])))-2,qmax=Math.ceil(Math.max(...cells.map(c=>c[0])))+2,rmin=Math.floor(Math.min(...cells.map(c=>c[1])))-2,rmax=Math.ceil(Math.max(...cells.map(c=>c[1])))+2;
  for(let q=qmin;q<=qmax;q++)for(let r=rmin;r<=rmax;r++){const x=X(q,r),y=Y(q,r);if(x<x0-1||x>x1+1||y<y0-1||y>y1+1)continue;
    out.push(`<polygon class="cell${winSet.has(key([q,r]))?' win hot':''}" points="${hexPts(x,y)}"/>`);}
  const ov=overlayCells(sl);
  if(ov){for(const row of ov.rows){if(played.has(key(row.c)))continue;const x=X(row.c[0],row.c[1]),y=Y(row.c[0],row.c[1]);
    if(ov.div){out.push(`<polygon class="heat" style="fill:var(${row.v>=0?'--pos':'--neg'});fill-opacity:${Math.min(1,Math.abs(row.v)).toFixed(2)}" points="${hexPts(x,y)}"/>`);}
    else{if(row.v<ov.floor&&row.v!==ov.max)continue;out.push(`<polygon class="heat" style="fill:var(--heat);fill-opacity:${(0.25+0.55*row.v/ov.max).toFixed(2)}" points="${hexPts(x,y)}"/>`);}
    out.push(`<text class="heatnum" x="${x.toFixed(3)}" y="${y.toFixed(3)}">${ov.label(row.v)}</text>`);}}
  if(S.tac&&sl.tactics&&sl.tactics.cells){for(const c of sl.tactics.cells)out.push(`<polygon class="tacwin" points="${hexPts(X(c[0],c[1]),Y(c[0],c[1]))}"/>`);
    for(const four of (sl.tactics.opp_fours||[]))for(const c of four)out.push(`<polygon class="tacfour" points="${hexPts(X(c[0],c[1]),Y(c[0],c[1]))}"/>`);}
  if(posCurrent()&&S.pos.win_line)for(const c of S.pos.win_line)out.push(`<polygon class="wincell" points="${hexPts(X(c[0],c[1]),Y(c[0],c[1]))}"/>`);
  st.forEach((c,i)=>{const x=X(c[0],c[1]),y=Y(c[0],c[1]),o=owner(i)?'p2':'p1';out.push(`<circle class="stone ${o}" cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="0.78"/><text class="num ${o}" x="${x.toFixed(3)}" y="${y.toFixed(3)}">${i}</text>`);});
  for(let i=Math.max(0,S.ply-2);i<S.ply;i++){const c=m[i];out.push(`<circle class="last" cx="${X(c[0],c[1]).toFixed(3)}" cy="${Y(c[0],c[1]).toFixed(3)}" r="0.9"/>`);}
  S.slots.forEach((o,i)=>{const a=argmaxOf(o);if(!a)return;const x=X(a[0],a[1]).toFixed(3),y=Y(a[0],a[1]).toFixed(3);
    if(o===sl)out.push(`<circle class="argmax" cx="${x}" cy="${y}" r="0.62"/>`);else out.push(`<text class="mark" x="${x}" y="${(+y+0.55).toFixed(3)}">${i}</text>`);});
  svg.innerHTML=out.join('');}
function drawReadout(sl){
  const raw=sl.el.querySelector('.r-raw').children,head=sl.el.querySelector('.r-head').children;
  if(sl.raw&&!sl.raw.absent){raw[1].textContent=f3(sl.raw.value);raw[2].textContent='argmax ('+sl.raw.argmax+')';raw[3].textContent=sl.raw.ms+' ms';}
  else{raw[1].textContent='—';raw[2].textContent=sl.raw&&sl.raw.absent?sl.raw.absent:(sl.engine?'reading…':'no engine');raw[3].textContent='';}
  if(sl.searching){head[1].textContent='…';head[2].textContent='searching · '+sl.searching.sims;head[3].textContent=((performance.now()-sl.searching.t0)/1000).toFixed(1)+' s';}
  else if(sl.search&&!sl.search.absent){head[1].textContent=f3(sl.search.root_value);head[2].textContent='argmax ('+sl.search.argmax+')';head[3].textContent=`${sl.search.sims} · ${sl.search.root_visits} visits · ${(sl.search.ms/1000).toFixed(1)} s · q-fires ${sl.search.quiescence_fires}`;}
  else{head[1].textContent='—';head[2].textContent=sl.search&&sl.search.absent?sl.search.absent:(S.sims>0?'not searched yet':'sims 0');head[3].textContent='';}}
function drawChildren(sl){
  const tb=sl.el.querySelector('.children tbody');tb.innerHTML='';let rows=[],arg=null;
  if(sl.search&&!sl.search.absent){rows=sl.search.children.map(r=>({c:[r[0],r[1]],p:r[2],n:r[3],q:r[4]}));arg=key(sl.search.argmax);}
  else if(sl.raw&&!sl.raw.absent){rows=sl.raw.policy.map(r=>({c:[r[0],r[1]],p:r[2],n:null,q:null}));arg=key(sl.raw.argmax);}
  for(const r of rows.slice(0,12)){const tr=document.createElement('tr');if(key(r.c)===arg)tr.className='arg';
    tr.innerHTML=`<td>(${r.c})</td><td>${(r.p*100).toFixed(1)} %</td><td>${r.n==null?'—':r.n}</td><td>${r.q==null?'—':f3(r.q)}</td>`;
    tr.onclick=()=>{$('hover').textContent=`(${r.c}) · p ${(r.p*100).toFixed(1)} %`+(r.n==null?'':` · n ${r.n} · Q ${f3(r.q)}`);};tb.appendChild(tr);}
  sl.el.querySelector('.more').textContent=rows.length>12?`… ${rows.length-12} more (board floor: visits ≥ 2, prior ≥ 2 %)`:'';}
function drawTactics(sl){
  const t=sl.tactics,e=sl.el.querySelector('.tac');if(!t){e.textContent='—';return;}
  if(t.class==='terminal'){e.innerHTML=`<span class="cls">game over</span> · ${esc(t.winner)} wins`;return;}
  const v=t.verdict,miss=s=>s&&s.includes('MISSES')?'miss':'';
  e.innerHTML=`<span class="cls">${esc(t.class.toUpperCase())}</span>${t.cells.length?' '+esc(JSON.stringify(t.cells)):''} · k=${+t.k}`
    +(v.raw?` · net: <span class="${miss(v.raw)}">${esc(v.raw)}</span>`:'')+(v.search?` · head: <span class="${miss(v.search)}">${esc(v.search)}</span>`:'')
    +(t.forced_win_move?` · forced_win_move(2) → (${esc(t.forced_win_move)})`:'')+`<div class="mute">${esc(t.derivation)}</div>`;}
function drawPanels(sl){
  const s=sl.el.querySelector('.sym');
  if(sl.sym){const y=sl.sym;s.innerHTML=`<b>symmetry</b> n=${+y.n} about (${esc(y.centre)}) · value ${f3(y.value_min)} … ${f3(y.value_max)} · spread ${y.spread.toFixed(4)} · argmax agrees ${esc(y.argmax_agreement)} · worst ${esc(y.worst.map)} ${f3(y.worst.value)} argmax (${esc(y.worst.argmax)})`
      +(y.translation.absent?`<div class="mute">translation: ${esc(y.translation.absent)}</div>`:` · translation ${f3(y.translation.value)}`);}
  else s.innerHTML=`<b>symmetry</b> <span class="mute">(s): not requested — 13 net reads, about ${sl.raw&&sl.raw.ms?((13*sl.raw.ms)/1000).toFixed(1)+' s':'a second'}</span>`;
  const t=sl.el.querySelector('.trace'),svg=t.querySelector('.tracesvg'),span=t.querySelector('span');
  if(sl.trace){const rows=sl.trace,n=rows.length;svg.hidden=false;svg.setAttribute('viewBox',`0 0 ${Math.max(n-1,1)} 100`);svg.setAttribute('preserveAspectRatio','none');
    const path=k=>{let d='',pen=false;rows.forEach((r,i)=>{if(r[k]==null){pen=false;return;}d+=(pen?'L':'M')+i+' '+(50-50*r[k]).toFixed(2)+' ';pen=true;});return d;};
    svg.innerHTML=`<line class="zero" x1="0" y1="50" x2="${n}" y2="50"/><path class="raw" d="${path('raw')}"/><path class="root" d="${path('root')}"/><line class="cur" x1="${S.ply}" y1="0" x2="${S.ply}" y2="100"/>`;
    span.textContent=` P1's perspective (the readout is the mover's) · thin: net · thick: head${rows.some(r=>r.root!=null)?'':' (not run)'} · gaps are unsearched or terminal plies`;}
  else{svg.hidden=true;span.textContent=`(v): not requested — ${S.moves.length+1} net reads`+(S.sims>0?` + ${S.moves.length+1} searches at ${S.sims}`+(sl.search&&!sl.search.absent?` (about ${((S.moves.length+1)*sl.search.ms/1000).toFixed(0)} s)`:''):'');}}

// ---- editing --------------------------------------------------------------------------
function place(q,r){if(posCurrent()&&S.pos.winner)return;S.moves=cur();S.moves.push([q,r]);S.ply=S.moves.length;edited();}
function stepTo(k){S.ply=Math.max(0,Math.min(S.moves.length,k));edited();}
function undo(){if(S.ply===0)return;S.moves=cur();S.moves.pop();S.ply=S.moves.length;edited();}
function parseText(t){const s=t.trim();const parts=!s?[]:s.startsWith('[')?JSON.parse(s):s.split(';').map(x=>x.split(',').map(Number));
  if(!parts.every(c=>Array.isArray(c)&&c.length===2&&c.every(Number.isInteger)))throw new Error('not q,r;q,r');return parts;}
function importText(t){try{S.moves=parseText(t);S.ply=S.moves.length;edited();}catch(e){status(activeSlot(),'import: '+e.message,true);}}
function link(){const u=new URL(location.href);u.searchParams.set('m',fmt(S.moves));u.searchParams.set('ply',S.ply);
  const es=S.slots.map(s=>s.engine).filter(Boolean);if(es.length)u.searchParams.set('e',es.join(','));else u.searchParams.delete('e');
  u.searchParams.set('sims',S.sims);u.searchParams.set('ov',S.ov);if(S.tac)u.searchParams.set('tac','1');else u.searchParams.delete('tac');history.replaceState(null,'',u);}
async function runSymmetry(){const sl=activeSlot();if(!sl.engine||(posCurrent()&&S.pos.winner))return;status(sl,'symmetry sweep…');await analyze(sl,0,true);}
async function runTrace(){const sl=activeSlot();if(!sl.engine)return;const seq=++S.seq;status(sl,'trace…');
  let out;try{out=await post('/trace',{engine:sl.engine,moves:fmt(S.moves),sims:S.sims,client:S.client,seq:seq});}catch(e){status(sl,'network: '+e,true);return;}
  if(out.body.superseded)return;if(!out.body.ok){status(sl,out.status+' '+out.body.refused,true);return;}sl.trace=out.body.trace;status(sl,'');drawPanels(sl);}

// ---- wiring ---------------------------------------------------------------------------
const board=$('board');
function cellAt(e){const pt=board.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const p=pt.matrixTransform(board.getScreenCTM().inverse());return pixelToHex(p.x,p.y);}
board.addEventListener('click',e=>{const [q,r]=cellAt(e);
  if(posCurrent()&&!S.pos.legal_window.some(c=>c[0]===q&&c[1]===r)){status(activeSlot(),`(${q},${r}) is outside the legal window`,true);return;}place(q,r);});
board.addEventListener('mousemove',e=>{const [q,r]=cellAt(e),sl=activeSlot();let t=`(${q},${r})`;
  const pr=sl.raw&&!sl.raw.absent&&sl.raw.policy.find(c=>c[0]===q&&c[1]===r),ch=sl.search&&!sl.search.absent&&sl.search.children.find(c=>c[0]===q&&c[1]===r);
  if(pr)t+=` · p ${(pr[2]*100).toFixed(1)} %`;if(ch)t+=` · n ${ch[3]} · Q ${f3(ch[4])}`;$('hover').textContent=t;});
$('budget').querySelectorAll('button').forEach(b=>b.onclick=()=>{const card=activeSlot().card;S.sims=b.dataset.sims==='deploy'?(card?card.deploy_sims:0):+b.dataset.sims;S.simsChosen=true;$('sims').value=S.sims;setDeployChip();link();if(S.auto)S.slots.forEach(sl=>analyze(sl,S.sims,false));});
$('sims').addEventListener('change',e=>{S.sims=Math.max(0,+e.target.value|0);S.simsChosen=true;setDeployChip();link();});
$('auto').addEventListener('change',e=>{S.auto=e.target.checked;});
$('go').onclick=()=>S.slots.forEach(sl=>analyze(sl,S.sims,false));
$('addslot').onclick=()=>{const sl=addSlot(activeSlot().engine);link();requestSlot(sl);if(S.auto&&S.sims>0)analyze(sl,S.sims,false);drawAll();};
$('moves').addEventListener('change',e=>importText(e.target.value));
$('copy').onclick=()=>navigator.clipboard.writeText(fmt(cur()));
document.addEventListener('keydown',e=>{const tag=e.target.tagName;if(tag==='INPUT'||tag==='SELECT'||tag==='TEXTAREA')return;const k=e.key;
  if(k==='ArrowLeft')stepTo(S.ply-1);else if(k==='ArrowRight')stepTo(S.ply+1);else if(k==='Home')stepTo(0);else if(k==='End')stepTo(S.moves.length);
  else if(k==='u')undo();else if(k==='h'||k==='p'||k==='q'){S.ov=k;link();drawBoard();}else if(k==='t'){S.tac=!S.tac;link();drawBoard();}
  else if(k==='s')runSymmetry();else if(k==='v')runTrace();else if(k==='l')navigator.clipboard.writeText(location.href);else if(k==='c')navigator.clipboard.writeText(fmt(cur()));
  else if(k==='?'){$('sheet').hidden=!$('sheet').hidden;}else if(k==='Escape'){$('sheet').hidden=true;}else return;e.preventDefault();});
$('sheet').onclick=()=>{$('sheet').hidden=true;};

// ---- boot -----------------------------------------------------------------------------
(async function boot(){
  const qs=new URL(location.href).searchParams;
  try{if(qs.get('m'))S.moves=parseText(qs.get('m'));}catch(e){S.moves=[];}
  S.ply=qs.get('ply')!=null?Math.max(0,Math.min(S.moves.length,+qs.get('ply'))):S.moves.length;
  if(qs.get('ov'))S.ov=qs.get('ov');S.tac=qs.get('tac')==='1';
  if(qs.get('sims')!=null){S.sims=Math.max(0,+qs.get('sims')|0);S.simsChosen=true;$('sims').value=S.sims;}
  try{S.engines=(await (await fetch('/engines')).json()).engines;}catch(e){$('hdr').textContent='cannot reach /engines: '+e;return;}
  const known=new Set(S.engines.map(r=>r.id)),wanted=(qs.get('e')||'').split(',').filter(id=>known.has(id));
  const first=(S.engines.find(r=>r.kind==='mantis')||{}).id||null;
  for(const id of (wanted.length?wanted:[first]))addSlot(id);
  setDeployChip();edited();
})();
})();
