(function(){
'use strict';
const $=id=>document.getElementById(id);
const {key}=HexBoard;
// The position is shared; everything an engine says (its card, reads, window, trace) lives on its slot.
const S={moves:[],ply:0,engines:[],slots:[],active:0,sims:0,simsChosen:false,ov:'h',tac:false,auto:true,seq:0,nextSlot:0,
  client:Math.random().toString(36).slice(2),timer:null,tick:null};
const fmt=m=>m.map(key).join(';');
const f3=v=>(v>=0?'+':'')+v.toFixed(3);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const has=x=>!!(x&&!x.absent);
const cur=()=>S.moves.slice(0,S.ply);
const activeSlot=()=>S.slots[S.active];
const posOf=sl=>(sl.pos&&fmt(sl.pos.moves)===fmt(cur()))?sl.pos:null;
const argmaxOf=sl=>(has(sl.search)&&sl.search.argmax)||(has(sl.raw)&&sl.raw.argmax)||null;
const meta=c=>`${c.search_kind} · ${c.device}${c.threads==null?'':' · '+c.threads+' threads'} · radius ${c.radius}`;
function hoverText(sl,q,r){let t=`(${q},${r})`;
  const pr=has(sl.raw)&&sl.raw.policy.find(c=>c[0]===q&&c[1]===r),ch=has(sl.search)&&sl.search.children.find(c=>c[0]===q&&c[1]===r);
  if(pr)t+=` · p ${(pr[2]*100).toFixed(1)} %`;if(ch)t+=` · n ${ch[3]} · Q ${f3(ch[4])}`;return t;}

// ---- slots ------------------------------------------------------------------------
function engineLabel(r){if(r.kind==='mantis')return `${r.run_id} · ${r.step.toLocaleString()} · ${r.sha8}`;
  if(r.kind==='strix')return 'strix'+(r.note?' — '+r.note:'');return `${r.id} — ${r.note}`;}
function addSlot(engine){
  const el=$('cardtpl').content.firstElementChild.cloneNode(true);
  const sl={id:S.nextSlot++,engine:engine||null,card:null,pos:null,raw:null,search:null,tactics:null,sym:null,trace:null,traceOf:null,searching:null,el:el};
  const sel=el.querySelector('.cardengine');
  for(const r of S.engines){const o=document.createElement('option');o.value=(r.kind==='mantis'||r.kind==='strix')?r.id:'';o.disabled=!o.value;o.textContent=engineLabel(r);sel.appendChild(o);}
  sel.value=sl.engine||'';
  sel.addEventListener('change',e=>{sl.engine=e.target.value||null;sl.card=null;setDeployChip();requestSlot(sl);if(S.auto&&S.sims>0)analyze(sl,S.sims,false);});
  el.querySelector('.cardhead').addEventListener('click',e=>{if(e.target.tagName==='SELECT'||e.target.tagName==='BUTTON')return;S.active=S.slots.indexOf(sl);drawAll();});
  el.querySelector('.drop').onclick=()=>{if(S.slots.length===1)return;const act=activeSlot();S.slots.splice(S.slots.indexOf(sl),1);el.remove();S.active=Math.max(0,S.slots.indexOf(act));link();drawAll();};
  S.slots.push(sl);$('cards').appendChild(el);
  return sl;
}
function status(sl,t,err){const e=sl.el.querySelector('.status');e.textContent=t;e.className='status'+(err?' err':'');}

// ---- requests ---------------------------------------------------------------------
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return {status:r.status,body:await r.json()};}
function endSearch(sl,seq){if(sl.searching&&sl.searching.seq===seq){sl.searching=null;if(!S.slots.some(s=>s.searching))tickStop();}}
async function analyze(sl,sims,symmetry){
  if(!sl.engine)return;
  const seq=++S.seq,moves=fmt(cur());
  if(sims>0){sl.searching={sims:sims,t0:performance.now(),seq:seq};tickStart();}
  let out;try{out=await post('/analyze',{engine:sl.engine,moves:moves,sims:sims,symmetry:!!symmetry,client:S.client+':'+sl.id,seq:seq});}
  catch(e){endSearch(sl,seq);status(sl,'network: '+e,true);return;}
  const b=out.body;
  if(b.superseded){endSearch(sl,seq);return;}
  endSearch(sl,seq);
  if(!b.ok){
    if(out.status===503||out.status===404)sl.card=null;
    if(out.status===400&&sl===activeSlot()&&moves===fmt(cur())&&S.ply>0&&b.refused.includes('at ply '+(S.ply-1))){
      S.moves=cur();S.moves.pop();S.ply=S.moves.length;status(sl,b.refused+' — undone',true);edited();return;}
    status(sl,out.status+' '+b.refused,true);drawCard(sl);return;}
  const rec=b.record;if(fmt(rec.position.moves)!==fmt(cur()))return;
  sl.card=rec.engine;sl.pos=rec.position;sl.tactics=rec.tactics;sl.raw=rec.raw;
  if(sims>0||!has(rec.raw))sl.search=rec.search;
  if(has(rec.symmetry))sl.sym=rec.symmetry;
  status(sl,'');setDeployChip();drawAll();
}
function requestSlot(sl){sl.raw=null;sl.search=null;sl.tactics=null;sl.sym=null;sl.searching=null;
  if(sl.traceOf!==fmt(S.moves)){sl.trace=null;sl.traceOf=null;}
  analyze(sl,0,false);drawCard(sl);}
function tickStart(){if(!S.tick)S.tick=setInterval(()=>S.slots.forEach(drawReadout),100);}
function tickStop(){if(S.tick){clearInterval(S.tick);S.tick=null;}}
function searchAll(){S.slots.forEach(sl=>analyze(sl,S.sims,false));}
function edited(){tickStop();link();S.slots.forEach(requestSlot);
  clearTimeout(S.timer);if(S.auto&&S.sims>0)S.timer=setTimeout(searchAll,300);drawAll();}
function setDeployChip(){const card=activeSlot().card,b=$('deploychip');
  b.dataset.value=card?card.deploy_sims:'';b.textContent=card?'deploy '+card.deploy_sims:'deploy';b.disabled=!card;
  if(card&&!S.simsChosen&&S.sims===0){S.sims=card.deploy_sims;$('sims').value=S.sims;link();if(S.auto)searchAll();}
  $('budget').querySelectorAll('button').forEach(x=>x.classList.toggle('on',+x.dataset.value===S.sims));}

// ---- drawing ----------------------------------------------------------------------
function drawAll(){drawBoard();S.slots.forEach(drawCard);drawHeader();}
function drawCard(sl){sl.el.classList.toggle('active',sl===activeSlot());sl.el.querySelector('.slotno').textContent=String(S.slots.indexOf(sl));
  sl.el.querySelector('.cardmeta').textContent=sl.card?meta(sl.card):'';drawReadout(sl);drawChildren(sl);drawTactics(sl);drawPanels(sl);}
function drawHeader(){const c=activeSlot().card,p=posOf(activeSlot());
  $('hdr').textContent=c?`${c.run_id} · ${c.step.toLocaleString()} · ${c.sha8||''} · ${meta(c)}`:'no engine loaded';
  $('pos').textContent=p?`ply ${p.ply} · ${p.winner?p.winner+' won':p.to_move+' to move · '+p.moves_remaining+' left'} · ${p.legal} legal`:`ply ${S.ply}`;
  if(document.activeElement!==$('moves'))$('moves').value=fmt(cur());}
function overlay(sl){
  if(S.ov==='p'&&has(sl.raw)){const max=Math.max(...sl.raw.policy.map(r=>r[2]),1e-9);
    return sl.raw.policy.filter(r=>r[2]>=0.02||r[2]===max).map(r=>({c:[r[0],r[1]],fill:'--heat',alpha:(0.25+0.55*r[2]/max).toFixed(2),label:(r[2]*100).toFixed(0)}));}
  if(!has(sl.search))return [];
  const ch=sl.search.children;
  if(S.ov==='q')return ch.filter(r=>r[3]>=2).map(r=>({c:[r[0],r[1]],fill:r[4]>=0?'--pos':'--neg',alpha:Math.min(1,Math.abs(r[4])).toFixed(2),label:f3(r[4])}));
  if(S.ov==='h'){const max=Math.max(...ch.map(r=>r[3]),1);
    return ch.filter(r=>r[3]>=2||r[3]===max).map(r=>({c:[r[0],r[1]],fill:'--heat',alpha:(0.25+0.55*r[3]/max).toFixed(2),label:String(r[3])}));}
  return [];}
function drawBoard(){const sl=activeSlot(),p=posOf(sl);
  HexBoard.draw($('board'),{moves:S.moves,ply:S.ply,window:p?p.legal_window:[],winLine:p?p.win_line:null,overlay:overlay(sl),
    tactics:S.tac&&sl.tactics&&sl.tactics.cells?{cells:sl.tactics.cells,fours:sl.tactics.opp_fours||[]}:null,
    marks:S.slots.map((o,i)=>({c:argmaxOf(o),label:i,active:o===sl})).filter(m=>m.c)});}
function drawReadout(sl){
  const raw=sl.el.querySelector('.r-raw').children,head=sl.el.querySelector('.r-head').children;
  if(has(sl.raw)){raw[1].textContent=f3(sl.raw.value);raw[2].textContent='argmax ('+sl.raw.argmax+')';raw[3].textContent=sl.raw.ms+' ms';}
  else{raw[1].textContent='—';raw[2].textContent=sl.raw?sl.raw.absent:(sl.engine?'reading…':'no engine');raw[3].textContent='';}
  if(sl.searching){head[1].textContent='…';head[2].textContent='searching · '+sl.searching.sims;head[3].textContent=((performance.now()-sl.searching.t0)/1000).toFixed(1)+' s';}
  else if(has(sl.search)){head[1].textContent=f3(sl.search.root_value);head[2].textContent='argmax ('+sl.search.argmax+')';head[3].textContent=`${sl.search.sims} · ${sl.search.root_visits} visits · ${(sl.search.ms/1000).toFixed(1)} s · q-fires ${sl.search.quiescence_fires}`;}
  else{head[1].textContent='—';head[2].textContent=sl.search?sl.search.absent:(S.sims>0?'not searched yet':'sims 0');head[3].textContent='';}}
function drawChildren(sl){
  const tb=sl.el.querySelector('.children tbody');tb.innerHTML='';let rows=[],arg=null;
  if(has(sl.search)){rows=sl.search.children.map(r=>({c:[r[0],r[1]],p:r[2],n:r[3],q:r[4]}));arg=key(sl.search.argmax);}
  else if(has(sl.raw)){rows=sl.raw.policy.map(r=>({c:[r[0],r[1]],p:r[2],n:null,q:null}));arg=key(sl.raw.argmax);}
  for(const r of rows.slice(0,12)){const tr=document.createElement('tr');if(key(r.c)===arg)tr.className='arg';
    tr.innerHTML=`<td>(${r.c})</td><td>${(r.p*100).toFixed(1)} %</td><td>${r.n==null?'—':r.n}</td><td>${r.q==null?'—':f3(r.q)}</td>`;
    tr.onclick=()=>{$('hover').textContent=hoverText(sl,r.c[0],r.c[1]);};tb.appendChild(tr);}
  sl.el.querySelector('.more').textContent=rows.length>12?`… ${rows.length-12} more (board floor: visits ≥ 2, prior ≥ 2 %)`:'';}
function drawTactics(sl){
  const t=sl.tactics,e=sl.el.querySelector('.tac');if(!t){e.textContent='—';return;}
  if(t.class==='terminal'){e.innerHTML=`<span class="cls">game over</span> · ${esc(t.winner)} wins`;return;}
  const v=t.verdict,side=(name,s)=>s?` · ${name}: <span class="${s.includes('MISSES')?'miss':''}">${esc(s)}</span>`:'';
  e.innerHTML=`<span class="cls">${esc(t.class.toUpperCase())}</span>${t.cells.length?' '+esc(JSON.stringify(t.cells)):''} · k=${+t.k}`
    +side('net',v.raw)+side('head',v.search)+(t.forced_win_move?` · forced_win_move(2) → (${esc(t.forced_win_move)})`:'')+`<div class="mute">${esc(t.derivation)}</div>`;}
function drawPanels(sl){
  const s=sl.el.querySelector('.sym');
  if(sl.sym){const y=sl.sym;s.innerHTML=`<b>symmetry</b> n=${+y.n} about (${esc(y.centre)}) · value ${f3(y.value_min)} … ${f3(y.value_max)} · spread ${y.spread.toFixed(4)} · argmax agrees ${esc(y.argmax_agreement)} · worst ${esc(y.worst.map)} ${f3(y.worst.value)} argmax (${esc(y.worst.argmax)})`
      +(y.translation.absent?`<div class="mute">translation: ${esc(y.translation.absent)}</div>`:` · translation ${f3(y.translation.value)}`);}
  else s.innerHTML=`<b>symmetry</b> <span class="mute">(s): not requested — 13 net reads, about ${has(sl.raw)?((13*sl.raw.ms)/1000).toFixed(1)+' s':'a second'}</span>`;
  const t=sl.el.querySelector('.trace'),svg=t.querySelector('.tracesvg'),span=t.querySelector('span');
  svg.toggleAttribute('hidden',!sl.trace);
  if(sl.trace){const rows=sl.trace,n=rows.length;svg.setAttribute('viewBox',`0 0 ${Math.max(n-1,1)} 100`);svg.setAttribute('preserveAspectRatio','none');
    const path=k=>{let d='',pen=false;rows.forEach((r,i)=>{if(r[k]==null){pen=false;return;}d+=(pen?'L':'M')+i+' '+(50-50*r[k]).toFixed(2)+' ';pen=true;});return d;};
    svg.innerHTML=`<line class="zero" x1="0" y1="50" x2="${n}" y2="50"/><path class="raw" d="${path('raw')}"/><path class="root" d="${path('root')}"/><line class="cur" x1="${S.ply}" y1="0" x2="${S.ply}" y2="100"/>`;
    span.textContent=` P1's perspective (the readout is the mover's) · thin: net · thick: head${rows.some(r=>r.root!=null)?'':' (not run)'} · gaps are unsearched or terminal plies`;}
  else{const n=S.moves.length+1,cost=has(sl.search)?` (about ${(n*sl.search.ms/1000).toFixed(0)} s)`:'';
    span.textContent=`(v): not requested — ${n} net reads`+(S.sims>0?` + ${n} searches at ${S.sims}${cost}`:'');}}

// ---- editing --------------------------------------------------------------------------
function place(q,r){const p=posOf(activeSlot());if(p&&p.winner)return;S.moves=cur();S.moves.push([q,r]);S.ply=S.moves.length;edited();}
function stepTo(k){S.ply=Math.max(0,Math.min(S.moves.length,k));edited();}
function undo(){if(S.ply===0)return;S.moves=cur();S.moves.pop();S.ply=S.moves.length;edited();}
function parseText(t){const s=t.trim();const parts=!s?[]:s.startsWith('[')?JSON.parse(s):s.split(';').map(x=>x.split(',').map(Number));
  if(!parts.every(c=>Array.isArray(c)&&c.length===2&&c.every(Number.isInteger)))throw new Error('not q,r;q,r');return parts;}
function importText(t){try{S.moves=parseText(t);S.ply=S.moves.length;edited();}catch(e){status(activeSlot(),'import: '+e.message,true);}}
function link(){const u=new URL(location.href);u.searchParams.set('m',fmt(S.moves));u.searchParams.set('ply',S.ply);
  const es=S.slots.map(s=>s.engine).filter(Boolean);if(es.length)u.searchParams.set('e',es.join(','));else u.searchParams.delete('e');
  u.searchParams.set('sims',S.sims);u.searchParams.set('ov',S.ov);if(S.tac)u.searchParams.set('tac','1');else u.searchParams.delete('tac');history.replaceState(null,'',u);}
async function runSymmetry(){const sl=activeSlot(),p=posOf(sl);if(!sl.engine||(p&&p.winner))return;status(sl,'symmetry sweep…');await analyze(sl,0,true);}
async function runTrace(){const sl=activeSlot();if(!sl.engine)return;const seq=++S.seq,of=fmt(S.moves);status(sl,'trace…');
  let out;try{out=await post('/trace',{engine:sl.engine,moves:of,sims:S.sims,client:S.client+':'+sl.id,seq:seq});}catch(e){status(sl,'network: '+e,true);return;}
  if(out.body.superseded)return;if(!out.body.ok){status(sl,out.status+' '+out.body.refused,true);return;}
  sl.trace=out.body.trace;sl.traceOf=of;status(sl,'');drawPanels(sl);}

// ---- wiring ---------------------------------------------------------------------------
const board=$('board');
board.addEventListener('click',e=>{const [q,r]=HexBoard.cellAt(board,e),p=posOf(activeSlot());
  if(p&&!p.legal_window.some(c=>c[0]===q&&c[1]===r)){status(activeSlot(),`(${q},${r}) is outside the legal window`,true);return;}place(q,r);});
board.addEventListener('mousemove',e=>{const [q,r]=HexBoard.cellAt(board,e);$('hover').textContent=hoverText(activeSlot(),q,r);});
$('budget').querySelectorAll('button').forEach(b=>b.onclick=()=>{if(b.dataset.value==='')return;S.sims=+b.dataset.value;S.simsChosen=true;$('sims').value=S.sims;setDeployChip();link();if(S.auto)searchAll();});
$('sims').addEventListener('change',e=>{S.sims=Math.max(0,+e.target.value|0);S.simsChosen=true;setDeployChip();link();});
$('auto').addEventListener('change',e=>{S.auto=e.target.checked;});
$('go').onclick=searchAll;
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
  const ply=parseInt(qs.get('ply'),10);S.ply=Number.isInteger(ply)?Math.max(0,Math.min(S.moves.length,ply)):S.moves.length;
  if(qs.get('ov'))S.ov=qs.get('ov');S.tac=qs.get('tac')==='1';
  const sims=parseInt(qs.get('sims'),10);if(Number.isInteger(sims)&&sims>=0){S.sims=sims;S.simsChosen=true;$('sims').value=S.sims;}
  try{S.engines=(await (await fetch('/engines')).json()).engines;}catch(e){$('hdr').textContent='cannot reach /engines: '+e;return;}
  const known=new Set(S.engines.map(r=>r.id)),wanted=(qs.get('e')||'').split(',').filter(id=>known.has(id));
  const first=(S.engines.find(r=>r.kind==='mantis')||{}).id||null;
  for(const id of (wanted.length?wanted:[first]))addSlot(id);
  setDeployChip();edited();
})();
})();
