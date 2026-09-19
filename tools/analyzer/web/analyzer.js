(function(){
'use strict';
const $=id=>document.getElementById(id);
const S={moves:[],ply:0,engines:[],engine:null,card:null,sims:0,simsChosen:false,ov:'h',tac:false,auto:true,seq:0,
  client:Math.random().toString(36).slice(2),raw:null,search:null,pos:null,tactics:null,sym:null,trace:null,
  searching:null,timer:null,tick:null};
const SQ3=Math.sqrt(3),X=(q,r)=>SQ3*(q+r/2),Y=(q,r)=>1.5*r;
function pixelToHex(x,y){const qf=SQ3/3*x-y/3,rf=2/3*y,sf=-qf-rf;let q=Math.round(qf),r=Math.round(rf),s=Math.round(sf);
  const dq=Math.abs(q-qf),dr=Math.abs(r-rf),ds=Math.abs(s-sf);if(dq>dr&&dq>ds)q=-r-s;else if(dr>ds)r=-q-s;return [q,r];}
const owner=p=>(((p+1)/2|0)%2);
const fmt=m=>m.map(c=>c[0]+','+c[1]).join(';');
const key=c=>c[0]+','+c[1];
const f3=v=>(v>=0?'+':'')+v.toFixed(3);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function hexPts(cx,cy){const p=[];for(let i=0;i<6;i++){const a=Math.PI/180*(60*i-30);p.push((cx+Math.cos(a)).toFixed(3)+','+(cy+Math.sin(a)).toFixed(3));}return p.join(' ');}
function cur(){return S.moves.slice(0,S.ply);}
function posCurrent(){return S.pos&&fmt(S.pos.moves)===fmt(cur());}
function status(t,err){const e=$('status');e.textContent=t;e.className='status'+(err?' err':'');}

// ---- requests --------------------------------------------------------------
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return {status:r.status,body:await r.json()};}
async function analyze(sims,symmetry){
  if(!S.engine)return;
  const seq=++S.seq,moves=fmt(cur());
  const req={engine:S.engine,moves:moves,sims:sims,symmetry:!!symmetry,client:S.client,seq:seq};
  if(sims>0){S.searching={sims:sims,t0:performance.now(),seq:seq};tickStart();}
  let out;try{out=await post('/analyze',req);}catch(e){status('network: '+e,true);return;}
  const b=out.body;
  if(b.superseded)return;
  if(S.searching&&S.searching.seq===seq){S.searching=null;tickStop();}
  if(!b.ok){
    if(out.status===503||out.status===404)S.card=null;
    if(out.status===400&&moves===fmt(cur())&&S.ply>0&&b.refused.includes('at ply '+(S.ply-1))){S.moves=cur();S.moves.pop();S.ply=S.moves.length;status(b.refused+' — undone',true);edited();return;}
    status(out.status+' '+b.refused,true);draw();return;}
  const rec=b.record;if(fmt(rec.position.moves)!==fmt(cur()))return;
  S.card=rec.engine;S.pos=rec.position;S.tactics=rec.tactics;S.raw=rec.raw;
  if(sims>0||(rec.search.absent&&!rec.search.absent.startsWith('sims=0')))S.search=rec.search;
  if(rec.symmetry&&!rec.symmetry.absent)S.sym=rec.symmetry;
  status('');setDeployChip();draw();
}
function tickStart(){tickStop();S.tick=setInterval(drawReadout,100);}
function tickStop(){if(S.tick){clearInterval(S.tick);S.tick=null;}}
function edited(){S.raw=null;S.search=null;S.tactics=null;S.sym=null;S.trace=null;S.searching=null;tickStop();link();
  analyze(0,false);clearTimeout(S.timer);if(S.auto&&S.sims>0)S.timer=setTimeout(()=>analyze(S.sims,false),300);draw();}
function setDeployChip(){const b=$('budget').querySelector('[data-sims="deploy"]');
  if(S.card){b.textContent='deploy '+S.card.deploy_sims;b.disabled=false;
    if(!S.simsChosen&&S.sims===0){S.sims=S.card.deploy_sims;$('sims').value=S.sims;link();if(S.auto)analyze(S.sims,false);}}
  else{b.textContent='deploy';b.disabled=true;}
  $('budget').querySelectorAll('button').forEach(x=>x.classList.toggle('on',(x.dataset.sims==='deploy'?(S.card&&S.card.deploy_sims):+x.dataset.sims)===S.sims));}

// ---- drawing ---------------------------------------------------------------
function draw(){drawBoard();drawReadout();drawChildren();drawTactics();drawPanels();drawHeader();}
function drawHeader(){const c=S.card;
  $('hdr').textContent=c?`${c.run_id} · ${c.step.toLocaleString()} · ${c.sha8||''} · ${c.search_kind} · ${c.device} · ${c.threads} threads · radius ${c.radius}`:'no engine loaded';
  $('cardhead').textContent=c?`${c.run_id} @${c.step.toLocaleString()} · ${c.search_kind} · ${c.device}`:'no engine loaded';
  const p=posCurrent()?S.pos:null;
  $('pos').textContent=p?`ply ${p.ply} · ${p.winner?p.winner+' won':p.to_move+' to move · '+p.moves_remaining+' left'} · ${p.legal} legal`:`ply ${S.ply}`;
  if(document.activeElement!==$('moves'))$('moves').value=fmt(cur());}
function overlayCells(){
  if(S.ov==='p'&&S.raw&&!S.raw.absent)return {rows:S.raw.policy.map(r=>({c:[r[0],r[1]],v:r[2]})),max:Math.max(...S.raw.policy.map(r=>r[2]),1e-9),floor:0.02,label:v=>(v*100).toFixed(0)};
  if(S.search&&!S.search.absent){const ch=S.search.children;
    if(S.ov==='h')return {rows:ch.map(r=>({c:[r[0],r[1]],v:r[3]})),max:Math.max(...ch.map(r=>r[3]),1),floor:2,label:v=>String(v)};
    if(S.ov==='q')return {rows:ch.filter(r=>r[3]>=2).map(r=>({c:[r[0],r[1]],v:r[4]})),max:1,floor:-2,label:v=>f3(v),div:true};}
  return null;}
function drawBoard(){
  const m=S.moves,st=cur(),win=(posCurrent()&&S.pos.legal_window)||[],cells=[...m,...win];if(!cells.length)cells.push([0,0]);
  const xs=cells.map(c=>X(c[0],c[1])),ys=cells.map(c=>Y(c[0],c[1])),pad=2;
  const x0=Math.min(...xs)-pad,x1=Math.max(...xs)+pad,y0=Math.min(...ys)-pad,y1=Math.max(...ys)+pad,w=x1-x0,h=y1-y0,Sz=Math.max(w,h);
  const svg=$('board');svg.setAttribute('viewBox',(x0-(Sz-w)/2)+' '+(y0-(Sz-h)/2)+' '+Sz+' '+Sz);
  const out=[],winSet=new Set(win.map(key)),played=new Set(st.map(key));
  const qmin=Math.floor(Math.min(...cells.map(c=>c[0])))-2,qmax=Math.ceil(Math.max(...cells.map(c=>c[0])))+2,rmin=Math.floor(Math.min(...cells.map(c=>c[1])))-2,rmax=Math.ceil(Math.max(...cells.map(c=>c[1])))+2;
  for(let q=qmin;q<=qmax;q++)for(let r=rmin;r<=rmax;r++){const x=X(q,r),y=Y(q,r);if(x<x0-1||x>x1+1||y<y0-1||y>y1+1)continue;
    out.push(`<polygon class="cell${winSet.has(key([q,r]))?' win hot':''}" points="${hexPts(x,y)}"/>`);}
  const ov=overlayCells();
  if(ov){for(const row of ov.rows){if(played.has(key(row.c)))continue;const x=X(row.c[0],row.c[1]),y=Y(row.c[0],row.c[1]);
    if(ov.div){out.push(`<polygon class="heat" style="fill:var(${row.v>=0?'--pos':'--neg'});fill-opacity:${Math.min(1,Math.abs(row.v)).toFixed(2)}" points="${hexPts(x,y)}"/>`);}
    else{if(row.v<ov.floor&&row.v!==ov.max)continue;out.push(`<polygon class="heat" style="fill:var(--heat);fill-opacity:${(0.25+0.55*row.v/ov.max).toFixed(2)}" points="${hexPts(x,y)}"/>`);}
    out.push(`<text class="heatnum" x="${x.toFixed(3)}" y="${y.toFixed(3)}">${ov.label(row.v)}</text>`);}}
  if(S.tac&&S.tactics&&S.tactics.cells){for(const c of S.tactics.cells)out.push(`<polygon class="tacwin" points="${hexPts(X(c[0],c[1]),Y(c[0],c[1]))}"/>`);
    for(const four of (S.tactics.opp_fours||[]))for(const c of four)out.push(`<polygon class="tacfour" points="${hexPts(X(c[0],c[1]),Y(c[0],c[1]))}"/>`);}
  if(posCurrent()&&S.pos.win_line)for(const c of S.pos.win_line)out.push(`<polygon class="wincell" points="${hexPts(X(c[0],c[1]),Y(c[0],c[1]))}"/>`);
  st.forEach((c,i)=>{const x=X(c[0],c[1]),y=Y(c[0],c[1]),o=owner(i)?'p2':'p1';out.push(`<circle class="stone ${o}" cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="0.78"/><text class="num ${o}" x="${x.toFixed(3)}" y="${y.toFixed(3)}">${i}</text>`);});
  for(let i=Math.max(0,S.ply-2);i<S.ply;i++){const c=m[i];out.push(`<circle class="last" cx="${X(c[0],c[1]).toFixed(3)}" cy="${Y(c[0],c[1]).toFixed(3)}" r="0.9"/>`);}
  const arg=(S.search&&!S.search.absent&&S.search.argmax)||(S.raw&&!S.raw.absent&&S.raw.argmax);
  if(arg)out.push(`<circle class="argmax" cx="${X(arg[0],arg[1]).toFixed(3)}" cy="${Y(arg[0],arg[1]).toFixed(3)}" r="0.62"/>`);
  svg.innerHTML=out.join('');}
function drawReadout(){
  const raw=$('r-raw').children,head=$('r-head').children;
  if(S.raw&&!S.raw.absent){raw[1].textContent=f3(S.raw.value);raw[2].textContent='argmax ('+S.raw.argmax+')';raw[3].textContent=S.raw.ms+' ms';}
  else{raw[1].textContent='—';raw[2].textContent=S.raw&&S.raw.absent?S.raw.absent:(S.engine?'reading…':'—');raw[3].textContent='';}
  if(S.searching){head[1].textContent='…';head[2].textContent='searching · '+S.searching.sims;head[3].textContent=((performance.now()-S.searching.t0)/1000).toFixed(1)+' s';}
  else if(S.search&&!S.search.absent){head[1].textContent=f3(S.search.root_value);head[2].textContent='argmax ('+S.search.argmax+')';head[3].textContent=`${S.search.sims} · ${S.search.root_visits} visits · ${(S.search.ms/1000).toFixed(1)} s · q-fires ${S.search.quiescence_fires}`;}
  else{head[1].textContent='—';head[2].textContent=S.search&&S.search.absent?S.search.absent:(S.sims>0?'not searched yet':'sims 0');head[3].textContent='';}}
function drawChildren(){
  const tb=$('children');tb.innerHTML='';let rows=[],arg=null;
  if(S.search&&!S.search.absent){rows=S.search.children.map(r=>({c:[r[0],r[1]],p:r[2],n:r[3],q:r[4]}));arg=key(S.search.argmax);}
  else if(S.raw&&!S.raw.absent){rows=S.raw.policy.map(r=>({c:[r[0],r[1]],p:r[2],n:null,q:null}));arg=key(S.raw.argmax);}
  for(const r of rows.slice(0,12)){const tr=document.createElement('tr');if(key(r.c)===arg)tr.className='arg';
    tr.innerHTML=`<td>(${r.c})</td><td>${(r.p*100).toFixed(1)} %</td><td>${r.n==null?'—':r.n}</td><td>${r.q==null?'—':f3(r.q)}</td>`;
    tr.onclick=()=>{$('hover').textContent=`(${r.c}) · p ${(r.p*100).toFixed(1)} %`+(r.n==null?'':` · n ${r.n} · Q ${f3(r.q)}`);};tb.appendChild(tr);}
  $('more').textContent=rows.length>12?`… ${rows.length-12} more (board floor: visits ≥ 2, prior ≥ 2 %)`:'';}
function drawTactics(){
  const t=S.tactics,e=$('tactics');if(!t){e.textContent='—';return;}
  if(t.class==='terminal'){e.innerHTML=`<span class="cls">game over</span> · ${esc(t.winner)} wins`;return;}
  const v=t.verdict,miss=s=>s&&s.includes('MISSES')?'miss':'';
  e.innerHTML=`<span class="cls">${esc(t.class.toUpperCase())}</span>${t.cells.length?' '+esc(JSON.stringify(t.cells)):''} · k=${+t.k}`
    +(v.raw?` · net: <span class="${miss(v.raw)}">${esc(v.raw)}</span>`:'')+(v.search?` · head: <span class="${miss(v.search)}">${esc(v.search)}</span>`:'')
    +(t.forced_win_move?` · forced_win_move(2) → (${esc(t.forced_win_move)})`:'')+`<div class="mute">${esc(t.derivation)}</div>`;}
function drawPanels(){
  const s=$('sym');
  if(S.sym){const y=S.sym;s.innerHTML=`<b>symmetry</b> n=${+y.n} about (${esc(y.centre)}) · value ${f3(y.value_min)} … ${f3(y.value_max)} · spread ${y.spread.toFixed(4)} · argmax agrees ${esc(y.argmax_agreement)} · worst ${esc(y.worst.map)} ${f3(y.worst.value)} argmax (${esc(y.worst.argmax)})`
      +(y.translation.absent?`<div class="mute">translation: ${esc(y.translation.absent)}</div>`:` · translation ${f3(y.translation.value)}`);}
  else s.innerHTML=`<b>symmetry</b> <span class="mute">(s): not requested — 13 net reads, about ${S.raw&&S.raw.ms?((13*S.raw.ms)/1000).toFixed(1)+' s':'a second'}</span>`;
  const t=$('trace'),svg=$('tracesvg'),span=t.querySelector('span');
  if(S.trace){const rows=S.trace,n=rows.length;svg.hidden=false;svg.setAttribute('viewBox',`0 0 ${Math.max(n-1,1)} 100`);svg.setAttribute('preserveAspectRatio','none');
    const path=k=>{let d='',pen=false;rows.forEach((r,i)=>{if(r[k]==null){pen=false;return;}d+=(pen?'L':'M')+i+' '+(50-50*r[k]).toFixed(2)+' ';pen=true;});return d;};
    svg.innerHTML=`<line class="zero" x1="0" y1="50" x2="${n}" y2="50"/><path class="raw" d="${path('raw')}"/><path class="root" d="${path('root')}"/><line class="cur" x1="${S.ply}" y1="0" x2="${S.ply}" y2="100"/>`;
    span.textContent=` P1's perspective (the readout is the mover's) · thin: net · thick: head${rows.some(r=>r.root!=null)?'':' (not run)'} · gaps are unsearched or terminal plies`;}
  else{svg.hidden=true;span.textContent=`(v): not requested — ${S.moves.length+1} net reads`+(S.sims>0?` + ${S.moves.length+1} searches at ${S.sims}`+(S.search&&!S.search.absent?` (about ${((S.moves.length+1)*S.search.ms/1000).toFixed(0)} s)`:''):'');}}

// ---- editing ----------------------------------------------------------------
function place(q,r){if(posCurrent()&&S.pos.winner)return;S.moves=cur();S.moves.push([q,r]);S.ply=S.moves.length;edited();}
function stepTo(k){S.ply=Math.max(0,Math.min(S.moves.length,k));edited();}
function undo(){if(S.ply===0)return;S.moves=cur();S.moves.pop();S.ply=S.moves.length;edited();}
function importText(t){try{const s=t.trim();const parts=!s?[]:s.startsWith('[')?JSON.parse(s):s.split(';').map(x=>x.split(',').map(Number));
  if(!parts.every(c=>Array.isArray(c)&&c.length===2&&c.every(Number.isInteger)))throw new Error('not q,r;q,r');S.moves=parts;S.ply=parts.length;edited();}catch(e){status('import: '+e.message,true);}}
function link(){const u=new URL(location.href);u.searchParams.set('m',fmt(S.moves));u.searchParams.set('ply',S.ply);if(S.engine)u.searchParams.set('e',S.engine);u.searchParams.set('sims',S.sims);u.searchParams.set('ov',S.ov);if(S.tac)u.searchParams.set('tac','1');else u.searchParams.delete('tac');history.replaceState(null,'',u);}
async function runSymmetry(){if(!S.engine||(posCurrent()&&S.pos.winner))return;status('symmetry sweep…');await analyze(0,true);}
async function runTrace(){if(!S.engine)return;const seq=++S.seq;status('trace…');
  let out;try{out=await post('/trace',{engine:S.engine,moves:fmt(S.moves),sims:S.sims,client:S.client,seq:seq});}catch(e){status('network: '+e,true);return;}
  if(out.body.superseded)return;if(!out.body.ok){status(out.status+' '+out.body.refused,true);return;}S.trace=out.body.trace;status('');drawPanels();}

// ---- wiring ------------------------------------------------------------------
const board=$('board');
function cellAt(e){const pt=board.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const p=pt.matrixTransform(board.getScreenCTM().inverse());return pixelToHex(p.x,p.y);}
board.addEventListener('click',e=>{const [q,r]=cellAt(e);
  if(posCurrent()&&!S.pos.legal_window.some(c=>c[0]===q&&c[1]===r)){status(`(${q},${r}) is outside the legal window`,true);return;}place(q,r);});
board.addEventListener('mousemove',e=>{const [q,r]=cellAt(e);let t=`(${q},${r})`;
  const pr=S.raw&&!S.raw.absent&&S.raw.policy.find(c=>c[0]===q&&c[1]===r),ch=S.search&&!S.search.absent&&S.search.children.find(c=>c[0]===q&&c[1]===r);
  if(pr)t+=` · p ${(pr[2]*100).toFixed(1)} %`;if(ch)t+=` · n ${ch[3]} · Q ${f3(ch[4])}`;$('hover').textContent=t;});
$('engine').addEventListener('change',e=>{S.engine=e.target.value||null;S.card=null;S.simsChosen=false;S.sims=0;setDeployChip();edited();});
$('budget').querySelectorAll('button').forEach(b=>b.onclick=()=>{S.sims=b.dataset.sims==='deploy'?(S.card?S.card.deploy_sims:0):+b.dataset.sims;S.simsChosen=true;$('sims').value=S.sims;setDeployChip();link();if(S.auto)analyze(S.sims,false);});
$('sims').addEventListener('change',e=>{S.sims=Math.max(0,+e.target.value|0);S.simsChosen=true;setDeployChip();link();});
$('auto').addEventListener('change',e=>{S.auto=e.target.checked;});
$('go').onclick=()=>analyze(S.sims,false);
$('moves').addEventListener('change',e=>importText(e.target.value));
$('copy').onclick=()=>navigator.clipboard.writeText(fmt(cur()));
document.addEventListener('keydown',e=>{const tag=e.target.tagName;if(tag==='INPUT'||tag==='SELECT'||tag==='TEXTAREA')return;const k=e.key;
  if(k==='ArrowLeft')stepTo(S.ply-1);else if(k==='ArrowRight')stepTo(S.ply+1);else if(k==='Home')stepTo(0);else if(k==='End')stepTo(S.moves.length);
  else if(k==='u')undo();else if(k==='h'||k==='p'||k==='q'){S.ov=k;link();draw();}else if(k==='t'){S.tac=!S.tac;link();draw();}
  else if(k==='s')runSymmetry();else if(k==='v')runTrace();else if(k==='l')navigator.clipboard.writeText(location.href);else if(k==='c')navigator.clipboard.writeText(fmt(cur()));
  else if(k==='?'){$('sheet').hidden=!$('sheet').hidden;}else if(k==='Escape'){$('sheet').hidden=true;}else return;e.preventDefault();});
$('sheet').onclick=()=>{$('sheet').hidden=true;};

// ---- boot ----------------------------------------------------------------------
(async function boot(){
  const qs=new URL(location.href).searchParams;
  if(qs.get('m')){const s=qs.get('m');S.moves=s?s.split(';').map(x=>x.split(',').map(Number)):[];S.ply=S.moves.length;}
  if(qs.get('ply')!=null)S.ply=Math.max(0,Math.min(S.moves.length,+qs.get('ply')));
  if(qs.get('ov'))S.ov=qs.get('ov');S.tac=qs.get('tac')==='1';
  if(qs.get('sims')!=null){S.sims=Math.max(0,+qs.get('sims')|0);S.simsChosen=true;$('sims').value=S.sims;}
  let rows;try{rows=(await (await fetch('/engines')).json()).engines;}catch(e){status('cannot reach /engines: '+e,true);return;}
  S.engines=rows;const sel=$('engine');
  for(const r of rows){const o=document.createElement('option');o.value=(r.kind==='mantis'||r.kind==='strix')?r.id:'';o.disabled=!o.value;
    o.textContent=r.kind==='mantis'?`${r.run_id} · ${r.step.toLocaleString()} · ${r.sha8}`:r.kind==='strix'?`strix${r.note?' — '+r.note:''}`:`${r.id} — ${r.note}`;sel.appendChild(o);}
  const want=qs.get('e');S.engine=(want&&rows.some(r=>r.id===want))?want:((rows.find(r=>r.kind==='mantis')||{}).id||null);sel.value=S.engine||'';
  setDeployChip();edited();
})();
})();
