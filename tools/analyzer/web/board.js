// The hex board renderer, forked from tools/viewer/html.py: a scene in, SVG out; it knows nothing about engines.
window.HexBoard=(function(){
'use strict';
const SQ3=Math.sqrt(3),X=(q,r)=>SQ3*(q+r/2),Y=(q,r)=>1.5*r;
const key=c=>c[0]+','+c[1];
const owner=p=>(((p+1)/2|0)%2);
function pixelToHex(x,y){const qf=SQ3/3*x-y/3,rf=2/3*y,sf=-qf-rf;let q=Math.round(qf),r=Math.round(rf),s=Math.round(sf);
  const dq=Math.abs(q-qf),dr=Math.abs(r-rf),ds=Math.abs(s-sf);if(dq>dr&&dq>ds)q=-r-s;else if(dr>ds)r=-q-s;return [q,r];}
function hexPts(cx,cy){const p=[];for(let i=0;i<6;i++){const a=Math.PI/180*(60*i-30);p.push((cx+Math.cos(a)).toFixed(3)+','+(cy+Math.sin(a)).toFixed(3));}return p.join(' ');}
const poly=(cls,c,style)=>`<polygon class="${cls}"${style?` style="${style}"`:''} points="${hexPts(X(c[0],c[1]),Y(c[0],c[1]))}"/>`;
const text=(cls,c,t,dy)=>`<text class="${cls}" x="${X(c[0],c[1]).toFixed(3)}" y="${(Y(c[0],c[1])+(dy||0)).toFixed(3)}">${t}</text>`;
// scene: {moves, ply, window:[[q,r]], winLine:[[q,r]]|null, overlay:[{c,fill,alpha,label}], tactics:{cells,fours}|null, marks:[{c,label,active}]}
function draw(svg,sc){
  const st=sc.moves.slice(0,sc.ply),cells=[...sc.moves,...sc.window];if(!cells.length)cells.push([0,0]);
  const xs=cells.map(c=>X(c[0],c[1])),ys=cells.map(c=>Y(c[0],c[1])),pad=2;
  const x0=Math.min(...xs)-pad,x1=Math.max(...xs)+pad,y0=Math.min(...ys)-pad,y1=Math.max(...ys)+pad,w=x1-x0,h=y1-y0,S=Math.max(w,h);
  svg.setAttribute('viewBox',(x0-(S-w)/2)+' '+(y0-(S-h)/2)+' '+S+' '+S);
  const out=[],winSet=new Set(sc.window.map(key)),played=new Set(st.map(key));
  const qs=cells.map(c=>c[0]),rs=cells.map(c=>c[1]);
  for(let q=Math.min(...qs)-2;q<=Math.max(...qs)+2;q++)for(let r=Math.min(...rs)-2;r<=Math.max(...rs)+2;r++){
    const x=X(q,r),y=Y(q,r);if(x<x0-1||x>x1+1||y<y0-1||y>y1+1)continue;out.push(poly('cell'+(winSet.has(key([q,r]))?' win hot':''),[q,r]));}
  for(const o of sc.overlay){if(played.has(key(o.c)))continue;out.push(poly('heat',o.c,`fill:var(${o.fill});fill-opacity:${o.alpha}`));out.push(text('heatnum',o.c,o.label));}
  if(sc.tactics){for(const c of sc.tactics.cells)out.push(poly('tacwin',c));for(const four of sc.tactics.fours)for(const c of four)out.push(poly('tacfour',c));}
  if(sc.winLine)for(const c of sc.winLine)out.push(poly('wincell',c));
  st.forEach((c,i)=>{const o=owner(i)?'p2':'p1';out.push(`<circle class="stone ${o}" cx="${X(c[0],c[1]).toFixed(3)}" cy="${Y(c[0],c[1]).toFixed(3)}" r="0.78"/>`);out.push(text('num '+o,c,i));});
  for(let i=Math.max(0,sc.ply-2);i<sc.ply;i++){const c=sc.moves[i];out.push(`<circle class="last" cx="${X(c[0],c[1]).toFixed(3)}" cy="${Y(c[0],c[1]).toFixed(3)}" r="0.9"/>`);}
  for(const m of sc.marks){if(m.active)out.push(`<circle class="argmax" cx="${X(m.c[0],m.c[1]).toFixed(3)}" cy="${Y(m.c[0],m.c[1]).toFixed(3)}" r="0.62"/>`);else out.push(text('mark',m.c,m.label,0.55));}
  svg.innerHTML=out.join('');
}
function cellAt(svg,e){const pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const p=pt.matrixTransform(svg.getScreenCTM().inverse());return pixelToHex(p.x,p.y);}
return {draw,cellAt,key,owner};
})();
