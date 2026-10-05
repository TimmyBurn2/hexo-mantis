// The board both views draw with: geometry and marks only. Every rule-of-the-game fact arrives from the server.
window.Hex = (function () {
  'use strict';
  const SQ3 = Math.sqrt(3), X = (q, r) => SQ3 * (q + r / 2), Y = (q, r) => 1.5 * r;
  const key = c => c[0] + ',' + c[1];
  const turnOf = p => (p + 1 >> 1) + 1;
  const NAME = ['Light', 'Dark'];
  const MINUS = '−', fmtC = c => `(${String(c[0]).replace('-', MINUS)}, ${String(c[1]).replace('-', MINUS)})`;

  function hexPts(cx, cy, rad) {
    let s = '';
    for (let i = 0; i < 6; i++) { const a = Math.PI / 180 * (60 * i - 30); s += (cx + rad * Math.cos(a)).toFixed(3) + ',' + (cy + rad * Math.sin(a)).toFixed(3) + ' '; }
    return s;
  }
  const hex = (cls, c, rad, extra = '') => `<polygon class="${cls}" points="${hexPts(X(c[0], c[1]), Y(c[0], c[1]), rad)}" ${extra}/>`;

  function pixelToHex(x, y) {
    const qf = SQ3 / 3 * x - y / 3, rf = 2 / 3 * y, sf = -qf - rf;
    let q = Math.round(qf), r = Math.round(rf); const s = Math.round(sf);
    const dq = Math.abs(q - qf), dr = Math.abs(r - rf), ds = Math.abs(s - sf);
    if (dq > dr && dq > ds) q = -r - s; else if (dr > ds) r = -q - s;
    return [q, r];
  }

  /* scene: {moves, owners, ply, frame, numbers, heat:[{c, rel, label, cls}], win:[c], block:[c], ghost:c, winLine:[c], focus:c} */
  function draw(svg, sc) {
    const frame = sc.frame && sc.frame.length ? sc.frame : (sc.moves.length ? sc.moves : [[0, 0]]);
    const xs = frame.map(c => X(c[0], c[1])), ys = frame.map(c => Y(c[0], c[1])), pad = 2.6;
    const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
    svg.setAttribute('viewBox', `${x0} ${y0} ${x1 - x0} ${y1 - y0}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const out = [], placed = new Set();
    for (let i = 0; i < sc.ply; i++) placed.add(key(sc.moves[i]));
    const qs = frame.map(c => c[0]), rs = frame.map(c => c[1]);
    for (let r = Math.min(...rs) - 4; r <= Math.max(...rs) + 4; r++) for (let q = Math.min(...qs) - 8; q <= Math.max(...qs) + 8; q++) {
      const x = X(q, r), y = Y(q, r);
      if (x >= x0 - 1 && x <= x1 + 1 && y >= y0 - 1 && y <= y1 + 1) out.push(hex('cell', [q, r], .95, `data-c="${q},${r}"`));
    }
    (sc.heat || []).forEach(h => { if (!placed.has(key(h.c))) out.push(hex(h.cls || 'heat', h.c, .26 + .56 * Math.sqrt(Math.max(0, Math.min(1, h.rel))))); });
    (sc.block || []).forEach(c => out.push(hex('block', c, .8)));
    (sc.win || []).forEach(c => { out.push(hex('wincell', c, .8)); out.push(hex('windot', c, .17)); });
    if (sc.ghost && !placed.has(key(sc.ghost))) out.push(hex('ghost', sc.ghost, .86));
    for (let i = 0; i < sc.ply; i++) out.push(hex(`s${sc.owners[i] + 1}`, sc.moves[i], .84));
    if (sc.winLine) {
      sc.winLine.forEach(c => out.push(hex('winrim', c, .84)));
      out.push(`<polyline class="winline" points="${sc.winLine.map(c => X(c[0], c[1]).toFixed(3) + ',' + Y(c[0], c[1]).toFixed(3)).join(' ')}"/>`);
    }
    const lastTurn = sc.ply ? turnOf(sc.ply - 1) : 0;
    for (let i = 0; i < sc.ply; i++) {
      const c = sc.moves[i], o = sc.owners[i] + 1, isLast = turnOf(i) === lastTurn;
      if (sc.numbers) {
        out.push(`<text class="n${o}" x="${X(c[0], c[1]).toFixed(3)}" y="${Y(c[0], c[1]).toFixed(3)}">${turnOf(i)}</text>`);
        if (isLast) out.push(hex(`ltr${o}`, c, .62));
      } else if (isLast) out.push(hex(`lt${o}`, c, .2));
    }
    (sc.heat || []).forEach(h => {
      if (h.label && !placed.has(key(h.c))) out.push(`<text class="heatnum" x="${X(h.c[0], h.c[1]).toFixed(3)}" y="${Y(h.c[0], h.c[1]).toFixed(3)}">${h.label}</text>`);
    });
    if (sc.focus) out.push(hex('focus', sc.focus, .95));
    out.push('<polygon class="hover" points=""/>');
    svg.innerHTML = out.join('');
  }

  function cellAt(svg, e) {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    return pixelToHex(p.x, p.y);
  }
  function hoverAt(svg, c) { const h = svg.querySelector('.hover'); if (h) h.setAttribute('points', c ? hexPts(X(c[0], c[1]), Y(c[0], c[1]), .95) : ''); }

  /* The strip: Light's chance fills from the bottom, Dark's is the rest; a costly turn is a tick along the top. */
  function trace(host, opts) {
    const W = host.clientWidth || 600, H = 84, m = { t: 10, b: 16 }, ih = H - m.t - m.b, T = Math.max(1, opts.turns);
    const X1 = t => (t - 1) / Math.max(1, T - 1) * W, Y1 = v => m.t + (1 - v) * ih, pts = opts.pts;
    let g = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Win chance by turn">`;
    g += `<rect x="0" y="${m.t}" width="${W}" height="${ih}" class="strip-dark" rx="3"/>`;
    if (pts.length > 1) {
      const line = pts.map(p => `${X1(p.turn).toFixed(1)},${Y1(p.light).toFixed(1)}`).join(' ');
      g += `<polygon class="strip-light" points="${X1(pts[0].turn).toFixed(1)},${m.t + ih} ${line} ${X1(pts[pts.length - 1].turn).toFixed(1)},${m.t + ih}"/>`;
    }
    g += `<line class="strip-mid" x1="0" x2="${W}" y1="${Y1(.5)}" y2="${Y1(.5)}"/>`;
    pts.forEach(p => { if (p.cost != null && p.cost >= .3) g += `<path class="${p.cost >= .5 ? 'tick-red' : 'tick-amber'}" d="M${X1(p.turn) - 4},${m.t - 9} h8 l-4,6z"/>`; });
    for (let t = 1; t <= T; t++) if (t === 1 || t === T || t % 5 === 0) g += `<text class="tick" x="${X1(t)}" y="${H - 3}" text-anchor="${t === 1 ? 'start' : t === T ? 'end' : 'middle'}">${t}</text>`;
    g += `<line class="cur" x1="0" x2="0" y1="${m.t - 2}" y2="${m.t + ih + 2}"/><rect class="hit" x="0" y="0" width="${W}" height="${H}" fill="transparent"/></svg><div class="tip"></div>`;
    host.innerHTML = g;
    const svg = host.querySelector('svg'), cur = svg.querySelector('.cur'), tip = host.querySelector('.tip');
    const setCur = t => { const x = X1(Math.min(T, Math.max(1, t))); cur.setAttribute('x1', x); cur.setAttribute('x2', x); };
    const turnAt = e => { const r = svg.getBoundingClientRect(); return Math.round(1 + (e.clientX - r.left) / r.width * (T - 1)); };
    let drag = false;
    svg.addEventListener('pointerdown', e => { drag = true; svg.setPointerCapture(e.pointerId); opts.onTurn(turnAt(e)); });
    svg.addEventListener('pointerup', () => { drag = false; });
    svg.addEventListener('pointermove', e => {
      const t = turnAt(e); if (drag) opts.onTurn(t);
      const p = pts.find(q => q.turn === t), r = svg.getBoundingClientRect();
      tip.style.left = (X1(t) / W * r.width) + 'px'; tip.style.opacity = 1;
      tip.textContent = p ? `Turn ${t}: Light ${Math.round(p.light * 100)} %` + (p.cost >= .3 ? `, cost ${Math.round(p.cost * 100)}` : '') : `Turn ${t}: no search recorded`;
    });
    svg.addEventListener('pointerleave', () => { tip.style.opacity = 0; });
    setCur(opts.turn);
    return { setCur };
  }

  const glyph = (cls, size = 14) => `<svg class="glyph" width="${size}" height="${size}" viewBox="-1.1 -1.1 2.2 2.2" aria-hidden="true"><polygon class="${cls}" points="${hexPts(0, 0, 1)}"/></svg>`;

  function theme() {
    const btn = document.getElementById('theme');
    if (!btn) return;
    const label = () => { btn.textContent = document.documentElement.dataset.theme === 'light' ? 'Dark' : 'Light'; };
    btn.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('mantis-theme', next); } catch (e) { /* storage refused: the toggle still works */ }
      label();
    });
    label();
  }

  return { X, Y, key, turnOf, NAME, fmtC, hexPts, draw, cellAt, hoverAt, trace, glyph, theme };
})();
