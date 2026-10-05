// The board both views draw with: geometry and marks only. Every rule-of-the-game fact arrives from the server.
window.Hex = (function () {
  'use strict';
  const SQ3 = Math.sqrt(3), X = (q, r) => SQ3 * (q + r / 2), Y = (q, r) => 1.5 * r;
  const key = c => c[0] + ',' + c[1];
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

  // The smallest area a board shows, so a few stones are never drawn huge: about 15 cells across, 13 rows down.
  const MIN_W = SQ3 * 15, MIN_H = 1.5 * 13;
  // Past this many cells in view the empty grid is left out and only stones and marks are drawn.
  const MAX_CELLS = 20000;

  /* The frame's box, padded and grown to the minimum area. */
  function baseBox(frame) {
    const xs = frame.map(c => X(c[0], c[1])), ys = frame.map(c => Y(c[0], c[1])), pad = 2.6;
    let x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
    if (x1 - x0 < MIN_W) { const m = (x0 + x1) / 2; x0 = m - MIN_W / 2; x1 = m + MIN_W / 2; }
    if (y1 - y0 < MIN_H) { const m = (y0 + y1) / 2; y0 = m - MIN_H / 2; y1 = m + MIN_H / 2; }
    return { x0, y0, w: x1 - x0, h: y1 - y0 };
  }

  /* scene: {moves, owners, turns, ply, frame, numbers, heat:[{c, rel, label, cls}], win, block, ghosts:[{c, label}], winLine, focus};
     owners and turns come from the server. The view (pan and zoom) lives on the svg and survives redraws. */
  function draw(svg, sc) {
    svg._scene = sc;
    const frame = sc.frame && sc.frame.length ? sc.frame : (sc.moves.length ? sc.moves : [[0, 0]]);
    const base = baseBox(frame), v = svg._view || { k: 1, dx: 0, dy: 0 };
    const w = base.w / v.k, h = base.h / v.k;
    const x0 = base.x0 + (base.w - w) / 2 + v.dx, y0 = base.y0 + (base.h - h) / 2 + v.dy, x1 = x0 + w, y1 = y0 + h;
    svg.setAttribute('viewBox', `${x0} ${y0} ${w} ${h}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const out = [], placed = new Set();
    for (let i = 0; i < sc.ply; i++) placed.add(key(sc.moves[i]));
    // The grid fills what the element shows: the viewBox widened to the element's aspect, as `meet` letterboxes it.
    const el = svg.getBoundingClientRect(), aspect = el.width > 0 && el.height > 0 ? el.width / el.height : w / h;
    const vw = Math.max(w, h * aspect), vh = Math.max(h, w / aspect), cx = x0 + w / 2, cy = y0 + h / 2;
    const gx0 = cx - vw / 2 - 1, gx1 = cx + vw / 2 + 1, gy0 = cy - vh / 2 - 1, gy1 = cy + vh / 2 + 1;
    if (vw * vh / (1.5 * SQ3) <= MAX_CELLS) {
      for (let r = Math.floor(gy0 / 1.5); r <= Math.ceil(gy1 / 1.5); r++) {
        for (let q = Math.floor(gx0 / SQ3 - r / 2); q <= Math.ceil(gx1 / SQ3 - r / 2); q++) {
          const x = X(q, r);
          if (x >= gx0 && x <= gx1) out.push(hex('cell', [q, r], .95, `data-c="${q},${r}"`));
        }
      }
    }
    (sc.heat || []).forEach(h => { if (!placed.has(key(h.c))) out.push(hex(h.cls || 'heat', h.c, .26 + .56 * Math.sqrt(Math.max(0, Math.min(1, h.rel))))); });
    (sc.block || []).forEach(c => out.push(hex('block', c, .8)));
    (sc.win || []).forEach(c => { out.push(hex('wincell', c, .8)); out.push(hex('windot', c, .17)); });
    (sc.ghosts || []).forEach(g => { if (!placed.has(key(g.c))) out.push(hex('ghost', g.c, .86)); });
    for (let i = 0; i < sc.ply; i++) out.push(hex(`s${sc.owners[i] + 1}`, sc.moves[i], .84));
    if (sc.winLine) {
      sc.winLine.forEach(c => out.push(hex('winrim', c, .84)));
      out.push(`<polyline class="winline" points="${sc.winLine.map(c => X(c[0], c[1]).toFixed(3) + ',' + Y(c[0], c[1]).toFixed(3)).join(' ')}"/>`);
    }
    const lastTurn = sc.ply ? sc.turns[sc.ply - 1] : 0;
    for (let i = 0; i < sc.ply; i++) {
      const c = sc.moves[i], o = sc.owners[i] + 1, isLast = sc.turns[i] === lastTurn;
      if (sc.numbers) {
        out.push(`<text class="n${o}" x="${X(c[0], c[1]).toFixed(3)}" y="${Y(c[0], c[1]).toFixed(3)}">${sc.turns[i]}</text>`);
        if (isLast) out.push(hex(`ltr${o}`, c, .62));
      } else if (isLast) out.push(hex(`lt${o}`, c, .2));
    }
    const numbered = new Set((sc.ghosts || []).filter(g => g.label).map(g => key(g.c)));
    (sc.heat || []).forEach(h => {
      if (h.label && !placed.has(key(h.c)) && !numbered.has(key(h.c))) out.push(`<text class="heatnum" x="${X(h.c[0], h.c[1]).toFixed(3)}" y="${Y(h.c[0], h.c[1]).toFixed(3)}">${h.label}</text>`);
    });
    (sc.ghosts || []).forEach(g => {
      if (g.label && !placed.has(key(g.c))) out.push(`<text class="ghostnum" x="${X(g.c[0], g.c[1]).toFixed(3)}" y="${Y(g.c[0], g.c[1]).toFixed(3)}">${g.label}</text>`);
    });
    if (sc.focus) out.push(hex('focus', sc.focus, .95));
    out.push('<polygon class="hover" points=""/>');
    svg.innerHTML = out.join('');
  }

  /* Pan by dragging, zoom with the wheel or the +, − and Fit buttons; a drag never counts as a click on the board.
     A touch drag pans only once zoomed, so at Fit a swipe still scrolls the page. */
  function viewport(svg, controls) {
    svg._view = { k: 1, dx: 0, dy: 0 };
    let frame = 0;
    const redraw = () => {
      const v = svg._view; svg.classList.toggle('zoomed', v.k !== 1 || v.dx !== 0 || v.dy !== 0);
      if (svg._scene && !frame) frame = requestAnimationFrame(() => { frame = 0; draw(svg, svg._scene); });
    };
    const unit = () => { const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal; return Math.max(vb.width / r.width, vb.height / r.height); };
    const zoom = (f, ex, ey) => {
      const v = svg._view, k = Math.min(12, Math.max(0.25, v.k * f));
      if (ex !== undefined) {
        const pt = svg.createSVGPoint(); pt.x = ex; pt.y = ey;
        const p = pt.matrixTransform(svg.getScreenCTM().inverse()), vb = svg.viewBox.baseVal;
        const cx = vb.x + vb.width / 2, cy = vb.y + vb.height / 2, s = 1 - v.k / k;
        v.dx += (p.x - cx) * s; v.dy += (p.y - cy) * s;
      }
      v.k = k; redraw();
    };
    svg.addEventListener('wheel', e => {
      if (!e.deltaY) return;
      e.preventDefault();
      const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
      zoom(Math.exp(-Math.max(-300, Math.min(300, px)) * 0.0015), e.clientX, e.clientY);
    }, { passive: false });
    let drag = null;
    svg.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, moved: false }; });
    svg.addEventListener('pointermove', e => {
      if (!drag || !(e.buttons & 1)) return;
      const ddx = e.clientX - drag.x, ddy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(ddx, ddy) < 5) return;
      if (!drag.moved) { drag.moved = true; svg.setPointerCapture(e.pointerId); svg.classList.add('panning'); }
      const u = unit(); svg._view.dx -= ddx * u; svg._view.dy -= ddy * u; drag.x = e.clientX; drag.y = e.clientY; redraw();
    });
    const end = () => { if (drag && drag.moved) { svg._dragged = true; setTimeout(() => { svg._dragged = false; }, 0); } drag = null; svg.classList.remove('panning'); };
    svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
    const fit = () => { svg._view = { k: 1, dx: 0, dy: 0 }; svg.classList.remove('zoomed'); };
    if (controls) {
      controls.hidden = false;
      controls.querySelector('[data-zoom="in"]').onclick = () => zoom(1.25);
      controls.querySelector('[data-zoom="out"]').onclick = () => zoom(1 / 1.25);
      controls.querySelector('[data-zoom="fit"]').onclick = () => { fit(); redraw(); };
    }
    return { reset: fit };
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

  return { X, Y, key, NAME, fmtC, hexPts, draw, viewport, cellAt, hoverAt, trace, glyph, theme };
})();
