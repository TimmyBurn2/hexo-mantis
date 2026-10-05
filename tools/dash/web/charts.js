// The Run view's script: the theme toggle, the run picker, and a crosshair readout over each server-drawn chart.
(function () {
  'use strict';
  const NBSP = ' ';
  const fmt = v => {
    const a = Math.abs(v);
    const s = a >= 100 ? Math.round(a).toLocaleString('en-GB').replace(/,/g, NBSP) : a >= 10 ? a.toFixed(1) : a >= 1 ? a.toFixed(2) : a.toFixed(3);
    return (v < 0 ? '−' : '') + s;
  };

  function theme() {
    const btn = document.getElementById('theme');
    if (!btn) return;
    const label = () => { const light = document.documentElement.dataset.theme === 'light'; btn.textContent = 'Theme'; btn.setAttribute('aria-label', light ? 'Switch to the dark theme' : 'Switch to the light theme'); };
    btn.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('mantis-theme', next); } catch (e) { /* storage refused: the toggle still works */ }
      label();
    });
    label();
  }

  function picker() {
    document.querySelectorAll('form.pick select').forEach(s => s.addEventListener('change', () => s.form.submit()));
  }

  function crosshair(plot) {
    const blob = plot.querySelector('script.xh');
    const svg = plot.querySelector('svg');
    if (!blob || !svg) return;
    const d = JSON.parse(blob.textContent);
    const m = d.m, iw = d.w - m.l - m.r, ih = d.h - m.t - m.b;
    const X = v => m.l + (v - d.x[0]) / ((d.x[1] - d.x[0]) || 1) * iw;
    const Y = v => m.t + (1 - (v - d.y[0]) / ((d.y[1] - d.y[0]) || 1)) * ih;
    const ns = 'http://www.w3.org/2000/svg';
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('class', 'xh-line'); line.setAttribute('y1', m.t); line.setAttribute('y2', m.t + ih); line.style.opacity = 0;
    const dots = document.createElementNS(ns, 'g');
    svg.append(line, dots);
    const tip = document.createElement('div'); tip.className = 'tip'; plot.appendChild(tip);
    const leave = () => { tip.style.opacity = 0; line.style.opacity = 0; dots.replaceChildren(); };
    svg.addEventListener('pointermove', ev => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * d.w / r.width;
      const xv = d.x[0] + (px - m.l) / iw * (d.x[1] - d.x[0]);
      const rows = []; let snap = null; dots.replaceChildren();
      d.series.forEach(s => {
        if (!s.pts.length) return;
        let best = s.pts[0];
        for (const p of s.pts) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
        if (s.dots && Math.abs(X(best[0]) - px) > 28) return;
        snap = snap === null ? best[0] : snap;
        const c = document.createElementNS(ns, 'circle');
        c.setAttribute('class', 'dot ' + s.cls); c.setAttribute('cx', X(best[0])); c.setAttribute('cy', Y(best[1])); c.setAttribute('r', 4);
        dots.appendChild(c);
        const row = document.createElement('div');
        row.textContent = `${s.name} ${fmt(best[1])}` + (best.length > 3 && best[2] !== null ? ` [${fmt(best[2])}, ${fmt(best[3])}]` : '');
        rows.push(row);
      });
      if (snap === null) { leave(); return; }
      line.setAttribute('x1', X(snap)); line.setAttribute('x2', X(snap)); line.style.opacity = 1;
      const head = document.createElement('div'); head.className = 'k'; head.textContent = `${d.xname} ${fmt(snap)}`;
      tip.replaceChildren(head, ...rows);
      const tx = X(snap) / d.w * r.width;
      tip.style.left = Math.min(r.width - tip.offsetWidth, Math.max(0, tx + 12)) + 'px'; tip.style.opacity = 1;
    });
    svg.addEventListener('pointerleave', leave);
  }

  theme();
  picker();
  document.querySelectorAll('figure.chart .plot').forEach(crosshair);
})();
