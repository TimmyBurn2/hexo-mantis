// The Analyzer's script: place stones, walk the line, switch nets and lenses. Every reading arrives composed by the server.
(function () {
  'use strict';
  const H = window.Hex, $ = id => document.getElementById(id);
  const S = JSON.parse($('state').textContent);
  const game = S.game ? S.game.moves : [];
  let moves = S.moves.slice(), redo = [], panel = S.panel, lens = 'net', num = false, tac = true, seq = 0;
  let a = panel ? panel.a_id : null, b = panel ? panel.b_id : null;
  const svg = document.querySelector('.boardwrap svg');

  const owners = n => Array.from({ length: n }, (_, i) => (i + 1 >> 1) % 2);
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  const pct = x => `${Math.round(x * 100)} %`;

  function heat() {
    if (!panel) return [];
    const rows = panel.lens[lens] || [];
    if (lens === 'diff') {
      const top = Math.max(1e-9, ...rows.map(r => Math.abs(r[2])));
      return rows.filter(r => Math.abs(r[2]) / top >= .04).map(r => ({ c: [r[0], r[1]], rel: Math.abs(r[2]) / top, cls: r[2] >= 0 ? 'heat' : 'heat2', label: '' }));
    }
    const top = Math.max(1e-9, ...rows.map(r => r[2]));
    return rows.map((r, i) => ({ c: [r[0], r[1]], rel: r[2] / top, label: i < 3 && r[2] >= .03 ? String(Math.round(r[2] * 100)) : '' }));
  }

  function draw() {
    const t = panel && tac ? panel.tactics : null;
    H.draw(svg, { moves, owners: owners(moves.length), ply: moves.length, frame: game.length ? game : moves, numbers: num, heat: heat(),
      win: t && t.cls === 'win' ? t.cells : [], block: t && t.cls === 'block' ? t.cells : [], ghost: panel && lens === 'net' ? panel.first : null });
  }

  function source(ctx) {
    const line = $('line');
    if (!line || !ctx) return;
    line.textContent = ctx.on_line ? 'On the game’s line' : `Variation, ${ctx.off} stone${ctx.off === 1 ? '' : 's'} off the game`;
    line.classList.toggle('var', !ctx.on_line);
  }

  function readout() {
    const box = $('read');
    if (!panel) return;
    box.querySelector('#verdict').innerHTML = panel.verdict;  // server-composed, record strings escaped there
    const chances = $('chances'); chances.replaceChildren();
    panel.chances.forEach(c => {
      const row = el('div', 'wc'), who = el('span', `who ${c.cls}`); who.append(el('i'), c.label);
      if (c.light == null) { row.append(who, el('span', 'muted', 'not read'), el('span'), el('span')); }
      else { const bar = el('span', 'bar'), f = el('i'); f.style.width = (c.light * 100) + '%'; bar.append(f); row.append(who, el('span', null, `Light ${pct(c.light)}`), bar, el('span', null, `${pct(1 - c.light)} Dark`)); }
      chances.append(row);
    });
    const tb = $('cands').tBodies[0]; tb.replaceChildren();
    panel.rows.forEach(r => {
      const tr = el('tr'); tr.dataset.c = `${r[0]},${r[1]}`;
      tr.append(el('td', 'num', H.fmtC(r)), el('td', 'num', pct(r[2])));
      if (panel.b) tr.append(el('td', 'num', pct(r[3])));
      tr.append(el('td', 'num', r[4] == null ? '—' : pct(r[4])));
      const tags = el('td'); r[5].forEach(t => tags.append(el('span', `tag ${t}`, t))); tr.append(tags);
      tr.onmouseenter = () => H.hoverAt(svg, r); tr.onmouseleave = () => H.hoverAt(svg, null);
      tr.onclick = () => place([r[0], r[1]]);
      tb.append(tr);
    });
    const note = { net: `Where ${panel.a} wants to play before any search: its policy, bigger is more.`,
      search: panel.search_source ? `Visits of ${panel.search_source}, bigger is more.` : 'No search here yet: run one below, or step onto the game’s line.',
      diff: `Blue where ${panel.a} puts more weight, orange where ${panel.b} does.` }[lens];
    $('lensnote').textContent = note;
    $('where').innerHTML = panel.where;  // server-composed
    document.querySelectorAll('#engines .eng').forEach(e => {
      const role = e.dataset.id === a ? 'reading' : e.dataset.id === b ? 'compare' : '';
      e.setAttribute('aria-pressed', String(Boolean(role))); e.classList.toggle('c1', role === 'reading'); e.classList.toggle('c2', role === 'compare');
      e.querySelector('small').textContent = role;
    });
  }

  async function read(extra = {}) {
    if (!a) { draw(); return; }
    const mine = ++seq, started = Date.now();
    const status = $('deeper'); const tick = setInterval(() => { if (status) status.textContent = `Reading… ${((Date.now() - started) / 1000).toFixed(1)} s`; }, 200);
    try {
      const r = await fetch('/api/read', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ a, b, moves: moves.map(c => c.join(',')).join(';'), run: S.run, g: S.g, seq: mine, client: 'page', ...extra }) });
      const out = await r.json();
      if (mine !== seq) return;
      if (!r.ok || !out.ok) { if (status) status.textContent = out.refused || `refused (${r.status})`; draw(); return; }
      panel = out.panel; source(out.context); readout(); draw();
      if (status) status.textContent = deeper(out.record, extra);
    } finally { clearInterval(tick); }
  }

  function deeper(rec, extra) {
    if (extra.symmetry && rec.symmetry && rec.symmetry.n) return `Symmetry over ${rec.symmetry.n} maps: value spread ${rec.symmetry.spread}, first choice agrees ${rec.symmetry.argmax_agreement}.`;
    if (extra.sims && rec.search && rec.search.sims) return `Search ${rec.search.sims} sims in ${(rec.search.ms / 1000).toFixed(1)} s: root value ${rec.search.root_value}, its choice ${H.fmtC(rec.search.argmax)}.`;
    return '';
  }

  function place(c) {
    if (moves.some(m => m[0] === c[0] && m[1] === c[1])) return;
    moves.push(c); redo = []; read();
  }

  function wire() {
    svg.addEventListener('click', e => place(H.cellAt(svg, e)));
    svg.addEventListener('pointermove', e => H.hoverAt(svg, H.cellAt(svg, e)));
    svg.addEventListener('pointerleave', () => H.hoverAt(svg, null));
    $('prev').onclick = () => { if (moves.length) { redo.push(moves.pop()); read(); } };
    $('next').onclick = () => {
      if (redo.length) moves.push(redo.pop());
      else if (moves.length < game.length && game.slice(0, moves.length).every((m, i) => m[0] === moves[i][0] && m[1] === moves[i][1])) moves.push(game[moves.length]);
      else return;
      read();
    };
    $('undo').onclick = () => { if (moves.length) { moves.pop(); redo = []; read(); } };
    $('lNum').onclick = () => { num = !num; $('lNum').setAttribute('aria-pressed', String(num)); draw(); };
    $('lTac').onclick = () => { tac = !tac; $('lTac').setAttribute('aria-pressed', String(tac)); draw(); };
    const seg = $('lens');
    if (seg) seg.onclick = e => { const btn = e.target.closest('button'); if (!btn) return; lens = btn.dataset.lens; seg.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === btn))); readout(); draw(); };
    document.querySelectorAll('#engines .eng').forEach(chip => chip.onclick = () => {
      if (chip.dataset.id === a) return;
      b = a; a = chip.dataset.id; read();
    });
    if ($('search')) $('search').onclick = () => read({ sims: +$('sims').value });
    if ($('sym')) $('sym').onclick = () => read({ symmetry: true });
    if ($('copy')) $('copy').onclick = () => { navigator.clipboard && navigator.clipboard.writeText(moves.map(c => c.join(',')).join(';')); $('copy').textContent = 'Copied'; };
    document.addEventListener('keydown', e => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === 'ArrowLeft') $('prev').click(); else if (e.key === 'ArrowRight') $('next').click();
      else if (e.key === 'u') $('undo').click();
    });
  }

  H.theme(); wire(); draw();
})();
