// The Analyzer's script: place stones, walk the line, switch nets and lenses. Every reading and turn fact arrives from the server.
(function () {
  'use strict';
  const H = window.Hex, $ = id => document.getElementById(id);
  const S = JSON.parse($('state').textContent);
  const game = S.game ? S.game.moves : [];
  const client = 'p' + Math.random().toString(36).slice(2) + Date.now().toString(36);
  const svg = document.querySelector('.boardwrap svg');
  // `shown` is the position the current panel was read for; `moves` is where the reader is going.
  let shown = { moves: S.moves.slice(), owners: S.owners, turns: S.turn_of }, panel = S.panel;
  let moves = S.moves.slice(), redo = [], lens = 'net', num = false, tac = true, seq = 0;
  let a = panel ? panel.a_id : null, b = panel ? panel.b_id : null;

  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  const pct = x => `${Math.round(x * 100)} %`;
  const same = (x, y) => x.length === y.length && x.every((m, i) => m[0] === y[i][0] && m[1] === y[i][1]);

  function heat() {
    if (!panel) return [];
    const rows = panel.lens[lens] || [];
    if (lens === 'diff') {
      const top = Math.max(1e-9, ...rows.map(r => Math.abs(r[2])));
      return rows.filter(r => Math.abs(r[2]) / top >= .04).map(r => ({ c: [r[0], r[1]], rel: Math.abs(r[2]) / top, cls: r[2] >= 0 ? 'heat' : 'heat2', label: '' }));
    }
    const top = Math.max(1e-9, ...rows.map(r => r[2]));
    return rows.map((r, i) => ({ c: [r[0], r[1]], rel: r[2] / top, label: i < 3 && r[2] >= .03 ? `${Math.round(r[2] * 100)}%` : '' }));
  }

  function draw() {
    const t = panel && tac ? panel.tactics : null;
    H.draw(svg, { moves: shown.moves, owners: shown.owners, turns: shown.turns, ply: shown.moves.length,
      frame: game.length ? game : shown.moves, numbers: num, heat: heat(), win: t && t.cls === 'win' ? t.cells : [],
      block: t && t.cls === 'block' ? t.cells : [],
      ghosts: panel && lens !== 'diff' ? panel.turn.ghosts.map(g => ({ c: [g[0], g[1]], label: g[2], cls: g[3] })) : [] });
  }

  function source(ctx) {
    const line = $('line'), back = $('back');
    if (!line || !ctx) return;
    line.textContent = ctx.on_line ? 'On the game’s line' : `Variation, ${ctx.off} stone${ctx.off === 1 ? '' : 's'} off the game`;
    line.classList.toggle('var', !ctx.on_line);
    if (back) back.hidden = ctx.on_line;
  }

  function readout() {
    if (!panel) return;
    $('verdict').innerHTML = panel.verdict;  // server-composed; record strings escaped there
    $('gameline').innerHTML = panel.game_line || '';
    const chances = $('chances'); chances.replaceChildren();
    panel.chances.forEach(c => {
      const row = el('div', 'wc'), who = el('span', `who ${c.cls}`); who.append(el('i'), c.label);
      if (c.light == null) row.append(who, el('span', 'muted', 'not read'), el('span'), el('span'));
      else { const bar = el('span', 'bar'), f = el('i'); f.style.width = (c.light * 100) + '%'; bar.append(f); row.append(who, el('span', null, `Light ${pct(c.light)}`), bar, el('span', null, `${pct(1 - c.light)} Dark`)); }
      chances.append(row);
    });
    if (!panel.b && lens === 'diff') lens = 'net';
    const seg = $('lens'); seg.replaceChildren();
    [['net', 'Net'], ['search', 'Search']].concat(panel.b ? [['diff', 'Difference']] : []).forEach(([k, label]) => {
      const btn = el('button', null, label); btn.type = 'button'; btn.dataset.lens = k; btn.setAttribute('aria-pressed', String(k === lens));
      btn.onclick = () => { lens = k; readout(); draw(); };
      seg.append(btn);
    });
    const head = el('tr'); [['Cell', ''], [panel.a, 'c1']].concat(panel.b ? [[panel.b, 'c2']] : [], [[panel.search_head, ''], ['', '']])
      .forEach(([h, cls]) => { const th = el('th'); th.append(cls ? el('span', cls, h) : h); head.append(th); });
    $('cands').tHead.replaceChildren(head);
    const tb = $('cands').tBodies[0]; tb.replaceChildren();
    panel.rows.forEach(r => {
      const tr = el('tr'); tr.dataset.c = `${r[0]},${r[1]}`;
      tr.append(el('td', 'num', H.fmtC(r)), el('td', 'pct', pct(r[2])));
      if (panel.b) tr.append(el('td', 'pct', r[3] == null ? '' : pct(r[3])));
      tr.append(el('td', 'pct', r[4] == null ? '' : pct(r[4])));
      const tags = el('td'); r[5].forEach(t => tags.append(el('span', `tag ${t}`, t))); tr.append(tags);
      tr.onmouseenter = () => H.hoverAt(svg, r); tr.onmouseleave = () => H.hoverAt(svg, null);
      tr.onclick = () => place([r[0], r[1]]);
      tb.append(tr);
    });
    $('lensnote').textContent = { net: `Policy before search. ${panel.turn.note}`,
      search: (panel.search_source ? `Visits of ${panel.search_source}.` : 'No search here yet. Run one below, or step onto the game’s line.') + ` ${panel.turn.note}`,
      diff: `Blue: ${panel.a} puts more weight there. Orange: ${panel.b} does.` }[lens];
    $('where').innerHTML = panel.where;  // server-composed
    if ($('netA')) { $('netA').value = a || ''; $('netB').value = b || ''; }
    const k = (g, t) => { const s = el('span'); s.innerHTML = g; s.append(t); return s; };
    const shared = panel.turn.ghosts.some(g => g[3] === 'gab');
    $('keys').replaceChildren(...[panel.turn.a.length ? k(H.glyph('ga'), panel.a) : null,
      panel.b && panel.turn.b.length ? k(H.glyph('gb'), panel.b) : null, shared ? k(H.glyph('gab'), 'both') : null].filter(Boolean));
  }

  async function read(extra = {}) {
    const status = $('deeper');
    if (!a) { if (status) status.textContent = 'Pick a net to read with.'; return; }
    const mine = ++seq, started = Date.now(), want = moves.slice();
    const tick = setInterval(() => { if (status && mine === seq) status.textContent = `Reading… ${((Date.now() - started) / 1000).toFixed(1)} s`; }, 200);
    try {
      const r = await fetch('/api/read', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ a, b, moves: want.map(c => c.join(',')).join(';'), run: S.run, g: S.g, seq: mine, client,
          line: !S.g && game.length ? game.map(c => c.join(',')).join(';') : null, ...extra }) });
      const out = await r.json();
      if (out.superseded || mine !== seq) return;
      if (!r.ok || !out.ok) {
        moves = shown.moves.slice(); redo = [];
        if (status) status.textContent = `Refused: ${out.refused || r.status}. The board is back at the last position read.`;
        draw(); return;
      }
      panel = out.panel; shown = { moves: want, owners: panel.owners, turns: panel.turn_of };
      source(out.context); readout(); draw();
      if (status) status.textContent = deeper(out.record, extra);
    } finally { clearInterval(tick); }
  }

  function deeper(rec, extra) {
    if (extra.symmetry && rec.symmetry && rec.symmetry.n) return `Symmetry over ${rec.symmetry.n} maps: value spread ${rec.symmetry.spread}, first choice agrees ${rec.symmetry.argmax_agreement}.`;
    if (extra.sims && rec.search && rec.search.sims) {
      const more = panel.turn.second_ms ? ` Its second stone took another search, ${(panel.turn.second_ms / 1000).toFixed(1)} s.` : '';
      return `Search ${rec.search.sims} sims in ${(rec.search.ms / 1000).toFixed(1)} s: root value ${rec.search.root_value}, its choice ${H.fmtC(rec.search.argmax)}.${more}`;
    }
    return '';
  }

  function place(c) {
    if (!a || moves.some(m => m[0] === c[0] && m[1] === c[1])) return;
    moves.push(c); redo = []; read();
  }

  function wire() {
    H.viewport(svg, document.querySelector('.boardwrap .zoom'));
    svg.addEventListener('click', e => { if (!svg._dragged) place(H.cellAt(svg, e)); });
    svg.addEventListener('pointermove', e => H.hoverAt(svg, H.cellAt(svg, e)));
    svg.addEventListener('pointerleave', () => H.hoverAt(svg, null));
    // Prev and next walk whole turns, by the server's turn starts: back to the start before here, forward one on the line.
    $('prev').onclick = () => {
      if (!moves.length) return;
      const target = Math.max(0, ...S.starts.filter(p => p < moves.length));
      redo.push(moves.slice(target)); moves = moves.slice(0, target); read();
    };
    $('next').onclick = () => {
      if (redo.length) moves = moves.concat(redo.pop());
      else if (moves.length < game.length && same(game.slice(0, moves.length), moves)) {
        moves = game.slice(0, Math.min(game.length, ...S.starts.filter(p => p > moves.length)));
      } else return;
      read();
    };
    // Back to the start a turn at a time onto the redo stack, so Next walks back over a typed position or a variation.
    $('first').onclick = () => {
      if (!moves.length) return;
      while (moves.length) { const target = Math.max(0, ...S.starts.filter(p => p < moves.length)); redo.push(moves.slice(target)); moves = moves.slice(0, target); }
      read();
    };
    $('last').onclick = () => { if (game.length && !same(moves, game)) { moves = game.slice(); redo = []; read(); } };
    $('undo').onclick = () => { if (moves.length) { moves.pop(); redo = []; read(); } };
    if ($('back')) $('back').onclick = () => { moves = game.slice(0, S.ply); redo = []; read(); };
    $('lNum').onclick = () => { num = !num; $('lNum').setAttribute('aria-pressed', String(num)); draw(); };
    $('lTac').onclick = () => { tac = !tac; $('lTac').setAttribute('aria-pressed', String(tac)); draw(); };
    document.querySelectorAll('#lens button').forEach(btn => btn.onclick = () => { lens = btn.dataset.lens; readout(); draw(); });
    if ($('netA')) {
      $('netA').onchange = () => { a = $('netA').value; if (b === a) b = null; read(); };
      $('netB').onchange = () => { b = $('netB').value || null; if (b === a) b = null; read(); };
    }
    if ($('search')) $('search').onclick = () => read({ sims: +$('sims').value });
    if ($('sym')) $('sym').onclick = () => read({ symmetry: true });
    // An import reads with the nets now chosen.
    if ($('importform')) $('importform').addEventListener('submit', () => {
      const form = $('importform');
      [['a', a], ['b', b]].forEach(([k, v]) => {
        let input = form.elements[k];
        if (!input) { input = el('input'); input.type = 'hidden'; input.name = k; form.prepend(input); }
        input.value = v || '';
      });
    });
    if ($('copy')) $('copy').onclick = () => {
      const text = panel && panel.htttx ? panel.htttx : shown.moves.map(c => c.join(',')).join(';');
      if (navigator.clipboard) navigator.clipboard.writeText(text);
      $('copy').textContent = !(panel && panel.htttx) ? 'Copied' : panel.htttx_moved ? 'Copied as htttx, first stone moved to the origin' : 'Copied as htttx';
    };
    document.addEventListener('keydown', e => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === 'ArrowLeft') $('prev').click(); else if (e.key === 'ArrowRight') $('next').click();
      else if (e.key === 'Home') $('first').click(); else if (e.key === 'End') $('last').click();
      else if (e.key === 'u') $('undo').click();
    });
  }

  H.theme(); wire(); if (panel) readout(); draw();
})();
