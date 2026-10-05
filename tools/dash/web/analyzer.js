// The Analyzer's script: place stones, walk the line, switch nets and lenses. Every reading and turn fact arrives from the server.
(function () {
  'use strict';
  const H = window.Hex, $ = id => document.getElementById(id);
  const S = JSON.parse($('state').textContent);
  const game = S.game ? S.game.moves : [];
  const client = 'p' + Math.random().toString(36).slice(2) + Date.now().toString(36);
  const svg = document.querySelector('.boardwrap svg');
  // `shown` is the position the current panel was read for; `moves` is where the reader is going.
  let shown = { moves: S.moves.slice(), owners: S.owners, turns: S.turn_of, starts: S.turn_starts }, panel = S.panel;
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
    return rows.map((r, i) => ({ c: [r[0], r[1]], rel: r[2] / top, label: i < 3 && r[2] >= .03 ? String(Math.round(r[2] * 100)) : '' }));
  }

  function draw() {
    const t = panel && tac ? panel.tactics : null;
    H.draw(svg, { moves: shown.moves, owners: shown.owners, turns: shown.turns, ply: shown.moves.length,
      frame: game.length ? game : shown.moves, numbers: num, heat: heat(), win: t && t.cls === 'win' ? t.cells : [],
      block: t && t.cls === 'block' ? t.cells : [],
      ghosts: panel && lens !== 'diff' ? panel.turn.a.map((c, i, all) => ({ c, label: all.length > 1 ? String(i + 1) : '' })) : [] });
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
    [['net', 'Net'], ['search', 'Search']].concat(panel.b ? [['diff', `${panel.a} vs ${panel.b}`]] : []).forEach(([k, label]) => {
      const btn = el('button', null, label); btn.type = 'button'; btn.dataset.lens = k; btn.setAttribute('aria-pressed', String(k === lens));
      btn.onclick = () => { lens = k; readout(); draw(); };
      seg.append(btn);
    });
    const head = el('tr'); ['Cell', panel.a].concat(panel.b ? [panel.b] : [], ['Search', '']).forEach(h => head.append(el('th', null, h)));
    $('cands').tHead.replaceChildren(head);
    const tb = $('cands').tBodies[0]; tb.replaceChildren();
    panel.rows.forEach(r => {
      const tr = el('tr'); tr.dataset.c = `${r[0]},${r[1]}`;
      tr.append(el('td', 'num', H.fmtC(r)), el('td', 'num', pct(r[2])));
      if (panel.b) tr.append(el('td', 'num', r[3] == null ? '—' : pct(r[3])));
      tr.append(el('td', 'num', r[4] == null ? '—' : pct(r[4])));
      const tags = el('td'); r[5].forEach(t => tags.append(el('span', `tag ${t}`, t))); tr.append(tags);
      tr.onmouseenter = () => H.hoverAt(svg, r); tr.onmouseleave = () => H.hoverAt(svg, null);
      tr.onclick = () => place([r[0], r[1]]);
      tb.append(tr);
    });
    $('lensnote').textContent = { net: `Where ${panel.a} wants to play before any search: its policy, bigger is more.`,
      search: panel.search_source ? `Visits of ${panel.search_source}, bigger is more.` : 'No search here yet: run one below, or step onto the game’s line.',
      diff: `Blue where ${panel.a} puts more weight, orange where ${panel.b} does.` }[lens];
    $('where').innerHTML = panel.where;  // server-composed
    document.querySelectorAll('#engines .eng').forEach(e => {
      const role = e.dataset.id === a ? 'reading' : e.dataset.id === b ? 'compare' : '';
      e.setAttribute('aria-pressed', String(Boolean(role))); e.classList.toggle('c1', role === 'reading'); e.classList.toggle('c2', role === 'compare');
      e.querySelector('small').textContent = role;
    });
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
      panel = out.panel; shown = { moves: want, owners: panel.owners, turns: panel.turn_of, starts: panel.turn_starts };
      source(out.context); readout(); draw();
      if (status) status.textContent = deeper(out.record, extra);
    } finally { clearInterval(tick); }
  }

  function deeper(rec, extra) {
    if (extra.symmetry && rec.symmetry && rec.symmetry.n) return `Symmetry over ${rec.symmetry.n} maps: value spread ${rec.symmetry.spread}, first choice agrees ${rec.symmetry.argmax_agreement}.`;
    if (extra.sims && rec.search && rec.search.sims) return `Search ${rec.search.sims} sims in ${(rec.search.ms / 1000).toFixed(1)} s: root value ${rec.search.root_value}, its choice ${H.fmtC(rec.search.argmax)}.`;
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
    // Prev and next walk whole turns: back to the turn's start before this position, forward a turn on the line.
    $('prev').onclick = () => {
      if (!moves.length) return;
      const target = Math.max(0, ...shown.starts.filter(p => p < moves.length));
      redo.push(moves.slice(target)); moves = moves.slice(0, target); read();
    };
    $('next').onclick = () => {
      if (redo.length) moves = moves.concat(redo.pop());
      else if (moves.length < game.length && same(game.slice(0, moves.length), moves)) {
        moves = game.slice(0, Math.min(...S.line_starts.filter(p => p > moves.length)));
      } else return;
      read();
    };
    $('undo').onclick = () => { if (moves.length) { moves.pop(); redo = []; read(); } };
    if ($('back')) $('back').onclick = () => { moves = game.slice(0, S.ply); redo = []; read(); };
    $('lNum').onclick = () => { num = !num; $('lNum').setAttribute('aria-pressed', String(num)); draw(); };
    $('lTac').onclick = () => { tac = !tac; $('lTac').setAttribute('aria-pressed', String(tac)); draw(); };
    document.querySelectorAll('#lens button').forEach(btn => btn.onclick = () => { lens = btn.dataset.lens; readout(); draw(); });
    document.querySelectorAll('#engines .eng').forEach(chip => chip.onclick = () => {
      if (chip.dataset.id === a) return;
      b = a; a = chip.dataset.id; read();
    });
    if ($('search')) $('search').onclick = () => read({ sims: +$('sims').value });
    if ($('sym')) $('sym').onclick = () => read({ symmetry: true });
    if ($('copy')) $('copy').onclick = () => {
      const text = panel && panel.htttx ? panel.htttx : shown.moves.map(c => c.join(',')).join(';');
      if (navigator.clipboard) navigator.clipboard.writeText(text);
      $('copy').textContent = panel && panel.htttx ? 'Copied as htttx' : 'Copied';
    };
    document.addEventListener('keydown', e => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === 'ArrowLeft') $('prev').click(); else if (e.key === 'ArrowRight') $('next').click();
      else if (e.key === 'u') $('undo').click();
    });
  }

  H.theme(); wire(); draw();
})();
