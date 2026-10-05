// The Games view's script: stepping, the strip, layers, keys, list paging. Sentences arrive composed and escaped by the server.
(function () {
  'use strict';
  const H = window.Hex, $ = id => document.getElementById(id);
  const S = JSON.parse($('state').textContent);
  const L = { num: false, search: true, tac: true };
  let G = S.game, ply = S.ply, timer = null, strip = null;

  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function trusted(tag, cls, html) { const e = el(tag, cls); e.innerHTML = html; return e; }  // server-escaped sentences only

  function scene() {
    const end = ply >= G.moves.length, pos = G.pos[Math.min(ply, G.moves.length)], think = pos.think || {};
    const t = end ? null : G.tactics[ply], cands = L.search && !end ? (think.cands || []) : [];
    const top = Math.max(1e-9, ...cands.map(c => c[2]));
    return {
      moves: G.moves, owners: G.owners, ply, frame: G.moves, numbers: L.num,
      heat: cands.map((c, i) => ({ c: [c[0], c[1]], rel: c[2] / top, label: i < 3 && c[2] >= .03 ? String(Math.round(c[2] * 100)) : '' })),
      win: L.tac && t && t.cls === 'win' ? t.cells : [], block: L.tac && t && t.cls === 'block' ? t.cells : [],
      ghost: !end && L.search ? G.moves[ply] : null, winLine: end ? G.win : null,
    };
  }

  function panel() {
    const box = $('panel'), pos = G.pos[Math.min(ply, G.moves.length)];
    box.replaceChildren();
    const game = el('section'); game.append(el('h2', null, 'Game'), trusted('p', 'verdict', G.head));
    if (G.turning) { const tp = trusted('p', 'say', G.turning); tp.querySelector('a').onclick = e => { e.preventDefault(); go(+e.target.dataset.ply); }; game.append(tp); }
    const dl = el('dl', 'facts'); G.facts.forEach(([k, v]) => dl.append(el('dt', null, k), el('dd', null, v))); game.append(dl);
    box.append(game);
    if (pos.threat) { const s = el('section'); s.append(el('h2', null, `Turn ${H.turnOf(ply)}`), trusted('div', 'tacline', pos.threat)); box.append(s); }
    if (pos.think) {
      const s = el('section'), th = pos.think; s.append(el('h2', null, 'What the bot thought'), trusted('p', 'say', th.text));
      if (th.light != null) {
        const wc = el('div', 'wc'), bar = el('span', 'bar'), fill = el('i');
        fill.style.width = (th.light * 100) + '%'; bar.append(fill);
        wc.append(el('span', null, `Light ${Math.round(th.light * 100)} %`), bar, el('span', null, `${Math.round((1 - th.light) * 100)} % Dark`));
        s.append(wc);
      }
      if (th.second) s.append(el('p', 'muted small', 'Second stone of the turn: the search averages its forced visits here, so its value reads low and the strip skips it.'));
      if (th.cands.length) {
        const tb = el('table', 'cands'), head = el('tr'); ['Cell', 'Visits', ''].forEach(h => head.append(el('th', null, h)));
        tb.append(el('thead')); tb.tHead.append(head); const body = el('tbody');
        th.cands.forEach(c => {
          const tr = el('tr'), bar = el('div', 'b'), fill = el('i'); fill.style.width = Math.max(2, c[2] * 78) + '%';
          bar.append(fill, el('span', 'num', `${Math.round(c[2] * 100)} %`));
          const tags = el('td'); c[3].forEach(t => tags.append(el('span', `tag ${t}`, t)));
          const cellTd = el('td', 'num', H.fmtC(c)), barTd = el('td', 'barcell'); barTd.append(bar);
          tr.append(cellTd, barTd, tags);
          tr.onmouseenter = () => H.hoverAt($('board-svg'), c); tr.onmouseleave = () => H.hoverAt($('board-svg'), null);
          body.append(tr);
        });
        tb.append(body); s.append(tb);
      }
      box.append(s);
    }
    const act = el('section'), row = el('div', 'actions'), a = el('a', 'btn', 'Open in Analyzer');
    a.href = `/analyzer?${new URLSearchParams({ run: S.run, g: G.id, ply })}`;
    const copy = el('button', 'btn', 'Copy position'); copy.type = 'button';
    copy.onclick = () => { navigator.clipboard && navigator.clipboard.writeText(G.moves.slice(0, ply).map(c => c.join(',')).join(';')); copy.textContent = 'Copied'; };
    row.append(a, copy); act.append(row); box.append(act);
  }

  function render() {
    H.draw($('board-svg'), scene());
    $('where').innerHTML = G.pos[Math.min(ply, G.moves.length)].where;  // server-composed
    if (strip) strip.setCur(H.turnOf(Math.max(0, ply >= G.moves.length ? ply - 1 : ply)));
    history.replaceState(null, '', `?${new URLSearchParams({ ...S.query, g: G.id, ply })}`);
    panel();
  }
  function go(p) { ply = Math.min(G.moves.length, Math.max(0, p)); render(); }
  function goTurn(t) { const i = G.moves.findIndex((_, k) => H.turnOf(k) === t); go(i < 0 ? G.moves.length : i); }
  function drawStrip() { strip = H.trace($('trace'), { pts: G.chances, turns: G.turns, turn: H.turnOf(Math.max(0, ply - 1)), onTurn: goTurn }); }

  async function open(id) {
    const r = await fetch(`/api/run/${encodeURIComponent(S.run)}/game/${encodeURIComponent(id)}`);
    if (!r.ok) return;
    G = (await r.json()).game; ply = G.moves.length;
    document.querySelectorAll('.lrow').forEach(x => x.setAttribute('aria-selected', String(x.dataset.id === id)));
    drawStrip(); render();
  }

  function keys() {
    const k = (g, t) => { const s = el('span'); s.innerHTML = g; s.append(t); return s; };
    $('keys').replaceChildren(k(H.glyph('s1'), 'Light'), k(H.glyph('s2'), 'Dark'), k(H.glyph('windot', 12), 'win available'),
      k(H.glyph('block'), 'must block'), k(H.glyph('heat'), 'where the search looked'), k(H.glyph('ghost'), 'stone played next'));
  }

  function wire() {
    const svg = document.querySelector('.boardwrap svg'); svg.id = 'board-svg';
    svg.addEventListener('pointermove', e => H.hoverAt(svg, H.cellAt(svg, e)));
    svg.addEventListener('pointerleave', () => H.hoverAt(svg, null));
    [['first', () => go(0)], ['prev', () => go(ply - 1)], ['next', () => go(ply + 1)], ['last', () => go(G.moves.length)]]
      .forEach(([id, f]) => { $(id).onclick = e => { e.preventDefault(); f(); }; });
    const play = $('play'); play.hidden = false;
    play.onclick = () => {
      if (timer) { clearInterval(timer); timer = null; play.textContent = 'Play'; return; }
      if (ply >= G.moves.length) go(0);
      play.textContent = 'Pause';
      timer = setInterval(() => { if (ply >= G.moves.length) { play.click(); return; } go(ply + 1); }, 450);
    };
    [['lNum', 'num'], ['lSearch', 'search'], ['lTac', 'tac']].forEach(([id, k]) => {
      $(id).onclick = () => { L[k] = !L[k]; $(id).setAttribute('aria-pressed', String(L[k])); render(); };
    });
    $('scroller').addEventListener('click', e => {
      const a = e.target.closest('a.lrow'); if (a) { e.preventDefault(); open(a.dataset.id); return; }
      const more = e.target.closest('a.more'); if (more) { e.preventDefault(); loadMore(more); }
    });
    document.querySelectorAll('form.filters select, form.filters input[type=checkbox]').forEach(x => x.addEventListener('change', () => x.form.submit()));
    document.addEventListener('keydown', e => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      const rows = [...document.querySelectorAll('.lrow')], i = rows.findIndex(r => r.dataset.id === G.id);
      if (e.key === 'ArrowRight') go(ply + (e.shiftKey ? 2 : 1));
      else if (e.key === 'ArrowLeft') go(ply - (e.shiftKey ? 2 : 1));
      else if (e.key === 'Home') go(0);
      else if (e.key === 'End') go(G.moves.length);
      else if (e.key === ' ') { e.preventDefault(); play.click(); }
      else if (e.key === 'j' && i < rows.length - 1) open(rows[i + 1].dataset.id);
      else if (e.key === 'k' && i > 0) open(rows[i - 1].dataset.id);
    });
    new ResizeObserver(() => drawStrip()).observe($('trace'));
  }

  async function loadMore(button) {
    const q = new URLSearchParams({ ...S.query, after: button.dataset.after });
    const r = await fetch(`/api/run/${encodeURIComponent(S.run)}/games?${q}`);
    if (!r.ok) return;
    const page = await r.json(), at = button;
    page.rows.forEach(row => {
      const a = el('a', 'lrow'); a.dataset.id = row.id; a.href = `?${new URLSearchParams({ ...S.query, g: row.id })}`;
      a.setAttribute('role', 'option'); a.setAttribute('aria-selected', 'false');
      const win = el('span'); win.innerHTML = row.res === 'p1' || row.res === 'p2' ? H.glyph(row.res === 'p1' ? 's1' : 's2') : '–';
      const kind = el('span', 'kind', ({ selfplay: 'Self-play', promotion: 'Gate', external: 'External', random_floor: 'Random' })[row.ch] || row.ch);
      a.append(win, kind, el('span', 'r', row.step >= 0 ? (row.step >= 1000 ? (row.step / 1000).toFixed(1).replace(/\.0$/, '') + 'k' : String(row.step)) : '—'),
        el('span', 'r', String(row.pl)), el('span', null, row.stats ? '·' : ''));
      at.before(a);
    });
    if (page.next) button.dataset.after = page.next; else button.remove();
  }

  H.theme();
  if (!G) return;
  keys(); wire(); drawStrip(); render();
})();
