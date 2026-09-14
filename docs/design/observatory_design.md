# OBSERVATORY — one display surface over the run record: analysis, design, plan (2026-09-14)

Status: **proposal for a ruling**. Nothing here is landed; no code, no producer, no gate moved.
The design canvas the operator can look at: `https://claude.ai/code/artifact/ec6f3827-b320-4e7f-8450-b85a8c6b7bf3`
(seventeen artboards on three pages — light, dark, the two directions not taken — every number read from the mirror at the time of writing). The research it draws
on: `docs/design/observatory_research.md`. Read together with `docs/design/repo_design.md` §1 and
its R333(d) / R352(g) amendments, `docs/governance/CARDS.md` (DASH-2) and R344(d).

## 0. What was measured before forming an opinion

| quantity | value | how |
|---|---|---|
| `tools/run_dashboard.py` (v2) over run6 — 166 MB, 160 621 rows, 35 084 steps | **5.0 s wall, 729 MB peak RSS, 366 KB page** | one invocation on the operator's machine, `resource.getrusage` |
| the same page the packet cites as "5 MB" | the mirror's `dashboard/run6.html` is 5.07 MB, dated 2026-09-13 10:35 — **the v1 page**; v2 landed at `3295cd8c` (11:20) and the run6 refresh unit was never re-run | file dates |
| `tools/game_viewer.py` over run6 — 153 shards, 38 988 games | **6.4 s, 493 MB RSS, index.html 6.8 MB + 77 shard files (34.5 MB, largest 1.2 MB)** | one invocation |
| run7 at the start of this session | events 5.9 MB after 52 min (**≈ 6.8 MB / h**); run6 grew 166 MB over ≈ 30 h (**≈ 5.5 MB / h**) | mirror file sizes |
| byte share of run6's stream | `iteration_complete` **52.9 %** (34 218 rows), `game_complete` 27.3 %, `trainer_step` 14.8 %; the ~35 round-level rows (`eval_round_*`, `monitor_gates`, `eval_channel_health`) are **< 0.1 %** | one streaming pass, `event_census.py` (scratch) |
| what a page needs of the stream | a dozen `(x, y)` series and the last row per event: ≈ 35 k points × 12 series × 16 B ≈ **7 MB** of state for run6 | derived from `tools/dashboard/tier2.py`'s reads |
| the viewer's light index | ≈ 175 B / game → 39 k games = 6.8 MB inline; at run7's ≈ 1 400 games / h a five-day run is ≈ 170 k games ≈ **30 MB of index on the phone** | measured bytes ÷ games |
| refresh machinery today | 2 generators, 2 refresh scripts (`refresh_run7.sh`, `refresh_viewer_run7.sh`) under 1 systemd timer + the puller unit; the timer fires 3 min after the puller's 10-min cycle | `~/.config/systemd/user/`, the mirror's `dashboard/` |
| the live-ness facts the record already carries and the dashboard does not read | `heartbeat_<run>.json` (`seq`, per-source `ages`, `wall_ts`; its path is published on `heartbeat_watchdog_armed.heartbeat_file`); `eval_spool.work/<run>/<round>_progress.txt` (named by contract #11 as the join partner of `game_index`) | the run7 mirror; `docs/contracts/game_record.md` |
| what `resolved_config` carries | **8 knobs**: `amp_dtype`, `eval.random_model_sims`, `eval.sealbot_model_sims`, `identity.encoding`, `identity.representation`, `run_id`, `schema_version`, `seed` — not `train.eval_interval`, not `deploy.search.kind`, not `monitor.alert_grad_norm_max` | run7's row |

Extrapolation, linear, stated as such: a five-day run7 record is ≈ 800 MB of events; the v2
dashboard's whole-file re-parse every 10 min becomes ≈ 25 s and ≈ 3.5 GB RSS per refresh. Time is
not the problem; memory is, and the index page's growth is the phone's.

`docs/governance/falsified.md` was read (R9): no row concerns a display surface; nothing below
re-litigates a falsified row.

## 1. ANALYSIS — should the two surfaces become one, and does that warrant a server

### 1.1 The governance position is not what the packet assumes

R344(d) **already ordered a server**: "`mantis dash serve --run <dir> [--port 8787] [--bind
127.0.0.1]`, read-only over the run record on disk, stdlib HTTP, no build step, no new hard
dependency; viewed locally or through `ssh -L`. Bind is loopback by default; 0.0.0.0 is a flag
documented as unsafe, never the default. It carries the GAME VIEWER". `CARDS.md` records it as
**ORDERED, NOT BUILT**, owing "an R9 amendment to repo_design.md in the SAME commit as the code".
The archived dispatcher note under R344 already flagged the two things that still bind:
(i) `mantis dash serve` is a console script and `pyproject.toml` declares none — "the command is
`python -m mantis.dash serve`", recorded, not decided; (ii) DASH-2 "is both named absences at once,
watching a live run" and needs the amendment written, not just ordered.

What landed instead were two OFFLINE tools (R333(d)'s dashboard, R352(g)'s viewer), each admitted
on the coupling rule — "an absent surface is one that would have to watch a run". The packet's
constraints move one thing in that rule: **nothing under `src/mantis` gains display code and
`monitor` is HEADLESS ONLY.** So R344(d)'s `src/mantis/dash/` cannot be the home. `tools/` can:
both admitted tools already live there, are loaded by path without `sys.path`, and inherit every
gate that scans `tools/` (14, 15, 16, 17).

Conclusion: a server is not a new deviation to argue from zero. It is the standing order, relocated
under `tools/`, with the amendment R344(d) owes finally written — and written honestly: the process
**watches a directory the puller wrote, never a run**; it opens a socket on loopback; it adds no
producer. That is a narrower coupling than R344(d)'s "the monitor becomes a server", and the
amendment should say so rather than pretend it is the offline shape.

### 1.2 Should the two surfaces become one — yes, because the jobs cross them

The four reader jobs (§2.2) are not two jobs on one page and two on another. The drill-down —
a round on the ladder → its games → one position with its root — starts on the dashboard and ends
on the viewer today, through a hand-typed `?g=run/game_id&ply=N`. The round page (canvas: *Desk ·
round*) is the join the operator asked for in GAME-QUALITY terms: 28 of 30 rung losses in
r000025_25000 end by the candidate's own missed block. That number needs the round's event rows
AND its game records AND a derivation over `moves`; neither tool can host it alone. One reader
layer over both contracts is the minimum; one surface over that reader is what makes the join
navigable.

### 1.3 The four options, costed for this repo

| | (a) two generators, shared design + reader | (b) one offline generator, index + lazy data | (c) small stdlib server + no-build front end, incremental readers | (d) built SPA |
|---|---|---|---|---|
| one surface | no — two pages linked by URL | yes | yes | yes |
| the join (round → games → position) | by link only | yes, pre-built for every game | yes, computed on request | yes |
| per-refresh cost at run6's size | 5 s + 729 MB and 6 s + 493 MB, every 10 min, growing linearly | the same unless the generator is made incremental (a state file) — then it is (c)'s reader with a file for a socket | **tail from a byte offset; state ≈ 7 MB for run6; closed shards indexed once** | as (c) |
| phone over LAN | the 6.8 MB index page is the blocker, unchanged | fixed only if the index is chunked per shard | pages ≤ 300 KB; game JSON ≤ 50 KB; lists paginated server-side | as (c) |
| self-contained, from disk | yes | yes for the page; `file://` breaks `fetch` of lazy data in Chromium, so the drill-down needs a file server anyway | served form needs the process; the **frozen form is one file** (§3.4) | needs a build to be self-contained at all |
| one file for the record | two files | one file | one file, written by the same readers (`freeze`) | one file after a build |
| new dependency | none | none | **none** (`http.server`, `json`, `threading`) | node as a build dependency: `mise.toml`'s ONE-consumer contract, a lockfile, `node_modules` hygiene, `dist/` under the artifact gate |
| new gates needed | none | none | none new; the existing 14–17 cover `tools/`; one census test for the no-`src` rule | a bundler output gate, a lockfile gate |
| `repo_design` amendment | one sentence (shared reader) | replace two admissions by one, still "opens no socket" | **the R344(d) amendment it already owes**, narrowed (§4.3) | a new top-level toolchain — a §1 tree change |
| what it does NOT fix | memory growth, the phone, the join | memory growth (unless incremental), `file://` lazy loads | nothing on the list; it adds one long-running process | nothing on the list; it adds a toolchain |

**(d) is rejected**: nothing in the jobs needs a framework; the cost is a second node consumer,
which `mise.toml` says in its own words is the contract it does not want. **(a) is rejected**: it
keeps the phone problem and the join by hand. **(b) is (c) without the socket** — the moment the
generator is made incremental (which the memory numbers demand) it holds the same state (c) holds,
and (c)'s `freeze` IS (b). So the honest choice is between (c) and "(b) now, (c) later", and the
second costs a second migration of the same readers.

### 1.4 Recommendation: (c), as ONE reader layer with TWO front doors

`tools/observatory/` — a stdlib HTTP server for the live case and a `freeze` command that writes
the one-file record from the same readers. Server-rendered HTML with inline SVG for every tier-1
number and every chart (the house rule "survives with script stripped" stays true on the served
page, the frozen page and the phone); ES modules, no bundler, for the board's stepping, the
crosshair readout and navigation; every derivation in Python (the JS draws, it never re-derives
a rule of the game). The frozen CLI flags of both existing tools stay until the last phase retires
them (§4.6).

Why stdlib and not one dep: one reader on a LAN, no TLS, no auth, no streaming — `ThreadingHTTPServer`
serves static files and JSON concurrently and has no supply chain. `starlette`/`uvicorn` would buy
async and a router the routes in §3.3 do not need, at the price of the strict dependency posture
`pyproject.toml` documents for the runtime.

### 1.5 Layout — why `tools/observatory/`, not an `apps/` root

An `apps/` root is a new top-level entry in `repo_design` §1's tree and says "product"; every rule
in this repo says display surfaces are instruments over a record, not products. `tools/` is where
both admitted surfaces and the puller already live, is scanned by gates 14–17, is loaded by path
(no `sys.path`), and may import `mantis.*` while nothing in `mantis` may import it (gate 9 sees
`src/` only; the census test in §4.4 pins the direction). The package is split so no directory
bloats:

```
tools/observatory.py              # the shim, like run_dashboard.py / game_viewer.py: serve | freeze
tools/observatory/
  __init__.py
  cli.py                          # argparse: serve --run ID=DIR … [--bind 127.0.0.1] [--port 8765]; freeze --run ID=DIR --out FILE|DIR
  readers/                        # the ONLY code that knows the two contracts
    events.py                     #   segment-aware tail: (path, offset, held partial line) per segment; reducers
    series.py                     #   the reduced state: (x, y) series, last-row-per-event, counters, game flags ring
    ladder.py                     #   eval_ladder_state.json + round join (today's strength.py, unchanged arithmetic)
    shards.py                     #   game shards: index once per CLOSED shard with byte offsets; the open shard re-read from its offset
    games.py                      #   one game by (shard, offset): moves, arms, stats; the derived facts below
    hexlogic.py                   #   owner, win line (today's viewer/hexlogic.py, moved)
    tactics.py                    #   fours / WIN1 / CHECK / LOST1 / missed block per turn, one-turn exact (§2.5)
    liveness.py                   #   heartbeat file age, mirror pull age, round in flight
  views/                          # Python → HTML/SVG/JSON; no reading here
    tokens.py                     #   the CSS custom properties, light + dark
    svg.py                        #   line/envelope, ladder, ticks, the BOARD (server-rendered for ?ply=N)
    overview.py  rounds.py  games.py  position.py  compare.py  states.py
    stats.py                      #   Wilson, Elo, OLS slope, quantiles (today's stats.py, moved)
  serve.py                        # ThreadingHTTPServer, routes (§3.3), the poll loop (mtime → re-tail)
  freeze.py                       # the one-file page + optional static game data
  web/                            # static, served verbatim, inlined at freeze
    app.css  board.js  charts.js  keys.js  nav.js
tests/tools/test_observatory_<module>.py   # flat, like test_dashboard_*.py; fixtures under tests/fixtures/observatory/
```

Tests mirror the modules one-to-one under `tests/tools/` (the existing flat convention; no package
named `tests` below the root, R5). The parity tests in §4 read the SAME fixture record the dashboard
tests read today (`MANTIS_DASH_FIXTURE_EVENTS`).

## 2. DESIGN

### 2.1 The reader and the direction

One reader: the operator, at a desk between other work and on a phone away from it, reading a run
that is written by a process they cannot see, through a mirror that is ten minutes behind. The
surface is an instrument, not a product: it must be believed, so it must say what it does not
know.

Direction committed: **instrument on paper** — the warm off-white surfaces and hairline rules the
two tools already have, system sans, monospace only where digits align, one accent (the viewer's
`#d94f2b`, kept), status colours only beside an icon and a word, and **absence drawn as a hatched
surface with a sentence**. Not a grey admin panel: there is no chrome to speak of, the largest
object on any page is a board or a chart. Not glass: no gradients, no shadows, no blur. Two
directions were sketched and not taken (canvas page 2): *Console* (dark, monospace, terminal
density — fastest for one expert at a desk, loses the phone and has no room for the absence
sentences) and *Broadsheet* (one sentence leads, serif, one chart at a time — the best five-second
read, nowhere for the drill-down to live).

### 2.2 The jobs, in order, and what each page owes them

| job | page | what answers it in five seconds | what the page must NOT do |
|---|---|---|---|
| (i) is the run healthy | overview, first cell | the badge: the worst of nine inputs with an icon and a word; its reasons inline; **an unread input is never green** (the badge reads "unmeasured for …"); the record's age and the heartbeat's age at the pull on the masthead | animate, tick, or imply the mirror is the run |
| (ii) strength and trend with their uncertainty | overview cells 3–5, the ladder | WR with its n, the Wilson band from the ladder file and the record's own pair-bootstrap CI as whiskers, ≈ Elo; the OLS slope with its t-interval and how many rounds are in it; PROVISIONAL note on every sealbot reading (R353(b)); promoted / rejected / no decision as three shapes | draw a curve from fewer than two rounds; draw a broken round as 0 %; smooth |
| (iii) drill in: round → games → position | rounds, round, games, position | the round's phases with games, sims, wins, and the tactical columns; the game list filtered and paginated; the board with the root's visits, the standing fours, the blocking cells, the arm and sims of the stone just placed, the per-turn state list, the root-value trace | infer an arm from `served_sims`; draw an empty heatmap on self-play; score a move |
| (iv) compare two runs | compare | small multiples with the run as the series (fixed colour per run), one step axis clipped to the shorter run, the ladder of each, and **the instrument behind each curve as a table** (rung, sims, games per round, cadence — each with the field it came from; a hatched row where the record does not carry it) | a dual axis; a curve where one run has no rows |

### 2.3 Information hierarchy, density, typography

Three tiers, as the dashboard has them, kept: tier 1 the hero cells (health first, then progress,
strength, trend, promotion, throughput, losses — reordered so job (i) is the first thing on the
page); tier 2 the panels (ladder + round-in-flight, losses, self-play quality, economy + health
timeline, latest games); tier 3 the collapsed roster, its absence notes last so they see every gap.
Every panel carries its `reads …` line in monospace: the event and field names a number came from.

Type: one family, `system-ui`; 26 / 600 hero values with proportional figures (20 on the phone);
14 / 600 panel titles; 13 body; 11 captions in the muted ink; `ui-monospace` with `tabular-nums`
for tables, axes, coordinates, plies and ids — never for a hero value. No webfont: the page must
work from disk with no network, so there is no font host to lean on (the design canvas honours
this too).

Density: 8 px rhythm; 24 px page gutter at a desk, 16 on the phone; a chart is 440 × 150 in a
three-column grid at a desk and full-width stacked on the phone; the board is 720 px beside a
column at a desk and the full width above the readout on the phone. Charts: hairline solid grid,
2 px lines, the per-pixel min–max band at 12 % opacity, an end-dot with a 2 px surface ring and
the last value as a direct label (leader-lined when two collide) — the extremes stay exact and
nothing is smoothed (TensorBoard's EMA slider was considered and rejected for exactly that: it
hides the spike a trainer reader is looking for).

### 2.4 Colour — one system for charts and board

The dataviz reference palette, validated with its script, light and dark, all pairs, three slots
(the most any page seats: run6 / run7 on compare, up to three rungs on the ladder): worst CVD ΔE
9.2 light / 9.4 dark, normal-vision 24.0 / 20.9. Slot 3 sits at 2.74:1 on the light surface, so a
third series always carries a direct label. The same three slots are the only series colours on
the site; a fourth series folds or facets.

| role | light | dark | where |
|---|---|---|---|
| surface / plane | `#fcfcfb` / `#f9f9f7` | `#1a1a19` / `#0d0d0d` | panels and the board / the page |
| ink / secondary / muted | `#0b0b0b` / `#52514e` / `#898781` | `#ffffff` / `#c3c2b7` / `#898781` | text; p1 stones are ink |
| hairline / axis | `#e1e0d9` / `#c3c2b7` | `#2c2c2a` / `#383835` | grid, borders |
| slot 1 / 2 / 3 | `#2a78d6` / `#eb6834` / `#1baf7a` | `#3987e5` / `#d95926` / `#199e70` | series, fixed by entity |
| sequential (visits) | blue 100 → 700 | the same steps | root visits on the board, count printed |
| status good / warning / critical | `#0ca30c` / `#fab219` / `#d03b3b` (text: `#006300` / `#8a5a00` / `#d03b3b`) | same marks; text `#0ca30c` / `#fab219` / `#e66767` | badges, WIN1 / CHECK / LOST1, the six, promoted / rejected — always with an icon + word |
| accent | `#d94f2b` (4.0:1 light, 4.2:1 dark) | same | the last two stones, "live", the current row — nothing else |
| absence | hatched 45° in the hairline colour over the surface, text in the secondary ink | same | every stated gap |

Board glyphs (canvas: *Tokens*): p1 = ink-filled stone with a paper numeral; p2 = paper-filled
stone with an ink ring; the last two stones ringed in the accent; a fast-arm stone dashed in the
accent (self-play only); a standing four = its six window cells outlined in warning; the cells that
block it = warning-filled, smaller; the six = outlined in good; root visits = sequential blue with
the count printed.

**Dark mode is general, not a phone special case** (operator direction during this packet): every
page has a dark twin on the canvas (page *Observatory · dark*) on the selected dark steps — the
dark surfaces, the series re-stepped and re-validated for them, the stones kept black-and-white
with a light ring on p1, the accent unchanged. The page follows `prefers-color-scheme` and keeps an
explicit toggle per viewer (`localStorage`, wrapped — a per-viewer convenience, never state); the
frozen file carries both palettes as CSS custom properties, so one file serves both.

### 2.5 The tactical overlay — derived, labelled, oracle-tested

The vocabulary the two measurement records used (fours, WIN1, CHECK, LOST1, missed block, missed
win, check-run) was computed by scratch scripts validated against the engine's `winning_moves ∪
threat_moves` on 18 190 positions with 0 disagreements. It is a pure function of `moves`, exactly
as the owner and the win line are, and the R352(g) amendment's rule for derived facts applies with
one difference: there is no record field to check a four against, so the check is the engine
oracle in tests (`mantis._engine.Board`), and the page says on every position that the state is
**derived by the observatory, not recorded by the run**. It is computed on request for one game
(milliseconds), never pre-built for 39 k games; the freeze carries it for the games it inlines.
It is not a judgement: no accuracy score, no move classification beyond the exact one-turn facts.

### 2.6 Keyboard and touch

`←` `→` ply; `Home` `End`; `space` play/pause (350 ms); `n` `p` next/previous game in the current
list; `[` `]` previous/next round; `h` visits; `t` tactics; `l` copy the position link; `g` focus
the games list; `/` focus the filter row; `?` the map; `Esc` close. On the phone: swipe the board to
step, tap a stone to jump to its ply, tap a blocking cell to see the four it blocks; controls are
44 px. Every position is an address: `/run/<id>/game/<game_id>?ply=N&heat=1&tac=1`, server-rendered
at that ply so the link works with script off and loads in one round trip on the phone; the
existing `?g=run/game_id&ply=N&heat=1` form is redirected for the record's existing links.

### 2.7 Empty, broken, stale, refused — the vocabulary (canvas: *States*)

One shape per fact class, each naming its event or producer: *absence* (no rows — hatched panel,
the event named); *banked* (no producer at HEAD — hatched, the producer that would fill it named);
*unmeasured input* (badge in secondary ink, never green); *broken round* (a hatched square ON the
axis at its step, listed, never omitted, never 0 %); *pre-producer record* (the arm "not recorded",
stones drawn plain); *stale mirror* (a warning badge with the pull age — the mirror is stale, the
run is not known to be); *refused* (an event-less record gets a refusal, never a clean page);
*self-play stats* (the visits toggle disabled and saying why); *round in flight* (rows land at block
end for a concurrent block, and the panel says so — no rows is not a wedge).

### 2.8 Deliberately NOT on the page

- No Elo across rungs, no Bradley–Terry: the record carries WR per rung and nothing that combines them.
- No accuracy / brilliancy / blunder scoring of moves: only exact one-turn facts, labelled derived.
- No smoothing, no EMA slider, no "ignore outliers" toggle: the envelope keeps every extreme.
- No auto-refresh animation, no live-ticking counter: the page states the record's age; a reload is a reload.
- No frontier / strix cells: they are `result.json` files, not a run record under either contract — a stated gap with the reader that would fill it (a follow-up).
- No config facts the record does not carry (`train.eval_interval`, `deploy.search.kind`, the grad-norm ceiling): hatched rows naming the missing producer — a producer under `src/` with its test is a follow-up packet, never this one.
- No write actions, no run control, no login, no notifications, no server-side accounts.
- No build step, no framework, no CDN, no webfont.

## 3. The served and frozen forms

### 3.1 Readers, incremental

`events.py` keeps, per run, the list of segment files (`events_<run>_seg*.jsonl`, sorted) and per
segment `(byte offset, held partial line)`. A poll stats every segment; a grown file is read from
its offset; a torn tail line is held until its newline arrives (the mirror is rsynced mid-write;
`read_shard`'s contract already says this is normal); a new segment is a resume and is appended.
Rows are parsed and REDUCED at once — the dicts are not kept; `series.py` holds `(x, y)` lists per
key, the last row per event, counters, and the per-game flag ring the ply-cap window reads. This
is today's `Record` with the same arithmetic and without the 729 MB. `shards.py` indexes a shard
once when the run's `games_<run>_index.jsonl` names it closed, storing `(game_id → shard, byte
offset)` and the light row; the open shard is re-read from its offset on each poll. A game is
fetched by one seek and one line parse.

### 3.2 Poll, not watch

The mirror advances every 10 min; the server polls the run directories' mtimes every 30 s and
re-tails on change. No inotify (a dependency and a platform), no push to the browser: the page
carries the record's timestamp and the pull's age, and a reload is the refresh. This is the
honest cadence: the freshness floor is the puller's, and the server buys only the 3-min timer
offset plus the whole-file re-parse — the point of the server is memory, the join and the phone,
not latency.

### 3.3 Routes

`/` (the run list: every `--run` given, with its state), `/run/<id>` (overview), `/run/<id>/rounds`,
`/run/<id>/round/<round_id>`, `/run/<id>/games?channel=&result=&term=&plies=&page=`,
`/run/<id>/game/<game_id>?ply=N&heat=1&tac=1`, `/compare?runs=a,b`, `/api/run/<id>/game/<game_id>`
(the JSON the board module steps through: moves, arms, stats, the per-turn tactical rows),
`/static/…`. Bind `127.0.0.1` by default; `--bind 0.0.0.0` exists for the phone and is documented
as unsafe on any network that is not the operator's own (R344(d)'s words). Read-only: no route
writes. Host names never appear in code or docs (rule 7).

### 3.4 Freeze — one file for the record

`freeze --run ID=DIR --out FILE` writes ONE self-contained HTML file: the overview, the rounds table
and every round's phase table WITH its derived tactical columns (a pass over the games at freeze
time, storing counts, not games), the same views and styles as the served form, every chart
server-rendered, both palettes inline. It carries NO game data by default and says so: measured on
run6, the eval-channel games alone are 12.5 MB slimmed to moves + roots (promotion 4.5 KB / game,
external 2.8 KB, random floor 1.6 KB; self-play 0.57 KB × 35 287 = 20 MB), which is not a file for
a phone or a record. `freeze --games eval|all --out DIR` writes a DIRECTORY instead (the page plus
per-shard data files, the board module included) for a record that must be browsable without the
process. The one-file page replaces today's dashboard page (366 KB) at the same order of size; the
size test is kept and its cap re-measured at landing.

### 3.5 What the operator's machine and phone carry

Server: one Python process, RSS ≈ 150–250 MB (the interpreter with `mantis._engine` imported,
≈ 7 MB of reduced state per run6-sized run, the shard index ≈ 7 MB per 39 k games); CPU: a poll is a
stat; a re-tail parses ≈ 1 MB of new rows per 10 min in well under a second; a position request is
one seek + a tactics pass (ms). Disk: nothing written except the freeze on demand. Under a systemd
user unit like the puller's. Phone: an overview page ≤ 300 KB (it is 305 KB today for run7 and is
mostly SVG), a games page of 50 rows ≈ 30 KB, a position page ≈ 60 KB + one game JSON ≤ 50 KB;
nothing over the LAN that scales with the run's length.

## 4. PLAN — phases, each one reviewable unit with tests

Each phase is one commit series on a worktree branch the operator fast-forwards; each lands with
its tests in the same commit; the full local gate set runs at each phase's exit (`make gates.exit`
at the packet exit). The frozen CLI flags of both old tools keep working until phase 6. No phase
touches `src/mantis`, the box, or the mirror's units (the operator swaps those at phase 6).

| phase | lands | tests (all under `tests/tools/`) | gates it must clear |
|---|---|---|---|
| **0 · ruling + amendment text** (docs only) | the amendment of §4.3 as text in this file's successor commit; CARDS.md's DASH-2 row updated to "built as OBSERVATORY under tools/" when phase 3 lands | — | none (docs) |
| **1 · readers** | `tools/observatory/readers/{events,series,ladder,shards,games,hexlogic}.py` (hexlogic and the ladder/stats arithmetic MOVED from the two tools, not copied — the old tools import them from the new home) | parity: the reduced series and the hero numbers equal `tools/dashboard`'s over the fixture record; incremental ≡ whole (a planted append yields the same state); a torn tail is held, not skipped; a new segment is appended; a closed shard is indexed once; a game is read by offset; the viewer's owner/win-line tests move with the code | 14 (pyright over `tools/`), 15, 16, gate 3 floor follows |
| **2 · views + freeze** | `views/`, `freeze.py`, `tools/observatory.py freeze`; the overview, rounds, round, games pages rendered from the readers; the size test | every tier-1 number survives script-stripped; the panel roster matches the contract order; every panel carries its `reads` line; the absence vocabulary of §2.7 has one test per shape (no zero, no empty axis, no flat line from no series); the page carries no absolute home path; the frozen page for the fixture record carries the same numbers as the old dashboard's | 6 (no output tracked), 14–17 |
| **3 · serve** (the socket; **the amendment commit**) | `serve.py`, routes, the poll loop, `liveness.py` (heartbeat age via `heartbeat_watchdog_armed.heartbeat_file` — resolved by BASENAME under the run's `logs/`, because the published path is the writing host's and the mirror's differs; pull age from the newest receipt's mtime; round in flight via `<round>_progress.txt` with the block-end caveat), the `repo_design` amendment of §4.3, the DASH-2 card moved | routes serve the same HTML as freeze for the same record; loopback is the default bind; a stale mirror is drawn as stale, not as a stalled run; a round in flight with no rows is drawn as "rows land at block end"; a census pins that nothing under `src/mantis` imports `tools` and that `tools/observatory` opens no socket at import | 9 (unchanged), 17 (no host), the census |
| **4 · games** | `readers/tactics.py` (the validated instrument, ported), the board module `web/board.js` (the viewer's JS as an ES module, stepping only), the server-rendered board at `?ply=N`, arms/sims, the visits overlay, the tactical overlay, the per-turn list, the value trace, pagination and filters | tactics vs the engine oracle on the fixture positions (`winning_moves ∪ threat_moves`, 0 disagreements — the GAME-QUALITY bar); a self-play game's visits toggle is disabled and says why; a pre-producer record says "arm not recorded"; the `?g=` redirect; the position page at `ply=N` carries that ply's stones with script stripped | 14–17 |
| **5 · compare** | `views/compare.py`: two runs on one step axis clipped to the shorter, small multiples, the instrument table with a hatched row per fact the record lacks | the series colour follows the run across a filter; a run with no rounds contributes an absence, not a curve; the instrument rows name their fields | 14–17 |
| **6 · retire** | delete `tools/run_dashboard.py`, `tools/dashboard/`, `tools/game_viewer.py`, `tools/viewer/`, their tests (the moved ones already live under the new names), the `make dashboard` / `make viewer` targets (replaced by `make observatory.freeze` / `make observatory.serve`); the operator replaces the mirror's `refresh_run7.sh`, `refresh_viewer_run7.sh` and `mantis-dashboard-run7.{service,timer}` with one `mantis-observatory.service` (a documented unit shape with placeholders, not a tracked unit — it would carry home paths, rule 7); the R333(d) and R352(g) amendments gain a closing line; the test-count floor follows the net count | the parity fixtures stay as the observatory's own; gate 10 (no Makefile/doc reference to the deleted paths) | 10, 3c |

Phase 4 may precede 3 (it does not need the socket); phases 1–2 must precede both. Phase 6 is the
only phase that deletes and it waits for the operator's word after phases 3–5 have run beside the
old tools for at least one live round of run7.

### 4.1 What is NOT in any phase

A producer under `src/` (a `resolved_config` knob roster wide enough for `eval_interval` and
`deploy.search.kind`; a self-play search-stats producer — `CARD-SELFPLAY-SEARCH-STATS`; a frontier
reader). Each is a follow-up with its LAW-07 producer test; the pages state the gaps until then.

### 4.2 New gate

None. The `tools/` gates cover the Python; the one new invariant — no `src/mantis` → `tools`
edge, no socket at import — is a census test, not a CI gate, because it guards one package.

### 4.3 The `repo_design` amendment (phase 3's commit carries it; proposed text)

> **AMENDMENT — R344(d) discharged under `tools/`: the OBSERVATORY is admitted; the two offline
> tools it replaces are retired at its last phase.** §1 says display surfaces are absent and names
> the event manifest + the game record as the contract any display builds against. R344(d) ordered
> a read-only stdlib server carrying the game viewer and owed this amendment; R333(d) and R352(g)
> admitted two offline tools on the rule that an absent surface is one that would have to WATCH A
> RUN. (1) What is admitted: `tools/observatory.py` (`serve`, `freeze`) with its package
> `tools/observatory/` — ONE reader layer over contract §4.7 and contract #11, an HTTP server on
> loopback by default that reads a run directory or its mirror INCREMENTALLY, and a `freeze` that
> writes one self-contained HTML file from the same readers. (2) What it is not: it watches a
> DIRECTORY, never a process — no connection to a run, no producer, no write; `src/mantis` gains no
> display code and `monitor` stays headless (a census pins the import direction); it needs no
> build step, no new dependency, no node. (3) The coupling rule is NARROWED, not lifted: the
> absence this file keeps is any surface a RUN would have to know about; a reader of a record's
> files is admitted on the record's own terms, whether it runs once (`freeze`) or stays up
> (`serve`). (4) The rule the tool carries: absent is not zero, on every page, in one vocabulary
> (docs/design/observatory_design.md §2.7); every derived fact is labelled derived and has an
> oracle test. (5) `tools/run_dashboard.py` and `tools/game_viewer.py` are retired when the last
> phase lands; their amendments above stay as history with a closing line.

### 4.4 The census test (phase 3)

`tests/tools/test_observatory_census.py`: nothing under `src/mantis` imports `tools` (the direction
that keeps `mantis` free of display code); importing `tools.observatory.*` binds no socket (the
socket is opened in `serve()` only); no module under `tools/observatory` writes anything but the
freeze's `--out`.

### 4.5 Risks named

- The tactics reader is the one derivation with no record field to check against; its oracle test
  is the engine and it is labelled derived on every page. If the oracle test cannot reach 0
  disagreements over the fixture set, the overlay does not land and the page states the gap.
- `file://` in Chromium blocks `fetch` and ES-module loads; the frozen ONE file inlines everything
  it carries and carries no game data; the directory freeze and the served form need a file
  server or the process. There is no fourth form and the docs say so.
- A five-day run's game list is ≈ 170 k rows; pagination is server-side from the shard index and
  the filters are one row above the list; the phone never receives an index.

### 4.6 What is deleted, and when

At phase 6, and not before: two generators (`tools/run_dashboard.py` + `tools/dashboard/`,
`tools/game_viewer.py` + `tools/viewer/`), their two Makefile targets, ten test files (their
behaviour re-homed), and on the operator's machine two refresh scripts and one timer/service pair,
replaced by one unit. Until then both frozen CLIs run unchanged beside the new tool.
