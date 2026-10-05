# DASH-2 — one local server, three views: Run, Games, Analyzer (2026-10-03)

R344(d) ordered DASH-2: a read-only stdlib server over the run record, carrying the game viewer,
loopback by default. `docs/design/observatory_design.md` (2026-09-14) is its first design. This
document keeps that design's server (§3: incremental readers, poll not watch, routes, bounded lists)
and its plan's shape (§4), REPLACES its views (§2) with the three the operator approved as mockups on
2026-10-03, and folds the ANALYZER-1 tool into the same server as the third view. Where the two
documents disagree, this one rules; observatory §3 and §4 are cited, not copied.

## 0. What exists, and what was learnt before designing

**Today.** Three tools, three looks, no links between them:

| tool | form | what is wrong with it for the operator |
|---|---|---|
| `make dashboard` (`tools/dashboard/`) | one static HTML per run, rebuilt by a timer from the mirror | headline is the sealbot rung R362 deleted; half the page is plumbing (determinism hash, eval-child memory, receipts, F-616-37 firings); a paragraph of ruling prose per panel; 0–1 axes; raw series; no value-instrument panel; no run overlay |
| `make viewer` (`tools/viewer/`) | static directory, 561 MB for run8's 62 481 games, an 11 MB index page | the whole list in one page; round stones with every ply number; the fast arm as dashed stones on the board; no win-chance trace although every sampled game carries per-ply search stats |
| `make analyzer` (`tools/analyzer/`) | loopback server over stamped checkpoints | sound engine layer; the page is a dense table; on the operator's main checkout its search fails (`MCTSTree` has no `last_tt_hits`: the compiled engine predates `cd22ed4e`; `make build.cuda` restores it) |

**Rivals, read statically** (nothing from their repos or releases was executed; reports in the
session record, §10): Six (state file per generation, a newest-vs-previous tape whose rows declare
which direction is better, verdict words only when the interval excludes zero), Bubble (JSONL +
heartbeats, a stdlib localhost server, a one-line definition under every chart, a smoothed line over
a faint raw trace, min/max downsampling), Strix (TensorBoard with a definition on every metric, an
eval graph dashed where unanalysed, a heatmap from searched moves only), Shrimp (the knowledge-horizon
chart: value sign accuracy by plies to the end, won vs lost; solid/dashed/dotted for now/as
played/compare). Taken: definitions on every chart, interval-gated verdict words, losses shown but
never scored, smoothed over faint range, the knowledge horizon, gaps as gaps. Not taken: Bubble's 25
ungrouped panels and three themes, Shrimp's one-metric-at-a-time chart and 6–7 px type, Six's
broadcast styling.

## 1. Scope

One process, `make dash`, three views under one menu: **Run** (§2), **Games** (§3), **Analyzer**
(§4), one board and one set of marks (§5), one visual system (§6). It reads a run directory or its
mirror and stamped checkpoints; it adds no producer, writes nothing, connects to no run, and needs no
build step or new dependency. The Analyzer's engines start only when `--checkpoints` is given; without
them the view says how to start it. When the last phase lands, `tools/dashboard`, `tools/viewer`,
`tools/run_dashboard.py`, `tools/game_viewer.py` and the analyzer's `web/` are deleted; the analyzer's
engine layer (`engines.py`, `analysis.py`, `instruments.py`, `strix.py`, `dispatch.py`) moves under
the new package unchanged.

Every number on every page is derived in Python by a reader with a test; the browser draws and steps,
it never re-derives a rule of the game or a statistic. The verdict sentences, the status line and
every chart are server-rendered (they survive with script stripped); script adds the crosshair, the
board stepping, the list paging, the theme toggle and the Analyzer's requests.

## 2. The Run view

Top to bottom, one question per section. Each section is a verdict sentence written from the data,
a one-line aside, and the evidence charts. A section whose input has no producer in the record says
so in place; it never draws a zero.

**Status line.** Live / stale / stopped (heartbeat age under 10 min / under 1 h / older, from the
heartbeat file's mtime and `wall_ts`; its format stays uncontracted and feeds nothing else), steps,
games, last save, last heartbeat in CET/CEST, and the training-alert count.

**Is it getting stronger?** Win rate against Strix per saved checkpoint from the follower's sidecars
(`*.ckpt.strix256.json`), whiskers the sidecar's 95 % interval, gate promotions as marks on the x axis,
and the PARENT as a band: the stem of `resolved_config.yaml`'s `warm_start.checkpoint`, joined to that
checkpoint's sidecar in whichever served run directory holds it. A series joins only sidecars of one
unit (unit, our search kind and sims, strix commit, checkpoint sha, sims, solver); any other unit
appears in the table as "other unit", never on the line. Six is a second panel, drawn as a stated gap
until a per-save Six sidecar exists.
The verdict compares the latest point with the parent and with the run's first point:
`d = p_a − p_b`, `se = sqrt(p_a(1 − p_a)/n_a + p_b(1 − p_b)/n_b)`, a word ("stronger than", "weaker
than") only when `d ± 1.96·se` excludes zero, otherwise "not separable from … yet". The sidecars carry
counts, not per-pair outcomes, so the interval ignores the pairing; paired openings make the true
interval narrower, so the rule errs toward "not separable" and the definition line says so.
Run8 reads: "45k is not separable from its parent yet, and clearly stronger than its own 3k"
(0.142 against the parent's 0.111, 0.056 at 3k; 288 games each).

**Is the value head learning?** The value instrument at each save (calibrated held-out CE, AUC,
temperature, train/held-out gap, by ply band) is the reading that rules (R382(b)); it has no per-save
producer, so the panel is a stated gap until one writes a sidecar (a follow-up card, not DASH-2).
Beside it, the knowledge horizon: the share of positions where the sign of the search's root value
matches the result, by full turns to the end, split by whether the side to move went on to win or
lose, the late window solid and the early window faint. Positions are each turn's FIRST stone in the
sampled self-play search stats; the windows are the first and last fifth of the self-play shards that
carry stats. The verdict states the reach: the largest k such that agreement is ≥ 90 % at every turn
≤ k (run8: 4 turns, as early in the run). The aside says it is the search's value, not the head.

**Is training stable?** No verdict on losses: they are shown, never scored ("lower is not
stronger"). The sentence reports aborts and warnings from `training_alert`. Small multiples:
value loss, policy loss, policy entropy, gradient norm, learning rate (×10⁻³, axis from zero), game
length. Each is a 2 px line of the bucket mean smoothed, over the bucket's min–max range at 10 %
opacity; the y axis is fitted to the line plus the median half-range, so a spike clips instead of
flattening the curve.

**Is self-play healthy?** Games per hour, the first-mover share over the last 2 000 games with its
95 % interval ("balanced sides" only when 0.5 is inside it; run8 reads 47 %, 45–49 %, so "the second
player slightly favoured"), the share ending at the cap, positions per hour, trainer steps per hour.

**Record details** (closed by default): gate rounds (step, promoted, wall), alerts and stops, the
config hash, the event inventory. Nothing else from today's dashboard survives on the page; the record
keeps it.

**Compare.** A second run overlays every chart in the second categorical colour; a run whose record
lacks a series shows a disabled legend entry "not recorded" (run7 has no `policy_entropy`). Colour
follows the run: run 1 blue, run 2 orange, run 3 aqua, validated in both themes (CVD ΔE ≥ 9.2;
the light-mode contrast warning is relieved by the legends and table views).

## 3. The Games view

An app frame: the page never scrolls, its three columns do.

- **List (left, its own scroll).** Filters (kind: Self-play, Gate, SealBot, Random; "with search";
  winner or "ended at the cap"; newest, longest, shortest; go to id). Rows: the winner as a small hex
  stone, the kind, the net step, the stone count, a search glyph when per-ply search stats exist. The
  list holds the 200-row window of observatory §3.5 with "load 200 more" at its foot (the operator's
  2026-09-14 direction stands; the pane scrolls within the window).
- **Board (centre).** §5's board for the position after `ply` stones, the whole game's extent as its
  frame so it never jumps; under it the key, the win-chance strip, the transport (first, previous,
  play, next, last; turn N of M, who places stone k of 2) and three layers: turn numbers (off), the
  bot's search (on), threats (on). Keys: arrows by stone, Shift+arrows by turn, Home/End, space, j/k
  between games.
- **Win-chance strip.** Light's chance fills from the bottom in the light stone's tone, Dark's is the
  rest, mirroring the stones. Points are each turn's first stone: the second stone's W/N is biased
  low (§9). A turn's cost to its mover is its own chance at the turn's start minus the opponent's
  view at the next turn's start; a cost ≥ 0.30 is an amber tick, ≥ 0.50 red. Drag to scrub.
- **Panel (right).** In sentences: "Dark wins with six in a row on turn 26", the turning point (the
  largest cost ≥ 0.30, linked), the kind, seats, net ("run8 at 50.8k, the actor's copy" for self-play;
  "before the actor's first sync" when the record says −1), search recorded or not, the hour and
  worker. At a ply: the threat line from the engine's tactics ("Light has a four. Dark must block
  (−6, −1) or (−2, −5) this turn, with either stone"), then what the bot thought ("Full search, 319
  visits. It played (−6, −4), its most-visited move"), a Light/Dark win-chance bar, and the top
  candidates with visit shares tagged played / blocks / wins. A stone without stats says why. Open in
  Analyzer carries the game and the ply.

**Per-game payload** (`/api/run/<id>/game/<game_id>`): moves, arms, sims, the stats as recorded (top
visits, root value), and per ply the engine's tactics (`mantis.diagnostics.tactics`: win cells, block
cells, the four count) and the win line at the end. The record keeps only each search's top 14 cells;
a cell outside them is "—", never 0 %.

## 4. The Analyzer view

The board on the left (the same component), the panel on the right.

- **Source line.** The game and ply it came from, "on the game's line", or "variation, N stones off
  the game" with Back to the game. Clicking the board places a stone; Undo and the arrows walk back.
- **Nets.** Chips per loaded engine (any stamped checkpoint, the vendored Strix opted in); one is
  read, another compared; colour is the engine (blue, orange), the same two hues as the lens.
- **Verdict.** Tactics first, then what each net does about it, then what the game's search did:
  "Dark must block Light's four this turn, at (−6, −1) or (−2, −5). Both nets block with their first
  choice, (−6, −1)." / "In the game the search put its first stone at (−6, −4) and blocked with the
  second at (−6, −1). With two stones a turn that is legal."
- **Win chance from the value head**, one Light/Dark bar per net and one for the game's search,
  labelled uncalibrated.
- **One lens on the board at a time:** Net (the read engine's policy), Search (the analyzer's own
  search, or the game's recorded search when the position is on the game's line), and A vs B (the
  difference of the two policies: blue where A puts more weight, orange where B does).
- **Candidates.** The read engine's top cells plus every block/win cell: A's prior, B's prior, the
  search's visit share, tags. Hover highlights the cell; click plays it.
- **Look deeper.** Search N sims, the symmetry check, copy position: the existing engine calls,
  their latency shown as a state, never a stale number (analyzer amendment, point 3).

## 5. The board and its marks

Stones are filled hexagons; **luminance tells the players apart** (Light, Dark, as in Go), so hue is
free for marks, and each mark owns one channel so none hides another:

| meaning | mark |
|---|---|
| a stone | filled hexagon, radius 0.84 of the cell; Light bone white, Dark near-black with a faint light rim on the dark board |
| the last turn | a small hexagonal dot in the opposite tone inside both of its stones; with turn numbers on, a thin inner outline |
| the winning six | an amber 0.2-wide line through the six centres and amber rims |
| a win available now | amber dashed outline with an amber centre dot |
| must block this turn | red outline over a faint red fill |
| where the bot looks | a blue inner hexagon sized by share (radius 0.26–0.82), the top three labelled in % — smaller than a stone, so never a third player |
| its choice | a dashed ink outline: the stone played next (Games) or the first choice (Analyzer) |
| A vs B | the same inner hexagon, blue where A leads, orange where B does |
| turn numbers | optional; both stones of a turn share one number |

The fast/full arm leaves the board: it is a word in the panel at that stone.

## 6. Visual system

Tokens on `:root`, a dark default with a light toggle (operator direction: dark everywhere), the page
plane and chart surface from the validated chart palette, colour meaning data only (runs, engines,
marks, status with an icon and a word), chrome monochrome. One family, Instrument Sans (OFL; its zero
is plain, which Atkinson Hyperlegible's hosted build is not), with tabular figures on axes and tables.
Charts: one axis, 2 px lines, hairline grid, a definition under each, a table view on the strength and
horizon charts, a crosshair readout. Phone: the Run sections stack; Games stacks board, panel, list.
No animation beyond the theme switch.

## 7. The server

Observatory §3 holds with these deltas:

1. **Package** `tools/dash/` (shim `tools/dash.py`, target `make dash RUNS="run8=<dir> …"
   [CHECKPOINTS=…] [STRIX=1] [PORT=8765]`): `readers/` (events, series, sidecars, shards, games,
   tactics, liveness, horizon, chances), `views/` (run, games, analyzer, board, svg, tokens, stats),
   `serve.py`, `web/` (one CSS, `board.js`, `charts.js`, `keys.js`). Every module ≤ 300 lines.
2. **Readers** come back from history (`3a563574..4678537d`: 3.0 s / 84 MB on run6 against the
   dashboard's 5.0 s / 729 MB) and gain three: sidecars (strength, the parent join, the unit rule),
   horizon (§2), chances (§3's strip, costs, turning point).
3. **Routes** add `/analyzer` and `POST /api/analyze` (today's analyzer request, unchanged) and
   `/api/run/<id>/game/<game_id>` gains the per-ply tactics.
4. **Freeze** shrinks to the Run view: one self-contained HTML with the charts inline, replacing
   `make dashboard` for records. The directory freeze of games is dropped.
5. **Escaping.** Every string from a record reaches the page through an escaping helper; ids are
   never interpolated raw.

## 8. Plan

Phases on a worktree branch the operator fast-forwards; each lands with its tests; `make gates` at
each phase exit, `make gates.exit` at the last; no phase touches `src/mantis`, the box or the mirror's
units.

| phase | lands | tests (under `tests/tools/`) |
|---|---|---|
| 1 · readers | the retired readers revived under `tools/dash/readers/`; sidecars, horizon, chances | parity with `tools/dashboard` on the fixture record; incremental ≡ whole; torn tail held; the unit rule (a planted other-unit sidecar stays off the line); the parent join from `warm_start`; horizon and chances against hand-computed fixtures; the second-stone exclusion |
| 2 · Run view + freeze | `views/run.py`, `svg.py`, `tokens.py`, the four sections, compare, freeze | every verdict word appears only when its interval excludes zero (planted pairs both sides); every gap is a sentence, never a zero; losses carry no verdict; the same record freezes byte-identically |
| 3 · serve (the amendment commit) | `serve.py`, routes, the poll loop, liveness, §8.1's amendment, CLAUDE.md's display line | observatory §4 phase 3's tests; the census (nothing under `src/mantis` imports `tools`; no socket at import) |
| 4 · Games | the list window, the board module, the strip, the panel, per-ply tactics | tactics against the engine oracle, 0 disagreements on the fixture games; a cell outside the top 14 renders "—"; a pre-sync record says so; the window's cursor is stable while the open shard grows |
| 5 · Analyzer | the engine layer moved, the view, the lenses, variations | today's analyzer tests, moved; A vs B sums to zero over the union of cells; a variation drops the game's search |
| 6 · retire | delete the three old tools, their targets and tests (moved tests land first, gate 3c), the operator swaps the mirror's timers for one `make dash` unit | gate 10 finds no reference to a deleted path |

### 8.1 The amendment (phase 3's commit carries it; the ruling number is the operator's)

> **AMENDMENT — R344(d) discharged under `tools/`: DASH-2 is admitted; the dashboard, the viewer and
> the analyzer's page it replaces are retired at its last phase.** One loopback stdlib server,
> `tools/dash.py`, reads run directories or their mirror and stamped checkpoints, and serves three
> views (Run, Games, Analyzer) and a one-file freeze of the Run view. It watches a directory, never
> a process: no connection to a run, no producer, no write; `src/mantis` gains no display code and
> `monitor` stays headless. Its engines are the analyzer's, admitted by R363 on the same terms. The
> coupling rule of R333(d) is narrowed, not lifted: what stays absent is any surface a run would have
> to know about. The R333(d), R352(g) and R363 amendments stay as history with a closing line.

## 9. Findings this work surfaced (for cards, not for DASH-2)

- **The recorded root value is a poor position value at a turn's second stone.** In run8's sampled
  self-play, at the stone that wins the game the search always plays the winning move, yet its W/N is
  ≤ 0 in about 31 % of those positions (264 of 842): Gumbel's halving forces visits onto losing
  children (19 of 64 on the winning move, 18 on the next) and W/N averages them in. It agrees with
  REG-1's 0a (completed Q beats W/N); the ring's root-value field held exactly this value until RUN11-PRE moved it to
  Σπ′·completedQ (the weight `train.value_target_search_weight` mixes into the target).
- **Self-play records carry the actor's step** from the first sync on; only games before it say −1.
  No producer gap.
- **The operator's main checkout runs a stale engine** (built 2026-09-29), so `make analyzer`'s
  search fails there until `make build.cuda`.

## 10. Open for the operator

1. The ruling number for §8.1.
2. Vendoring the Instrument Sans woff2 (OFL, with its licence) under `tools/dash/web/`, or the system
   font stack.
3. Keeping the one-file freeze (§7.4) or dropping it with `make dashboard`.
4. Where the mockups and the four rival reports go: `mantis-records/dash2/` (they are in the session
   scratchpad now).

## 11. Annotations (the tree moved after 2026-10-03)

- **2026-10-05, delta (a): the value-head panel has a producer.** `tools/run_monitor` writes one record per save at
  `<records>/saves/<step:08d>.json`, plus `HALT.json` and `GAP_RULE.json`. The server takes them as `--records
  ID=DIR` beside the run directory and draws §2's value panel from them instead of a stated gap: calibrated held-out
  CE, AUC and temperature per save, CE by ply band, the exams against their floors, the train/held-out gap against
  the gap rule's line, the lagged read. The keys it reads are pinned against the monitor's real output (LAW-07). A
  null temperature or `holds: null` is "not measured", never a zero.
- **2026-10-05, delta (b): strength is read against Six first.** The primary panel is Six (`<ckpt>.six30_16*.json`;
  run11's cells are `.six30_16.full.json`, 288 pairs). Strix is a second panel only where a run has strix sidecars.
  The going-forward read is drawn as a marked band: the mean logit of the last four cells of the line against the
  parent's logit + 0.17 (fewer than four, those available, stated). The cells come from `--cells DIR` (repeatable),
  joined to a run by the sidecar's `run_id`. The parent is the stem of `identity.warm_start.checkpoint` (§2 said
  `warm_start.checkpoint`; the key lives under `identity`).
- **2026-10-05, delta (c): a resumed run has several segments.** The readers take every `events_<run>_seg*.jsonl` in
  segment order, and the status line reads the newest segment (a failed boot leaves a short segment between two
  lives).
- **2026-10-05, build decisions.** §10.2: the system font stack, nothing vendored. §10.3: the one-file freeze is kept.
  §10.4: the mockups and rival reports are in `mantis-records/dash2/`. The horizon's windows are the first and last
  fifth of the sampled games in record order (a run of a few shards has no fifth of shards). The Games list labels
  each kind by its channel (Self-play, Gate, External with its rung, Random).
- **2026-10-05, delta (b) extended (operator, relayed by the RUN11-GO session and recorded in the packet): several
  rulers and a ladder.** Every sidecar unit is its own series with its own parent anchor (the parent's sidecar on
  the same unit); none is pooled or joined across units. The pre-registered rule's unit is a server input
  (`--rule-unit ID=UNIT`, matched against the sidecar's `unit` or `unit.arm`); every other unit is labelled
  report-only, and the going-forward band is drawn on the rule's unit only. The Run view shows the rule's ruler in win
  rate with the parent and going-forward bands, and every ruler as logit(win rate) − logit(parent's win rate on that
  ruler), with win rate, interval and the cell's host load in the table. The ruler ladder's state file
  (`--ladder ID=FILE`: `{current_unit, streak, history, changes}`) marks rung changes on the x axis and names the
  bridge pair, the same checkpoint read on both rungs; a missing file is a stated gap. A sidecar without `eff_n` is
  refused, never read on its raw game count (LAW-04).
- **2026-10-05, build decisions (continued).** The tokens live once, in `web/dash.css` (§7.1's `views/tokens.py` is
  not a separate module: one stylesheet serves the served and the frozen form). The status line's steps are the live
  segment's own: a run resumed from an earlier save reads its new life, not the dead one's maximum.
- **2026-10-05, after the second review.** The Analyzer's routes are `GET /analyzer`, `GET /api/engines`, `POST
  /api/read` (a position read by one net and compared with another, the panel composed server-side) and `POST
  /api/analyze` (the old analyzer's request, unchanged); the old `POST /trace` has no caller and is dropped (the engine
  layer keeps its op, moved unchanged). Keys live in `games.js` and `analyzer.js` (no separate `keys.js`). The browser
  steps with turn facts the server sends (each stone's owner and turn, each turn's first ply); it derives none. A game's
  payload carries every position's sentences, so a 256-ply sampled game is about 160 KB (above observatory §3.6's
  50 KB) and the server keeps 24 of them. The owner rule appears twice in the package: in the moved engine layer
  (`engine/position.py`, unchanged by design) and in the readers. Gate 10 records the retired tools under
  `DISSOLVED_PATHS`, as it records every removed path, rather than finding no reference to them in history.
- **2026-10-05, after deployment (operator's asks): turns, a movable board, htttx.** The transports step by turn: the
  arrows and the buttons go from one turn's start to the next, and Shift with an arrow steps one stone in Games. At a
  turn's start Games says what the bot did with each of its stones, and the board numbers them 1 and 2. The Analyzer
  reads each net's whole turn: its choice now and, with two stones to place, its choice on the position after that
  one. The choice is the search's when it searched, else the net's highest prior. So the second stone costs a second
  read at the same depth. The read net's two stones are drawn numbered, and both nets' turns are named in the verdict.
  Every board pans by dragging and zooms with the wheel or its +, − and Fit buttons. A drag never places a stone, and
  the view survives stepping. A board shows at least 15 cells across and 13 rows down, so a few stones are never
  drawn huge. The Analyzer imports a game in the HeXO site's htttx notation (`version[1];`, then `N. [q,r][q,r];` a
  turn, after the origin stone the site places): `POST /analyzer` replays it through the engine's board and refuses
  a line, a number or a stone that does not replay, naming it. Its routes moved to `analyzer_routes.py` with the
  import. "Copy position" copies htttx whenever the position starts at the origin.
