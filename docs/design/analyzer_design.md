# ANALYZER — an interactive position analyzer over the engine: analysis, design, plan (2026-09-19)

Status: **proposal for a ruling** (ANALYZER-1). Nothing here is landed; no code, no producer, no
gate moved. Read together with `docs/design/repo_design.md` §1 and its R333(d) / R352(g)
amendments, `docs/design/observatory_research.md` §"Takes and rejects" (the display conventions
this reuses), `docs/governance/CARDS.md` (DASH-2) and R344(d).

The job: import or create a Hex Tac Toe position, run one or more of OUR engines on it, and see
how each evaluates it and where it would place its stones — for general analysis of the bot and
for finding defects (a missed block, a policy that ignores a five, a value that flips under a
rotation). The operator's constraints, verbatim in spirit: not bloated, clean and usable, not
"AI slop", integrated cleanly or — if it makes more sense — in a separate repo.

## 0. What exists, and what was measured, before forming an opinion

**The engine is done.** `DeployHeadPlayer` (`src/mantis/arena/deploy_head.py`) searches a
`Board` with the run's own search kind and leaves `last_root = (root_value, [(cell, idx, prior,
visits, q), …])`. `build_candidate_player` (`src/mantis/eval/worker.py`) composes it from a
`LocalInferenceEngine` by a closed match on the declared representation. A stamped `.ckpt`
carries `config`, `metadata.arch`, `metadata.encoding_name`, `step` and `run_id`
(`src/mantis/train/checkpoints.py`), so one file is enough to rebuild the deploy-matched head:
the resolve helpers (`resolve_fused_graph_caps`, `resolve_inference_batching`,
`resolve_leaf_build_threads`) already take a plain config dict. `Board` exposes the tactical
oracle (`winning_moves`, `threat_moves`, `forced_win_move(depth)`, `get_threats`,
`find_winning_line`) and `mantis.diagnostics.tactics.analyze` gives the census vocabulary
(`RowTactics.forced(k)` → `win | block | lost1 | quiet` with the cell sets).

**The board is done.** `tools/viewer/html.py` carries the hex SVG (axial → pointy-top), stones
with ply numbers, last-stone rings, the win line, a visits heatmap, the warm-neutral palette with
dark mode, keyboard and swipe. No framework, no build step, one page.

**The conventions are decided.** `observatory_research.md` ruled on layout, colour, absence
grammar, key map and the KataGo-shaped root JSON. Several of its REJECTS — "policy overlay: no
producer on the record", "PV: not stored", "no deeper-on-demand" — were about a record reader.
The analyzer IS the producer: it runs the net and the search, so those become its features.

**Governance.** A display surface is "absent" because it would WATCH A RUN (R333(d) amendment,
clause 2). The analyzer watches nothing: it reads immutable checkpoints (R3) and answers on
demand. It needs a process (inference is Python + Rust), and a loopback stdlib server is the
standing order — R344(d), card DASH-2, ordered and unbuilt. Every admission is one R9 amendment
in the same commit as the code. No display code under `src/mantis`; `tools/` is the decided home
(`observatory_design.md` §1.5); gates 14–17 scan it; `tests/tools/conftest.py::load_tools_package`
loads a `tools/<pkg>` by path.

**Measured** (throwaway probe, the operator's CPU host — 8 cores / 16 threads, CPU torch —
`run8_00018000_98fe0e8c.ckpt`, 302 466 params, `puct` deploy kind, `leaf_batch_size` 8):

| what | time |
|---|---|
| `load_checkpoint` + `build_net` + `LocalInferenceEngine` | 50 ms |
| raw net read on the root (one batch) | 143 ms |
| search, 32 sims | 1.7 s |
| search, 128 sims | 3.3 s |
| search, 256 sims (the run's `eval.gate.deploy_sims`) | 5.6 s |
| search, 512 sims | 11.7 s |

≈ 22 ms per sim on CPU. On the box (CUDA) the same searches are sub-second, reached through
`ssh -L`. This fixes the interaction model: **two tiers** — the net's prior and value land on
every edit, the search lands seconds later — and the page must show the search as a STATE
("searching · 256 · 3.1 s"), never as a stale number.

Three facts found on the way, to carry rather than rediscover: `best_model.pt` is the worker-side
SNAPSHOT format (`load_model_snapshot`), not a stamped checkpoint — `load_checkpoint` refuses it
by name; `Board.ply` / `current_player` / `moves_remaining` are PROPERTIES while
`src/mantis/_engine.pyi` declares them as methods (stub drift; fixed on contact, one line each);
and **`MCTSTree.root_raw_value()` is 0.0 under `puct`** at every budget (the kind keeps no raw
values; `crates/mantis-search/src/mcts/mod.rs` says so) while `gumbel` keeps it — measured on the
same checkpoint: puct 1 sim `root_value` +0.0644 / `root_raw_value` +0.0000; gumbel 1 sim +0.0644 /
+0.0644, 64 sims −0.0377 / +0.0644. Reading it off a puct tree would draw a silent zero, so the
analyzer never reads it: the net's raw value is a 1-sim search's `root_value()`, under both kinds.
The same probe showed `root_visits` 79 for a 64-sim puct search (terminal and transposition hits
add visits), so the record carries the sims REQUESTED and the visits the tree REPORTS, both.

## 1. ANALYSIS

### 1.1 Separate repo? No.

The backend is entirely mantis internals that move per ruling — `mantis._engine`,
`LocalInferenceEngine`, `load_checkpoint`, the encoding registry, the config resolvers; contract
v33 landed three days before this document. A second repo would pin mantis and drift the way R9
forbids, or track HEAD and be a worktree by another name. The renderer, the palette and the key
map already live here. This is ≈ 2 000 lines of instrument, not a product; §1.5's rule —
instruments over a record, not products — places it under `tools/`. A separate repo makes sense
only if this becomes a player-facing HeXO product, which is the opposite of a debugging
instrument.

### 1.2 Three shapes, costed

| | (A) `tools/analyzer/`: stdlib server on loopback + one page, engine in-process | (B) offline report CLI: position in, one HTML out | (C) in-browser inference: ONNX export + wasm search |
|---|---|---|---|
| create positions interactively | yes | no — every edit is a CLI round trip | yes |
| the run's OWN search (deploy-matched) | yes, `DeployHeadPlayer` | yes | **no** — a re-implementation, the parity trap a debugging tool exists to avoid |
| new dependency / toolchain | none (`http.server`, `json`, `threading`) | none | onnx export of a GNN with scatter ops, a wasm build of `mantis-core` + `mantis-search`, JS glue |
| `repo_design` amendment | one, narrow: a loopback socket, no watching | one line | a §1 tree change |
| strix beside mantis | yes, the vendored driver | yes | no |

**(B) is rejected** — it fails the "create positions" job. **(C) is rejected** — a toolchain, and
not the run's own search. **(A) is the recommendation**, and the ruling this document asks for.

### 1.3 Decisions taken with the operator (2026-09-19)

- Engines in v1: mantis stamped checkpoints **and strix**; several engine slots on one page.
- Editing: **play-order** (click = the next stone) **plus import** (paste, deep link, and a
  "copy position" affordance in the viewer). No free placement, no variations.
- Instruments in v1: the **tactics oracle line**, the **symmetry sweep**, the **raw-vs-searched
  value trace**. The value readout, the ranked children and the visits / prior / Q overlays are
  always in. NOT in v1: record-game import with the recorded root beside a fresh one.

## 2. DESIGN

### 2.1 Layout

```
tools/analyzer.py              the shim, like game_viewer.py: `python tools/analyzer.py …`
tools/analyzer/
  __init__.py
  cli.py         serve (default) | once. --checkpoints DIR (repeatable) --strix
                 --bind 127.0.0.1 --port 8766 --device cpu|cuda --threads N --timeout-sec 120
  engines.py     discovery over the --checkpoints dirs; EngineInfo; lazy load + cache;
                 the composition load_checkpoint → build_net → LocalInferenceEngine → build_candidate_player
  strix.py       the strix engine over mantis.bots.strix's DriverTransport + the driver's `analyze` op
  position.py    moves ↔ Board; the compact text form; refusals by name; the legal window
  analysis.py    the ONE record: raw (a 1-sim search) + search + the tactics verdict; perspective stated once
  instruments.py the 12 hex symmetries + one translation and the sweep; the tactics verdict helpers
  serve.py       ThreadingHTTPServer; routes; ONE analyst thread fed by a queue
  html.py        the page shell; inlines web/analyzer.css + web/analyzer.js at render → one page
  web/analyzer.css
  web/analyzer.js
tests/tools/test_analyzer_<module>.py     one per module, flat (R5); checkpoints minted in tmp_path, no fixture files
Makefile: `analyzer` target (CHECKPOINTS=… [STRIX=1] [PORT=…])
```

Estimate: ≈ 900 lines Python, ≈ 450 CSS + JS, ≈ 500 tests, + 40 in `tools/strix_driver.py`,
+ 3 in `tools/viewer/html.py`. Every `.py` under the R8 cap; a file that grows past it carries a
reason, never a count.

**One analyst thread.** `MCTSTree` is `unsendable` (thread-affine) and `ThreadingHTTPServer`
runs each request on its own thread. So every engine, every tree and every strix transport lives
on ONE worker thread; a handler enqueues `(request, Future)` and waits up to `--timeout-sec`.
Concurrency is 1 by design and the page says "queued" when a search is in flight. Nothing in the
server cancels a running search: budgets are bounded and the page discards stale replies by a
monotonic `seq` it stamps on every request.

**No `src/mantis` display code.** `tools/analyzer/` imports `mantis.*`; nothing imports it.
`tools/` packages do not import each other (path-loaded); the analyzer needs nothing from
`tools/viewer` — the owner of a ply is `((ply + 1) >> 1) & 1` in JS and `Board.current_player`
in Python, and the win line is `Board.find_winning_line()`.

### 2.2 Engines

**Discovery** (`engines.py`): walk each `--checkpoints DIR` for files matching
`mantis.train.bundle_receipts.CHECKPOINT_NAME_RE` (`<run>_<step8>_<sha8>.ckpt`); each becomes an
`EngineInfo{id (the stem), path, run_id, step, sha8}`, read from the NAME only — the stamp is
opened on first use. `best_model.pt` files are listed as a stated gap — "snapshot format, no
stamp; its `.provenance.json` names step N of run R" when that sidecar exists — and are not
loadable here. Strix, when `--strix` is given: one `EngineInfo{id: "strix", …}` whose
availability comes from `mantis.bots.strix.strix_availability()`; unavailable → listed with that
reason, which names `make vendor.strix`.

**Loading** (first use, cached for the process): `load_checkpoint(path)` → `lookup(encoding_name)`
→ `build_net(metadata.arch)` + `load_state_dict` → `LocalInferenceEngine(net, device,
encoding_spec, fused_graph_caps, inference_batching, max_in_flight=leaf_batch_size,
leaf_build_threads)` with every knob read from `ck.config` through the resolve helpers — the
worker's own composition, not a second one. The engine card carries `run_id, step, sha8,
encoding, search_kind (= config.deploy.search.kind), deploy_sims (= config.eval.gate.deploy_sims),
params, device`. A `DeployHeadPlayer` is built PER REQUEST with the run's `c_visit, c_scale,
q_rescale, leaf_batch_size, gumbel_m` and the request's `sims`; `gumbel_seed` is fixed (0) so a
gumbel engine answers the same position the same way twice. A stamp whose config lacks a key
this needs is REFUSED by the key's name (R1 — no default supplied here).

**Strix** (`strix.py` + the driver): `mantis.bots.strix.locate_strix()` (sha-verified pin) and
`DriverTransport` are reused; `load_request(...)` is sent once. `tools/strix_driver.py` gains an
`analyze` op beside `select`, taking the SAME `{stones, to_move, moves_remaining}` and returning
what `gumbel_mcts_with_diagnostics` already computes — `{legal, improved, visits, move, sims,
origin, ms}` — plus the raw net value of the seated position from `_eval_fn([game])`. What
strix's diagnostics do not expose (a per-child Q, a searched root value) is a STATED GAP on the
strix card, never a zero; if `_rest` turns out to carry a root value it is added at implementation
and the record says which. Strix's `m_actions` and its solver posture come from the pin's rung as
`mantis.bots.strix` composes the `load` request; the `analyze` op takes an optional `sims` that
rebuilds the driver's `MCTSConfig` for that request alone, so the page's budget chips apply to
strix too and the card states the sims strix actually ran.

### 2.3 Position model (`position.py`)

The position is a move list `[[q, r], …]` in axial coordinates — contract #11's own form — with
a ply cursor. Text form for paste and deep links: `q,r;q,r;…` (`0,0;1,0;0,1`); a JSON list is
accepted on paste too. `moves → Board`: `Board.with_encoding_name(encoding)` then `apply_move` in
order; the server REFUSES by name — `illegal move (q, r) at ply k`, `move after the game ended at
ply k`, `cell outside the legal radius` — and never truncates silently. The position record:
`{moves, ply, to_move (1|2), moves_remaining (1|2), legal (count), winner (null|1|2),
win_line (cells|null), legal_window (the empty cells within the legal radius — drawn faint)}`.
The owner of ply p is p1 for p = 0, then turns alternate in pairs (`Ply::turn`).

The page: click or tap a cell = the next stone at the cursor; placing while the cursor is behind
the end REPLACES the tail (stated in the help line — no variations in v1); `u` undoes the last
stone; ← → Home End move the cursor. `tools/viewer/html.py` gains one key, `c`, that copies the
current game's compact text up to the current ply to the clipboard — the viewer's join to the
analyzer, no URL assumed.

**Deep link:** `?e=<engine id>[,<engine id>…]&m=<compact moves>&ply=N&sims=256&ov=v|p|q&tac=1`
kept with debounced `replaceState`. Reproducible given the same checkpoint: a stamp is immutable
and `gumbel_seed` is fixed.

### 2.4 The analysis record (`analysis.py`) — KataGo-shaped, the perspective stated once

```
{
  "engine":      {"id", "run_id", "step", "sha8", "encoding", "search_kind", "deploy_sims", "device"},
  "position":    {"moves", "ply", "to_move", "moves_remaining", "legal", "winner", "win_line"},
  "perspective": "to_move",
  "raw":         {"value", "argmax": [q, r], "policy": [[q, r, p], …], "ms"},
  "search":      {"sims", "root_visits", "kind", "root_value", "argmax": [q, r],
                  "children": [[q, r, prior, visits, q], …], "ms"}
                 | {"absent": "sims=1 requested"} | {"absent": "position is terminal"},
  "tactics":     {"class": "win|block|lost1|quiet", "k", "cells": [[q, r], …],
                  "opp_fours": [[[q, r], …], …], "forced_win_move": [q, r] | null,
                  "verdict": {"raw": "…", "search": "…"} , "derivation": "mantis.diagnostics.tactics.analyze(k=…)"},
  "symmetry":    {"n": 13, "value_min", "value_max", "spread", "argmax_agreement": "k/13",
                  "worst": {"map": "rot60^3·refl", "value", "argmax": [q, r]}}
                 | {"absent": "not requested"},
  "elapsed_ms"
}
```

- **raw** is a 1-sim search through the same `DeployHeadPlayer`: its `root_value()` is the net's
  value (the root's own evaluation is the only backup) and `get_root_children_info()`'s priors
  are the decoded policy over the legal children. `root_raw_value()` is NEVER read (§0: 0.0 under
  puct). On a `sims > 1` request the analyst runs the 1-sim read FIRST and then the search, so
  every record is complete and a deep link reproduces the whole of it; the cost is one extra
  raw read (25–125 ms measured) beside a multi-second search. One code path for raw and
  searched; no second decode. `search` carries `sims` (requested) and `root_visits` (reported).
- **perspective**: every value is for the side to move at this ply. A test pins it: a WIN1 child
  is terminal, so the searched root reads ≈ +1 for the mover regardless of the net. The page
  states the perspective ONCE, beside the readout.
- **tactics**: `mantis.diagnostics.tactics.analyze(q, r, p, mover, k)` on `Board.get_stones()`
  with `k = moves_remaining`; `RowTactics.forced(k)` gives the class and the cell set. The
  verdict compares an argmax to the class: `win` → "argmax WINS" | "argmax MISSES WIN at …";
  `block` → "argmax BLOCKS" | "argmax MISSES THE BLOCK — block set …"; `lost1` → "no block exists
  (LOST1)"; `quiet` → "quiet". One verdict for the raw argmax and one for the searched argmax.
  `Board.forced_win_move(2)` rides beside it as the engine's own detector.
- **symmetry** (`instruments.py`): the 12 symmetries of the hex grid in axial coordinates —
  rotation by 60° `(q, r) → (−r, q + r)` composed 0–5 times, each with and without the reflection
  `(q, r) → (r, q)` — plus ONE translation by `(+3, −2)`. Each map is applied to the whole move
  list (cadence and ownership are preserved), a raw read is taken, and the argmax is mapped back
  by the inverse. The panel reports `value_min / value_max / spread`, how many of the 13 argmaxes
  agree with the identity's, and names the worst map. 13 raw reads ≈ 1.3 s on CPU; on demand,
  cost stated. A net that is exactly equivariant reads spread 0; the panel draws the number, the
  reader draws the conclusion.
- **value trace**: client-orchestrated. The page sweeps `POST /analyze` with `sims=1` over every
  ply of the current move list (raw line, instant), then again at the TRACE budget (default the
  quick chip, 32 sims) for the searched root; plies not yet computed are drawn as gaps, never
  interpolated (Sabaki's gap grammar, decided). The chart is in P1's FIXED perspective —
  `v_p1 = v if to_move == 1 else −v` — and its caption says so; the readout stays the mover's.
  `viewBox="0 0 <plies> 100"`, `vector-effect: non-scaling-stroke`, the current ply a dashed line.

### 2.5 Routes (`serve.py`)

| route | in | out |
|---|---|---|
| `GET /` | — | the page (`html.py`) |
| `GET /engines` | — | `[EngineInfo…]` incl. stated gaps and strix's availability |
| `POST /analyze` | `{engine, moves, sims, symmetry: bool, seq}` | the record; `400` with `{"refused": reason}` on a position refusal; `503` `{"refused": "engine unavailable: …"}`; `504` on the analyst timeout |

`ThreadingHTTPServer`, bind `127.0.0.1` by default; `--bind 0.0.0.0` is documented in `--help`
as unsafe (no auth, no TLS) and never the default. `once` runs one `/analyze` body from the
command line and prints the record — the same function, for scripts and for tests without a
socket.

### 2.6 The page (`html.py`, `web/`)

```
┌ mantis analyzer · run8 · 18 000 · 98fe0e8c · puct · cpu ─────── keys: ← → u v p q t s g ? ┐
│                                   │ engine [run8_00018000 ▾] [+ slot]   budget (32)(256•)(1024) [n]│
│        ┌ hex board ┐              │ ply 7 · p2 to move · 2 left · 280 legal                         │
│        │ stones w/ ply numbers    │ 0,0;1,0;0,1;-1,0;0,-1;2,0;1,1                  [copy]           │
│        │ overlay: visits|prior|Q  │ ── run8 @18 000 ── values for the side to move ─────────────── │
│        │ argmax ringed            │  raw   −0.019   argmax (−2,1)        43 ms                       │
│        │ tactics outlines (t)     │  256   −0.049   argmax (−2,1)        5.6 s      ("searching…")  │
│        │ legal window faint       │  CHECK · block {(2,1),(3,0)} · search argmax (2,1) BLOCKS        │
│        └──────────┘               │  cell     prior   visits    Q                                    │
│  hover: (q,r) · p 26 % · n 169 · Q −0.03   (−2,1)   26 %   169   −0.03  ●                          │
│                                   │  (1,−2)   24 %   159   −0.02                                     │
│                                   │  … 8 more under the floor                                        │
│                                   │ symmetry (s): not requested — 13 net reads, ≈ 1.3 s on cpu       │
│                                   │ value trace (g): not requested                                   │
└───────────────────────────────────┴─────────────────────────────────────────────────────────────────┘
```

- **Grid.** Desk: one CSS grid `board | side` (side ≈ 380 px). Phone (portrait, decided by aspect
  ratio): one column `board / controls / readout / cards`, the board sized to keep the readout on
  screen. Pure CSS, no script.
- **Tokens.** The viewer's `:root` custom properties VERBATIM (`--bg --fg --mute --line --p1 --p2
  --hi --win --heat`) plus `--pos` / `--neg` for the Q overlay; `prefers-color-scheme: dark` as
  the viewer. `system-ui`, 12–14 px, `font-variant-numeric: tabular-nums` on every number. No
  shadows, gradients, rounded cards, icons or emoji; borders are `1px var(--line)`.
- **Board.** The viewer's SVG: empty cells around the played area, the LEGAL WINDOW drawn faint
  (a real constraint of the engine — the radius from the stamp), stones with ply numbers, the
  last two stones ringed, the win line outlined. Tap-to-cell by the inverse map + `cube_round`
  (Red Blob, decided). Hover/focus on a cell writes `(q, r) · p · n · Q` under the board — no
  floating tooltips.
- **Overlays** (one at a time; keys `v` visits, `p` prior, `q` Q): visits and prior share the
  sequential `--heat` at OGS's rule `alpha = 0.25 + 0.55 × x / max`, the number printed above a
  floor (visits ≥ 2, prior ≥ 2 %; the floor stated on the panel and never blanking the board); Q is
  diverging (`--neg` … transparent at 0 … `--pos`) over the children with visits ≥ 2. The active
  card's argmax is RINGED; the other cards' argmaxes are small labelled markers (engine id
  initials). `t` toggles the tactics outlines: the class's cells in `--win`, the opponent's fours'
  empties dotted in `--hi`.
- **Engine cards.** One card per slot: header `run8 @18 000 · puct · cpu`, the two-row readout
  (raw / searched: value, argmax, ms), the tactics line with its derivation, the ranked children
  (sorted by visits, or by prior when raw only; 12 rows + "n more under the floor"; a row click
  highlights its cell), then the on-demand panels. The first card is active; clicking a card
  header makes it the overlay's source.
- **Auto-analyze** on by default: every edit sends `sims=1` at once and the budget's search
  after; the searched row shows `searching · 256 · 3.1 s` with a live elapsed counter, or
  `queued` while another request is in flight. Off: a "search" button. The deep chip (1024 sims,
  ≈ 23 s on CPU) says its cost in its title.
- **Keys** (single keys, no modifiers, printed in the header and on `?`): ← → Home End step, `u`
  undo, `v` `p` `q` overlays, `t` tactics, `s` symmetry sweep, `g` value trace, `c` copy position,
  `?` the key sheet. Keys are ignored while an input has focus.

### 2.7 Empty, refused, absent, stale — the vocabulary

| state | what the page shows |
|---|---|
| empty board | the legal window (25 cells), the raw read runs (an empty board is a position) |
| engine loading (first use) | `loading run8 @18 000 …` on the card; strix's subprocess `load` can take seconds and says so |
| searching / queued | the state and the elapsed time in the searched row; the raw row stays live |
| refused | the reason verbatim from the server, in `--hi`, on the card; the board is unchanged |
| absent panel | `symmetry: not requested — 13 net reads, ≈ 1.3 s on cpu` — a cost, never a blank |
| strix unavailable | the card carries `strix_availability()`'s reason and names `make vendor.strix` |
| stale reply | discarded by `seq`; nothing drawn |
| terminal position | `search: absent — the game is over (p1 wins)`; the raw read still shown |

Absent is not zero: no panel ever draws an empty axis, a flat line or a bare 0 for something that
was not computed.

### 2.8 Deliberately NOT in v1

Variations and branches (one line, tail replaced); free placement (`ring_audit.reconstruct` is
the seam when wanted); reading game shards in the server (the viewer's `c` is the join);
streaming search progress and cancellation (an `on_batch` hook on `DeployHeadPlayer` is the
seam — a one-argument, None-guarded addition to a deploy-matched head, so it is its own small
ruling); sealbot; the observatory's run pages; any `src/mantis` display code; a `freeze` (the
deep link IS the record of an analysis, given the checkpoint).

## 3. Tests — the load-bearing ones

- `test_analyzer_position.py`: text ↔ list round trip; the three refusals by name; the owner
  cadence agrees with `Board.current_player` over 40 random legal plies; the legal window equals
  `Board.legal_moves()`.
- `test_analyzer_engines.py`: discovery lists stamped names and reports snapshots as a stated
  gap; a stamp missing a needed key is refused by the key's name; the composition on a
  stamped checkpoint MINTED in `tmp_path` by `save_checkpoint` with a tiny `GnnArchV2` (as
  `tests/train/test_arch_stamp_authority.py` does — no fixture file, nothing near the 1 MB gate)
  runs `sims=1` and `sims=8` and the record has every field of §2.4, with `raw.value` equal
  under both search kinds.
- `test_analyzer_analysis.py`: the PERSPECTIVE pin (a WIN1 position for the mover → searched
  root ≥ 0.9 at sims ≥ 8); `search.absent` on a terminal position; the tactics verdicts on
  constructed positions — WIN / MISSED WIN / BLOCK / MISSED BLOCK / LOST1 / QUIET — with a stub
  argmax.
- `test_analyzer_instruments.py`: the 12 maps form a group (closure, inverses, identity at
  index 0); the translation's inverse; a stub engine whose value depends only on the stone
  count gives spread 0 and 13/13 agreement; the worst map is named.
- `test_analyzer_serve.py`: with a STUB analyst (no torch): `GET /engines`, `POST /analyze`
  happy path, `400` on a refusal, `504` on a timeout, `seq` echoed; the default bind is loopback.
- `test_analyzer_strix.py`: the driver's `analyze` op — skipped LOUDLY naming `make vendor.strix`
  when the vendor is not fetched (the house convention), otherwise the shape and that `improved`
  sums to 1 over `legal`.
- `test_analyzer_html.py`: the page renders with the viewer's tokens present and the CSS/JS
  inlined; the viewer's `c` key exists (`test_game_viewer.py` gains one assertion).
- JS is not unit-tested (house convention: the viewer's script is not either); every derivation
  is Python-side and tested there.

## 4. PLAN — phases, each a reviewable unit with tests

1. **Position + engines + the record, headless.** `position.py`, `engines.py`, `analysis.py`,
   `instruments.py`, `cli.py once`. The perspective pin lands here. Gate set at exit.
2. **The server + the page.** `serve.py`, `html.py`, `web/`, the shim, the Makefile target, the
   viewer's `c`. The R9 amendment (§5) in THIS commit. Gate set at exit.
3. **Strix.** The driver's `analyze` op, `strix.py`, the strix card. Its own commit, so the
   mantis-only tool is usable before the vendor is built anywhere.

The full local gate set (`make gates.exit`) at each phase exit; targeted tests while iterating.
Commits are one line, no trailers (R36/R47). Rustfmt: no `.rs` is touched.

### 4.1 Risks named

- CPU latency at the deploy budget (5.6 s) makes auto-analyze feel slow on the operator's host;
  the mitigations are the quick chip, the instant raw tier and the elapsed counter — and the box.
  If that is not enough, the `on_batch` hook (§2.8) is the next step, ruled separately.
- Strix's subprocess holds its own torch; two torches in one process tree on the box is fine
  (the eval worker already does this) but the analyzer's memory is stated on the card.
- `LocalInferenceEngine` was built for many workers; at one request in flight its collector waits
  its own deadline. Measured above as acceptable (143 ms raw); if the graph path's batching
  deadline dominates at sims=1, `inference_batching` is read from the stamp and stated, not
  tuned here.

## 5. The `repo_design` amendment (phase 2's commit carries it; proposed text)

> ### AMENDMENT — R<nnn>, ANALYZER-1: an interactive position analyzer is ADMITTED on the
> viewer's terms plus one loopback socket
>
> **R<nnn>.** §1 keeps display surfaces absent because each would watch a run. ANALYZER-1 is a
> position analyzer: a position in (typed, pasted, or a deep link), one or more engines' readings
> out. Under R9 it lands as an amendment in the same commit as the server.
>
> 1. **What is admitted, narrowly.** `tools/analyzer.py` (`make analyzer CHECKPOINTS=… [STRIX=1]`;
>    the implementation is `tools/analyzer/`) — a stdlib `ThreadingHTTPServer` bound to loopback
>    by default (`--bind 0.0.0.0` is documented as unsafe, never the default), serving ONE page
>    and answering `POST /analyze` by running the run's own `DeployHeadPlayer` on stamped
>    checkpoints (immutable, R3) and, opted in, the vendored strix pin through its driver. It
>    watches no run, reads no run record, adds no producer and writes no artifact. It is dev-only
>    tooling under `tools/`, beside the dashboard and the viewer.
> 2. **What stays absent, unchanged.** The web dashboard and the TUI monitor; the coupling rule
>    of the R333(d) amendment is untouched — this process holds no file open and polls nothing.
> 3. **The rule the tool carries.** Every number on the page is derived in Python from a named
>    engine call and carries its derivation; values are for the side to move, stated once; a
>    search not yet run is a STATE with its elapsed time, never a stale number; a panel not
>    requested is a stated cost, never a blank; a position the engine refuses is refused by name;
>    strix's gaps (no per-child Q) are stated on its card. Absent is not zero, applied to a live
>    engine.
> 4. **Measured at landing:** (filled at landing — engine load, raw read, and the four search
>    budgets on the host that ran the gate set, in the same table as §0 of the design.)
> 5. **Contracts #11 and §4.7 are unchanged.** Nothing is emitted; nothing is read from a record.

The ruling number is the operator's; this document only drafts the text.
