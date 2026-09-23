# ANALYZER — an interactive position analyzer over the engine: analysis, design, plan (2026-09-19)

Status: **proposal for a ruling** (ANALYZER-1); the code is on branch `worktree-analyzer` awaiting the ruling number and the ff. No producer, no
gate moved. Read together with `docs/design/repo_design.md` §1 and its R333(d) / R352(g)
amendments, `docs/design/observatory_research.md` §"Takes and rejects" and
`docs/design/observatory_design.md` §2.4–2.7 (the display conventions this reuses),
`docs/governance/CARDS.md` (DASH-2) and R344(d). Revised 2026-09-19 after an independent
review against the code; §0 carries what that review found.

The job: import or create a Hex Tac Toe position, run one or more of OUR engines on it, and see
how each evaluates it and where it would place its stones — for general analysis of the bot and
for finding defects (a missed block, a policy that ignores a five, a value that flips under a
rotation). The operator's constraints: not bloated, clean and usable, not "AI slop", integrated
cleanly or — if it makes more sense — in a separate repo.

## 0. What exists, and what was measured, before forming an opinion

**The engine is done.** `DeployHeadPlayer` (`src/mantis/arena/deploy_head.py`) searches a
`Board` with the run's own search kind and leaves `last_root = (root_value, [((q, r), idx,
prior, visits, q), …])`. `build_candidate_player` (`src/mantis/eval/worker.py`) composes it from
a `LocalInferenceEngine` by a closed match on the declared representation; importing that module
as a library has no side effects (`mantis.diagnostics.acceptance_witness` already does). A
stamped `.ckpt` carries `config`, `metadata.arch`, `metadata.encoding_name`, `step` and `run_id`
(`src/mantis/train/checkpoints.py`), so one file rebuilds the deploy-matched head; the resolve
helpers (`resolve_fused_graph_caps`, `resolve_inference_batching`, `resolve_leaf_build_threads`,
`resolve_deploy_search_kind`) take a plain mapping, and `stamped_checkpoints(dir)`
(`src/mantis/train/bundle_receipts.py`) already walks a directory for stamped names. `Board`
exposes the tactical oracle (`winning_moves`, `forced_win_move(depth)`,
`find_winning_line`, `is_legal`; `threat_moves`/`get_threats` since removed, unused) and `mantis.diagnostics.tactics.analyze` gives
the census vocabulary (`RowTactics.forced(k)` → `win | block | lost1 | quiet` with the cells).

**The board is drawn.** `tools/viewer/html.py` carries the hex SVG (axial → pointy-top), stones
with ply numbers, last-stone rings, the win line, a visits heatmap, the warm-neutral palette
with dark mode, keyboard and swipe. No framework, no build step, one page. Path-loaded `tools/`
packages cannot import each other, so the analyzer's `web/analyzer.js` FORKS that SVG code
(≈ 50 lines) rather than sharing it; the tokens are copied verbatim.

**The conventions are decided.** `observatory_research.md` ruled on layout, colour, absence
grammar, key map and the KataGo-shaped root JSON. Several of its REJECTS — "policy overlay: no
producer on the record", "PV: not stored", "no deeper-on-demand" — were about a record reader.
The analyzer IS the producer: it runs the net and the search, so those become its features.

**Governance.** A display surface is "absent" because it would WATCH A RUN (R333(d) amendment,
clause 2): the criterion is COUPLING. The analyzer watches nothing — it reads immutable
checkpoints (R3) and answers on demand — but it needs a process (inference is Python + Rust) and
so a socket, which no admitted tool has opened yet; R344(d) shows a loopback stdlib server is a
shape the house has accepted in principle (DASH-2 orders one over the RUN RECORD for the viewer;
that card is neither discharged nor engaged by this). Every admission is one R9 amendment in the
same commit as the code. No display code under `src/mantis`; `tools/` is the decided home
(`observatory_design.md` §1.5); gates 14–17 scan it; `tests/tools/conftest.py::load_tools_package`
loads a `tools/<pkg>` by path.

**Measured** (throwaway probes, the operator's CPU host — 8 cores / 16 threads, CPU torch —
`run8_00018000_98fe0e8c.ckpt`, 302 466 params, `gnn_axis_r8`, radius 8, `puct` deploy kind,
`leaf_batch_size` 8):

| what | time |
|---|---|
| `load_checkpoint` + `build_net` + `LocalInferenceEngine` | 50 ms |
| `MCTSTree()` construction (a 4 000 000-node pool, ≈ 140 MB) | 45–50 ms |
| raw net read through a CACHED quiescence-off tree (§2.4) | 31 ms |
| search, 32 sims | 1.7 s |
| search, 128 sims | 3.3 s |
| search, 256 sims (the run's `eval.gate.deploy_sims`) | 5.6 s |
| search, 512 sims | 11.7 s |

≈ 22 ms per sim on CPU. On the box (CUDA) the same searches are sub-second, reached through
`ssh -L`. This fixes the interaction model: **two tiers** — the net's prior and value land on
every edit, the search lands seconds later — and the page shows the search as a STATE
("searching · 256 · 3.1 s"), never as a stale number.

**Facts the review found, each of which changed the design** (verified in the code and by probe):

1. **Root quiescence.** `finish_expansion` applies `apply_quiescence` to the ROOT's own value
   before backup and stores the CORRECTED value in `raw_values` (`crates/mantis-search/src/mcts/
   backup.rs`); `DeployHeadPlayer._fresh_tree` builds `MCTSTree()` with `quiescence_enabled=True`.
   So a 1-sim `root_value()` is the solver's verdict wherever it fires — measured on a CHECK
   position (p1 open four, p2 to move at k=2): net −0.2987, deploy head 1 sim −0.5987
   (`quiescence_fire_count` 1), 64 sims +0.3429 (3 fires). And under `puct` `root_raw_value()`
   is 0.0 at every budget (the kind keeps no raw values). The raw tier therefore reads the NET
   through a `MCTSTree(quiescence_enabled=False)` driven by the worker's own expand collaborator
   — measured −0.2987, equal to `engine.infer_ls` — and the searched tier is stated to INCLUDE the
   deploy head's quiescence override, with the fire count printed as its own instrument.
2. **Terminal positions.** `select_move` RAISES on a won board (no root children, every kind
   and budget); strix's MCTS refuses too. The analyzer checks for a win BEFORE any engine call.
3. **`Board.apply_move` refuses only an occupied cell** — no radius check, no game-over check
   (`crates/mantis-core/src/board/state/core.rs`: "those constraints are the caller's"); a stone
   after the six is ACCEPTED and `check_win()` then reads False (it checks the last move only).
   The position builder checks `is_legal` and the win after every stone.
4. **`Board` and `MCTSTree` are both `unsendable`**; touching either from another thread is a
   Rust panic → `PanicException`, a `BaseException` that `except Exception` does not catch. The
   analyst thread builds every `Board` itself and guards each request with `BaseException`.
5. **The argmax at 1 sim is not the policy argmax**: under puct `get_top_visits(1)` sorts
   all-zero visits; under gumbel it is a seeded Gumbel SAMPLE. The raw argmax is `max(prior)`
   over `get_root_children_info()` rows, always.
6. **Player numbering**: the engine's `current_player`, `winner()`, `get_stones()` and
   `winning_moves(player)` use 1 / −1; the viewer's `owner()` 0 / 1.
   The record uses the game record's own strings `"p1"` / `"p2"` and ONE mapping table.
7. **Gumbel's draw** is seeded from `gumbel_seed` mixed with the player's game and move
   counters, so only a FRESH player repeats an answer. A search builds a fresh player.
8. **The empty-board legal set is the 5 × 5 axial square**, not hex-symmetric; the symmetry
   sweep is centred on the first stone so ply 0 stays legal under every map.
9. **strix's `gumbel_mcts_with_diagnostics` returns 7 values** — `(action, improved_policy,
   visit_counts, per_child_q, per_child_prior, candidate_indices, chosen_action_forced_
   candidates)`, `per_child_q` in the root player's view — so strix's card has full children
   rows; its gaps are a searched root value and the perspective of its raw value. Its driver
   refuses a board with no p1 stone.
10. **Gate 14's comment ratchet has ZERO headroom** (`comment_excess 3405/3405`,
    `docstring_excess 13105/13105`, `private_docstring_excess 1506/1506` at HEAD): every new
    docstring is ONE line (`Raises` inline, the viewer/dashboard precedent) and every own-line
    comment run ≤ 2 lines. This is the dominant style constraint on ≈ 1 400 new Python lines.
11. **The worktree trap**: the main checkout's venv resolves `mantis` to the MAIN checkout's
    `src/` and `find_vendor_root()` to its `vendor/`, so a `src/mantis` edit made in a worktree is
    invisible to tests run there. This design edits NOTHING under `src/mantis` (the `_engine.pyi`
    "stub drift" of the first draft was false: the stub already declares `ply` /
    `current_player` / `moves_remaining` as properties). strix's driver path is resolved from the
    analyzer's OWN tree (§2.2), so phase 3 tests run in the worktree too.
12. `best_model.pt` is the worker-side SNAPSHOT format (`load_model_snapshot`), not a stamped
    checkpoint — `load_checkpoint` refuses it by name; `root_visits` exceeds `sims` under puct
    (terminal and transposition re-backups: 79 for 64), so the record carries both numbers.

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

- Engines in v1: mantis stamped checkpoints **and strix**; several engine slots on one page
  (one slot in phase 2, slots in phase 2b — §4).
- Editing: **play-order** (click = the next stone) **plus import** (paste, deep link, and a
  "copy position" key in the viewer). No free placement, no variations.
- Instruments in v1: the **tactics oracle line**, the **symmetry sweep**, the **raw-vs-searched
  value trace**. The value readout, the ranked children and the visits / prior / Q overlays are
  always in. NOT in v1: record-game import with the recorded root beside a fresh one.

## 2. DESIGN

### 2.1 Layout

```
tools/position_analyzer.py     the shim, like game_viewer.py (a distinct name beside the package)
tools/analyzer/
  __init__.py
  cli.py         serve (default) | once. --checkpoints DIR (repeatable) --strix
                 --bind 127.0.0.1 --port 8766 --device cpu --threads N --timeout-sec 120
  engines.py     discovery (stamped_checkpoints); EngineInfo; lazy load + cache; the raw tree; the search
  strix.py       the strix engine over mantis.bots.strix's DriverTransport + the driver's `analyze` op
  position.py    moves ↔ Board on the analyst thread; the compact text form; refusals by name; the legal window
  analysis.py    the ONE record: raw + search + the tactics verdict; perspective stated once
  instruments.py the 12 hex symmetries + the translation and the sweep; the value trace
  dispatch.py    the engine registry + `handle(request) -> {status, body}`; no threads, no HTTP
  serve.py       ThreadingHTTPServer; routes; ONE analyst thread with a superseding queue, the engines closed on it
  html.py        the page shell; inlines web/analyzer.css + web/board.js + web/analyzer.js at render → one page
  web/analyzer.css
  web/board.js   the hex renderer forked from tools/viewer (a scene in, SVG out; knows nothing about engines)
  web/analyzer.js the state machine: slots, requests, cards
tests/tools/test_analyzer_<module>.py     one per module, flat (R5); checkpoints minted in tmp_path, no fixture files
tests/tools/conftest.py                   + one session fixture `analyzer` (append-only, the `viewer` pattern)
tests/test_meta_ci.py                     + "analyzer" in the pinned Makefile target set
Makefile: `analyzer` target (CHECKPOINTS=… [STRIX=1] [PORT=…]) + .PHONY
```

Landed (2026-09-20, after an independent review of the code): ≈ 1 000 lines Python, ≈ 300 CSS + JS, ≈ 600 tests,
+ 40 in `tools/strix_driver.py`, + 3 in `tools/viewer/html.py`. Every `.py` under the R8 cap; a file that grows past it carries a
reason, never a count.

**One analyst thread.** `Board` and `MCTSTree` are `unsendable` (§0 item 4) and
`ThreadingHTTPServer` runs each request on its own thread. So every engine, tree, board and
strix transport lives on ONE worker thread: a handler parses JSON only and enqueues `(request,
Future)`; the analyst builds the `Board`, refuses or answers, and the handler waits up to
`--timeout-sec`. Each request runs under `except BaseException` (a `PanicException` included):
the reply is a `500` naming the exception type and the loop survives. **Supersession:** before
starting a request the analyst drains the queue and keeps only the NEWEST request per `client`
(a page-generated id) per engine; superseded requests are answered at once with
`{"superseded": seq}`. So stepping ten plies on CPU costs one search, not ten. Nothing cancels a
RUNNING search: budgets are bounded, the page discards stale replies by `seq`, and the raw tier is
never behind a search because the page requests it separately (§2.4).

**No `src/mantis` display code, no `src/mantis` change at all.** `tools/analyzer/` imports
`mantis.*`; nothing imports it. `tools/` packages do not import each other; the owner of a ply is
`((ply + 1) >> 1) & 1` in JS and `Board.current_player` in Python; the win line is
`Board.find_winning_line()`.

### 2.2 Engines

**Discovery** (`engines.py`): `stamped_checkpoints(dir)` over each `--checkpoints DIR`; each
name becomes `EngineInfo{id (the stem), path, run_id, step, sha8}` — the stamp is opened on
first use. `best_model.pt` files are listed as a stated gap ("snapshot format, no stamp; its
`.provenance.json` names step N of run R" when that sidecar exists) and are not loadable here.
Strix, when `--strix` is given: one `EngineInfo{id: "strix"}` whose availability is
`mantis.bots.strix.strix_availability()`; unavailable → listed with that reason, and the card
appends `make vendor.strix` (the target exists; the reason names the script).

**Loading** (first use, cached for the process, on the analyst thread): `load_checkpoint(path)`
→ `lookup(encoding_name)` → `build_net(metadata.arch)` + `load_state_dict` →
`LocalInferenceEngine(net, device, encoding_spec, fused_graph_caps, inference_batching,
max_in_flight=leaf_batch_size, leaf_build_threads)` with every knob read from `ck.config`
through the resolve helpers and `search_kind = resolve_deploy_search_kind(ck.config)` — the
worker's own composition, not a second one. A stamp whose config lacks a key this needs is
REFUSED by the key's name (R1: nothing is defaulted here). `--device` defaults to `cpu`
explicitly (the box passes `cuda`); `--threads` absent means torch's own default, and the card
states the count in use. The card carries `run_id, step, sha8, encoding, radius, search_kind,
deploy_sims (= eval.gate.deploy_sims), params, device, threads`.

**Two trees per engine.** (i) THE RAW TREE, cached: `MCTSTree(quiescence_enabled=False)` +
`configure_search(kind, c_visit, c_scale, q_rescale)` once; per read `new_game(board)` →
`select_leaves(1)` → the worker's expand collaborator (`_graph_expand_fn(engine, spec)` for
graph; `expand_and_backup` on `engine.infer` for grid) → `root_value()` is the NET's value and
`get_root_children_info()`'s priors are the decoded policy (31 ms measured). (ii) THE SEARCH: a
FRESH `DeployHeadPlayer` per request via `build_candidate_player(engine, sims, …)` with the
run's `c_visit, c_scale, q_rescale, leaf_batch_size, gumbel_m` and `gumbel_seed =
ANALYZER_GUMBEL_SEED` (a named module constant, 0, carried in the record); fresh so a gumbel
answer repeats (§0 item 7). The 45–50 ms and ≈ 140 MB transient of `MCTSTree()` per search is
< 3 % of the smallest budget and is freed with the player; the card says "one tree per search".

**Strix** (`strix.py` + the driver): the interpreter, working directory, checkpoint and pin come
from `mantis.bots.strix.locate_strix()` (sha-verified); the DRIVER PATH is the analyzer's own
`tools/strix_driver.py` (resolved from `strix.py`'s location), so the driver that runs is the
one in this tree (§0 item 11). `load_request(...)` is sent once. `tools/strix_driver.py` gains an
`analyze` op beside `select`, taking `{stones, to_move, moves_remaining, sims?}` — `sims` rebuilds
the driver's `MCTSConfig` for that request alone — and returning `{legal, improved, visits,
per_child_q, per_child_prior, move, sims, origin, raw_value, ms}`, `raw_value` from
`_eval_fn([game])`. The strix card's rows are `(cell, prior = per_child_prior, visits, q =
per_child_q)`; its `search.root_value` is `{"absent": "strix reports no root value"}` unless the
visit-weighted mean of `per_child_q` is shown, LABELLED as derived; the perspective of
`raw_value` is stated as UNVERIFIED on the card until phase 3 pins it with a probe. A board with
no p1 stone is refused by the driver and the card says so ("strix needs a p1 stone").

### 2.3 Position model (`position.py`)

The position is a move list `[[q, r], …]` in axial coordinates — contract #11's own form — with
a ply cursor. Text form for paste and deep links: `q,r;q,r;…` (`0,0;1,0;0,1`); a JSON list is
accepted on paste too. `moves → Board` runs on the analyst thread: `Board.with_encoding_name
(encoding)`, then for every stone in order — `if won: refuse("move after the game ended at ply
k")`; `if not board.is_legal(q, r): refuse("illegal move (q, r) at ply k: occupied or outside
the legal radius")`; `apply_move`; `if board.check_win(): won, winner = True, board.winner()`.
Nothing is truncated silently. The position record: `{moves, ply, to_move ("p1"|"p2"),
moves_remaining (1|2), legal (count), winner (null|"p1"|"p2"), win_line (cells|null),
legal_window (the empty cells the active engine's radius admits — drawn faint)}`. ONE mapping
table: engine `1` ↔ `"p1"`, `−1` ↔ `"p2"`; the engine's own ints go to `tactics.analyze(…,
mover=board.current_player, …)` and `winning_moves(board.current_player)`.

The page: click or tap a cell = the next stone at the cursor; placing while the cursor is behind
the end REPLACES the tail (stated in the help line — no variations in v1); `u` undoes the last
stone; ← → Home End move the cursor. `tools/viewer/html.py` gains one key, `c`, that copies the
current game's compact text up to the current ply — the viewer's join to the analyzer, no URL
assumed.

**Deep link:** `?e=<engine id>[,<engine id>…]&m=<compact moves>&ply=N&sims=256&ov=h|p|q&tac=1`
kept with debounced `replaceState`. On CPU it reproduces the record exactly (a stamp is
immutable, the seed is fixed, a fresh player per search); on CUDA the scatter ops are not
bit-deterministic and the card says so.

### 2.4 The analysis record (`analysis.py`) — KataGo-shaped, the perspective stated once

`POST /analyze` with `sims: 0` is a RAW-ONLY read (the page sends it on every edit, at once);
`sims ≥ 1` is raw + search in ONE record (the page sends it debounced, at the budget). Every
record carries `raw`, so a deep link reproduces the whole of it. The reply envelope is
`{"seq", "ok": true, "record"}` | `{"seq", "ok": false, "refused": reason}` |
`{"seq", "superseded": true}`.

```
{
  "engine":      {"id", "run_id", "step", "sha8", "encoding", "radius", "search_kind", "deploy_sims", "device", "threads"},
  "position":    {"moves", "ply", "to_move", "moves_remaining", "legal", "winner", "win_line"},
  "perspective": "to_move",
  "raw":         {"value", "argmax": [q, r], "policy": [[q, r, p], …], "ms",
                  "derivation": "quiescence-off tree, 1 leaf, the run's decode"}
                 | {"absent": "position is terminal (p1 wins)"},
  "search":      {"sims", "root_visits", "kind", "seed", "root_value", "argmax": [q, r],
                  "children": [[q, r, prior, visits, q], …], "quiescence_fires", "ms",
                  "derivation": "DeployHeadPlayer, the run's own head; root_value includes its quiescence override"}
                 | {"absent": "sims=0 (raw only)"} | {"absent": "position is terminal (p1 wins)"},
  "tactics":     {"class": "win|block|lost1|quiet", "k", "cells": [[q, r], …],
                  "opp_fours": [[[q, r], …], …], "forced_win_move": [q, r] | null,
                  "verdict": {"raw": "…", "search": "…" | null},
                  "derivation": "mantis.diagnostics.tactics.analyze(k=…, radius=…)"},
  "symmetry":    {"n", "centre": [q, r], "value_min", "value_max", "spread",
                  "argmax_agreement": "k/n", "worst": {"map", "value", "argmax": [q, r]},
                  "translation": {…} | {"absent": "the first stone would leave the opening window"}}
                 | {"absent": "not requested"},
  "elapsed_ms"
}
```

- **raw** comes from the engine's cached quiescence-off tree (§2.2): `root_value()` is the net's
  value; `argmax = max(children, key=prior)`; `policy` lists every legal child with its prior.
  `root_raw_value()` is never read.
- **search** is the deploy head's own answer: `root_value` INCLUDES its quiescence override
  (§0 item 1) and `quiescence_fires` (`tree.quiescence_fire_count`) says how often it fired;
  `argmax` is the move `select_move` returned (the head's choice: most-visited under puct,
  Sequential Halving's answer under gumbel); `children` from `last_root`; `sims` requested,
  `root_visits` reported; `seed` the constant.
- **perspective**: every value is for the side to move at this ply — `root_value()` is the
  mover's and children Q are flipped into the mover's view (`q_sign`). The page states it ONCE,
  beside the readout. A test pins it (§3): a WIN1 position for the mover reads ≈ +1 at the
  searched root — via root quiescence (an own open window at k → +1), which is exactly the
  deploy head's behaviour; `raw.value` on that position is whatever the net says, in [−1, 1].
- **terminal**: `winner` is set while building the board; `raw` and `search` are both absent by
  name; the tactics line names the winner; no engine is called.
- **tactics**: `mantis.diagnostics.tactics.analyze(q, r, p, mover, k, radius=spec.
  legal_move_radius)` on `Board.get_stones()` with `k = moves_remaining`; `RowTactics.forced(k)`
  gives the class and the cell set. The verdict compares an argmax to the class: `win` →
  "argmax WINS" | "argmax MISSES WIN at …"; `block` → "argmax BLOCKS" | "argmax MISSES THE BLOCK
  — block set …"; `lost1` → "no block exists (LOST1)"; `quiet` → "quiet". One verdict for the
  raw argmax, one for the searched (null at `sims=0`). `Board.forced_win_move(2)` rides beside
  it as the engine's own detector. Tactics are ALWAYS computed; `tac=1` only draws them.
- **symmetry** (`instruments.py`): the 12 symmetries of the hex grid CENTRED ON THE FIRST STONE
  `c` — `m(x) = M(x − c) + c` with `M` = rotation by 60° `(q, r) → (−r, q + r)` composed 0–5
  times, each with and without the reflection `(q, r) → (r, q)` — so ply 0 stays where the
  opening window admits it (§0 item 8). Plus ONE translation by `(+1, −1)`, applied only when
  the translated first stone is still in the opening window; otherwise that row is absent by
  name. Each map is applied to the whole move list (cadence and ownership are preserved), a raw
  read is taken, and the argmax is mapped back by the inverse. The panel reports `value_min /
  value_max / spread`, how many of the n argmaxes agree with the identity's, and names the worst
  map. 12–13 raw reads ≈ 0.4 s on CPU; on demand, cost stated. A net that is exactly
  equivariant reads spread 0; the panel draws the number, the reader draws the conclusion.
- **value trace** (`instruments.py`, `POST /trace {engine, moves, sims, client, seq}`): ONE
  server-side request over the cached raw tree — one raw read per ply of the move list (≈ 31 ms
  each) — returning `[{ply, to_move, raw, root | null}]`; the searched root per ply only when
  `sims ≥ 1`, opt-in, with the cost PRINTED on the button before it runs (`plies × per-search s`,
  from the last search's measured ms). The P1-perspective flip `v_p1 = v if to_move == "p1" else
  −v` is done in Python; the chart caption says "P1's perspective (the readout is the mover's)".
  Plies not computed are gaps, never interpolated. `viewBox="0 0 <plies> 100"`, `vector-effect:
  non-scaling-stroke`, the current ply a dashed line.

### 2.5 Routes (`serve.py`)

| route | in | out |
|---|---|---|
| `GET /` | — | the page (`html.py`) |
| `GET /engines` | — | `[EngineInfo…]` incl. stated gaps and strix's availability |
| `POST /analyze` | `{engine, moves, sims, symmetry: bool, client, seq}` | the envelope (§2.4) |
| `POST /trace` | `{engine, moves, sims, client, seq}` | `{seq, ok, trace: [...]}` or the refusal envelope |

Status codes: `200` any envelope with `ok` or `superseded`; `400` a position refusal; `404` an
unknown engine id; `500` an analyst exception (named); `503` an engine that failed to load
(named — a bad stamp, CUDA OOM, strix unavailable); `504` the analyst timeout.
`ThreadingHTTPServer`, bind `127.0.0.1` by default; `--bind 0.0.0.0` is documented in `--help`
as unsafe (no auth, no TLS) and never the default. CLI defaults are the house tools convention
(`tools/viewer/cli.py`, `tools/bench_server.py`); R1 governs configs. `once` runs one `/analyze`
body from the command line and prints the record — the same function, for scripts and for tests
without a socket.

### 2.6 The page (`html.py`, `web/`)

```
┌ mantis analyzer · run8 · 18 000 · 98fe0e8c · puct · cpu ─────── keys: ← → u h p q t s v l ? ┐
│                                   │ engine [run8_00018000 ▾] [+ slot]   budget (32)(256•)(1024) [n]│
│        ┌ hex board ┐              │ ply 7 · p2 to move · 2 left · 280 legal                         │
│        │ stones w/ ply numbers    │ 0,0;1,0;0,1;-1,0;0,-1;2,0;1,1                  [copy]           │
│        │ overlay: visits|prior|Q  │ ── run8 @18 000 ── values for the side to move ─────────────── │
│        │ argmax ringed            │  net     −0.019   argmax (−2,1)      31 ms                       │
│        │ tactics outlines (t)     │  head    −0.049   argmax (−2,1)      256 · 5.6 s · q-fires 0     │
│        │ legal window faint       │  CHECK · block {(2,1),(3,0)} · head argmax (2,1) BLOCKS          │
│        └──────────┘               │  cell     prior   visits    Q                                    │
│  hover: (q,r) · p 26 % · n 169 · Q −0.03   (−2,1)   26 %   169   −0.03  ●                          │
│                                   │  (1,−2)   24 %   159   −0.02                                     │
│                                   │  … 8 more under the floor                                        │
│                                   │ symmetry (s): not requested — 13 net reads, ≈ 0.4 s on cpu       │
│                                   │ value trace (v): not requested — 7 raw reads ≈ 0.2 s             │
└───────────────────────────────────┴─────────────────────────────────────────────────────────────────┘
```

- **Grid.** Desk: one CSS grid `board | side` (side ≈ 380 px). Phone (portrait, decided by aspect
  ratio): one column `board / controls / readout / cards`, the board sized to keep the readout on
  screen. Pure CSS, no script.
- **Tokens.** The viewer's `:root` custom properties VERBATIM (`--bg --fg --mute --line --p1 --p2
  --hi --win --heat`) plus `--pos` / `--neg` for the Q overlay; `prefers-color-scheme: dark` as
  the viewer. `system-ui`, 12–14 px, `font-variant-numeric: tabular-nums` on every number. No
  shadows, gradients, rounded cards, icons or emoji; borders are `1px var(--line)`.
- **Board.** The viewer's SVG, forked (§0): empty cells around the played area, the LEGAL WINDOW
  of the ACTIVE card's engine drawn faint (its radius is on the card), stones with ply numbers,
  the last two stones ringed, the win line outlined. Tap-to-cell by the inverse map +
  `cube_round` (Red Blob, decided); a Python twin of the inverse is round-trip tested and the JS
  is its transliteration. Hover/focus on a cell writes `(q, r) · p · n · Q` under the board — no
  floating tooltips.
- **Overlays** (one at a time; `h` visits, `p` prior, `q` Q): visits and prior share the
  sequential `--heat` at the OBSERVATORY's rule `alpha = 0.25 + 0.55 × x / max` (the viewer draws
  `0.12 + 0.78 x`; the two boards differ by decision, stated here), the number printed above a
  floor (visits ≥ 2, prior ≥ 2 %; the floor stated on the panel and never blanking the board);
  Q is diverging (`--neg` … transparent at 0 … `--pos`) over children with visits ≥ 2. The ring
  marks the HEAD's argmax when a search exists on the active card, else the NET's; the other
  cards' argmaxes are small labelled markers. `t` toggles the tactics outlines: the class's
  cells in `--win`, the opponent's fours' empties dotted in `--hi`.
- **Engine cards.** One card per slot (phase 2: one slot): header `run8 @18 000 · puct · cpu ·
  4 threads`, the two-row readout (`net` / `head`: value, argmax, ms; the head row adds sims and
  `q-fires`), the tactics line with its derivation, the ranked children (by visits, or by prior
  at `sims=0`; 12 rows + "n more under the floor"; a row click highlights its cell), then the
  on-demand panels. The first card is active; clicking a card header makes it the overlay's
  source.
- **Auto-analyze** on by default: every edit sends `sims=0` at once and, debounced 300 ms, the
  budget's search; the head row shows `searching · 256 · 3.1 s` with a live elapsed counter, or
  `queued`; a superseded reply is simply dropped. Off: a "search" button. The deep chip
  (1024 sims, ≈ 23 s on CPU) says its cost in its title.
- **Keys** — the observatory §2.6 map where a key exists there, new keys otherwise; single keys,
  no modifiers, printed in the header and on `?`: ← → Home End step; `u` undo; `h` visits, `p`
  prior, `q` Q; `t` tactics; `s` symmetry sweep; `v` value trace; `l` copy the deep link; `c` copy
  the position text; `?` the key sheet. Keys are ignored while an input has focus.

### 2.7 Empty, refused, absent, stale — the vocabulary

| state | what the page shows |
|---|---|
| empty board | the legal window (25 cells), the raw read runs (an empty board is a position); strix's card: "strix needs a p1 stone" |
| engine loading (first use) | `loading run8 @18 000 …` on the card; strix's subprocess `load` can take seconds and says so |
| engine failed to load | the card carries the `503` reason (a bad stamp, CUDA OOM, strix unavailable + `make vendor.strix`) |
| unknown engine id | `404` — the card says the id is not in `/engines` |
| searching / queued | the state and the elapsed time in the head row; the net row stays live |
| superseded | dropped; the newest request's reply replaces it |
| refused | the reason verbatim from the server, in `--hi`, on the card; the board is unchanged |
| analyst exception | `500` naming the exception type on the card; the server stays up |
| absent panel | `symmetry: not requested — 13 net reads, ≈ 0.4 s on cpu` — a cost, never a blank |
| stale reply | discarded by `seq`; nothing drawn |
| terminal position | `net: absent — the game is over (p1 wins)`, `head: absent — …`; the tactics line names the winner |

Absent is not zero: no panel ever draws an empty axis, a flat line or a bare 0 for something that
was not computed.

### 2.8 Deliberately NOT in v1

Variations and branches (one line, tail replaced); free placement (`ring_audit.reconstruct` is
the seam when wanted); reading game shards in the server (the viewer's `c` is the join);
streaming search progress and cancellation (an `on_batch` hook on `DeployHeadPlayer` is the
seam — a None-guarded addition to a deploy-matched head under `src/mantis`, so it is its own
small ruling and, by §0 item 11, not a worktree change); sealbot; the observatory's run pages;
any `src/mantis` change; a `freeze` (the deep link IS the record of an analysis, given the
checkpoint); JS unit tests (house convention: the viewer's script has none; every derivation is
Python-side and tested there).

## 3. Tests — the load-bearing ones

Style throughout (§0 item 10): one-line docstrings with `Raises` inline, comment runs ≤ 2 lines,
`encoding="utf-8"` on every text read (gate 16), no host content (gate 17), no R8 marker in a
file under the cap (gate 15).

- `test_analyzer_position.py`: text ↔ list round trip; the three refusals by name (occupied,
  outside the radius, after the game ended — the last one is the case `apply_move` alone
  accepts); the owner cadence agrees with `Board.current_player` over 40 random legal plies; the
  legal window equals `Board.legal_moves()`; the `1/−1 ↔ "p1"/"p2"` table; the Python inverse
  map round-trips `hex → pixel → hex` over a ring of cells.
- `test_analyzer_engines.py`: discovery lists stamped names and reports snapshots as a stated
  gap; a stamp missing a needed key is refused by the key's name; the composition on a stamped
  checkpoint MINTED in `tmp_path` by `save_checkpoint` with a tiny `GnnArchV2` (as
  `tests/train/test_arch_stamp_authority.py` does — no fixture file) runs `sims=0` and `sims=8`
  and the record has every field of §2.4; `raw.value` equals `engine.infer_ls`'s value on the
  same board; a second identical gumbel request repeats its `search.argmax`.
- `test_analyzer_analysis.py`: the PERSPECTIVE pin (a WIN1 position for the mover → searched
  root ≥ 0.9 at sims ≥ 8, the mechanism named in the test: root quiescence); `raw` and `search`
  absent by name on a terminal position and no engine call made (a stub engine that raises if
  called); the tactics verdicts on constructed positions — WIN / MISSED WIN / BLOCK / MISSED
  BLOCK / LOST1 / QUIET — with a stub argmax; `raw.argmax` is the max-prior cell, not the
  returned move.
- `test_analyzer_instruments.py`: the 12 maps form a group about the centre (closure, inverses,
  identity at index 0); the translation's inverse and its opening-window rule; a stub engine
  whose value depends only on the stone count gives spread 0 and n/n agreement; the worst map is
  named; the trace's P1 flip and its gaps.
- `test_analyzer_serve.py`: with a STUB analyst (no torch): `GET /engines`; `POST /analyze` happy
  path with `seq` echoed; `400` on a refusal; `404` on an unknown engine; `500` naming a raised
  `BaseException` subclass with the loop alive for the next request; `504` on a timeout;
  supersession (two queued requests from one client → the older answers `superseded`); the
  default bind is loopback.
- `test_analyzer_strix.py`: the driver's `analyze` op — skipped LOUDLY naming `make vendor.strix`
  when the vendor is not fetched (the house convention), otherwise the shape, `improved` sums to
  1 over `legal`, `per_child_q` and `per_child_prior` aligned with `legal`, and the empty-board
  refusal; the driver launched is THIS tree's (`§2.2`).
- `test_analyzer_html.py`: the page renders with the viewer's tokens present and the CSS/JS
  inlined; `test_game_viewer.py` gains one assertion for the viewer's `c` key.
- `test_meta_ci.py`'s target-set pin gains `analyzer` in the same commit as the Makefile target.

## 4. PLAN — phases, each a reviewable unit with tests

1. **Position + engines + the record, headless.** `position.py`, `engines.py` (the raw tree,
   the search), `analysis.py`, `instruments.py`, `cli.py once`. The perspective pin and the
   quiescence-off raw read land here. Gate set at exit.
2. **The server + the page, one engine slot.** `serve.py` (the superseding analyst), `html.py`,
   `web/`, the shim, the Makefile target + the `test_meta_ci` pin, the viewer's `c`. The readout,
   the three overlays, the tactics line, deep links, auto-analyze. The R9 amendment (§5) in THIS
   commit. A usable tool at this exit. Gate set at exit.
3. **2b — instruments and slots.** The symmetry panel, the value trace (`/trace`), engine slots
   with markers. Gate set at exit.
4. **Strix.** The driver's `analyze` op, `strix.py`, the strix card, the raw-value perspective
   probe that turns "UNVERIFIED" on the card into a stated fact. Its own commit, so the
   mantis-only tool is usable before the vendor is built anywhere.

Mechanics: the worktree runs on the main checkout's venv with `UV_NO_SYNC=1`; nothing under
`src/mantis` is edited (§0 item 11); the strix driver path is resolved from this tree. The full
local gate set (`make gates.exit`) at each phase exit; targeted tests while iterating. Commits
are one line, no trailers (R36/R47). No `.rs` is touched. Governance text (CARDS/STATE) written
before phase 2 lands may name `tools/analyzer/` but not the shim (gate 10 scans docs/governance).

### 4.1 Risks named

- CPU latency at the deploy budget (5.6 s) makes auto-analyze feel slow on the operator's host;
  the mitigations are supersession, the 300 ms debounce, the quick chip, the instant net tier
  and the elapsed counter — and the box. If that is not enough, the `on_batch` hook (§2.8) is
  the next step, ruled separately.
- The comment ratchet (§0 item 10) is the largest hidden cost: ≈ 1 400 lines written to
  one-line docstrings from the start, never retrofitted.
- `web/analyzer.js` (≈ 250 lines after moving the trace server-side) is the most defect-prone
  code and has no unit test; the Python twin of the inverse map and the small surface are the
  mitigation.
- Strix's subprocess holds its own torch; the card states the analyzer's memory. The
  perspective of strix's raw value is unverified until phase 4's probe; the card says so.
- `LocalInferenceEngine` was built for many workers; at one request in flight its collector
  waits its own deadline. Measured acceptable (31 ms raw through the cached tree); if the graph
  path's batching deadline dominates, `inference_batching` is read from the stamp and stated,
  not tuned here.

## 5. The `repo_design` amendment (phase 2's commit carries it; proposed text)

> ### AMENDMENT — R363, ANALYZER-1: an interactive position analyzer is ADMITTED on the
> viewer's terms plus one loopback socket
>
> **R363.** §1 keeps display surfaces absent because each would WATCH A RUN: the R333(d)
> criterion is coupling, not sockets, and every tool admitted so far happened to need neither.
> ANALYZER-1 is a position analyzer: a position in (typed, pasted, or a deep link), one or more
> engines' readings out. It needs a process, so it is the first admitted tool that opens a
> socket; under R9 it lands as an amendment in the same commit as the server.
>
> 1. **What is admitted, narrowly.** `tools/position_analyzer.py` (`make analyzer CHECKPOINTS=…
>    [STRIX=1]`; the implementation is `tools/analyzer/`) — a stdlib `ThreadingHTTPServer` bound
>    to loopback by default (`--bind 0.0.0.0` is documented as unsafe, never the default),
>    serving ONE page and answering `POST /analyze` by running the run's own `DeployHeadPlayer`
>    on stamped checkpoints (immutable, R3) and, opted in, the vendored strix pin through its
>    driver. It watches no run, reads no run record, holds no file open, polls nothing, adds no
>    producer and writes no artifact. It is dev-only tooling under `tools/`, beside the
>    dashboard and the viewer.
> 2. **What stays absent, unchanged.** The web dashboard and the TUI monitor; the coupling rule
>    of the R333(d) amendment is untouched. DASH-2 (a server over the run record carrying the
>    viewer) is neither discharged nor engaged by this.
> 3. **The rule the tool carries.** Every number on the page is derived in Python from a named
>    engine call and carries its derivation; the NET's value is read through a quiescence-off
>    tree and the HEAD's value is the deploy head's own, its quiescence override included and
>    counted; values are for the side to move, stated once; a search not yet run is a STATE with
>    its elapsed time, never a stale number; a panel not requested is a stated cost, never a
>    blank; a position the engine refuses is refused by name; strix's gaps are stated on its
>    card. Absent is not zero, applied to a live engine.
> 4. **Measured at landing:** (filled at landing — engine load, the raw read, and the four
>    search budgets on the host that ran the gate set, in the same table as §0 of the design.)
> 5. **Contracts #11 and §4.7 are unchanged.** Nothing is emitted; nothing is read from a record.

The ruling number is the operator's; this document only drafts the text.
