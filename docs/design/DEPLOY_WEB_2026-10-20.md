# DEPLOY-WEB — the browser bot: one wasm engine, two net backends, witnessed from the stamp (2026-10-20)

Status: **DESIGN** (DEPLOY-WEB L0, under R398(b)–(c); CARD-DEPLOY-WEB). The design is R398's. This document records it
and fills in the mechanisms. It does not reopen a decision. Where it adds a choice R398 did not make, it says so in
place, and every choice with a law or a ruling behind it is put to the architect in §13. No code, config or mint rides
it. It orders nothing until the architect's word before L1.

- **Evidence:** RESEARCH-WEB, `mantis-records/research-web/REPORT.md` (accepted by R398(a)), with its `export/`,
  `bench/gforward/` and `bench/gbench/`. Every number below without a source is that report's, or is marked INFERENCE.
- **A caveat on every browser number.** Each one was taken on a cross-origin-isolated page. The shipped page is not
  isolated. Every WebGPU number is the ScatterND graph's: the ScatterElements graph this design ships never ran on
  WebGPU (§3.3).
- **Tree:** read at `fd26b1d5` (origin/dev). This doc sits on R398's landing commit, which is docs-only over it.
- **Records:** `mantis-records/deploy-web/` (local): `review/REVIEW-1.md`, `review/REVIEW-2.md` and their
  dispositions.
- **Ruled by R399 (2026-10-21).** §13's eight items are ruled, and each carries its ruling. The body follows the
  rulings where they change a mechanism: the 16-sim floor (§2.5, §6.4), the fp16 line (§4.1, §5), the WebGPU fallback
  (§3.3), the trap rebuild (§2.1, §6.2) and the retired reservation (§8). L1 may start.

## 0. Decisions — one screen

1. **One engine.** `crates/mantis-player` is the deploy head's per-stone loop ported to Rust, single-threaded. It
   carries the early stop, TR-D's reroot, the tactics resolve and the per-game exact cache. It sits behind a pull/push
   step machine: `next_batch(now) → Batch | Turn` and `submit(outputs, now)`. It builds natively and for wasm32
   through wasm-bindgen. It calls the same Rust the served path calls (§2.1), so the port is the loop and the cache,
   not the search.
2. **PUCT only** (this design's choice). The deploy block of record is `puct`; reuse and the early stop are PUCT-only.
   The player refuses `gumbel` by name, and the detector refuses it first.
3. **One batch for both backends** (§3.1). From the served path's own `pack_wire`, the engine builds every array the
   net reads beyond `x` as int32 (R398(b)): indices, masks and counts. The ONNX graph then holds only float math and
   int32 gathers and scatters. `RustForward` reads the same arrays.
4. **Two backends behind one trait.**
   - `RustForward` is the research's fused GINE forward, in the crate. It reads the weights file and is the
     one-CPU-thread path, with no runtime.
   - The host backend is ORT-web's native WebGPU EP over the ScatterElements graph, one session per page.
   - No ScatterND anywhere, no threads anywhere, so no COEP.
   - **Both backends are driven by one JS loop in the Worker** (§6.3). It times every phase on the host's clock and
     yields between batches so an abort is heard.
5. **One export path.** `tools/export_web.py` writes the ONNX in fp32 and fp16, the weights file and a manifest, each
   named after the stamp.
   - The manifest binds them to the stamp, the exporter commit, the resolved deploy block, the player's own constants
     and every witness row.
   - The detector is a closed table that refuses everything outside it by name (§4.2).
6. **Three witnesses, pinned, each with a planted break shown red** (§5):
   - **W1:** golden equality at every shipped backend and thread count.
   - **W2:** the Rust forward against the ONNX on the same goldens.
   - **W3:** search parity. One Python stub feeds the player and the Python deploy head, which must play the same
     stones and count the same rows, at concurrency 1 (R394(f)).
7. **The budget is set on the device, each turn** (§2.5):
   - sims = clamp(⌊(0.8·T − root) / (k·ρ·c_leaf)⌋, 32, cap), where k is the turn's stones. R398(b) writes 2; k = 1 for
     a one-stone turn is this design's.
   - root and c_leaf come from the previous searched turn, on the host's clock. Turn 1 uses the start-up warm-up.
   - **ρ = 1**, its hard bound. A measured ρ buys 27–36 % more sims and risks an overrun; that is §13 A6.
   - The engine reports `minTurnMs`, the shortest clock at which the 32 floor fits. Below it a turn plays at the floor
     (16 sims on the CPU path, R399(d)), late, counted and shown.
   - L4 holds the clock to a pre-stated line: no clock loss, and p99 turn time ≤ 0.8 T (§11).
8. **Memory by budget** (§2.6).
   - `MAX_NODES` becomes a constructor parameter: `4·1024·S_cap` nodes, with S_cap = clamp(2 × the rule's quotient at
     k = 2, 32, 976). The pool is sized at `newGame` from that game's clock.
   - The pool alone, at 10 s: about 20 MB on one CPU thread and about 132 MB on WebGPU. Whole-engine peaks are about
     90 and 190 MB (§2.6). All are INFERENCE, and the WebGPU figures are nd-graph ones.
   - The per-game cache keeps Python's 8 192 entries, FIFO, at about 12 MB in Rust.
9. **The integrator surface is `web/`** (§6):
   - a Worker that owns the engine and the backend;
   - a typed API (`newGame`, `play`, `playTurn`, `analyse`, `abort`);
   - the arena's analysis-slot packet adapter, a demo page with the LAW-18 rows, and node tests.

   It is released as an npm package from a tag, with the net files beside the code inside it: "every byte ships in the
   package", as the packet reads R398(c)'s "beside it". The integrator serves those files from its own origin, so
   nothing is fetched from a third party at runtime. Contract #13 is `docs/contracts/web_engine.md`.
10. **Bad news first:**
    - The shipped WebGPU graph is unmeasured on WebGPU. Its expanded scatter index is [E, H], about 116 MB of int32 at
      B 8 (§3.3). L3 measures it against pre-stated numbers, and §13 A7 is its fallback.
    - fp16 cannot be held to the fp32 lines. It ships to WebGPU only under R399(a)'s line, read in L1 (§5).
    - The 32 floor exceeds a 5-s clock on a slow CPU (§2.5).
    - The WebGPU path is about 6–8 MB on a first visit, above R398(d)'s ~4–6 MB.
    - The player is a second consumer of the switches the protected set says only the ladder throws (§13 A3).
    - The port touches `mantis-search` additively and copies three pieces of bridge glue for now (§13 A2).

## 1. What exists, and what is new

| piece | today | DEPLOY-WEB |
|---|---|---|
| search, solver, reroot | Rust (`mantis-search`), driven from Python through the bridge's `MCTSTree` | called directly by the player; one additive constructor (§2.6) |
| per-stone loop, early stop, reuse switch, tactics resolve | Python (`DeployHeadPlayer.select_move`, `_search`, `_drive_puct`, `_leader_fixed`) | ported to `mantis-player` (§2.2); the Python head is not edited |
| per-game exact cache | Python (`GameEvalCache`), keyed by Rust's `LeafKey::of` through the bridge | ported (§2.3), keyed by `LeafKey::of` directly |
| leaf graph, wire, pack | Rust (`build_leaf_graph`, `GraphWire::from_axis_graphs`, `pack_wire` in `mantis-selfplay`), then torch | the same Rust, then the batch derivation (§3.1) |
| net | torch `served_outputs`: `forward_batch`, `segment_softmax`, `decode_binned_value` | ONNX (ORT-web WebGPU) or `RustForward` |
| export | none in the tree; RESEARCH-WEB's `export_served.py` in records | `tools/export_web.py` |
| browser surface | none | `web/` |

## 2. The crate `mantis-player`

### 2.1 Its place in the DAG

- **Dependencies:** `mantis-player → core, graph, encoding, search, selfplay`. It takes from `mantis-selfplay` exactly
  the served path's own leaf code:
  - `build_leaf_graph`;
  - `LeafKey::of`, the cache key;
  - `GraphWire::from_axis_graphs`;
  - `pack_wire`, at `threads = 1`, where the one share runs on the calling thread (L2's wasm test pins that nothing
    spawns);
  - `assemble_ls_from_gnn_probs`.

  No second leaf build, key or pack exists. Of the runner it takes only `runner::params::resolve_geometry`, the one
  geometry resolution; it never reaches the queues' threaded pop or `EvalCache`.
- **wasm32.** The wasm-only dependencies live under `[target.'cfg(target_arch = "wasm32")'.dependencies]` in this
  crate's manifest only:
  - `wasm-bindgen`, pinned exactly;
  - `getrandom = { version = "0.4", features = ["wasm_js"] }`.

  Cargo should unify the feature into `rand`'s getrandom, leaving `mantis-search` and `mantis-selfplay` unedited. That
  is INFERENCE until L2 builds it; the research put the line in `mantis-search`. The player's paths draw nothing
  random, since PUCT draws nothing. The build sets `+simd128` (§9).
- **Panics.** The workspace's `panic = "unwind"` builds for wasm32 on the pinned 1.97.1 (checked on a scratch crate,
  2026-10-20). A panic in wasm is a trap: the Worker sees an error, the instance is dead, and the API rejects every
  pending call with `EngineCrashed`. The Worker then rebuilds the engine and counts the trap (`engine_traps`); R399(h)
  accepts this as LAW-13's reading. Production paths return the named
  `PlayerError` (§2.7), so a trap is a bug, never a control path.
- **One FFI place each:**
  - `#[wasm_bindgen]` lives only in `crates/mantis-player/src/wasm.rs`, which is cfg'd to wasm32 and checked by a
    census test.
  - The Python bindings live in `mantis-bridge`, like every pyclass: `web_batch`, `PyPlayer` and `PyRustForward`.
    They exist for W2 and W3 and for the export's native forward row (§5).
  - LAW-17 stands: no pyo3 outside the bridge.

### 2.2 The per-stone loop (the port)

The player plays a turn as one stone, or two, with the board updated between them. Each stone follows
`DeployHeadPlayer.select_move` step for step, in its order:

1. **Empty board:** play `(0, 0)`, unsearched, with `last_sims = 0`.
2. **Reroot or reset:** `tree.reroot_or_new_game(board)` when reuse is on, which it is in the shipped posture; else
   `new_game`. The reroot itself decides whether a subtree is kept: only at the turn's second stone, under PUCT,
   with an expanded root.
3. **Root offence:** `root_offence()` when tactics are armed. A decided stone is played unsearched, with the tactics
   counters taken and the reuse rows not counted. `Err(ProofStoneIllegal)` reads as `None`, as the bridge reads it.
4. **Reuse rows:** if any visits were inherited, `reuse_reroots += 1` and `reuse_inherited += inherited`.
5. **Root step:** `select_leaves(1)` and, if it returns a leaf, one expand. Then
   `sims_done = leaves + last_inline_descents() + last_tt_hits()`.
6. **PUCT drive:** while `sims_done < n`, with `left = n − sims_done`:
   - The early stop: if `get_top_visits(2)` has one child, or leader − runner-up > `left`, then
     `stop_fired += 1`, `stop_saved += left`, and the loop ends.
   - Otherwise `select_leaves_filled(leaf_batch_size, left)`, with `unserved = last_inline_descents() + last_tt_hits()`.
   - It ends on no leaves and no unserved. Otherwise it expands the leaves and adds `leaves + unserved` to
     `sims_done`.
7. **Move:** `get_top_visits(1)[0]`. The player calls the same function, so ties among equal visits break as the
   Python head breaks them; nothing reimplements the argmax. `last_sims = sims_done`.
8. **Rows, read before the audit as Python reads them:**
   - `select_counters()` into `select_calls`, `select_overlaps` and `select_network_leaves`;
   - the root value and `get_root_children_info()`.

   No root children is `PlayerError::NoRootChildren`, as the head raises.
9. **Audit:** `root_audit(move, audit_order(None))` when tactics are armed; then `tactics_counters()`.

**"Expand"** is the cache step (§2.3), then `assemble_ls_from_gnn_probs` per served leaf, then one
`expand_and_backup_ls_at` over the batch in pending order. The probabilities and values are exactly the f32 the
backend returned. Three pieces of this loop are bridge glue today, not `mantis-search` code:
- the `ProofStoneIllegal → None` mapping;
- the `audit_order` call;
- `expand_and_backup_ls_graph`'s arity, stride-362, centre and trunk checks.

The player carries its own copy of each, named as interim duplication. An error-parity test feeds both the same
malformed inputs, and the copies are retired by CARD-DEPLOY-HEAD-DRIVER when the bridge delegates (§13 A2).

> **L2, as built:** the expand checks are `player::expand_refusal`, and the error-parity test
> (`mantis-bridge`'s `player::tests`) feeds it and the bridge's `expand_and_backup_ls_graph` the same five inputs.
> Three more interim copies, each named in CARD-DEPLOY-HEAD-DRIVER and held to the head by W3: the legal rows'
> coordinates (`batch.rs` `leaf_meta`, as the bridge's `inference.rs` derives them), the board-to-request conversion
> (`position_of`) and the spec-to-`BoardGeometry` one (`Player::new`). The tactics block's ranges are
> `TacticsConfig::check` in `mantis-search`, which the player calls; the bridge's parser keeps its own until it delegates.

**The deploy block** has two sources, and the manifest keeps them apart (§4.3).
- **Resolved from the stamp's own `config`**, by the run's resolvers (`resolve_puct_constants`,
  `resolve_deploy_tactics`), at export. Values of record (`configs/run14.yaml`, resolved):

  | key | value | source |
  |---|---|---|
  | kind | PUCT | `deploy.search.kind` |
  | `c_puct` / `fpu_reduction` | 1.5 / 0.25 | `selfplay.mcts` |
  | quiescence / `blend_2` | on / 0.3 | `selfplay.mcts` |
  | `leaf_batch_size` | 8 | `selfplay.leaf_batch_size` |
  | tactics: leaf | 3 turns / 256 nodes | `deploy.search.tactics` |
  | tactics: root | 8 turns / 20 000 nodes | `deploy.search.tactics` |
  | tactics: audit | 8 turns / 2 000 nodes, k 4, m 4, total 40 000 | `deploy.search.tactics` |
  | audit mode | `hold` | added by the resolver, not in the yaml |

- **The player's own constants**, each pinned by a test equal to the source it mirrors:

  | constant | value | the source it equals |
  |---|---|---|
  | early stop | on | `LADDER_EARLY_STOP` (`tools/ladder/backends.py`) |
  | tree reuse | on | `LADDER_TREE_REUSE` |
  | cache capacity | 8 192 | `MAX_ENTRIES` (`src/mantis/arena/eval_cache.py`) |
  | virtual loss | 1.0 | `VIRTUAL_LOSS_PENALTY`, used directly from `mantis-search` |

### 2.3 The per-game exact cache (the port)

It is `GameEvalCache`, line for line:

- **Key:** `LeafKey::of(stones, side, stones_left, win_length, radius, trunk, empty_edges)`. That returns
  `Result<LeafKey, String>`; the error is `PlayerError::Leaf`, and the `LeafKey` is the map key. The geometry comes
  from the encoding row.
- **One batch:**
  - Hits are counted.
  - Unseen keys are deduplicated in first-seen order, and the backend sees each only once.
  - Outputs are assembled in leaf order, and only then are the new entries inserted.
- **Capacity:** 8 192 entries. At capacity the oldest goes before an insert (FIFO by insertion), and a hit does not
  refresh an entry.
- **Entry:** the `LegalSetPolicy` (the dense 362 row plus the off-window map), the value, and the builder's window
  centre. That is about 1.5 KB in Rust against Python's ~5 KB (INFERENCE from the row's size).
- **Lifetime:** `new_game` empties the entries; the counters stay cumulative.
- **Rows:** `calls`, `positions`, `hits`, `served`, `in_batch_repeats`, `all_hit_calls`, `evicted`, with
  `positions = hits + served + in_batch_repeats`.

The batch the backend sees is therefore the misses alone: up to `leaf_batch_size`, and fewer after hits.

### 2.4 The pull/push step machine

The engine never calls the net. It hands a batch out and waits for the outputs, so one engine serves an in-module
backend and an async host backend alike.

```
new_game(position, pool) ─► Ready ─ begin_turn(budget, now) ─► Root(stone)
Root(stone): reroot/new_game, root_offence, root step ─► Pull | StoneDone
Search(stone): early stop, select_leaves_filled, cache ─► Pull | StoneDone
Pull ── next_batch(now) returns Batch(&WebBatch) ── submit(outputs, now) ─► Search(stone)
StoneDone: rows, audit; the stone is placed; game won or k−1 = 0 ─► Ready, next_batch returns Turn(result)
           else ─► Root(next)
play(stones) (the other side's turn, rules-checked) ─► Ready
abort() in any turn state ─► Ready, the board as before the turn
```

- **`next_batch(now_ms)`** runs the engine forward until it needs the net or the turn is done. It returns
  `Batch(&WebBatch)` (borrowed arrays, §3.1) or `Turn(TurnResult)`.
- **`submit(outputs, now_ms)`** takes `probs [L]` and `value [G]`, and may take `logits [L]` (read only by the
  witness). Wrong lengths are `PlayerError::OutputShape`.
- **`begin_turn`** takes a `Budget`:
  - `Clock { turn_ms }`: the budget rule, §2.5;
  - `Sims(n)`: a fixed `n`, used by W3 and the tests;
  - `Unlimited`: the pool's cap.
- **Errors in a turn.** A `PlayerError` in the middle of a turn rolls the turn back, leaving the board as before it,
  and the engine stays usable. Calling a method from the wrong state (`next_batch` with a batch outstanding,
  `begin_turn` mid-turn, `play` mid-turn) is `PlayerError::OutOfStep`.
- **One board per engine.** `analyse` is a turn on a fresh game at the given position, and it replaces the game in
  play. An integrator that needs both keeps two engines. No scratch tree exists.
- **Time** comes only from the host. Every call takes `now_ms`: the Worker's `performance.now()`, or a test's
  synthetic clock. The engine never reads a clock, because `Instant::now` panics on wasm32.
- **The warm-up and the on-device witness** are the same machine on the manifest's witness positions (§3.4):
  `begin_witness()` pulls their batches and returns the deltas and the timings.
- **`TurnResult`:**
  - per stone: the cell, `last_sims`, `last_stopped`, decided-unsearched, the root value, and the top children
    (cell, visits, Q, prior);
  - per turn: the sims the budget set, the turn's wall ms, root ms and leaf ms;
    (L2, as built: the wall ms and the served leaves; the root phases fold into the cost a leaf, `wall_ms / served_leaves`);
  - flags: `floor_over_clock` and `budget_pool_capped`;
  - the cache, reuse, select and tactics rows.

### 2.5 The budget rule

`sims_for_turn(turn_ms, root_ms, leaf_ms, rho, stones, cap) -> u32` is a pure function in `budget.rs`:

```
min(max(floor((0.8·turn_ms − root_ms) / (stones · rho · leaf_ms)), 32), cap),   with 32 ≤ cap ≤ 976 by construction
```

| input | turn 1 of the engine | each later turn (games included) |
|---|---|---|
| `root_ms` | the warm-up's root offence plus audit on two witness positions, × stones | 0: the engine sees no time inside a call, so a turn's root phases fold into its `leaf_ms` (L2; ≈ 1 % fewer sims, the safe side) |
| `leaf_ms` | the warm-up's wall time per leaf on the witness's largest positions, at B 8, plus their timed `analyze` and one leaf solve each | the previous turn's whole wall time ÷ its served leaves, root phases included; a turn that served none keeps the last finite value |
| `rho` | 1 | 1 (§13 A6 is the measured alternative) |
| `cap` | `armed_sims_for(max_nodes)` (§2.6), ≥ 32 | same |

**Why ρ = 1 cannot overrun from ρ.** `leaf_ms` is measured per served leaf, but it holds the whole search phase:
net time, graph builds, leaf solves, table hits, bookkeeping and the JS↔wasm hop. A turn at S sims spends
S·k·(wall ÷ descents) ≤ S·k·`leaf_ms`, since descents ≥ served leaves. Two things are left to the 0.8 margin
(1.25×):
- leaf growth between turns;
- turn 1's warm-up. It is timed on the largest witness positions, which are larger than an opening's leaves, at a full
  B 8 (WebGPU's partial batches cost ~12 % more a leaf: 7.82 against 7.0 ms, nd graph).

**Measurements carry across games.** A new game's first turn uses the last game's late-game costs. Those are higher
than an opening's, so the error is on the safe side.

**Pinned tests:**
- RESEARCH-WEB's Q5 guaranteed column at T 10 s, root 93 ms, k 2: `leaf_ms` 8.84 gives 447 and 58.71 gives 67.
- The floor, the cap, k = 1, and monotonicity in T.
- `cap ≥ 32` for every pool the player can build.
- The 16-sim floor: on the CPU path under a `Clock` below `minTurnMs`, and nowhere else (R399(d)).

**The floor can break the clock.** At 32 sims, one CPU thread on the desk needs (2·32·58.71 + 93) / 0.8 = 4.81 s.
A laptop core at 0.7× the desk's needs 6.8 s (INFERENCE; no laptop measured).
- The engine reports `minTurnMs = (2·32·leaf_ms + root_ms) / 0.8` after the warm-up and after every turn.
- A `Clock` below it plays at the floor anyway, with `floor_over_clock` set, counted and shown (R399(d)): 16 sims on the
  CPU path, 32 on WebGPU. A weak turn beats a forfeit. The 32-sim floor stands for the budget rule and for
  `minTurnMs`.
- The arena's mode offers only clocks at or above it (CARD-WEB-ARENA-SIDE).
- The engine does not cut a search short on the clock (§13 A4).

**LAW-18 rows** per turn: the sims set, the turn's wall ms, `turn_over_0.8T`, `floor_over_clock`,
`budget_pool_capped`, root ms and leaf ms.

### 2.6 Memory plan

**The pool.** `MCTSTree::with_pool(c_puct, virtual_loss, fpu, max_nodes)` is added in `mantis-search`:

- `new_full` becomes `with_pool(.., MAX_NODES)`, so self-play is unchanged.
- `armed_sims_for(nodes) = nodes / (4·1024)` is the existing derivation as a function, and
  `MAX_ARMED_SIMS = armed_sims_for(MAX_NODES)` keeps its value.
- `configure_search(Gumbel)` sizes `raw_values` from `pool.len()`. That is the same value in self-play. The player
  never configures Gumbel.

**Sizing.**
- `newGame(position, {turnMs})` sizes the pool from that game's clock:
  `max_nodes = 4·1024·S_cap`, with S_cap = clamp(2 × the rule's uncapped quotient at k = 2 under the engine's current
  costs, 32, 976). A one-stone turn's quotient doubles and is capped at S_cap, counted.
  - The factor 2 is one doubling of headroom. Early-game leaves are smaller than the warm-up's, and WebGPU warms up.
  - An unlimited game takes 976.
- Each game builds a fresh tree sized for its clock, as the Python head does (a tree's stored proof and hold stones
  belong to one game); the old pool is dropped before the new one is allocated. The memory keeps its high-water mark:
  a wasm memory never shrinks (L2).
- A turn's sims never exceed `armed_sims_for(max_nodes)`. When that cap binds, `budget_pool_capped` counts it.
- The pool's overflow panic stays the last line, and is unreachable through the cap. `cap ≥ 32` is pinned.

**What each piece costs** (INFERENCE unless measured; L2 measures each):

| piece | size | note |
|---|---|---|
| tree pool | 36 B × 4 096 × S_cap = 147 KB a sim | one CPU thread, 10 s: S_cap 134 → 19.8 MB; WebGPU, 10 s: 894 → 132 MB (nd-graph timing); 976 → 144 MB |
| reroot's copy | the kept subtree, ≤ S·1 024 × 36 B | ≤ a quarter of the pool, transient, at the second stone |
| transposition table | ≤ one entry per served leaf of the stone, ~1.5 KB each | ≤ 1.5 MB at 976; cleared every stone |
| solver | 2^18 × 32 B table + 4.75 MiB grid | ≈ 12.8 MiB, as in self-play |
| per-game cache | ≤ 8 192 × ~1.5 KB | ≈ 12 MB |
| batch | the mean leaf (555 nodes, 14 122 edges) gives ≈ 1.6 MB for 8; the peak follows the largest leaf (the goldens reach 171 stones), unknown until L2 measures it | ×2–3 transient through the pack |
| `RustForward` | 3.3 MB weights; activations at B 8 (emb [N, 4·256] plus three [N, 256]) | ≈ 35 MB at the mean leaf, more at the max |
| ORT-web WebGPU | runtime wasm 26.8 MB in memory; ONNX 3.4 / 1.7 MB; **GPU**: the [E, H] messages and the expanded scatter index, ≈ 116 MB each at B 8 and the mean leaf | §3.3 caps it per call |
| engine module | core + graph + solver measured 239 KB; the forward 61 KB; MCTS, the encoding row and the pack not yet linked | < 1 MB; L2 measures |

**Peak by path:**
- One CPU thread at 10 s: about 90 MB at the mean leaf, more at the max.
- WebGPU at 10 s: about 190 MB, plus GPU memory.

A wasm memory never shrinks, so the end-of-game `memory.buffer.byteLength` is the peak. L2 and L4 report it as a row.

**First-visit download.** The player fetches only the selected backend's files (§3.4).
- One CPU thread: the weights file, 3.3 MB, plus the engine: about 4 MB.
- WebGPU fp32: the ONNX 3.4 MB, ORT's WebGPU runtime 3.9 MB brotli, and the engine: about 8 MB.
- WebGPU fp16, if A1 lets it ship: about 6 MB.

R398(d) says ~4–6 MB, and the CARD-WEB-ARENA-SIDE card carries this gap.

### 2.7 Errors and determinism

- **`PlayerError`** is a named enum with a variant per cause:
  - `OutOfStep`, `OutputShape`, `IllegalStone`, `NoRootChildren`, `Refused(manifest field)`, `Expand`;
  - `Selection(SelectionDesync)`, `Tactics(TacticsError)`, `Pack(PackError)`, `Leaf(String)`, `Weights`.

  No `unwrap` or `expect` on a production path. wasm-bindgen turns each into a JS `Error` whose `name` is the variant.
- **Determinism.**
  - PUCT draws no random number.
  - Selection breaks ties first-equal-score, children by prior then packed key, the audit by visits, prior and index.
    All of that is the shared Rust.
  - One thread and fixed-order loops: the same position, budget and outputs give the same stones.
  - Different backends' outputs differ within the W1 lines, so they may break a near-tie differently. That is the
    precision of record, not a defect.
  - A `Clock` budget is as deterministic as the host's clock, which is not deterministic. Only `Sims` is.

## 3. The backends

### 3.1 The batch: one format, built in the engine

From `pack_wire`'s seven arrays the engine derives the batch both backends read. All arrays are little-endian; G is
the batch's graphs.

| array | dtype | shape | meaning |
|---|---|---|---|
| `x` | f32 | [N, 11] | node features, as packed |
| `src`, `dst` | i32 | [E] | the global node rows of each edge (`edge_index` split, narrowed by a checked cast) |
| `code` | i32 | [E] | the edge's vocabulary row (u8 widened; an ONNX Gather index is int32 or int64) |
| `legal` | i32 | [L] | the legal cells' global node rows, strictly ascending (`legal_node_gather`) |
| `node_graph`, `legal_graph` | i32 | [N], [L] | the graph of each node and of each legal row (from the offsets) |
| `divisor` | i32 | [N] | 1 on a real node (a stone or legal), else max(in-degree, 1) |
| `real` | i32 | [N] | 1 on a real node, else 0 (the max readout's mask) |
| `mean_mask` | i32 | [N] | 1 on a stone, or on every node of a graph with no stones |
| `mean_den` | i32 | [G] | that graph's mask count; it also carries G to the graph |

- **Dtypes.** Everything the engine builds is int32, as R398(b) has it. The graph casts the three count and mask arrays
  it divides or multiplies by (`divisor`, `mean_mask`, `mean_den`) to f32 once, on the GPU, and compares `real` to 0.
- **What it removes from the graph.** Every int64 data op RESEARCH-WEB saw fall back to the CPU (Range,
  GreaterOrEqual, ReduceSum, GatherND, Sub, Less, Or, Cast, Expand), together with the `in_degree`, `real` and
  `counts` scatter-sums.
- **What L1 found, and fixed (REVIEW-1 of L1).** torch's scatter wants an int64 index, so the dynamo export re-casts
  the int32 inputs to int64 inside the graph, and `t[index]` lowers to GatherND. The exporter reads every gather
  through `index_select`, and a pass after the export (`narrow_index_casts`) drops every Cast of an int32 value to
  int64, found by inferred type (so `index_select`'s own Reshape → Cast → Gather is caught), wherever every read takes
  either width (Gather, GatherElements and ScatterElements's index, Unsqueeze, Expand, Reshape). It refuses a graph
  that still holds a GatherND, or a batch-sized index that a Gather or a scatter reads as int64 (REVIEW-2 of L1 found
  the name-matched first cut missed the Reshape hop). Every such index, the [E, H] expanded one included, is int32.
- **What is left:**
  - Gather, f32 arithmetic and LayerNorm;
  - ScatterElements(add): one per layer (four at 4×256), the mean, and the softmax denominator;
  - ScatterElements(max): the real-node max and the softmax segment max.
- **Outputs:** `logits [L]`, `probs [L]` (softmax per graph over its legal rows) and `value [G]` (the 65-bin
  expectation, clamped to [−1, 1]). That is `served_outputs` plus the logits.
- **Ownership.** The derivation is the one implementation, in `mantis-player`'s `batch.rs`. The exporter reaches it
  through the bridge's `web_batch(positions, encoding)`. The torch re-expression the exporter traces reads these
  arrays, so the ONNX's inputs are the engine's arrays by construction.

### 3.2 `NetBackend` and `RustForward`

> **L2, as built:** no `NetBackend` trait and no Rust `drive` loop: `RustForward` is the one Rust backend, so
> `forward(&WebBatch) -> NetOutputs` is called directly (the Rust tests drive `next_batch`/`submit` with a stub, W2
> reads it through the bridge, the wasm `Engine` through `forward_pending`). It reads the batch's own `divisor`, `real`,
> `mean_mask` and `mean_den`, as the exported graph does. Measured in node on run13's 120k: 63–65 ms a leaf, a
> 697 KB module after wasm-bindgen, and 116 MB (S 32) / 126 MB (S 128) peak memory over 8 goldens (24.8 MB after the
> weights load). The peak is above §2.6's 90 MB for the mean leaf: the goldens reach the large leaves.

```rust
pub trait NetBackend {
    fn forward(&mut self, batch: &WebBatch<'_>, out: &mut NetOutputs) -> Result<(), BackendError>;
}
```

- **In Rust**, `drive(player, backend, Budget::Sims(n))` is the synchronous loop for the Rust tests under a fixed
  budget only. `PyPlayer` exposes `next_batch` and `submit` to Python, and W3 pulls and pushes through them.
- **In the browser**, the Worker's one JS loop drives both backends (§6.3):
  - `RustForward` through `wasm.rs`'s `forward_pending()`, which runs it on the pending batch in place and holds the
    outputs, then `submit_pending(now)`. The batch never crosses into JS and back, so nothing is copied, and no
    view is held over a memory that a malloc could grow;
  - ORT-web through `await session.run`, over typed arrays copied out of the batch.
- **`RustForward`** is `bench/gforward` brought into the crate:
  - fused edge messages, so no [E, H] tensor exists;
  - 4-row GEMM blocks; sequential f32 accumulation in edge order;
  - LayerNorm eps 1e-5;
  - the stone-mean and real-max readout, the 65-bin value expectation, and a stable per-graph softmax.
- **Its weights** are the weights file: raw f32 little-endian, linear weights transposed (in × out). The tensor table
  (name, rows, cols, offset) is in the manifest, not a side file. The order is the exporter's `ServedGraph` order:
  - `input_proj`;
  - per layer: norm, the folded edge table [91 × H], `eps`, the MLP;
  - `final_norm`, the policy MLP, `fc1`, `fc2`, `support`.
- **What RESEARCH-WEB measured:** max |Δ logit| 2.1e-4 against torch, and 58 ms a leaf for 4×256 in Chromium, flat in
  B. L2 brings it in as measured, with no GEMM work. Speed is L4's to read, not L2's to tune.

### 3.3 ORT-web on WebGPU

- **Import:** `onnxruntime-web/webgpu`, the native WebGPU EP. The default import loads the deprecated JSEP, which
  has no ScatterElements. The version is pinned exactly in `web/package-lock.json`; L3 picks the newest release that
  passes W1.
- **Session:** `executionProviders: ['webgpu']`, `graphOptimizationLevel: 'all'`, `env.wasm.numThreads = 1`, no
  proxy, one session per page. No graph capture: the shapes are dynamic, and a WebGPU scatter fix of 2026-10-15 is
  not in 1.31.0.
- **Feeds:** `ort.Tensor`s over the batch's typed arrays, f32 and int32 only.
- **Fetches:** `probs` and `value` in play; `logits` only in the witness.
- **Placement.** "No float or index op falls back" (R398(b)) is read as: no **data** op on the CPU, meaning no op
  whose output is sized by N, E, L or G. Ops that only touch shape tensors may run on the CPU, as ORT keeps them
  under dynamic shapes. L3 records the placement log, and a data op on the CPU is a W1 row failure.
- **GPU memory.** ScatterElements needs its index shaped like its updates. The exporter's `scatter_add` over
  `index[:, None].expand(-1, H)` therefore materialises an [E, H] int32 index beside the [E, H] f32 messages: about
  116 MB each at B 8 and the mean leaf. The max readout adds an [N, layers·H] index.
  - So the backend splits a batch into calls at graph boundaries, under a cap on edges a call that L3 sets from its
    measurement.
  - A graph's outputs do not depend on its batch-mates, so the split is invisible to the step machine. The release's
    WebGPU witness runs once more with a forced small cap, so the split itself is witnessed, and a node test pins
    the splitting against a stub session.
- **The measurement L3 owes, pre-stated before it runs.** The elements graph on WebGPU, on a non-isolated page, on the
  desk:
  - W1 and the placement log;
  - ms a leaf at B 1 and B 8;
  - peak GPU memory at the goldens' p99 leaf.

  It passes if it meets W1, places no data op on the CPU, and is faster a leaf than `RustForward` on the same page.
  Otherwise R399(g) applies.
  - The ScatterND graph ships for WebGPU only, as a second ONNX. Its manifest row is marked "wrong under threaded
    CPU kernels (ORT #18048)", no path but the WebGPU backend loads it, and a node test pins that.
  - The integrator ONNX stays the ScatterElements graph.
  - If the ScatterND graph misses too, `RustForward` alone ships until CARD-WEB-RUNTIME-ALTERNATIVES.

### 3.4 Selection and the on-device witness

At start, the Worker first checks for wasm SIMD (`WebAssembly.validate` on a 1-instruction simd128 module). Without
it, it refuses to start with `Unsupported`. Then it picks a backend once:
1. If `navigator.gpu` exists and gives a non-fallback adapter, it builds the WebGPU backend and runs the on-device
   witness. On a pass, it takes it.
2. Otherwise, or on a miss, it disposes the WebGPU backend, fetches the weights file and runs the same witness on
   `RustForward`. On a pass, it takes it.
3. If neither passes, the engine refuses to start (`NoBackendPassed`, with both rows).

**The witness itself:**
- It uses 16 positions in the export's witness file:
  - the empty board and the 1–4-stone openings;
  - eleven goldens spread over stone count, the largest included.
- The engine builds their batches itself, at the deploy batch size, so the witness also checks the engine's own
  builder in the browser. It compares to the export's reference outputs at the W1 lines.
- Its wall time on the largest positions gives the warm-up's `leaf_ms`, and its two-stone positions give `root_ms`.
- A backend that misses refuses itself and says by how much. The demo shows the row.

## 4. The export path: `tools/export_web.py`

### 4.1 What it writes

`uv run python tools/export_web.py --checkpoint <stamp> --out <dir>`. It starts from RESEARCH-WEB's `export_served.py`
and goes through four steps.

1. **Load.** `load_checkpoint(path, declared_encoding=None, declared_future_head=None)` adopts the stamp. Then
   `build_net(metadata.arch)` and `deploy_state`. The manifest records whether the EMA shadow or the learner was
   exported.
2. **Detect** (§4.2). Refusal comes before anything is written.
3. **Re-express.** `WebServedGraph` reads the §3.1 arrays:
   - the per-layer edge tables `conv.lin(edge_proj(vocab))[:91]` are folded into initializers;
   - every scatter-sum is `scatter_add` over the expanded index, so it lowers to ScatterElements;
   - the maxes are `scatter_reduce(amax)`.

   It must read **bit-identical** to `served_outputs` in torch on the goldens before any ONNX is written, as the
   research's did.
4. **Write**, each file named `<stem>.<kind>`, where `<stem>` is the stamp's own `{run_id}_{step:08d}_{sha8}`. That
   carries the run id and a content hash, as LAW-12 asks of an artifact (§13 A8). Each file's own sha256 is in the
   manifest.
   - **`<stem>.fp32.onnx`:** torch.onnx dynamo, opset 18, dynamic N, E, L and G, no external data, `onnx.checker`.
     The op census must show **zero ScatterND**; the exporter refuses its own output otherwise.
   - **`<stem>.fp16.onnx`:** fp16 storage of the initializers, with a Cast to fp32 at the graph's head. Compute and
     the scatter accumulation stay fp32 (LAW-06, accepted by R399(h)). It ships to WebGPU only, under R399(a)'s line
     (§5), pre-stated in L1's PRESTATED and read on run13's 120k. Until that read, fp32 is the only shipped file.
   - **`<stem>.weights.bin`:** dumped from the same `WebServedGraph`, so its tensors are the ONNX's.
   - **`<stem>.witness.json` and `<stem>.manifest.json`:** §4.3, §5. (L1 wrote the device witness as JSON, the
     16 positions' stones with their served logits, probs and values, rather than a binary file.)

**A dirty tree.** The exporter records `dirty: true` in the manifest rather than refusing, so tests run while
iterating. The release refuses a dirty manifest (§10).

**Dependencies.** A new uv dependency group, `web-export`: `onnx`, `onnxscript` and `onnxruntime` (CPU), pinned to
the research's 1.23.2 / 0.7.2 / 1.31.0.
- It is in `default-groups`, so the gates have it.
- It is declared in conflict with the `teacher` extra, whose `onnxruntime-gpu` provides the same module, so
  `--extra teacher` needs `--no-group web-export`.
- `tools/distill/teacher.py`'s install hint moves with it.
- ORT 1.31 reports to an outside endpoint at import. The exporter sets `ORT_DISABLE_TELEMETRY=1` before it
  imports ORT, so neither an export nor its gate sends anything (L1).

### 4.2 The detector

The detector is a closed table, matched exhaustively, with no default arm (LAW-11).

| stamp field | carried | refused by name |
|---|---|---|
| `arch_kind` | `GnnArchV2`, `GnnArchV2SoftPolicy`, `GnnArchV2SoftPolicyFuture`; the served path is V2's `forward_batch`, and the aux and future heads are not served or exported | `GnnArch`: its mean-only readout with no divisor is another graph; no lineage net uses it |
| encoding row | `gnn_axis_r8`, `gnn_axis_r8_pruned`: the same net, with the builder's `empty_edges` taken from the row | `hexcanvas_r8` (no model, R398(b)); `gnn_axis_v1` (radius 6: no goldens at its radius, no lineage net) |
| `deploy.search.kind` | `puct` | `gumbel` (the player carries PUCT only) |

**The registry.** The stamp carries no registry hash. The exporter checks two things:
- the stamp's encoding name resolves through `mantis.encoding.lookup`;
- the import-time handshake held, so the on-disk TOML's sha equals the compiled module's `registry_sha()`.

It writes that `registry_sha` to the manifest. The player refuses a manifest whose `registry_sha` differs from its
own compiled `registry_sha()`, so the wasm engine can never serve a net under another registry.

Each refusal is a test row. A later carried row needs its own goldens and its own gate-20 row.

### 4.3 The manifest (`<stem>.manifest.json`, format `mantis-web-manifest-v2`)

| section | fields |
|---|---|
| `stamp` | file name, sha256 of the file's bytes, `run_id`, `step`, `lineage`, `commit_sha`, `arch_kind`, `encoding`, `weights` (`ema` or `learner`), `net_param_hash` (recomputed after `deploy_state`; the stamp carries none) |
| `exporter` | the tree's commit, `dirty` (tracked changes, read before anything is written; true when git cannot say), torch, onnx, onnxscript, onnxruntime, opset and the export's seconds |
| `encoding` | name, `registry_sha`, `win_length`, `graph_radius`, `trunk_size`, `empty_edges`, `node_feat_dim`, vocabulary rows (91) |
| `deploy.resolved` | §2.2's first table, as the resolvers returned it from the stamp's `config` |
| `deploy.player` | §2.2's second table, as the player's constants (the player refuses a manifest that disagrees with its own). It arrived with the player in L2, as manifest v2; manifest v1 held `deploy.resolved` alone |
| `files` | per file: name, sha256, bytes; per ONNX file its dtype, op census and `passes_its_line` (every row of that file holds its line); for the weights file the tensor table |
| `witness` | the lines; the goldens' path and sha256 (`d5c5a7ba…`); and one row per (backend, file, threads, B): max \|Δ\| logit, prob and value, argmax agreed / compared, its line (fp32 or R399(a)'s fp16) and pass; a WebGPU-only ScatterND file's rows are marked "wrong under threaded CPU kernels (ORT #18048)" (R399(g)) |

- **Rows filled at export:** `ort-cpu` at 1 thread and at default threads, on both ONNX files, at B 1, 8 and 16; and
  `rust-forward-native` (through the bridge, from L2).
- **Rows filled at release** (§10): `rust-forward-wasm-node` and `ort-web-webgpu` (with browser and adapter).
- The player and the Worker refuse a manifest whose format, encoding, `registry_sha` or deploy kind they do not carry,
  or whose file sha256 does not match the bytes fetched.

## 5. The witnesses

| | what | positions | lines | where it runs | planted break, shown red where the witness runs |
|---|---|---|---|---|---|
| W1 | each shipped backend's outputs against the deploy path's `served_outputs` (CPU, fp32, eager) | the 512 goldens at B 1, 8, 16 plus the five openings | fp32: \|Δ logit\| ≤ 1e-3, \|Δ prob\| ≤ 1e-4, \|Δ value\| ≤ 1e-4, same argmax wherever the reference's top-2 gap > 2e-3 (RESEARCH-WEB's pre-stated lines, now in code); fp16 (R399(a), WebGPU only): \|Δ prob\| ≤ 2e-3, \|Δ value\| ≤ 2e-3, same argmax wherever the top-2 gap > 2e-2, logits reported, not gated; its known-bad, the fp16 file's last policy bias +0.05, must fail the prob line | export (ORT CPU, 1 and default threads; gate 20), release (WebGPU in Chromium on a hardware adapter; wasm in node), each device at start (16 positions) | the **written** ONNX's last policy bias +1.0 (every logit moves; softmax ignores it), its last policy weight ×50 (the probs), one value bin +4 (the value), each red on each line that gates it; an argmax-only flip (gate 20, tiny stamps) |
| W2 | `RustForward` against the ONNX (ORT CPU, 1 thread) | the same | the same | gate 20 (native, through the bridge), gate 21 (wasm in node, on the tiny stamp) | `RustForward` with the degree divisor dropped (gates 20 and 21) |
| W3 | the player against the Python deploy head: same stones, same per-stone rows | the 512 goldens, each one full turn (1 or 2 stones) at `Sims(128)` in the shipped posture | identical stones; identical `last_sims`, `last_stopped`, reuse, select, cache and tactics rows | the integration tier (512); the default tier (32) | (i) the player's reroot replaced by `new_game`, which reds the reuse rows on every searched two-stone turn (the test pins that the reference run has at least one among the 32); (ii) the early stop comparing `≥ left`, which reds only at a tie (default tier, 32 positions) |

**W3's stub.** One Python function, `stub_outputs`, computes probs and value from a graph's legal rows: their node
features and their graph. It hashes the features' bytes into small integers c_j. Then probs_j = c_j / Σ c over the
graph, in f32, and the value is a fixed map of the graph's sum.
- **The Python head** gets it by monkeypatching `served_outputs`, as RESEARCH-WEB's `capture()` patched it, so the
  head's whole path runs: engine, cache, collate, assemble and expand.
- **The player** gets it through `PyPlayer`: pull a batch, call the same function, push.
- **The two adapters**, from GraphBatch and from WebBatch to the per-graph legal rows, are pinned by their own test:
  on the goldens they give identical rows.
- Given that, and one Rust builder under both, the outputs are bitwise equal.
- The positions run at concurrency 1, one game at a time: bit identity is a claim at concurrency 1 only (R394(f)).

**The tiny stamps.** Gates 20 and 21 export a random-init stamp of each carried kind on `gnn_axis_r8`, and the
`gnn_axis_r8_pruned` row on the future kind (four of the six pairs; the row changes only the batch), seeded, written
through the one checkpoint writer into a temp directory. There are no loose weights in the tree. The real stamp's
rows are a release step (§10).

## 6. The `web/` package

### 6.1 Layout

```
web/
├── package.json          # "private": true until the operator's release word (§13 O2)
├── package-lock.json     # onnxruntime-web (exact), typescript, puppeteer-core (dev, release witness only)
├── tsconfig.json
├── src/
│   ├── index.ts          # the typed API (§6.2), main thread
│   ├── worker.ts         # owns the wasm Player and the backend; the message handler and the loop (§6.3)
│   ├── backends.ts       # RustForward (into wasm) and OrtWebGpu with its per-call edge cap; selection and witness (§3.4)
│   └── arena.ts          # the analysis-slot packet adapter (§6.4)
├── demo/                 # index.html + demo.ts: play vs mantis, the LAW-18 rows (§6.5)
├── test/                 # node --test: the API, the worker handler, the adapter, W2 in wasm, the budget in situ
└── scripts/
    ├── build.sh          # RUSTFLAGS=-Ctarget-feature=+simd128 cargo build -p mantis-player --target wasm32-unknown-unknown --release; wasm-bindgen --target web
    └── release.sh        # §10
```

- The package ships `dist/`: the JS, the `.d.ts`, the wasm module and the bindgen glue, the ORT WebGPU runtime files
  copied from `node_modules`, and `net/` (§10). `web/dist/` and `web/node_modules/` are gitignored.
- The integrator serves the package's files from its own origin and passes `assets`, their base URL. The engine
  fetches nothing from anyone else.
- `+simd128` is set by the committed build script. LAW-13 bans `target-cpu`. simd128 is a portable target feature
  that every evergreen browser has, and the engine refuses where it is missing (§3.4).

### 6.2 The typed API

```ts
type Cell = { q: number; r: number };
type Position = { stones: Array<Cell & { p: 1 | -1 }>; toMove: 1 | -1; stonesLeft: 1 | 2 };
type Budget = { turnMs: number } | { sims: number } | { unlimited: true };
createEngine(opts: { assets: string; backend?: 'auto' | 'webgpu' | 'cpu' }): Promise<Engine>;
interface Engine {
  info(): EngineInfo;   // backend, adapter, witness row, leafMs, rootMs, minTurnMs, module and peak bytes, the manifest's stamp
  newGame(opening: Position, clock: { turnMs: number } | { unlimited: true }): Promise<void>;  // sizes the pool (§2.6)
  play(stones: Cell[]): Promise<void>;                        // the other side's turn, rules-checked by the engine
  playTurn(budget: Budget): Promise<TurnResult>;
  analyse(position: Position, budget: Budget): Promise<TurnResult>;  // replaces the game in play (§2.4)
  abort(): void;        // the turn in flight rejects with AbortError; the board is as before it
  dispose(): void;
}
```

- `TurnResult` is §2.4's, in camelCase.
- Errors are typed. Every §2.7 `PlayerError` variant reaches JS by name, `OutOfStep` among them (also for `analyse`
  or `playTurn` while a turn runs). The Worker adds `AbortError`, `NoBackendPassed`, `Unsupported` and
  `EngineCrashed`.
- After `EngineCrashed` the Worker has already rebuilt the engine (re-instantiated, the backend re-selected) and
  counted the trap in `info().engineTraps` (R399(h)). The game in play is gone, and the integrator starts a new one.
- A match clock is the integrator's to map to a turn's `turnMs`; the arena's mode decides it (§13 O1).

### 6.3 The Worker and its loop

- The main thread posts `{id, op, args}` and receives `{id, ok, value | error}`.
- `worker.ts` exports its handler, so node tests drive it without a browser worker.
- **One loop for both backends:**

  ```
  begin_turn(budget, performance.now())
  loop:
    step = next_batch(performance.now())
    if step is Turn: resolve
    if backend is cpu: forward_pending(); submit_pending(performance.now())
    else: submit(await ort(copy(step.batch)), performance.now())
    await macrotask()            // a MessageChannel round trip, so a queued abort message runs
    if aborted: player.abort(); reject with AbortError
  ```

  So every phase is timed on the host's clock on both paths, and an abort is heard within one batch. A microtask
  `await` would not let the message run.

### 6.4 The arena adapter

The adapter speaks the analysis slot's `postMessage` packets, as RESEARCH-WEB Q3 read them at HeXO-Arena `develop`
8817871a (`apps/web/src/analysis/worker-source.ts`, `packages/contract/src/htttx.ts`).
- **On start:** it posts the capabilities `free_setup`, `resettable_state`, `dual_sided`, `request_id` and
  `interruptible`.
- **`setup`:** holds the position. The adapter calls `newGame` at the first `move_request`, sized for 5 s, the
  slot's largest request, so the pool fits every deepening step and is not regrown. A play mode passes its own
  clock (O1).
- **`move_request {side, previous, move_time_limit?, request_id}`:**
  - The adapter applies the unseen turns with `play`.
  - It runs `playTurn({turnMs: move_time_limit·1000})`, or `{unlimited: true}` when the limit is absent.
  - It answers `move_response {move: {pieces: [2 cells], evaluation}, considerations ≤ 2, request_id}`.
  - The arena requires exactly two pieces. When the turn's first stone wins, the adapter adds as the second piece the
    first legal empty cell in the engine's root children. A packet fixture pins it.
- **What `move_time_limit` means.** It is the seconds left when the frame is sent. In the analysis slot that is the
  analysis time, and on a turn clock it is the turn's time; both map straight to `turnMs`. On a match clock it is the
  whole remaining clock. The adapter never maps a match clock: that mapping is the play mode's, the operator's (O1).
- **The slot deepens from 0.25 s**, below any `minTurnMs`.
  - On WebGPU a floor turn (about 0.66 s) fits inside the slot's 3-s grace, and plays late and flagged.
  - On one CPU thread the floor is 16 sims (R399(d)). A floor turn costs about 2.0 s on the desk and about 2.8 s on a
    0.7× laptop core, so it fits inside 0.25 s plus the 3-s grace too (INFERENCE from §2.5's costs).
  - Every request is answered, late and flagged where it is below `minTurnMs`.
- **An interrupt** gives `abort`.
- **`evaluation`'s mapping** (`heuristic` from the root value; `win_in` from a root proof's turns) is L3's to fix
  against the arena's definition at that sha.
- The fixtures are packets transcribed from those files. Nothing of the arena's is run.

### 6.5 The demo page

- `demo/` plays a game against the engine on a local static server. It shows the board, the clock and the
  per-turn LAW-18 rows:
  - sims set and spent per stone, stops fired;
  - reuse inherited, cache hits and served;
  - tactics decided;
  - turn wall ms, `turn_over_0.8T`, `floor_over_clock`, `leaf_ms`, `root_ms` and `minTurnMs`;
  - the backend and its witness row, the module's bytes and the wasm memory's peak.
- It connects to no run and reads no run record. R398(c) admits it, and the amendment names it (§8).

## 7. Contract #13 — `docs/contracts/web_engine.md`

- **Owners:** `mantis-player` for the batch, the step machine and the manifest reader; `web/` for the typed API and
  the Worker messages.
- **Sections:**
  - Summary;
  - the batch (§3.1) and its outputs;
  - the step machine (§2.4) and its named errors;
  - the manifest (§4.3) and its refusals;
  - the witness lines (§5);
  - the typed API and the Worker messages (§6.2, §6.3);
  - the adapter's packet map (§6.4);
  - who asserts what where, and the pinning tests.
- **Versions:** it lands at L1 as v1, holding the batch, the outputs and the manifest. L2 and L3 each extend it and
  bump its version in the commit that changes it, as every contract does. The first release tag records the version
  it ships.

## 8. The repo_design amendment and its touch-points (each lands with the code it describes)

**L1's first code commit:**
- **repo_design §1 layout:**
  `crates/mantis-player/  # the single-thread deploy driver (pull/push), the web batch, RustForward; native + wasm32`.
- **repo_design §2 DAG:**
  `mantis-player → core, graph, encoding, search, selfplay  # wasm32-clean on its paths; no pyo3; one #[wasm_bindgen] module`.
  The bridge row reads unchanged and now includes the player; the bridge gains `web_batch`.
- **repo_design §4 contracts.** Row 13: `web engine API | v1 | the batch both net backends read, the pull/push step
  machine, the export manifest and its refusals, the witness lines, the typed API and Worker messages`.
- **repo_design §7:** the `web-export` dependency group and its conflict with `teacher`. `tools/distill/teacher.py`'s
  install hint moves with it.
- **CLAUDE.md map:** the crate's line.

**L2:**
- **repo_design §2:** the `mantis-search` row notes `with_pool` and `armed_sims_for`. The bridge gains `PyPlayer` and
  `PyRustForward`.
- **Gate 4:** it gains `cargo check -p mantis-player --target wasm32-unknown-unknown --locked`, so the player's wasm
  build is gated from L2 on.

**L3:**
- **repo_design §1:** `web/  # the browser package: Worker, typed API, backends, arena adapter, demo, node tests; npm
  from a tag`, and the Makefile comment's target list gains `web`.
- **"Deliberately absent"** (repo_design §1 and CLAUDE.md) names the demo page as admitted by R398(c), with no run
  input. CLAUDE.md's "One ruled exception exists" becomes two.
- **repo_design §7:** `make web` builds the player for wasm32 (`+simd128`) and runs wasm-bindgen. The wasm-only
  dependencies sit in the player's manifest alone. The third toolchain is npm under the node already pinned.
- **`mise.toml`:**
  - it pins `"cargo:wasm-bindgen-cli"` at the crate's exact version;
  - its header's "the ONE tool rust-toolchain.toml does not cover" and "Node is a TYPE-CHECKING dependency only"
    are rewritten: node now builds and tests a shipped package.
- **repo_design §9:** gate 21.

`src/mantis/deploy/`'s reservation retired by R399(h). R399's landing commit deletes the empty package and amends
repo_design §1 and CLAUDE.md's map line.

## 9. Gates and toolchains

| gate | runs | reds on |
|---|---|---|
| 20 export pipeline witness (L1) | `pytest tests/tools/test_export_web.py`: the tiny stamps through `export_web.py`; W1 at ORT CPU 1 and default threads; W2 native (from L2); W1's and W2's planted breaks red; the re-expression's refusal red; the detector's refusals; the census of no ScatterND, no GatherND and every index a Gather or a scatter reads int32 (an untyped index refuses); the manifest's fields | any gating line, any refusal not raised, any ScatterND, a planted break that stays green |
| 21 web package (L3) | `make web` (wasm32 build + wasm-bindgen), `tsc --noEmit`, and `node --test web/test` on the tiny stamp's export | a build failure, a type error, a test, W2 in wasm or its break |
| 4 (from L2) | adds the player's wasm32 `cargo check` | a dependency or call the wasm build cannot carry |
| 2a / 2b | `mantis-player`'s Rust tests and clippy, as a workspace member: the budget function, the cache port, the step machine, the served-sims pin (§13 A3), the error-parity test | as today |
| 3a / 3b | W3: 32 positions in the default tier with both breaks red, 512 in integration | any stone or row differing |

- **No `web` marker (L1).** The default-tier expression is pinned in six places and a tier census, so a deselecting
  marker would move all of them. The module stays in the default tier, and gate 20 runs it again by path, about
  15 s twice.
- **Toolchains:** Rust 1.97.1 already pins `wasm32-unknown-unknown`, and node 26.7.0 is already pinned. New pins:
  `wasm-bindgen-cli` in `mise.toml`, and npm's lock in `web/`.
- **Network:** gate 21's `npm ci` needs the registry once per clone, like `uv sync`.

## 10. Release procedure

A release is reproducible from three names in the manifest: the stamp's sha256, the exporter commit, and the witness
rows.

1. **The operator's words first:**
   - the licence of the released artefacts (§13 O2);
   - the package's name and scope;
   - the stamp of record. It is run14's pick by R398(e), or the uniform average of its last eight saves if that average
     wins the end read; such a stamp is lineage `avg(run14:…)`. run13's 120k is the development net until then.
2. **Export.** On a clean tree at the release commit: `tools/export_web.py --checkpoint <stamp> --out web/dist/net/`.
   This writes the export rows. Any gating W1 or W2 miss is a HALT: report it, never patch the net.
3. **Build.** `web/scripts/release.sh` refuses a manifest with `dirty: true`. It builds the package, then runs the
   node tests and the wasm witness row on the real files.
4. **WebGPU witness on the desk.** Headless Chromium (`/usr/lib/chromium/chromium`, `--disable-vulkan-surface`), with
   an 8-s pause between WebGPU launches and an adapter check that refuses SwiftShader (RESEARCH-WEB's traps), on a
   non-isolated page. It writes the `ort-web-webgpu` rows for each shipped ONNX.
5. **Pack and tag.** `npm pack` writes the tarball (npm fixes the tar's mtimes). The tarball's sha256 goes into the
   release record. The tag is `web-v<semver>` on that commit.
6. **Publish.** Publishing to the npm registry is outward-facing and is the operator's act. The session hands over the
   tarball and its sha256.

The net files are built at release and live in the tarball and the mirror. They are never committed.

## 11. The build plan

| leg | sessions | lands | its pins | HALT |
|---|---|---|---|---|
| L1 EXPORT | 1 | `mantis-player` opened with `batch.rs` only; the bridge's `web_batch`; `tools/export_web.py`; the `web-export` group; contract #13 v1; gate 20; the amendment's L1 part | W1 at ORT CPU; the detector; zero ScatterND; the torch re-expression bit-identical; W1's break red | any export failing a gating golden-equality line |
| L2 DRIVER | 2 | the per-stone loop, the cache port, the step machine, `budget.rs`, `RustForward`, `with_pool` and `armed_sims_for`, `PyPlayer` and `PyRustForward`, `wasm.rs`, gate 4's player row; LAWS' protected set gains W3 and the served-sims pin (R399(c)) | W3 with both breaks; W2 native with its break; the budget's Q5 rows, `cap ≥ 32` and the 16-sim floor; `with_pool(.., MAX_NODES)` and `MAX_ARMED_SIMS` with a planted break on the pool size, self-play's goldens untouched (R399(b)); the served-sims pin; the constants' pins; the error-parity test; the `#[wasm_bindgen]` census; the wasm32 build, with module size and peak memory reported | a witness failing; any edit to self-play, the cells or the Python head beyond A2's additive set; a dependency the wasm build cannot carry |
| L3 PACKAGE | 1–2 | `web/`: the Worker and its loop, both backends, selection and the on-device witness, the typed API, the adapter, the demo, node tests, gate 21, the release script; contract #13's API sections | W2 in wasm; the adapter's packet fixtures; §3.3's pre-stated WebGPU measurement and W1 on WebGPU in Chromium (desk) | a runtime fetch from a third party; a data op on the CPU; §3.3's measurement missing its line (then §13 A7) |
| L4 BENCH + READ | 1 | numbers only (§14) | — | a witness failing |

Every leg ends with `make gates` and a fresh read-only review. L2's two sessions each get one.

**L4's read, its lines pre-stated in L4's PRESTATED before any cell.**
- **The bench:** leaf cost per backend, on the desk in headless Chromium on a non-isolated page and on one real laptop
  if the operator has one; and the sims the budget sets at 5, 10 and 20 s.
- **The clock line:** zero clock losses in every cell, and p99 turn time ≤ 0.8 T. That is R396(c)'s standard for the
  ladder's flip, re-aimed at the browser by R397(b). Its known-bad: the same bot with ρ pinned at 0.5 under the 0.8·T
  rule. Its turns run about 1.2–1.5 T, so the pre-stated reading is p99 > T, with clock losses.
- **The strength line:** the shipped bot at the 10-s clock in the browser against Six at n128 and Bubble.
  - It must not be below the comparison cell beyond that cell's CI. The comparison cell is the native deploy head on
    the same net, book and opponent, 576 games, at a fixed sims equal to the browser run's median sims a stone.
  - Bubble's budget is the same fixed time per move in both arms: the browser run's measured mean turn time.
  - The known-bad: the browser bot at 8 sims a stone, three or more doublings below what the budget gives on every
    backend. It must read below the line.
- **The harness** lives in records, not the tree. Bubble is the operator-sanctioned 410k release in its isolated
  directory, as RUN12-GO and TR-D ran it; any other Bubble build needs the operator's word.
- **What the read is.** It is a wall-clock read of the product, not a promotion bar (LAW-15), and its CIs count
  trajectory-distinct games (LAW-04, eff_n stated per cell).
- **Its biases, stated now:**
  - The budget varies sims turn by turn while the comparison is fixed. Strength is concave in sims, so the varying arm
    reads slightly low.
  - The browser runs fp32 while the box's cells run bf16 under CUDA autocast. The browser is the closer to the trained
    net.
  - At 576 games the strength line catches only a gross loss: about −0.27 logit at 0.8 power, at SAVE-AVG's measured
    spread. L4's PRESTATED restates the power for its own cells.

## 12. Risks, each with what would show it

| risk | shows as | answer |
|---|---|---|
| the elements graph is slow or too big on WebGPU | §3.3's pre-stated L3 measurement | the per-call edge cap; §13 A7 |
| int32 inputs still leave a data op on the CPU under WebGPU | L3's placement log | a W1 row failure; move the op into the engine's batch |
| `pack_wire` or the leaf build touches a thread or a clock on wasm | a trap in node at L2 | the wasm32 node test packs a batch ≥ 65 536 edges at `threads = 1` |
| fp16 storage misses its line | W1 fp16 rows against R399(a)'s line | fp32 only |
| leaf size grows past the previous turn's cost late in a game | `turn_over_0.8T` and the p99 turn time | the 0.8 margin; L4's clock line |
| a weak integrated GPU is slower than one CPU thread | L4's laptop row | selection by measured `leaf_ms` is the change it would order |
| ORT-web's runtime (26.8 MB in memory) is too heavy for a phone | the demo's peak row | CARD-WEB-RUNTIME-ALTERNATIVES (burn-wgpu) |

## 13. Open items and owners

**The operator's:**
- **O1 — the arena side** (CARD-WEB-ARENA-SIDE):
  - `'wasm-unsafe-eval'` in the CSP, where a pinned test forbids it today;
  - the play-vs-mantis mode on the analysis packets, with its clock: a turn clock maps straight to `turnMs`, and a
    match clock needs the mode's own rule;
  - serving ~4 MB (CPU path) to ~6–8 MB (WebGPU path) with compression. Caddy compresses `application/wasm`, not
    `application/octet-stream`;
  - offering a device only clocks at or above the engine's `minTurnMs`. On one CPU thread that is 4.81 s on the desk
    and about 6.8 s on a 0.7× laptop, against the arena's 5-s floor. This is the one guard against a clock loss
    (§2.5, §13 A4).

  COEP stays off.
- **O2 — the licence** before the first tag.
  - The source in this repository is MIT (`pyproject.toml`); the released artefacts include the net's weights, whose
    licence is the operator's decision.
  - Until then `package.json` stays `"private": true`, with no licence field claimed.
  - Also the package's name and scope, the npm account that publishes it, and the publish itself.
- **O3 — a laptop for L4**, if there is one.

**The architect's, ruled by R399 (2026-10-21).** Each item carries its ruling.
- **A1 — fp16's line.** Rounding the weights to fp16 moves logits by about 1e-3 relative (INFERENCE), so the fp32
  lines will likely fail.
  - The recommendation: fp16 ships to WebGPU only under a line pre-stated in L1's PRESTATED before its first read,
    with a known-bad that must fail it.
  - The line: same argmax wherever the gap > 2e-2, and \|Δ value\| and \|Δ logit\| at 3× the rounding floor measured
    on the development net (run13 120k), not on a random-init one.
  - Until it is ruled, fp16 rows gate nothing and fp32 is the only shipped file.
  - **Ruled — R399(a).** fp16 ships to WebGPU only, under a line pre-stated in L1's PRESTATED and measured on run13's
    120k: \|Δ prob\| ≤ 2e-3, \|Δ value\| ≤ 2e-3, the same argmax wherever the reference's top-2 gap > 2e-2. Logits
    are reported, not gated. The known-bad, the fp16 file's last policy bias shifted by 0.05, must fail the prob line.
    Until the read, fp32 is the only shipped file.
  - **L1's read (2026-10-21, `mantis-records/deploy-web/l1/READ.md`).** On run13's 120k the fp16 file MISSES the line:
    |Δ prob| 2.26e-3 and |Δ value| 2.75e-3 against 2e-3, argmax 502/502. fp32 is the only shipped file. The ruled
    known-bad cannot fail: the last policy layer has one output, so its bias shifts every logit alike and the softmax
    ignores it. It read exactly the unperturbed file's prob and value. The proposed substitute, the last policy weight
    ×1.05, read |Δ prob| 3.45e-2. Which control stands is the architect's, needed only if fp16 is ever to ship.
- **A2 — scope.** The packet HALTs on "any edit to self-play". The port adds, additively:
  - `MCTSTree::with_pool` and `armed_sims_for` to `mantis-search`, with `new_full`'s and `MAX_ARMED_SIMS`'s values
    unchanged;
  - `configure_search` sizing `raw_values` from `pool.len()` (the same value in self-play);
  - three bridge bindings: `web_batch`, `PyPlayer` and `PyRustForward`. The last gives the export its native forward
    row.

  This reads "edit" as behaviour, as R396(c) read TR-D's M1. The bridge's three pieces of glue (§2.2) are copied, not
  moved: moving them would edit the cells' path. The copies are pinned by W3 and an error-parity test, and are retired
  when CARD-DEPLOY-HEAD-DRIVER moves the bridge onto the shared code.
  - **Ruled — R399(b).** Not an edit to self-play. `new_full` stays `with_pool(.., MAX_NODES)` and `MAX_ARMED_SIMS`
    keeps its value, both pinned with a planted break on the pool size. The glue is copied, not moved, pinned by W3
    and the error-parity test, and retired by CARD-DEPLOY-HEAD-DRIVER. Self-play's goldens stay untouched and green.
- **A3 — the protected set.** R386(d) and R396(c) annotate served-sims exactness for "the ladder alone" as the
  consumer of the early stop and reuse. The player is a second consumer, with both on. Its constants are pinned equal
  to `LADDER_EARLY_STOP` and `LADDER_TREE_REUSE`, so the test is a third reader of those switches. Its own pins
  should be named in the protected set by a ruling:
  - W3;
  - a Rust served-sims test: each stone spends its `n` in new descents, or a fired stop with the leader fixed.
  - **Ruled — R399(c).** Both enter the protected set with L2. R386(d) and R396(c) are annotated: the player is the
    second consumer of the early stop and reuse, its constants pinned equal to `LADDER_EARLY_STOP` and
    `LADDER_TREE_REUSE`.
- **A4 — no deadline stop.** A host-side stop at 0.95·T would make a timeout impossible but would end a search with
  the leader unfixed, which weakens served-sims exactness. It is not built: ρ = 1, the 0.8 margin, `minTurnMs` and L4's
  clock line stand in its place. Building it takes a ruling naming the invariant.
  - **Ruled — R399(d).** No deadline stop: served-sims exactness is the invariant. One change: under a clock below
    `minTurnMs` the CPU path plays a 16-sim floor, counted as `floor_over_clock` and shown (§2.5). The 32-sim floor
    stands for the budget rule and the arena's offered clocks.
- **A5 — a mis-citation in R398(b).** It cites "(R396(f))" for concurrency 1. R394(f) holds that rule ("Tree bridges:
  bit identity is a claim at concurrency 1 only"), and R396(f) holds annotations only. This design cites R394(f). A
  correcting annotation on R398's entry is the architect's.
  - **Ruled — R399(e).** R398(b)'s "(R396(f))" is annotated to R394(f).
- **A6 — a measured ρ.** ρ = 1 is RESEARCH-WEB's guaranteed column. A measured ρ would buy more sims: per descent the
  anchor cell reads 0.734 served leaves, and with reuse about 0.79 a nominal sim (INFERENCE, another net). Sims scale
  by 1/ρ, so that is 27–36 % more (equivalently, ρ = 1 plays 21–27 % fewer). It risks an overrun when one turn's ρ
  is low and the next's high. If ruled, the estimator takes the maximum per-descent cost over the game's last k turns,
  and L4's clock line must hold with it.
  - **Ruled — R399(f).** ρ stays 1. A measured-ρ estimator is a later card (CARD-WEB-MEASURED-RHO), admissible only
    after L4's clock line reads zero overruns at ρ = 1 on the desk. The 21–27 % fewer sims are the first release's
    price.
- **A7 — the WebGPU fallback**, if §3.3's measurement misses. Each option amends R398(b) ("the ScatterElements graph";
  "WebGPU where present"), so either takes a ruling:
  - (i) the ScatterND graph on WebGPU only. The research measured it exact there (atomics; the race is the CPU and
    WASM kernels'), at 7.0 ms a leaf at B 8. A second ONNX would ship, never on the WASM EP.
  - (ii) Or `RustForward` alone until CARD-WEB-RUNTIME-ALTERNATIVES.
  - **Ruled — R399(g).** (i) first: the ScatterND graph ships for WebGPU only, as a second ONNX, marked "wrong under
    threaded CPU kernels (ORT #18048)" in its manifest row, loaded by no path but the WebGPU backend, pinned by a node
    test. The integrator ONNX stays the ScatterElements graph. If the ScatterND graph misses too, (ii).
- **A8 — the law readings this design makes:**
  - **LAW-13:** a wasm trap kills the instance and surfaces as an error the Worker catches; that is read as "a panic
    crosses the FFI catchable".
  - **LAW-12:** exported files are artifacts named by the stamp's stem plus their own sha256 in the manifest.
  - **LAW-06:** a deploy re-expression accumulates in fp32, like the run's, without being the run's one
    implementation; W1 and W2 hold it to the served output.
  - **The reservation:** whether `src/mantis/deploy/`'s reservation retires, now that R398(b) put the export in
    `tools/`.
  - **Ruled — R399(h).** All three readings accepted. LAW-13: the Worker also rebuilds the engine after a trap and
    counts it. LAW-06: the export re-expression is a witnessed view of the served path, not a second implementation.
    The reservation retires: the export lives in `tools/`, the driver in a crate, the package in `web/`; CLAUDE.md's
    line goes with R399's landing.

## 14. Record (L4 fills)

**The sims the budget sets, per backend.** Measured leaf and root costs; sims a stone, median [p10, p90] over a game.

| device, backend | leaf ms (B 8) | root ms | T 5 s | T 10 s | T 20 s | `minTurnMs` | p99 turn / T | peak memory |
|---|---|---|---|---|---|---|---|---|
| desk, WebGPU fp32 | | | | | | | | |
| desk, `RustForward` | | | | | | | | |
| laptop, … | | | | | | | | |

**The read at 10 s.**

| opponent | browser bot (sims median) | native cell at that sims | Δ logit [CI] | eff_n | clock losses | line |
|---|---|---|---|---|---|---|
| Six n128 | | | | | | |
| Bubble, equal time | | | | | | |
