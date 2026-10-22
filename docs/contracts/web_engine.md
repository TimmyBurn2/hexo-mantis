# Contract: web engine API (#13)

- version: v4
- owner: crate mantis-player (`batch.rs`, the batch both net backends read; `player.rs`, `cache.rs` and `budget.rs`,
  the step machine; `forward.rs`, `RustForward`; `wasm.rs`, the browser surface); `tools/export_web.py` (the manifest
  and its witness rows); `web/` (the typed API, the Worker's messages and checks, the backends, the arena adapter)
- status: v4 — the `web/` package lands with DEPLOY-WEB L3: the typed API, the Worker's messages, its manifest checks,
  selection and the on-device witness, the WebGPU backend's per-call edge cap, and the arena adapter's packet map (v3,
  L2: `RustForward`, the wasm32 surface, W2's rows, manifest v2; v2, L2: the step machine, the cache, the budget; v1,
  L1: the batch, its outputs, the export's files, the manifest and the witness lines).

## Summary

The browser bot is one engine, `mantis-player`, with two net backends: `RustForward` on one CPU thread and ORT-web on
WebGPU (R398(b)). Both read ONE batch, built in the engine from the served path's own leaf build, fuse and pack
(`build_leaf_graph`, `GraphWire::from_axis_graphs`, `pack_wire`): every array beyond `x` is int32, so no index or mask
is derived inside the net's graph. The export writes, from a stamp, the ONNX graph in fp32 and fp16 over that batch,
the Rust forward's weights, a device witness and a manifest that binds them to the stamp and to every witness row.

## The batch (`mantis_player::batch::WebBatch`; `_engine.web_batch`)

G graphs block-diagonally, rows global: N nodes, E edges, L legal rows. All little-endian.

| array | dtype | shape | meaning |
|---|---|---|---|
| `x` | f32 | [N, node_feat_dim] | node features, as packed (11 at the graph rows) |
| `src`, `dst` | i32 | [E] | each edge's global node rows |
| `code` | i32 | [E] | each edge's row in `edge_vocabulary(win_length)` (91 rows at win length 6) |
| `legal` | i32 | [L] | the legal cells' global node rows, strictly ascending |
| `node_graph`, `legal_graph` | i32 | [N], [L] | the graph of each node and of each legal row |
| `divisor` | i32 | [N] | 1 on a real node (a stone or legal), else max(in-degree, 1) |
| `real` | i32 | [N] | 1 on a real node, else 0 (the value head's max) |
| `mean_mask` | i32 | [N] | 1 on a stone, or on every node of a graph with no stones (the value head's mean) |
| `mean_den` | i32 | [G] | each graph's `mean_mask` count; it also carries G |

A graph's rows are laid out `[stones | legal | dummy]`, as the builder lays them; a stone is a row below the graph's
`n_stones`. The batch is refused (`BatchError`) on no positions, a row without graph geometry (`hexcanvas` by name), a
leaf's seam guard, any pack refusal (checks 4–16 of `graph_wire.md`, check 14 as a coded row outside the
vocabulary), or a count past int32.

## The outputs

| output | dtype | shape | meaning |
|---|---|---|---|
| `logits` | f32 | [L] | the policy head's logit per legal row |
| `probs` | f32 | [L] | the per-graph softmax over its legal rows |
| `value` | f32 | [G] | the 65-bin head's expectation over linspace(−1, 1, 65), clamped to [−1, 1] |

`probs` and `value` are what `served_outputs` returns; `logits` is the witness's.

## The export (`tools/export_web.py`)

Every file is named `<stem>.<kind>`, `<stem>` the stamp's own `{run_id}_{step:08d}_{sha8}` (LAW-12's reading, accepted by
R399(h)), and listed in the manifest with its sha256 and bytes.

| file | contents |
|---|---|
| `<stem>.fp32.onnx` | torch.onnx's dynamo export of the served forward over the batch, opset 18, N, E, L and G dynamic; every scatter-sum is ScatterElements(add), the maxes ScatterElements(max), every gather a Gather; every index a Gather or a scatter reads is int32 by inferred type (`narrow_index_casts`), a shape op's static int64 aside; a ScatterND, a GatherND or any other index, an untyped one included, refuses the export |
| `<stem>.fp16.onnx` | the same graph with every float initializer fp16 holds finite stored fp16 and cast to fp32 where read (a constant like finfo(f32).min stays fp32); compute stays fp32 |
| `<stem>.weights.bin` | the Rust forward's tensors, raw f32, linear weights transposed (in × out), in the manifest's tensor table's order |
| `<stem>.witness.json` | the device witness: the five openings and eleven goldens spread over stone count, with the served `logits`, `probs` per position and `value` |
| `<stem>.manifest.json` | the manifest below |

## The manifest (`mantis-web-manifest-v2`)

| section | fields |
|---|---|
| `stamp` | `file`, `sha256` of its bytes, `run_id`, `step`, `lineage`, `commit_sha`, `arch_kind`, `encoding`, `weights` (`ema` or `learner`), `net_param_hash` (recomputed; the stamp carries none), `arch` (its widths) |
| `exporter` | `commit`, `dirty` (tracked changes, read before any file is written; true when git cannot say), `torch`, `onnx`, `onnxscript`, `onnxruntime`, `opset`, `export_s` |
| `encoding` | `name`, `registry_sha` (the compiled registry's), `win_length`, `graph_radius`, `trunk_size`, `empty_edges`, `node_feat_dim`, `edge_vocab_rows` |
| `deploy.resolved` | from the stamp's own config through the run's resolvers: `search_kind`, `c_puct`, `fpu_reduction`, `quiescence_enabled`, `quiescence_blend_2`, `leaf_batch_size`, `tactics` (the audit's `mode` included) |
| `deploy.player` | the player's own constants as the engine reports them (`_engine.player_constants()`): `early_stop`, `tree_reuse`, `cache_entries`, `virtual_loss` (v2) |
| `inputs`, `outputs` | the names above, in the graph's order |
| `files` | per file: `name`, `sha256`, `bytes`; per ONNX file `dtype`, `ops` (its op census) and `passes_its_line` (every row of that file holds its line); for the weights `tensors` (`name`, `rows`, `cols`, `offset` in floats) and `passes_its_line` (W2's rows) |
| `witness` | `lines`; `goldens` (path, sha256, positions, openings); `rows`, one per (backend, file, threads, B): max \|Δ\| `logits`, `probs`, `value`, `argmax_agreed` / `argmax_compared`, `line`, `pass`. Backends: `ort-cpu` (W1, both ONNX files, 1 and default threads) and `rust-forward-native` (W2: `RustForward` over the weights file against the fp32 ONNX at ORT CPU 1 thread, held to the fp32 line) |

## The detector

A closed table; a value it does not carry is refused by name (`ExportRefused`, its `field`), before anything is written.

| field | carried | refused |
|---|---|---|
| `arch_kind` | `GnnArchV2`, `GnnArchV2SoftPolicy`, `GnnArchV2SoftPolicyFuture` | `GnnArch` (another readout); any other |
| `encoding` | `gnn_axis_r8`, `gnn_axis_r8_pruned` | `hexcanvas_r8` (no model), `gnn_axis_v1` (no goldens at radius 6); any other |
| `deploy.search.kind` | `puct` | `gumbel` (the player carries PUCT only); any other |

The export also refuses (`field`): `registry`, a stamp whose encoding the import-time registry handshake did not check;
`batch`, an engine batch whose `x` is not the served batch's; `re-expression`, one not bit-identical to `served_outputs`
on every golden batch; `graph`, a written graph with a ScatterND, a GatherND or an int64 batch-sized index, or a
narrowing that does not type-check (this stem's files are removed); `stamp` and `config`, a stamp with no arch or config mapping. A venv synced with `--extra teacher
--no-group web-export` has no ONNX toolchain and runs no export.

## The witness lines

The reference is the deploy path's served output: `LocalInferenceEngine` on the CPU, eager, fp32, at its
`served_outputs` call, on the 512 golden positions (`tests/fixtures/graph_build/positions_r8.txt`) at B 1, 8 and 16 and
on the five openings as one batch. A graph's argmax is compared where the reference's top-2 logit gap exceeds the line's.

| line | \|Δ logit\| | \|Δ prob\| | \|Δ value\| | argmax gap | holds for |
|---|---|---|---|---|---|
| fp32 | ≤ 1e-3 | ≤ 1e-4 | ≤ 1e-4 | > 2e-3 | every shipped fp32 backend; a miss at export is a HALT |
| fp16 | reported | ≤ 2e-3 | ≤ 2e-3 | > 2e-2 | the fp16 file, WebGPU only (R399(a)); it ships only after its pre-stated read on the development net |

## The step machine (`mantis_player::player::Player`; `_engine.Player`)

One board at a time. The engine reads no clock and calls no net: every call takes the host's `now_ms`, and the net's
work is the host's between `next_batch` and `submit`.

| call | in | does | refuses (`PlayerError`) |
|---|---|---|---|
| `new(block, encoding, cpu_path)` | the manifest's `deploy.resolved` (PUCT only) | — | `Refused` (an unknown or non-graph row; `leaf_batch_size` 0; tactics outside `TacticsConfig::check`'s ranges; a block not PUCT, at the bridge) |
| `set_costs(leaf_ms, root_ms)` | the start-up warm-up's measured costs | — | `Refused` unless `leaf_ms` is finite above 0 and `root_ms` finite, not below 0 |
| `new_game(moves, clock)` | an opening as moves; `Sims(n)`, `TurnMs(t)` or `Unlimited` | a fresh tree with `4·1024·S_cap` nodes (`S_cap`: `n` held to [32, 976], `budget::pool_sims(t, …)`, or 976, so every pool's cap is at least the floor) and an empty cache | `OutOfStep` mid-turn, `IllegalStone`, `NoCosts` |
| `play(stones)` | the other side's stones | rules-checked, all or none | `OutOfStep`, `IllegalStone` |
| `begin_turn(budget, now_ms)` | `Sims(n)`, `Clock { turn_ms }` or `Unlimited` | the turn's sims a stone: `n`; `sims_for_turn` at the costs (floor 32, or 16 on the CPU path under `min_turn_ms`); the pool's cap | `OutOfStep`, `NoCosts`, `BudgetPastPool` |
| `next_batch(now_ms)` | — | runs the turn to its next cache miss or its end: `Pull::Batch(&WebBatch)` (the misses, deduplicated, first-seen order) or `Pull::Turn(TurnResult)` | `OutOfStep` with a batch outstanding or no turn; any search refusal |
| `submit(probs, values, now_ms)` | `probs [L]`, `value [G]` for the outstanding batch | assembles each leaf's policy by its slots, remembers it, expands | `OutOfStep`, `OutputShape`, `NonFinite` (a value), `Expand` |
| `abort()` | — | drops the turn; the board is as before it | — |

- **The per-stone loop** is the Python deploy head's `select_move`: the origin on an empty board; `reroot_or_new_game`
  (reuse on) after `check_tactics_board`; `root_offence` (a decided stone unsearched, `ProofStoneIllegal` read as none);
  the reuse rows; the root step `select_leaves(1)`; then, while descents < `n`: the early stop (one child, or leader −
  runner-up > descents left), `select_leaves_filled(leaf_batch_size, left)`, the cache, the expand; then the select
  rows, `get_top_visits(1)`, `root_audit` with `audit_order(None)`, the tactics rows. A turn ends on a win or its last
  stone.
- **The cache** is `GameEvalCache`'s: `LeafKey` keys, hits counted before the call's fresh entries go in, FIFO at
  `CACHE_ENTRIES` (8 192), a hit refreshing nothing, emptied by `new_game`.
- **A refusal mid-turn** rolls the turn back (the board as before it, the tree reset to it, so no virtual loss
  survives into a later re-root) and leaves the player usable.
- **`TurnResult`**: per stone the cell, its new descents, whether the stop fired, whether the root solver decided it,
  the visits a re-root kept, the root's visits and value, its two most visited children (cell, visits, prior, Q) and
  the tactics rows; per turn
  the sims set, `floor_over_clock`, `budget_pool_capped`, the wall ms and the served leaves. After a turn that served
  leaves, the costs become its wall time a served leaf, the root folded in (the engine sees no time inside a call); a
  turn that served none keeps them.
- **The rows**, cumulative and named as the head's `search_rows`: `stop_fired`, `stop_saved`, `select_calls`,
  `select_overlaps`, `select_network_leaves`, `reuse_reroots`, `reuse_inherited`, and `cache_calls`, `cache_positions`,
  `cache_hits`, `cache_served`, `cache_in_batch_repeats`, `cache_all_hit_calls`, `cache_evicted`.
- **The constants** (`_engine.player_constants()`): `early_stop` and `tree_reuse` on (the ladder's), `cache_entries`
  8 192 (the head's cache), `virtual_loss` 1.0 (`mantis-search`'s, read directly).

## `RustForward` (`mantis_player::forward`; `_engine.RustForward`)

`WebServedGraph`'s forward over the batch's own arrays, f32, one thread: the input projection; per layer the norm, the
fused edge message `relu(h[src] + table[code])` summed into `dst` in edge order, ÷ `divisor`, `+ (1 + eps)·h`, the
MLP and the residual, then the final norm into that layer's slice of the embedding; the policy MLP on the legal rows; the
`mean_mask` mean and the `real` max into the value MLP and its bins' expectation, clamped to ±1; a per-graph softmax.
`load(table, floats, node_feat_dim)` refuses a missing tensor, a shape that disagrees with another (`Shape`) and a
tensor past the file (`OutOfFile`); `forward` refuses a batch whose lengths or indices its own shape does not allow.

## The browser surface (`mantis_player::wasm`, wasm32 only)

`Engine` is the step machine and `RustForward` behind `#[wasm_bindgen]`, the one module that carries it (a census test
pins that). Each refusal is a JS `Error` whose `name` is the `PlayerError` variant, or `Weights` for the forward's.

| method | does |
|---|---|
| `new(encoding, cpu_path, c_puct, fpu_reduction, quiescence_enabled, quiescence_blend_2, leaf_batch_size, tactics?, audit?, audit_mode?)` | `deploy.resolved`, flat: `tactics` `[leaf_turns, leaf_nodes, root_turns, root_nodes]`, `audit` `[turns, nodes, k, m, total_nodes]` and `"hold"` or `"inverted"` |
| `load_weights(names, dims, floats, node_feat_dim)` | `RustForward` over the weights file: names one a line, `(rows, cols, offset)` each |
| `set_costs`, `min_turn_ms`, `new_game(moves, sims?, turn_ms?)`, `play(stones)`, `begin_turn(now_ms, sims?, turn_ms?)`, `abort` | as the step machine; cells as flat `q, r` pairs |
| `next_batch(now_ms)` | `true` when a batch waits for the net, `false` when the turn ended |
| `pending_batch()` | a copy of the pending batch, as a `Batch`, for a host backend |
| `forward_batch(batch)` | `RustForward` on any `Batch`: the on-device witness and W2 read it |
| `forward_pending()`, `submit_pending(now_ms)` | `RustForward` on the pending batch in place, then its outputs to the search: the batch never crosses into JS |
| `submit(probs, values, now_ms)` | a host backend's outputs |
| `turn_json()`, `rows_json()` | the last `TurnResult` and the rows, as JSON (a non-finite float is `null`) |
| `memory_bytes()` | the module's linear memory: its high-water mark |
| `Batch(x, src, …, mean_den)`, `.x()`, `.i32(name)`, `.n_graphs` | contract #13's batch in JS, from its arrays (the forward checks them) |
| `web_batch(encoding, stones, counts, sides, lefts)` | the engine's own `Batch` of positions: stones as flat `(q, r, p)` triples |
| `registry_sha()`, `player_constants()` | the compiled registry's sha256 and the player's constants (JSON), which a manifest must match |

The wasm-only dependencies (`wasm-bindgen` pinned to the lock's one version, `js-sys`, `serde_json`, and `getrandom`'s
`wasm_js`, which unifies into `rand`'s) live in the player's manifest alone; gate 4 checks the player for wasm32.

## The typed API and the Worker (`web/src/index.ts`, `web/src/worker.ts`)

`createEngine({assets, backend?})` starts a module Worker at `${assets}/src/worker.js` (`assets` resolved against the
page; it must be absolute in a blob: or data: context); every file is fetched from `assets` alone (the integrator's
origin): `net/index.json` → the manifest, the net files, `pkg/` (the engine), `ort/`
(the WebGPU bundle and the one runtime it loads, `env.wasm.wasmPaths` pointing there).

| method | message `op` | does |
|---|---|---|
| `info()` | — (cached) | backend, adapter, witness row and every candidate's row, `leafMs`, `rootMs` (0: folded), `minTurnMs`, module and peak bytes, the manifest's stamp, `engineTraps` |
| `newGame(opening, clock)` | `newGame` | the opening's stones as moves in turn cadence (x one, then two each, each side's in the order given); the engine's rules check them |
| `play(stones)` | `play` | the other side's stones |
| `playTurn(budget)` | `playTurn` | the one loop: `begin_turn`; per batch `forward_pending` + `submit_pending` (CPU) or the backend on `pending_batch()` + `submit` (WebGPU); a MessageChannel round trip between batches, so an `abort` is heard; `TurnResult` in camelCase with the rows |
| `analyse(position, budget)` | `analyse` | a new game at `position` (pool sized for the budget), then a turn |
| `abort()` | `abort` (no reply) | the turn in flight rejects with `AbortError`; the board is as before it |
| `dispose()` | — | terminates the Worker |

- **Messages:** `{id, op, args}` → `{id, ok: true, value}` or `{id, ok: false, error: {name, message}}`. The handler is
  exported (`handler(host)`), so node drives it with a host of its own.
- **Init refuses** (`Refused`): a manifest format other than `mantis-web-manifest-v2`, a `registry_sha` other than the
  engine's, a deploy kind other than PUCT, a `deploy.player` constant other than the engine's, a file whose bytes miss
  its sha256. `Unavailable`: a file the server did not serve (an HTTP error; on a candidate's file it is that
  candidate's failed row). `BackendFailed`: an ORT run that failed, its own abort included (a candidate's failed row,
  never read as a trap of the engine's module). `Unsupported`: no wasm SIMD. `NoBackendPassed`: no candidate's witness held (the rows in
  the message).
- **Selection:** WebGPU first when a non-fallback adapter exists (`backend: 'auto'`), ORT pinned to that adapter; else,
  on a miss, or when the candidate throws (a failed row with its error), the CPU path. Each runs the export's
  16-position witness on batches the engine builds at `leaf_batch_size`, at the fp32 line; outputs of another length
  than the reference's fail every line, and a witness that compared no argmax does not pass. A file missing its sha256
  is `Refused` for the net, not a failed backend. The
  engine is built for the backend that won (`cpu_path` arms the 16-sim floor). The warm-up cost a leaf is the
  witness's wall ms over its leaves, root folded in, until a turn measures its own.
- **WebGPU calls** split a batch at graph boundaries under `EDGE_CAP` edges (a larger graph alone), re-based; a graph's
  outputs do not depend on its batch-mates, so the split is invisible.
- **A failure mid-turn** (an abort, a backend's, the engine's refusal) rolls the turn back before it is answered, so
  the engine stays usable. A trap (`WebAssembly.RuntimeError`) is passed on untouched — the trapped module cannot be
  called again — and selection stops on one rather than running the next candidate on the same module.
- **A trap** (`RuntimeError`) or a Worker error rejects every pending call, the trapping one included, with
  `EngineCrashed`; the API terminates the Worker, starts a new one, re-runs `init` and counts the trap. A call made
  meanwhile waits for the new engine. It respawns only after a first successful `init` and at most `MAX_RESPAWNS` (3)
  times in a row; past that, or when a respawn's `init` fails, every call rejects with `EngineCrashed`. A first `init`
  that fails terminates its Worker. `dispose()` rejects every pending and later call with `AbortError`.

## The arena adapter (`web/src/arena.ts`)

HeXO-Arena's analysis slot (develop 8817871a: `packages/contract/src/htttx.ts`, `apps/web/src/analysis/worker-source.ts`),
htttx basic_websocket v1-alpha over `postMessage`; wire `(q, r)` is the engine's `(q, r)`, as `tools/ladder/wire.py` maps it.

| packet in | the adapter |
|---|---|
| (on attach) | posts `{meta: {version}, basic_websocket: {versions: {"v1-alpha": {free_setup, resettable_state, dual_sided, request_id, interruptible: true}}}}` |
| `setup {board: {cells: [{q, r, p}]}}` | holds the stones, queued behind the request in flight |
| `move_request {side, previous, move_time_limit?, request_id?}` | after a setup, `newGame` at the setup's stones plus `previous`, sized for 5 s; else `play` of the unseen pieces; then `playTurn({turnMs: limit·1000})`, or `{unlimited: true}` without a limit. A refusal answers nothing (the slot times it out); the next position is served after its setup |
| `interrupt {request_id?}` | `abort` of the request in flight; an interrupted request (one queued or in flight; with no id, the one in flight) is never started, played on or answered, whenever the interrupt lands |

- **`move_response`:** `{type, move: {pieces: [2], evaluation: {heuristic}}, request_id}`. A turn whose first stone wins
  is padded by that stone's first empty neighbour (q+1; r+1; q−1,r+1; q−1; r−1; q+1,r−1), then the first empty cell of
  each wider ring. `heuristic` is x-positive (htttx counts x positive): the last searched stone's root value for the
  mover, signed by side, 1 for a decided turn or one that placed fewer stones than it was due. No `win_in` and no
  considerations are sent; an interrupted request is answered with nothing.
- **A stale board:** after a refused request or a dropped answer the engine's board is no longer the slot's, so every
  request is refused (answered with nothing) until the next `setup`.

## Who asserts what where

| fact | asserted where | pinning test |
|---|---|---|
| the batch's derived arrays agree with the offsets they came from; an empty board's mean is every node; the pruned row keeps the nodes; a canvas row and an empty request are refused | `crates/mantis-player/src/batch.rs` | `batch::tests::every_derived_array_agrees_with_the_offsets_it_came_from`, `an_empty_board_takes_the_mean_over_every_node`, `the_pruned_row_builds_fewer_edges_on_the_same_nodes`, `a_canvas_row_and_an_empty_request_are_refused_by_name` |
| every carried kind and row exports, re-expresses bit for bit, and holds the fp32 lines at 1 and default threads | `tools/export_web.py` | `tests/tools/test_export_web.py::test_every_carried_kind_and_row_exports_and_holds_the_fp32_lines_at_one_and_default_threads` (gate 20) |
| no written graph holds a ScatterND, a GatherND or an int64 batch-sized index; the narrowing reaches through a Reshape; an unnarrowed graph is refused and removed (planted break) | `export_web.narrow_index_casts`, `graph_refusal` | `test_no_written_graph_holds_a_scatternd_a_gathernd_or_an_int64_batch_index`, `test_a_widened_index_is_refused_and_the_narrowing_takes_it_through_a_reshape`, `test_an_unnarrowed_graph_is_refused_and_leaves_no_file` |
| the fp16 file stores only what fp16 holds finite | `export_web.fp16_storage` | `test_the_fp16_file_stores_only_what_fp16_holds_finite` |
| the manifest binds the stamp, the files and the resolved block | `export_web.export` | `test_the_manifest_binds_the_stamp_the_files_and_the_resolved_block` |
| the witness reds a perturbed graph on the line each break moves, on every line that gates it: logits (a bias shift, gated in fp32 alone), probs (a weight scale), value (one bin); the argmax rule reds a flip the value lines cannot see (planted breaks) | `export_web.witness`, `gaps`, `passes` | `test_a_perturbed_written_graph_misses_the_line_it_moves`, `test_the_argmax_rule_reds_a_flip_the_value_lines_cannot_see` |
| a drifted re-expression, a foreign batch and an unchecked registry are refused before anything is written (planted breaks) | `export_web.export`, `witness_cases` | `test_a_re_expression_that_is_not_bit_identical_is_refused`, `test_a_foreign_batch_and_an_unchecked_registry_are_refused` |
| the player plays the Python deploy head's stones and counts its rows on one stub's outputs, at concurrency 1, over up to three turns a game with one engine on both sides (the forced-win position and the goldens), with reuse, cache hits and a decided stone each seen; a head that re-roots by a new game or stops at a tie is told apart (planted breaks) | `player.rs`, `cache.rs` | `tests/player/test_search_parity.py::test_the_player_plays_the_heads_stones_and_counts_its_rows`, `test_a_head_without_reuse_or_stopping_at_a_tie_is_told_apart`, `test_the_player_plays_the_heads_stones_on_every_golden` |
| each stone spends its `n` in new descents or stops with the leader fixed; a re-root keeps the chosen child's visits | `player.rs` | `crates/mantis-player/tests/served_sims.rs::each_stone_spends_its_budget_or_stops_with_the_leader_fixed_and_the_reroot_keeps_visits` |
| the player's constants are the ladder's switches and the head's cache size | `player.rs`, `cache.rs` | `test_the_players_constants_are_the_ladders_and_the_caches` |
| the cache serves hits and each unseen key once; FIFO at capacity, a hit refreshing nothing | `cache.rs` | `cache::tests::*` |
| the budget reproduces RESEARCH-WEB's guaranteed column, holds its floors and cap, rises with the clock; the pool is one doubling over the clock | `budget.rs` | `budget::tests::*`; `served_sims.rs::a_short_clock_plays_the_cpu_floor_and_flags_it` |
| the player refuses exactly the expands the bridge's `expand_and_backup_ls_graph` refuses (arity, stride, centre, trunk) | `player.rs` | `crates/mantis-bridge/src/player.rs::tests::the_player_refuses_exactly_the_expands_the_bridge_refuses` |
| a call out of step is refused and an abort restores the board | `player.rs` | `served_sims.rs::a_call_out_of_step_is_refused_and_an_abort_restores_the_board` |
| every game clock sizes a pool whose cap is at least the floor; a clock past the pool plays the cap and flags it; a turn that served leaves sets the costs, one that served none keeps them | `player.rs`, `budget.rs` | `served_sims.rs::every_game_clock_sizes_a_pool_of_at_least_the_floor`, `a_clock_past_the_pool_plays_the_cap_and_flags_it`, `a_turn_that_served_leaves_sets_the_costs_and_one_that_served_none_keeps_them` |
| a block or an output the head would refuse is refused (`leaf_batch_size` 0, a 0-turn solver, a non-finite value) | `player.rs` | `served_sims.rs::a_block_or_an_output_the_head_would_refuse_is_refused` |
| W2: `RustForward` holds the fp32 line against the fp32 ONNX on every tiny stamp and batch; fed every divisor as 1 it misses (planted break) | `forward.rs`, `tools/export_web.py` | `tests/tools/test_export_web.py::test_the_rust_forward_holds_the_fp32_lines_against_the_onnx_on_every_batch`, `test_a_rust_forward_without_the_degree_divisor_misses_the_line` |
| `RustForward` refuses a table its file does not hold, and a batch its shape does not allow | `forward.rs` | `forward::tests::*` |
| the bindgen attribute lives in `wasm.rs` alone, gated to wasm32 | `wasm.rs` | `crates/mantis-player/tests/wasm_census.rs` |
| W2 in wasm: the engine's `RustForward` holds the fp32 line against the fp32 ONNX on ORT's one-thread CPU kernels at B 1 and 8; every divisor fed as 1 misses it | `forward.rs`, `wasm.rs` | `web/test/w2.test.ts` (gate 21) |
| the Worker refuses a foreign manifest by name, admits the CPU backend by its witness, falls back to it when the WebGPU candidate throws, plays turns through the one loop, hears an abort between batches, stays usable after a failure mid-turn, passes a trap through untouched, refuses a second turn and a position no cadence reaches | `web/src/worker.ts` | `web/test/worker.test.ts` (gate 21) |
| the adapter declares the five capabilities, maps setup and requests to the engine, signs the heuristic x-positive, pads a winning first stone, answers nothing for an interrupt wherever it lands, serves the next position after a refused one, and refuses to play onto a stale board | `web/src/arena.ts` | `web/test/arena.test.ts` (gate 21) |
| a WebGPU batch split under the edge cap answers what the whole batch answers | `web/src/backends.ts` | `web/test/split.test.ts` (gate 21) |
| a trapping call rejects with `EngineCrashed` and the engine respawns; calls wait out a respawn; respawns are capped; a failed first `init` and `dispose` clean up | `web/src/index.ts` | `web/test/index.test.ts` (gate 21) |
| a witness fails short outputs and one that compared no argmax | `web/src/backends.ts` | `web/test/split.test.ts` (gate 21) |
| the detector refuses by name | `export_web.detect` | `test_the_detector_refuses_what_it_does_not_carry_by_name`, `test_a_gumbel_deploy_block_is_refused_before_anything_is_written` |

## Pinning tests

- `crates/mantis-player/src/batch.rs`, `cache.rs`, `budget.rs` (unit tests) and
  `crates/mantis-player/tests/served_sims.rs` (gate 2a)
- `tests/player/test_search_parity.py` (the default tier on the forced-win position and 32 goldens, the integration
  tier on it and 512, up to three turns each)
- `crates/mantis-player/src/forward.rs` (unit tests), `crates/mantis-player/tests/wasm_census.rs` and the bridge's
  `player::tests` error-parity test (gate 2a)
- `tests/tools/test_export_web.py` (gate 20, and the default tier)
- `web/test/*.test.ts` (gate 21: `tools/ci_gates/web_gate.sh` builds the engine, type-checks the package and runs them on
  a tiny stamp `tests/tools/_web_stamps.py` exports)
