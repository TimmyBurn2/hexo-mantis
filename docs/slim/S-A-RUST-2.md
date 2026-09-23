# S-A-RUST-2 — R2 crates/mantis-selfplay
scope: `git ls-files crates/mantis-selfplay` (63 files, 15 115 lines); method: `git grep -n -w` over crates/ src/ tools/ tests/ docs/ per pub item, `mantis_selfplay::` import census, `cargo check -p mantis-selfplay --all-targets --locked` (shared target dir; rc 0, 2 warnings), awk span extraction + md5 over test helpers, `r8_header_gate.py` (green, 13 R2 files over cap, all justified).

LAW-11 question ("is dense still live in production?"): NO. `mantis_encoding::Representation` has one variant (`Graph`), the dense batcher and the HEXB ring are gone (queues/mod.rs, replay/mod.rs say so). What is left of dense is residue: a dead training-row channel (01), an unreachable branch (07), an always-`Some` Option plus a one-variant enum (08), and stale comments (05/06). The carded `push_dense` (B-15) is Python and belongs to C2 (HANDOFF).

## Summary
- 17 findings. By class: DEAD 6 (01 02 03 07 16 17), DUP 4 (09 10 14 15), SIMPLIFY 2 (04 08), DOC 2 (05 06), ONE-SHOT 2 (12 13), TEST 1 (11).
- By lane: A 3 (04 05 16), B 2 (03 15), C 12. Most are C because PZ's glob covers `crates/mantis-selfplay/tests/**`, runner/{mod,search_drive,stats}.rs, records.rs and replay/hexg/mod.rs.
- Top 3 by Δ: S-A-RUST-2-09 (-328), S-A-RUST-2-13 (-285), S-A-RUST-2-12 (-277).
- Rust test deletions do not move gate 3c. That floor counts pytest collection only.

## Findings

### S-A-RUST-2-01 | DEAD | C
subject: crates/mantis-selfplay/src/runner/mod.rs::WorkerResultRow, ::SelfPlayRunner.results, ::SelfPlayRunner::drain_training_rows; runner/params.rs::WorkerChannels.results_queue; runner/spawn.rs (`results_queue: self.results.clone()`); runner/game.rs::run_one_game `_results_queue` param + its destructure/arg; the in-src test mod.rs::seam_roundtrip::drain_training_rows_returns_pushed_rows_then_empties
claim: the dense per-row training queue has no producer and no consumer. The only worker-side handle is the underscore-named `_results_queue` param, which is never read. The drain is called only by its own in-src round-trip test.
evidence: `git grep -n -E "results_queue|self\.results\b|WorkerResultRow" -- crates/mantis-selfplay/src` -> field, ctor, spawn clone, game.rs destructure/arg, `_results_queue: &Mutex<..>` param, drain, test. No `push_back` on it outside the test.
callers:
- AST/cross-crate: `git grep -n -w -E "drain_training_rows|WorkerResultRow|_results_queue" -- crates/mantis-bridge src tools tests` -> 0.
- pyo3: crates/mantis-bridge/src/runner.rs has no `drain_training_rows` wrapper (same grep) -> nothing reaches `_engine`.
- entry points/python -m/subprocess/importlib/conftest: n/a (Rust-internal type; `mantis_selfplay::` import census lists no `WorkerResultRow`).
- config keys: `results_queue_cap` STAYS. It is live on the GRAPH queue in finalize.rs::finalize_game_graph (`gq.len() > results_queue_cap`).
- gates/STATE/docs: `git grep -n -E "drain_training_rows|WorkerResultRow" -- . ':!docs/*/archive'` -> crates only. The "P-04 pin destructures this carrier" doc names a pin that does not exist (tests/fixtures/manifest.toml mentions P-04 only for worker/*.bin).
- include_str/[[bench]]/cfg(test): only the in-src cfg(test) test deleted with it.
Δlines: -71. Derived from awk spans: type+doc 13, drain+doc 5, in-src test 46, plus 7 single plumbing lines (mod.rs field+ctor, params.rs field, spawn.rs, game.rs destructure/arg/param).
witness: none reds except the deleted test. `cargo check --all-targets` proves the plumbing gone.
depends: 08

### S-A-RUST-2-02 | DEAD | C
subject: the `positions_dropped` counter chain. runner/finalize.rs::finalize_game_graph (`positions_dropped.fetch_add`), runner/stats.rs::WorkerStats.positions_dropped, runner/mod.rs::RunnerStatsSnapshot.positions_dropped and field/ctor/snapshot/test lines, runner/game.rs plumbing
claim: the graph-queue drop-oldest counter is produced but no run reads it. Its only reader is the bridge getter `SelfPlayRunner.positions_dropped`, which has zero Python references. Choose: wire it into the event stream (LAW-18) or delete it. The drop itself stays either way.
evidence: `git grep -n -w positions_dropped -- src tools tests` -> only the two `_engine.pyi` stubs. src/mantis/selfplay/pool_hooks.py's 15 getattr counters (CALLERS §7) do not include it.
callers:
- Rust: `git grep -n -w positions_dropped -- crates` -> selfplay src (15 lines) + crates/mantis-bridge/src/runner.rs getter.
- pyo3 from Python: 0 (above). getattr-by-string: pool_hooks list checked, absent.
- dashboard event names (CALLERS §7): absent. STATE: `git grep positions_dropped docs/governance/STATE.md` -> 0.
- config: no key.
Δlines: -15 in-slice (`git grep -n -w positions_dropped -- crates/mantis-selfplay | wc -l` = 15; each is a one-line site) + 3 bridge + 2 pyi (R3/HANDOFF).
witness: mod.rs::seam_roundtrip::stats_snapshot_reads_back_each_private_atomic (edits with it).
depends: —

### S-A-RUST-2-03 | DEAD | B
subject: crates/mantis-selfplay/src/replay/hexg/storage.rs::HexgBuffer::set_weight_schedule_impl; replay/schedule.rs::WeightBracket and its test `test_weight_schedule_lookup`; plus out of slice: crates/mantis-bridge/src/hexg.rs::set_weight_schedule, src/mantis/selfplay/buffers.py::set_weight_schedule, both `_engine.pyi`
claim: no production path ever sets a game-length weight schedule. Every slot is therefore weighted by `WeightSchedule::uniform()` (1.0), and the setter chain is dead. The per-slot `weights` field and `weight_for` stay, because weights are part of the HEXG v2 on-disk record (golden `replay/hexg_v2_golden.hexg`).
evidence: `git grep -n -E "\.set_weight_schedule\(" -- src tools | grep -v self.raw` -> empty. `git grep -n "set_weight_schedule" -- tests` -> only tests/selfplay/test_buffer_facade.py, whose fake raw buffer records the call.
callers:
- AST: buffers.py facade defines it; no caller in src/ or tools/ (above).
- pyo3: bridge `HexgBuffer.set_weight_schedule` is reached only through that facade.
- config keys: `git grep -n -i weight_schedule -- src/mantis/config` -> only `policy_loss_weight_schedule` (unrelated loss warm-up).
- STATE/docs: docs/contracts/replay_persist.md cites `replay/schedule.rs` (O-31). The file stays, so gate 10 is unaffected.
- tests: schedule.rs in-src O-31 lookup test (deleted with WeightBracket); the uniform test stays.
Δlines: -56 in slice (awk spans: set_weight_schedule_impl 24, WeightBracket 9, lookup test 23) + bridge/facade/pyi/facade-test rows out of slice.
witness: tests/selfplay/test_buffer_facade.py (its set_weight_schedule row must go with the facade method).
depends: —

### S-A-RUST-2-04 | SIMPLIFY | A
subject: crates/mantis-selfplay/src/queues/graph.rs::GraphQueue::with_contract_version
claim: this is a pure forwarder whose only caller is `GraphQueue::new()` in the same file. Fold it into `new()`, which calls `with_contract_version_and_supply(1, 0)`.
evidence: `git grep -n -w with_contract_version -- .` -> graph.rs:136 (`new`) and graph.rs:142 (def) only.
callers: all other channels are empty for a Rust-internal method (no pyo3 name, no string lookup). The bridge uses `with_contract_version_and_supply` (inference.rs:299).
Δlines: -6 (2 doc + `#[must_use]` + 3-line fn; `new()` body changes 1:1).
witness: tests/queue_roundtrip.rs (16 `GraphQueue::new()` sites) keeps the contract-version-1 behaviour.
depends: —

### S-A-RUST-2-05 | DOC | A
subject: stale dense/grid/WP-era comments in non-PZ R2 files:
- queues/graph.rs: header "DISJOINT from the dense queue … dense pool"; `pop_graph_batch_blocking` "Graph counterpart of the dense pop"; `close` "Disjoint from the dense queue".
- runner/params.rs: `resolve_geometry` "`is_graph` / `legal_set` are set from the arms"; WorkerChannels "dense/graph inference queues are now the two disjoint".
- runner/game.rs: "hoisted `is_graph` finalize branch" twice; PerGameInit "`Vec::new()` … for grid games"; "skipped entirely for a seeded game"; `init_per_game_board` "dry-replays an optional seed prefix, samples per-game rotation", which the body does not do.
- runner/finalize.rs: header says "The in-src ply-cap unit test exists because…" but there is none (`grep -c "cfg(test)" finalize.rs` = 0); "WP7"; "dense drain"; "dense backpressure drop".
- runner/atomics.rs: "until WP7 wires the real setter". The setter is wired: bridge `set_model_version`.
- replay/hexg/storage.rs: three "ReplayBuffer/HEXB" parity lines.
- lib.rs: "WP0 scaffold: compiles empty".
claim: each comment describes an arm, test or port stage that does not exist at HEAD. R316(e) applies on contact.
evidence: `grep -n -i -E "dense|grid|ReplayBuffer|HEXB|WP7|WP0|seeded|rotation" crates/mantis-selfplay/src -r` (hits listed above).
Δlines: 25 lines touched. Net ≤ 0: most are one-line rewrites, and the finalize.rs false test claim is 2 lines deleted outright.
witness: gate 14 comment ratchet (rust_doc_excess may only fall; deletion lowers it).
depends: —

### S-A-RUST-2-06 | DOC | C
subject: the same stale class in PZ-glob files:
- runner/mod.rs: `WorkerResultRow` "P-04 pin" (goes with 01); `visit_capacity` "`None` on grid runs — dense-362 records".
- runner/search_drive.rs: header target order "forced-win one-hot -> solver soft-inject" (stats.rs says those hooks are deleted); a 6-line dense-arm docstring ("submits to the dense inference queue, forward/inverse-scatters under the per-game symmetry…") fused above `infer_and_expand_graph`'s own doc; `select_move` "ZOI-filters when enabled"; "`Some` iff this is a graph run".
- records.rs: `#[allow]` note "mirrors the dense record fns'"; "dense scatter-max path" preface.
- replay/hexg/mod.rs: "PARALLEL ring beside the dense `ReplayBuffer`"; "mirror `ReplayBuffer::weight_bucket`"; "parallel to `ReplayBuffer`"; the refusal text "(use ReplayBuffer for dense encodings)" (unreachable, see 07).
claim: same as 05, but in protected files.
evidence: same grep. `sed -n 256,262p search_drive.rs` shows the orphaned dense docstring.
Δlines: -6 for the orphaned docstring (lines 256–261). The remaining ~10 lines are rewrites (≤ 0).
witness: gate 14 ratchet; rotation_parity.rs reads search_drive.rs source text (`include_str!`) but pins only the builder call, not comments.
depends: 07

### S-A-RUST-2-07 | DEAD | C
subject: crates/mantis-selfplay/src/replay/hexg/mod.rs::HexgBuffer::new, the `if !spec.is_graph() { return Err(… "use ReplayBuffer for dense encodings") }` branch
claim: this branch is unreachable. `Representation` has the single variant `Graph`, and the registry parse refuses `"grid"` by name, so every resolvable spec is graph.
evidence: `git grep -n -A3 "enum Representation" crates/mantis-encoding` -> `Graph,` only. `RegistrySpec::is_graph` is `matches!(self, Representation::Graph)`.
callers: n/a (a branch, not a symbol). tests/replay_hexg.rs::grid_encoding_rejected_at_construction reaches the registry `lookup` refusal, not this branch (its own comment says so).
Δlines: -6 (the if-block, lines 264–269).
witness: NONE (unreachable). PZ contact is file-level: MAX_STONES (R329/R331(ii)) lives in this file.
depends: —

### S-A-RUST-2-08 | SIMPLIFY | C
subject: single-arm residue in runner/:
- `visit_capacity: Option<usize>`: always `Some(..)` in mod.rs::SelfPlayRunner::new; threaded through params.rs::WorkerParams, game.rs::WorkerMoveCfg and search_drive.rs::MovePlayContext; unwrapped by a 4-line `.expect("… graph arm …")` in search_drive.rs::play_one_move.
- search_drive.rs::MovePolicy: one-variant enum `Ls` plus its `impl`, destructured in record.rs.
- `MovePlayContext.game_start_ply`: only writer is game.rs `game_start_ply: 0`, and the test `explore_gate_tests::the_span_is_relative_to_the_games_start` (11 lines) covers the deleted seeded deep-prefix start.
- search_drive.rs::select_move: unused `_move_history` param and a `let legal = full_legal;` alias.
claim: each is a two-arm shape whose second arm (grid, seeded corpus) was deleted at R346(f). The value is now constant or the wrapper carries one case.
evidence: `git grep -n "game_start_ply" -- crates` -> one writer (0). `git grep -n "visit_capacity" -- crates/mantis-selfplay/src/runner` -> Some-wrap + expect. `grep -n "MovePolicy::" runner/*.rs` -> only `Ls`.
Δlines: ≈ -30 (explore test 11; expect block 4; MovePolicy decl/impl match arms ≈ 8; game_start_ply field/doc/arg 4; `_move_history` param + arg 2; alias 1).
witness: tests/served_sims_exact.rs, gumbel_round_batching.rs, pcr_arm_matches_the_record.rs (end-to-end drives); explore_gate_tests' two remaining tests.
depends: 01

### S-A-RUST-2-09 | DUP | C
subject: the healthy mock graph producer (`spawn_producer` / `spawn_healthy_graph_producer` / `spawn_counting_producer` / `spawn_graph_producer`) in 11 test files: dirichlet_inert_on_gumbel, game_result_carries_move_arms, game_result_carries_search_stats, gumbel_round_batching, target_support_is_sims_bounded, ply_cap_game_coverage, zero_visit_export_pin, target_wire_carry, pcr_arm_matches_the_record, served_sims_exact, target_latch_propagation
claim: 4 copies are byte-identical (md5 6da4d6ef). 6 differ only in the `pop_graph_batch(N, 5)` batch constant or the fn name. target_latch_propagation differs only in formatting. One shared `tests/common` helper taking `pop_max` would replace all 11. tests/common/mod.rs already exists with `#![allow(dead_code)]`.
evidence: awk extraction + md5 in scratchpad (`prod_*.rs`); `diff` of each pair shows only the name or batch-constant lines.
deliberate?: not seam, oracle or twin. It is scaffolding that feeds uniform probs through the production `assemble_ls_from_gnn_probs`. No fixture is bound to it, and each file's subject is the runner, not the producer. The 3 real variants stay local (search_seam_fatal's failure injection, drain_shutdown's seeded mock-NN).
Δlines: -328 = -(34×5 + 38×5 + 28 = 388 derived by `wc -l prod_*.rs`) + 38 shared helper + 22 (`#[path]`/`mod common;` 2 lines × 11).
witness: the 11 files themselves (a behaviour change in the helper reds them all).
depends: 12 (tests/common/mod.rs content)

### S-A-RUST-2-10 | DUP | C
subject: other copied test helpers:
- `splitmix64`: 8 byte-identical 7-line bodies (tests/common/mod.rs, drain_shutdown, hexg_sample_parallel_parity, leaf_graph_parallel_parity, queue_fuse_reserve_parity, worker_output_pin, benches/queue_fuse_bench, benches/graph_build_bench).
- Fixture parsers `fixture_text/value_of/ints/floats/scalar/pairs`, duplicated between graph_child_parity.rs and target_export_parity.rs (60 lines; 4 identical, `ints`/`scalar` differ only in panic text).
- `inv_sym` in replay_hexg.rs and rotation_parity.rs (same function, branches ordered differently).
- `wide_board/ls_on_first_legal/record` across target_integrity_postfix.rs, target_sign_integrity.rs and zero_visit_export_pin.rs (`wide_board` identical in all three).
claim: local duplication of test helpers that tests/common could hold once.
evidence: md5 of splitmix bodies (all 0962311e); `diff` of parser fns (0/2 changed lines); awk spans.
deliberate?: the parser pair reads the same fixture dir (tests/fixtures/eval_selfplay_parity), so it is not a twin by design. The postfix file is the byte-frozen oracle bank (PZ-3). Moving its helpers out is a frozen-file touch, so leave that row to the ruling.
Δlines: ≈ -116 (splitmix -49 + 7 kept; parsers -60; inv_sym -7; `mod common` lines not netted).
witness: the host tests (the goldens graph_child_parity / target_export_parity red on any parser drift).
depends: 09

### S-A-RUST-2-11 | TEST | C
subject: tests/replay_hexg.rs::rotate_axial_roundtrips_under_inverse (+ its local `inv_sym`); tests/replay_hexg.rs::grid_encoding_rejected_at_construction
claim:
- The rotate test is a strict subset of rotation_parity.rs::rotate_axial_forward_inverse_is_identity: 5 coords × 12 syms, against the full 13×13 grid × 12.
- The grid test makes the same assertion as audit1_named_errors.rs::an_unknown_encoding_is_an_ERR_naming_the_registered_set: an unregistered name passed to `HexgBuffer::new`, and a message naming it and `gnn_axis_v1`. "v6" is now just unknown.
evidence: awk bodies of the four tests (same call, same asserts).
Δlines: -32 (rotate 14 + inv_sym 7 + grid 11; `#[test]` lines not netted).
witness: rotation_parity.rs and audit1_named_errors.rs keep the assertions. No pytest floor moves (cargo tests are not in gate 3c).
depends: 10

### S-A-RUST-2-12 | ONE-SHOT | C
subject: the S-PREFUSE harness: tests/prefuse_concat_parity.rs; tests/common/mod.rs::{corpus, concat_by_offset, fuse}; benches/queue_fuse_bench.rs::queue_concat_cross_thread_pop40 + its `#[path] mod common`
claim: this proves and times a worker-side pre-fuse that was never adopted. R336 records "S-PREFUSE REFUTED with HOT-14 re-opened". `concat_by_offset` has no production caller.
evidence: `git grep -n -w concat_by_offset -- .` -> common/mod.rs, prefuse_concat_parity.rs, queue_fuse_bench.rs only. `git grep -n -i prefuse -- docs/governance/RULINGS.md docs/governance/CARDS.md` -> R336 "REFUTED", CARDS HOT-14 row.
deliberate?: ruling-named (R336, HOT-14 re-open cites it), hence lane C. If HOT-14 re-litigates a pre-fuse, the harness is its starting point.
Δlines: -277 (prefuse_concat_parity 91 + common/mod.rs 143 + bench arm 39 by awk span + 4 `#[path]` lines). Nets to -134 if 09 reuses tests/common.
witness: NONE for production. `cargo clippy --all-targets` (gate 2b) must stay green after the bench group edit.
depends: —

### S-A-RUST-2-13 | ONE-SHOT | C
subject: crates/mantis-selfplay/tests/queue_fuse_reserve_parity.rs (whole file)
claim: this is a perf-rewrite proof (PERF-TRANCHE-1 A1). It keeps a transcribed copy of the pre-A1 fuse (`fuse_reference`) as a second implementation, to show the shipped `GraphWire::from_axis_graphs` is byte-identical to it. The fuse arithmetic is independently golden-pinned by queue_fuse_pin.rs (P-09, fixture worker/graphwire_multigraph_input.bin).
evidence: file header ("the pre-A1 fuse is transcribed here verbatim as `fuse_reference`"); `wc -l` 285.
deliberate?: partly unique coverage. Its synthetic corpus hits degenerate arrays and the dst-walk offset, which the 3-graph golden may not. The decision is whether that coverage is re-homed into queue_fuse_pin or the file is kept as a standing oracle.
Δlines: -285 (`wc -l`), less any re-homed degenerate-array case.
witness: queue_fuse_pin.rs (golden) remains.
depends: —

### S-A-RUST-2-14 | DUP | C
subject: crates/mantis-selfplay/src/runner/mod.rs::SelfPlayRunner::store_fatal_defect vs runner/search_drive.rs::FatalDefectLatch::store / ::store_counted
claim: the same write-once-slot, count, then `running=false` sequence is implemented twice. The production path uses the latch. The runner method's only caller is the frozen oracle bank.
evidence: `git grep -n store_fatal_defect -- .` -> def + tests/target_integrity_postfix.rs (o4b) only.
deliberate?: target_integrity_postfix.rs is PZ-3 frozen ("byte-frozen post-fix oracle bank") and names `store_fatal_defect(String)` in its gate contract. Re-pointing it is a frozen-file touch.
Δlines: -15 (awk span of the method + doc).
witness: target_integrity_postfix.rs::o4b_latch_stores_the_named_variant_and_halts_the_runner; target_latch_propagation.rs (the production latch).
depends: —

### S-A-RUST-2-15 | DUP | B
subject: crates/mantis-selfplay/src/queues/graph.rs::build_leaf_graphs_batch vs crates/mantis-selfplay/src/replay/hexg/sample.rs::build_and_align_batch
claim: both write the same in-index-order chunked `std::thread::scope` map: `threads = n.max(1).min(len)`, the serial path at 1, `chunks(div_ceil)`, join→named error, then flatten with `?`. One generic helper would serve both.
evidence: side-by-side read (graph.rs lines ~472–499, sample.rs ~287–322).
deliberate?: not seam or twin. Both headers give the same reason ("rayon is absent"). The bodies differ only in the item type and the error string.
Δlines: net ≈ -15. Spans 36 + 43 by awk; the shared thread-scope bodies are ~22 lines each, replaced by one ~25-line generic plus two ~3-line calls.
witness: tests/leaf_graph_parallel_parity.rs, tests/hexg_sample_parallel_parity.rs (bit-identity at every width).
depends: —

### S-A-RUST-2-16 | DEAD | A
subject: crates/mantis-selfplay/src/lib.rs::CRATE_NAME, the cfg(test) `crate_name_pinned` and `dag_deps_compile`, and the "WP0 scaffold" doc line
claim: this is WP0 scaffolding. CRATE_NAME has no reader except these two tests, which assert constants equal their own literals. The dependency edges they claim to prove are already proven by the crate compiling.
evidence: `git grep -n CRATE_NAME -- .` -> only each crate's own lib.rs. `git grep -n -E "CRATE_NAME|dag_deps_compile" -- tests tools docs Makefile` -> 0.
callers: pyo3 none (not re-exported by the bridge); include_str/[[bench]] none; string lookups none (grep above).
Δlines: -18 (lib.rs line 2 + lines 9–25).
witness: NONE.
depends: —

### S-A-RUST-2-17 | DEAD | C
subject: pub items with test-only readers: runner/mod.rs::SelfPlayRunner::feature_len / ::policy_len; queues/wire.rs::GraphWire::is_available
claim: no production code (bridge, Python) reads these. feature_len/policy_len exist only for inv19/inv23. The bridge derives its own strides from the spec. `is_available` exists only for queue_fuse_pin's single-read test, which `take()`'s `Err(WireAlreadyConsumed)` already pins.
evidence: `git grep -n -E "\.(feature_len|policy_len)\(\)" -- crates` -> inv19_reanchor.rs, inv23_reanchor.rs only. `git grep -n -w is_available -- crates` -> wire.rs def + queue_fuse_pin.rs:267/269.
callers: pyo3: `feature_len`/`policy_len` bridge hits are InferenceBatcher's own fields, not this method (`git grep -n "inner\.feature_len" crates/mantis-bridge` -> 0). Python: n/a.
Δlines: -15 (two 5-line accessors + the 5-line `is_available`). inv23's shape assertions lose their subject: rewrite them against `spec` directly or drop them.
witness: inv23_reanchor.rs::every_registered_encoding_name_resolves… (rewrite), queue_fuse_pin.rs::take_is_single_read.
depends: —

## DEFECTS
- The LAW-18 fire-rate counters `pcr_full_moves`, `pcr_quick_moves`, `gumbel_round_leaves` and `gumbel_rounds` are on `RunnerStatsSnapshot` but have no bridge or Python reader (`git grep -w` -> bridge 0, src/tools/tests 0). They are observable only in cargo tests, never in-run. `dirichlet_root_fires` is declared "not a bridge field" by R359(d); the other four carry no such note.
- tests/audit1_named_errors.rs: two non-snake-case test fns (`…_at_BOOT`, `…_ERR_…`) emit rustc `non_snake_case` warnings on every build (`cargo check --all-targets`).
- runner/finalize.rs header asserts an in-src ply-cap unit test that does not exist (0 `cfg(test)` in the file). Coverage claimed, not present (also listed in 05).

## PARKED
- queues/graph.rs::GraphQueue::submit_graph_results / ::fail_remaining take the waiter-map lock once per id rather than once per batch.

## HANDOFF
- R3 (bridge): `SelfPlayRunner.positions_dropped` getter has zero Python refs (02). `HexgBuffer.set_weight_schedule` has no production caller (03). `get_buffer_stats` has only tests/data readers. The same CRATE_NAME/`dag_deps_compile` scaffold exists in mantis-core/-encoding/-graph (16). R1 (mantis-search) has it too.
- C2: `src/mantis/selfplay/pool_push.py::push_dense` is CARDED (CARD-STYLE-BACKLOG B-15): dead in production, and the golden drain-parity suite uses it as its oracle. It is not in this slice. `buffers.py::set_weight_schedule` facade goes with 03.
- T5: tests/selfplay/test_buffer_facade.py's `set_weight_schedule` row (03).
- C3: src/mantis/diagnostics/ring_reader.py re-declares HEXG_MAGIC/HEXG_VERSION in Python. Deliberate if the reader must run engine-free; the owner should say.

## Not covered
- records.rs body (frozen 1:1 port, PZ-3) beyond its pub-item census; replay/hexg/{persist,push}.rs internals; the in-src tests of records.rs, sym.rs and atomic.rs.
- tests/replay_hexg.rs (1 021 lines) beyond the helper and duplicate scan; per-assertion uniqueness across all 36 test files beyond the pairs named in 11.
- benches/graph_build_bench.rs beyond confirming its one floor row in tools/bench_floors.toml.
- No `cargo test` probe was run (CPU budget). Only `cargo check --all-targets` (green, 2 warnings).
- `GraphQueue::submit_graph_and_wait` (single-graph path): live only through the bridge's `spawn_mock_graph_games` test hook. Not classified.
