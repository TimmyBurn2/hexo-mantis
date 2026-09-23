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

## Review
reviewer: fresh read-only agent (not the author); probes in throwaway worktrees, removed
| ID | verdict | lane | Δlines (probe-measured for lane A) | note |
|---|---|---|---|---|
| S-A-RUST-2-01 | CONFIRMED | C | -71 (scout, unprobed: lane C) | only pushes into `results` are the in-src test's |
| S-A-RUST-2-02 | CONFIRMED | C | -15 in slice (scout) | wire-or-delete is the architect's choice |
| S-A-RUST-2-03 | AMENDED | B | ≈ -63 in slice (was -56) | WeightBracket drags `brackets` + the weight_for loop; replay_persist.md O-31 row moves |
| S-A-RUST-2-04 | CONFIRMED | A | -7 (probe; scout -6) | the blank separator goes too |
| S-A-RUST-2-05 | CONFIRMED | A | -1 on the probed subset (the rest are 1:1 rewrites) | comment floor file moves; lib.rs line 2 is shared with 16 |
| S-A-RUST-2-06 | AMENDED | C | -6 (scout) | the witness claim is wrong: rotation_parity pins a COMMENT marker in the header being rewritten |
| S-A-RUST-2-07 | CONFIRMED | C | -6 (scout) | |
| S-A-RUST-2-08 | CONFIRMED | C | ≈ -30 (scout estimate) | search_drive.rs is source-text-pinned (see 06) |
| S-A-RUST-2-09 | CONFIRMED | C | -328 (scout) | "4 byte-identical" is loose wording; the class holds |
| S-A-RUST-2-10 | CONFIRMED | C | ≈ -116 (scout) | 3 of the 8 copies are named `splitmix64_step`; the set includes a golden and a floor-row bench |
| S-A-RUST-2-11 | CONFIRMED | C | -32 (scout) | |
| S-A-RUST-2-12 | CONFIRMED | C | -277 (scout) | ARCHITECT (HOT-14 re-open) stands |
| S-A-RUST-2-13 | REFUTED | C | 0 | a parity oracle is not ONE-SHOT |
| S-A-RUST-2-14 | CONFIRMED | C | -15 (scout) | |
| S-A-RUST-2-15 | CONFIRMED | B | ≈ -15 (scout estimate) | |
| S-A-RUST-2-16 | CONFIRMED | A | -19 (probe; scout -18) | trailing blank line; lib.rs line 2 counted here, not in 05 |
| S-A-RUST-2-17 | CONFIRMED | C | -15 (scout) | inv19_reanchor also calls both accessors |

### Per-finding notes
Probe (04 + 05 subset + 16, one worktree, independent edits): `sed` removed graph.rs::with_contract_version and folded `new()` into `with_contract_version_and_supply(1, 0)`; deleted lib.rs line 2 plus 8–25; rewrote finalize.rs lines 2–3/11/74/115 and atomics.rs "until WP7" -> `cargo check --workspace --all-targets --locked` rc 0 (0 errors); `cargo clippy -p mantis-selfplay --all-targets --locked -- -D clippy::all` rc 0; `cargo test -p mantis-selfplay --locked --lib --test queue_roundtrip --test queue_fuse_pin` -> 27 + 17 + 5 passed (lib 29 -> 27 = the two deleted tests); `r8_header_gate.py` green (graph.rs stays over the cap and keeps its header); `comment_lint.py` GREEN but notes rust_doc_excess 2676 -> 2673 and comment_excess 3385 -> 3384, "ratchet the floor down in this commit" (HEAD: GREEN with no notes). So tools/ci_gates/comment_length_floor.txt moves with this batch. `git diff --numstat`: lib.rs 0/19, graph.rs 1/8, finalize.rs 4/5, atomics.rs 1/1. No Python probe: neither subject has a pyo3 export (`git grep -w -E "CRATE_NAME|with_contract_version" crates/mantis-bridge` -> 0).
S-A-RUST-2-01 — CONFIRMED: `git grep -n -E "results\.lock|\.push_back\(" crates/mantis-selfplay/src/runner` -> production pushes go only to `gq`/`rg` in finalize.rs; `results` is pushed only at mod.rs's in-src test. Cross-crate grep for the four names over bridge/src/tools/tests -> 0. finalize.rs's header line 3 also names `WorkerResultRow` (05 rewrites it).
S-A-RUST-2-02 — CONFIRMED: `git grep -n -F positions_dropped -- src tools tests configs docs/contracts docs/governance/STATE.md` -> only `src/mantis/_engine.pyi:416`. pool_hooks.py's getattr list does not name it. Lane C because mod.rs and stats.rs are PZ.
S-A-RUST-2-03 — AMENDED (Δ, contract): `sed -n 1,50p replay/schedule.rs` shows `WeightSchedule.brackets: Vec<WeightBracket>`, which weight_for loops over. Deleting WeightBracket therefore also deletes that field, `brackets: Vec::new()` in `uniform()` and the 5-line loop (≈ -7 more). `grep -n "schedule.rs\|O-31" docs/contracts/replay_persist.md` -> the O-31 row names schedule.rs `#[cfg(test)]` as the weight-schedule pin. That is the lookup test being deleted, so the contract row must be re-pointed. Lane B stands. Setter callers re-derived: `git grep set_weight_schedule -- crates src tools tests` -> bridge hexg.rs, buffers.py facade, the 2 pyi, the facade test's fake only.
S-A-RUST-2-04 — CONFIRMED: `git grep -n -w with_contract_version -- . ':!docs/slim'` -> graph.rs:136 and :142 only. graph_wire.md cites other graph.rs symbols, not this one. Probe above.
S-A-RUST-2-05 — CONFIRMED: `grep -n -i -E "dense|grid|ReplayBuffer|HEXB|WP7|WP0|seeded|rotation|is_graph|in-src"` over the 7 files reproduces every listed site. game.rs has no seed or rotation code (`grep -i "seed\|rotation\|sym" runner/game.rs` -> only the doc lines 384/454). None of the 7 files is in the PZ list or named by a ruling or card (`git grep -c -F` over RULINGS/CARDS/STATE/LAWS -> 0). Probe above covers the deletion part.
S-A-RUST-2-06 — AMENDED (witness): `sed -n 165,195p tests/rotation_parity.rs` -> `graph_build_call_passes_no_sym_idx` asserts `SEARCH.contains("rotation-free")` on RAW source ("Doc-marker, matched on RAW source"). The only lowercase hit is search_drive.rs header line 6 (`grep -n rotation-free`), right beside the header lines 4–5 this finding rewrites. A header edit must keep that token, or the test reds. The orphaned dense docstring is confirmed (`sed -n 254,268p`).
S-A-RUST-2-07 — CONFIRMED: `git grep -A4 "pub enum Representation" crates/mantis-encoding` -> `Graph,` only. The branch is hexg/mod.rs:264–269.
S-A-RUST-2-08 — CONFIRMED: `git grep -w game_start_ply crates` -> one writer, `0` (game.rs:358). `MovePolicy::` -> `Ls` only (7 sites). `visit_capacity` -> the `Some(` at mod.rs:261 and the `.expect(` at search_drive.rs:681. `_move_history` and `let legal = full_legal;` are both in select_move. Any edit to search_drive.rs must keep rotation_parity's `call_args(build_leaf_graph)` and its "rotation-free" marker.
S-A-RUST-2-09 — CONFIRMED: my brace-matched extractor normalised the fn name, `pop_graph_batch(N,M)` and whitespace, then hashed. Result: one class of 4 (dirichlet, gumbel_round_batching, move_arms, search_stats) and pcr = zero_visit. Against pcr: ply_cap and served_sims differ by 4 lines (name and batch constant); target_support and dirichlet by 8 (signature formatting); target_wire_carry by 10 (a `let res` restructure); target_latch by 20 (formatting). Raw byte-identity modulo name holds for 2 pairs, not 4. The copies sum to 388 lines. served_sims_exact.rs is a PZ witness file.
S-A-RUST-2-10 — CONFIRMED: the same extractor over `splitmix64,splitmix64_step` -> 8 bodies in one hash class. Three are named `splitmix64_step`: drain_shutdown, worker_output_pin, graph_build_bench. worker_output_pin.rs is a PZ golden ("pinned splitmix64 inputs"). graph_build_bench is the only R2 bench with a tools/bench_floors.toml row (`[floor.graph_build_bench_graph_build_gnn_axis_v1_leafcorpus]`), so moving its helper touches a LAW-09 floor bench's source. Lane C.
S-A-RUST-2-11 — CONFIRMED: I printed all four bodies. The rotate test is 5 coords × 12 syms against rotation_parity's 13×13 × 12, with the same assertion. The grid test and audit1's unknown-encoding test make the same `HexgBuffer::new` refusal and the same `contains(name) && contains("gnn_axis_v1")`. `grep -c -w inv_sym replay_hexg.rs` = 2 (def + this test), so -7 holds. Gate 3c is pytest-only (test_count_gate.sh collects via pytest), so no floor moves.
S-A-RUST-2-12 — CONFIRMED: `grep -i prefuse RULINGS.md CARDS.md` -> R336 "S-PREFUSE REFUTED with HOT-14 re-opened" and the CARDS HOT-14 row. `grep -l "mod common"` -> prefuse_concat_parity.rs and queue_fuse_bench.rs only. `grep queue_fuse tools/bench_floors.toml` -> 0, so no floor move.
S-A-RUST-2-13 — REFUTED (class): the file header reads "The reference is a LOAD-BEARING transcription … the only remaining copy of the arithmetic … must not be simplified". It is a differential parity oracle with 3 tests: batch shapes, degenerate arrays and the dst-walk. queue_fuse_pin covers 3 frozen graphs plus the single-graph case. REVIEW_BRIEF: a parity oracle is not ONE-SHOT. What is left is an ARCHITECT question: does this oracle retire now that PERF-TRANCHE-1 is ratified (R320)?
S-A-RUST-2-14 — CONFIRMED: `git grep -w store_fatal_defect crates src tests` -> the mod.rs:326 def plus target_integrity_postfix.rs (PZ-3 frozen oracle bank) only.
S-A-RUST-2-15 — CONFIRMED: `grep -n "max(1).min\|div_ceil\|thread::scope"` -> graph.rs:478/482/484 and sample.rs:289/296/298, the same skeleton. sym.rs's `include_str!("hexg/sample.rs")` pins only `self.rng.random_range(0..N_SYMS)`, which the merge does not touch.
S-A-RUST-2-16 — CONFIRMED: `git grep -n -E "CRATE_NAME|dag_deps_compile|crate_name_pinned" -- . ':!docs/slim'` -> only the five crates' lib.rs. mantis-selfplay's `CRATE_NAME` has no reader outside its own lib.rs. Every dependency edge is used by real code (`git grep -l -w mantis_{core,encoding,graph,search}` over selfplay src/tests/benches -> 15/25/17/29 files). There is no `unused_crate_dependencies` or machete gate. Probe above.
S-A-RUST-2-17 — CONFIRMED: `git grep -E "\.(feature_len|policy_len)\(\)" crates` -> inv19_reanchor.rs:146–149 and inv23_reanchor.rs only, so inv19 needs the same rewrite. `is_available` -> wire.rs plus queue_fuse_pin.rs:267/269. No Python `.is_available` hit on a GraphWire (all hits are torch).
DEFECT LAW-18 (four counters) — CONFIRMED real: `git grep -F` for each of `pcr_full_moves pcr_quick_moves gumbel_round_leaves gumbel_rounds` over crates/mantis-bridge, src, tools, tests -> 0. `git log -S` over bridge/src/tests -> never exposed in visible history. Both levers are ARMED in the live run: STATE.md:121 `selfplay.search.kind: gumbel` and configs/run10.yaml `full_search_prob: 0.25`. Their fire-rates are therefore unobservable in-run, which is the defect LAW-18 names. The other two DEFECTS rows also reproduce (the `non_snake_case` warnings appear in my cargo check; finalize.rs has 0 `cfg(test)`).

### Missed by the scout
- NEW-1 | DEAD | HANDOFF R3: crates/mantis-bridge/src/inference.rs::PyInferenceBatcher.feature_len. At HEAD, rustc warns "field `feature_len` is never read" in `cargo check --workspace --all-targets` (the bridge was untouched by the probe).
- NEW-2 | DEAD | B/C, HANDOFF R3: PyInferenceBatcher.spawn_mock_graph_games. `git grep -w spawn_mock_graph_games -- src tools tests` -> only _engine.pyi, and its only caller is an in-src Rust test (inference.rs:958). It is the only non-test caller of GraphQueue::submit_graph_and_wait, which graph_wire.md cites as the consumer-side handshake, so the pair is contract-cited.
- NEW-3 | DOC | C: runner/mod.rs::RunnerStatsSnapshot.gumbel_round_leaves doc: "whose reader publishes the ABSENCE rather than a 0/0" names a reader that does not exist (the LAW-18 defect).

### Tally: raised 17 | confirmed 14 | amended 2 (03, 06) | refuted 1 (13) | pending 0 | architect 0 (12 and 13 carry residual architect questions)
