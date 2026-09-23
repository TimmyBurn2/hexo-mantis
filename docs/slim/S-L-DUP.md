# S-L-DUP — LENS: duplication (whole tree)

scope: `.py` and `.rs` under src/, tools/, tests/, crates/ (tests/fixtures/** excluded) at `69e1532`.
method (all throwaway, outputs under scratchpad/ldup/):
- `npx --yes jscpd@4 --min-lines 8 --min-tokens 60 --reporters json --format python,rust --ignore "**/.venv/**,**/target/**,**/tests/fixtures/**" src tools tests crates`
  → 311 clone pairs, 4 623 duplicated lines of 168 494 (2.74 %); python 230 pairs / 3 415 lines, rust 81 / 1 208.
- a finer jscpd pass over non-test code only (`--min-lines 6 --min-tokens 50`, tests/ and benches/ ignored) → 46 pairs.
- scratchpad/ldup/asthash.py: Python function bodies, docstrings stripped, every Name/arg renamed positionally,
  `ast.dump` hashed, ≥ 6 statements → 27 groups (≥ 4 statements: no new src/tools group).
- scratchpad/ldup/tophash.py: top-level def/class, docstrings stripped, names kept, ≥ 4 lines → 107 groups,
  2 058 lines beyond the first copy (tests dominate; the src hits are exception declarations, see K-D20).
- scratchpad/ldup/samename.py (same-name helpers across a file set, difflib ratio), rsfn.py (Rust fn bodies,
  comments/whitespace normalised), sinks.py and fakes.py (spy-sink and `_Pool`/`_Buffer` censuses).
- Δlines: scratchpad/ldup/delta.py — removes every listed copy (AST span incl. decorators; Rust span incl. `///`
  lines), keeps ONE copy per symbol (the largest), adds one import line per touched file plus the stated header
  lines. Blank separators are not counted, so every Δ is conservative. Δs are per finding and are not additive
  where findings share files (their import lines would merge).
- Protected zone: scratchpad/map/PZ.md (glob list). Any PZ member forces lane C; lane-B findings list their PZ
  members as excluded and those members are gathered in S-L-DUP-29.

## Summary

- Clusters examined: 64 (table below). ACCIDENTAL raised: 29 findings. DELIBERATE: 24. ACCIDENTAL not raised
  (net < 10 lines after a helper, or parametrize-shaped in-file repetition): 11.
- Findings by lane: A 1 (S-L-DUP-18) · B 22 (two of them CARDED under CARD-MECHANISM-SWEEP: S-L-DUP-11, -12) ·
  C 6 (S-L-DUP-01, -02, -03, -05, -06, -29). All class DUP (S-L-DUP-23 also moves the test count).
- Top 3 by Δlines: S-L-DUP-01 −332 (Rust self-play uniform producer ×11) · S-L-DUP-12 −305 (carded `_Pool`/`_Buffer`
  exact groups) · S-L-DUP-13 −298 (40 spy event sinks). Next: S-L-DUP-14 −260, S-L-DUP-16 −157, S-L-DUP-15 −145.
- Sum of raised Δ ≈ −2 248 (upper bound; overlapping import lines not merged).
- Root cause of most test duplication: 22 test files justify a private copy with "cross-test imports are barred" /
  "R5 bars cross-test imports" (`git grep -l -i "cross-test import" -- tests | wc -l` → 22). R5 bars `sys.path`
  mutation and a package named `tests`, not helper imports; `tests/_drivable.py` (R367(a)) is imported across
  directories today, and REVIEW2 G3.4 already recorded this reason as false for one file. See DEFECTS.

### Cluster table (ALL clusters)

size = lines per copy × members. ACC = accidental (raised as the finding named), ACC-nil = accidental, not raised.

| K | cluster | size | verdict | reason (one line) |
|---|---|---|---|---|
| K01 | selfplay Rust tests: uniform graph producer | 30–40 × 11 | ACC → -01 | same fn up to batch literal; common/mod.rs exists |
| K02 | selfplay Rust tests/benches: `splitmix64(_step)` | 7 × 8 | ACC → -02 | byte-identical PRNG; common/mod.rs has it |
| K03 | search Rust tests: Gumbel drive + `stub_policy` + `r8_board` | 49–51/8/8 × 2 | ACC → -03 | identical up to `c_scale` param |
| K04 | core `moves.rs` tests `fwm_board` vs `Board::from_stones` | 22 × 2 | ACC → -04 | re-implements a feature-gated builder |
| K05 | `derived_hexg_visit_capacity` kwargs mapping (src) | 9–16 × 3 | ACC → -05 | three hand-copied config→kwargs maps |
| K06 | `DrainCaps` vs `DrainCapsSpec` | 9–13 × 2 | ACC → -06 | same four fields, lifted by hand |
| K07 | `corpus_metrics` ply-entropy block | 17 × 2 | ACC → -07 | verbatim block in two functions |
| K08 | `next_segment_index` (sink / game_record) | 9–10 × 2 | ACC → -08 | same body, only the regex differs |
| K09 | shard-index reader (`first_closed_shard` / tools `closed_shards`) | 16–19 × 2 | ACC → -09 | same index loop, src and tool |
| K10 | `_ppid_of` (src) vs `_ppid_of_pid` (test) | 11 × 2 | ACC → -10 | test re-types a src helper it already imports beside |
| K11 | `hashlib.sha256(path.read_bytes())` beside `sha256_file` | 1–8 × 4 | ACC, CARDED → -11 | CARD-MECHANISM-SWEEP (+1 site the card misses) |
| K12 | composition `_Pool`/`_Buffer` fakes | 5–48 × 27 in 11 exact groups (74 classes total) | ACC, CARDED → -12 | CARD-MECHANISM-SWEEP |
| K13 | spy event sinks (`_Sink`/`_SpySink`/`SpySink`/`SpyEventSink`/…) | 3–14 × 50 | ACC → -13 | one 14-line superset exists in two conftests |
| K14 | composition fakes the card does not name (`_RunnerStats`, `_filled_hexg`, `_fake_run_safety`, `_mirrored`, `_Trainer`) | 5–21 × 41 | ACC → -14 | same family as K12 |
| K15 | eval pipeline harness (`_bounded`, `_eval_cfg`, `fake_mp`, `FakeClock`, `_FakeCtx`, `_FakeProcess`, `_promotion_hooks`, `_tiny_model`) | 5–34 × 22 | ACC → -15 | "house convention" copies; no ruling |
| K16 | schema-test config block builders | 6–22 × 17 | ACC → -16 | identical `_valid_*_block` bodies |
| K17 | AST/token census helpers (`_called_name`, `_enclosing_defs`, `_root_name`, `_call_sites`, `_top_level_imports`, `_code_text`) | 7–14 × 16 | ACC → -17 | identical walkers in 9 test files |
| K18 | module autouse signal-restore = root conftest autouse | 8 × 2 (+root) | ACC → -18 | zero-effect duplicate of an autouse |
| K19 | named `restore_signals` fixtures = root conftest autouse | 7–8 × 2 | ACC → -19 | redundant under the root autouse |
| K20 | run-root recorder harness (`_install_recorders`, `_Recorders`) | 6–29 × 4 | ACC → -20 | identical harness in two root tests |
| K21 | selfplay inference harness (`_TelemetryPool`, `_wire_for`, `_cfg`, `_emit`, `_RStats`) | 6–23 × 10 | ACC → -21 | home `_fused_graph_harness.py` exists |
| K22 | selfplay drain harness (`_ScriptedTime`, `_games_from_golden`) | 6–15 × 5 | ACC → -22 | harness, not the drain oracle |
| K23 | `_p2` twin tests (regime O9/O10/O11; resolver self-test) | 3–17 × 8 | ACC → -23 | identical body over identical input |
| K24 | bridge uniform consumer, in-file ×3 | 13–18 × 3 | ACC → -24 | Python twin of K01 |
| K25 | small tests/train + root pairs (11 groups) | 5–32 × 22 | ACC → -25 | identical stubs/drivers |
| K26 | small monitor/lifecycle pairs (`_alive`, `_Clock`, `FakeClock`, `_events`) | 6–13 × 9 | ACC → -26 | identical stubs |
| K27 | small eval pairs (`_net`, `_caps_for`, `_RuleNet`, `graph_engine`, `_board`) | 6–24 × 11 | ACC → -27 | "cross-test imports are barred" copies |
| K28 | small tools/config/selfplay pairs (10 groups) | 4–17 × 20 | ACC → -28 | identical stubs |
| K29 | PZ test members of K13/K14/K12 families + `deterministic_algorithms`, `_graph_step`, `_coordinator`, `_Opening` | 4–21 × 14 | ACC → -29 | protected-zone contact |
| K30 | queues/graph.rs `submit_graph_results`/`fail_remaining` waiter removal | 8 × 2 | ACC-nil | helper nets ≈ −3 |
| K31 | core benches `board_with_n_stones` (board_bench/smoke_bench) | 10 × 2 | ACC-nil | −8; separate bench binaries, "restated" on purpose |
| K32 | tools/dashboard external/ladder chart scaffold | 7 × 2 | ACC-nil | helper nets < 5 |
| K33 | data/corpus_analysis.py in-file | 7 × 2 | ACC-nil | < 10 |
| K34 | selfplay pool-attribute stubs (game_complete_delivery vs lifecycle_events) | 26 × 2, near | ACC-nil | not identical (ratio < 0.95); rides K22 on contact |
| K35 | selfplay/train full-config literal dicts (search_stats_end_to_end, server_owned_copy, pool_lifecycle, mcts_playout_cap vs selfplay_schema) | 11–12 × 2–3 | ACC-nil, CARD-adjacent | card: "each is a complete config a schema test owns" |
| K36 | in-file test repetition (eval_result_tmp_litter, round_completion_error, lifecycle_events, resolver_wiring, sparse_gumbel_row_target, graph_visit_capacity_relation, sealbot_vendored, deploy_head_vram, coordinator_gates, queue_roundtrip.rs, target_wire_carry.rs, inv27) | 9–26 × 2 each | ACC-nil | parametrize-shaped; net small per file |
| K37 | PZ in-file repetition (graph_contract.rs tests, mint_config.py, eval/worker.py, dispatch.py, aggregate.py, served_sims_exact.rs, target_export_parity.rs, test_bc_graph_reroute.py) | 7–26 × 2 each | ACC-nil (PZ) | protected; below value |
| K38 | conformance memory_envelope/perf_floor wire builder; gnn_v2_witnesses vs conformance | 9–10 × 2 | ACC-nil (PZ) | protected suite; below value |
| K39 | Rust `legal_node_gather` → coords idiom (records.rs, many tests) | 9–12 × many | ACC-nil | idiom, not a unit; records.rs frozen port |
| K40 | core board/moves.rs, board/mod.rs in-file | 7–8 × 2 | ACC-nil | < 10 |
| K-D1 | gnn.py vs gnn_v2.py `forward_batch`/readout | 29 × 2 | DELIBERATE | per-arch hot path (R321), PZ seam |
| K-D2 | arch.py `GnnArch`/`GnnArchV2`/`GnnArchV2SoftPolicy` | 12–20 × 3 | DELIBERATE | per-arch contract dataclasses, PZ seam |
| K-D3 | policy.rs dense vs `_ls` exporters | 10–12 × 4 pairs | DELIBERATE | "completed-Q math is FROZEN" (PZ-3) |
| K-D4 | backup.rs `expand_and_backup_single` vs `_ls_framed` prelude; `pick_topk_children(_ls)` | 9–18 × 2 | DELIBERATE | dense/legal-set twin on the search hot path |
| K-D5 | selection.rs `select_leaves` vs `select_leaves_forced` | 16 × 2 | DELIBERATE | twin with documented divergence (TT fast path absent) |
| K-D6 | gumbel_mctx.rs `round_batch` vs `argmax_at` scoring loop | 9 × 2 | DELIBERATE | scorer already factored (`score_considered`); hot path |
| K-D7 | mcts/parity_tests.rs repeated sigma setup | 10 × 3 | DELIBERATE | Mctx parity oracle (PZ golden) |
| K-D8 | queue_fuse_reserve_parity.rs `fuse_reference` vs queues/wire.rs | 23 × 2 | DELIBERATE | "LOAD-BEARING transcription … must not be simplified" |
| K-D9 | graph_child_parity.rs / target_export_parity.rs flat fixture readers | 71 × 2 | DELIBERATE | frozen oracle files, reader co-located by R8 header (PZ goldens) |
| K-D10 | r153_target_mass / r153_leg2 / target_export_stage1 harnesses | 9–19 × 2–3 | DELIBERATE | prereg'd per-leg measurement harness ("frozen semantics") |
| K-D11 | registry_census.rs per-row field pins | 24 × 3 | DELIBERATE | per-row oracle (AUDIT-1 F-41) |
| K-D12 | records.rs blocks vs selfplay tests | 11–12 × 2 | DELIBERATE | verbatim frozen port (PZ-3) |
| K-D13 | mantis-graph bbox loops vs core | 6 × 3 | DELIBERATE | dep-free crate, independent oracle |
| K-D14 | queue_fuse_bench corpus vs `common::corpus` | 11 × 2 | DELIBERATE | bench input pinned to the measured ledger shape (LAW-09) |
| K-D15 | test_every_key_has_consumer(_p2) registry + mutation tests | 201 × 2 | DELIBERATE | "two independently-maintained copies" (contract doc) |
| K-D16 | tools/check_import_dag.py vs test_worker_sweep_reachability walker | 8–22 × 2 | DELIBERATE | twin detector, "lifted in SHAPE and not imported" |
| K-D17 | test_run_buffer_route.py `_derived` vs src mapping (K05) | 18 × 1 | DELIBERATE | independent oracle for `_select_buffer` |
| K-D18 | pool_hooks `RecorderLike`/`NullRecorder` vs `GameRecorder` | 13–15 × 3 | DELIBERATE | Protocol/impl signature |
| K-D19 | losses `ragged_policy_ce` wrapper; Trainer eval/train step signatures | 9–21 × 2 | DELIBERATE | signature-only, sibling contract |
| K-D20 | exception classes with docstring-only bodies (13+4+3+2 groups) | 4–8 × 2–13 | NOT A CLONE | distinct types; AST false positive |
| K-D21 | tools/audit_bootstrap_corpus.py `sha256_file` | 11 × 1 | DELIBERATE | tool "imports no mantis module" by design |
| K-D22 | tests computing `hashlib.sha256(x.read_bytes())` in assertions | 1 × 15 | DELIBERATE | independent expected-value oracles |
| K-D23 | selfplay tests `drain_shutdown`/`search_seam_fatal` producers | 37–51 × 2 | DELIBERATE | mock-infer and fault-injecting variants of K01 |
| K-D24 | pool_hooks `RecorderLike` vs `ActorSyncTarget` shape; `game_recorder.py` | 13 × 2 | DELIBERATE | typed seam (same as K-D18) |

## Findings

### S-L-DUP-01 | DUP | C
subject: crates/mantis-selfplay/tests/{dirichlet_inert_on_gumbel,gumbel_round_batching,game_result_carries_move_arms,game_result_carries_search_stats,target_support_is_sims_bounded,pcr_arm_matches_the_record}.rs::spawn_producer, served_sims_exact.rs::spawn_counting_producer, {zero_visit_export_pin,ply_cap_game_coverage}.rs::spawn_healthy_graph_producer, {target_wire_carry,target_latch_propagation}.rs::spawn_graph_producer → home crates/mantis-selfplay/tests/common/mod.rs (existing; `#![allow(dead_code)]`, header "ONE implementation")
claim: eleven uniform-prior graph-queue producer threads are one function up to the `pop_graph_batch(N, 5)` batch argument and rustfmt layout.
evidence: `scratchpad/ldup/rsfn.py producer <selfplay tests>` → body hashes cf32ac6f ×2, 17e90fea ×2, b1e39c04 ×2 plus five singletons; `diff` of each singleton against zero_visit_export_pin.rs::spawn_healthy_graph_producer → only the fn name and the batch literal (8 / 4 / LEAF_BATCH / PROD_LEAF_BATCH) differ; target_wire_carry.rs adds a `res` temporary.
callers: n/a (DUP, not a DEAD claim)
deliberate?: a stand-in inference server, not an oracle: every assertion sits in the tests, no parity test pins the producer, and common/mod.rs already exists for exactly this ("two copies would let the proof drift"). Excluded as deliberate variants: drain_shutdown.rs (mock_graph_infer) and search_seam_fatal.rs (fault injection), K-D23. Lane C: crates/mantis-selfplay/tests/** is PZ (served_sims_exact.rs pins served-sims exactness).
Δlines: −332 (`python3 scratchpad/ldup/delta.py 12 <11 members>` → copies 395, keep 40, +11 `use`, +11 `mod common;`, +1 batch parameter)
witness: the eleven test binaries (`cargo test -p mantis-selfplay --test <name>`); not run here (CPU budget)
depends: —

### S-L-DUP-02 | DUP | C
subject: crates/mantis-selfplay/tests/{hexg_sample_parallel_parity,leaf_graph_parallel_parity,queue_fuse_reserve_parity}.rs::splitmix64, {drain_shutdown,worker_output_pin}.rs::splitmix64_step, benches/graph_build_bench.rs::splitmix64_step, benches/queue_fuse_bench.rs::splitmix64 → tests/common/mod.rs::splitmix64 (made `pub`)
claim: seven byte-identical SplitMix64 steps beside the one already in common/mod.rs (queue_fuse_bench.rs even includes common via `#[path]` and still carries its own).
evidence: `sed -n '/^fn splitmix64/,/^}/p' <file> | md5sum` → 578e361e… for common + 3 tests, d44bcc48… for the three `_step` copies (differ only in the name); `rsfn.py '^splitmix64'` → 7 lines each.
callers: n/a
deliberate?: a PRNG step, not the object under test; output is unchanged by sharing it, so goldens (worker_output_pin.rs) and bench inputs do not move. Lane C: PZ (crates/mantis-selfplay/tests/**).
Δlines: −36 (7 copies × 7 = −49; +2 lines `mod common;`/`use` in 5 tests, +3 `#[path]`/`mod`/`use` in graph_build_bench.rs)
witness: the seven targets; worker_output_pin.rs reds if a golden-feeding sequence changed
depends: S-L-DUP-01

### S-L-DUP-03 | DUP | C
subject: crates/mantis-search/tests/gumbel_sparse_row_kl.rs::{search,stub_policy,r8_board} and crates/mantis-search/tests/r8_game_target_mass.rs::{search,stub_policy,r8_board} → new crates/mantis-search/tests/common/mod.rs
claim: the Gumbel search drive, the skewed stub prior and the R8 board are identical in both files, `search` differing only by a `c_scale` parameter (r8_game_target_mass hard-codes 50.0/0.1).
evidence: `diff <(sed -n 56,103p gumbel_sparse_row_kl.rs) <(sed -n 46,95p r8_game_target_mass.rs)` → signature, the two `QSigma` literals and two comments; `rsfn.py` → stub_policy 244bed93 ×2, r8_board bfe30e7b ×2.
callers: n/a
deliberate?: gumbel_sparse_row_kl.rs's R8 header argues the drive must not change without the measurement; a shared drive enforces that across both files. Not a frozen harness (unlike the R153 legs, K-D10). Lane C: crates/mantis-search/tests/** is PZ.
Δlines: −58 (`delta.py 5 …` → copies 133, keep 68, +2 `use`, +5 = 2 `mod common;` + 3-line module header)
witness: `cargo test -p mantis-search --test gumbel_sparse_row_kl --test r8_game_target_mass`
depends: —

### S-L-DUP-04 | DUP | B
subject: crates/mantis-core/src/board/moves.rs::tests::fwm_board vs crates/mantis-core/src/board/state/core.rs::Board::from_stones
claim: the in-crate test builder re-implements the `test-fixtures`-gated builder because `from_stones` is invisible to mantis-core's own unit tests.
evidence: `sed -n 729,750p moves.rs` vs `sed -n 538,568p state/core.rs` → same bbox loop and field writes; crates/mantis-search/src/tactics/search.rs already wraps it as `Board::from_stones(stones, player, mr, stones.len() as u32, None)`; `grep -c "fwm_board(" moves.rs` → 10 (1 def, 9 calls).
callers: n/a
deliberate?: no oracle role (it builds inputs); the only reason is the cfg gate. Proposal: `#[cfg(any(test, feature = "test-fixtures"))]` on the impl (and its test module), fwm_board becomes the 3-line wrapper. One behavioural edge to check: fwm_board sets `has_stones`/bbox even for an empty slice, from_stones only for a non-empty one — all 9 calls pass stones.
Δlines: −19 (22-line fn → 3-line wrapper; the cfg edit is 0 lines)
witness: `cargo test -p mantis-core --lib board::moves`
depends: —

### S-L-DUP-05 | DUP | C
subject: src/mantis/config/schema/core.py::RunConfig (the hexg visit-capacity validator's kwargs), src/mantis/run.py::_derived_visit_capacity, src/mantis/diagnostics/worker_sweep.py (graph arm of the buffer builder) → one module-level `derived_visit_capacity(config)` beside the validator in schema/core.py
claim: the same ten-keyword config→`derived_hexg_visit_capacity` mapping is hand-typed three times in src.
evidence: `git grep -n "derived_hexg_visit_capacity(" -- '*.py'` → schema/core.py, run.py, worker_sweep.py, tests/test_run_buffer_route.py, tests/bridge/test_hexg_visit_capacity.py; `awk 'NR>=587&&NR<=601' schema/core.py` (15), `run.py::_derived_visit_capacity` span 16, `worker_sweep.py` 653–661 (9).
callers: n/a
deliberate?: not a mirror — no test pins the three against each other; the test copy in tests/test_run_buffer_route.py::_derived IS an independent oracle for `_select_buffer` and stays (K-D17). Lane C: run.py and schema/core.py are PZ.
Δlines: −20 (−13 schema, −8 worker_sweep, −16 run.py def; +16 home fn, +1 import)
witness: tests/test_run_buffer_route.py (compares `_select_buffer(...).visit_capacity` against its own derivation), tests/config/test_graph_visit_capacity_relation.py
depends: —

### S-L-DUP-06 | DUP | C
subject: src/mantis/eval/pipeline.py::DrainCaps vs src/mantis/config/resolve/drain.py::DrainCapsSpec
claim: two frozen dataclasses with the same four float fields, the spec lifted field-by-field through `StepCoordinatorConfig` into `DrainCaps` in run.py.
evidence: `grep -n "class DrainCaps" -A8 eval/pipeline.py` and resolve/drain.py → identical field lists; `tophash` group 8e600dd3ea; pipeline.py already imports `mantis.config.resolve.*` (fused_graph_caps, inference_batching), so the drain.py docstring's layering reason ("mantis.config may not import mantis.eval") does not bar `DrainCaps = DrainCapsSpec` in the eval direction.
callers: n/a
deliberate?: layering was the stated reason and it holds only in the config→eval direction; eval importing the config spec is already done twice in the same file. Lane C: eval/pipeline.py and run.py are PZ.
Δlines: −8 (9-line class → 1 import alias; the six-line lift in run.py could go too, not counted)
witness: tests/config/test_drain_caps_wiring.py, the eval pipeline suites that construct `DrainCaps(...)`
depends: —

### S-L-DUP-07 | DUP | B
subject: src/mantis/data/corpus_metrics.py::analyse_move_entropy and ::_compute_per_game_entropies → a private `_ply_entropies(records)` in the same module
claim: the 17-line per-ply move-entropy computation is pasted verbatim in both functions.
evidence: jscpd `corpus_metrics.py:225-243 ↔ 490-508` (19); `awk` 225–241 and 490–506 → 17 lines each, identical.
callers: n/a
deliberate?: same module, same inputs, no oracle or parity role.
Δlines: −12 (−34; +20 helper incl. def/docstring/return; +2 call lines)
witness: tests/data/test_sources_metrics.py (`run_analysis` key sets) — torch-free, green at HEAD (66 passed in the targeted run below)
depends: —

### S-L-DUP-08 | DUP | B
subject: src/mantis/monitor/game_record.py::next_segment_index vs src/mantis/monitor/sink.py::next_segment_index
claim: same body, only the filename regex differs (`_SHARD_RE` vs `_SEGMENT_RE`); game_record.py already imports from sink.py.
evidence: asthash group 475f7265b344 (locals renamed); `git grep -n next_segment_index -- src tests tools` → one caller each, none in tests.
callers: n/a
deliberate?: two segment conventions, one scan; not a mirror.
Δlines: −9 (game_record copy; sink's gains a `pattern` parameter on the same line)
witness: tests/monitor/test_sink.py, test_rotation_on_resume.py, test_game_record.py (torch-free parts green at HEAD)
depends: —

### S-L-DUP-09 | DUP | B
subject: src/mantis/diagnostics/mirror_receipts.py::first_closed_shard vs tools/mirror_pull.py::closed_shards
claim: both read the shard index line by line with the same skip/parse/`shard_closed` filter; the first is `closed_shards(...)[0][0]`.
evidence: finer jscpd `mirror_receipts.py:41-49 ↔ mirror_pull.py:127-135`; spans 19 and 16 (AST).
callers: n/a
deliberate?: the puller is the producer the halt reads after; one index reader beside `index_filename` (monitor/game_record.py) serves both. mirror_pull.py already imports `mantis.*`.
Δlines: −11 (first_closed_shard 19 → 7; closed_shards moves into src, net 0; +1 import in the tool)
witness: tests/diagnostics/test_mirror_receipts.py, tests/tools/test_mirror_pull.py, tests/tools/test_preflight_start_halts.py
depends: —

### S-L-DUP-10 | DUP | B
subject: tests/test_run_pdeathsig.py::_ppid_of_pid vs src/mantis/train/lifecycle/signals.py::_ppid_of
claim: the test re-types the src helper byte-for-byte while already importing from the same module.
evidence: asthash group 5c389393938b; `grep -n "^from mantis.train.lifecycle.signals import" tests/test_run_pdeathsig.py` → present.
callers: n/a
deliberate?: used only to reap a wrapper process in teardown, not as an oracle for `_ppid_of`.
Δlines: −10 (11-line def; +1 name in the existing import)
witness: tests/test_run_pdeathsig.py
depends: —

### S-L-DUP-11 | DUP | B (CARDED: CARD-MECHANISM-SWEEP)
subject: src/mantis/bots/strix.py, src/mantis/diagnostics/fusion_calibrate.py::_sha256, src/mantis/encoding/__init__.py (the card's three) + src/mantis/diagnostics/worker_sweep.py::_sha256 (NOT named by the card) vs src/mantis/util/hashing.py::sha256_file
claim: already carded; one further site exists that the card's list misses (worker_sweep.py::_sha256, which returns None on OSError).
evidence: `git grep -n "hashlib.sha256(.*read_bytes())" -- src tools` → the four src sites; tools/audit_bootstrap_corpus.py::sha256_file is deliberate (K-D21).
callers: n/a
deliberate?: no; encoding/__init__.py needs `.digest()` bytes (the registry handshake) so it converts rather than deletes.
Δlines: −3 (fusion_calibrate.py::_sha256 4 lines → import; the others are 0 net)
witness: tests/encoding/test_registry_sha_handshake.py, tests/diagnostics/test_worker_sweep_authority.py (`ws._sha256` assertion)
depends: —

### S-L-DUP-12 | DUP | B (CARDED: CARD-MECHANISM-SWEEP)
subject: the composition tests' private `_Pool`/`_Buffer` fakes — exact-duplicate groups only: _Buffer ×4 (clean_stop_save, cluster_stat_wiring, gate_interval_decoupling, iteration_complete_decoupling::_FakeBuffer); _SyncRecordingPool/_Pool ×2 (tests/test_actor_sync_composition.py, tests/test_run_strict_composition.py); _Pool ×2 (inference_seam_events, target_counter_events); _SyncRecordingPool/_Pool ×2 (actor_sync_production_posture, actor_sync_real_config); _Buffer ×3 (inference_seam_events, target_counter_events, terminal_eval_rc); _Pool ×2 (abort_exit_signal, clean_stop_save); _Pool ×2 (periodic_checkpoint, train_step_dispatch); _Buffer ×4 (config/drain_caps_wiring, actor_lag_wiring_live, actor_sync_production_posture, actor_sync_real_config); FakeBuffer ×2 (coordinator_gates, eval_result_routing); _Buffer ×2 (ply_cap_gate, policy_loss_trough_gate); _Buffer/_DrivableBuffer ×2 (test_actor_sync_composition, test_run_composition) → tests/_drivable.py
claim: carded; the census counts 74 private Pool/Buffer fake classes (1 518 lines) in tests, of which 27 fall into 11 byte-identical groups.
evidence: `scratchpad/ldup/fakes.py` → 74 / 1 518; `tophash.py` groups 907f9cac44, 78ef6218e6, 9da8c6331c, 3d7c6862df, bd5854866c, ce0f105f0b, 2aefab26e6, 6d21c46705, 6f9cee97c5, e1b8371c07, 3ed1f79877.
callers: n/a
deliberate?: the reason given in-file is "R5 bars cross-test imports" (false, see DEFECTS); `tests/_drivable.py` is the R367(a) precedent. PZ member excluded: the drawrate pair (c8bef7dfb3) → S-L-DUP-29.
Δlines: −305 (`delta.py 0 <27 members>` → copies 592, keep 266, +21 imports)
witness: the 21 member modules (torch-dependent; not runnable here)
depends: —

### S-L-DUP-13 | DUP | B
subject: 40 top-level spy event sinks in tests (e.g. tests/train/test_policy_loss_trough_gate.py::_Sink, tests/eval/test_round_completion_error.py::_SpySink, tests/selfplay/test_lifecycle_events.py::_RecordingSink, tests/train/_microbatch_harness.py::SpySink, tests/train/conftest.py::SpyEventSink, tests/monitor/conftest.py::SpyEventSink; full list: `scratchpad/ldup/sinks.py`) → tests/_drivable.py::SpyEventSink (the 14-line superset: `emit`/`named`/`has`)
claim: 50 classes (411 lines) implement the same "append `dict(event)`, filter by `event` name" spy; 40 non-PZ, top-level ones reduce to one.
evidence: `sinks.py` → 50 / 411; emit bodies all `self.events.append(dict(event|payload))` except tests/eval/test_eval_round_observability.py (raw append) — excluded, as are the nested/3-line class-attribute variants (test_round_instruments_are_measured, test_quiescence_fires_producer, test_rates_are_measured).
callers: n/a
deliberate?: a recorder, not an oracle; no variant carries behaviour the superset lacks. PZ members excluded (test_draw_rate_is_a_fraction, test_drawrate_abort_threading, test_drawrate_gate_branch_flipset, test_drawrate_gate_capacity, test_resume_owned_paths, test_strength_floor_verdict_on_the_routed_mapping) → S-L-DUP-29. Note _microbatch_harness.py is imported by PZ tests (it is not itself PZ).
Δlines: −298 (`delta.py 0 <40>` → copies 352, keep 14, +40 imports)
witness: the 40 member modules
depends: S-L-DUP-12 (same home)

### S-L-DUP-14 | DUP | B
subject: composition-harness fakes the card does not name: `_RunnerStats` ×16 (tests/config/{test_coordinator_knobs_wiring,test_drain_caps_wiring}.py, tests/{test_actor_sync_composition,test_run_composition,test_run_strict_composition}.py, tests/train/{actor_lag_wiring_live,actor_sync_production_posture,actor_sync_real_config,coordinator_gates,eval_result_routing,gate_interval_decoupling,iteration_complete_decoupling,periodic_checkpoint,policy_loss_trough_gate,terminal_eval_rc,train_step_dispatch}); `_filled_hexg` ×12 (tests/train/test_{abort_exit_signal,clean_stop_save,cluster_stat_wiring,coordinator_gates,eval_result_routing,gate_interval_decoupling,inference_seam_events,iteration_complete_decoupling,ply_cap_gate,policy_loss_trough_gate,target_counter_events,terminal_eval_rc}); `_fake_run_safety` ×4; `_mirrored` ×4; `_Trainer` ×3 (cluster_stat_wiring, gate_interval_decoupling, iteration_complete_decoupling::_FakeTrainer) and ×2 (inference_seam_events, target_counter_events) → tests/_drivable.py
claim: the same family as S-L-DUP-12, byte-identical across up to 16 files.
evidence: `tophash.py` groups 08dc134d7d (17 incl. PZ), 1f4af7d746 (13 incl. PZ), da8300a399, d3978269e9, 18f92ba53c, 9c9c6ada5e.
callers: n/a
deliberate?: `_filled_hexg`'s own docstring gives the reason "R5 bars cross-test imports, so each file that needs one builds it" — false (DEFECTS). The card's "the trainer stub is one since F2" holds for the root composition tests only: these 5 `_Trainer` copies are not `DrivableTrainerStub`. PZ member excluded: tests/train/test_drawrate_abort_threading.py (its `_RunnerStats`, `_filled_hexg`, `_fake_run_safety`).
Δlines: −260 (`delta.py 0 <41>` → copies 350, keep 68, +22 imports)
witness: the 22 member modules
depends: S-L-DUP-12

### S-L-DUP-15 | DUP | B
subject: tests/eval/{test_round_completion_error,test_escalate_join_timeout_bound,test_eval_broken,test_eval_broken_reason_routes,test_pipeline_isolation}.py::{_bounded,_eval_cfg,fake_mp,FakeClock,_FakeCtx,_FakeProcess,_promotion_hooks,_tiny_model} → new tests/eval/_pipeline_harness.py
claim: the eval-pipeline test rig is copied per suite; 22 copies are ≥ 0.95 similar to the canonical one.
evidence: `samename.py <7 eval files + train/test_terminal_eval_rc.py>` → e.g. `_promotion_hooks` 1.00 ×4, `_FakeCtx` 1.00/0.96/1.00, `_eval_cfg` 1.00/0.96/1.00, `_FakeProcess` 1.00/0.95; lower-ratio variants (`_pipeline_kwargs` 0.65–0.90, test_pipeline_isolation's `fake_mp` 0.53) left out of the Δ.
callers: n/a
deliberate?: the in-file reason is "the eval-suite house convention" (test_round_completion_error.py, test_escalate_join_timeout_bound.py R8 header); no ruling or doc defines it (`git grep -i "house convention"` → only these comments and the analyzer design doc). test_escalate_join_timeout_bound.py's "frozen suites" wording has no freeze behind it (PZ-3: no freeze manifest). R8: test_escalate_join_timeout_bound.py (315) falls under 300 and must drop its header; the header text of test_eval_broken_reason_routes.py cites the same false reason.
Δlines: −145 (`delta.py 3 <22>` → copies 254, keep 101, +5 imports, +3 header)
witness: the five member modules (torch-dependent)
depends: S-L-DUP-13 (their `_SpySink`s are counted there)

### S-L-DUP-16 | DUP | B
subject: tests/config/{test_schema,test_schema_strict}.py::_valid_{monitor,selfplay,eval,inference}_block, tests/config/{test_eval_schema_bounds,test_actor_sync_schema,test_train_policy_value_target_consistency}.py::_{monitor,selfplay,eval,inference}_block → new tests/config/_schema_blocks.py
claim: four config-block builders are body-identical across five schema test modules.
evidence: `tophash.py` groups 72a6d5dd40 (monitor), 12d7e18bd6 (selfplay), dcc18fdbde (eval), 92832d789a (inference); jscpd test_schema.py ↔ test_schema_strict.py 73 lines.
callers: n/a
deliberate?: CARD-MECHANISM-SWEEP names "the nine full-config literal dicts … each is a complete config a schema test owns" and leaves them; these are the per-block BUILDERS feeding such dicts, byte-identical, so ownership is not at stake — flagged as card-adjacent for the dispatcher's call. PZ members excluded: tests/train/test_resume_wiring_integration.py (`_monitor_block`, `_selfplay_block`), tests/train/test_checkpoint_conformance.py (`_make_eval_block`).
Δlines: −157 (`delta.py 3 <17>` → copies 223, keep 58, +5 imports, +3 header)
witness: the five member modules (tests/config/test_schema.py and test_schema_strict.py are torch-free)
depends: —

### S-L-DUP-17 | DUP | B
subject: tests/{test_run_import_authority,test_run_main_authority,test_run_one_authority}.py::{_called_name,_enclosing_defs,_root_name,_call_sites}; tests/monitor/test_monitor_census.py, tests/train/test_run_safety_wiring.py, tests/train/test_train_import_dag.py::_top_level_imports; tests/test_run_one_authority.py, tests/tools/{test_preflight_child_convergence,test_preflight_mint,test_preflight_parent_census}.py::_code_text → new tests/_ast_census.py
claim: six AST/token walkers are body-identical across nine test modules.
evidence: asthash groups d2225b280c45, 57f0522a0796, 8d7d03f0367d, 51d4fc7c948e, dcc146efc4b9; `tophash.py` f94e133b59 (`_code_text` ×4).
callers: n/a
deliberate?: each copy is a tool used by its census, not an oracle of the walker itself. Two further `_code_text` variants (tests/config/test_armed_abort_manifest.py, tests/config/test_drawrate_arming_authority.py [PZ]) differ and are not counted. The R8 header of tests/test_run_one_authority.py cites the barred-import reason.
Δlines: −77 (`delta.py 3 <16>` → copies 151, keep 62, +9 imports, +3 header)
witness: the nine member modules
depends: —

### S-L-DUP-18 | DUP | A
subject: tests/test_run_partial_composition.py::restore_signal_dispositions and tests/test_run_root_lifecycle.py::restore_signal_dispositions (both `@pytest.fixture(autouse=True)`) vs tests/conftest.py::_restore_signal_dispositions (autouse, every test)
claim: two module autouse fixtures repeat the root conftest's autouse save/restore of SIGINT/SIGTERM around the same tests, so they have no effect at any test boundary.
evidence: `sed -n 44,57p tests/conftest.py` → autouse, "restore-AROUND, so inner save/restores nest cleanly"; the two module fixtures have the identical body (`tophash.py` 848f6f4197).
callers: by-name: `git grep -n restore_signal_dispositions -- .` → only the two defs, the root def and a docs/slim/00_MAP.md mention; `git grep -n usefixtures -- tests | grep -i signal` → none; autouse fixtures are never requested by name, and the root one wraps every test in tests/ (both files sit directly under tests/).
deliberate?: a copy of an autouse that already runs; no mutation or planted break reads it.
Δlines: −17 (8 + 8 lines; tests/test_run_partial_composition.py also loses its now-unused `import signal`)
witness: tests/test_run_root_lifecycle.py (its handler-installed assertion), tests/test_run_partial_composition.py — torch-dependent; the probe needs the box or a torch venv
depends: —

### S-L-DUP-19 | DUP | B
subject: tests/train/test_lifecycle_contract.py::restore_signals and tests/train/test_orphan_workers_census.py::restore_signals (named fixtures, 5 requesting tests)
claim: both save/restore SIGINT/SIGTERM exactly as the root autouse already does around every test.
evidence: `sed -n '/^def restore_signals/,/^$/p'` → getsignal/yield/signal for the same two signals; `grep -n restore_signals` → 3 + 2 requesting tests.
callers: n/a
deliberate?: redundant with tests/conftest.py::_restore_signal_dispositions. Lane B, not A: five test signatures change, and test_orphan_workers_census.py's mutation test should be read for fixture-order reliance before the fixture goes.
Δlines: −15 (8 + 7; signature edits 0 lines)
witness: the two modules
depends: S-L-DUP-18

### S-L-DUP-20 | DUP | B
subject: tests/test_run_partial_composition.py::{_install_recorders,_Recorders} vs tests/test_run_root_lifecycle.py::{_install_recorders,_Recorders} → tests/_drivable.py
claim: the composed-boot recorder harness is identical in both root tests.
evidence: `tophash.py` f1cf74ca01 (28/29), 896d890f92 (6/8); asthash a3b2c2d8a966, 2408b0eb4c9d (nested `_recording_build`).
callers: n/a
deliberate?: both R8 headers give the barred-import reason for keeping a copy. R8: tests/test_run_partial_composition.py (317) falls under 300 with S-L-DUP-18 and must drop its header.
Δlines: −32 (`delta.py 0 <4>` → copies 71, keep 37, +2 imports)
witness: the two modules
depends: S-L-DUP-18

### S-L-DUP-21 | DUP | B
subject: tests/selfplay/{test_fusion_counters,test_inference_batch_timing}.py::{_TelemetryPool,_emit,_RStats}, tests/selfplay/{test_inference_batch_timing,test_inference_server}.py::{_wire_for,_cfg} → tests/selfplay/_fused_graph_harness.py (existing: "Shared rig for the … graph-inference-fusion oracles")
claim: five inference-server test helpers are body-identical in pairs.
evidence: `tophash.py` 13966c56d6, 4f5be129b6, cfed354dbc, 60947cb1dc, 5b32010944; jscpd test_inference_batch_timing ↔ test_inference_server 43, ↔ test_fusion_counters 40.
callers: n/a
deliberate?: rig, not oracle; the home module already exists for this rig. R8: tests/selfplay/test_fusion_counters.py (329) likely falls under 300 (with its `_ListSink` from S-L-DUP-13) — check `wc -l` after.
Δlines: −64 (`delta.py 0 <10>` → copies 136, keep 69, +3 imports)
witness: the three modules
depends: —

### S-L-DUP-22 | DUP | B
subject: tests/selfplay/test_game_complete_delivery.py::_ScriptedTime, test_pool_drain_arms.py::{_Clock,_games_from_golden}, test_pool_drain_parity.py::{ScriptedTime,_games_from_golden} → new tests/selfplay/_drain_harness.py
claim: the scripted clock and the golden-games loader are body-identical across the drain suites.
evidence: `tophash.py` 6fcd1c130c (13/13/15), f807add2b5 (6/7).
callers: n/a
deliberate?: harness, not the drain oracle (PZ-6 names `pool_push.py::push_dense` as the drain-parity oracle; that is untouched). The near-identical hand-built pool attribute stubs (K34, 26 lines, ratio < 0.95) ride this on contact.
Δlines: −26 (`delta.py 3 <5>` → copies 54, keep 22, +3 imports, +3 header)
witness: the three modules
depends: —

### S-L-DUP-23 | DUP | B (test-floor move)
subject: tests/config/test_regime_parity_p2.py::{test_o9_sims_regime_parity_unchanged,test_o10_amp_is_bf16_on_graph_unchanged,test_o11_encoding_regime_parity_unchanged} (twins of tests/config/test_regime_parity.py::test_o9_sims_regime_parity/test_o10_amp_is_bf16_on_graph/test_o11_encoding_regime_parity) and tests/config/test_every_key_has_consumer_p2.py::test_the_resolver_bites_on_a_dead_symbol
claim: four tests are AST-identical to a test in the sibling file over the same fixture, so they cannot disagree with it.
evidence: `scratchpad/ldup/p2cmp.py <orig> <p2>` → IDENTICAL for all four (spans 17, 3, 3, 5); the p2 file's own docstring says it is a rewrite whose O9–O11 are "UNCHANGED"; the resolver self-test reads no `CONSUMER_REGISTRY`.
callers: n/a
deliberate?: the every-key REGISTRY pair and its per-copy bijection and mutation tests are deliberate (K-D15, contract doc "a mutation self-test in both copies") and stay; these four check nothing file-specific. Both _p2 files survive (the p2 keeps the radius-absence tests the contract doc cites).
Δlines: −28 (17 + 3 + 3 + 5); collected count −4. Floor: tools/ci_gates/test_count_floor.txt reads 4862 and need not move (the HEAD commit subject records 5 112 collected; the count cannot be re-measured here without torch).
witness: tests/config/test_regime_parity.py, tests/config/test_every_key_has_consumer.py (torch-free; green in the targeted run)
depends: —

### S-L-DUP-24 | DUP | B
subject: tests/bridge/test_mcts_inference_roundtrip.py — two nested `consumer()` closures and one inline loop serving uniform graph batches
claim: the same uniform-prob serve step is written three times in one module (the Python twin of S-L-DUP-01).
evidence: jscpd self 36 lines; `diff <(sed -n 83,105p) <(sed -n 203,225p)` → one line differs; `awk 'NR>=183&&NR<=195'` the third copy.
callers: n/a
deliberate?: stand-in server, no oracle role.
Δlines: −21 (18 + 18 + 13 = 49 → module helper 14 + two 6-line loops + 2-line inline = 28)
witness: tests/bridge/test_mcts_inference_roundtrip.py
depends: —

### S-L-DUP-25 | DUP | B (batch: small tests/train + root pairs)
subject: tests/train/test_{aux_soft_policy,policy_loss_warmup}.py::_step_once; test_{ply_cap_gate,policy_loss_trough_gate}.py::_drive; test_{periodic_checkpoint,train_step_dispatch}.py::_graph_buffer; test_{coordinator_gates,iteration_complete_decoupling}.py::_make_config; test_actor_lag_{sample_emission,watchdog}.py::_registry; test_{eval_heartbeat,run_safety_wiring}.py::_ExitSpy; test_actor_sync_{production_posture,real_config}.py::_bounded_config; tests/test_run_disk_guard_abort_rc.py + tests/train/test_terminal_eval_rc.py::{_await_signal,_fake_disk_usage}; tests/test_{run_composition,run_strict_composition}.py::_no_terminal_eval_config; tests/train/test_{inference_seam_events,target_counter_events}.py::_drive → tests/_drivable.py (cross-directory pairs) or the existing tests/train helpers
claim: eleven helper pairs are body-identical.
evidence: `tophash.py` 3d8bccc98c, 53002f6705, c1c83d52fa, ebc3e0116f, ebbded1bdc, 4548eb4b2d, 82c4ff9501, a44bcb97e0, cb11d1a3cd, e8b6239da5, 13a28cdba0.
callers: n/a
deliberate?: none is an oracle; several carry the barred-import reason.
Δlines: −83 (`delta.py 0 <22>` → copies 219, keep 116, +20 imports)
witness: the 20 member modules
depends: S-L-DUP-12

### S-L-DUP-26 | DUP | B (batch: monitor/lifecycle)
subject: tests/monitor/test_supervisor_signal_posture.py, tests/test_run_pdeathsig.py, tests/train/test_parent_death_signal.py::_alive; tests/monitor/test_supervisor.py + tests/train/test_heartbeat_watchdog.py::_Clock; tests/monitor/conftest.py + tests/train/test_drain_hardcap_wiring.py::FakeClock; tests/monitor/test_supervisor_{config_witness,signal_posture}.py::_events → tests/_drivable.py
claim: four helpers are body-identical across 8 files.
evidence: `tophash.py` d697728ce3, 61befef937, 6c7e1ab1e6, 1260756332.
callers: n/a
deliberate?: none is an oracle.
Δlines: −37 (copies 89, keep 44, +8 imports; the conftest `SpyEventSink` pair is counted in S-L-DUP-13)
witness: the 8 member modules
depends: —

### S-L-DUP-27 | DUP | B (batch: eval)
subject: tests/eval/test_{eval_concurrency_row,game_record_eval_channel}.py::_net; test_{eval_decode_guard_ordering,graph_round_encoding}.py::_caps_for; test_{eval_selfplay_child_parity,rung_seat_off_window}.py::{_RuleNet,graph_engine}; + test_eval_value_channel.py::_board (×3) → tests/eval/_pipeline_harness.py (S-L-DUP-15's home)
claim: five helpers are body-identical across 7 eval modules.
evidence: `tophash.py` eece2a8c25, cb8ca43d3d, 33c4dcb0e5, bf27ef0dcb, f699f687ec.
callers: n/a
deliberate?: test_rung_seat_off_window.py says "The stub net is the ONE stand-in and is duplicated rather than imported, since cross-test imports are barred" — the premise is false and sharing makes it literally one. test_eval_selfplay_child_parity.py is a parity oracle, but the shared parts are its stub net and board reader, not the golden or the predicate.
Δlines: −52 (copies 126, keep 67, +7 imports)
witness: the 7 member modules
depends: S-L-DUP-15

### S-L-DUP-28 | DUP | B (batch: tools/config/selfplay)
subject: tests/tools/test_strix_{net_only_cell,ruler_r6_cell}.py::{_load,external}; test_strength_frontier_{book,strix}.py::base; test_{dashboard_reader,run_dashboard}.py::_write; test_{dashboard_strength,run_dashboard}.py::_ladder; test_preflight_mint{,_process}.py::_model_samples; tests/config/test_preflight_stamp.py + tests/tools/test_preflight_stamp_writer.py::state_home; tests/config/test_{armed_abort_cadence,cadence_clock_mutations}.py::_revalidated; tests/selfplay/test_pool_hparams{,_arms}.py::(_)RecordingRunnerConfig; tests/diagnostics/test_mirror_receipts.py::_receipt_everything + tests/tools/test_preflight_start_halts.py::_receipt_run_dir → the directory's existing helper (tests/tools/_ladder_stub.py) or tests/_drivable.py for cross-directory pairs
claim: ten helper pairs are body-identical.
evidence: `tophash.py` f1147f9e3e, 86e724cbbb, 9d85f7da6d, 7f674e30c1, 58584e13fa, 2bc8ae2f06, f0733eb7c7, 2f8111564f, f234816dbd; jscpd test_mirror_receipts ↔ test_preflight_start_halts 11.
callers: n/a
deliberate?: none is an oracle.
Δlines: −50 (`delta.py 0 <20>` → copies 140, keep 73, +17 imports)
witness: the 17 member modules
depends: —

### S-L-DUP-29 | DUP | C (protected-zone members of the families above)
subject: tests/model/_bf16_parity.py + tests/train/_microbatch_harness.py::deterministic_algorithms (LAW-06 parity helper); tests/train/test_{finite_gradient_guard,nonfinite_guard}.py::_graph_step; tests/train/test_drawrate_gate_{branch_flipset,capacity}.py::{_coordinator,_Buffer,_SpySink}; tests/arena/{test_battery_concurrency,test_legality_boundary,test_match_fairness,test_ply_cap_adjudication}.py::_Opening; plus the PZ members excluded from S-L-DUP-12/-13/-14/-16 (test_drawrate_abort_threading, test_draw_rate_is_a_fraction, test_resume_owned_paths, test_strength_floor_verdict_on_the_routed_mapping, test_resume_wiring_integration, test_checkpoint_conformance) and K38 (conformance suite)
claim: the same byte-identical stubs reach into protected-set test files (finite-gradient guard, draw-rate abort, arena legality, resume round-trip, strength_floor, the conformance suite).
evidence: `tophash.py` 3490b55864, 65d60fff00, bcbfa22488, c8bef7dfb3, 21e3a075cd, 8104433339; PZ.md glob list.
callers: n/a
deliberate?: stubs, not the protected assertions; merging weakens nothing, but PZ contact is lane C by rule.
Δlines: −63 for the six named groups (`delta.py 3 <14>` → copies 148, keep 72, +10 imports, +3 header); the excluded PZ members of S-L-DUP-12/-13/-14/-16 add ≈ −100 more if they follow their families (not derived member by member)
witness: the protected-set test files named
depends: S-L-DUP-12, -13, -14, -16

## DEFECTS
- 22 test files state "cross-test imports are barred" / "R5 bars cross-test imports" as the reason for a local copy (`git grep -l -i "cross-test import" -- tests`); R5 bars `sys.path` writes and a `tests` package only, tests/_drivable.py is imported across directories under R367(a), and REVIEW2 G3.4 already ruled the reason false for one file. Several of these are R8 headers, so gate 15's stale-justification class applies.
- CARD-MECHANISM-SWEEP's `hashlib.sha256(path.read_bytes())` list misses src/mantis/diagnostics/worker_sweep.py::_sha256 (S-L-DUP-11).
- CARD-MECHANISM-SWEEP's "the trainer stub is one since F2" is true only for the root composition tests; five `_Trainer` copies in tests/train are not `DrivableTrainerStub` (S-L-DUP-14).
- tests/eval/test_escalate_join_timeout_bound.py calls test_round_completion_error/test_eval_broken "the frozen suites"; no freeze exists at HEAD (PZ-3).

## PARKED
none

## HANDOFF
- T6 (tests/eval): the "eval-suite house convention" wording in the R8 headers of test_escalate_join_timeout_bound.py and test_eval_broken_reason_routes.py is a stale justification whether or not S-L-DUP-15 lands.
- T4 (tests/config): the full-config literal dicts (K35) are the card's own item; this lens did not re-derive them.

## Not covered
- Nested functions and classes are grouped only where jscpd saw them (≥ 8 lines / 60 tokens in tests); the AST passes hash top-level defs and function bodies, so a clone that is a nested class inside one test and top-level in another is missed.
- Rust was scanned by jscpd plus a regex fn-body hash; there is no Rust AST pass, and trait-impl or macro-generated repetition was not examined.
- Near-duplicates under 0.95 similarity are named (K34, the `_pipeline_kwargs` variants) but never counted in a Δ.
- tools/viewer and tools/dashboard JavaScript/CSS/HTML assets were not scanned (`--format python,rust`); docs/ and tests/fixtures/ are out of scope.
- No member test was run beyond one torch-free probe: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/data/test_sources_metrics.py tests/monitor/test_sink.py tests/monitor/test_game_record.py tests/monitor/test_rotation_on_resume.py tests/config/test_regime_parity.py tests/config/test_regime_parity_p2.py tests/config/test_every_key_has_consumer.py tests/config/test_every_key_has_consumer_p2.py` → 66 passed, 1 failed (test_game_record.py::test_the_production_pool_is_BUILT_with_a_real_recorder: `mantis.run` imports torch). Every other witness is torch-dependent or a cargo test not run under the shared CPU budget.
- R8 crossings are flagged only where the arithmetic is obvious; each lane-B merge must re-derive `wc -l` on every touched file.

## Review
reviewer: fresh read-only agent (not the author); probes in throwaway worktrees, removed
method: own tools, not the scout's — scratchpad/rv-ldup/idx.py (every top-level def/class and method, docstrings
stripped, own name blanked, `ast.unparse` sha1) + q.py/d.py (group, PZ-filter via PZ.md glob list, Δ = −(Σspan − max
span per exact group) + one import per touched file + stated header); rs.py (Rust fn brace-match, comments and
whitespace stripped, fn name blanked, difflib ratio vs the largest copy); sinks.py (own spy-sink census: emit body,
`named` body, `__init__`). MATCH = identical after that normalisation; NEAR = ratio stated.

| ID | verdict | lane | Δlines | note |
|---|---|---|---|---|
| 01 | CONFIRMED | C | −332 (re-derived) | NEAR 0.95–1.00: batch literal, a `res` temp, layout; no member has `mod common;` today |
| 02 | CONFIRMED | C | −36 | MATCH ×8 incl. common/mod.rs (private `fn`, needs `pub`) |
| 03 | CONFIRMED | C | −58 | `search` NEAR 0.978 (c_scale + 2 literals); stub_policy, r8_board MATCH |
| 04 | CONFIRMED | B | −18 | NEAR 0.84; semantically `from_stones(s, p, mr, len, None)` for non-empty s; fn is 21 lines (730–750) |
| 05 | CONFIRMED | C | −20 | MATCH: same 10-kwarg map ×3; no arch code moves (search_kind, validator already in schema) |
| 06 | CONFIRMED | C | −8 | MATCH (4 fields); pipeline.py already imports 3 `mantis.config.resolve.*` modules |
| 07 | CONFIRMED | B | −12 | MATCH, 17 lines identical |
| 08 | CONFIRMED | B | −9 | NEAR 0.94: param name, regex constant, docstring |
| 09 | AMENDED | B | −11 | NEAR 0.66/0.81; `closed_shards()[0][0]` is not behaviour-exact (see note) |
| 10 | CONFIRMED | B | −10 | MATCH |
| 11 | AMENDED | C | −3 | lane B→C: src/mantis/encoding/__init__.py is PZ (src/mantis/encoding/**) and the card names the paths |
| 12 | AMENDED | C | −305 (re-derived; −313 if 8 existing `_drivable` imports extend) | clone and Δ hold; lane B→C: open card names the subject (BRIEF: card-named → C) |
| 13 | CONFIRMED | B | −298 (re-derived) | superset, not MATCH: 4 variants use `e['event']`, 10 have no `named`; home is NEW in tests/_drivable.py |
| 14 | CONFIRMED | B | −260 (re-derived) | 6 MATCH groups as stated; PZ member test_drawrate_abort_threading excluded |
| 15 | AMENDED | B | −101 (was −145) | only exact groups count; the ≥0.95 "near" copies carry different timeouts / process classes |
| 16 | CONFIRMED | B | −157 (re-derived) | 17 exact non-PZ members; see NEW-1 |
| 17 | AMENDED | B | −69 (was −77) | test_train_import_dag.py::_top_level_imports is NEAR 0.95, not a member |
| 18 | PENDING-PROBE | A | −19 (probe; scout −17) | torch-bound: both member modules error at collection at HEAD |
| 19 | CONFIRMED | B | −15 | MATCH |
| 20 | CONFIRMED | B | −32 | MATCH (28/29, 6/8) |
| 21 | CONFIRMED | B | −64 (re-derived) | MATCH ×5 pairs; all 3 modules already import torch, so the torch-importing home adds no coupling |
| 22 | CONFIRMED | B | −26 | MATCH (13/13/15, 6/7) |
| 23 | CONFIRMED | B (test-floor move) | −28 | MATCH ×4; both copies import the resolver from tests/config/_consumer_resolver.py, so it is one check; floor file reads 4862 |
| 24 | CONFIRMED | B | −21 | closures differ by 0 lines inside the span; third is the inline form |
| 25 | CONFIRMED | B | −83 (re-derived) | 11 MATCH pairs |
| 26 | CONFIRMED | B | −37 | 4 MATCH groups |
| 27 | CONFIRMED | B | −52 | 5 MATCH groups; the child-parity oracle shares only its stub net/board |
| 28 | AMENDED | B | −50 | 9 of 10 pairs MATCH; `_receipt_everything`/`_receipt_run_dir` NEAR 0.99 |
| 29 | CONFIRMED | C | −63 (re-derived) | 6 MATCH groups (_Opening ×4, deterministic_algorithms, drawrate trio, _graph_step) |

### Per-finding notes
S-L-DUP-01 — CONFIRMED: `rs.py` over the 11 → ratios 0.949–1.000 vs target_wire_carry.rs; the diffs are the pop batch literal (8/4/LEAF_BATCH/PROD_LEAF_BATCH), `let res =` vs a direct push, and a trailing comma. `grep -c "^mod common;"` → 0 in all 11, so 395 − 40 + 11 + 11 + 1 = −332 holds. Not an oracle; PZ glob crates/mantis-selfplay/tests/**.
S-L-DUP-02 — CONFIRMED: `rs.py` → one normalised hash (49ec7a3e) for all 8, common/mod.rs included.
S-L-DUP-03 — CONFIRMED: `rs.py` → `search` 0.978 (the `c_scale` param vs 50.0 and 0.1 literals); stub_policy and r8_board identical. PZ glob crates/mantis-search/tests/**.
S-L-DUP-04 — CONFIRMED: from_stones guards the bbox with `if !stones.is_empty()` and sets `last_move`; fwm_board does neither. It equals `from_stones(.., None)` only for non-empty input, as the scout says. The crate's own `from_stones_tests` are `cfg(all(test, feature = "test-fixtures"))`, which confirms the cfg gate is the only barrier. Δ: 21-line fn → 3 = −18.
S-L-DUP-05 — CONFIRMED: `git grep -A13 "derived_hexg_visit_capacity("` → the same 10 kwargs in schema/core.py, run.py and worker_sweep.py. DUP question: `search_kind` is a search knob, not a model arch, and the validator already sits in the schema, so nothing arch-specific moves. tests/test_run_buffer_route.py::_derived stays as the oracle (K-D17 spot-check below).
S-L-DUP-06 — CONFIRMED: idx key a658a5899e shared by DrainCapsSpec and DrainCaps; `grep "^from mantis.config" eval/pipeline.py` → eval_posture, fused_graph_caps, inference_batching.
S-L-DUP-07 — CONFIRMED: `diff` of lines 225–241 against 490–506 (indent stripped) → identical.
S-L-DUP-08 — CONFIRMED: `diff` of the two `sed` ranges → param name (`record_dir` vs `log_dir`), `_SHARD_RE` vs `_SEGMENT_RE`, docstring.
S-L-DUP-09 — AMENDED (claim): first_closed_shard returns at the first `shard_closed` row and never parses `bytes`. closed_shards parses `int(row.get("bytes", -1))` on EVERY row, so a malformed `bytes` field raises ValueError/TypeError where the halt read returns a path today. A shared reader needs that edge settled (or a reader that yields rows lazily). Lane B unchanged.
S-L-DUP-10 — CONFIRMED: idx key a9ebb57faf for both; the test already imports from mantis.train.lifecycle.signals (line 33).
S-L-DUP-11 — AMENDED: `git grep "hashlib.sha256(.*read_bytes())" -- src tools` → the 4 src sites plus tools/strix_follower.py, tools/select_balanced_book.py and tools/ci_gates/preflight_mint_parent.py (PZ), none of them card-named. Lane C for two reasons: src/mantis/encoding/__init__.py matches the PZ glob src/mantis/encoding/** (registry handshake, gate 8), and CARD-MECHANISM-SWEEP names the paths. worker_sweep.py::_sha256 keeps its OSError→None wrapper.
S-L-DUP-12 — AMENDED (lane only): `q.py` → exactly the 11 exact groups and 27 members named; d.py → 326 removed over 21 files → −305. Per BRIEF ("a ruling or open card names the symbol/path" → C), CARD-MECHANISM-SWEEP ("the 21 remaining private `_Pool`/`_Buffer` fakes") makes this C. The card counts 21 fakes, the scout counts 74 classes. The dispatcher may keep B if card-ordered work is exempt.
S-L-DUP-13 — CONFIRMED: `sinks.py` → 40 non-PZ top-level classes over 40 files, 352 lines; −352 + 14 + 40 = −298. Emit bodies: 39 `dict(event|payload)` plus 1 raw append (test_eval_round_observability, correctly excluded). Two amendments to the wording:
- `named` has 3 forms: 26 `.get('event')`, 4 `e['event']` (KeyError on an event without the key; the superset silently skips it) and 10 absent.
- tests/_drivable.py holds NO SpyEventSink today (`grep -n "class SpyEventSink" tests/_drivable.py` → none). The superset lives in tests/{train,monitor}/conftest.py, which are 2 of the 40.
S-L-DUP-14 — CONFIRMED: idx groups `_RunnerStats` ×17 (16 + PZ), `_filled_hexg` ×13 (12 + PZ), `_fake_run_safety` ×5, `_mirrored` ×4, `_Trainer` ×3 and ×2. test_cluster_stat_wiring.py::_RunnerStats is NEAR 0.96 and correctly not listed. d.py → 282 removed over 22 files → −260.
S-L-DUP-15 — AMENDED (Δ): the exact groups are `_promotion_hooks` ×4, `fake_mp` ×3, `_bounded` ×3, `_tiny_model` 2+2, FakeClock ×2 and `_FakeCtx` ×2 (reason_routes, terminal_eval_rc) → 111 removed, 7 files, −101. The "≥0.95" copies are not clones:
- `_eval_cfg` defaults differ in `round_timeout_sec` (0.3 / 5.0 / 0.05) and `worker_kill_grace_sec`, which are the values the timeout-bound suites exist to set.
- escalate's `_FakeCtx` builds `_RealisticFakeProcess`.
- The `_FakeProcess` copies differ in their `join` signature.
Merging those needs parameters (lane B, not counted).
S-L-DUP-16 — CONFIRMED: 17 exact non-PZ members; PZ test_checkpoint_conformance and test_resume_wiring_integration fall in the same exact groups and are excluded as stated. d.py → −157.
S-L-DUP-17 — AMENDED (Δ): exact groups `_code_text` ×4, `_called_name` ×3, and pairs of `_top_level_imports`, `_enclosing_defs`, `_call_sites` and `_root_name` → 15 members, 8 files, 80 removed → −69. tests/train/test_train_import_dag.py::_top_level_imports is NEAR 0.95 vs test_monitor_census.py.
S-L-DUP-18 — PENDING-PROBE (torch-bound), probe in scratchpad/wt/rv-ldup-18:
- Edit: deleted both fixtures + `import signal` (partial).
- `python -S -c "import mantis"` → ok. `pytest --collect-only -q -m ''` → "2289 tests collected, 167 errors" before and after; ERROR lines identical (`diff` of `grep ^ERROR`). Both member modules are among the 167: they import mantis.run → torch.
- `ruff check` on both → clean; r8_header_gate → 0 stale (partial goes 317 → 307, keeps its header); `cargo check --workspace --all-targets --locked` → Finished.
- `git diff --stat` → 19 deletions (10 + 9, blank lines included).
- Reasoning: the root autouse `_restore_signal_dispositions` is function-scoped and outer (conftest before module autouse), so the module copies are strictly nested. Zero effect is sound, but no member test can run here.
S-L-DUP-19 — CONFIRMED: idx key 9956133785 (8/7 lines).
S-L-DUP-20 — CONFIRMED: idx keys bdb789096b and 0164c2034a; 34 removed + 2 imports = −32. With -18, partial drops under 300, so its header goes.
S-L-DUP-21 — CONFIRMED: 5 pair keys; d.py −64. The home tests/selfplay/_fused_graph_harness.py imports torch, and all three members already do (grep).
S-L-DUP-22 — CONFIRMED: keys 4b4eccd978 (×3) and 8cd17886db (×2). The drain-parity oracle (PZ-6 push_dense) is untouched. No ruling or card cites the member files (`git grep` over RULINGS/CARDS/STATE/LAWS → none).
S-L-DUP-23 — CONFIRMED: the name-blanked keys match for all 4 twins, and `_consumer_resolver` is imported by both files, so the p2 resolver self-test re-checks the same function. The contract doc's "mutation self-test in both copies" is test_bijection_bites_on_a_real_schema_mutation, present in both and untouched. test_count_floor.txt → 4862. Direction note: the p2 file calls itself the SC-A4 prereg oracle and a "rewrite of test_regime_parity.py"; deleting the originals' O9–O11 instead is equally valid and is the dispatcher's pick.
S-L-DUP-24 — CONFIRMED: `diff <(sed -n 83,105p) <(sed -n 203,225p)` → only line 23 differs, and it lies outside the closure. The third copy (lines 183–195) is the same serve step inline.
S-L-DUP-25/26/27 — CONFIRMED: q.py exact keys as listed. 25 → 103 removed / 20 files; 26 → 45 removed; 27 → 59 removed.
S-L-DUP-28 — AMENDED: 9 MATCH keys. tests/diagnostics/test_mirror_receipts.py::_receipt_everything vs tests/tools/test_preflight_start_halts.py::_receipt_run_dir is NEAR 0.99, not identical.
S-L-DUP-29 — CONFIRMED: 6 exact keys; 76 removed + 10 imports + 3 header = −63.

### Top-10 by Δ, re-derived (d.py / rs.py; copies removed minus imports and headers)
01 −332 ✓ · 12 −305 ✓ · 13 −298 ✓ · 14 −260 ✓ · 16 −157 ✓ · 15 −145 → **−101** · 25 −83 ✓ · 17 −77 → **−69** · 21 −64 ✓ · 29 −63 ✓.
Revised raised sum ≈ −2 196 (was ≈ −2 248; still an upper bound, and the import lines overlap).

### DELIBERATE spot-checks (4)
- K-D8 holds: queue_fuse_reserve_parity.rs line 12 reads "The reference is a LOAD-BEARING transcription".
- K-D16 holds, and it is not a clone at all: test_worker_sweep_reachability.py says check_import_dag "walks top-level statements only, and this walk must not".
- K-D17 holds: `_derived` is the expected value in 4 asserts against `_select_buffer(...).visit_capacity`.
- K-D23 holds: `rs.py` → drain_shutdown and search_seam_fatal producers score 0.75/0.79 against the canonical; they add `after: ThenDo, parked: Arc<AtomicBool>` and mock inference.

### Overlaps with area scouts (same subject; for the cross-check, not double-counting)
01/02 ↔ S-A-RUST-2-09, -10, -17, S-A-RUST-3-18 · 03 ↔ S-A-RUST-1-18 · 09 ↔ S-A-TOOLS-2-11 · 10 ↔ S-A-TESTS-7-05 ·
11 ↔ S-A-CORE-1-20/-21/-32, S-A-CORE-3-14, S-A-TOOLS-2-12/-18 · 12/14/25 ↔ S-A-TESTS-7-01, S-A-TESTS-1-05, S-A-TESTS-2-03/-07,
S-A-TESTS-4-21 · 13 ↔ S-A-TESTS-1-06, S-A-TESTS-2-01/-04, S-A-TESTS-5-21, S-A-TESTS-6-02, S-A-TESTS-7-08 · 15 ↔ S-A-TESTS-6-02/-13 ·
16 ↔ S-A-TESTS-2-08, S-A-TESTS-4-20 · 17 ↔ S-A-TESTS-2-10, S-A-TESTS-3-06/-15, S-A-TESTS-7-04 · 18 ↔ S-A-TESTS-7-03, S-A-TESTS-2-16 ·
19 ↔ S-A-TESTS-2-16 · 20 ↔ S-A-TESTS-7-02 · 21 ↔ S-A-TESTS-5-04/-20, S-A-TESTS-8-05 · 23 ↔ S-A-TESTS-4-01/-02 · 24 ↔ S-A-TESTS-5-05 ·
26 ↔ S-A-TESTS-1-06, S-A-TESTS-6-02 · 27 ↔ S-A-TESTS-5-11, S-A-TESTS-6-04 · 29 ↔ S-A-TESTS-1-19, S-A-TESTS-8-11/-14.
The "cross-test imports are barred" premise is refuted independently in S-A-TESTS-1/-2/-4/-7 and REVIEW2 G3.4. Bare-name helper modules are live: `import _microbatch_harness` ×28, `from _drivable import` ×15, `import _fused_graph_harness` ×7. So no proposed home is blocked by R5.

### Missed by the scout
NEW-1 | DUP | B | tests/train/conftest.py::_make_{eval,selfplay,monitor}_block and tests/train/test_launch_path_smoke.py::_{eval,selfplay,inference,monitor}_block are NEAR 0.95–1.00 to the S-L-DUP-16 builders (q.py). Not counted; they ride -16 on contact.

### Tally: raised 29 | confirmed 22 | amended 6 (09, 11, 12, 15, 17, 28) | refuted 0 | pending 1 (18) | architect 0
