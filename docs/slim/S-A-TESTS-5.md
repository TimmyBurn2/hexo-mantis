# S-A-TESTS-5 — T5: tests/selfplay, tests/bridge, tests/arena
scope: tests/selfplay/** (57 files), tests/bridge/** (16), tests/arena/** (11) — 84 files, 13 578 lines;
method: full static read of every file; `.venv/bin/python -m pytest --collect-only -q -m '' -p no:cacheprovider tests/selfplay tests/bridge tests/arena`
(129 collected, 57 collection errors, all `No module named 'torch'`: only tests/bridge/*, test_selfplay_census, test_drain_row_shape_parity,
test_target_export_parity, test_selection_desync_is_named, test_books, test_book_geometry_pairing collect here);
`uvx ruff check --isolated --select F401,F811,F841`; `uvx vulture --min-confidence 60`; AST spans via a throwaway script
(`scratchpad/span.py`: `ast.parse` → node `lineno..end_lineno`, decorators included); `git grep -n -w` over the whole tree for every
DEAD claim; runtime `dir(mantis._engine)` vs an AST read of `src/mantis/_engine.pyi`.
Gate-3 context for every test deletion: CALLERS.md measures 5 112 collected vs floor 4 862 (tools/ci_gates/test_count_floor.txt);
no deletion below moves the floor file. None of the deleted tests is a tier_declaration.txt row unless stated.

## Summary
- Findings: 31. By class: DEAD 4 · TEST 13 · DUP 9 · ONE-SHOT 1 · SIMPLIFY 2 · DOC 2.
- By lane: A 5 (01, 02, 04, 05, 06) · B 22 · C 4 (08, 24, 25, 28).
- Top 3 by Δlines: S-A-TESTS-5-20 (≤ −104, InferenceServer fakes twin), S-A-TESTS-5-17 (−89, nine small subsumed tests),
  S-A-TESTS-5-08 (−76, lane C, tautological lifecycle "producer tests").
- Headline defect: two LAW-07 "producer tests" (test_lifecycle_events first_inference_*) execute no production code.

## Findings

### S-A-TESTS-5-01 | DEAD | A
subject: tests/selfplay/test_pool_surface.py::FILL_CAPACITY, ::FILL_ROWS, ::FILL_SELF_PLAY_PUSHED, ::TERMINAL_SCRIPT, ::EXPECTED_COMPOSITION (+ its unused `threading`, `numpy` imports, + the R8 header)
claim: a "captured deterministic fill … encoding v6" block no test reads; with it and two unused imports gone the file drops under 300 lines and must shed its R8 justification (which cites that very fill).
evidence: `git grep -n -w <each name>` → exactly 1 hit each (the definition); ruff F401 → test_pool_surface.py:17 `threading`, :21 `numpy`; `wc -l` → 314.
callers: AST imports: `git grep -w` whole tree → definition only · entry points/`-m`/subprocess/STATE: none name a test constant (CALLERS §1-6) · importlib/getattr: no string form (`git grep -n -E "[\"']EXPECTED_COMPOSITION[\"']"` → 0) · conftest: not a fixture name · pyo3/config keys/gate paths/box procedures: n/a (test-module constants; gate 15 reads only the header).
Δlines: −27 (lines 35–54 = 20 via `sed -n 35,54p`; imports 2; R8 header paragraph lines 4–8 = 5); file 314 → 287.
witness: gate 15 (r8_header_gate.py) reds if the header is left behind; tests/selfplay/test_pool_surface.py stays collectable only with torch.
depends: —

### S-A-TESTS-5-02 | DEAD | A
subject: tests/selfplay/test_inference_server.py::_NO_CUDA, ::_GPU_ONLY; tests/selfplay/test_arm8_reachable_paths.py::_a_board; tests/selfplay/test_graph_batch_dead_transfer_retired.py::_RETIRED_2; the `legal_mask` locals inside test_inference_server.py::_hand_built_batch and test_inference_batch_timing.py::_hand_built_batch
claim: defined, never read — a skip marker nobody applies, a board factory nobody calls, a retired-field name nobody asserts, and a mask built for a GraphBatch field retired by R298(d).
evidence: `git grep -n -w _GPU_ONLY` → 1 hit (definition); `_NO_CUDA` → 2 (definition + use inside `_GPU_ONLY`); `_a_board` → 1; `_RETIRED_2` → 1; vulture 60% flags `_a_board`; both `_hand_built_batch` bodies build `legal_mask` then pass no `legal_mask=` to `GraphBatch(...)` (read at test_inference_server.py L151-175, test_inference_batch_timing.py L121-144).
callers: AST: whole-tree `git grep -w` as above · `-m`/subprocess/STATE: none · getattr/strings: `git grep -n -E "[\"'](_GPU_ONLY|_a_board|_RETIRED_2)[\"']"` → 0 · conftest/fixtures: not parameter names (`git grep -n "def test_.*_a_board"` → 0) · pyo3/config/gates/box: n/a.
Δlines: −23 (_NO_CUDA+_GPU_ONLY 9, _a_board 4 by AST span; _RETIRED_2 comment+assignment 2; legal_mask 4+4 by `sed`).
witness: NONE (dead locals); tests stay green by construction.
depends: —

### S-A-TESTS-5-03 | DEAD | B
subject: unused imports in 12 slice files (ruff F401/F811, pyproject-ignored under tests/**)
claim: 16 whole import lines + 1 name import nothing used: test_arm8 `lookup`, test_buffer_facade `numpy` (+`BufferKindMismatch` name), test_fusion_counters `numpy`, test_game_complete_absence/_delivery/test_inference_local/test_lifecycle_events/test_target_export_parity `pytest`, test_graph_collate_masking_authority `textwrap` + `Trainer` + the in-function re-imports of `inspect`/`ragged_policy_ce`, test_inference_server `math`/`threading`/`mock`, test_pool_hparams_arms `lookup`.
evidence: `uvx ruff check --isolated --select F401,F811 tests/selfplay tests/bridge tests/arena` → 19 findings (test_pool_surface's 3 are in -01).
callers: an unused import has no caller by definition; import side effects: the only non-stdlib one is `mantis.train.trainer.core` (masking_authority), whose module is imported by the test's other imports anyway.
Δlines: −16 (whole lines; `sed -n <line>p` per ruff hit).
witness: NONE.
lane note: B, not A — pyproject.toml `[tool.ruff.lint.per-file-ignores] "tests/**"` deliberately admits F401/F811 ("edit-averse by frozen-oracle discipline"), so a sweep is a policy choice.
depends: —

### S-A-TESTS-5-04 | DEAD | A
subject: tests/selfplay/test_fusion_counters.py::_RStats and test_inference_batch_timing.py::_RStats fields `cluster_value_std_mean`, `cluster_policy_disagreement_mean`, `cluster_variance_sample_count`; tests/selfplay/test_game_complete_delivery.py::_ScriptedBuffer.push_many
claim: fake fields/methods for surfaces the tree no longer has (R10 "no compatibility shim for a state the tree no longer has"): production reads no `cluster_*` stat and has no `push_many`.
evidence: `git grep -n -E "\bcluster_value_std_mean\b" -- src/mantis` → 0 (crates: a comment only); same for the other two; `git grep -n -E "\bpush_many\b" -- src/mantis crates` → 0; tests/selfplay/test_pool_surface.py::test_snapshot_dataclass_field_sets_are_frozen already asserts `cluster_*` are GONE from RunnerStats.
callers: attribute reads by production of these names: `git grep -w` over src/ + crates/ → 0 · events.py::regime_gated_cluster_stats reads only `mcts_mean_root_concentration` (read L99-113) · getattr-by-string: `git grep -n "\"cluster_" -- src` → 0 · conftest/pyo3/config/gates/box: n/a.
Δlines: −9 (3+3 field lines; push_many def 2 + blank 1).
witness: tests/selfplay/test_inference_batch_timing.py::test_the_batching_block_reaches_the_sink_on_iteration_complete (drives emit_iteration_complete_event with the trimmed _RStats).
depends: — (the same fake fields exist out of slice: HANDOFF)

### S-A-TESTS-5-05 | SIMPLIFY | A
subject: tests/bridge/test_mcts_inference_roundtrip.py::test_mctstree_forced_root_child_round_trip (`hasattr(tree, "root_children_info")` arm)
claim: a compat branch for a method the bridge has never exported; collapse to `tree.get_root_children_info()[0][1]`.
evidence: `git grep -n "fn root_children_info\|fn get_root_children_info" -- crates/mantis-bridge/src` → only `get_root_children_info` (mcts.rs); runtime `hasattr(_engine.MCTSTree, "root_children_info")` is False (dir() compared in the stub check).
Δlines: −2 (3 lines → 1).
witness: the same test (runs torch-free).
depends: —

### S-A-TESTS-5-06 | TEST | A
subject: exact / verbatim-subset duplicate tests: test_collate_check_rewrites_parity.py::test_a_row_outside_the_global_range_still_raises_EdgeIndexOutOfBounds (≡ ::test_THE_PRECEDENCE_…, same array `[0,1,3,99,1,0,4,3]`, same raise); test_instrumentation_arms.py::test_g09_… (its one line is g10's first line); tests/bridge/test_graph_wire_adv.py::test_adv8_clean_input_passes (⊂ ::test_adv8_the_real_registry_dim_still_passes); test_fused_forward_seam.py::test_fg7_04_the_successful_parts_output_is_discarded_not_submitted (same payload/caps/net as ::test_fg7_04_a_mid_plan_failure…, whose `_assert_uniform_seam_failure` already asserts `results == []`); tests/bridge/test_gil_release_on_native_calls.py::test_a_concurrent_reader_is_NOT_refused… (same drive; `error is None` + `reads > 0` ⊂ OVERLAPS' `error is None` + `reads > 50`); test_check17_vectorized_parity.py::test_canonical_slot_helper_is_still_the_one_used (`callable()` of two imported names — vacuous; its docstring's claim is false, the reference never calls `_canonical_slot_vec`)
claim: deleting them loses no assertion.
evidence: side-by-side read (bodies quoted above); collection of the bridge ones is torch-free.
Δlines: −40 (AST spans 4+4+3+15+11+3) — note g09 is ALSO inside -07; count once.
witness: the surviving twin in each pair.
lane note: 6 collected tests fewer; 5 112 → 5 106 > floor 4 862, no floor file moves.
depends: —

### S-A-TESTS-5-07 | TEST | B
subject: tests/selfplay/test_instrumentation_arms.py::test_g06, g07, g08, g09, g10, g11, g14, g15, g15b (+ ::_TWO_CLUSTER_P1, 4 now-unused `_compute_*` import names)
claim: hand-ported copies of the old suite's pure-function vectors that the captured golden battery (tests/selfplay/test_instrumentation.py::test_pure_function_goldens, "the old test file's vectors verbatim") already pins exactly.
evidence: battery dump (`json.load(tests/fixtures/selfplay/instrumentation/pure_function_battery.json)`): two_stones_far col (2,2)=G-06; six_in_a_row_p1 ll ct5_wc1 (6,1.0)=G-07; seven_collinear_p1 (6,0.857…)=G-08; two_cluster_p1 nc ct5=2, ct19=1 =G-09/G-10; empty ll (0,0.0), nc 0, s5 (0,0)=G-11/G-14; stride5_chain_of_4 s5 (4,4)=G-15; three_adjacent_r0 s5 (1,3)=G-15b.
Δlines: −67 (AST spans 63 + 4 import-name lines).
witness: tests/selfplay/test_instrumentation.py (G-16) battery rows `empty`, `six_in_a_row_p1`, … .
lane note: −9 collected (−8 if -06 takes g09).
depends: S-A-TESTS-5-06 (g09)

### S-A-TESTS-5-08 | TEST | C
subject: tests/selfplay/test_lifecycle_events.py::test_first_inference_enqueued_emits_once, ::test_first_inference_served_emits_once (+ dead nested ::fake_start_thread)
claim: both "LAW-07 producer tests" re-implement the emit inside the test body on `InferenceServer.__new__(...)` and never call production code, so they cannot red; the real emit (inference_server.py, `"representation": "graph"`) has NO test — these assert their own `"dense"` payload.
evidence: read L196-269: the `if not srv._first_enqueued_emitted: … srv._sink.emit({…"dense"})` blocks are test code; `git grep -n "first_inference_enqueued" -- src` → inference_server.py emits `"representation": "graph"`; `git grep -l "first_inference" -- tests` → this file only.
Δlines: −76 (AST 35+37+4).
witness: NONE (that is the defect).
lane note: C — ruling-named: docs/governance/archive/rulings_register.md R223 accepts "tests/selfplay/test_lifecycle_events.py, 6 tests" as the lifecycle producer tests. Replacing them with a real drive is a LAW-07 act, not a deletion.
depends: —

### S-A-TESTS-5-09 | TEST | B
subject: tests/bridge/test_engine.py (whole file, 3 tests); tests/bridge/test_inv17_registryspec_retired.py::test_from_registry_unknown_raises
claim: test_surface.py already asserts the all_specs set (plus RegistrySpec types) and the registry_sha shapes (plus hex↔raw agreement); test_panic_exception.py's `RegistrySpec.from_registry` parametrize case asserts the same unknown-name ValueError with its message.
evidence: read both; residue not covered elsewhere: test_engine's `len(specs) == 2` duplicate-name guard (fold one line into test_surface) and `_engine.__doc__` (also checked by gate_01_fresh_sync.sh).
Δlines: −24 (`wc -l` test_engine.py 22 − 1 line moved; inv17 AST span 3).
witness: tests/bridge/test_surface.py::test_all_specs_binding_matches_registered_set, ::test_registry_sha_shapes; test_panic_exception.py::test_unknown_encoding_lookup_raises_a_NAMED_error[RegistrySpec.from_registry].
depends: —

### S-A-TESTS-5-10 | TEST | B
subject: tests/arena/test_regime_key.py::test_key_equality_is_full_tuple
claim: every assertion is made by ::test_canonical_roundtrip (`restored == key`) and ::test_any_field_change_changes_key (deploy_matched, model_sims changes ⇒ `!=`).
Δlines: −9 (AST span).
witness: the two surviving tests.
depends: —

### S-A-TESTS-5-11 | TEST | B
subject: tests/selfplay/test_fused_graph_caps_construction.py::_round_spec_base, ::test_fg6_08_the_round_spec_carries_the_caps_across_the_process_seam, ::test_fg6_08_a_grid_round_carries_none_across_the_same_seam, ::test_fg6_04_a_graph_engine_cannot_be_built_without_the_caps
claim: the two fg6_08 rows are the same round trip as tests/selfplay/test_inference_batching_threaded.py::test_the_round_spec_carries_the_batching_across_the_process_seam (`RoundSpec.to_dict` IS `dataclasses.asdict`, rounds.py L199-200; both assert `back == spec` with a FusedGraphCapsSpec, and the grid arm with both fields None); fg6_04 is implied by fg6_03's signature check and passes for the wrong reason (DEFECTS).
Δlines: −48 (scratchpad span.py total over the four named nodes).
witness: test_inference_batching_threaded.py round-trip test; test_fg6_03.
depends: — (tests/eval/test_eval_posture_inert.py carries a third RoundSpec round trip: HANDOFF T6)

### S-A-TESTS-5-12 | DUP | B
subject: tests/selfplay/test_pool_hparams_arms.py::_RecordingRunnerConfig, ::assemble, ::test_killed_knobs_are_never_read vs tests/selfplay/test_pool_hparams.py::RecordingRunnerConfig, ::record_runner_config, ::test_killed_and_relocated_fields_never_reach_the_runner
claim: the recording proxy + fixture is transcribed twice (identical `__init__/__setattr__/__getattr__` bodies); the killed-knob test is duplicated, and test_pool_hparams' own killed test is already implied by its golden set-equality (expected kwargs exclude NOT_CROSSING names).
deliberate?: not a seam or oracle — both drive the same `build_runner_config`; no twin-pin comment in either file.
Δlines: ≤ −42 (span.py total over the three arms-file nodes) before a shared helper's cost.
witness: test_pool_hparams.py::test_runner_config_assembly_golden.
depends: —

### S-A-TESTS-5-13 | ONE-SHOT | B
subject: tests/selfplay/test_radius_chain_removed.py (5 tests)
claim: a spent removal gate for the R25 radius_override chain (docstring still says "RED at HEAD (507c23b)"); it shells `grep -rn` over src/tests/crates on every run, and its positive control keeps `Board.set_legal_move_radius` alive with no Python caller.
evidence: `wc -l` → 70; `git grep -n -w set_legal_move_radius -- src tools tests '*.py'` → only this file (CALLERS §9 "test-only").
Δlines: −70.
witness: NONE after deletion (graves-style guard; the slimming decision is whether a spent grave stays).
depends: S-A-RUST-3 finding on `Board.set_legal_move_radius` (R3 slice; this file is its only Python-side reference).

### S-A-TESTS-5-14 | TEST | B
subject: tests/selfplay/test_check17_vectorized_parity.py::test_zero_legal_node_graph_parity
claim: drives only the test's own transcription `_check17_reference` ("only the reference is pinned here"), never the shipped `_check_semantic`: it asserts nothing about product code.
Δlines: −16 (AST span).
witness: NONE.
depends: —

### S-A-TESTS-5-15 | TEST | B
subject: tests/selfplay/test_inference_server.py::test_run_dispatches_to_the_graph_loop_for_a_graph_spec
claim: pins that `InferenceServer.run` forwards to `_run_graph_loop` (a one-line body, "One loop") and asserts "not the dense one" — no dense loop exists; every run()-driven row (e.g. ::test_finite_graph_output_submits_results) already fails if the forward breaks.
evidence: `sed -n 979,981p src/mantis/selfplay/inference_server.py` → `self._run_graph_loop()`.
Δlines: −7.
witness: ::test_finite_graph_output_submits_results.
depends: —

### S-A-TESTS-5-16 | TEST | B
subject: tests/selfplay/test_graph_batch_dead_transfer_retired.py::test_the_dead_device_tensor_is_gone_from_the_batch (second assert) and ::_RETIRED alias use; tests/selfplay/test_graph_collate_parity.py::_RETIRED_FIELDS alias
claim: `_RETIRED = "node_coords"` is a member of RETIRED_BATCH_FIELDS, so its assert repeats the loop above it; `_RETIRED_FIELDS = RETIRED_BATCH_FIELDS` is a pure rename. The test named "the_other_four" checks a seven-field tuple (DOC drift).
Δlines: −6 (assert 4 lines + alias 1 + its comment 1, `sed`).
witness: the RETIRED_BATCH_FIELDS loop.
depends: S-A-TESTS-5-02 (`_RETIRED_2` in the same file)

### S-A-TESTS-5-17 | TEST | B
subject: batch of small assertion-subsumed / tautological tests: test_selfplay_census.py::test_j01_census_covers_every_frozen_row (each parametrized J-01 row already asserts `func is not None`; the `>= 10` floor is on a literal in the same module); test_fused_forward_planner.py::test_fg2_01_near_cap_graphs_force_one_forward_each (= bank row "every graph exactly at the cap", whose `_reference_plan` equality fixes 8 parts); test_arm8_reachable_paths.py::test_every_construction_site_is_censused (implied by ::test_the_selfplay_path_threads_its_spec_explicitly); test_pool_encoding_resolve.py::test_resolved_kept_planes_match_indices_length (implied by the golden test's two equalities + a fixture self-property); test_collate_geometry_is_declared.py::test_every_graph_row_hands_out_a_complete_geometry (compares `geometry_kwargs(row)` to `spec_for(row)` — the helper against the spec it reads, a tautology)
claim: none adds an assertion about product code another test does not make.
Δlines: −54 (AST spans 18+7+3+13+13).
witness: the named survivors.
lane note: −5 collected (−6 counting the parametrized geometry row ×2 graph rows).
depends: —

### S-A-TESTS-5-18 | DUP | B
subject: tests/arena/test_match_fairness.py, test_battery_concurrency.py, test_ply_cap_adjudication.py (+ PZ twin tests/arena/test_legality_boundary.py): `_Opening`, `_board_factory`, `_regime_key`, `_FirstLegalBot`
claim: the same frozen-dataclass opening, board factory and RegimeKey builder are written four times; `_FirstLegalBot` twice.
deliberate?: no seam role — pure fixtures; legality_boundary's copy is PZ-1 (arena legality) and stays unless lane C agrees.
Δlines: ≤ −44 (AST: 11 per file ×3 + `_FirstLegalBot` 11) before a shared `tests/arena/_arena_stubs.py` (~15 lines).
witness: the arena suites.
depends: —

### S-A-TESTS-5-19 | DUP | B
subject: tests/selfplay/test_graph_collate_gather_order.py::_collate, test_graph_collate_edge_containment.py::_collate, tests/selfplay/_wire_geometry.py::spec_for
claim: two identical 7-line `_collate` helpers re-derive the four geometry kwargs from `lookup` instead of `_wire_geometry.geometry_kwargs` (the shared authority F-41 introduced); `spec_for` re-implements `mantis.encoding.registry.lookup` by scanning `all_specs()`.
Δlines: ≤ −24 (7+7 AST spans + spec_for 10).
witness: the collate suites (torch).
depends: —

### S-A-TESTS-5-20 | DUP | B
subject: tests/selfplay/test_inference_server.py::_cfg, ::_wire_for, ::_FakeGraphBatcher, ::_FiniteGraphNet, ::_hand_built_batch vs tests/selfplay/test_inference_batch_timing.py same five (and `_fused_graph_harness.py::ScriptedGraphBatcher/SentinelGraphNet/graph_cfg`)
claim: the InferenceServer drive rig exists twice near-verbatim (both hardcode dims 11/5); both files' R8 headers justify their size by "shared fakes".
deliberate?: no twin/oracle marker; the harness module already exists for exactly this.
Δlines: ≤ −104 (batch_timing's five helpers; AST total 149 minus the telemetry fakes 45 counted in -21), less the variant parameters a shared batcher needs.
witness: both suites (torch).
depends: S-A-TESTS-5-02, -21

### S-A-TESTS-5-21 | DUP | B
subject: tests/selfplay/test_fusion_counters.py::_ListSink, ::_TelemetryPool, ::_Buffer, ::_RStats, ::_emit vs tests/selfplay/test_inference_batch_timing.py same five
claim: identical telemetry fakes; test_fusion_counters drops from 329 to 284 lines and must shed its R8 header.
Δlines: −48 (AST 45 + R8 header 3).
witness: both suites' iteration_complete rows.
depends: S-A-TESTS-5-04; S-L-DUP spy-sink row (tests/_drivable.py::SpyEventSink)

### S-A-TESTS-5-22 | DUP | B
subject: tests/selfplay/test_pool_lifecycle.py::_cfg, ::_graph_pool vs test_pool_surface.py::_cfg, ::_graph_pool (and test_search_stats_end_to_end.py::_cfg)
claim: a 30-line hand-built pool config and the pool factory repeated three times, differing only in `n_simulations`/`fast_sims`.
Δlines: ≤ −35 (lifecycle AST span 35).
witness: both suites (torch).
depends: —

### S-A-TESTS-5-23 | SIMPLIFY | B
subject: tests/selfplay/conftest.py::pure_function_battery
claim: its only consumer (test_instrumentation.py) already loads the same JSON at module level as `_BATTERY` (needed for parametrize ids), so the fixture is a second read of one file.
evidence: `git grep -l -w pure_function_battery -- tests | grep -v conftest` → test_instrumentation.py (+ manifest filename).
Δlines: −4 (AST span 4).
witness: test_instrumentation.py.
depends: —

### S-A-TESTS-5-24 | DUP | C
subject: private `_Pool` fakes: tests/selfplay/test_alpha_full_counter.py::_Pool ≡ test_game_id_push.py::_Pool (same 8 attributes); test_inference_seam_counter.py and test_target_law18_counters.py (both test `runner_stats` threading with the same `_Pool`, 52+65 lines — mergeable into one file)
claim: carded duplication.
Δlines: ≤ −14 (game_id_push `_Pool` AST span) plus a merge of the two runner_stats files (−3 `_Pool` + headers).
lane note: C — CARD-MECHANISM-SWEEP names "the 21 remaining private `_Pool`/`_Buffer` fakes", applied on contact (R367(a)); slice total of such fakes: 11 (`git grep -n -E "^class _(Pool|Buffer)\b"` in slice).
depends: —

### S-A-TESTS-5-25 | DUP | C
subject: the drain stub-pool harness: tests/selfplay/test_pool_drain_parity.py (PZ-6), test_pool_drain_arms.py (9 helpers, 108 lines), test_game_complete_delivery.py (6 helpers, 97), test_lifecycle_events.py::_make_drain_pool (+sink, 56)
claim: four copies of the ~40-attribute pool the drain body reads, all with `pool._is_graph = False` (the dense arm, dead in production since every spec is graph); drain_parity also carries a `RecordingBuffer.push_many` twin and a `call["_method"] in ("push_many","push_dense_many")` shim, and `ScriptedPool._WINNER_NAMES` that the drain never reads (it uses the module constant).
evidence: `git grep -n "_WINNER_NAMES" -- src/mantis/selfplay` → pool_drain.py module constant only; `git grep -n -w push_many -- src/mantis crates` → 0.
Δlines: ≤ −153 (delivery 97 + lifecycle 56 AST spans, if both consume drain_arms' builder).
lane note: C — PZ-6 B-15: `push_dense` is the drain-parity instrumentation oracle; flipping these drives to the graph arm re-bases the dense oracles. tests/selfplay/test_drain_row_shape_parity.py pins test_game_complete_delivery.py and test_lifecycle_events.py BY PATH (`_FAKE_FILES`), so a merge edits that list; test_lifecycle_events is R223-named.
depends: S-A-TESTS-5-08

### S-A-TESTS-5-26 | DUP | B
subject: tests/bridge/test_mcts_inference_roundtrip.py: the graph consumer loop (pop → uniform per-segment probs → submit) in ::test_mctstree_expand_and_backup_ls_graph_round_trip, ::test_inference_batcher_graph_mock_round_trip, ::test_inference_batcher_submit_graphs_and_wait
claim: one 16-line loop written three times in one file (a fourth near-copy in test_inference_server.py::test_wire_round_trips_to_assemble_and_completes).
Δlines: ≤ −32 (two of the three copies; consumer AST span 18 each minus call sites).
witness: the bridge roundtrip suite (torch-free).
depends: —

### S-A-TESTS-5-27 | TEST | B
subject: test-only pyo3 exports and the bridge/selfplay tests that are their only Python callers: free fns `mcts_pool_overflow_count`, `take_mcts_pool_overflow_count` (tests/bridge/test_surface.py::FREE_FNS only); methods `MCTSTree.get_policy`, `.forced_root_child`, `SelfPlayRunner.get_win_stats`, `.set_model_version`, `.max_sims_per_search` (PZ), `InferenceBatcher.policy_len_py`, `.representation_py`, `.graph_max_in_flight`, `.lock_recoveries`, `.check_graph_request`, `.completed_graph_games`, `.spawn_mock_graph_games`, `.has_pending_graph_requests`, `HexgBuffer.game_id_at`; and the explicit `InferenceBatcher(feature_len=…, policy_len=…)` ctor arm (tests/bridge/test_pyclass_roundtrips.py::test_mcts_and_inference_batcher_construct; production reaches it only from InferenceServer's non-graph ctor arm, HANDOFF C2)
claim: for each, the test is the export's only Python caller, so export and test go (or stay) together.
evidence: per-name `git grep -n -E "\b<name>\b" -- src tools | grep -v _engine.pyi | wc -l` → 0 for every name listed.
lane note: B — a design choice per export: `spawn_mock_graph_games`/`has_pending_graph_requests`/`completed_graph_games` are a deliberate mock-game test harness exported from Rust (they drive the queue for 5 test files incl. the integration seam smoke); `max_sims_per_search` is PZ-1 served-sims (tests/bridge/test_runner_derived_means.py) → C.
Δlines: not derived here — owned by the R3 slice; each removal deletes its list entry/test.
depends: S-A-RUST-3 (the bridge-export findings for these names; IDs not yet published when this was written)

### S-A-TESTS-5-28 | TEST | C
subject: run-named tests and by-name production pins: tests/arena/test_book_geometry_pairing.py::test_run6_is_not_one_of_them (subsumed by ::test_every_shipped_config_pairs_its_encoding_with_a_replayable_book, which covers configs/*.yaml with an exact inventory); tests/selfplay/test_qsigma_rescale_reaches_target.py::test_run8_and_run7_configs_build_different_targets_on_the_same_root; test_pool_encoding_bridge.py::_run5_dump (loads configs/run6.yaml); test_compile_trunk.py::test_the_schema_defaults_to_eager_through_the_one_loader and test_edge_geometry_checker_thread.py::test_the_schema_defaults_to_inline_through_the_one_loader (pin `run6.yaml` by name); test_drawrate_pooled_statistic.py::RUN5_* constants
claim: R10 ("no runN in a symbol, test, pin"; production pins are a census) violations; the first is also assertion-subsumed.
Δlines: −8 for test_run6_is_not_one_of_them (AST span); the rest are renames/census rewrites (≈ 0).
lane note: C — CARD-MECHANISM-SWEEP inventories "32 test functions carrying a runN token" and "RUN5 = …/run6.yaml"; test_qsigma is STATE.md's named witness (CALLERS §5); test_drawrate_pooled_statistic and test_edge_geometry_checker_thread are PZ-1 pins.
depends: —

### S-A-TESTS-5-29 | DOC | B
subject: stale oracle-era docstrings: tests/arena/test_regime_key.py, test_books.py, test_match_fairness.py ("RED-at-import until IMPL writes …"); tests/selfplay/test_pool_encoding_resolve.py, test_instrumentation.py ("RED at import until IMPL writes"); test_radius_chain_removed.py, test_pool_encoding_bridge.py ("RED at HEAD"); test_target_law18_counters.py, test_target_export_parity.py ("PRE-FIX status at HEAD"); _fused_graph_harness.py ("Written before the feature exists … the suites … go RED"); test_lifecycle_events.py ("inference_server.py, dense + graph")
claim: each describes a pre-implementation state the tree left long ago.
evidence: `grep -n "RED at\|RED-at-import\|PRE-FIX status\|Written before the feature exists\|dense + graph" tests/{selfplay,arena,bridge}/*.py | wc -l` → 12.
Δlines: ≥ −12 (hit lines; most sit in 2–4 line paragraphs).
lane note: B — CLAUDE.md comment rule is "applied ON CONTACT, never as a cleanup pass"; a sweep needs the slimming phase's explicit sanction. Gate-14 docstring ratchet only allows the fall.
depends: —

### S-A-TESTS-5-30 | DOC | B
subject: tests/bridge/test_engine_stub_twins_agree.py
claim: pins the two `_engine.pyi` copies against EACH OTHER, never against the runtime, so both copies carry the same drift: the stub declares `MY_STONE_PLANE`, `OPP_STONE_PLANE`, `MOVES_REMAINING_PLANE`, `PLY_PARITY_PLANE` (not exported), and omits `RegistrySpec.cluster_threshold`, `.cluster_window_size`, `SelfPlayRunner.inference_failures_total`, `.max_sims_per_search` (exported).
evidence: `.venv/bin/python` AST read of src/mantis/_engine.pyi vs `dir(mantis._engine)` → exactly those 4 pyi-only and 4 runtime-only names.
Δlines: −4 stub lines per copy for the retired constants (8 total); the missing members add lines (net ≈ 0).
lane note: B — the fix is a stub edit in two files (one under crates/mantis-bridge, R3) plus, optionally, a runtime arm in this test.
depends: S-A-RUST-3 (the wheel stub lives in crates/mantis-bridge/python/mantis/_engine.pyi)

### S-A-TESTS-5-31 | TEST | B
subject: tests/selfplay/test_buffer_facade.py::test_graph_arm_missing_getter_propagates; test_pool_surface.py::test_graph_pool_buffer_composition_is_nan_and_that_is_parity
claim: both pin a facade forwarder (`ReplayFacade.outcome_in_range_count`) whose only raw buffer never has the method, i.e. they pin that a dead forwarder raises AttributeError and that `draw_target_fraction` is permanently NaN. If C2 deletes the forwarder, these go with it.
evidence: `git grep -n outcome_in_range_count -- src tools crates` → buffers.py forwarder + pool_push.py call inside `except (AttributeError, TypeError)`; HexgBuffer has no such method (the test itself asserts it).
Δlines: ≈ −20 (AST spans of the two tests; not re-derived individually).
witness: NONE after the pair deletion.
depends: C2 slice finding on buffers.py::ReplayFacade.outcome_in_range_count (HANDOFF)

## DEFECTS
- tests/selfplay/test_lifecycle_events.py: `first_inference_enqueued` / `first_inference_served` have no real producer test — the two "producer tests" emit their own event (with `"dense"`; production emits `"graph"`) (-08).
- tests/selfplay/test_fused_graph_caps_construction.py::test_fg6_04 omits three required kwargs (`fused_graph_caps`, `inference_batching`, `max_in_flight`), so its TypeError would still fire if `fused_graph_caps` gained a default.
- tests/selfplay/test_collate_check_rewrites_parity.py::_wire docstring says "Geometry comes from the registry row, never from literals" but hardcodes `6 * 11`, `4 * 5`, `edge_attr[0::5]`.
- tests/selfplay/_fused_graph_harness.py: `NODE_FEAT_DIM = int(GRAPH_SPEC.node_feat_dim or 11)` / `or 5` — a silent literal fallback of the F-41 class the harness's neighbours were written to remove.
- `_engine.pyi` (both twins) drift from the runtime surface (4 phantom constants, 4 missing members) — test_engine_stub_twins_agree cannot see it (-30).
- tests/arena/test_books.py::test_book_v1_reproducible_from_minter_args never compares the minter's output with the packaged book_v1 file or its manifest sha — it only checks two runs agree (the v2 pool has the real check in tests/tools/test_mint_opening_book.py).
- cwd-relative config paths: tests/selfplay/test_qsigma_rescale_reaches_target.py (`Path("configs") / config`) and test_pool_encoding_bridge.py (`"configs/run6.yaml"`) red when pytest runs from outside the repo root; siblings use `Path(__file__).parents[2]`.
- tests/arena/test_match_fairness.py::test_argmax_only_no_temperature_token_in_arena_or_eval `continue`s past a missing package dir (vacuous if `src/mantis/arena` moves) and reads with `read_text()` without `encoding=` (gate-16 function-scope backlog; same pattern in tests/selfplay/conftest.py, test_selfplay_census.py, test_pool_surface.py, test_target_export_parity.py, test_game_complete_delivery.py, test_radius_chain_removed.py subprocess `text=True`).

## PARKED
- tests/bridge/test_graph_batch_dispatch_parity.py::_round_trip busy-polls `next_graph_batch` for up to 30 s per call; tests/selfplay/test_fused_forward_planner.py::test_fg1_05_the_plan_is_identical_over_repeated_calls runs 100 identical plans.

## HANDOFF
- C2 (src/mantis/selfplay): `buffers.py::ReplayFacade.outcome_in_range_count` forwards to a method no raw buffer has (pool_push.py::buffer_composition's draw_target_fraction is always NaN); `pool_drain.py` dense arm and `inference_server.py` non-graph ctor arm (`InferenceBatcher(feature_len=…, policy_len=…)`) unreachable (every spec is graph; `is_graph_representation` raises otherwise); `pool.py::WorkerPool._chain_len` written, never read; `buffers.py::BufferKind` has one member and duplicates `hparams.is_graph_representation`'s dispatch; `PoolDims` feat/chain always 0 on graph.
- T8: tests/model/test_pmask_gather_parity.py::payload_fields re-implements tests/selfplay/conftest.py::payload_fields.
- T4/T7: the dead `cluster_*` fake RunnerStats fields (-04) also sit in tests/config/test_coordinator_knobs_wiring.py, tests/config/test_drain_caps_wiring.py, tests/test_actor_sync_composition.py, tests/test_run_composition.py, tests/test_run_disk_guard_abort_rc.py.
- T6: tests/eval/test_eval_posture_inert.py carries a third RoundSpec process-seam round trip (-11).
- D1/C1 (PZ-6 B-15 "the two segment softmaxes, second UNRESOLVED"): tests/selfplay/test_graph_collate_parity.py::test_train_reads_the_selfplay_segment_softmax_and_never_a_copy asserts `mantis.train.losses.segment_softmax is graph_collate.segment_softmax` — the second is a re-export, not a copy.
- Dispatcher (CALLERS.md §9): "SelfPlayRunner.{inference_failures_total, max_sims_per_search} … all four are used from Python" — `max_sims_per_search` has no src/tools reader (`git grep -n max_sims_per_search -- src tools` → 0); its only Python caller is tests/bridge/test_runner_derived_means.py.
- R3: -13, -27, -30 depend on the bridge-export findings.

## Not covered
- No test was executed beyond collection: 57 of 84 slice files cannot import without torch, so every torch-side "subsumed/duplicate" claim is static; no deletion was probed.
- tests/selfplay/test_graph_collate_adv.py (PZ-1, 511 lines) was read structurally (defs, helpers, ADV-3/A-18 rows) — its overlap with test_collate_check_rewrites_parity and test_graph_collate_edge_containment (same error classes on other shapes) was judged not strictly duplicate and not raised.
- test_gnn_seam_smoke.py vs test_inference_server.py::test_wire_round_trips_to_assemble_and_completes overlap not evaluated; golden VALUES under tests/fixtures/selfplay were not re-validated (T9).
- Some Δlines for merges (-18..-22, -24..-26) are the derived span of the removable copy, not a net after the shared helper is written.
