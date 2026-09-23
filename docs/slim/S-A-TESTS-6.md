# S-A-TESTS-6 — T6: tests/eval, tests/bots, tests/diagnostics
scope: tests/eval/** (47 files), tests/bots/** (5), tests/diagnostics/** (19): 71 files, 14 668 lines, all `test_*.py` (no conftest, no helper module).
method: AST surveys over the slice (imports, test lists, byte-identical and near-identical helper bodies via `ast.dump` hashing and `difflib`, one-reference helper scan); `uvx ruff --isolated --select F401,F811,F841,F821` (throwaway); `git grep` over src/tools/docs/contracts/tier_declaration/producer_manifest; the tier census; the collection of the torch-free files. Tests were run only in isolation: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/bots tests/diagnostics/{test_f816_37_rate_bar,test_replay_ratio_producer,test_ring_audit,test_ring_reader,test_tactics,test_worker_sweep_reachability}.py` → 107 passed, 2 failed, 14 deselected. Both failures are `No module named 'torch'` in a subprocess import of worker_sweep, which is the baseline. `--collect-only -m ''` over the slice → 170 collected, 54 collection errors, all torch (`mantis.eval` imports torch), so tests/eval was read statically. Read against docs/slim/S-A-CORE-3.md: its IDs are cited in `depends:`.

## Summary
- Findings: 13. By class: TEST 7, DUP 3, DEAD 1, SIMPLIFY 1, DOC 1. By lane: A 3 (T6-09, T6-10, T6-13); B 6 (T6-02, T6-05, T6-06, T6-07, T6-08, T6-12); C 4 (T6-01, T6-03, T6-04, T6-11).
- Top 3 by Δlines: **T6-01 −1 212** (sealbot adapter tests, if CORE-3-11 is ruled), **T6-02 up to −480** (the eval-pipeline fake harness copied into 8 files), **T6-03 up to −198** (the RoundSpec/net builder copied into 7 files).
- One-shot diagnostics: none. S-A-CORE-3-18 rules `src/mantis/diagnostics/` KEEP ALL with a live caller for every module, so no diagnostics test goes with its subject. The one spent constant test in the slice, test_worker_sweep_determinism.py::test_the_bands_constant_is_SUPERSEDED_but_still_pinned_for_history, is already raised in S-A-CORE-3-04 and is not repeated here.
- Deleted eval subjects: no test in the slice imports or names `mantis.eval.{ladder,bt,channel_health}`. `git grep -n -E "ladder|channel_health|\bbt\b" -- tests/eval tests/bots tests/diagnostics` returns only worker_sweep's own "ladder" (worker counts) and prose. What survives of R362 is (a) the sealbot adapter's tests (T6-01), (b) `wr_sealbot` residue (T6-08, T6-13), and (c) the grid-arm pins (T6-11).
- Test-floor arithmetic (gate 3c, floor 4862, collected 5112 per CALLERS §0): the B/C test deletions below total −31 (T6-01) −3 (T6-05) −1 (T6-06) −3 (T6-07) −4 (T6-08) = −42 collected. That stays above the floor, but every one of them is a test-floor move under the brief.

## Findings

### S-A-TESTS-6-01 | TEST | C
subject: tests/bots/test_sealbot_adapter.py, tests/bots/test_sealbot_resolve.py, tests/bots/test_sealbot_vendored.py; plus the sealbot arms of tests/bots/test_protocol.py (`_KNOWN_KINDS`, `_ENV_KEYS`) and tests/eval/test_resolver_wiring.py (`test_sealbot_rung_model_sims_route_through_resolve_eval_model_sims`, the `"sealbot"` parametrize rows)
claim: The subject has no production caller. These tests go with the adapter if and only if S-A-CORE-3-11 is ruled. One test must NOT go with them: test_sealbot_adapter.py::test_no_sys_path_write_anywhere_under_src_tools_or_tests (+ `_sys_path_writes`) is the repo's only src+tools+tests LAW-17/R5 `sys.path` census, and it has to be relocated.
evidence: `sed -n 1038,1070p docs/design/repo_design.md` → R362(c) amendment item 3: "The sealbot ADAPTER … stays as a vendored opponent with no production caller; deleting it is a separate decision nobody has taken". src/mantis/eval/worker.py::_model_sims_for_kind raises `ValueError("… the sealbot rung was deleted by R362(c)…")` before resolve_bot can reach `_resolve_sealbot`; src/mantis/eval/pipeline.py passes `rung_jobs=[]`. `git grep -n -E "def test_.*sys_?path|no_sys_path" -- tests` → this test, plus two narrower tools-only censuses (tests/tools/test_preflight_mint.py, test_preflight_parent_census.py).
Δlines: −1 212 = −1 246 (`wc -l` of the 3 files: 408+249+589) + 34 relocated (AST spans: `_sys_path_writes` 21 + the test 13). Also −26 (AST span of the test_resolver_wiring sealbot test) and T6-06's −35. Collected: −32 (10+9+13 via `--collect-only -m ''`) +1 relocated = −31.
witness: gate 3c (floor); tools/ci_gates/tier_declaration.txt rows 22–24 (test_sealbot_vendored.py) turn STALE and must go in the same commit; gate 10, because docs/contracts/eval_instrument.md rows cite all three files plus test_protocol.py (a contract doc, PZ-2).
depends: S-A-CORE-3-11 (the adapter PACK, ruled-kept), S-A-CORE-3-10

### S-A-TESTS-6-02 | DUP | B
subject: the eval-pipeline fake-subprocess harness: `_tiny_model`, `_eval_cfg`, `_promotion_hooks`, `_pipeline_kwargs`, `_SpySink`, `_FakeProcess`, `_FakeCtx`, `fake_mp` (fixture), `_bounded`, `FakeClock`, `_InjectedCompletionError` in tests/eval/{test_escalate_join_timeout_bound, test_eval_broken, test_eval_broken_reason_routes, test_pipeline_isolation, test_round_completion_error, test_eval_mp_context_whitelist, test_eval_result_tmp_litter, test_round_end_to_end}.py
claim: One harness is copied into 8 files. The copies differ only in override values (round_timeout_sec 0.05/0.3/5.0, kill grace 0.05–1.0, drain caps 2.0 vs 5.0) and in annotation style. A shared `tests/eval/_pipeline_harness.py` taking `**overrides` would replace them, following the rootdir-imported helper pattern tests/train already uses (`_microbatch_harness`, CALLERS §8).
evidence: `ast.dump` hashing → byte-identical bodies (docstrings stripped) for `_promotion_hooks` ×4, `_SpySink` ×3, `fake_mp` ×3, `_bounded` ×3, `FakeClock` ×2, `_pipeline_kwargs` ×2, `_tiny_model` ×2+2. A `difflib` pass shows the remaining copies differ only in the values listed above.
deliberate?: Two comments claim a "house convention": test_round_completion_error.py "duplicated per suite by house convention rather than shared through a conftest", and test_escalate_join_timeout_bound.py's R8 header "the self-contained fixture copy the eval-suite house convention requires". test_eval_broken_reason_routes.py says "R5 bars cross-test imports". R5 bars `sys.path` writes and a `tests` package. It does not bar underscore helper modules, and 13 of those are imported by bare name today (CALLERS §8). No ruling or design doc states the convention (`git grep -n -i "house convention" -- docs` → only analyzer_design.md, which is about something else), so this is a design choice for the dispatcher and not a seam.
Δlines: up to −480 = 637 (the AST spans of the 11 helpers summed over the 8 files) − 157 (one kept copy, the maximum span per helper). The helper's import lines (≤ 3 per file) come off that. R8: test_escalate_join_timeout_bound.py (315 → 219), test_eval_broken_reason_routes.py (355 → 280) and test_eval_result_tmp_litter.py (330 → 292) fall under the cap and MUST drop their justification headers (gate 15). test_eval_broken.py (447 → 313) keeps its header. The four `_FakePipeline` copies (8/10/10/15 lines: test_eval_posture_inert, test_gate_fields_ride_the_round_complete_row, test_round_instruments_are_measured, test_strength_floor_verdict_on_the_routed_mapping) are a second, smaller family that could share the same module.
witness: the 8 files themselves (torch-dependent, so they cannot run here); gate 15.
depends: —

### S-A-TESTS-6-03 | DUP | C
subject: the real-round builders `_net` (×7) and `_round_spec` (×6) in tests/eval/{test_eval_concurrency_row, test_f816_37_instrument, test_game_record_eval_channel, test_gate_sequential, test_graph_round_encoding, test_strength_floor_refuses_the_round, test_snapshot_payload_keys}.py
claim: One RoundSpec/GateSpec/snapshot builder is re-typed in six files, and the tiny-GnnArch builder in seven. The variations are a single parameter each (concurrency, GameRecordTarget, sequential dict, encoding name, StrengthFloorSpec) plus the net's width, so a shared builder with keyword overrides would replace them.
evidence: `difflib` against test_f816_37_instrument.py::_round_spec → 18–43 changed lines per copy, all in parameter values, round_id strings and the rung_jobs stub.
deliberate?: No stated reason in any of the copies. Three of the files are PZ-1 pinning tests: f816_37 (1-in-1 / dump-on-fire), gate_sequential (gate statistics), strength_floor_refuses (strength_floor). Hence lane C. The other four files alone would be lane B.
Δlines: up to −198 = 245 (AST spans: `_round_spec` 170 + `_net` 75) − 47 (one kept copy of each).
witness: the 7 files (torch).
depends: S-A-TESTS-6-02 (same helper-module decision)

### S-A-TESTS-6-04 | TEST | C
subject: tests/eval/test_rung_seat_off_window.py (whole file); its `_RuleNet`, `_rule_logit`, `graph_engine`, `_board` copies from tests/eval/test_eval_selfplay_child_parity.py
claim: The file repeats child_parity::test_head_plays_an_off_window_move_against_random_bot. Both run the same `worker.build_candidate_player` → 8-ply loop against RandomBot → assert an off-window head move. Only the sims count (4 vs 1) and the fixture positions differ. Its one extra assertion, `_model_sims_for_kind(spec, "strix") == strix_model_sims`, is test_strix_rung_sims.py::test_the_strix_rung_tool_threads_its_sims_through_the_same_lookup. A sims-parametrized row in child_parity would carry the claim, and the stub net and engine fixture would stop being duplicated ("duplicated rather than imported, since cross-test imports are barred", which is the same misreading as in T6-02).
evidence: `ast.dump` near-dups: `_RuleNet` 24/17 lines, `graph_engine` 14/12, `_board` 7/6, `_rule_logit` 3/2. The two test bodies were compared side by side (`grep -A32 test_head_plays_an_off_window_move_against_random_bot`).
Δlines: −146 (`wc -l` of the file) + the parametrize row added to child_parity (≈ 3 lines). Collected: −2, or 0 if parametrized over the same two positions.
witness: docs/contracts/eval_instrument.md cites test_rung_seat_off_window.py in two claim rows, so gate 10 reds unless the contract doc changes in the same commit. That contract-doc contact (PZ-2) is why this is lane C.
depends: —

### S-A-TESTS-6-05 | TEST | B
subject: tests/eval/test_resolver_wiring.py::test_unknown_opponent_still_raises_after_extension, ::test_none_value_still_raises_for_every_known_opponent, ::test_random_floor_routes_through_resolver; plus the module docstring
claim: All three are assertion-subsumed. (a) The unknown-opponent test is tests/config/test_resolve_nsims.py::test_unknown_opponent_raises. (b) The None-for-every-known test repeats the `pytest.raises(ValueError)` arm of this file's own parametrized test_unknown_opponent_and_none_value_raise_pre_existing_green over the same two opponents. (c) test_random_floor_routes_through_resolver is the `[random]` row of test_every_bot_kind_routes_its_sims_through_the_resolver_after_the_rewrite: the same pass-through spy and the same `("random", N) in calls` assertion, 96 vs 128. The docstring is stale. It says "`mantis.bots` does not exist yet, so the whole file fails collection today", it names kraken, and it names the retired key `eval.sealbot_model_sims`.
evidence: the three tests were read side by side with the rows named above (`cat` of both files).
Δlines: −29 (AST spans 3+4+22) − the blank separators. Collected −3.
witness: tests/config/test_resolve_nsims.py; test_resolver_wiring.py's parametrized rows.
depends: —

### S-A-TESTS-6-06 | TEST | B
subject: tests/bots/test_protocol.py::test_external_kinds_carry_a_reason_that_names_no_env_key (+ `_ENV_KEYS`)
claim: The test is subsumed, and half of it cannot run. It parametrizes over `sorted(_ENV_KEYS)` == `["sealbot"]`, and the sealbot arm `return`s, so everything after the `if kind == "sealbot":` block never executes. The sealbot arm's assertions are covered, with the env var both set and unset, by test_sealbot_resolve.py::test_setting_the_deleted_env_key_changes_nothing, ::test_sealbot_refusal_reason_names_exactly_its_own_missing_step (`"MANTIS_BOT_" not in reason`) and ::test_no_mantis_bot_env_literal_survives_under_src.
evidence: `sed -n 18,97p tests/bots/test_protocol.py`; `sed -n 190,249p tests/bots/test_sealbot_resolve.py`.
Δlines: −35 (AST spans: test 32 + `_ENV_KEYS` 3). Collected −1. If T6-01 lands, this goes with it.
witness: tests/bots/test_sealbot_resolve.py (passes here); docs/contracts/eval_instrument.md cites the file (not the test), and the file stays.
depends: —

### S-A-TESTS-6-07 | TEST | B
subject: tests/eval/test_value_pool_guard.py::test_every_registered_encoding_declares_an_implemented_value_pool, ::test_run5_encoding_passes_the_value_channel_guard, ::test_policy_channel_refusal_is_unchanged_by_the_new_guard
claim: All three are subsumed. The first two follow from tests/eval/test_graph_round_encoding.py::test_the_decode_capability_set_is_closed_over_the_registry, which asserts that the full guard fires for NO registered spec (policy and value, `gnn_axis_v1` included). The third is the same synthesis as tests/eval/test_eval_selfplay_child_parity.py::test_no_drop_pooling_encoding_is_still_refused: gnn_axis_v1 with `policy_pool="legal_set_scatter_max"` and an implemented value pool (registry `value_pool = "none"`), refused, with the pool named in the message.
evidence: `sed -n 150,168p tests/eval/test_graph_round_encoding.py`; `sed -n 555,581p tests/eval/test_eval_selfplay_child_parity.py`; `grep -n value_pool crates/mantis-encoding/src/registry.toml` → "none" ×2.
Δlines: −23 (AST spans 10+3+10) − the blank separators. Collected −3. test_value_pool_guard.py survives (it is cited by eval_instrument.md).
witness: the two kept rows (torch).
depends: —

### S-A-TESTS-6-08 | TEST | B
subject: small subsumed rows: tests/eval/test_round_timeout_route.py::test_the_wire_spelling_round_trips_through_the_taxonomy; tests/eval/test_round_events.py::test_round_complete_wall_sec_feeds_the_routed_result_key; tests/diagnostics/test_tool_absences.py::test_the_shipped_plan_still_loads; the `wr_sealbot` residue asserts in tests/eval/test_eval_broken.py (`result.get("wr_sealbot") is None`) and test_pipeline_isolation.py (`"wr_sealbot" not in ack`)
claim: Each row asserts nothing another test does not.
- The wire-spelling row is the enum census loop in test_eval_broken_reason_enum.py (every member re-parses to itself).
- The wall_sec row calls the same emitter and reads the same key as test_round_emits_start_and_complete_wall_events (the manifest's producer_test, which stays). The routed-result claim in its name is never exercised.
- The shipped-plan row is test_worker_sweep_plan.py::test_the_committed_plan_loads_and_carries_the_operator_ladder, since `load_plan` itself enforces the A03 inequality.
- `wr_sealbot` has had no producer since R362(c). The two asserts are vacuous, and test_promote_reason_guard.py::_result still seeds the dead key.
evidence: the files were read side by side. `git grep -n wr_sealbot -- src/mantis/eval` → 0 hits.
Δlines: −21 (AST spans 3+14+4) −2 assert lines. Collected −3.
witness: the kept siblings named above.
depends: —

### S-A-TESTS-6-09 | DEAD | A
subject: tests/eval/test_graph_round_encoding.py::_openings_at
claim: A module-level helper that is never called and is not a fixture (no decorator), which leaves an orphaned monkeypatch of `worker.round_openings`.
evidence: `git grep -n -w _openings_at` → the def only; `git grep -n -E "[\"']_openings_at[\"']"` → 0.
callers: AST imports: nothing imports a test module, and the whole-tree `git grep -w` gives the def only · entry points / python -m / subprocess: n/a for a test-module private · importlib/getattr/string: `git grep -E "['\"]_openings_at['\"]"` → 0 · conftest/fixtures: not a fixture, so pytest cannot inject it, and no `def test_` takes it as a parameter · pyo3 / config keys / gate tool paths / STATE procedures: n/a. A search by the bare word covers every one of these channels for a test-module private.
Δlines: −27 (AST span 25 + 2 blank separators).
witness: NONE. The file still collects (torch).
depends: —

### S-A-TESTS-6-10 | SIMPLIFY | A
subject: stale "RED-at-import anchor" top-level imports and the per-test re-imports that shadow them in tests/eval/test_gate_parity.py (5), test_resolver_wiring.py (3) and test_round_events.py (4); unused imports in test_tool_absences.py (`json`), test_eval_broken.py (`json`, `time`, `ResultContractError`, unused local `ack`), test_eval_work_dir_run_id_scope.py (`SimpleNamespace`), test_round_end_to_end.py (`Any`), and the dead local `import torch` in `_tiny_model` of test_escalate_join_timeout_bound.py and test_round_completion_error.py
claim: The anchors' comments say "mantis.eval does not exist yet" / "mantis.bots does not exist yet", which is false at HEAD. Each test then re-imports the same name locally (F811). Keeping the top-level import, dropping the noqa comment and removing the local re-imports changes nothing.
evidence: `uvx ruff check --isolated --select F401,F811,F841,F821 --output-format concise tests/eval tests/bots tests/diagnostics` → 21 findings: F811 ×12, F401 ×8, F841 ×1.
Δlines: −32 = 11 re-import lines + their 11 blank followers, plus 10 unused-import lines (the two local `import torch` lines each carry a blank follower). The docstring "RED-at-import" sentences in test_aggregate_regime.py and test_round_events.py fall under T6-13.
witness: gate 14 (ruff at zero stays zero); the files' own tests (torch).
depends: —

### S-A-TESTS-6-11 | TEST | C
subject: grid-arm pins: tests/eval/test_leaf_build_threads_wiring.py::test_the_grid_arm_is_the_serial_width (and the `IfExp` assertion in ::test_run_py_threads_a_DERIVED_width_and_not_a_literal); the harness rows that run the pipeline on "the GRID arm" with the deleted encoding `"v6_live2_ls"` (16 sites in 8 tests/eval files); the `infer_fn=` exactly-one test in test_eval_selfplay_child_parity.py
claim: These tests pin a representation arm that no registered encoding can reach, and the harness names a registry row that R346(f) deleted. They move or go when the grid arm does.
evidence: `.venv/bin/python -c "from mantis.encoding import all_specs; …"` → [('gnn_axis_r8','graph'), ('gnn_axis_v1','graph')]; `git grep -c v6_live2_ls -- tests/eval` → 16; crates/mantis-encoding/tests/registry_census.rs names v6_live2_ls as deleted.
Δlines: −8 (AST span of the grid-arm test 7 + 1 separator). The harness re-stub is value-only (0 net).
witness: tests/eval/test_leaf_build_threads_wiring.py
depends: S-A-CORE-3-12 (the grid arm, lane C through src/mantis/run.py and eval/worker.py, both PZ); slice C2 owns `IdentityConfig.representation: Literal["grid","graph"]`

### S-A-TESTS-6-12 | DUP | B
subject: tests/diagnostics/test_worker_sweep_{authority,markers,verdicts}.py::plan (fixture) + each file's `_PLAN` path constant
claim: One fixture, `ws.load_plan(_PLAN)`, is byte-identical in three files. A tests/diagnostics/conftest.py would hold it once.
evidence: `ast.dump` hash b2a194ca9d is shared by all three.
deliberate?: none stated.
Δlines: −6 net (3×(2-line def + decorator + separator) = 12, less one kept copy of 4 + a conftest import of 2). A new conftest is a structural choice, hence lane B.
witness: the three files (torch).
depends: —

### S-A-TESTS-6-13 | DOC | A
subject: stale prose in slice tests:
- test_round_events.py docstring: `wr_sealbot` in the `eval_round_complete` payload; "RED-at-import: `mantis.eval.pipeline` does not exist yet".
- test_aggregate_regime.py docstring: "RED-at-import until IMPL writes `mantis.eval.aggregate`".
- test_eval_concurrency_row.py assertion message: cites `test_eval_rung_concurrency_row.py`.
- test_promote_call_site.py comment: "no promote.py yet … fail today".
- test_eval_child_memory.py: phase label `rung:sealbot_d5`.
claim: Each sentence describes a tree that no longer exists.
- The cited file test_eval_rung_concurrency_row.py does not exist (`git ls-files | grep -c test_eval_rung_concurrency_row` → 0).
- `eval.rung_concurrency` left the schema (R362(c)).
evidence: `git grep -n -i -E "RED-at-import|does not exist yet|sealbot" -- tests/eval`.
Δlines: −6 (prose lines removed or reworded, counted from the grep hits).
witness: gate 14's comment ratchet (may only fall).
depends: —

## DEFECTS
- tests/eval/test_deploy_matched_hparam_coincidence.py (PZ-1 deploy-matched): `_RUN5 = configs/run6.yaml`, while every test and message says "run5" (configs/run5.yaml does not exist). It is an R10 run-keyed pin read off ONE config instead of the census, and `_INFERENCE_KEYS` is an empty list kept in a "partition".
- R10 run-named tests: test_gate_parity.py::test_gate_truth_table_matches_run3; test_value_pool_guard.py::test_run5_encoding_passes_the_value_channel_guard; test_ring_audit.py::test_a_clean_ring_passes_the_run8_bands, ::test_the_prereg_carries_the_five_run8_bands; test_worker_sweep_authority.py produced_by labels "run5@…".
- tests/diagnostics/test_ring_audit.py opens `Path("docs/design/measurements/RUN8_PREREG_2026-09-17.md")` cwd-relative, so it reds when pytest is launched from any directory but the repo root.
- tests/eval/test_escalate_join_timeout_bound.py, test_eval_broken_reason_routes.py, test_pipeline_isolation.py and 5 more pass `encoding="v6_live2_ls"`, a registry row deleted by R346(f). The pipeline never looks it up, so the fixture is silently meaningless (see T6-11).
- tests/eval/test_round_timeout_route.py docstring calls test_eval_broken_reason_routes.py "frozen … OUT OF SCOPE of the grant". No freeze manifest exists at HEAD (PZ-3).

## PARKED
none

## HANDOFF
- T2 (tests/train h–z): tests/train/test_terminal_eval_rc.py carries a 9th copy of the T6-02 harness (`_FakeProcess`/`_FakeCtx`/`_pipeline_kwargs`/`_eval_cfg`) and a `"wr_sealbot": None` row.
- T3 (tests/tools): if T6-01 relocates the repo-wide `sys.path` census, check tests/tools/test_preflight_mint.py::test_no_sys_path_mutation_in_the_tool_or_its_tests for subsumption.
- T4 (tests/config): tests/config/test_resolve_nsims.py is the kept twin for T6-05. Its `test_sealbot_reads_config_value` rides on S-A-CORE-3-11.
- C3: src/mantis/eval/worker.py::_assert_policy_pool_implemented's message still describes the GRID arm (already inside S-A-CORE-3-12's subject).

## Not covered
- Per-assertion subsumption inside the worker_sweep suite (authority/determinism/markers/selection/verdicts/reachability, 2 351 lines): I read the headers and the helper duplication only.
- The bulk of test_eval_selfplay_child_parity.py (581 lines) beyond the rows named above.
- test_eval_posture_inert.py, test_acceptance_witness.py, test_fusion_calibrate_refusals.py / test_fusion_margin_is_measured.py, test_tactics.py, test_ring_reader.py, test_eval_child_memory.py vs test_eval_child_memory_reader.py, and test_mirror_receipts.py: headers and test lists only.
- None of the 54 torch-dependent files was executed (environment). No reviewer probe was run for the A-lane rows.

## Review
reviewer: fresh read-only agent (not the author); probes in throwaway worktrees, removed
| ID | verdict | lane | Δlines (probe-measured for lane A) | note |
|---|---|---|---|---|
| T6-01 | AMENDED | C | −1 210 (−1 246 + 36 relocated) | relocation also carries `_REPO` + `_SCAN_ROOTS`; gate 10 + tier census red measured |
| T6-02 | CONFIRMED | B | up to −480 | spans reproduced 637 / 157 |
| T6-03 | CONFIRMED | C | up to −198 | spans reproduced 170 + 75 − 47 |
| T6-04 | CONFIRMED | C | −146 + row | gate 10 red measured (2 rows); sims 4 vs 1 is a real condition, so a parametrize row is needed, not a bare delete |
| T6-05 | CONFIRMED | B | −29 | the "three rows above are HEAD's pins" comment goes stale with it |
| T6-06 | AMENDED | B | smaller, unmeasured | `"env key" not in reason` is asserted ONLY here; the dead part is the post-`return` branch |
| T6-07 | CONFIRMED | B | −23 | |
| T6-08 | AMENDED | B | −21 −3 asserts | a third vacuous `wr_sealbot` assert: test_round_completion_error.py |
| T6-09 | PENDING-PROBE | A | −27 | torch-bound |
| T6-10 | AMENDED | A −10 (pending) / B −22 | −10 (A) | F811 half → B: pyproject declares "the deliberate re-import idiom"; gate 14 is not a witness |
| T6-11 | CONFIRMED | C | −8 | |
| T6-12 | CONFIRMED | B | ≈ −6 | |
| T6-13 | PENDING-PROBE | A | −6 | torch-bound; `rung:sealbot_d5` is test data, not prose (drop from subject) |

### Per-finding notes
Probe recipe (REVIEW_BRIEF, `-S` + isolating PYTHONPATH). Worktree rev-tests6-A carries T6-09, T6-10 and T6-13 edits: `import mantis` ok. Full-tree `--collect-only -m ''`: HEAD 2289 collected / 167 errors, probe 2289 / 167, `diff` of the sorted ERROR lines empty. All 13 edited files `ast.parse` clean. `uvx ruff --isolated --select F401,F811,F841,F821,E9` over them: clean. gate 15: 0 stale. gate 10: rc 0. comment_lint: GREEN, but the floors FALL (docstring_excess 13079→13073, private_docstring_excess 1485→1481), so tools/ci_gates/comment_length_floor.txt moves in the landing commit. cargo check was not run because no probe touched Rust. Worktree rev-tests6-C `git rm`s the 3 sealbot test files + test_rung_seat_off_window.py.
S-A-TESTS-6-01 — AMENDED:
- Uniqueness check: `git grep -n -E "sys\.path" -- tests tools` returns 2 code censuses besides this one, test_preflight_mint.py and test_preflight_parent_census.py, both tool-scoped. No gate script checks sys.path. So the adapter-file test is the ONLY src+tools+tests LAW-17 census, and it must move.
- Relocation cost: it needs `_REPO` and `_SCAN_ROOTS` too, which adds 2 lines to the 34.
- Probe rev-tests6-C: check_tracked_refs.py rc 1 on eval_instrument.md (lines citing test_sealbot_{adapter,resolve,vendored}.py). tier_census.py: "0 undeclared, 3 stale" (tier_declaration.txt rows 22–24). Collected 2289→2257 (−32).
- Lane C stands (R362(c) item 3: "deleting it is a separate decision nobody has taken").
S-A-TESTS-6-02 — CONFIRMED: an AST span/hash script reproduces sum 637, kept 157, net −480. `_eval_cfg` has 7 distinct bodies across 8 files, so the copies do differ by values, as claimed.
- docs/audits/REVIEW2_2026-09-21.md G3.4 already finds "cross-test imports are evidently not barred". That supports the scout's reading of the "house convention" comments.
- Also weigh pyproject.toml's tests/** grounds ("an oracle-write corpus: files are edit-averse"). Lane B.
S-A-TESTS-6-03 — CONFIRMED: `_round_spec` 6 copies / 170 lines and `_net` 7 / 75 reproduced. PZ.md rows (1-in-1, gate pair statistics, strength_floor) name test_f816_37_instrument, test_gate_sequential and test_strength_floor_*, so lane C.
S-A-TESTS-6-04 — CONFIRMED (C):
- Both bodies use `build_candidate_player` → 8-ply loop against RandomBot → a non-empty off-window list. The sims differ (4 vs 1), which is a search-depth condition, so the merge must be a parametrize row, not a plain delete.
- The rung-sims assert maps to test_strix_rung_sims.py::test_the_strix_rung_tool_threads_its_sims_through_the_same_lookup (`_model_sims_for_kind(spec,"strix") == 256`).
- Probe: gate 10 reds on eval_instrument.md, two rows for this file.
- The path is also cited in docs/governance/archive/{RULINGS_ACTIVE,rulings_register}.md. Those files are frozen and gate 10 does not scan them, so there is nothing to do there.
S-A-TESTS-6-05 — CONFIRMED:
- (a) test_resolve_nsims.py::test_unknown_opponent_raises: `"mystery"` vs `"nnue"`, same ValueError arm.
- (b) the parametrized pre_existing_green covers `pytest.raises(ValueError)` for random and sealbot with None.
- (c) the `[random]` row of test_every_bot_kind… is the same pass-through spy and `(kind,128) in calls`.
- No external citation of the 3 names (`git grep -l <name> -- ':!docs/slim'` → the file only). Torch-free, so if it lands the file can be run here.
S-A-TESTS-6-06 — AMENDED:
- `_ENV_KEYS == {"sealbot": …}`, so the block after `return` is unreachable. That part is right.
- The subsumption claim is wrong. `grep -n "env key" tests/bots/*.py` finds the `"env key" not in reason` assertion ONLY in test_protocol.py. test_sealbot_resolve checks `"MANTIS_BOT_"` alone, and src/mantis/bots/protocol.py still says "env key" in prose, so the regression is plausible.
- Correct scope: delete the unreachable branch. The whole test goes only if the "env key" check first moves into test_sealbot_refusal_reason_names_exactly_its_own_missing_step.
S-A-TESTS-6-07 — CONFIRMED:
- The worker guard is `_assert_policy_pool_implemented` then `_assert_value_pool_implemented`, so the empty-fire census covers both census rows.
- child_parity's synthesis (value_pool "none", implemented) vs this file's "min" (implemented): both reach only the policy message. The claimed ORDERING is exercised by neither, which is not a loss.
S-A-TESTS-6-08 — AMENDED:
- The wire spelling is covered by test_eval_broken_reason_enum.py:83 (`EvalBrokenReason(member.value) is member`), together with this file's own `.value == "round_timeout"`.
- wall_sec is covered by the sibling's `complete["wall_sec"] == 12.5`.
- The shipped-plan row is covered by src/mantis/diagnostics/worker_sweep.py `if sampler_interval_sec >= round_sec` inside load_plan, plus test_worker_sweep_plan loading the same toml.
- MISSED: tests/eval/test_round_completion_error.py also asserts `result.get("wr_sealbot") is None`, so there are 3 vacuous sites, not 2.
- `git grep -c wr_sealbot -- src tools` → tools/dashboard/{strength,tier2,tier3}.py still read it (historical run records). The test asserts stay vacuous because there is no producer.
S-A-TESTS-6-09 — PENDING-PROBE (torch-bound):
- `git grep -n -w _openings_at` → the def only. The AST shows no decorator, and no test takes it as a parameter.
- Probe: deleted lines 89–115, diff −27. Every collect/gate step is green.
- The file errors at collection at HEAD (torch), so its tests cannot be shown green.
S-A-TESTS-6-10 — AMENDED:
- The ruff re-run reproduces 21 findings.
- Probe: the F401 half (8 lines + 2 blank followers) = −10, the F841 `ack` edit = 0 net, and the F811 half = −22 (11 re-imports + 11 blanks; the 3 anchor lines only lose their `# noqa … does not exist yet` text).
- tests/eval/test_resolver_wiring.py after the F811 edit: `8 passed`. The other edited files are torch-bound, so the lane-A half is PENDING.
- The F811 half moves to lane B. pyproject.toml `[tool.ruff.lint.per-file-ignores] "tests/**"` ignores F401/F811/F841 and names "the deliberate re-import idiom", so which import to keep is a design choice. For the same reason the scout's witness (gate 14) is REFUTED: ruff does not check these codes over tests/.
S-A-TESTS-6-11 — CONFIRMED:
- test_the_grid_arm_is_the_serial_width asserts `leaf_build_threads` is an IfExp with orelse 1 in run.py.
- `git grep -c v6_live2_ls -- tests/eval` → 16 in 8 files.
- Lane C follows S-A-CORE-3-12.
S-A-TESTS-6-12 — CONFIRMED: the AST body hash of `plan` is identical in authority, markers and verdicts (3-line span each). A new conftest is structural, so lane B.
S-A-TESTS-6-13 — PENDING-PROBE (torch-bound):
- Probe edits: test_round_events docstring drops `wr_sealbot` and the RED-at-import line + blank (−2); test_aggregate_regime clause (0); the test_eval_concurrency_row message (−1); test_promote_call_site comment 4→1 (−3). Net −6, and the collect/gate steps are green.
- `git grep rung_concurrency -- src`: only the RoundSpec field and checkpoints.py's retired-key list. So the message's config-key and file citations are stale, as claimed.
- AMEND: the test_eval_child_memory.py `rung:sealbot_d5` hits are a phase-label tuple the test iterates, i.e. data, not prose. Remove them from this DOC subject.

### Missed by the scout (optional, max 5)
- NEW-1 | DOC | B: the stale "house convention" / "cross-test imports are barred" reasons in tests/eval/test_escalate_join_timeout_bound.py (R8 header + fixture comment), test_round_completion_error.py and test_eval_broken_reason_routes.py (R8 header). This is the class REVIEW2 G3.4 rated should-fix, and gate 15 accepts a false reason. Lands with T6-02. Out-of-slice siblings in tests/config → T4.
- NEW-2 | DOC | C: pyproject.toml's tests/** ruff and pyright exclusions are grounded on "frozen-oracle discipline". PZ-3 says no freeze manifest exists at HEAD, and LAWS.md LAW-14's annotation cites those grounds. So the grounds are governance-linked. Architect call.
- NEW-3 | TEST | B: test_promote_reason_guard.py::_result still seeds `"wr_sealbot": 0.6`. This is dead-key data (the scout names it in prose only). It goes with T6-08.

### Tally: raised 13 | confirmed 7 | amended 4 | refuted 0 | pending 2 (+ T6-10's lane-A half) | architect 0
