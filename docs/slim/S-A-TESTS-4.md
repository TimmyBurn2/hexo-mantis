# S-A-TESTS-4 — T4: tests/config, tests/data, tests/encoding, tests/env, tests/util
scope: tests/config/** (56 files), tests/data/** (7), tests/encoding/** (15), tests/env/** (1), tests/util/** (6); 85 files, 14 353 lines (00_MAP).
method: an AST outline of every file (docstring + top-level defs + spans); per-file `pytest --collect-only -q -m '' --continue-on-collection-errors` (895 collected, 11 torch collection errors: config/test_{coordinator_knobs_wiring,drain_caps_wiring,drawrate_arming_authority,fused_graph_caps_authority,gnn_widths_resolver,resolved_config_record,schema,train_device_authority}.py, encoding/test_{geometry_crosses_the_ffi,resolver_agreement}.py, util/test_determinism.py); `uvx vulture` over the slice; `git grep` for callers and citations; throwaway scripts that compare the literal payloads with the validated `configs/dev_example.yaml` dump and the three CONSUMER_REGISTRY copies; a targeted torch-free run of the 20 files these findings touch (`.venv/bin/python -m pytest -q -p no:cacheprovider …` → 158 passed). Every Δ is a `wc -l` or an AST span (`ast` node `end_lineno - first decorator lineno + 1`).
Test floor: collected at HEAD is 5112 against the floor of 4862 (CALLERS §0), so there are 250 tests of headroom. The B/C findings below remove at most 121 collected tests together. None of them touches a tier_declaration.txt row (`grep -E "tests/(config|data|encoding|env|util)/" tools/ci_gates/tier_declaration.txt` → only the train_device_authority, audit_cli and util/test_device rows, and no finding touches those).

## Summary
- By class: DUP 9 · TEST 13 · DEAD 2 · PACK 1 · ONE-SHOT 1 · DOC 1 (27 findings).
- By lane: A 1 · B 15 · C 11.
- Top 3 by Δlines:
  - S-A-TESTS-4-17: −489, the tests/data pack. It rides S-A-CORE-2-01.
  - S-A-TESTS-4-20: −379, nine full-config literal dicts. Already carded.
  - S-A-TESTS-4-14: −234, cpu_budget parked in tests (R289(q)).
- S-A-TESTS-4-03 is larger at −336, but it is a deliberate twin that a contract doc names, so it is not recommended.
- Sum of Δ excluding -03: about −1 963.

## Findings

### S-A-TESTS-4-01 | DUP | C
subject: tests/config/test_regime_parity_p2.py (whole file)
claim: Three of its four tests have byte-identical bodies to test_regime_parity.py (o9/o10/o11). The fourth, o12, repeats the `legal_move_radius*` ∉ `SelfplayConfig.model_fields` check that test_radius_removed.py::test_selfplay_config_has_no_radius_field and test_selfplay_schema.py::test_selfplay_has_no_legal_move_radius_field already make. The file was a PREREG "new-file" oracle, and its twin was never retired.
evidence: `cat` of both files → o9, o10 and o11 have the same statements, and o9 even has the same docstring. `grep -n -A6 test_selfplay_has_no_legal_move_radius_field tests/config/test_selfplay_schema.py` → the same two `model_fields` asserts.
deliberate?: no. Its own docstring says "IMPL retires the old file's O12 content at port time", and that never happened.
Δlines: −59 (`wc -l`); −4 collected. The docstring pointer in test_regime_parity.py also moves.
witness: tests/config/test_regime_parity.py, test_radius_removed.py
depends: — (lane C: docs/contracts/run_config_schema.md row "regime parity per LAW knob …" cites the path, so gate 10 reds until that row is edited)

### S-A-TESTS-4-02 | DUP | C
subject: tests/config/test_resolved_config_emit_p2.py (whole file)
claim: Every assertion is already made in test_resolved_config_emit.py:
- The key-set equality in ::test_payload_event_and_seven_knob_key_set implies the `_p2` len==7 check, the radius-absent check and the no-`train.`/`selfplay.`/`monitor.` prefix check.
- The source tags are the same assertion.
- ::test_payload_pins_production_values_unchanged_by_radius_removal is statement-for-statement ::test_payload_pins_production_values.
evidence: `cat` of both files. `_SIX_SCHEMA_LEAVES` (p2) and `_SEVEN_SCHEMA_LEAVES` (emit) are the same six-element set.
deliberate?: no. Its docstring says "IMPL retires the old 9-knob assertions in the existing file at port time". It was written as a replacement and never merged.
Δlines: −79 (`wc -l`); −5 collected
witness: tests/config/test_resolved_config_emit.py
depends: — (lane C: run_config_schema.md row "resolved-config emit …" cites the path, so gate 10 applies)

### S-A-TESTS-4-03 | DUP | C
subject: tests/config/test_every_key_has_consumer_p2.py (the second CONSUMER_REGISTRY)
claim: A second 160-entry registry plus four tests. It doubles the cost of adding any schema leaf. 103 of the 160 strings are identical to the first copy.
evidence: a throwaway load of both modules with `PYTHONPATH=tests/config` → `160 160`, `identical strings 103 of 160`
deliberate?: YES. It is a documented twin ("deliberately duplicated so the two copies must agree"), and docs/contracts/run_config_schema.md names "two independently-maintained copies". This is listed for visibility only and is not recommended without a ruling.
Δlines: −336 (`wc -l`) if a ruling retired it; −4 collected
witness: tests/config/test_every_key_has_consumer.py
depends: —

### S-A-TESTS-4-04 | TEST | B
subject: per-section "no pydantic default" censuses:
- test_train_schema.py::test_no_field_has_a_pydantic_level_default_EXCEPT_the_arch_scoped_ones
- test_selfplay_schema.py::test_selfplay_has_no_pydantic_level_default_EXCEPT_the_declared_operational_ones and ::test_inference_has_no_pydantic_level_default_EXCEPT_the_arch_scoped_ones
- test_monitor_schema.py::test_monitor_has_no_pydantic_level_default_EXCEPT_the_declared_operational_ones
- test_mcts_playout_cap_schema.py::test_mcts_has_no_pydantic_level_default and ::test_playout_cap_has_no_pydantic_level_default
- test_actor_sync_schema.py::test_schema_fields_are_required_with_no_pydantic_level_default
claim: Each one is a strict subset of test_schema.py::test_o16_all_fields_required_no_code_side_defaults. That test walks every field of every block reachable from RunConfig against the same ARCH_SCOPED_KEYS / OPERATIONAL_DEFAULT_KEYS registries, both ways, and adds `seen == exempt`.
evidence: reading test_schema.py (sed of the census section) against each per-section body. Every body is `is_required()` per field with the same exemption source.
Δlines: −80 (AST spans 25+13+17+15+3+3+4); −7 collected. Caveat: test_schema.py needs torch at import (mantis.model, eval.rounds, train.warmstart), so on a torch-free host only the per-section copies run. The box has torch.
witness: tests/config/test_schema.py::test_o16_all_fields_required_no_code_side_defaults
depends: — (test-floor move)

### S-A-TESTS-4-05 | TEST | B
subject: per-section extra-key rejections:
- test_train_schema.py::test_extra_key_rejected
- test_selfplay_schema.py::test_selfplay_extra_key_rejected, ::test_selfplay_nested_extra_key_rejected, ::test_inference_extra_key_rejected
- test_monitor_schema.py::test_monitor_extra_key_rejected, ::test_drain_caps_extra_key_rejected
- test_mcts_playout_cap_schema.py::test_mcts_extra_key_rejected, ::test_playout_cap_extra_key_rejected
- test_disk_guard_keys.py::test_disk_guard_extra_key_rejected
claim: All nine are covered by test_schema.py::test_o16_every_schema_block_is_strict, which asserts `StrictModel` and `extra == "forbid"` for every reachable block. The behavioural witnesses ::test_top_level_unknown_key_rejected and ::test_nested_unknown_key_rejected are already present.
evidence: `grep -A5 _extra_key_rejected` in each file: one `model_validate(… bogus_x=1)` per block
Δlines: −32 (AST spans); −9 collected. The same torch caveat as -04 applies.
witness: tests/config/test_schema.py::test_o16_every_schema_block_is_strict
depends: — (test-floor move)

### S-A-TESTS-4-06 | DUP | B
subject: tests/config/test_eval_config_remint.py::_NEW_LEAF_CONSUMERS, ::test_new_keys_all_have_consumers_in_o15_registry, ::test_screen_confirm_hi_key_is_rejected_everywhere (+ ::_PARITY_GATE), and the `"grid.yaml"` arm of ::_config_paths
claim: The first three items are covered elsewhere:
- `_NEW_LEAF_CONSUMERS` is a third copy of 14 registry entries, all byte-identical to the main registry. Its test only asserts that those keys are leaves, which both bijections already force.
- screen_confirm_hi is refused by the strictness census, and ::test_run3_parity_values_pinned already asserts `not hasattr(gate, "screen_confirm_hi")`.
- The grid template does not exist.
evidence: a throwaway comparison → `14 … set() set()`, `c strings identical to a: 14`. `ls tools/config_templates/` → `dev.yaml` only.
Δlines: −41 (AST spans 16+12+8+5) plus the one-token grid arm; −2 collected
witness: tests/config/test_every_key_has_consumer{,_p2}.py::test_schema_leaves_equal_consumer_registry_bijection
depends: —

### S-A-TESTS-4-07 | TEST | B
subject: tests/config/test_eval_schema_bounds.py::test_minted_configs_still_load_after_f_rt2_1_bounds
claim: Despite its name, it validates the file's own literal `_payload()`, not a minted config. So it is ::test_valid_payload_still_loads_after_bounds_added again, and its two asserted literals are properties of that literal. The real minted configs are already validated by gate 7 and test_schema::test_o16_every_committed_config_validates.
evidence: `sed` of the test → `payload = _payload()`; the asserted 3600.0/10.0 come from `_payload`
Δlines: −6 (AST span); −1 collected
witness: ::test_valid_payload_still_loads_after_bounds_added; gate 7
depends: —

### S-A-TESTS-4-08 | TEST | B
subject: tests/config/test_loader_duplicate_key.py::test_clean_minted_config_loads_and_validates; tests/config/test_allocator_posture_authority.py::test_ap01_every_committed_config_declares_the_posture_key
claim:
- The first is statement-identical to test_schema.py::test_example_config_validates: load dev_example.yaml and assert `run_id == "dev_example"`.
- The second asserts `hasattr(cfg, "allocator_posture")` on a validated RunConfig. That is always True because the field is required (::test_ap02). ::test_ap03 reads the posture of every discovered config.
evidence: `grep -A5` of both tests; `grep -A10 test_example_config_validates tests/config/test_schema.py`
Δlines: −10 (AST spans 3+7); −2 collected
witness: test_schema.py::test_example_config_validates; test_allocator_posture_authority.py::test_ap02_*, ::test_ap03_*
depends: —

### S-A-TESTS-4-09 | TEST | B
subject:
- tests/config/test_selfplay_playout_cap_mutual_exclusion.py::test_equal_quick_and_full_sims_raises and ::test_valid_differing_sims_and_mid_probability_constructs_cleanly
- tests/config/test_selfplay_schema.py::test_dirichlet_epsilon_field_name_equals_config_key
- tests/config/test_mcts_playout_cap_schema.py::test_playout_cap_field_name_matches_config_key_for_temperature_threshold
claim:
- The first two drive the same validator branch as ::test_quick_equal_to_full_now_raises_no_op and ::test_valid_move_level_cap_regime_constructs_cleanly (same predicate, same `match=`, different numbers).
- The two field-name tests are implied by the bound-violation parametrizations on the same keys (`("dirichlet_epsilon", 1.1)`, `("temperature_threshold_compound_moves", -1)`), which only red if the key reaches the field.
evidence: `sed -n 1,123p` of the mutual-exclusion file; the MCTS/PLAYOUT_CAP_BOUND_VIOLATIONS tables
Δlines: −24 (AST spans 5+6+7+6); −4 collected
witness: the sibling rows named in the claim
depends: —

### S-A-TESTS-4-10 | TEST | B
subject: tests/config/test_resolve_nsims.py::test_config_value_always_wins; tests/config/test_minted_values_are_provenance_not_expectation.py::test_the_relation_still_BITES_on_a_resolver_that_re_derives
claim: The one-line passthrough `resolve_eval_model_sims(k, v) == v` is asserted by seven tests.
- The first subject is ::test_random_reads_config_value with 64 in place of 96.
- The second asserts only on a lambda defined inside the test, plus the same passthrough arm 1 already asserts.
evidence: the `cat` of both files
Δlines: −12 (AST spans 2+10); −2 collected
witness: test_resolve_nsims.py::test_random_reads_config_value; arm 1 in the same file
depends: S-A-CORE-2-13. If that lands, the whole family goes instead: test_resolve_nsims.py (−32 `wc -l`, 5 tests), both o9 rows, and arms 1–2. test_the_one_provenance_pin_still_exists… then has to be re-pointed.

### S-A-TESTS-4-11 | DUP | B
subject: tests/encoding/test_resolvers_no_fallback.py (whole file); tests/encoding/test_resolvers_nested_identity.py::test_nested_identity_encoding_resolves and ::test_disagreeing_dual_shape_raises_not_a_precedence_pick
claim: Four files pin the same `resolve_from_config` arms, and the duplicates are:
- no_fallback::test_resolve_from_config_mapping_without_version_raises is body-identical to agreement::test_mapping_without_version_still_raises_missing.
- no_fallback's `None` and `{}` rows are parameters of agreement::test_absent_declarations_still_raise_missing.
- Its explicit-string and explicit-mapping rows are covered by agreement::test_single_shapes_resolve_unchanged.
- nested::test_nested_identity_encoding_resolves has the same input as the next test in its file.
- nested::test_disagreeing_dual_shape… has the same input as agreement::test_disagreeing_flat_string…, which also checks the message.
Only normalize(None) and normalize("gnn_axis_v1") are unique; move them to test_resolvers_mapping_no_fallback.py.
evidence: `sed` of all four files. Caveat: test_resolver_agreement.py imports mantis.train.anchor (torch), so torch-free coverage of `resolve_from_config(None)` would lapse.
Δlines: −54 (`wc -l`) −12 (AST 4+8), plus about 7 lines for the two moved rows; net ≈ −59; −7 collected
witness: tests/encoding/test_resolver_agreement.py, test_resolvers_mapping_no_fallback.py
depends: —

### S-A-TESTS-4-12 | DUP | B
subject: tests/encoding/test_encoding_round_trip.py::test_detect_graph_marker_resolves_when_unique, ::test_detect_stamp_beats_the_graph_marker
claim: Both are already covered in test_r8_identity.py:
- The first equals ::test_r328c_07: the same monkeypatch of `resolvers._graph_specs` to [v1] and the same `name == v1` assertion.
- The second is covered by ::test_r328c_06, which resolves stamped v1 AND r8.
round_trip::test_detect_graph_marker_REFUSES… and r8::test_r328c_05 also overlap, but each asserts something the other does not, so both stay. The round_trip module docstring ("deterministic shape fallback") is stale.
evidence: `sed` of both files
Δlines: −19 (AST spans 11+8); −2 collected
witness: tests/encoding/test_r8_identity.py (::test_r328b_03 is cited by crates/mantis-encoding/tests/registry_census.rs and stays)
depends: —

### S-A-TESTS-4-13 | TEST | B
subject: tests/encoding/test_inv22_spec_parity.py::test_registered_set_is_nonempty_and_derived, ::test_inv22_alias_read_equals_direct_read
claim:
- The first is the same assertion as test_registered_names_absence.py::test_name_set_is_the_compiled_all_specs_set: shim names == `_engine.all_specs()` names.
- The second is tautological given ::test_inv22_encoding_spec_is_engine_registry_spec_alias. Because `EncodingSpec is _engine.RegistrySpec`, `EncodingSpec.from_registry` and `_engine.RegistrySpec.from_registry` are the same call.
evidence: `sed -n 30,98p`; `.venv/bin/python -c "…type(lookup('gnn_axis_v1')) is _engine.RegistrySpec"` → True
Δlines: −17 (AST spans 4+13); −3 collected (the second is parametrized over 2 encodings)
witness: ::test_inv22_encoding_spec_is_engine_registry_spec_alias; test_registered_names_absence.py
depends: —

### S-A-TESTS-4-14 | DEAD | C
subject: tests/util/_cpu_budget.py + tests/util/test_cpu_budget.py
claim: This is product code with zero consumers, parked under tests/ by R289(q). Only its own smoke test reads it, and `apply_torch_interop_cap` is unused even by that test.
evidence: `git grep -n "cpu_budget"` (excluding the two files) → only src/mantis/util/__init__.py (a docstring narrative) and the archive ruling. vulture → `unused function 'apply_torch_interop_cap'`.
callers:
  AST imports: `git grep -n "_cpu_budget"` → only test_cpu_budget.py
  entry points / python -m / subprocess: CALLERS §1, §4 and §6 → none
  importlib/getattr: the `git grep` string search → none
  conftest: CALLERS §8 lists it only as a bare-name helper, with no fixture
  pyo3 / config keys / gates / STATE: none (it is stdlib-only; `git grep OMP_NUM_THREADS` → a RULINGS prose line only)
Δlines: −234 (`wc -l` 136+98); −11 collected
witness: NONE
depends: — (lane C: R289(q) placed it there. The src/mantis/util/__init__.py narrative goes too (S-A-CORE-3 notes it).)

### S-A-TESTS-4-15 | TEST | B
subject: tests/util/test_coordinates.py rows for flat_to_axial / axial_to_flat / cell_to_flat (::test_flat_to_axial_round_trip_all_cells, ::test_axial_to_flat_out_of_window_returns_none, ::test_flat_to_axial_known_values + KNOWN_TRIPLES, ::test_cell_to_flat_{origin,corners,whitespace_and_parens,invalid_format_raises,out_of_window_raises})
claim: The subjects have no product caller; only axial_distance is live (selfplay/instrumentation.py).
evidence: `git grep -n -w -E "flat_to_axial|axial_to_flat|cell_to_flat" -- src tools crates` → only the util/__init__ docstring example
Δlines: −52 (AST span 40 + KNOWN_TRIPLES 12); −17 collected
witness: the remaining axial_distance rows
depends: S-A-CORE-3-05

### S-A-TESTS-4-16 | TEST | B
subject: tests/env/test_game_state.py; tests/util/test_b4_history_len_sot.py
claim: The subject is mantis.env.game_state (and HISTORY_LEN, whose only src consumer is game_state). Its only product importer is data/generate.py, which has no caller.
evidence: `git grep -n -E "game_state|GameState" -- src tools` → data/generate.py plus comments; `git grep -n -w HISTORY_LEN -- src tools crates` → game_state.py, constants.py and a tools/hardcode_scan.py label
Δlines: −142 (`wc -l` 113+29); −13 collected
witness: NONE after the subject is deleted
depends: S-A-CORE-2-02 (and S-A-CORE-2-01)

### S-A-TESTS-4-17 | PACK | C
subject: tests/data/{test_corpus_io.py, test_sources_metrics.py, test_data_loss_counters.py, _frozen_games.py}
claim: This is the old corpus pipeline's test pack.
- test_bootstrap_encode.py, test_bootstrap_split_and_truncation.py and test_no_identity_blind_board.py stay (bootstrap_encode is a live `python -m` CLI).
- The rest tests data/{corpus_io, generate, human_seeding, corpus_metrics, corpus_analysis, sources}, which have no product caller outside mantis.data.
- `_frozen_games.py` is read only by test_sources_metrics. Its `ENCODINGS = ("v6","v6w25","v6_live2_ls")` is unused (vulture) and names deleted encodings.
- The test_data_loss_counters census of blind excepts can stay as long as data/ keeps any module.
evidence: `git grep -n "_frozen_games"` → test_sources_metrics.py only; the S-A-CORE-2-01 caller census; `git grep -n "ENCODINGS\b" -- tests/data` → the definition only
Δlines: −489 (`wc -l` 89+142+209+49) at most; −25 collected (6+7+12)
witness: NONE after the subject goes
depends: S-A-CORE-2-01 (lane C: tests/data/_frozen_games.py is in the PZ glob list)

### S-A-TESTS-4-18 | DEAD | A
subject: tests/config/test_mint_header_roundtrip.py::_BASELINE_KNOWN_BAD; tests/encoding/test_encoding_round_trip.py::_grid_state
claim: Two module-private names are never read. The first names the deleted key `eval.ladder.rungs`. The second builds a retired dense state dict.
evidence: `git grep -n -w -E "_BASELINE_KNOWN_BAD|_grid_state" -- .` → only the two definitions
callers:
  AST imports: the whole-tree `-w` grep above → definitions only. The two test modules are loaded by path nowhere: the tests/config spec_from_file_location loads are test_actor_sync_schema.py, the consumer registries, preflight_mint.py and mint_config.py.
  entry points / python -m / subprocess: none, since these are test-module privates (CALLERS §1, §4, §6)
  importlib/getattr/strings: `git grep -n -E "[\"'](_BASELINE_KNOWN_BAD|_grid_state)[\"']"` → 0
  conftest: neither is a fixture (no decorator); CALLERS §8
  pyo3 / config keys / gates / STATE: n/a (test-private Python names; no gate tool reads test source for them)
Δlines: −10 (AST spans 5+5)
witness: NONE
depends: —

### S-A-TESTS-4-19 | ONE-SHOT | B
subject: tests/config/test_docstring_debt_discharge.py
claim: This is a spent SC-A1/A2 text-absence oracle for two docstrings fixed long ago. Its docstring still says "this suite is RED at HEAD". Its pattern is also narrower than the defect: src/mantis/selfplay/hparams.py::InferenceHParams still says "same R1-exception as `SelfPlayHParams`" (S-A-CORE-2-19), and the test passes.
evidence: `grep -n "R-TRAINCONFIG-SCHEMA\|R1-exception" src/mantis/train/trainer/core.py src/mantis/selfplay/hparams.py` → hparams.py `"""Every ctor-time inference-server knob (same R1-exception as …)"""`
Δlines: −48 (`wc -l`); −2 collected
witness: NONE
depends: S-A-CORE-2-19 (the docstring it failed to catch)

### S-A-TESTS-4-20 | DUP | C
subject: the full-config literal builders (`_eval_block`/`_train_block`/`_selfplay_block`/`_inference_block`/`_monitor_block`/`_payload`) in tests/config/test_schema.py, test_schema_strict.py, test_actor_sync_schema.py, test_eval_schema_bounds.py and test_train_policy_value_target_consistency.py, plus test_actor_sync_reachability.py::_load_frozen_schema_oracle
claim: Five of the carded "nine full-config literal dicts" live here. After validation each one differs from the `configs/dev_example.yaml` dump by only 4–5 scalar leaves: allocator_posture, run_id, seed, selfplay.n_workers, eval.random_floor_games. test_schema.py's `_MINTED_TRAIN` already derives `train` that way. test_actor_sync_reachability.py re-executes a whole test module by path ("frozen oracle", but there is no freeze at HEAD, PZ-3) just to reuse `_payload`.
evidence: a throwaway `RunConfig.model_validate(_payload()).model_dump()` diff against dev_example → 5/5/4/5 differing leaves (test_schema.py cannot load torch-free)
deliberate?: no, but carded by CARD-MECHANISM-SWEEP ("each is a complete config a schema test owns"; applied on contact)
Δlines: −379 removed (AST spans 72+76+79+75+77) −16 (the loader); the shared derivation that replaces them is not counted
witness: every rejection test in those five files
depends: — (the card makes this lane C)

### S-A-TESTS-4-21 | DUP | C
subject: tests/config/test_drain_caps_wiring.py::_RunnerStats, ::_Pool, ::_Buffer, ::_fake_run_safety vs tests/config/test_coordinator_knobs_wiring.py's copies
claim: The composition fakes are near-verbatim copies:
- `_RunnerStats` and `_fake_run_safety` are identical.
- The drain `_Pool` is knobs `_Pool`+`_ComposePool` merged.
tests/_drivable.py and tests/train/_coordinator_pool.py show that a bare-name helper module is the house route, even though the knobs file's docstring claims "cross-test imports are barred".
evidence: `sed -n 60,160p` of the knobs file and `sed -n 1,140p` of the drain file
deliberate?: no; carded by CARD-MECHANISM-SWEEP ("the 21 remaining private `_Pool`/`_Buffer` fakes")
Δlines: −52 (AST spans in the drain file 6+33+6+7); both files are torch-dependent
witness: both files
depends: —

### S-A-TESTS-4-22 | DUP | B
subject: tests/config/test_cadence_clock_mutations.py::RUN5, ::run5, ::_revalidated (copies of test_armed_abort_cadence.py's) beside the unused-here tests/config/conftest.py::production_config
claim: This is the same module fixture and the same dump-mutate-revalidate helper in two sibling files. The production config is also hard-named in 57 lines across 24 slice files rather than taken from the census (R10). That naming half is carded.
evidence: `sed` of both headers; `git grep -c "run6\.yaml" -- tests/config tests/encoding` → 24 files
Δlines: −10 (AST spans 1+3+6)
witness: both files
depends: S-A-TESTS-4-27 (the RUN5 naming)

### S-A-TESTS-4-23 | TEST | C
subject: tests/encoding/test_registry_sha_handshake.py::test_missing_toml_skips_not_silently_passes
claim: It is covered by ::test_missing_toml_logs_a_reason, which makes the same no-raise call on an absent path and also asserts the logged SKIPPED line.
evidence: `sed -n 1,84p`
Δlines: −7 (AST span); −1 collected
witness: ::test_missing_toml_logs_a_reason
depends: — (lane C: the gate-8 handshake is on the seam)

### S-A-TESTS-4-24 | DOC | B
subject: oracle-era docstrings and names that disagree with the tree:
- "RED at HEAD / RED-at-import until IMPL lands" in 9 files (13 lines).
- tests/config/conftest.py "loads the live run5" (it loads run6).
- test_resolved_config_emit.py "7 schema leaves … = 8 knobs" (6+1=7).
- test_no_identity_blind_board.py says data/corpus_metrics.py was deleted (it exists).
- test_eval_config_remint.py::test_parity_config_mints_random_floor_disabled… asserts 20 (enabled). Its `if not path.exists()` guard is dead.
- test_allocator_posture_authority.py::test_ap01_both_mint_templates… (one template exists).
- test_armed_abort_manifest.py::_load_tool's "RED anchor … does not exist at HEAD" arm (the tool exists).
claim: These are stale narratives, and CLAUDE.md fixes them on contact only (R316(e)).
evidence: `git grep -c -E "RED[- ]at[- ](HEAD|import)|RED at HEAD" -- <slice>` → 13 lines in 9 files; `ls tools/config_templates/` → dev.yaml
Δlines: ≤ −13 (git grep count), plus the dead guard and the dead arm; applied on contact
witness: NONE
depends: —

### S-A-TESTS-4-25 | TEST | C
subject: tests/encoding/test_no_dead_resolver_export.py::_QUEUED_DEAD_EXPORTS, ::test_the_queued_dead_exports_are_still_dead, ::test_transitively_dead_cluster_is_recorded_not_silently_deleted; tests/encoding/test_one_artifact_pin_authority.py::test_an_encoding_with_no_anchor_pin_RAISES_rather_than_guessing, and the anchor loop of ::test_the_live_authority_answers_for_every_encoding_it_claims
claim: These are anti-rot pins over dead src exports: resolve_arch, expand_auto_paths, resolve_anchor_path. `_ANCHOR_PATHS` is `{}`, so the anchor loop iterates nothing and resolve_anchor_path always raises. The pins die with the src deletion.
evidence: `.venv/bin/python -c "from mantis.encoding.resolvers import _ANCHOR_PATHS; print(_ANCHOR_PATHS)"` → `{}`; `git grep -n -w "resolve_arch|expand_auto_paths|resolve_anchor_path" -- src tools crates` → resolvers.py plus re-exports only
Δlines: −35 (AST spans 1+9+17+8) plus the 3-line loop; −3 collected
witness: NONE
depends: S-A-CORE-1-24 (lane C: src/mantis/encoding/** is PZ; resolve_arch is "operator-sign-off-locked")

### S-A-TESTS-4-26 | TEST | C
subject: tests/config/test_every_key_has_consumer.py::test_the_walker_descends_into_an_OPTIONAL_block_not_only_a_required_one
claim: Its fixture half is covered by test_one_schema_leaf_walker.py::test_the_default_mode_hands_out_only_paths_a_config_can_write, whose `_Fixture` is a superset. Its real-schema half (the three draw_rate_abort inner keys are leaves, and the block is not) is forced by the bijection.
evidence: `sed -n 330,422p` of the consumer file; `sed -n 1,80p` of the walker file
Δlines: −38 (AST span); −1 collected
witness: test_one_schema_leaf_walker.py; the bijection
depends: — (lane C: run_config_schema.md row "… the walker's descent into an OPTIONAL block … in both copies" names it)

### S-A-TESTS-4-27 | TEST | C
subject: run-named mechanisms in the slice:
- `RUN5 = …/run6.yaml` in test_armed_abort_cadence.py, test_cadence_clock_mutations.py and test_config_discovery_authority.py.
- `run5`/`_run5`/`_run5_payload` helpers.
- test_eval_config_remint.py::test_run3_parity_values_pinned. Its name is also pinned as source text by test_minted_values…::test_the_one_provenance_pin_still_exists_and_still_names_its_grounds.
- test_r8_identity.py::test_r328b_01_the_run6_row…, ::test_r328b_02_run5s_identity…
- test_resolvers_nested_identity.py::test_nested_identity_resolves_the_graph_encoding_run5_declares.
claim: These are R10 violations that CARD-MECHANISM-SWEEP already cards. They are reported here as carded, not re-raised.
evidence: `git grep -n -E "def (test_)?[a-zA-Z0-9_]*run[0-9]+|RUN5 = " -- tests/config tests/encoding`
Δlines: 0 (renames)
witness: n/a
depends: —

## DEFECTS
- A flat `glob("*.yaml")` config census appears in three places. Each is blind to configs/ subdirectories, which the loader and gates 7/12 make legal (the invariant test_config_discovery_authority.py pins). Every other site uses discover_configs. The three:
  - tests/config/test_ply_cap_within_ring_stone_ceiling.py::test_every_shipped_graph_config_is_inside_the_ceiling
  - tests/config/test_resolved_config_record.py (the `parametrize` list)
  - tests/data/test_bootstrap_encode.py::test_no_config_key_selects_the_encoder
- tests/config/test_resolved_config_emit.py::_SEVEN_SCHEMA_LEAVES holds six names; the docstring's "8 knobs" is 7.
- tests/config/test_drawrate_arming_authority.py's comment says the `resolve.draw_rate` module "does not exist". src/mantis/config/resolve/draw_rate.py does exist (PZ file; on contact).

## PARKED
- tests/encoding/test_no_dead_resolver_export.py runs one `git grep` subprocess per exported function. The config mint/diff tests spawn one `tools/*.py` subprocess per committed config.

## HANDOFF
- C2: src/mantis/config/schema/core.py `representation: Literal["grid","graph"]` keeps a representation no registered encoding has. The test-side `_GRID_CONFIGS == []` guards in test_fused_graph_caps_authority.py (fg5_01c/fg5_08) exist only because of it (PZ schema, lane C).
- C3: tools/hardcode_scan.py rows for `HISTORY_LEN` / `game_state` go stale with S-A-CORE-2-02.
- S-L-DUP: tool-by-path loaders (`spec_from_file_location` on tools/ci_gates/preflight_mint.py and tools/mint_config.py) are re-typed per module in tests/config and tests/tools. tests/tools/conftest.py::load_tools_package is the one existing helper.

## Not covered
- The torch-dependent files were read but not run: the 11 collection errors listed in scope. That includes test_schema.py, the owner of the census that -04/-05 lean on.
- Line-level review was not done for the biggest behavioural suites: test_allocator_posture_authority.py (87 tests), test_armed_abort_manifest.py, test_armed_abort_cadence.py, test_inference_batch_reachability.py, test_consumer_citation_arrows.py, test_preflight_stamp.py and test_audit_cli.py. Only their outlines were checked for duplication against siblings. For the PZ files among them (test_drawrate_*, test_config_census, test_mint_*), proposals would be lane C in any case.
- The literal payload in test_schema.py could not be diffed against dev_example torch-free.
- The deliberately-different 57 strings between the two CONSUMER_REGISTRY copies were not audited.

## Review
reviewer: fresh read-only agent (not the author); probes in throwaway worktrees, removed
| ID | verdict | lane | Δlines (probe-measured for lane A) | note |
|---|---|---|---|---|
| S-A-TESTS-4-01 | CONFIRMED | C | −59 | o9/o10/o11 AST-equal to test_regime_parity.py; o12 covered; contract row cites the path |
| S-A-TESTS-4-02 | CONFIRMED | C | −79 | `_SIX == _SEVEN` (6 names); the pins test differs only by helper spelling |
| S-A-TESTS-4-03 | ARCHITECT | C | −336 if ruled | a deliberate twin that the contract doc names twice |
| S-A-TESTS-4-04 | CONFIRMED | B | −80 | census emulated torch-free: green over 30 models; the covering test is torch-bound |
| S-A-TESTS-4-05 | CONFIRMED | B | −32 | all 30 census models strict; the covering test is torch-bound |
| S-A-TESTS-4-06 | CONFIRMED | B | −41 | 14 of 14 strings identical in BOTH registries; only dev.yaml template exists |
| S-A-TESTS-4-07 | CONFIRMED | B | −6 | validates its own `_payload()` |
| S-A-TESTS-4-08 | CONFIRMED | B | −10 | covering test_example_config_validates is a superset but torch-bound |
| S-A-TESTS-4-09 | CONFIRMED | B | −24 | same `match="no-op"` branch; bound rows exist |
| S-A-TESTS-4-10 | CONFIRMED | B | −12 | 64 vs 96 only; family rides S-A-CORE-2-13 (C) |
| S-A-TESTS-4-11 | CONFIRMED | B | ≈ −59 | covering file test_resolver_agreement.py is torch-bound |
| S-A-TESTS-4-12 | CONFIRMED | B | −19 | inputs differ (marker-only vs `_gnn_state()`), but it is the same branch |
| S-A-TESTS-4-13 | CONFIRMED | B | −17 | `EncodingSpec is _engine.RegistrySpec` → True |
| S-A-TESTS-4-14 | CONFIRMED | C | −234 | R289(q) (archive RULINGS_ACTIVE) holds the path |
| S-A-TESTS-4-15 | CONFIRMED | B | −52 | 17 items, matches S-A-CORE-3-05 |
| S-A-TESTS-4-16 | CONFIRMED | B | −142 | test_b4 imports mantis.env.game_state at module scope, so the whole file rides S-A-CORE-2-02 |
| S-A-TESTS-4-17 | CONFIRMED | C | −489 max | `_frozen_games` read only by test_sources_metrics; PZ glob |
| S-A-TESTS-4-18 | AMENDED | A (`_grid_state`) + C (`_BASELINE_KNOWN_BAD`) | −14 probe (−7 + −7) | probe green; half is in PZ glob `tests/config/test_mint_*.py` |
| S-A-TESTS-4-19 | CONFIRMED | B | −48 | hparams.py still says "same R1-exception"; the oracle passes |
| S-A-TESTS-4-20 | AMENDED → ARCHITECT | C | −395 (scout, unprobed) | the card lists these files for their `model` line, not as a merge; gate-15 header coupling missed |
| S-A-TESTS-4-21 | CONFIRMED | C | −52 | `_RunnerStats`, `_fake_run_safety` AST-equal; both files torch-bound |
| S-A-TESTS-4-22 | CONFIRMED | B | −10 | `run5`, `_revalidated` AST-equal |
| S-A-TESTS-4-23 | CONFIRMED | C | −7 | the sibling makes the same no-raise call and adds the caplog assert |
| S-A-TESTS-4-24 | CONFIRMED | B | ≤ −13 | 9 files, 13 lines, re-counted |
| S-A-TESTS-4-25 | CONFIRMED | C | −35 | `_ANCHOR_PATHS == {}`; rides S-A-CORE-1-24 |
| S-A-TESTS-4-26 | CONFIRMED | C | −38 | the p2 copy has no such test (count 0) |
| S-A-TESTS-4-27 | CONFIRMED | C | 0 | the 4th `RUN5 =` site is tests/tools/test_preflight_mint_process.py (out of slice), which matches the card's "four" |

### Per-finding notes
S-A-TESTS-4-01 — CONFIRMED: an AST body compare (a scratchpad script, docstrings stripped) found identical bodies: p2 o9/o10/o11 == test_regime_parity.py test_o9/o10/o11. o12's `model_fields` asserts are test_selfplay_schema.py::test_selfplay_has_no_legal_move_radius_field. Its `hasattr` asserts on the instance follow from the model_fields check. `grep` of docs/contracts/run_config_schema.md → the "regime parity" row cites `test_regime_parity_p2.py`, which leaves gate 10 and a PZ doc edit, so lane C.
S-A-TESTS-4-02 — CONFIRMED: `_SEVEN_SCHEMA_LEAVES == _SIX_SCHEMA_LEAVES` → True (6 names). The pins test differs from ::test_payload_pins_production_values only by `_run5_payload()` vs `_run5().to_event_payload()`. The contract "resolved-config emit" row cites the path.
S-A-TESTS-4-03 — ARCHITECT: may the second CONSUMER_REGISTRY be retired? run_config_schema.md names "two independently-maintained copies" in the "every schema leaf key" row and "in both copies" in the "every-key-has-consumer bijection" row. Re-derived: 160/160, same key set, 103 identical strings.
S-A-TESTS-4-04 — CONFIRMED (torch-bound witness): I exec'd test_schema.py's own `_schema_census(RunConfig)` without its three torch imports and took the three `*_ROW` constants by AST from their source files. The census holds 30 models, including Train/Selfplay/Mcts/PlayoutCap/Inference/MonitorSchema/DrainCaps/DiskGuard. The o16 default census replayed on it gives bad=[] and seen==exempt. Collect-only → 7 subject tests. test_schema.py itself cannot run here.
S-A-TESTS-4-05 — CONFIRMED (torch-bound witness): the same census → every model is `StrictModel` with `extra == "forbid"` (nonstrict=[]). Collect-only → 9 subjects. The claim is structural-implies-behaviour; the behavioural witnesses (::test_top_level/nested_unknown_key_rejected) are torch-bound too.
S-A-TESTS-4-06 — CONFIRMED: loading the three modules → `_NEW_LEAF_CONSUMERS` has 14 entries, identical to registry a (14) and to registry b (14). `ls tools/config_templates` → dev.yaml, so the `("dev.yaml", "grid.yaml")` loop arm is dead.
S-A-TESTS-4-07 — CONFIRMED: `awk` of the test → `payload = _payload()` plus two literal asserts. ::test_valid_payload_still_loads_after_bounds_added does `RunConfig.model_validate(_payload())`. The contract doc cites the file, which stays.
S-A-TESTS-4-08 — CONFIRMED: the AST compare shows test_example_config_validates = the same two statements plus a `representation == "graph"` assert, so it is a superset. It is torch-bound, so a torch-free host loses the only dev_example load test in tests/config.
S-A-TESTS-4-09 — CONFIRMED: `grep match=` → `"no-op"` at both ::test_quick_equal_to_full_now_raises_no_op and ::test_equal_quick_and_full_sims_raises. Bound rows `("dirichlet_epsilon", 1.1)` and `("temperature_threshold_compound_moves", -1)` are in test_mcts_playout_cap_schema.py's tables.
S-A-TESTS-4-10 — CONFIRMED: the AST compare shows the bodies differ only by 64 vs 96. `git grep "resolve_eval_model_sims(...) =="` → 8 sites across 5 files. S-A-CORE-2-13 is DEAD|C and has no review section yet.
S-A-TESTS-4-11 — CONFIRMED: no_fallback::…mapping_without_version is AST-equal to agreement::…still_raises_missing. The agreement parametrize carries None and {}. agreement imports `mantis.train.anchor` (torch), so the covering file cannot run here. Collect → 9 subjects − 2 moved rows = −7.
S-A-TESTS-4-12 — CONFIRMED with a nuance: r328c_07 feeds a marker-only dict with the default `strict`. The default is False (resolvers.py signature), so `strict=False` in round_trip is a no-op. round_trip's extra `representation == "graph"` is a registry property. r328c_06 loops the stamped V1 and R8 over a marker, which covers stamp-beats.
S-A-TESTS-4-13 — CONFIRMED: `.venv/bin/python -c "…EncodingSpec is _engine.RegistrySpec"` → True. test_registered_names_absence.py::test_name_set_is_the_compiled_all_specs_set exists. Collect → 3 items.
S-A-TESTS-4-14 — CONFIRMED lane C: `git grep -i "cpu_budget\|R289(q)" -- docs/governance` → archive/RULINGS_ACTIVE.md: R289(q) "held the path reserved". `wc -l` → 136+98. Collect → 11.
S-A-TESTS-4-15 — CONFIRMED: collect → 17 flat/axial/cell items, which equals S-A-CORE-3-05's count (DEAD|B).
S-A-TESTS-4-16 — CONFIRMED: test_b4_history_len_sot.py imports `mantis.env.game_state` at module scope, so the file cannot outlive S-A-CORE-2-02. HISTORY_LEN then survives only in constants.py, the util/__init__ example and hardcode_scan (a handoff already noted). Collect → 11+2.
S-A-TESTS-4-17 — CONFIRMED: `wc -l` → 89/142/209/49. `git grep -E "_frozen_games" -- tests src tools` → test_sources_metrics.py:12 only. This CONTRADICTS S-A-CORE-2-01's conftest line ("bare-name helper for bootstrap tests"), and that line is wrong. Collect → 25. It is lane C because of the PZ glob `tests/data/_frozen_games.py`.
S-A-TESTS-4-18 — AMENDED: probe worktree scratchpad/wt/rev-tests4-18 (removed). Both symbols were dropped, including `_BASELINE_KNOWN_BAD`'s `#:` doc-comment line. Results: `import mantis` OK (the path shows the WT src first); collect → "2289 tests collected, 167 errors" (baseline unchanged, no test deleted); the 2 files → 21 passed; gate 10 rc 0; gate 15 "0 stale"; `uvx ruff --select F` clean; `cargo check --workspace --all-targets --locked` Finished. `git diff --stat` → 14 deletions (7+7), not −10. test_mint_header_roundtrip.py matches PZ glob `tests/config/test_mint_*.py`, so that half is lane C and only `_grid_state` (−7) stays lane A. Neither file crosses the R8 cap (228, 166).
S-A-TESTS-4-19 — CONFIRMED: `grep -n "R1-exception" src/mantis/selfplay/hparams.py` → the InferenceHParams docstring. test_docstring_debt_discharge.py passed in my targeted run, so the oracle is blind to it. Collect → 2.
S-A-TESTS-4-20 — AMENDED → ARCHITECT: CARDS.md CARD-MECHANISM-SWEEP inventories "the nine full-config literal dicts whose `model` line the fix leg left (each is a complete config a schema test owns)". It cards the stale `model` line on contact. It does NOT card a merge, and its parenthetical asserts R1 ownership of each complete literal. So the "already carded" basis fails; the merge is a new proposal. Question: may a schema test's owned complete literal become `dev_example.yaml` + a scalar delta? Spot check: test_schema_strict's `_payload` validated dump vs dev_example → exactly 5 leaves (allocator_posture, eval.random_floor_games, run_id, seed, selfplay.n_workers). MISSED coupling: test_eval_schema_bounds.py is 307 lines, so dropping its ~75-line builder takes it under 300 and its R8 header must go (gate 15).
S-A-TESTS-4-21 — CONFIRMED: AST-equal `_RunnerStats` and `_fake_run_safety`. The card names "the 21 remaining private `_Pool`/`_Buffer` fakes". The knobs docstring's "cross-test imports are barred" must be rewritten in the same leg. Both files are torch-bound, so this is unprobed here.
S-A-TESTS-4-22 — CONFIRMED: `run5` and `_revalidated` are AST-equal across the two files. The conftest `production_config` is used 27 times elsewhere in tests/config ("unused-here" = unused in these two files).
S-A-TESTS-4-23 — CONFIRMED: the bodies differ by path only, and the sibling adds the caplog SKIPPED assert. Lane C (gate-8 seam).
S-A-TESTS-4-24 — CONFIRMED: `git grep -c` → 9 files, 13 lines. conftest line 1 reads "loads the live run5".
S-A-TESTS-4-25 — CONFIRMED: `_ANCHOR_PATHS` → `{}`. Rides S-A-CORE-1-24 (DEAD|C, "operator-sign-off-locked"). Collect → 3.
S-A-TESTS-4-26 — CONFIRMED lane C: `grep -c` of the OPTIONAL-descent test in the _p2 copy → 0. The contract row's "in both copies" never held for this property, so the row edit is needed either way. test_one_schema_leaf_walker.py::_Fixture carries `optional_block: _Inner | None`.
S-A-TESTS-4-27 — CONFIRMED: `git grep "^RUN5 = "` → 3 slice files + tests/tools/test_preflight_mint_process.py.

### Missed by the scout (optional, max 5)
NEW-1 | coupling | -20 drops tests/config/test_eval_schema_bounds.py (307 lines, `wc -l`) under the R8 cap, so its justification header must be removed in the same leg (gate 15). -07 alone leaves it at 301.
NEW-2 | DOC | B | tests/encoding/test_resolvers_no_fallback.py docstring: "currently silently resolve to the v6 default" names a retired encoding. It dies with -11; if -11 is not taken, fix on contact.
NEW-3 | cross-slice | S-A-CORE-2-01's conftest callers line says `_frozen_games` serves the bootstrap tests. `git grep` shows test_sources_metrics.py is its only reader, so -17's reading is the correct one.
Couplings re-checked: tier_declaration.txt slice rows = train_device_authority ×2, audit_cli, util/test_device ×2, and no finding touches them. Floor file = 4862, and test_count_ratchet_down.txt is absent. Contract-doc citations (gate 10) exist for -01/-02/-03 and are ruled lane C. Governance docs cite no finding subject by test name (`git grep -F` for each name over docs/tools/src/crates → docs/slim only).
Torch-free run of the touched files: 534 passed, 1 failed. The failure is tests/data/test_bootstrap_split_and_truncation.py::test_the_policy_DENOMINATOR…, which fails from a function-scope `import torch` via mantis.train.losses. That is environmental and not a subject.

### Tally: raised 27 | confirmed 24 | amended 2 (-18, -20) | refuted 0 | pending 0 | architect 2 (-03, -20)
