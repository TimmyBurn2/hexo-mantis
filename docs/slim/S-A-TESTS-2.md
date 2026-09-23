# S-A-TESTS-2 — T2 (tests/train/test_[h-z]*)

scope: `git ls-files tests/train | grep -E '^tests/train/test_[h-z]'` (49 files, 10 397 lines by `wc -l`).
method: every file read (the four largest PZ files read in outline); an AST outline per file (imports, docstrings,
top-level helpers); AST hash-compare of same-named helpers across `tests/` (scratchpad `t2_dupsym.py`, `t2_ring.py`,
`t2_sinks.py`); unused top-level names by occurrence count (`t2_unused_toplevel.py`); `uvx ruff check --select
F401,F841,F811 --isolated`; `git grep` for citations in producer_manifest.yaml, tier_declaration.txt, docs/contracts,
docs/governance. Torch cannot be imported here, and `tests/train/conftest.py` imports it, so NO tests/train module can
be collected normally. With `--noconftest`, only test_steps_budget_carry, test_train_import_dag and
test_init_trainer_device_required collect (16 passed). Every other claim below comes from reading the code, and the
reviewer should probe it where torch is available.
Test-count context: collected 5112, floor 4862 (`tools/ci_gates/test_count_floor.txt`), so there is 250 of headroom and
none of the deletions below needs the floor file to move.

## Summary
- Counts by class: TEST 9 (01 02 09 10 11 12 14 15 20); DUP 8 (03 04 05 06 07 08 16 19); DEAD 3 (13 17 18); runN mark 1 (21).
- Counts by lane: A 4 (09 13 16 17a); B 15 (01–08 10 11 12 14 15 19 20), where 08's test_resume_wiring_integration half is C; C 2 (17c 18). 21 is carded and only marked.
- Top 3 by Δlines: S-A-TESTS-2-01 -257 · S-A-TESTS-2-02 -168 · S-A-TESTS-2-04 -93 gross (S-A-TESTS-2-03 -76 is next).

## Findings

### S-A-TESTS-2-01 | TEST | B
subject: tests/train/test_inference_seam_events.py (whole file)
claim: tests/train/test_target_counter_events.py already makes every assertion in this file, using the same rig (the `_Pool`, `_Trainer`, `_Buffer`, `_SpySink` and `_drive` there are byte-identical: AST hash 2db62198/c4e9590d/f71456ed/5276737b).
evidence: `iteration_complete_carries_the_inference_seam_counter` (the key is present, the 3 slots exist, total/delta) is covered by target_counter::test_iteration_complete_carries_the_target_integrity_fire_rates (all 3 counters × 3 slots) plus ::test_the_delta_is_the_interval_change_and_the_total_is_cumulative (the `inference_failures_total` total and delta). `idle_seam_counter_is_visible_at_zero` is covered by ::test_an_idle_lever_stays_visible_at_zero (every counter: present, 0/0, per_position 0.0). `two_conjunct_counters_are_distinct` is covered by ::test_the_three_counters_do_not_crosswire (distinct 22/222 and 33/333 values). Each of the file's own 3 named mutations reds one of those rows.
callers: none needed (a test module). `git grep test_inference_seam_events` returns only a docstring pointer in tests/selfplay/test_inference_seam_counter.py, which needs re-pointing. It has no producer_manifest.yaml row (that row cites target_counter_events), no tier row, and no ruling or contract citation (R275(b) requires the stream carry the counter, not this file).
Δlines: -257 (`wc -l`), -3 collected tests
witness: tests/train/test_target_counter_events.py (the producer_manifest `target_integrity_counters` row)
depends: —

### S-A-TESTS-2-02 | TEST | B
subject: tests/train/test_parent_death_signal.py (whole file)
claim: the file's four rows are the same harness and the same assertions as tests/test_run_pdeathsig.py; that file's own docstring says it re-created "Harness shape, re-created locally because R5 bars cross-test imports". The direct-primitive path is also covered by tests/monitor/test_arm_exec_trampoline.py.
evidence: `test_a_SIGKILLED_parent_takes_its_armed_child_with_it` matches pdeathsig::test_a_supervised_run_entry_dies_with_its_SIGKILLED_parent, which arms through `arm_parent_death_if_supervised`, and that gate delegates to `arm_parent_death_signal` (signals.py). `test_an_UNARMED_child_survives_the_same_kill` matches ::test_an_UNSTAMPED_run_entry_SURVIVES_the_same_kill. `test_a_child_that_HANDLES_sigterm_is_still_taken_down` has a same-named twin in test_run_pdeathsig.py. The unconditional primitive, as arm_exec calls it, is covered by test_arm_exec_trampoline::test_a_direct_launch_through_the_trampoline_still_dies. `test_arming_reports_true_on_linux_and_false_elsewhere` is covered by pdeathsig's supervised row, which asserts `armed` on Linux; the "false elsewhere" half runs only off Linux.
callers: 3 rows in tools/ci_gates/tier_declaration.txt name this file (skipif). They go STALE and must be deleted in the same commit, or gate 3c reds. CARDS F-816-14 (SIGKILL leg closed) does not name the path.
Δlines: -168 (`wc -l`), -4 tests, -3 tier_declaration rows
witness: tests/test_run_pdeathsig.py; tests/monitor/test_arm_exec_trampoline.py; tier_census (gate 3c)
depends: —

### S-A-TESTS-2-03 | DUP | B
subject: the ring builder `_filled_hexg` / `_graph_buffer` / `_synthetic_ring` in test_inference_seam_events, test_iteration_complete_decoupling, test_launch_path_smoke, test_periodic_checkpoint, test_ply_cap_gate, test_policy_loss_trough_gate, test_target_counter_events, test_terminal_eval_rc, test_train_step_dispatch
claim: 9 in-slice copies of one 8-record graph ring (stones `[: 2 + (i % 2)]`, policy `[(2,0,0.6),(1,1,0.4)]`) duplicate the existing tests/conftest.py::mk_graph_buffer factory, which test_resume_ring_roundtrip.py already uses from tests/train.
evidence: `t2_ring.py` finds 9 in-slice functions (76 lines), plus 8 more copies outside the slice (T1 + tests/config). `t2_dupsym.py _filled_hexg` gives 13 byte-identical copies (hash 6b247d5b). test_target_stage45_pins::_graph_buffer is DIFFERENT (varied policy, mr=2) and is excluded.
deliberate?: several docstrings say "cross-test imports are barred, so each file builds it". The root fixture, and the `_coordinator_pool.py` / `_drivable.py` bare-name helper modules, show that sharing is allowed. The copies are not a seam, oracle or twin.
Δlines: up to -76 gross in the slice (AST spans). `_Buffer.__init__` builds the ring at construction, so a helper-module function (not the fixture) is the likely home, and the net figure depends on that choice.
witness: every coordinator drive in the listed files
depends: —

### S-A-TESTS-2-04 | DUP | B
subject: local spy sinks: heldout_gap::_Sink, inference_seam_events::_SpySink, iteration_complete_decoupling::_SpySink, monitor_liveness_arming::_Sink, ply_cap_gate::_Sink, policy_loss_trough_gate::_Sink, resume_owned_paths::spy_sink (a local fixture), run_safety_wiring::SpySink, target_counter_events::_SpySink, terminal_eval_rc::_SpySink
claim: 10 local recording sinks duplicate tests/train/conftest.py::SpyEventSink (the `spy_sink` fixture, which has emit/named/has) and tests/train/_microbatch_harness.py::SpySink.
evidence: `t2_sinks.py` gives 93 lines over the 10 classes. The AST-identical groups are 25ab73e7 (heldout, ply, trough, plus 3 outside the slice), f182ce5b (liveness plus 2 outside) and 5276737b (seam, target_counter).
deliberate?: partly. target_counter, terminal_eval_rc and inference_seam subscript `e["event"]` on purpose ("a payload without it is a producer defect"), while the conftest spy uses `.get`, so merging means choosing one semantic. terminal_eval_rc adds `order()`.
Δlines: -93 gross, minus the shared-helper growth
witness: the rows that read `sink.named(...)`
depends: —

### S-A-TESTS-2-05 | DUP | B
subject: `_GRAPH_FULL_CONFIG` in test_inference_seam_events, test_iteration_complete_decoupling, test_target_counter_events, test_terminal_eval_rc, plus the same literal inline in ply_cap_gate::_harness, policy_loss_trough_gate::_harness and heldout_gap (the coordinator drive)
claim: 4 byte-identical module constants (hash a396a6a6, 6 lines each; 10 across tests/train) and 3 inline copies of the same 3-section dict.
evidence: `t2_dupsym.py _GRAPH_FULL_CONFIG` → 10 copies, 60 lines
deliberate?: no. The literal is the "non-binding caps" drive declaration in every copy, and none is a schema-owned full config.
Δlines: -24 (the in-slice constants) plus about 12 (the inline literals), plus 1 shared definition
witness: the coordinator drives in those files
depends: 03 (same helper module)

### S-A-TESTS-2-06 | DUP | B
subject: test_inference_seam_events::_Trainer, test_target_counter_events::_Trainer, test_iteration_complete_decoupling::_FakeTrainer
claim: all three have the loss dict and the train_step/save surface of tests/_drivable.py::DrivableTrainerStub (R367(a): "The ONE trainer double"), which terminal_eval_rc and heldout_gap already subclass.
evidence: the loss dict is identical to `DrivableTrainerStub.loss_info`. `_FakeTrainer` only differs by incrementing in both entry points, as the stub also does.
deliberate?: no. CARDS says "the trainer stub is one since F2"; these three are residue of that sweep.
Δlines: -55 gross (17+17+21, AST spans)
witness: the files' drives
depends: 01 (if 01 lands, one copy goes with it)

### S-A-TESTS-2-07 | DUP | B (carded, CARD-MECHANISM-SWEEP)
subject: the private `_Pool`/`_Buffer`/`_RunnerStats` fakes: inference_seam_events+target_counter_events `_Pool`/`_Buffer` (identical), terminal_eval_rc `_Buffer` (identical to those), ply_cap_gate+policy_loss_trough_gate `_Buffer` (identical, cf55b39c), periodic_checkpoint+train_step_dispatch `_Pool`+`_RunnerStats`+`_NON_BINDING_CAPS` (identical, 72bf44ea/4ccfc064), `_RunnerStats` ×5 in the slice (12 across tests/train, 70328923), policy_loss_trough_gate::_Pool
claim: carded as "the 21 remaining private _Pool/_Buffer fakes". The one new observation: policy_loss_trough_gate::_Pool+_RunnerStats (32 lines) is CoordinatorPoolStub minus `ply_cap_window_counts`, and the ply-cap gate is None in that harness, which reads no window (ply_cap_gate::test_the_explicit_off_posture… asserts `window_calls == []`). So the in-slice helper tests/train/_coordinator_pool.py::CoordinatorPoolStub replaces it as is.
evidence: `t2_dupsym.py _Pool/_Buffer/_RunnerStats`
deliberate?: no (carded)
Δlines: -32 for the trough case. The rest is sized by the card.
witness: test_policy_loss_trough_gate.py
depends: —

### S-A-TESTS-2-08 | DUP | B (C for test_resume_wiring_integration: PZ glob `tests/train/test_resume_*.py`) (carded, nine full-config literals)
subject: test_launch_path_smoke::{_eval_block,_MINTED_TRAIN,_REPRESENTATION,_drop_foreign_arch_keys,_train_block,_selfplay_block,_inference_block,_monitor_block,_config}; the same set in test_resume_wiring_integration (plus `_full_config`)
claim: the two files' block builders have identical bodies (only the annotations differ; `_train_block` adds lr/lr_schedule in resume_wiring), and tests/train/conftest.py::make_run_config(run_id=…) already builds the same config except for the eval block's values (cpu/1 vs cuda/96…) and the seed.
evidence: AST unparse diff shows `-def _eval_block(): +def _eval_block() -> dict:` and the same for selfplay, inference and monitor. `t2_dupsym.py` groups them (f249ae96, 8de638b5, bfb0a407, c6830200).
deliberate?: no. Both are carded full-config literals.
Δlines: -85 (launch_path_smoke) and -84 (resume_wiring_integration) gross (AST spans), each replaced by a 1–3 line builder call
witness: test_launch_path_smoke (integration tier); test_resume_wiring_integration
depends: —

### S-A-TESTS-2-09 | TEST | A
subject: tests/train/test_run_safety_wiring.py::test_stall_exit_code_equality_pin
claim: an exact duplicate: `assert SELFPLAY_STALL_EXIT_CODE == WATCHDOG_STALL_EXIT_CODE == 42` is also the whole body of tests/monitor/test_supervisor.py::test_exit_code_equality_pin.
evidence: `git grep "SELFPLAY_STALL_EXIT_CODE =="` → exactly those two lines. After the deletion, the file's two imports of these constants have no other use (grep shows only the import lines and line 200).
Δlines: -3 (AST span) -2 (the imports) = -5; -1 test; collected 5112→5111, floor file unchanged
witness: tests/monitor/test_supervisor.py::test_exit_code_equality_pin
depends: —

### S-A-TESTS-2-10 | TEST | B
subject: tests/train/test_run_safety_wiring.py::test_no_top_level_eval_import_under_train_or_monitor + ::_top_level_imports
claim: the union of two other tests makes this assertion, with the same top-level-import semantics (the helper is byte-identical to theirs): tests/train/test_train_import_dag.py (no `mantis.eval`/`mantis.arena` under train/**) and tests/monitor/test_monitor_census.py::test_monitor_mantis_imports_are_within_the_allowed_set (monitor/** may import only {util, encoding, monitor}).
evidence: `t2_dupsym.py _top_level_imports` → test_monitor_census and test_run_safety_wiring are identical (1c5e9bf6), and train_import_dag's copy has the same body plus a docstring. `_ALLOWED_MONITOR_MANTIS_IMPORTS = ("mantis.util", "mantis.encoding", "mantis.monitor")`.
Δlines: -11 -9 -1 (`import ast` becomes unused) = -21 (AST spans); -1 test
witness: tests/train/test_train_import_dag.py; tests/monitor/test_monitor_census.py
depends: —

### S-A-TESTS-2-11 | TEST | B
subject: tests/train/test_rates_are_measured.py::test_the_measured_games_per_hour_still_exists_on_the_coordinator
claim: `assert callable(StepCoordinator._games_per_hour)` is implied by ::test_a_rate_over_zero_elapsed_is_absent_not_zero in the same file, which calls it and checks both of its values.
evidence: lines 103/107 call `StepCoordinator._games_per_hour(stub)`
Δlines: -5 (AST span); -1 test
witness: ::test_a_rate_over_zero_elapsed_is_absent_not_zero
depends: —

### S-A-TESTS-2-12 | TEST | B (runN token carded)
subject: tests/train/test_pretrain_cli_states_no_training_knob.py::test_the_run5_divergences_the_row_measured_are_now_GONE
claim: its three `training_terms(run6).…` asserts pin minted literals (0.001 / 256 / 0.0005). ::test_training_terms_reproduces_the_YAML_own_numbers[run6.yaml] already asserts the same terms against the YAML's own numbers. This is also the F-49 "minted value as expectation" shape that tests/config/test_minted_values_are_provenance_not_expectation.py retired elsewhere.
evidence: `grep -n "lr:\|batch_size:\|eta_min:" configs/run6.yaml` → 0.001 / 256 / 0.0005, and the parametrize runs over `sorted(configs/*.yaml)`
Δlines: -6 (AST span); -1 test
witness: ::test_training_terms_reproduces_the_YAML_own_numbers
depends: —

### S-A-TESTS-2-13 | DEAD | A
subject: tests/train/test_pretrain_cli_states_no_training_knob.py::_DEAD_FLAGS
claim: an 8-line dict whose comment says "the assertions derive their subject from SHADOWED_TRAIN_KEYS, never from this map", and it has no reader. It is also stale: it names aux_* keys that SHADOWED_TRAIN_KEYS (`lr, weight_decay, batch_size, eta_min`) does not.
callers: AST/text: `git grep -n -w _DEAD_FLAGS` → the definition only. importlib/getattr/string: the name is private to the test module and no `getattr(` in tests names it. conftest/plugins: none (a test-module private). entry points, `-m`, subprocess, pyo3, config keys, gates, STATE procedures: not applicable to a test-private constant (CALLERS §1–§11 list no tests/ constants).
Δlines: -10 (8 lines of AST span + the 2-line comment above it)
witness: NONE
depends: —

### S-A-TESTS-2-14 | TEST | B
subject: tests/train/test_pretrain_encoding_no_fallback.py checkpoint arm (`_config_encoding` and 5 test functions)
claim: `_config_encoding` is a test-local copy of a veneer that no longer exists. tests/encoding/test_resolver_agreement.py's docstring records that "the pretrain validator's `_config_encoding` went with the dense pretrain pipeline, R346(f)", so the arm exercises `resolve_from_config` itself, and tests/encoding already makes most of its assertions.
evidence: `no_encoding_at_all` is made by test_resolvers_nested_identity::test_neither_shape_present_still_raises (same class and match). `explicit_encodings_still_resolve` (4 items) is made by test_resolver_agreement::test_single_shapes_resolve_unchanged and ::test_agreeing_dual_shape_resolves_like_single_shape. `conflicting_dual_shape` is made by ::test_disagreeing_flat_string_and_nested_identity_raise_the_named_error. UNIQUE and to be moved, not dropped: the `"no 'version' key"` message match, `{"encoding":{"version":6}}` → "must be a string", and `{"identity":{"encoding":6}}` → EncodingRegistryError. All three were probed live here (`resolve_from_config` imports without torch). The CLI arm (`_resolve_encoding_name`, `pretrain`) is live and stays.
Δlines: -26 for the duplicates (3+4+14+5, AST spans including the parametrize decorator); -6 collected items; about 0 net for the 3 moved rows
witness: tests/encoding/test_resolver_agreement.py, test_resolvers_nested_identity.py
depends: —

### S-A-TESTS-2-15 | TEST | B
subject: tests/train/test_train_step_dispatch.py::test_graph_spec_never_calls_the_dense_entry_point, plus the dead `train_step_from_tensors` stub methods (test_train_step_dispatch::_RecordingTypedTrainer, ::_HalfTrainer (inside test_missing_graph_entry_point…); test_inference_seam_events::_Trainer, test_target_counter_events::_Trainer, test_iteration_complete_decoupling::_FakeTrainer)
claim: the grid route has been deleted. `run_declared_train_step` has one `if representation == "graph"` arm and then raises. `rec.tensor_calls == []` is therefore vacuous, and `len(rec.graph_calls) == 1` is also asserted by ::test_graph_arm_threads_recency_weight_as_recent_frac. No src caller reaches a `train_step_from_tensors` stub.
evidence: `git grep -n -E "\.train_step_from_tensors|['\"]train_step_from_tensors['\"]" -- src tools` returns nothing (rc=1). `grep -n "def train_step_from" src/mantis/train/trainer/*.py` lists only `train_step_from_graph_batch`. There is no `isinstance(…, TrainerLike)` anywhere.
Δlines: -8 (the test) -4 (the `_RecordingTypedTrainer` stub + `tensor_calls`) plus small edits in 4 doubles; -1 test
witness: the rest of test_train_step_dispatch
depends: HANDOFF H1 (drop the protocol member first, or in the same change)

### S-A-TESTS-2-16 | DUP | A
subject: test_lifecycle_contract::restore_signals, test_orphan_workers_census::restore_signals (5 test signatures use them)
claim: both fixtures save and restore SIGINT+SIGTERM. The autouse tests/conftest.py::_restore_signal_dispositions already does exactly that around every test ("inner save/restores nest cleanly").
evidence: `t2_dupsym.py restore_signals` → 2 identical (2d74a20b); tests/conftest.py autouse fixture body `saved = {sig: signal.getsignal(sig) for sig in (SIGINT, SIGTERM)}`.
Δlines: -15 (8+7, AST spans) plus 5 parameter removals; test count unchanged
witness: the 5 signal rows (restoration still happens through the autouse fixture)
depends: —

### S-A-TESTS-2-17 | DEAD | A (17a) / C (17c)
subject: unused imports (ruff F401, which pyproject exempts for tests/**)
claim: 18 import lines are unused. 17a, lane A, 11 lines in unprotected files: iteration_complete_decoupling `pytest`; launch_path_smoke `numpy`; lifecycle_contract `sys`; orphan_workers_census `sys` + the in-function `signals as sig_mod` (line 93); parent_death_signal `textwrap`; periodic_checkpoint `numpy`; training_step_absence `numpy`, `torch`, `lookup`, `arch_from_spec_and_config, build_net`. 17c, lane C, 7 lines in PZ or self-declared-frozen files: heartbeat_watchdog `Path`; resume_bundle `os`, `torch`; resume_semantics `pytest`; resume_wiring_integration `lookup`, `build_net`; target_stage45_pins `pytest` ("byte-frozen through IMPL").
callers: `uvx ruff check --select F401 --isolated <slice>` → 19 names. They are imports, so no other channel applies. conftest already imports torch, so no import side effect is lost.
Δlines: -11 (17a), -7 (17c)
witness: NONE (ruff F401 is ignored for tests/** by pyproject)
depends: 02 (parent_death_signal's line goes with the file)

### S-A-TESTS-2-18 | DEAD | C
subject: test_resume_wiring_integration::_LADDER_RUNGS (the eval.ladder subject was deleted by R362); test_trainer_seam_conformance::trainer_accesses (PZ-2 file); test_heartbeat_watchdog::_Clock.advance (unused method); test_resume_owned_paths::spy_sink (a local fixture shadowing the conftest one with the same behaviour)
claim: each has zero readers. All four sit in PZ files (the `test_resume_*` glob, PZ-2 seam conformance, and the PZ-6 heartbeat_watchdog pins), so they are lane C.
callers: `git grep -n -w _LADDER_RUNGS` / `trainer_accesses` → the definition only. `grep "\.advance("` over the file → none. fixture-by-parameter: `spy_sink` in resume_owned_paths is used, but the conftest `spy_sink` provides the same emit/named.
Δlines: -5, -2, -2, -14 = -23
witness: NONE (nothing reads them)
depends: —

### S-A-TESTS-2-19 | DUP | B
subject: test_quiescence_fires_producer::_iteration_complete, test_rates_are_measured::_iteration_complete (+ `_Rstats`/`_StubPool`)
claim: two copies of "drive `emit_iteration_complete_event` with a one-event sink and inline pool/buffer fakes", differing only in which pool and rstats they pass.
evidence: AST spans of 31 and 22 lines; same kwargs block
deliberate?: no
Δlines: about -20 net for one shared helper
witness: both files' rows
depends: —

### S-A-TESTS-2-20 | TEST | B
subject: tests/train/test_warmstart.py::test_no_personal_path_in_warmstart_source[/home/] and [/Users/]
claim: the same file's ::test_no_absolute_path_literal_anywhere_in_the_module "is strictly wider than the two tests above" (its own docstring) for string literals. Gate 17 (rule7_gate) scans any edit to the file for home paths, which covers comments. `expanduser` and `_HEADSWAP_AB_DIR` stay.
evidence: docstring text; tools/ci_gates/rule7_gate.py patterns (CLAUDE.md gate 17)
Δlines: 0 lines (turn the parametrize into a single needle); -2 collected items
witness: ::test_no_absolute_path_literal_anywhere_in_the_module; gate 17
depends: —

### S-A-TESTS-2-21 | TEST | B (carded, CARD-MECHANISM-SWEEP: mark only, no rename proposed)
subject: runN tokens: test_periodic_checkpoint::test_run5_config_produces_a_periodic_checkpoint_on_its_declared_route (+ its `configs/run6.yaml` pin), test_pretrain_cli_states_no_training_knob::test_the_run5_divergences… (see 12) (+ its run6.yaml pin), test_steps_budget_carry::test_the_carry_holds_the_ratio_under_run8s_arrival_mix, test_monitor_liveness_arming::_PRODUCTION = "configs/run6.yaml" (a relative path, which also assumes the cwd is the repo root), and the literals `run6_00000750_abcdef12.ckpt` / `run_id="run6"` in test_resume_state and test_resume_ring_roundtrip
claim: carded; recorded only
evidence: `git grep -n -E "def test\w*run[0-9]" -- 'tests/train/test_[h-z]*'` → 3; `configs/run[0-9]+\.yaml` → 3 pins
Δlines: 0
witness: —
depends: —

## DEFECTS
- tests/train/test_parent_death_event.py::test_a_real_boot_writes_the_arming_event_into_the_runs_own_jsonl sets `signals_mod._LAST_DECISION = None` and pops `PARENT_DEATH_PPID_ENV` without monkeypatch. That module and process state leaks into later tests.
- tests/train/test_survivability.py::_tmp_out_dir uses `tempfile.mkdtemp` and never cleans up (each run leaves a directory in the system temp dir).
- tests/train/test_train_import_dag.py: the test name says `…_eval_monitor_arena_import`, but monitor left the ban (docstring WP13-A). Its `path.read_text()` has no `encoding=` (function scope, gate-16 backlog), and the same holds in test_run_safety_wiring and test_periodic_checkpoint.
- tests/train/test_training_step_absence.py: the docstring and the "on both arms" comment still name the deleted `train_step_from_tensors` tail.
- tests/train/test_lifecycle_contract.py: `DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC  # noqa: F401` is in fact used (`_watchdog` default), so the noqa is stale.

## PARKED
none

## HANDOFF
- H1 (owner of src/mantis/train/coordinator, PZ-2, lane C): `config.py::TrainerLike.train_step_from_tensors` is a phantom protocol member. It has no src caller (the grep above), the production `Trainer` does not implement it, and dispatch.py's module docstring and `run_declared_train_step` docstring still describe a grid arm and `_grid_step` that do not exist. Also check `GridRouteBufferLike`. test_trainer_seam_conformance checks call sites ⊆ members, so it cannot see a member that nothing calls.
- docs/contracts/checkpoint_envelope.md "Pinning tests" cites tests/train/test_warmstart.py for "warm-start / weights-only load from a foreign lineage". That file now pins only the no-host-path census and the deleted-arm graves; the live pin is tests/train/test_bc_warm_start_entry.py.
- docs/design/repo_design.md (the "SECOND STANDING BLOCKER … NOT fixed here" item citing test_pre_search_kind_checkpoint) is partly overtaken by src/mantis/train/checkpoints.py::_validate_stamped_config / RETIRED_STAMP_PATHS, which now log a missing newer leaf or a retired path instead of refusing.
- T1 (tests/train a–g + helpers): 7 more `_filled_hexg` and 6 `_GRAPH_FULL_CONFIG` copies (see 03, 05). tests/train/conftest.py::SpyEventSink and _microbatch_harness.SpySink are two shared spies for one job. Also ask whether make_run_config's default `run_id="run5"` is carded.
- tests/selfplay owner: tests/selfplay/test_inference_seam_counter.py's docstring points at test_inference_seam_events.py (re-point it if 01 lands).

## Not covered
- The body of test_heartbeat_watchdog.py rows 200–545, test_sparse_gumbel_row_target.py, test_server_owned_copy.py, test_nonfinite_guard.py, test_periodic_bundle.py and the resume_* PZ files beyond their outlines. All are PZ or contract-cited (lane C), so I only scanned their helpers for duplicates.
- Nothing was executed except 3 torch-free modules (16 passed with `--noconftest`) and the resolver probes. The coordinator-drive equivalences in 01, 03 and 06–07 are unexecuted and must be probed where torch is available.
- The terminal_eval_rc `main()` harness (lines 330–853) is read in outline only. Its `_Pool` and `_FakeEvalPipeline` are carded fakes.

## Review
reviewer: fresh read-only agent (not the author); probes in throwaway worktrees, removed
Probe: ONE batch worktree (scratchpad/wt/rev-tests2-A, detached at af4d37e) carrying 09+13+16+17a together; removed at the end.
Recipe result: `import mantis` OK from the worktree src; `pytest --collect-only -q -m ''` = "2289 tests collected, 167 errors", same as
baseline. That recipe CANNOT see this slice: tests/train is ONE directory-level error ("ERROR tests/train - ModuleNotFoundError: torch",
via tests/train/conftest.py), so every tests/train edit is invisible to it. Second route used: `--noconftest -p conftest` with the
worktree's tests/ on PYTHONPATH, which loads the ROOT conftest (so the autouse `_restore_signal_dispositions` is live) and skips the torch
conftest. Of the 9 edited modules, 7 import torch themselves (collection error at HEAD too); only test_orphan_workers_census +
test_parent_death_signal ran: 9 passed at HEAD, 9 passed in the worktree. cargo check skipped: the diff touches only tests/train/*.py.
Worktree diff: `git diff --shortstat` = 9 files, 5 insertions(+), 55 deletions(-) (net -50).

| ID | verdict | lane | Δlines (probe-measured for lane A) | note |
|---|---|---|---|---|
| 01 | CONFIRMED | B | -257 (wc -l); -3 tests | each assertion mapped to a target_counter row (below) |
| 02 | CONFIRMED | B | -168; -4 tests; -3 tier rows (95–97) | covering rows ran live here and PASSED; not ruling-named |
| 03 | CONFIRMED | B | ≤ -76 gross (scout) | 10 slice files define a builder, 9 after the stage45 exclusion; conftest::mk_graph_buffer exists |
| 04 | CONFIRMED | B | -93 gross (scout) | own AST hash (docstrings stripped): heldout/ply/trough `_Sink` one group, seam/target_counter `_SpySink` one group |
| 05 | CONFIRMED | B | ≈ -36 (scout) | 10 tests/train files define `_GRAPH_FULL_CONFIG`, 4 of them in the slice |
| 06 | CONFIRMED | B | -55 gross (scout) | seam/target_counter `_Trainer` are AST-identical; `DrivableTrainerStub.loss_info` has the same dict except `grad_norm` is an attribute |
| 07 | CONFIRMED | B | -32 (trough case) | trough `_Pool` methods = CoordinatorPoolStub's minus `ply_cap_window_counts`; bodies NOT executed (torch) |
| 08 | NOT RE-DERIVED | B/C | — | only the lane split was checked (PZ glob `tests/train/test_resume_*.py`). The AST identity was not re-run (budget), so this is NOT confirmed |
| 09 | PENDING-PROBE (torch-bound) | A | -7 (scout -5: + 2 separator blanks) | twin tests/monitor/test_supervisor.py::test_exit_code_equality_pin PASSED live |
| 10 | CONFIRMED | B | -21 (scout) | + R8 consequence missed, see NEW-1 |
| 11 | CONFIRMED | B | -5 | the rows at the call sites call `_games_per_hour`, so the callable check is implied |
| 12 | CONFIRMED | B | -6 | the parametrize asserts lr/batch_size/eta_min against run6.yaml's own values |
| 13 | PENDING-PROBE (torch-bound) | A | -11 (scout -10: + 1 blank) | no reader anywhere (`git grep DEAD_FLAGS`, whole repo, substring) |
| 14 | AMENDED | B | -26 (scout) | claim text wrong, see note |
| 15 | CONFIRMED | B | -12 + edits (scout) | + H1 addendum; + R8 consequence (NEW-1) |
| 16 | AMENDED + PENDING-PROBE | A | -21 (scout -15) | orphan half probe-green; lifecycle half torch-bound; `import pytest` also becomes unused |
| 17a | PENDING-PROBE (5 of 7 files torch-bound) | A | -11 | orphan + parent_death lines probe-green; see the sig_mod trap below |
| 17c | CONFIRMED | C | -7 | own ruff run reproduces all 19 names |
| 18 | AMENDED | C | -23 (scout) | the spy_sink sub-item is DUP, not DEAD |
| 19 | AMENDED | B | undetermined | spans 31/22 and distinct hashes verified; "about -20 net" is an estimate, not a derived figure |
| 20 | REFUTED | — | 0 | the needle rows cover comments that neither the literal test nor gate 17 covers |
| 21 | CONFIRMED (mark only) | B | 0 | — |

### Per-finding notes
S-A-TESTS-2-01 — CONFIRMED: read both files → carries_the_inference_seam_counter (key, 3 slots) ⊂ target_counter::…carries_the_target_integrity_fire_rates (`_COUNTERS` includes inference_failures_total; checks all slots); its total/delta ⊂ ::the_delta_is_the_interval_change (200→260). idle ⊂ ::an_idle_lever_stays_visible_at_zero (loops over every counter: total 0, delta 0, per_position 0.0). distinct ⊂ ::do_not_crosswire (seam 222 vs defects 333, which catches seam→defects leakage too). The only citer is the docstring pointer in tests/selfplay/test_inference_seam_counter.py. No tier row and no governance citation (`git grep` over the whole repo).
S-A-TESTS-2-02 — CONFIRMED: ran `pytest -rA tests/test_run_pdeathsig.py` + trampoline::test_a_direct_launch… here → supervised-SIGKILL, UNSTAMPED-survives, HANDLES-sigterm and direct-launch all PASSED. The 4 pdeathsig FAILs are torch-import (mantis.run), unrelated. signals.py: the gate calls `arm_parent_death_signal()` with its default signal and returns that value through `_record`, so `assert armed` pins the Linux True path and the SIGKILL default. Residual: the off-Linux `return False` arm loses its only pin, which never executes on the Linux hosts anyway. Ruling check: `git grep -i "parent_death_signal|pdeathsig|F-816-14"` over docs/governance/docs/contracts/docs/design → CARDS rows and R300(d) (archived text) close F-816-14's SIGKILL leg on a GPU measurement and name no test path. LAW-16 names the lifecycle subsystem, not this file, so lane B, not C. The 3 tier rows must go in the same commit (the 4th row is unmarked).
S-A-TESTS-2-09 — PENDING-PROBE: the module imports torch transitively, so it errors at collection even with --noconftest. The worktree edit (test + `WATCHDOG_STALL_EXIT_CODE,` + the SELFPLAY import line) is ruff F401/F821 clean. Δ is -7 because deleting a mid-file def also removes its 2 separator blanks.
S-A-TESTS-2-10 — CONFIRMED: census::test_monitor_mantis_imports… flags any top-level `mantis.*` outside {util, encoding, monitor}, which includes mantis.eval. train_import_dag bans mantis.eval/arena under train/**. The top-level-only semantics are the same.
S-A-TESTS-2-13 — PENDING-PROBE: the pretrain cli module imports torch (collection error at HEAD with --noconftest). The removal (2 comment lines + 8-line dict + 1 blank) is ruff-clean.
S-A-TESTS-2-14 — AMENDED: the claim that `_config_encoding` is "a test-local copy of a veneer that no longer exists" is inaccurate. src/mantis/train/checkpoints.py::_config_encoding EXISTS at HEAD (it returns the raw `identity.encoding` or None, with no resolution and no raise). The test helper copies no src veneer: it is a 1-line alias of `resolve_from_config(...).name`. The covering test names exist (grep in tests/encoding). The partial-merge verdict and lane B stand.
S-A-TESTS-2-15 — CONFIRMED: dispatch.py::run_declared_train_step = `if representation == "graph":` then raise. `git grep train_step_from_tensors -- src tools` → the config.py protocol member, dispatch.py's module docstring AND a comment in src/mantis/train/coordinator/step.py, all dead text, to be added to H1. The seam and target_counter `_Trainer.train_step_from_graph_batch` delegate to their own `train_step_from_tensors`, so the double edit is an inline, not a deletion.
S-A-TESTS-2-16 — AMENDED Δ + PENDING-PROBE: autouse tests/conftest.py::_restore_signal_dispositions saves and restores exactly SIGINT+SIGTERM, the same as both local fixtures. Teardown order relative to monkeypatch is unchanged. After the fixture goes, test_orphan_workers_census has no `pytest` use left, so `import pytest` goes too (ruff F401 in the worktree), plus blank-line tidy → -21. Orphan half: 9 passed at HEAD and in the worktree through the root-conftest route. Lifecycle half: torch-bound. Note: test_lifecycle_contract.py is the LAW-16 contract file and docs/contracts/event_manifest.md cites 2 of its OTHER rows. No asserted line changes, so lane A holds.
S-A-TESTS-2-17 — 17a PENDING-PROBE / 17c CONFIRMED: `uvx ruff check --select F401,F811` → the same 19 names. TRAP: the line `from mantis.train.lifecycle import signals as sig_mod` occurs TWICE in test_orphan_workers_census. Only the first (test_second_signal…) is unused, and the second (test_mutation_disabled_teardown…) is live. My first text-matched removal produced F821. Remove it by position, not by text.
S-A-TESTS-2-18 — AMENDED (class): `git grep -w` confirms the definition only for _LADDER_RUNGS and trainer_accesses; `_Clock.advance` has no `.advance(` call. spy_sink in test_resume_owned_paths IS read, by parameter injection, and its emit/named are identical to tests/train/conftest.py::SpyEventSink (`.get("event")`, minus `has`). So it is a shadowing DUP, not DEAD. Lane C stands (PZ glob).
S-A-TESTS-2-20 — REFUTED: `grep -n -i "users|/home" tools/ci_gates/rule7_gate.py` → the only home pattern is `/home/[A-Za-z_]…`, and there is NO `/Users/` pattern. The needle rows scan raw source (`_WARMSTART_SRC`), comments included. test_no_absolute_path_literal… sees string literals only. A `/Users/x` comment, or a `/home/…` comment (like the one quoted in the row's own docstring), is caught by these rows alone.
S-A-TESTS-2-03/04/05/06/07/19 — own AST hash with docstrings stripped (a different tool from the scout's script): the groups reproduce. All stay lane B: they are test-helper merges that add no runtime arch branch.

### Missed by the scout (optional, max 5)
NEW-1 | R8/gate 15 | B — test_run_safety_wiring.py (312) falls below 300 if 09+10 land (-7 plus about -21), so its ">300 justify (R8)" docstring line must go in the same commit. test_train_step_dispatch.py (314) does the same under 15+03 (its line-1 header).
NEW-2 | DOC | C (H1 addendum) — a comment in src/mantis/train/coordinator/step.py still names the dead `train_step_from_tensors` entry point, beside dispatch.py's docstring.
NEW-3 | env — the "collected 5112" figure cannot be re-derived here (torch absent). Arithmetic on the scout's number: 01 -3, 02 -4, 09 -1, 10 -1, 11 -1, 12 -1, 14 -6 (+3 re-homed), 15 -1 → -18 gross, -15 net → about 5097, still above the 4862 floor, so the floor file does not move. Only 02 touches tier_declaration.txt (rows 95–97, the 3 skipif rows).

### Tally: raised 22 (01–21, 17 split a/c) | confirmed 12 (01 02 03 04 05 06 07 10 11 12 15 21) + 17c | amended 4 (14 16 18 19) | refuted 1 (20) | pending 4 (09 13 16 17a; 16 also amended) | architect 0 | not re-derived 1 (08)
