# S-A-TESTS-7 — T7: tests/ root modules + tests/monitor
scope: `^tests/([^/]+$|monitor/)` (42 files, 10 091 lines at 69e1532); method: full read of the 14 test_run_*/composition
modules, tests/conftest.py, tests/_drivable.py, tests/monitor/conftest.py and the monitor unit/census files; `git grep -w` /
quoted-name / path-string sweeps per CALLERS.md §0; `uvx vulture --min-confidence 60` over the slice; AST span counts
(`.venv/bin/python` + `ast`, scripts in scratchpad `t7_spans*.py`); torch-free rows run in isolation
(`.venv/bin/python -m pytest -q -p no:cacheprovider` over 13 monitor/root files → 124 passed, 2 failed, both
`No module named 'torch'`: baseline). 14 slice modules import `mantis.run` (torch) and cannot collect here.

## Summary
- Counts by class: DUP 5 · DEAD 3 · TEST 9 · SIMPLIFY 2 · DOC 2 = 21.
- Counts by lane: A 3 · B 10 · C 8.
- Top 3 by Δlines: S-A-TESTS-7-01 (-341, composition pool/buffer fakes, lane C card-named) · S-A-TESTS-7-15 (-69,
  phantom-beat redundant/inert rows, lane C) · S-A-TESTS-7-02 / -05 (-68 each, root-drive recorders; process harness).
- Test-floor: the TEST rows together remove 23 collected tests (4862 floor in tools/ci_gates/test_count_floor.txt;
  none is declared in tools/ci_gates/tier_declaration.txt — `grep -E "^tests/([^/]+\.py|monitor/)"` → only rows for
  tests not touched here).
- Enabler: S-A-TESTS-7-21 — seven R8 headers justify their size with "R5 bars cross-test imports"; `tests/_drivable.py`
  (R367(a), REVIEW2 F2.1 "R5-clean") shows a bare-name helper module is permitted, so the DUP hoists are not barred.

## Findings

### S-A-TESTS-7-01 | DUP | C
subject: tests/test_run_composition.py::FakePoolNeverStarted,_RunnerStats,_no_terminal_eval_config; tests/test_run_strict_composition.py::_Pool,_RunnerStats,_Buffer,_fake_run_safety,_no_terminal_eval_config; tests/test_actor_sync_composition.py::_SyncRecordingPool,_RunnerStats,_Buffer; tests/test_run_eval_enabled_authority.py::_Pool; tests/test_run_disk_guard_abort_rc.py::_Pool; tests/test_run_partial_composition.py::_Pool; tests/test_run_root_lifecycle.py::_Pool
claim: the drivable WorkerPool double (same 8 attributes, same 9 methods, `games_completed` one-game-per-read) and its buffer / run-safety / no-terminal-eval patches are written 7 times in-slice; one `DrivablePoolStub` (+ buffer, `fake_run_safety`, `no_terminal_eval_config`) beside `DrivableTrainerStub` in tests/_drivable.py covers every variant (started/stopped recorders, `on_start`, raise-on-unstarted-stop, sync recorders).
evidence: `git grep -n -E "^class (_RunnerStats|_Pool|FakePool\w*|_Buffer)\b|def _fake_run_safety|def _no_terminal_eval_config" -- tests/*.py` → 16 defs in 7 files; also 8 inline `registry=SimpleNamespace(beat=` run-safety closures and 3 inline fake eval pipelines (`git grep -c`).
deliberate?: not a seam or oracle: every copy is harness (each docstring says "drivable stand-in"); CARD-MECHANISM-SWEEP names "the 21 remaining private `_Pool`/`_Buffer` fakes of the composition tests" as the next hoist after F2.1 — card-named, hence lane C (applied on contact per the card).
Δlines: -341 = 428 (AST spans of the 15 non-dead defs, scratchpad t7_spans.py: total 437 minus `_DrivableBuffer` 9) − 87 kept once (largest `_Pool` 60 + `_Buffer` 13 + `_fake_run_safety` 7 + `_no_terminal_eval_config` 7); inline closures not counted.
witness: every compose_run drive in the 7 files (torch; not runnable here).
depends: S-A-TESTS-7-21

### S-A-TESTS-7-02 | DUP | C
subject: tests/test_run_partial_composition.py::_RecordedDiskGuard,_Recorders,_install_recorders,_bounded (twins of tests/test_run_root_lifecycle.py's)
claim: the real-subsystem recording harness (REAL `build_run_safety` wrapped, `DiskGuard` subclass recording kwargs, sink-close finalizer, dev_example bounded config with a disk_guard block) is copied near-verbatim between the two LAW-16 root files; one shared copy serves both. `_events(run_safety)` is a third twin in tests/test_run_disk_guard_abort_rc.py.
evidence: t7_spans2.py → partial {_RecordedDiskGuard 9, _Recorders 6, _install_recorders 28, _bounded 13}, root_lifecycle {14, 8, 29, 12}; bodies diff only in `stop_calls` recording and docstrings.
deliberate?: the partial file's R8 header says the split was forced because "the sibling lifecycle file is BYTE-FROZEN" — false at HEAD (`git log -- tests/test_run_root_lifecycle.py` → edited in 596f9f9 and deb16b8). LAW-16 lifecycle tests → lane C.
Δlines: -68 = 56 (four spans) + 8 blank separators + 4-line R8 header of test_run_partial_composition.py, which falls from 317 to ≤ 253 and must DROP its justification (gate 15).
witness: the 3 partial rows + 9 root_lifecycle rows (torch).
depends: S-A-TESTS-7-01, S-A-TESTS-7-21

### S-A-TESTS-7-03 | DUP | C
subject: tests/test_run_partial_composition.py::restore_signal_dispositions; tests/test_run_root_lifecycle.py::restore_signal_dispositions
claim: both module-local autouse fixtures are exact copies of the root autouse tests/conftest.py::_restore_signal_dispositions (same SIGINT/SIGTERM save/restore, function scope, which already wraps every test), so they do nothing.
evidence: `git grep -n -A6 "signal.getsignal(sig) for sig in" -- tests` → the three bodies are identical; vulture flags both local copies unused (60%).
deliberate?: the root one's docstring states the restore-AROUND semantics that make nesting redundant. Lane C only for file contact (LAW-16 files).
Δlines: -21 = 2 × (8 span + 2 blank) + `import signal` in test_run_partial_composition.py (its only uses are lines 45/48 of the fixture: `grep -n "signal\." ` → 2 hits, both in it).
witness: tests/test_run_root_lifecycle.py signal rows (a leaked handler would red a later drive).
depends: —

### S-A-TESTS-7-04 | DUP | B
subject: tests/test_run_one_authority.py::{_called_name,_root_name,_func,_enclosing_defs,_code_text,_body_without_docstring,_production_sources,_rel,_call_sites}; tests/test_run_main_authority.py::{_called_name,_root_name,_func}; tests/test_run_import_authority.py::{_called_name,_enclosing_defs,_production_sources,_rel,_call_sites}
claim: the AST-census helper family is copied across the three run-authority censuses (plus an inline `walk`/`owner` copy inside test_run_root_lifecycle.py::test_the_signal_install_has_exactly_two_call_sites_and_one_of_them_is_the_root and `_compose_run_body` in strict); one `tests/_ast_census.py` holds them.
evidence: `git grep -n -E "^def (_called_name|_root_name|_func|_enclosing_defs)\(" -- tests` → 3/2/2(+1 other slice)/2 copies; t7_spans2.py spans 65 + 23 + 33.
deliberate?: not twins by design — O-A2/O-A4 are twins at the ASSERTION level, the helpers are byte-copies. Lane B: `_call_sites` reads module globals that test_the_pool_allowlist_BITES_on_a_third_construction_site rebinds, so the shared form needs a roots parameter (signature choice).
Δlines: -56 = 121 (three in-slice copies) − 65 (one union copy).
witness: the 18 census rows in the three files (torch-import at top; not runnable here).
depends: S-A-TESTS-7-21

### S-A-TESTS-7-05 | DUP | C
subject: tests/monitor/test_supervisor_config_witness.py::{_spawn_supervisor,_events}; tests/monitor/test_supervisor_signal_posture.py::{_spawn_supervisor,_events,_reap,_alive}; tests/monitor/test_arm_exec_trampoline.py::{_reap,_alive,_ppid_of}; tests/test_run_pdeathsig.py::{_reap,_alive,_ppid_of_pid}
claim: the process-harness helpers of the LAW-16 supervisor/parent-death suites are re-created per file (`_alive` zombie-aware /proc read ×3, `_ppid_of`≡`_ppid_of_pid`, `_reap` ×3, `_events` ×2, `_spawn_supervisor` ×2); tests/test_run_pdeathsig.py says it re-created them "because R5 bars cross-test imports".
evidence: `git grep -n -E "^def (_alive|_reap|_ppid_of\w*|_events|_spawn_supervisor)\b" -- tests` ; t7_spans4.py → 128 lines in-slice.
deliberate?: same bodies (read side by side); LAW-16 lifecycle → lane C.
Δlines: -68 = 128 − 60 (one copy each: 17 + 13 + 10 + 9 + 11).
witness: the pdeathsig / trampoline / signal-posture rows (Linux, subprocess).
depends: S-A-TESTS-7-21

### S-A-TESTS-7-06 | DEAD | A
subject: tests/test_run_composition.py::_DrivableBuffer
claim: defined, never instantiated.
evidence: `git grep -l -w _DrivableBuffer -- .` → the file itself (+ a sibling slim doc).
callers:
- AST imports: `git grep -w _DrivableBuffer -- tests src tools` → definition only (test modules are not importable by name).
- entry points / `python -m` / subprocess strings: n/a for a test-local class; `git grep -E "[\"']_DrivableBuffer[\"']"` → none.
- importlib/getattr: none (quoted-name grep above).
- conftest/plugins: not a fixture; no `pytest_plugins` (CALLERS §8).
- pyo3 / config keys / gate tool paths / STATE procedures: n/a (test-private name; whole-tree `-w` grep is empty outside the file).
Δlines: -11 (AST span 9 + 2 blank).
witness: NONE
depends: —

### S-A-TESTS-7-07 | DEAD | A
subject: tests/test_run_strict_composition.py::_Buffer.sample_batch_with_pos, ::_ExplodingTrainer.train_step_from_tensors
claim: both are the retired GRID route's entry points on fakes: src dispatches graph only (src/mantis/train/coordinator/dispatch.py::run_declared_train_step raises on anything else), `_Buffer`'s two drives never reach a step, and `_ExplodingTrainer` also overrides `train_step_from_graph_batch`, the one path taken.
evidence: `git grep -n "sample_batch_with_pos\|train_step_from_tensors(" -- src` → only the Protocol stubs in coordinator/config.py; vulture flags the `augment` arg 100%.
callers:
- AST: `git grep -n sample_batch_with_pos -- src tools tests` → Protocol decl + this fake; no call site.
- string/getattr: `git grep -E "[\"']sample_batch_with_pos[\"']"` → none.
- conftest/entry points/pyo3/config/gates/STATE: n/a (methods of test-private classes).
Δlines: -7 (4 incl. comment + blank; 3 incl. blank; read off `sed -n 135,155p`).
witness: test_a_drive_failure_propagates_and_close_out_still_ran, test_the_run_length_ceiling_is_ABSOLUTE_not_per_process
depends: —

### S-A-TESTS-7-08 | DEAD | A
subject: tests/monitor/conftest.py::{mutable_counter, SpyEventSink.has, ExitSpy.first, CallSpy(ret=, raises=)}; tests/conftest.py::MINTED_CONFIGS
claim: the monitor conftest's fixtures serve exactly one module (test_persist_fatal.py), which uses none of these members; `MINTED_CONFIGS` is cited only by a docstring, and that docstring is stale (callers pass run7/run8/run10 through the eval_enabled axis).
evidence: `git grep -l -w mutable_counter -- .` → conftest + 00_MAP; `git grep -n -E '\.has\b|\.first\b' -- tests/monitor` → 0; `git grep -n -E 'CallSpy\(.+\)' -- tests` → 0; `git grep -l -w MINTED_CONFIGS -- .` → tests/conftest.py only. Vulture: `mutable_counter`, `has` unused; `Callable` import unused once `mutable_counter` goes.
callers:
- conftest channel: fixtures resolve by parameter name only under tests/monitor/ (no subdirs); `git grep -w mutable_counter -- tests` → definition only.
- AST/string: quoted-name grep → none for all three names.
- entry points / subprocess / pyo3 / config keys / gates / STATE: n/a (test-private).
Δlines: -26 = mutable_counter 13 (lines 102-114 incl. blanks) + has 3 + first 4 + CallSpy ret/raises 5 (lines 74,75,79-81) + MINTED_CONFIGS 1.
witness: tests/monitor/test_persist_fatal.py (runs torch-free: green here).
depends: —

### S-A-TESTS-7-09 | TEST | B
subject: tests/test_run_composition.py::test_wired_sources_include_eval_round_iff_pipeline_built
claim: tests/test_run_eval_enabled_authority.py::test_the_config_key_alone_decides_whether_the_eval_pipeline_is_built makes the same two-direction `wired_sources` ∋/∌ "eval_round" assertion on the same dev_example config, through the REAL `build_run_safety`, and also asserts the pipeline is (not) built.
evidence: read both: identical predicate, the sibling strictly stronger.
Δlines: -41 (AST span 39 + 2); collected -1 (floor move).
witness: the eval_enabled_authority row.
depends: —

### S-A-TESTS-7-10 | TEST | B
subject: tests/test_run_eval_enabled_authority.py::test_no_parameter_can_force_the_eval_posture_or_the_run_identity, ::test_the_key_is_required_with_no_code_side_default
claim: the first is implied by tests/test_run_strict_composition.py::test_compose_runs_parameter_list_is_pinned_so_no_re_add_can_be_silent (exact 8-tuple, which excludes both names); the second by tests/config/test_schema.py::test_o16_all_fields_required_no_code_side_defaults, whose own comment says "a re-added SCHEMA default on `eval_enabled` or `run_id` reds it".
evidence: `sed -n 297,300p tests/config/test_schema.py`; strict tuple read.
Δlines: -26 (10+2, 12+2); collected -2.
witness: the two named siblings.
depends: —

### S-A-TESTS-7-11 | TEST | B
subject: tests/test_run_eval_enabled_authority.py::test_every_minted_config_declares_the_key_explicitly (×6 params), ::test_the_minted_axis_is_not_empty
claim: its stated subject ("declares explicitly") is structural now (required field + `extra="forbid"`, R1); what remains is an `eval_enabled is True` value pin over a second `configs/*.yaml` glob beside the census, written for a spent "zero-behaviour mint" migration. Design choice: it is the ONLY pin that production configs evaluate — keep a census-based one or drop.
evidence: docstring "today's effective posture is the code default True everywhere, so True is a zero-behaviour mint (§6)".
Δlines: -15 (8+2, 3+2); collected -7.
witness: NONE for the value; S-A-TESTS-7-10's schema census for explicitness.
depends: —

### S-A-TESTS-7-12 | TEST | B
subject: tests/test_run_launcher.py::test_the_launcher_prints_no_config_ok_readiness_line
claim: a tombstone for the deleted validate-and-exit print (`"config OK" not in run.py`); the launcher is now pinned by the exact flag-set census and the process-boundary usage row. Keep only if the grave class is wanted here.
evidence: docstring: "asserts a line that EXISTS at HEAD is gone" (stale: it has been gone since the rewrite).
Δlines: -14 (12+2); collected -1.
witness: NONE
depends: —

### S-A-TESTS-7-13 | TEST | B
subject: tests/test_run_strict_composition.py::test_the_minted_PRODUCTION_config_ships_the_actor_lag_abort_ARMED
claim: tests/config/test_armed_abort_manifest.py::test_arming_audit_fails_a_disarmed_production_config asserts `audit_arming(run6).disarmed == []`, and `actor_lag` is a REQUIRED `CONFIG_BOOL` row on `monitor.actor_lag_abort_enabled`; gate 12 audits the whole production census. Its docstring still says run5.
evidence: `grep -n -A8 'name="actor_lag"' src/mantis/config/armed_aborts.py`; `sed -n 101,128p tests/config/test_armed_abort_manifest.py`.
Δlines: -14 (12+2); collected -1.
witness: the armed-abort manifest row; gate 12.
depends: —

### S-A-TESTS-7-14 | TEST | B
subject: tests/monitor/test_sink.py::test_write_failure_counts_and_does_not_raise, ::test_persist_errors_total_starts_at_zero; tests/monitor/test_best_effort.py::test_counters_get_unknown_label_is_zero; tests/monitor/test_rules.py::test_math_import_available_for_finite_guards (+ the `assert not (float("nan") > 10.0)` stdlib sanity in ::test_grad_norm_spike_boundary_and_nonfinite_fires)
claim: (a) the same `sink_mod.json` monkeypatch, "+1 exactly" and no-raise assertions are the first half of the manifest producer test tests/monitor/test_persist_fatal.py::test_sink_failure_counts_and_aborts; (b) "starts at zero / stays zero" is asserted by ::test_emit_missing_event_key_raises_valueerror and the concurrency row; (c) `get` of a never-incremented label is exactly what ::test_success_returns_true_value_and_leaves_counter_unchanged reads (success never increments, `_counts.get(label, 0)`); (d) asserts stdlib `math.isfinite` only, no mantis subject.
evidence: rows read side by side; src/mantis/monitor/best_effort.py::BestEffortCounters.get.
Δlines: -43 (19 + 8 + 6 + 6 + 4); collected -4. test_rules.py also holds draw-rate rows (PZ-1); these two edits do not touch them.
witness: the named siblings (all torch-free, green here).
depends: —

### S-A-TESTS-7-15 | TEST | C
subject: tests/test_run_phantom_beat.py::test_producer_live_beat_drops_age (×2), ::test_deferred_heartbeat_bind_makes_it_live, ::test_all_four_sources_are_covered_by_the_census, and the inert arms of ::test_phantom_wired_deferred_heartbeat_forwards_beat
claim: `test_producer_live_beat_drops_age` touches no producer — its "mutation arm" is `x = None; if x is not None:` (never runs), so it re-asserts `HeartbeatRegistry.beat` drops age (tests/monitor/test_heartbeat.py::test_beat_resets_only_that_source; producer wiring is pinned by the manifest's test_step_loop_beats / test_poller_thread_beats_eval_round); `bind_makes_it_live` is the `[selfplay_drain]` case of `forwards_beat`; `all_four_sources` is the same tuple equality as tests/monitor/test_heartbeat.py::test_heartbeat_sources_name_pins (its extra set equality is implied); the forwards row carries the same inert constant-None arm.
evidence: `cat -n tests/test_run_phantom_beat.py` lines 24-53, 73-89.
Δlines: -69 = 32 (span 28 + 2 comment + 2) + 15 + 10 + 12 (inert arm lines 73-75→1, 80-89); collected -4.
witness: tests/monitor/test_heartbeat.py rows; the remaining forwards / conjunct rows. LAW-16 stall watchdog → lane C.
depends: —

### S-A-TESTS-7-16 | TEST | B
subject: tests/monitor/test_manifest_contract.py::test_the_retired_dense_instruments_have_no_manifest_row
claim: its manifest half is implied by ::test_shipped_manifest_every_row_resolves (a row whose producer is gone fails resolution — ::test_dead_producer_symbol_bites proves it); the rest is a grave for four deleted `mantis.train.events` names (R346(f) deletion). Keep only if graves are wanted outside the conformance suite.
evidence: read.
Δlines: -23 (21+2); collected -1.
witness: test_shipped_manifest_every_row_resolves (torch here).
depends: —

### S-A-TESTS-7-17 | TEST | C
subject: tests/monitor/test_heartbeat.py::test_exit_code_constants_are_pinned; tests/monitor/test_supervisor.py::test_exit_code_equality_pin
claim: literal re-pins (42, 43, SELFPLAY==WATCHDOG==42) of values tests/test_exit_code_table_census.py already ties to the design table in both directions (row 42 with both names is its own row); the census docstring calls a hand list "a second authority". Lane C: supervisor exit-code contract (LAW-16).
evidence: read the three files.
Δlines: -11 (4+2, 3+2); collected -2.
witness: tests/test_exit_code_table_census.py (green here).
depends: —

### S-A-TESTS-7-18 | SIMPLIFY | C
subject: tests/_drivable.py::DrivableTrainerStub.train_step_from_tensors
claim: the grid-route entry survives only as the body `train_step_from_graph_batch` forwards to; fold it into the graph method (the real Trainer has no `train_step_from_tensors`: `git grep "train_step_from_tensors(" -- src` → Protocol stub only).
evidence: tests/_drivable.py read; R367(a) "the ONE trainer double" → lane C.
Δlines: -3.
witness: every `from _drivable import` drive.
depends: S-A-TESTS-7-07

### S-A-TESTS-7-19 | SIMPLIFY | C
subject: tests/monitor/conftest.py (after S-A-TESTS-7-08)
claim: its four remaining fixtures (`spy_sink`, `fake_clock`, `exit_spy`, `snapshot_spy`) are used by one module; instantiating the classes there drops the fixture layer. Lane C: that module is the manifest's `persist_fatal` producer test.
evidence: `git grep -l -w -E "spy_sink|fake_clock|exit_spy|snapshot_spy" -- tests/monitor` → conftest + test_persist_fatal.py.
Δlines: -20 (4 fixtures × (3 lines + 2 blank)).
witness: tests/monitor/test_persist_fatal.py
depends: S-A-TESTS-7-08

### S-A-TESTS-7-20 | DOC | B
subject: stale oracle-first claims in tests/monitor/{conftest,test_best_effort,test_manifest_contract,test_persist_fatal,test_rules,test_sink}.py and tests/{test_run_composition,test_run_launcher,test_run_one_authority,test_run_strict_composition}.py
claim: 16 lines state "RED-at-import until IMPL", "does not exist yet", "absent at HEAD", "GREEN TODAY BY VACANCY … while `mantis.eval` does not exist" — all false at HEAD; plus the 2-line narrative note about the removed stride5-spam rule in tests/monitor/test_rules.py.
evidence: `git grep -n -i -E "RED-at-import|does not exist yet|until IMPL|before any port code|absent at HEAD|GREEN TODAY BY VACANCY" -- <slice>` → 16 lines.
Δlines: -13 derived as whole-line deletions (test_run_composition:93, test_run_launcher:29, test_run_one_authority:28, monitor/conftest docstring 11→1); the rest rewrite in place. R316(e): on contact only — ride with the rows touching these files; gate 14 floors may only fall.
witness: gate 14 comment ratchet.
depends: S-A-TESTS-7-01, -04, -08, -14

### S-A-TESTS-7-21 | DOC | B
subject: R8 headers of tests/test_run_composition.py, test_run_disk_guard_abort_rc.py, test_run_one_authority.py, test_run_partial_composition.py, test_run_root_lifecycle.py, test_run_pdeathsig.py
claim: each justifies its size or its private harness with "R5 bars cross-test imports" (partial adds "sibling … BYTE-FROZEN"); CLAUDE.md's R5 bars `sys.path` writes and a package named `tests`, not a bare-name helper module — tests/_drivable.py is exactly that and REVIEW2 F2.1 measured it R5-clean. The premise blocks -01/-02/-04/-05.
evidence: `git grep -n -i -E "cross-test import|R5 bars|BYTE-FROZEN" -- <slice>` → 7 hits in 6 files; CLAUDE.md hard rule 5.
Δlines: 0 (rewording; the header deletions are counted in -02).
witness: gate 15 (tools/ci_gates/r8_header_gate.py).
depends: —

## DEFECTS
- tests/monitor/test_arm_exec_trampoline.py::_reap carries `@_LINUX_ONLY` (a `skipif` mark on a non-test helper: inert).
- tests/test_run_partial_composition.py R8 header: "the sibling lifecycle file is BYTE-FROZEN" is false (edited in 596f9f9, deb16b8).
- tests/conftest.py::make_run_config_from_minted docstring: "`name` is any of `MINTED_CONFIGS`" — callers pass configs outside that tuple.
- tests/test_run_phantom_beat.py: two "mutation arms" are constant-`None` branches that cannot fail (see -15).

## PARKED
none

## HANDOFF
- train/src slice (PZ-2): src/mantis/train/coordinator/config.py::TrainerLike.train_step_from_tensors and ::GridRouteBufferLike declare the retired grid route; the real Trainer lacks the method; dispatch.py's module docstring still names `_grid_step` / `train_step_from_tensors`.
- tests/train + tests/config slices: the other ~14 private `_Pool`/`_Buffer` composition fakes (CARD-MECHANISM-SWEEP); `_fake_disk_usage` ×3 (tests/config/test_disk_guard_keys.py, tests/train/test_lifecycle_contract.py, tests/train/test_terminal_eval_rc.py); `_await_signal` (test_terminal_eval_rc); `_alive` (tests/train/test_parent_death_signal.py); `_top_level_imports` twin of tests/monitor/test_monitor_census.py in tests/train/test_train_import_dag.py, whose `mantis.eval` half is subsumed by tests/test_run_composition.py::test_no_train_module_imports_eval_even_lazily.
- tests/tools + tests/config slices: `_code_text` copies (tests/config/test_armed_abort_manifest.py, tests/tools/test_preflight_{child_convergence,mint,parent_census}.py).

## Not covered
- The 14 torch-importing modules could not collect or run here; their behavioural rows were read, not executed.
- Bodies of tests/monitor/test_supervisor{,_config_witness,_signal_posture,_spawn_contract}.py, test_arm_exec_trampoline.py and tests/test_run_pdeathsig.py read at helper/outline level only (LAW-16, lane C either way).
- tests/test_fixtures_manifest.py (PZ goldens) and tests/test_line_endings.py read at outline level; no deletion probes run (reviewer's job).
- Assertion-level overlaps with no count change were left out (e.g. strict's burst-parser half vs the launcher flag-set census; eval_enabled's run.py CLI half; meta_ci's ci.yml needles vs the orphan census — ci.yml is suspended, R348(a)).
