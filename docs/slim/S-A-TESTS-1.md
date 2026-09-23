# S-A-TESTS-1 — T1 (tests/train helpers + test_a* … test_g*)
scope: the 46 files `git ls-files tests/train | grep -E '^tests/train/([^t]|t[^e]|te[^s]|tes[^t]|test[^_]|test_[a-g])'` (12 031 lines, `xargs wc -l`) at origin/dev 69e1532; method: I read every file statically (torch is not importable: `tests/train/conftest.py` imports torch, so nothing under tests/train collects, and 35 of 39 test modules fail even under `--noconftest`). I used throwaway AST scripts in the scratchpad to measure spans and body hashes of duplicated helpers, to verify imports resolve (`t1_imps2.py`: every `from mantis… import X` in the slice is bound, rc 0), and to find names left unused inside their own file. I ran `.venv/bin/ruff check --isolated --select F401,F841,F811`, used `git grep` for callers over the whole tree, and checked `tools/ci_gates/tier_declaration.txt` for rows.

## Summary
- Counts: TEST 15 (lane A 5, lane B 10) · DUP 6 (lane B 4, lane C 2) · DOC 2 (lane A 1, lane B 1). 23 findings in total.
- Carded, marked and not re-raised (CARD-MECHANISM-SWEEP):
  - the private `_Pool`/`_Buffer`/`_Trainer`/`FakePool`/`FakeBuffer`/`FakeTrainer`/`_RunnerStats` fakes in 14 slice files.
  - `runN` tokens: five test names in `test_graph_microbatch_authority.py`, `test_graph_microbatch_bound.py::_run5_caps`, `test_checkpoint_conformance.py::test_a_run8_shaped_stamp_with_the_retired_rung_rows_still_loads_and_says_so`.
  - `conftest.py::_make_*_block` and `make_run_config`, which may be among the card's nine full-config literal dicts.
- Test-floor effect of every TEST(c) deletion proposed below: −20 collected (5112 → 5092, which is still above the floor of 4862, so gate 3c stays green and `test_count_ratchet_down.txt` is not needed). None of the proposed tests has a row in `tier_declaration.txt` (`grep -n "tests/train/(_|conftest|test_[a-g])"` returns only rows 86–91, and none of those is proposed), so no row goes stale.
- Top 3 by Δlines: S-A-TESTS-1-05 (−205 gross / −172 net, coordinator-harness preamble), S-A-TESTS-1-06 (−123 gross, private sink spies), S-A-TESTS-1-16 (−45, drain-hardcap twins).

## Findings

### S-A-TESTS-1-01 | TEST | A
subject: tests/train/test_actor_lag_wiring_live.py::_Buffer, tests/train/test_actor_sync_production_posture.py::_Buffer, tests/train/test_actor_sync_real_config.py::_Buffer
claim: All three classes are defined and never referenced. The drives pass `mk_graph_buffer(...)` (the root-conftest fixture) instead.
evidence: `grep -c -w _Buffer <file>` → 1 in each file (the `class` line only).
callers: AST imports: `git grep -n -w _Buffer -- tests` → each hit is a same-file `class _Buffer` or a use inside another file's own class; no cross-file import (R5: test modules are not imported by name) · entry points / `-m` / subprocess / importlib: not applicable to a test-private class (CALLERS §1, §4, §6, §7 list no tests/train module) · conftest/plugins: no fixture named `_Buffer` · pyo3, config keys, gate tool paths, STATE procedures: not applicable (none names a test-private class). Each search is `-w` over the whole of tests/, so it would find any reference by name.
Δlines: −24 (AST span 6 × 3, plus 2 PEP 8 separator lines × 3)
witness: the three modules' collection (needs torch); `ruff F821` would red on any surviving reference
depends: —

### S-A-TESTS-1-02 | TEST | A
subject: tests/train/test_anchor.py::_core_and_optional_keys
claim: The helper has zero callers. Its docstring-era "optional-subset twin" test was deleted, and the file itself says "The optional-subset twin is gone".
evidence: `git grep -n _core_and_optional_keys` → only its own `def` in test_anchor.py
callers: AST imports, attribute access, string lookups: `git grep -n _core_and_optional_keys -- .` → 1 hit (the def) · no conftest, entry-point, subprocess, pyo3, config or STATE channel can reach a test-private helper
Δlines: −7 (AST span 5 + 2 separators). test_anchor.py stays over 300 lines (311 → 304), so its R8 header stays.
witness: NONE (dead code)
depends: —

### S-A-TESTS-1-03 | TEST | A
subject: unused imports and a dead local in 10 non-protected slice files
claim: ruff finds 22 whole-line unused imports and one dead 2-line local. Two further `# noqa: F401 — RED-at-import anchor` imports are dead now that their modules exist.
evidence:
- `.venv/bin/ruff check --isolated --select F401,F841 <slice>` → 24 hits.
- Whole lines: test_actor_lag_wiring_live `inspect`; test_bc_heldout_stop `math`; test_bc_pretrain_cli_stopflags `json`, `Path`; test_bc_pretrain_dtype `torch`; test_cluster_stat_wiring `runner_stats`; test_graph_microbatch `inspect`, `lookup`, `build_net`, `dispatch_mod`, `Trainer`; test_graph_microbatch_authority `hashlib`, `json`, `platform`, `sys`, `Any`, `torch`, `ArchScopedKeyOutsideItsArchError`, `lookup`, `Trainer`.
- test_eval_heartbeat.py::test_stale_eval_poller_fires_42_under_fake_clock carries the F841 local `registry`.
- The anchors: test_actor_sync_isolation.py `import mantis.train.actor_sync`, and test_eval_result_routing.py `import mantis.eval.pipeline`, whose only occurrence is that line.
- test_checkpoint_conformance.py has 2 more such lines; that file is lane C and they are listed there, not here.
callers: not a DEAD-symbol claim; ruff's F401 is the static proof
Δlines: −24 (22 import lines + 2 lines of the F841 assignment, from ruff's line list)
witness: ruff F401/F841 (not in gate 14's select, so no gate reds today)
depends: —

### S-A-TESTS-1-04 | TEST | A
subject: the `cluster_value_std_mean` / `cluster_policy_disagreement_mean` / `cluster_variance_sample_count` attributes on the runner-stats stubs in tests/train/_coordinator_pool.py, test_abort_exit_signal, test_actor_lag_wiring_live, test_actor_sync_production_posture, test_actor_sync_real_config, test_clean_stop_save, test_coordinator_gates, test_eval_result_routing, test_gate_interval_decoupling (and test_drawrate_abort_threading, which is lane C)
claim: The subject is deleted: R346(f) and R250 removed the getters and every src reader, so these stub attributes feed nothing. test_cluster_stat_wiring.py keeps its own copies on purpose, because it asserts their absence from the payload.
evidence:
- `git grep -n -E "cluster_(value_std_mean|policy_disagreement_mean|variance_sample_count)" -- src tools crates` → 0 code hits (one comment in crates/mantis-bridge/src/runner.rs).
- docs/contracts/event_manifest.md names them as retired.
- `grep -c` over the slice without the wiring test → 29 lines.
callers: src read path: `git grep` as above → none · getattr-by-string: `git grep -E "[\"']cluster_" -- src tools` → only unrelated `cluster_window_size`/`cluster_threshold` · no `vars(rstats)`/`asdict(rstats)` in src (`git grep` returns none)
Δlines: −28 in slice (29 lines, minus the 1 line of _coordinator_pool.py that also carries the live `mcts_*` fields). About 29 files tree-wide.
witness: the coordinator suites (need torch)
depends: rides CARD-MECHANISM-SWEEP if that card consolidates the fakes first

### S-A-TESTS-1-05 | DUP | B
subject: the StepCoordinator-harness preamble. `_filled_hexg`, `_GRAPH_FULL_CONFIG`, `_mirrored`, `_DRAIN_CAPS`/`_KNOBS`/`_GATE_INTERVAL` in test_abort_exit_signal, test_actor_deploy_independence, test_clean_stop_save, test_cluster_stat_wiring, test_coordinator_gates, test_eval_result_routing, test_gate_interval_decoupling (lane C part: test_drawrate_abort_threading, test_drawrate_gate_branch_flipset, test_drawrate_gate_capacity)
claim: The copies are identical in body.
- `_filled_hexg` has 13 copies tree-wide, all body hash `9038bac3`. It is the same builder as tests/conftest.py::mk_graph_buffer's inner `make` (same stones, policy and push arguments).
- `_GRAPH_FULL_CONFIG` has 10 copies, hash `35af3759`.
- `_mirrored` has 4 copies, hash `ce5b3d26`.
- The dev_example drain/knobs/gate triple has 13 semantically equal copies.
A torch-free shared home already exists (tests/train/_coordinator_pool.py, or tests/_drivable.py next to mk_graph_buffer).
evidence: `t1_span.py` hashes as quoted; `t1_dupspan.py` over the 10 slice files → TOTAL 205 lines (non-protected 173, drawrate files 32)
deliberate?: The files say they duplicate only because "R5 bars cross-test imports". That is false: R5 bars `sys.path` writes and a `tests` package. Bare-name helper modules are live: `_microbatch_harness` is imported by 22 files and `_coordinator_pool` by 2. REVIEW2 G3.4 flagged the same false premise in tests/test_run_strict_composition.py. These are not oracles or twins; no test compares two copies.
Δlines: −205 gross (dupspan). Net −172 = gross − one retained copy (8+7+5+3 = 23 lines, the smallest spans) − one import line per file (10).
witness: the 10 modules' collection and their drives (need torch)
depends: S-A-TESTS-1-23

### S-A-TESTS-1-06 | DUP | B
subject: the private event-sink spies. `_Sink`/`_SpySink`/`SpySink` in 15 slice files (abort_exit_signal, actor_lag_watchdog, actor_sync, augment_sym_counter, clean_stop_save, cluster_stat_wiring, coordinator_gates, disk_guard_failure_is_visible, eval_heartbeat, eval_result_routing, gate_interval_decoupling; lane C part: draw_rate_is_a_fraction, drawrate_abort_threading, drawrate_gate_branch_flipset, drawrate_gate_capacity). Also tests/train/conftest.py::SpyEventSink next to tests/train/_microbatch_harness.py::SpySink, and test_drain_hardcap_wiring.py::FakeClock.
claim: Every copy is the same recorder (`events` list, `emit` appends `dict(event)`, optional `named`). The directory already has TWO shared copies: the conftest `spy_sink` fixture, which no slice test takes, and `_microbatch_harness.SpySink`. FakeClock repeats conftest.FakeClock plus tests/monitor/conftest.py::FakeClock.
evidence: `t1_sinks.py` over the slice → 17 classes, 148 lines; the 15 private copies = 123 lines (hash differences are annotations only)
deliberate?: none is compared to another; the train conftest docstring states the fixtures exist so every suite draws "from ONE place"
Δlines: −123 gross (private copies; lane C part −27). The replacement cost is one torch-free import or a fixture parameter per file, because `_microbatch_harness` imports torch.
witness: the suites' collection (need torch)
depends: —

### S-A-TESTS-1-07 | TEST | B
subject: tests/train/test_actor_sync_real_config.py::test_a_real_run_config_actually_reaches_the_cadence_resolver, ::test_lag_callables_read_live_sources_under_a_real_config
claim: The file's premise is retired. `src/mantis/run.py::_resolve_actor_sync_cadence_steps` says "The retired smoke arm…", and the sibling suites already compose with a real `RunConfig` (tests/conftest.py::make_run_config_from_minted). Test 1 is subsumed by tests/test_run_strict_composition.py::test_every_minted_config_resolves_through_every_composition_seam, which asserts `_resolve_actor_sync_cadence_steps(cfg) == cfg.train.actor_sync_cadence_steps` for every minted config. Test 3 is subsumed by test_actor_lag_wiring_live (::test_actor_ckpt_step_fn_reads_the_live_sync_engine, ::test_learner_step_fn_reads_the_live_trainer), and its last assertion reduces to `trainer.step - actor_before == trainer.step - actor_before`. Test 2 (cadence 2) is the only compose-level witness that a cadence ≠ 1 reaches ActorSync, so KEEP it. It could fold into test_actor_sync_production_posture, which has the same `_bounded_config`/pool harness (hashes `564f5277`, `533005ea`).
evidence: `sed` of both run.py docstrings; `grep` of tests/test_run_strict_composition.py (in the seam test); the docstring's "every other drive … composes with config=SimpleNamespace()" is contradicted by the `smoke_run_config` fixture
Δlines: −38 (AST spans 7 + 27, plus 4 separators); collected −2
witness: test_run_strict_composition seam test; test_actor_lag_wiring_live
depends: —

### S-A-TESTS-1-08 | TEST | B
subject: tests/train/test_actor_lag_wiring_live.py::test_composition_root_supplies_both_lag_callables
claim: The test asserts `name in captured` and `callable(captured[name])`. Every sibling in the file calls `captured["…_fn"]()` and asserts on the value, so a missing or non-callable entry reds all of them (KeyError / TypeError).
evidence: the file's own 5 sibling drives, read above
Δlines: −7 (span 5 + 2); collected −1
witness: ::test_the_two_lag_callables_are_not_swapped
depends: —

### S-A-TESTS-1-09 | TEST | B
subject: tests/train/test_anchor_wiring.py::test_publication_is_in_place_so_prebuilt_holders_see_it, ::test_no_anchor_passed_still_binds_the_resolved_one
claim: In the first test, `holder_a`/`holder_b` are aliases of `shared` that the test itself assigns, so its checks are those of ::test_resolved_anchor_is_published_onto_the_callers_object plus a tautological `is`. The second test asserts nothing. The binding it names is a dead store: after `anchor_state = resolved`, `src/mantis/train/loop.py` never reads `anchor_state` again (`awk 'NR>=85' loop.py | grep anchor_state` → none). The same no-anchor call path is also driven, with an assertion, by tests/eval/test_promotion_integrity.py.
evidence: the file as read; the loop.py read
Δlines: −25 (spans 16 + 5, plus 4); collected −2
witness: ::test_resolved_anchor_is_published_onto_the_callers_object
depends: —

### S-A-TESTS-1-10 | TEST | B
subject: tests/train/test_anchor.py::test_anchor_state_carries_declared_representation
claim: The test runs the same save and load as ::test_load_best_model_resilient_loads_valid_anchor, which already asserts `representation == "graph"`. It then wraps the tuple in `AnchorState`, a plain 4-field dataclass (src/mantis/train/anchor.py::AnchorState), and reads the field back.
evidence: both bodies read; `grep -n "class AnchorState" -A 8`
Δlines: −13 (span 11 + 2); collected −1
witness: ::test_load_best_model_resilient_loads_valid_anchor
depends: —

### S-A-TESTS-1-11 | TEST | B
subject: tests/train/test_eval_heartbeat.py::test_eval_round_is_a_registered_heartbeat_source, tests/train/test_actor_lag_watchdog.py::test_actor_lag_is_not_a_heartbeat_source
claim: tests/monitor/test_heartbeat.py::test_heartbeat_sources_name_pins asserts the exact 4-tuple. That implies both of the first test's assertions (`"eval_round" in`, `len == 4`) and both of the second's (the same tuple, and no `actor` substring).
evidence: `git grep -n "HEARTBEAT_SOURCES ==" -- tests` → 4 exact-tuple pins (monitor/test_heartbeat, test_run_phantom_beat ×2, train/test_actor_lag_watchdog)
Δlines: −18 (spans 8 + 6, plus 4); collected −2
witness: tests/monitor/test_heartbeat.py::test_heartbeat_sources_name_pins
depends: —

### S-A-TESTS-1-12 | TEST | B
subject: tests/train/test_eval_heartbeat.py::test_monitor_config_carries_eval_round_deadline
claim: ::test_build_run_safety_arms_eval_round_deadline passes `MonitorConfig()` and asserts `deadlines["eval_round"] == 1800.0`. `src/mantis/train/subsystems.py` fills that entry from `cfg.heartbeat_deadline_eval_round_sec`, so the default is asserted there too.
evidence: `git grep heartbeat_deadline_eval_round_sec -- src` → monitor/config.py default, subsystems.py read
Δlines: −9 (span 7 + 2); collected −1
witness: ::test_build_run_safety_arms_eval_round_deadline
depends: —

### S-A-TESTS-1-13 | TEST | B
subject: tests/train/test_bc_heldout_stop.py::test_pb8_the_train_ring_hazard_is_DEFENDED_where_the_ring_is_CHOSEN
claim: The test calls itself "A POINTER". Its one real assertion (`"split_part" in inspect.getsource(cli.pretrain)`) is made by test_bc_pretrain_cli_stopflags.py::test_the_split_part_guard_is_WIRED_and_names_the_train_ring_hazard, which checks more. The other assertion, `m.ring is not None`, is a tautology: `_monitor` passes `ring=object()`.
evidence: both bodies read
Δlines: −19 (span 17 + 2); collected −1
witness: test_bc_pretrain_cli_stopflags::test_the_split_part_guard_is_WIRED_and_names_the_train_ring_hazard
depends: —

### S-A-TESTS-1-14 | TEST | B
subject: tests/train/test_bc_pretrain_cli_stopflags.py::test_a_PARTIAL_stopping_rule_is_refused (6 cases)
claim: The assertions are made only about the test's own parametrize input (any flag set, not all flags set). The all-or-none `SystemExit` lives in `src/mantis/train/pretrain/cli.py::pretrain`, which the test never calls. So the 6 cases pass with the refusal deleted, and the refusal has no witness (see DEFECTS).
evidence: `grep -n "_stop_flags" src/mantis/train/pretrain/cli.py` → the check is inside `pretrain()`, not `_build_arg_parser`
Δlines: −17 (span 15 + 2); collected −6
witness: NONE (this is the defect)
depends: —

### S-A-TESTS-1-15 | TEST | B
subject: tests/train/test_graph_microbatch_authority.py::test_of2_8_run5s_caps_are_typed_and_inside_the_schema_range (carded `runN`; also pins `run6.yaml` by name, the R10 census class)
claim: ::test_n1_run5_is_ARMED_with_a_sized_cap_not_the_templates_non_binding_default[run6.yaml] loads the same file through `load_config`, which enforces the schema's int `Field(ge=1)` (src/mantis/config/schema/train.py::MicrobatchCapsConfig). The resolver `int()`-casts, so the typed-and-≥1 test can only red where test_n1 also reds.
evidence: schema fields and `resolve_microbatch_caps` read
Δlines: −9 (span 7 + 2); collected −1
witness: ::test_n1_run5_is_ARMED_…[run6.yaml]
depends: —

### S-A-TESTS-1-16 | TEST | B
subject: tests/train/test_drain_hardcap_wiring.py::test_drain_within_budget_is_not_broken, ::test_terminal_round_bounded_by_terminal_eval_hard_cap_sec
claim: The first test's setup (`exitcode = 0`) and all three assertions (`is None`, no terminate, no kill) repeat the clean half of ::test_drain_or_kill_returns_a_typed_reason_or_none. The second passes `budget_sec=caps.terminal_eval_hard_cap_sec` in itself, so it tests only `drain_or_kill`'s overrun arm, whose assertions (`"join_timeout"`, terminate and kill called) ::test_drain_overrun_kills_worker_and_yields_eval_broken already makes. That the terminal cap is wired is covered by ::test_all_four_drain_cap_fields_have_live_consumers.
evidence: the four bodies read
Δlines: −45 (spans 20 + 21, plus 4); collected −2
witness: ::test_drain_or_kill_returns_a_typed_reason_or_none, ::test_drain_overrun_kills_worker_and_yields_eval_broken
depends: —

### S-A-TESTS-1-17 | DUP | B
subject: tests/train/test_actor_lag_sample_emission.py::_registry, ::_spec, ::_watchdog vs tests/train/test_actor_lag_watchdog.py (same names)
claim: `_registry` is identical (hash `ee4c1337`) and `_spec` has the same body (`2db7120b`). `_watchdog` differs only in `file_interval_sec`/`clock` parameters. One suite could hold both, or the helpers could move to a helper module.
evidence: `t1_span.py` output (spans 5/5, 3/5, 12/15)
deliberate?: The emission file says the watchdog file is "byte-frozen" (`:64`). That is an oracle-phase freeze and no ruling names it.
Δlines: −20 gross (the emission file's copies, 5 + 3 + 12). A merged file would be 488 lines and need an R8 header.
witness: both suites
depends: —

### S-A-TESTS-1-18 | DUP | C
subject: tests/train/test_bc_warm_start_entry.py::_arch, ::_write_source vs tests/train/test_f32_launch_pin_wiring.py (same names). PZ-1: net-param hash on the warm start.
claim: The two files use the same narrow arch (`hidden=32, num_layers=2, …`) and the same `save_checkpoint` source writer. Both already import `_warmstart_config.minimal_config`, which is the natural shared home.
evidence: `t1_span.py` → `_arch` 6/6, `_write_source` 14/14 (bodies differ only in `_spec()` vs `lookup`, the `run_id` kwarg and the `Path()` wrap)
deliberate?: no; neither file compares against the other
Δlines: −20 gross (one file's copies)
witness: both PZ-1 suites
depends: —

### S-A-TESTS-1-19 | DUP | C
subject: tests/train/test_finite_gradient_guard.py::_graph_step vs tests/train/test_nonfinite_guard.py::_graph_step (T2), plus the paired healthy-step and "five keys" tests. PZ-1: finite-gradient guard.
claim: `_graph_step` is body-identical (hash `f3b355cf`). Each file has its own healthy-step mutation half. Both "…stays_five_keys" tests assert a SEVEN-key set, and the healthy-path copy (T2) repeats test_graph_microbatch.py::test_of2_4_one_optimizer_step_and_seven_keys_at_every_m[1]. Also, inside test_of2_4 itself, the `for key in (…)` loop duplicates the `set(r.info) == {…}` assertion below it.
evidence: `t1_span.py` (14/15 lines); the bodies read
deliberate?: no twin/oracle claim in either header
Δlines: −14 (one `_graph_step` copy); the rest is left to the ruling
witness: both PZ-1 suites
depends: —

### S-A-TESTS-1-20 | DUP | B
subject: tests/train/conftest.py::make_tiny_arch vs tests/train/_microbatch_harness.py::tiny_graph_arch
claim: Both build the same `GnnArch(in_dim=…, edge_dim=…, hidden=16, num_layers=1, policy_hidden=16, value_hidden=16)` in the same directory. The conftest already imports from `_microbatch_harness` (`graph_hparams`), so it could import this one too.
evidence: `git grep -n -E "def (tiny_graph_arch|make_tiny_arch)"`; T2 has 3 more inline copies (HANDOFF)
deliberate?: no
Δlines: −3 (make_tiny_arch body 5 lines replaced by a 2-line delegation)
witness: tests/train/test_checkpoint_conformance.py (tiny_arch users)
depends: —

### S-A-TESTS-1-21 | TEST | A
subject: tests/train/conftest.py::mk_optim (fixture)
claim: No test takes this fixture as a parameter, names it as a string, or requests it through `usefixtures`.
evidence: `git grep -n -w mk_optim -- .` → the def, plus a listing line in docs/slim/00_MAP.md (inventory, not a caller)
callers: fixture by parameter name: `git grep -w mk_optim -- tests` → the def only · `getfixturevalue`/`usefixtures` strings: `git grep -n -E "getfixturevalue|usefixtures" -- tests` → `preflight_stamped`/`local_puller` only · no plugin or entry point exists (CALLERS §1, §8) · AST imports: conftest is never imported by name
Δlines: −5 (span 3 + 2). conftest.py drops from 301 to 296 lines, so its `>300 justify` docstring paragraph must go under R8 (gate 15, third rule).
witness: NONE
depends: —

### S-A-TESTS-1-22 | DOC | A
subject: oracle-phase text that no longer matches the tree, in tests/train/test_actor_sync.py, test_actor_sync_isolation.py, test_actor_deploy_independence.py, test_actor_lag_watchdog.py, test_eval_heartbeat.py, test_eval_result_routing.py, test_eval_kick_burst.py
claim: The files still say "RED-at-import until IMPL lands …", "module does not exist yet" and "Byte-frozen through IMPL", but every named module exists (`t1_imps2.py` rc 0). test_eval_kick_burst's module docstring describes the `% eval_interval` rule that its own third test says B-3/R355(e) replaced. test_graph_microbatch::test_of2_4's docstring says "all five keys" while the test asserts seven.
evidence: `grep -n -E "RED-at-import|until IMPL|does not exist yet|Byte-frozen"` → 10 lines across 6 files
Δlines: −5 (whole docstring lines test_actor_sync.py 3–4, test_actor_sync_isolation.py 5, test_actor_deploy_independence.py 5–6); the rest are trailing-comment edits (Δ 0)
witness: NONE
depends: S-A-TESTS-1-03 (the two anchor imports)

### S-A-TESTS-1-23 | DOC | B
subject: the R8 justification headers of tests/train/test_abort_exit_signal.py and test_clean_stop_save.py, the `_filled_hexg` docstrings in test_coordinator_gates.py and test_gate_interval_decoupling.py, and the headers of test_clean_stop_save.py ("the ten rows") and test_graph_microbatch_authority.py ("the four rows")
claim: The first two headers justify the file size with "R5 bars cross-test imports", which is false (see S-A-TESTS-1-05; REVIEW2 G3.4 is the same class). The row tallies are wrong: the files hold 14 and 10 tests (`grep -c "^def test_"`). These are transcribed counts, the SF-7 class gate 15 targets, although gate 15 only catches line counts.
evidence: `git grep -n "R5 bars" -- tests/train`; the two `grep -c` results
Δlines: 0 (rewording); the two headers become moot if S-A-TESTS-1-05 lands and the files still exceed 300 lines
witness: gate 15 (presence only)
depends: S-A-TESTS-1-05

## DEFECTS
- tests/train/test_bc_pretrain_cli_stopflags.py::test_a_PARTIAL_stopping_rule_is_refused: nothing tests R328(d)'s all-or-none refusal. The `SystemExit` is in `cli.pretrain`, and no test calls it with a partial set.
- tests/train/test_abort_exit_signal.py::test_an_abort_with_no_authored_code_resolves_to_None[sealbot_wr_abort] names a rule R362(c) deleted. The docstring ("both share `_fire_hard_abort`") is false and the premise check is vacuous (no row). Even so, it is the ONLY witness of `exit_code_for_abort`'s no-row branch, so it should be re-pointed at a synthetic unknown name, not deleted.
- tests/train/test_ema_lever_is_reachable.py::test_the_trainer_builds_an_ema_model_only_when_the_config_arms_it ends on `ema.update_parameters(net)` with no assertion that the shadow moved (the comment says it does), and never drives the disarmed arm that its "only when" claims.
- tests/train/test_eval_heartbeat.py::test_poller_thread_beats_eval_round writes to the fixed `/tmp/mantis-eval-heartbeat-test` spool/record dirs instead of `tmp_path` (so parallel runs collide) and gates on `time.sleep(0.05)`.
- "five keys" test names assert seven keys: test_finite_gradient_guard.py::test_the_loss_info_contract_stays_five_keys_on_a_skipped_step (and T2's test_nonfinite_guard twin).
- The `resolve_anchor` stubs in test_actor_sync_production_posture.py and test_actor_sync_real_config.py return `representation="grid"`, a representation R346(f) deleted.
- test_gate_interval_decoupling.py::test_p6b's docstring says "The count ratchets in BOTH directions", but the test asserts no count.

## PARKED
- test_bc_graph_reroute.py: the three `_BURIED` census tests each re-parse every `.py` under src/, tools/ and tests/ (three full-tree AST walks; PZ-6 lane C).
- test_actor_lag_wiring_live.py runs `compose_run` 7 times (5 tests plus 2 params) for assertions that one captured drive could make.

## HANDOFF
- src/mantis/train/anchor.py::_OPTIONAL_HEAD_PREFIXES lists dense-era heads (opp_reply_*, value_var, ownership/threat/chain/ply_index heads, input_channel_index) that a graph net lacks (the test_anchor comment says so). This is PZ glob, lane C → src/mantis/train slice.
- src/mantis/train/loop.py::run_training_loop: `if anchor_state is None: anchor_state = resolved` is a dead store → src/mantis/train slice.
- The unused `_Buffer` class in tests/config/test_drain_caps_wiring.py (T4) and in tests/test_actor_sync_composition.py (T7) (the `grep -c -w _Buffer == 1` sweep).
- tests/test_run_phantom_beat.py::test_all_four_sources_are_covered_by_the_census asserts the same tuple twice (tuple, then set); it duplicates tests/monitor/test_heartbeat.py::test_heartbeat_sources_name_pins (T7).
- T2: the S-A-TESTS-1-05 preamble copies in test_inference_seam_events, test_iteration_complete_decoupling, test_target_counter_events, test_terminal_eval_rc (which also carries the "R5 bars" header), test_ply_cap_gate and test_policy_loss_trough_gate; tiny-arch copies in test_periodic_checkpoint, test_server_owned_copy and test_train_step_dispatch; test_nonfinite_guard's five-key test ⊂ test_graph_microbatch::test_of2_4[1] (lane C).
- tests/config/test_actor_sync_schema.py (T4) is loaded BY PATH from test_actor_sync_real_config::_frozen_payload. If T4 renames `_payload`, S-A-TESTS-1-07's kept test breaks.

## Not covered
- Nothing was executed except the 7 torch-free modules' collection (`--noconftest`: 39 tests collected, 35 modules error on torch). No drive-level claim is probe-verified. Every "subsumed" claim rests on reading both bodies.
- test_checkpoint_conformance.py (770 lines, the Suite A lane-C ambiguity in PZ-1), test_graph_microbatch.py (676) and test_graph_microbatch_bound.py (423) were read only at the level of their test lists, headers and imports, not assertion by assertion. The same goes for the bodies of test_clean_stop_save.py, test_coordinator_gates.py and test_gate_interval_decoupling.py beyond the pairs compared above.
- The protected draw-rate files (test_drawrate_*, test_draw_rate_is_a_fraction), test_f816_37_train_path_dump.py and test_firing_halt_resume_witness.py were checked for helper duplication only (lane C regardless).
- I did not re-derive how CARD-MECHANISM-SWEEP counts its 21 `_Pool`/`_Buffer` fakes; those fakes are marked carded, not re-inventoried.
