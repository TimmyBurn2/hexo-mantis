# S-A-TESTS-3 — T3: tests/tools/**
scope: tests/tools/** (73 files, 14 286 lines; `git ls-files tests/tools | xargs wc -l`). Subjects: tools/*.py, tools/{analyzer,dashboard,ladder,probe1,viewer}, tools/ci_gates/**.
method: `pytest --collect-only -q -m ''` over the slice (840 collected, 5 collection errors, all `No module named 'torch'`: test_bench_server, test_ladder_backends, test_ladder_openings, test_preflight_mint_process, test_preflight_start_halts); AST span scripts for every Δ; `npx --yes jscpd@4` (min-lines 5 and 8) for clones; AST scans for repeated fixtures and for helpers calling `spec_from_file_location`; `git grep` over the tree for callers, ruling/card names and tier_declaration rows; one probe of gate 8 in a scratch dir; `pytest` on test_registry_gate.py (2 passed).

## Summary
15 findings. By class: DUP 8 (lane B 4, lane C 4), TEST 5 (lane C 5), SIMPLIFY 2 (lane A 1, lane C 1). By lane: A 1, B 4, C 10.
Top 3 by own Δ: **02** −106 (ci-gate and frozen-oracle loaders), **01** −78 (tool-module loaders), **03** −59 (merge the strength_frontier trio). The largest test removal is **14**: test_audit_bootstrap_corpus.py, −654. It depends on S-A-TOOLS-2-01, and that finding's −1 793 already includes these lines, so do not add the two.
Gate 3c: 5 112 collected against the floor of 4 862, so no deletion here needs a floor-file move. Deletions still lower the count, so they are lane B or C. tier_declaration.txt has rows for tests/tools; only **13** removes a declared test (its `skip` row). tools/ci_gates/** is in PZ, which forces lane C.
Frozen-oracle note: RULINGS R310 grants a "frozen edit" on tests/tools/test_preflight_mint_process.py. test_preflight_parent_census.py calls test_preflight_mint.py "the byte-frozen oracle". R43 (standing) says any edit to a byte-frozen oracle queues. Every finding that touches either file is lane C.

## Findings

### S-A-TESTS-3-01 | DUP | B
subject: per-file by-path loaders of ONE tools/ module in 15 non-gate test files: test_audit_bootstrap_corpus::_load_tool, test_bench_server::bench, test_ladder_backends::_minimal_config, test_ladder_bot::bot_mod, test_mint_opening_book::minter, test_mirror_pull::_tool, test_probe1::_package, test_run_dashboard::shim, test_select_balanced_book::selector, test_strength_frontier_{book,sigma,strix}::frontier, test_strix_follower::follower_mod, test_strix_net_only_cell::_load, test_strix_ruler_r6_cell::_load
claim: each file re-types the same 6–12-line `spec_from_file_location` → `module_from_spec` → `sys.modules[...]` → `exec_module` block. One bare-name helper module can hold it, the way tests/tools/_ladder_stub.py is already imported (`from _ladder_stub import ...`).
evidence: AST scan for non-test functions that call `spec_from_file_location` → 33 helpers in 33 files (263 lines). jscpd min-lines 5 → 36 clones, most of them this block (e.g. test_strix_net_only_cell 15–30 ↔ test_strix_ruler_r6_cell 14–29, 16 lines; test_strength_frontier_book 16–32 ↔ _strix 16–32, 17 lines).
deliberate?: not a seam or oracle. Every docstring gives the same reason (R5: no sys.path write), so this is one mechanism copied, not a second authority. The load names differ per file; the helper takes the name.
Δlines: −78. Derivation: 103 lines removed across the 15 helpers (AST spans; a fixture keeps its decorator, def and a one-line `return`), +15 import lines, +10 for the helper. test_probe1::_package is a package loader, which S-A-TOOLS-2-07 also counts.
witness: collection of each migrated file; gate 3c count unchanged.
depends: S-A-TOOLS-2-07 (package-loader twin; one helper can serve both); overlaps 03.

### S-A-TESTS-3-02 | DUP | C
subject: the same loader block in 18 files whose subject is tools/ci_gates/** (the LAW-07 gate self-tests test_comment_lint, test_encoding_io_gate, test_r8_header_gate, test_rule7_gate, test_silent_encoding_gate, test_tier_census, test_gate_vacuity, test_contract_doc_gate, test_drawrate_arming_surface_named_failure [PZ-1]) or the preflight tool, including the frozen test_preflight_mint.py and test_preflight_mint_process.py
claim: the mechanism is the same as 01, but these files are LAW-07 self-tests, PZ-named files or byte-frozen oracles (R43/R310).
evidence: the same AST scan as 01; jscpd e.g. test_r8_header_gate 27–38 ↔ test_rule7_gate 27–38 (12 lines), test_encoding_io_gate 22–33 ↔ test_rule7_gate 27–38.
deliberate?: as in 01; nothing in these files depends on the loader's shape.
Δlines: −106. Derivation: 124 lines removed (AST spans, same rule as 01), +18 import lines. The helper is shared with 01.
witness: every gate self-test in the list (collection).
depends: S-A-TESTS-3-01

### S-A-TESTS-3-03 | DUP | B
subject: tests/tools/test_strength_frontier_book.py, test_strength_frontier_sigma.py, test_strength_frontier_strix.py
claim: three files for one tool (tools/strength_frontier.py). Each opens with the same imports, a `frontier` loader fixture and a `base` fixture, so they can merge into one module with one prologue.
evidence: AST prologue through the `base` fixture: 29 / 32 / 29 lines, against 75 / 56 / 52 lines total. jscpd: book 16–32 ↔ strix 16–32 (17 lines), sigma 18–30 ↔ strix 16–28 (13 lines). The sigma rows parametrize (0.1, True) and (1.0, False), which differ from both run6's and run7's `c_scale: 1.0 / q_rescale: true`, so the config the `base` fixture reads does not matter.
deliberate?: no. The `base` fixtures differ only in which minted config they name (run7, run6, run7), and each `frontier` fixture loads the same file.
Δlines: −59 (the sigma and strix prologues, 32 + 29, removed; +2 lines for the merged docstring). Collected count unchanged (4 + 3 + 2). The configs are named by run, which is CARD-MECHANISM-SWEEP's inventory; leave that to the sweep.
witness: the 8 merged test functions.
depends: overlaps 01 by the two dropped `frontier` fixtures (14 of 01's lines); the two Δs do not add.

### S-A-TESTS-3-04 | DUP | B
subject: module fixtures that load a tools/dashboard submodule, re-declared per file: `reader` → dashboard.reader in 5 files (test_dashboard_{external,health,reader,strength}, test_run_dashboard); `html` → dashboard.html in 2 (test_dashboard_external, test_run_dashboard); `external` → dashboard.external in 3 (test_dashboard_external, test_strix_net_only_cell, test_strix_ruler_r6_cell)
claim: 10 copies of a 3–5-line `importlib.import_module("dashboard.<x>")` fixture. The conftest already serves the `dashboard` package they all depend on.
evidence: AST scan for module-level fixtures defined in more than one file (tests/tools) → reader ×6 (one is viewer.reader), html ×4 (dashboard ×2, viewer, analyzer), external ×3.
deliberate?: no; each fixture returns the same module object. The viewer and analyzer `reader`/`html` fixtures stay module-local, since pytest lets a module fixture override a conftest one.
Δlines: −36 (54 lines of fixture spans with trailing blanks removed, by AST; +18 for three session fixtures in tests/tools/conftest.py).
witness: the dashboard tests named above.
depends: —

### S-A-TESTS-3-05 | SIMPLIFY | A
subject: tests/tools/conftest.py::load_dashboard_package
claim: a pure forwarder: its body is `return load_tools_package("dashboard")`, and the conftest `dashboard` fixture is its only caller.
evidence: `git grep -n load_dashboard_package` → conftest.py def + the `dashboard` fixture's `return load_dashboard_package()`, plus docs/slim/00_MAP.md (census text).
callers: (it is not DEAD, but every channel was ticked)
  AST imports: `git grep -n -w load_dashboard_package -- '*.py'` → conftest.py only (no test imports a conftest).
  entry points / python -m / Makefile / shell: `git grep -n load_dashboard_package -- Makefile '*.sh' '*.toml' docs` → 00_MAP.md only.
  subprocess strings: none (the same grep covers string forms: `git grep -n -E "[\"']load_dashboard_package[\"']"` → 0).
  importlib/getattr: the one conftest loaded by path (test_preflight_mint_process.py::test_the_probe_sweep_survives_a_SYMLINK...) reads only `PROBES`/`_sweep`.
  conftest/pytest: fixtures resolve by parameter name, and this is a plain function, not a fixture.
  pyo3 / config keys / gate tool paths / STATE procedures: not applicable; `git grep` in STATE.md → 0.
Δlines: −5 (def + docstring + return + 2 blanks, lines 153–157 by AST + blanks); the fixture's return becomes `load_tools_package("dashboard")`.
witness: every dashboard test (the `dashboard` fixture).
depends: —

### S-A-TESTS-3-06 | DUP | B
subject: `_code_text` (source with COMMENT/STRING/FSTRING_MIDDLE tokens stripped): tests/tools/test_preflight_child_convergence.py::_code_text, test_preflight_parent_census.py::_code_text, test_preflight_mint.py::_code_text (frozen), plus tests/config/test_armed_abort_manifest.py::_code_text and tests/test_run_one_authority.py::_code_text (out of slice)
claim: 5 copies of one 7–16-line tokenizer. A single helper at the tests root could serve all of them.
evidence: `git grep -n -E "^def _code_text" -- tests` → 5 hits; AST spans 9 / 10 / 9 / 16 / 7.
deliberate?: no; the docstrings differ, the token set is identical.
Δlines: −9 in this slice (the two non-frozen copies with blanks, 11 + 12 lines by AST, −23; +2 imports; +12 for the helper). −28 tree-wide without the frozen copy (51 − 9 frozen = 42 removed, +4 imports, +10 helper).
witness: test_preflight_parent_census.py censuses, test_preflight_child_convergence.py::test_the_tool_no_longer_builds_a_single_collaborator_for_itself.
depends: — (the frozen copy stays under R43; if 11 lands first, the child_convergence copy moves with it)

### S-A-TESTS-3-07 | DUP | C
subject: tests/tools/test_preflight_mint_process.py: `_P`, `_STEP_SEC`, `_SAMPLE_TS`, `_THRESHOLD`, `_SyncTarget`, `_real_syncs`, `_model_samples`, `_stream`, `_assertions`
claim: the process file re-declares test_preflight_mint.py's stream-builder rig. Values match, and the bodies match up to run-id tags. The one difference: the process file's `_SyncTarget` is a no-op, while the mint file's records.
evidence: `git grep -n -E "^def (_real_syncs|_model_samples|_stream|_assertions)\b|^class _SyncTarget|^_(P|STEP_SEC|SAMPLE_TS|THRESHOLD) = " -- tests` → exactly these two files. Side-by-side read: the same `ActorSync`/`JsonlEventSink` drive, the same `ts` re-basing, the same `shutdown_save` ground truth, the same `evaluate_assertions` call.
deliberate?: arguably. The process file's R8 header argues against forking its own `_mini_tree` rig, yet this rig is forked across the two files. Both files are frozen oracles (R43/R310), so a shared helper module is an adjudication item.
Δlines: −48 (lines 1326–1375, 50 lines by AST span through `_assertions`; +2 import lines).
witness: test_preflight_mint.py::test_each_mutation_flips_exactly_its_declared_predicates; process test_b0_needs_TWO_samples..., test_a4_discriminates_a_LOST_SINK_LINE...
depends: —

### S-A-TESTS-3-08 | TEST | C
subject: tests/tools/test_preflight_mint_process.py::test_a_child_rc_48_is_the_runs_own_ARMED_ABORT_and_is_never_collapsed_to_33
claim: other tests already make every assertion in this test.
- test_a_child_rc_46_... loops over ARMED_ABORT_CODES, asserts == (46, 47, 48), and checks for each code: propagation, rc ≠ 33, not in PASS_THROUGH, disjoint from WATCHDOG_CODES.
- test_the_failure_code_table_is_the_designs_table asserts RESERVED_CODES == (42…48).
- tests/config/test_armed_abort_manifest.py::test_the_terminal_eval_broken_row_is_required_and_imports_its_exit_code binds TERMINAL_EVAL_BROKEN_EXIT_CODE == 48.
evidence: `git grep -n -E "RESERVED_CODES|TERMINAL_EVAL_BROKEN_EXIT_CODE|ARMED_ABORT_CODES" -- tests tools src` → the three covering sites above; a read of all four test bodies.
Δlines: −33 (AST span 2955–2985, 31 lines, +2 preceding blanks; last function in the file). Collected −1; no tier_declaration row.
witness: the three covering tests.
depends: — (frozen oracle, R43/R310 → lane C)

### S-A-TESTS-3-09 | SIMPLIFY | C
subject: tests/tools/test_preflight_mint.py::_load_tool (the "RED anchor" branch) and its "RED-at-import anchor" / "RED anchor #2" / `_real_samples` "RED at HEAD by construction … returns []" text
claim: a compatibility shim for a state the tree no longer has. The branch raises "IMPL owes C-2" when tools/ci_gates/preflight_mint.py is absent, but the tool exists and the emission `_real_samples` waits for is live (R10: no shim for a gone state).
evidence: `git grep -n -i -E "RED anchor|RED-at-import|RED at HEAD|IMPL owes" -- tests/tools` → only this file; `git ls-files tools/ci_gates/preflight_mint.py` → present.
Δlines: −6 (the `if not TOOL_PATH.is_file(): raise …` block, `sed -n 51,56p`), plus comment and docstring rewording at 0 net.
witness: the file's collection (TOOL loads at import).
depends: — (frozen oracle → lane C)

### S-A-TESTS-3-10 | TEST | C
subject: tests/tools/test_registry_gate.py::test_trigger_arms_when_registry_present
claim: the test plants `crates/mantis-encoding/registry.toml`, the crate-root path CLAUDE.md warns has been mis-cited before. Gate 8 reads `crates/mantis-encoding/src/registry.toml` (registry_gate.sh `REG=`). So this "registry present" test actually runs the ABSENT arm, which test_hard_fail_when_registry_absent already covers. Its one unique assertion ("registry not yet ported" not in stdout) checks a message that exists nowhere in the tree.
evidence: a scratch-dir probe with the planted crate-root file → `gate 8: FAIL — expected registry at crates/mantis-encoding/src/registry.toml … absent.` rc=1. `git grep -n "not yet ported"` → the test itself only. `pytest tests/tools/test_registry_gate.py` → 2 passed.
Δlines: −8 (AST span 21–26 + 2 blanks). The module docstring's "the armed-stub path exits 0" sentence is also stale. Collected −1; no tier_declaration row. The alternative, re-aiming it at src/registry.toml, is net 0 and restores the arm (see DEFECTS).
witness: test_hard_fail_when_registry_absent (same arm).
depends: — (LAW-07 self-test of gate 8 → lane C)

### S-A-TESTS-3-11 | DUP | C
subject: tests/tools/test_preflight_child_convergence.py::preflight_child (+ its prologue), tests/tools/test_preflight_armed_smoke.py::test_armed_smoke_config_completes_a_bounded_burst_through_the_real_preflight, tests/tools/test_preflight_pfc_cards.py::test_a_foreign_run_ids_litter_does_not_trip_the_refusal
claim: all three spawn the SAME real green preflight on the same inputs (config configs/smoke_preflight_armed.yaml, burst 16, --timeout-sec preflight_budget_sec, --receipt-wait-sec 120). They differ only in out-dir contents (foreign litter) and in env (only armed_smoke redirects XDG_STATE_HOME). One module-scoped boot, with the foreign litter planted in its out-dir first, can feed every assertion.
evidence: a read of the three `subprocess.run` argv lists (identical flags and config); tier_declaration rows 70, 71, 81 (all integration).
deliberate?: child_convergence says "Spawn ONE real preflight, shared by every assertion below", which already accepts a shared boot, and it cites armed_smoke as the burst-length authority ("one authority").
Δlines: −42 (child_convergence lines 1–47, the prologue through the fixture, less the kept `import tokenize`, `_COMPOSER_EVENTS` and 2 docstring lines). Collected count unchanged.
witness: the 4 child_convergence tests + armed_smoke + the foreign-litter row, on the merged fixture.
depends: — (armed_smoke is PZ-4's named live consumer of an exempt config; tier_declaration.txt rows move) → lane C

### S-A-TESTS-3-12 | DUP | C
subject: tests/tools/test_artifact_gate.py::_run_gate, ::_run_gate_rename
claim: both re-implement the file's own `_repo_with` (repo + one commit per dict) and `_gate` (run the script). `_run_gate(tree, added)` is `_gate(tree, _repo_with(tree, [{"base.txt": b"base\n"}, added]), "--base", "HEAD~1")`. `_run_gate_rename` needs only the extra `git mv` + commit on top of `_repo_with`.
evidence: AST spans _run_gate 24, _run_gate_rename 22, _repo_with 17, _gate 3. jscpd: lines 106–113 ↔ 37–44, 120–127 ↔ 53–61, 158–165 ↔ 37–44, 166–171 ↔ 48–53 (all within this file).
deliberate?: no; the B-6 section added `_repo_with`/`_gate` later without folding the older builders in.
Δlines: −35 (46 lines → a 3-line `_run_gate` and an 8-line `_run_gate_rename`).
witness: all 13 gate-6 rows in the file.
depends: — (LAW-07 self-test of gate 6 → lane C)

### S-A-TESTS-3-13 | TEST | C
subject: tests/tools/test_run_dashboard.py::test_the_shakedown_record_renders_under_the_size_cap
claim: the test runs only when `MANTIS_DASH_FIXTURE_EVENTS` names "the 173 MB run6 record". No procedure, Makefile target or gate sets that variable, so it is a declared skip on every host. It is also keyed to one run's record (R10).
evidence: `git grep -n MANTIS_DASH_FIXTURE_EVENTS` → this test + docs/design/observatory_design.md (design prose); `grep -n -i "MANTIS_DASH\|shakedown" docs/governance/STATE.md` → no procedure sets it; tier_declaration.txt row 85 `skip`.
Δlines: −14 (AST span 306–317 + blanks). The file stays at 356 lines, over 300, so its R8 header stays. Collected −1; row 85 goes with it.
witness: NONE (it never executes).
depends: — (tier_declaration.txt is under tools/ci_gates/**, and a docs/design file cites the variable → lane C)

### S-A-TESTS-3-14 | TEST | C
subject: tests that ride on S-A-TOOLS-2 findings: tests/tools/test_audit_bootstrap_corpus.py (whole, 654 lines, 38 collected); tests/tools/test_probe1.py (3 of its 4 tests); the ladder fixtures in test_run_dashboard / test_dashboard_strength / test_dashboard_reader; test_ladder_client.py's `cancel` row; test_game_viewer.py's hexlogic rows
claim: each test's subject is proposed for deletion by the owning tools slice, so the test goes, or is re-pointed, in the same commit.
evidence: S-A-TOOLS-2-01 (audit_bootstrap_corpus, ONE-SHOT, its Δ includes this test file); S-A-TOOLS-2-02 (probe1 decompose/gap/proofs/swa, "−33 tests (3 of test_probe1.py's 4 tests)"); S-A-TOOLS-2-03 (dashboard ladder surface, witnesses name these three test files); S-A-TOOLS-2-15 (LadderClient.cancel is test-only); S-A-TOOLS-2-08 (hexlogic, "−26 tests"). `pytest --collect-only -m ''` → test_audit_bootstrap_corpus 38, test_probe1 4.
Δlines: −654 for test_audit_bootstrap_corpus.py (`wc -l`), plus the probe1, hexlogic and client rows. ALL of this is already counted in the owning L2 Δs, so do not add it. Collected ≥ −41; floor 4 862 not reached. A `tests/diagnostics/test_tool_absences.py` row (T6) also goes with S-A-TOOLS-2-01.
witness: the owning findings' witnesses.
depends: S-A-TOOLS-2-01, S-A-TOOLS-2-02, S-A-TOOLS-2-03, S-A-TOOLS-2-08, S-A-TOOLS-2-15

### S-A-TESTS-3-15 | TEST | C
subject: tests/tools/test_strix_ruler_r6_cell.py::test_the_follower_unit_composes_the_cell_and_names_its_sidecar, ::test_the_sidecar_records_the_radius_and_the_dashboard_labels_it
claim: these two pin the follower UNIT `ruler_r6` (256/256, sidecar strix256_r6) built for one measurement. CARD-E1-RULER-R6 ordered that measurement, and STATE records it as RUN (R365: "the E1 ruler-r6 cell: 0.080 [0.049, 0.115]"). If the unit is retired, these two tests go. The file's first three tests pin src/mantis/bots/strix.py's `:r<N>` variant mechanism and stay while it does.
evidence: CARDS.md CARD-E1-RULER-R6 ("the unit `ruler_r6` of `tools/strix_follower.py --once`"); STATE.md item text quoted above; `git log -1 -- tests/tools/test_strix_ruler_r6_cell.py` → d4804e5 "the strix ruler-r6 cell (R365 E1)".
Δlines: −27 (AST spans with blanks 11 + 16). Collected −2.
witness: NONE beyond the tests themselves.
depends: a tools-slice decision to retire `tools/strix_follower.py::UNITS["ruler_r6"]`/`RADIUS_UNITS`. S-A-TOOLS-2 raised none, and the card is open → lane C.

## DEFECTS
- tests/tools/test_registry_gate.py: no test exercises gate 8 with the registry PRESENT. The "present" row plants the crate-root path the gate never reads (see 10), so the handshake arm's only witness is the script's own inline self-test.
- tests/tools/test_preflight_child_convergence.py::preflight_child and test_preflight_pfc_cards.py::test_a_foreign_run_ids_litter_does_not_trip_the_refusal run a green preflight with no XDG_STATE_HOME redirect. `_run_preflight` calls `clear_stamp` then `write_stamp` in `stamp_dir()`, so the smoke config's R348(c) stamp is rewritten in the HOST's real store. test_preflight_armed_smoke.py redirects for exactly this reason.
- tests/tools/test_preflight_mint_process.py::test_a_child_rc_46_…: the failure message still says the codes are "46 (draw-rate) and 47 (disk guard)" while asserting (46, 47, 48).
- tests/tools/test_silent_encoding_gate.py::_load_corpus: `read_text()` without `encoding=`. Function scope, so it is gate 16's registered backlog, but it breaks the whole-tree rule.

## PARKED
- The integration tier boots the smoke config three times for one set of inputs (see 11). Each boot is a real torch boot plus a 16-step burst with a 1 800 s budget; merging saves two per integration run.

## HANDOFF
- L1: tools/ci_gates/rule7_gate.py::_justified and encoding_io_gate.py::_justified are two definitions of the escape-hatch check (not compared in this slice).
- L1: if .github/workflows/ci.yml is retired (remote CI suspended), the ci.yml-parity rows in tests/tools/test_local_gate_runner.py (test_the_parse_finds_a_plausible_number_of_gates, test_every_CI_gate_has_a_local_invocation, test_the_runner_invents_no_gate_CI_does_not_have) lose their subject.
- T4/T7 (or S-L-DUP): the two out-of-slice `_code_text` copies named in 06.

## Not covered
- Line-level assertion overlap inside the 97-test test_preflight_mint_process.py beyond 07/08. One near-duplicate is not claimed: test_preflight_verdict_is_reached.py::test_the_not_run_reason_no_longer_claims_no_boot_for_a_spawned_child vs process::test_the_not_run_reason_is_DERIVED_from_the_reports_own_child_block; they differ in the child's rc (None vs 33).
- analyzer (8 files), ladder (7 files), game_viewer, vendor_* and dashboard_{health,svg,stats,envelope} were checked only by the clone scan and fixture/loader scans, not read for assertion overlap.
- Runtime behaviour of the 5 torch-dependent modules (not collectable here) and of every integration row. Torch-free status is also order-dependent in this venv (test_select_balanced_book collects in the full-slice run but errors alone), so it was not relied on.
- Run-named symbols (`RUN5 = configs/run6.yaml` and `_mint_run5_*` in the process file; the analyzer tests' synthetic "run9" ids; configs named per test). CARD-MECHANISM-SWEEP carries them "on contact". Renames have no Δ, so they are not raised.
- Placement: test_bridge_cache_keys.py and test_law13_no_target_cpu_in_committed_config.py test crates/pyproject/build files, not tools. Moving them has no Δ.
