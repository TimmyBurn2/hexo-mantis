# 01_DEFECTS — bugs tripped over by the census (roll-up, never fixed here)

Source: the `## DEFECTS` sections of the 24 scout files (88 raw lines, `awk '/^## DEFECTS/…'`) plus defects the
reviewers confirmed, refuted or added. Duplicates are folded; the owning file is named for the evidence. "(rev:…)" is
the reviewer's verdict where one was given. Out of scope for slimming: each is a separate fix, never ridden on a
slimming leg unless that leg already touches the file.

## Behaviour, witnesses and gates

1. `tools/ci_gates/rule7_gate.py::main` compares an exemption's recorded sha against `git hash-object` (40-hex SHA-1)
   while tests/tools/test_rule7_gate.py demands a 64-hex sha256, so no gate-17 exemption can ever match (S-A-TOOLS-1; rev: confirmed).
2. tests/selfplay/test_lifecycle_events.py: the two `first_inference_*` "producer tests" emit their own event (with
   "dense"; production emits "graph"), so neither event has a real LAW-07 producer test (S-A-TESTS-5-08; rev: confirmed).
3. RunnerStatsSnapshot's LAW-18 fire-rate counters `pcr_full_moves`, `pcr_quick_moves`, `gumbel_round_leaves`,
   `gumbel_rounds` never reach the bridge or Python although both levers are armed in the run10 config (S-A-RUST-2; rev: confirmed).
4. `monitor_gates.buffer_save_errors_total` is published every gate tick from a counter no production path increments
   (S-A-CORE-1-07; LAW-07 class).
5. src/mantis/monitor/rules.py names the live `policy_entropy_selfplay` "legacy" and a producer-less key "canonical";
   the rule still works through `.get` fall-through (S-A-CORE-3-13; rev: real, no behaviour change, lane C).
6. `iteration_complete.corpus_selfplay_frac` is a constant 1.0 published as a measurement; `training_step` carries ten
   always-None fields; docs/contracts/event_manifest.md claims producers that do not exist (S-A-CORE-1-02/-05).
7. src/mantis/train/pretrain/cli.py: `--corpus-npz` (and `--no-compile`) are parsed and never read, and are not in the
   refused dense-arm set, so they are silently ignored (S-A-CORE-1; rev: confirmed).
8. tests/train/test_bc_pretrain_cli_stopflags.py: the partial-stopping-rule test never calls `cli.pretrain`, so R328(d)'s
   all-or-none refusal has no witness (S-A-TESTS-1-14; rev: confirmed, repair not delete).
9. tests/train/test_ema_lever_is_reachable.py ends on `update_parameters` with no assertion and never drives the
   disarmed arm (S-A-TESTS-1; rev: confirmed).
10. tests/model conformance self-tests: the grave-guard self-test raises from its own inline code and the arch-kind-table
    self-test never calls `_arch_keyed_dict_literals`, so neither proves its census bites (S-A-TESTS-8; rev: confirmed).
11. No test round-trips `GnnArchV2SoftPolicy` through the checkpoint arch serializer; test_arch_v2_dispatch's "every
    arch" parametrises V1/V2 only (S-A-TESTS-8; rev: confirmed).
12. tests/tools/test_registry_gate.py: no row exercises gate 8 with the registry PRESENT (the "present" row plants a
    path the gate never reads) (S-A-TESTS-3; rev: confirmed — a fix needs a real fixture).
13. Two preflight integration tests run a green preflight without redirecting XDG_STATE_HOME, writing the smoke
    config's stamp into the host's real state store (S-A-TESTS-3; rev: confirmed).
14. The ban censuses (tests/model/test_arch_ban.py regexes, the net-constructor set in test_one_amp_dtype_authority) do
    not name GnnNetV2/GnnNetV2SoftPolicy (S-L-SEAM; rev: narrower — isinstance still catches them via GnnNet).
15. `TrainerLike` (runtime_checkable) declares `train_step_from_tensors`, which the production Trainer lacks; docstrings
    and one comment in coordinator/step.py still describe the deleted tail (S-A-CORE-1, S-A-TESTS-2/-7 handoffs).
16. src/mantis/selfplay/inference_server.py::submit_and_wait reshapes to `self._shape`, which is None on the only live
    (graph) arm (S-A-CORE-2).
17. src/mantis/data/corpus_metrics.py creates `reports/corpus_analysis` in the CWD at import time (S-A-CORE-2).
18. `mcts/selection.rs::MCTSTree::select_leaves_forced` stores the forced child unchecked though its doc claims the
    `set_forced_root_child` validation (S-A-RUST-1; rev: real, but the bridge path validates first — reach overstated).
19. `mcts/dirichlet.rs::sample_dirichlet` calls `.expect` on a production path (CLAUDE.md Rust style) (S-A-RUST-1; rev:
    confirmed, low reach — the schema requires alpha > 0).
20. crates/mantis-core/tests/miri_cache.rs: `drop(g1)` drops a reference (no-op), so the borrow "proof" step proves
    nothing (S-A-RUST-3).
21. `_engine.pyi` (both copies) declares 4 `*_PLANE` constants the extension does not export and omits 4 runtime members
    Python reads; test_engine_stub_twins_agree compares the stubs only with each other (S-A-RUST-3, S-A-TESTS-5; rev:
    confirmed against runtime `dir()`). Both the root wheel and the maturin wheel ship `mantis/_engine.pyi` (S-A-RUST-3 rev NEW-1).
22. tools/ci_gates/tier_census.py leaks one temp file per `--self-test` (S-A-TOOLS-1; rev: confirmed by a run).
23. tools/ci_gates/comment_lint.py::_RULING misses bare `R\d{2,3}` cites and never reads docstrings, so
    `ruling_cite_comment_lines` under-measures its class (S-L-STYLE; rev: confirmed).
24. tools/ci_gates/test_count_floor.txt = 4862 against 5112 collected at the last box exit: up to 250 tests can vanish
    with gate 3c green (S-A-TOOLS-1; not re-measurable here without torch).
25. tools/hardcode_scan.py::_registry_targets asks RegistrySpec for `n_chain_planes`, which it lacks; `getattr(…, None)`
    hides the dead name (S-A-TOOLS-2).
26. Makefile `dashboard`/`viewer`/`analyzer` recipes hard-code `uv run` instead of `$(UV)` (S-A-TOOLS-1; rev: confirmed).
27. src/mantis/monitor/config.py::MonitorConfig repeats every schema default as a literal (R1) (S-A-CORE-3).
28. A flat `glob("*.yaml")` config census appears at three test sites, blind to configs/ subdirectories that the loader
    and gates 7/12 accept (S-A-TESTS-4).

## Test hygiene

29. tests/train/test_eval_heartbeat.py writes fixed temp-dir paths instead of `tmp_path` and gates on a sleep (S-A-TESTS-1).
30. tests/train/test_parent_death_event.py mutates module state and an env var without monkeypatch (S-A-TESTS-2);
    tests/train/test_survivability.py leaves a `mkdtemp` directory per run (S-A-TESTS-2).
31. cwd-relative paths: tests/selfplay/test_qsigma_rescale_reaches_target.py, test_pool_encoding_bridge.py and
    tests/diagnostics/test_ring_audit.py red when pytest is launched outside the repo root (S-A-TESTS-5/-6).
32. tests/arena/test_books.py's reproducibility test never compares the minter's output with the packaged book or its
    manifest sha (S-A-TESTS-5).
33. tests/selfplay/_fused_graph_harness.py falls back to literal feature dims (`or 11` / `or 5`), the F-41 class
    (S-A-TESTS-5); test_fg6_04 omits three required kwargs, so its TypeError is not the one it claims (S-A-TESTS-5).
34. Eight eval tests pass `encoding="v6_live2_ls"`, a registry row R346(f) deleted; stubs in two actor-sync tests return
    representation "grid" (S-A-TESTS-6, S-A-TESTS-1).
35. The `[sealbot_wr_abort]` case in tests/train/test_abort_exit_signal.py names a rule R362(c) deleted and is the only
    witness of its branch (S-A-TESTS-1).
36. Test names/docstrings that lie about their assertion: "five keys" tests assert seven; `_SEVEN_SCHEMA_LEAVES` holds six;
    a "ratchets in BOTH directions" test asserts no count; a docstring calls an existing module absent (S-A-TESTS-1/-4).
37. tests/monitor/test_arm_exec_trampoline.py puts a `skipif` mark on a non-test helper (inert) (S-A-TESTS-7).
38. rustc warnings at HEAD: non-snake-case test names in mantis-search and mantis-selfplay tests; an unused bridge field
    `PyInferenceBatcher.feature_len`; `PyRegistrySpec::from_static` test-only in the lib build (S-A-RUST-1/-2/-3).
39. pyo3 deprecation: `#[pyclass] + Clone` on InferenceBatcher and SelfPlayRunnerConfig relies on the automatic
    `FromPyObject` that pyo3 is making opt-in (S-A-RUST-3).

## Stale text that misleads (docs, headers, comments)

40. 22 test files justify a local copy with "R5 bars cross-test imports"; R5 bars `sys.path` writes and a `tests`
    package only, and tests/_drivable.py is an R5-clean shared helper (S-L-DUP; rev: confirmed by four reviewers).
41. "BYTE-FROZEN" and "frozen suites" claims in test headers/docstrings with no freeze at HEAD (the sibling was edited
    since) (S-A-TESTS-7, S-A-TESTS-6, S-L-DUP).
42. docs/governance/STATE.md carries box-local specifics (run paths, a tunnel invocation, a provider CLI with an
    instance id, pids) that gate 17's tracked floor does not catch (S-A-DOCS-1; rev: 21 lines by a broader class).
43. docs/design/repo_design.md §3 says GnnArchV2 subclasses GnnArch — it has no base (S-L-SEAM, S-A-DOCS-3; rev: confirmed).
44. Two protected Rust tests (crates/mantis-search/tests/r153_*.rs) cite the R153 prereg at its pre-R355(f) path
    (S-A-DOCS-3; rev: nothing reds — gate 10 scans .md only).
45. Stale line pins: rust-toolchain.toml's `bench_floors.toml:11`; the bridge pyproject cache-key comment
    (`golden_tests.rs:267`, `mod.rs:271`); gate_01's `_engine.pyi:650` (file is shorter); R38's pin into
    test_pool_hparams_arms.py now lands on another test (S-A-DOCS-4, S-A-RUST-3, S-A-TESTS-5 rev NEW-1).
46. "28 bench floors" in CLAUDE.md, rust-toolchain.toml, mise.toml, the Makefile and PZ-6 — tools/bench_floors.toml has
    23 `[floor.*]` tables; Makefile/run_all.sh say 8/7 benches, HEAD has 6 (S-A-TOOLS-1, S-A-DOCS-4).
47. RULINGS.md coverage note: "322 numbers" (the tree gives 321 for R23–R345) and "Four entries" that names five
    (S-A-DOCS-1; rev: confirmed).
48. CARDS.md: `MAX_CHILDREN_PER_NODE = 192` (tree: 1024); six of seven "in-source-only" markers gone and eleven real
    ones missing; falsified.md F-04 points at an absent function/pin (S-A-DOCS-1; rev: confirmed).
49. runner/finalize.rs's header claims an in-src ply-cap unit test that does not exist (S-A-RUST-2).
50. Run-named symbols/defaults (R10): `_RUN5 = configs/run6.yaml` in a PZ-1 eval test and `RUN5 = … run6.yaml` in
    tests/config; `--run-id` defaulting to "run8" in tools/probe1; `configs/run7.yaml` in tools/select_balanced_book.py
    (S-A-TESTS-6, S-A-DOCS-4 rev NEW-1, S-A-TOOLS-2) — CARD-MECHANISM-SWEEP covers the class on contact.
51. CARD-MECHANISM-SWEEP's inventory misses `diagnostics/worker_sweep.py::_sha256` and five `_Trainer` copies in
    tests/train (S-L-DUP).

## Refuted

- "gate 16 false-positives on `os.open`" (S-L-STYLE): REFUTED — a deliberate, pinned limitation
  (tests/tools/test_encoding_io_gate.py::test_known_limitation_any_dot_open_is_flagged_regardless_of_receiver).
