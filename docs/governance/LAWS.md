# LAWS — the standing laws of mantis

Read this first. Each law was bought by a measured failure; the earned mechanism and the
pre-R346 wording are in docs/governance/archive/laws.md. Changing or dropping a law takes an
amendment commit and operator sign-off.

## The eighteen

- LAW-01 Prime directive. Context first, measurement mandatory: no architectural change
  without reading the design docs and the source, no perf-sensitive change without a bench.
- LAW-02 Re-validation. Never drop a candidate driver or a fix on a falsified row or a banked
  prior without testing that the prior's context transfers. Retested fresh, one driver INVERTED.
- LAW-03 Measurement-unit. Verify a founding measurement's unit before framing on it — turn
  vs ply (a compound turn places TWO stones), and WHICH cell of a multi-stone win completes it.
- LAW-04 Effective-n. A strength CI counts DISTINCT games (trajectory-hash dedupe), not the
  game count; argmax/temp-0 regimes collapse to ~2 distinct games per pairing.
- LAW-05 Falsified-register-first. Read docs/governance/falsified.md before proposing any optimization or experiment, and apply LAW-02 when citing a row from it.
- LAW-06 bf16-graph (R369(b)). bf16 storage and GEMMs; graph message aggregation accumulates in
  fp32, deterministically, in one implementation shared by server and trainer. No key restores the
  old path; this is not a new arch kind. Pinned in code and by parity tests.
  R349(a): fp32 on `train.device: cpu` is the ONE carve-out (the autocast context, never the
  dtype pin), pinned by its own parity test; it applies to no production path.
- LAW-07 Producer-test. No gate or monitor input without a live producer test, and the checker
  carries a mutation self-test proving it bites.
- LAW-08 Live-consumer. Every config key and every registered encoding has a live consumer; a
  dead knob dies with its freeze-tests in one commit. Pinned by
  tests/config/test_every_key_has_consumer.py and, for encodings, gate 11.
- LAW-09 Bench discipline. Pre-registered hotspots + expected-gain bracket + abort threshold;
  one change = one commit = one IQR-gated bench; profile first; a measured floor is a finding.
- LAW-10 DELETED by R347(d) — grid-era, no producer, gating nothing. The number is retired, not
  reused: every other law keeps the number it was cited by.
- LAW-11 Identity-keys. No dense-by-default anywhere. An absent encoding/representation is an
  error, never a default; representation is a closed enum on both sides of the FFI.
- LAW-12 Checkpoint-stamp. Stamps are written once and immutable, never re-stamped from a
  loaded config; one envelope format (docs/contracts/checkpoint_envelope.md) and one loader;
  weights-only strip is the one sanctioned encoding-change path. Artifact filenames carry the
  run id and a content hash, and an artifact that cannot be stamped cannot be written.
- LAW-13 FFI/build. panic = "unwind" so a panic crosses the FFI catchable, never as a process
  abort; no target-cpu in committed build config — a native build is `make build.native`,
  env-only, and its artifacts are host-specific and never distributed.
- LAW-14 Persistence-fatal. Persistence failures are run-fatal; `except Exception: pass` is
  lint-banned; an optional effect goes through best_effort() and requires a counter.
  ANNOTATION (2026-09-17, INVESTIGATION-1 C-11): the lint ban is ruff `BLE` over `src/` and
  `tools/`; `tests/` carries a declared `BLE001` exclusion with its grounds in pyproject.toml (an
  oracle-write corpus), so the ban's scope is the production tree, not "repo-wide".
- LAW-15 Eval-instrument. Deploy-matched eval is the DEFAULT promotion bar and a missing deploy
  decision blocks promotion; strength bars are fixed-depth instruments, never wall-clock.
- LAW-16 Lifecycle. One subsystem, contract-tested: signals save-then-exit, self-play stall watchdog ALWAYS armed, disk guard.
- LAW-17 Structure. Zero sys.path writes; one tests/ collection root with no package named
  `tests` below it; pyo3 only in crates/mantis-bridge, so every other crate compiles without it;
  configs explicit and complete; a >300-line file justifies itself and states no line count.
- LAW-18 In-run observability. A lever under test logs its own fire-rate in-run — a post-hoc probe cannot tell "starved" from "ineffective".
- LAW-19 Controls first (R370(b)). Before a pre-registered criterion gates work, a correct design
  must pass it and the known-bad path must fail it; a criterion no correct design can pass is void.

## The protected set

R346 §1(d): this list is the whole of what ruling-protected code means from now on. Each item
exists because a measured failure bought it. None may be weakened, disarmed, narrowed or
deleted except by a ruling that names it. Each is held by the tests named beside it (R370(f));
tests/test_protected_set_pins.py fails if a named test stops existing.

- net-param hash on the warm-start —
  `tests/train/test_bc_warm_start_entry.py::test_a_checkpoint_that_is_NOT_the_declared_net_is_REFUSED`,
  `tests/train/test_bc_warm_start_entry.py::test_a_row_missing_its_hash_is_REFUSED_not_defaulted`,
  `tests/train/test_f32_launch_pin_wiring.py::test_a_SWAPPED_artifact_at_the_pinned_path_REFUSES`
- served-sims exactness —
  `crates/mantis-selfplay/tests/served_sims_exact.rs::both_kinds_serve_exactly_sixty_four`,
  `crates/mantis-selfplay/tests/served_sims_exact.rs::r8_at_fifty_sims_serves_exactly_fifty_per_search`,
  `tests/arena/test_deploy_head_budget_spent.py::test_every_kind_spends_exactly_its_budget`
- the suite's conformance sections (tests/model/conformance/ and the search-kind suite) —
  `tests/model/conformance/test_conformance_roster_guard.py::test_a_SHRUNKEN_roster_is_refused`,
  `crates/mantis-search/tests/search_kind_conformance.rs::a_gumbel_round_is_exactly_the_halving_phase_wide`
- 1-in-1 collate checks —
  `tests/eval/test_f816_37_instrument.py::test_every_collate_path_asks_for_one_in_one`,
  `tests/eval/test_f816_37_instrument.py::test_period_one_runs_the_semantic_layer_on_every_batch`
- arena legality —
  `tests/arena/test_legality_boundary.py::test_a_candidate_playing_off_the_legal_set_forfeits_and_the_move_is_not_applied`,
  `tests/arena/test_legality_boundary.py::test_an_opening_that_does_not_replay_is_a_fatal_corpus_error`
- finite-gradient guard —
  `tests/train/test_finite_gradient_guard.py::test_a_nonfinite_gradient_from_a_finite_loss_never_reaches_the_optimizer`,
  `tests/train/test_nonfinite_guard.py::test_a_nonfinite_microbatch_loss_is_skipped_and_counted`
- resume bundle round-trip —
  `tests/train/test_resume_ring_roundtrip.py::test_the_ring_comes_back_and_it_is_the_same_ring`,
  `tests/train/test_resume_bundle.py::test_a_bundle_whose_member_changed_underneath_is_refused`,
  `tests/train/test_resume_semantics.py::test_full_save_resume_roundtrip_restores_state`
- gate pair statistics —
  `tests/eval/test_gate_pair_statistics.py::test_the_two_legs_of_an_opening_are_one_unit`,
  `tests/eval/test_gate_pair_statistics.py::test_the_gate_ci_and_eff_n_are_both_over_pairs`
- F-816-37 dump-on-fire —
  `tests/eval/test_f816_37_instrument.py::test_a_planted_corruption_DUMPS_and_REDS`,
  `tests/train/test_f816_37_train_path_dump.py::test_the_dump_never_replaces_the_raise`
- strength_floor —
  `tests/eval/test_strength_floor_refuses_the_round.py::test_a_no_signal_round_REFUSES_before_the_gate_block_ever_runs`,
  `tests/eval/test_strength_floor_gate.py::test_both_bars_are_reported_even_when_both_fail`
- draw-rate abort —
  `tests/selfplay/test_drawrate_pooled_statistic.py::test_a_true_pool_draw_rate_of_0968_fires_the_abort`,
  `tests/train/test_drawrate_gate_branch_flipset.py::test_every_branch_of_the_draw_rate_gate_has_an_input_that_takes_it`

## Rule numbers R1–R11

Comments and docs across the tree cite CLAUDE.md's former hard rules by number. Each now lives
in one place:
R1 config — LAW-08, LAW-11, LAW-17, docs/design/repo_design.md §5, gates 7 and 12, and CLAUDE.md
(minted, never hand-edited); R2 — LAW-13; R3 — LAW-12; R4 — LAW-07, LAW-08; R5 — LAW-17;
R6 — LAW-17; R7 — gate 6 (tools/ci_gates/artifact_gate.py); R8 — LAW-17 and gate 15
(tools/ci_gates/r8_header_gate.py); R9, R10, R11 — CLAUDE.md.
