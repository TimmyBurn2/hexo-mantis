# S-A-RUST-1 — R1 crates/mantis-search

scope: `git ls-files crates/mantis-search` (35 files, 11 382 lines) at HEAD e7c1d08 (= 69e1532 + the 00_MAP commit).
method: full read of src/ (lib, legal_set, temperature, mcts/*, tactics/{mod,ordering,eval,tt}, search.rs outline), every
tests/*.rs header + helper list, benches/mcts_bench.rs; per-`pub` census across other crates, the bridge and Python
(`git grep -w`, `git grep -h -o "mantis_search::…"`); `cargo check -p mantis-search --all-targets --locked` (rc 0, 2 warnings:
non-snake-case test names in mcts/tests.rs); a VISIBILITY PROBE (a scratchpad copy of the tracked crates with the lib's
`pub mod`s made private, the re-exports narrowed to the cross-crate set, every `#[allow(dead_code)]` stripped, then
`cargo check -p mantis-search --lib`, which listed the production-dead items below). No cargo test was run.

Checklist method used for every Rust DEAD row (DEAD channels for a Rust symbol): (1) whole tree `git grep -n -w <sym> -- .`
covers docs, Makefile, *.sh, *.toml, *.py and string forms; (2) other crates `git grep -l -w <sym> -- crates ':!crates/mantis-search'`;
(3) pyo3: bridge registrations are in crates/mantis-bridge (covered by 2), Python by `git grep -w <sym> -- src tools tests`;
(4) include_str!/[[bench]]/[[test]]: mantis-search/Cargo.toml has one `[[bench]] mcts_bench`, no [[test]]/[[bin]], no
build.rs; the crate's include_str! is golden_tests.rs::GOLDEN_BITS only; (5) cfg(test)/tests/benches:
`git grep -w <sym> -- crates/mantis-search/tests crates/mantis-search/benches` + in-src `#[cfg(test)]` reads; (6) generics/trait
impls: noted per row. Config keys, gate tool paths, `python -m`, subprocess, STATE procedures and conftests cannot name a
Rust item except as a string, which (1) would find.

## Summary
- 18 findings. By class: DEAD 8 (A 1, B 4, C 3) · ONE-SHOT 1 (C) · TEST 3 (C) · DUP 3 (B 2, C 1) · SIMPLIFY 1 (B) · DOC 2 (A 1, C 1).
- By lane: A 2 · B 7 · C 9. Most of the slice is PZ: mcts/{mod,policy,seq_halving}.rs, mcts/*_tests.rs, all of
  crates/mantis-search/tests/**, SearchKind (seam), and completed-Q/golden (frozen). Not proposed: completed_q.rs,
  golden_tests.rs, parity_tests.rs, mctx_parity.rs, temperature_parity_golden.rs, search_kind_conformance.rs.
- Top 3 by Δlines: S-A-RUST-1-09 (−243), S-A-RUST-1-03 (−207), S-A-RUST-1-10 (−53).
- Rust tests are outside the gate 3c floor (tools/ci_gates/test_count_gate.sh counts `pytest --collect-only` only), so no
  row below moves test_count_floor.txt.

## Findings

### S-A-RUST-1-01 | DEAD | A
subject: crates/mantis-search/src/legal_set.rs::is_covered (+ its `pub use` in lib.rs)
claim: The coverage predicate has no caller anywhere. Its doc still names callers that are gone ("the O1/solver injection gates", "PRIOR producers (`aggregate_policy_ls`)").
evidence: `git grep -n "aggregate_policy_ls\|is_covered" -- .` → legal_set.rs (def + module doc), lib.rs (re-export), mantis-selfplay/src/records.rs (a prose mention of aggregate_policy_ls only), docs/design/archive/measurements/*R153* (frozen prose). Probe → `warning: function is_covered is never used`.
callers:
  whole tree: `git grep -n -w is_covered -- .` → def, re-export, legal_set module doc, frozen archive prose only
  other crates: `git grep -l -w is_covered -- crates ':!crates/mantis-search'` → empty
  pyo3/Python: not registered in mantis-bridge (same grep); `git grep -w is_covered -- src tools tests` → empty
  include_str!/bench/test tables: bench imports {LegalSetPolicy, MCTSTree} only; no [[test]]
  tests/benches: `git grep -l -w is_covered -- crates/mantis-search/tests crates/mantis-search/benches` → 0
  generics/traits: free fn, none. Strings: the whole-tree grep would find `"is_covered"` and found none
Δlines: −16 (`awk '/^\/\/\/ Coverage predicate/{f=1} f' legal_set.rs | wc -l`). The lib.rs re-export line and the module-doc line 3 are edited, not removed.
witness: `cargo check --workspace --all-targets --locked` (gate 2b compiles every target)
depends: —

### S-A-RUST-1-02 | DEAD | C
subject: crates/mantis-search/src/mcts/mod.rs::MCTSTree::{selection_overlap_count, max_depth_observed}
claim: Two `pub` counter fields are written on every descent and reset in new_game, but nothing in the tree reads them.
evidence: `git grep -n -w "selection_overlap_count\|max_depth_observed" -- .` → 10 lines, all in mcts/mod.rs (decl, init, reset) and mcts/selection.rs (writes). No read.
callers:
  whole tree: the grep above covers docs, Python and strings → writes only
  other crates/bridge: none. The bridge's last_search_stats reads depth_accum/sim_count, not these
  Python: `git grep -w` over src tools tests → empty
  tests/benches: none
Δlines: −11 (the 10 grep lines, plus the `}` that closes the `if depth > …` block in select_one_leaf)
witness: `cargo check --workspace --all-targets --locked`
depends: —  (lane C because mcts/mod.rs is on the PZ glob. selection.rs is the PUCT hot path; this only removes writes.)

### S-A-RUST-1-03 | DEAD | B
subject: crates/mantis-search/src/tactics/ordering.rs::{PolicyPrior, OrderingState.policy, OrderingState::with_policy} + tests ordering.rs::net_policy_reorders_quiet_tier_but_never_above_tt_or_killers, search.rs::{run_scored_pol, verdict_invariant_to_net_policy_ordering}
claim: The net-policy ordering hook has no production path. `prove` always builds `OrderingState::new()` (policy None), with_policy carries `#[cfg_attr(not(test), allow(dead_code))]`, and the trait's only impls live in the two tests. tactics/mod.rs calls "net-policy ordering" deferred.
evidence: `git grep -n -w "PolicyPrior\|with_policy" -- crates/mantis-search` → ordering.rs (def, field, ctor, test impl) and search.rs (test helper and test impl) only. Probe → `associated function with_policy is never used`.
callers:
  whole tree: `git grep -n "PolicyPrior\|with_policy" -- docs crates/mantis-bridge crates/mantis-selfplay src tools tests` → no Rust hit (only an unrelated Python `_with_policy_loss` test helper)
  bridge/pyo3: tactics.rs builds TacticalSolver from TacticalConfig {cand_cap, window_half, neighbor_dist}, with no prior
  generics: the trait object `Box<dyn PolicyPrior>` is built only in cfg(test)
Δlines: −207 (ordering.rs: trait 9 + field 3 + `policy: None` 1 + with_policy 9 + prior block 5 + test 48; search.rs 2043–2174: 132; each span measured by `sed -n A,Bp | wc -l`, blank separators not counted)
witness: `cargo test -p mantis-search --lib tactics` (the remaining ordering/soundness tests)
depends: —  (lane B because this removes a deferred feature seam that tactics/mod.rs's doc names)

### S-A-RUST-1-04 | DEAD | B
subject: crates/mantis-search/src/tactics/tt.rs::{Bound (Upper never constructed), Slot.bound (never read)} + ProofTt::store_bound's `bound` parameter
claim: The α-β bound flag is stored but never read, and `#[allow(dead_code)]` hides that. Production stores only `Bound::Lower` (search.rs, 2 sites) and `Exact`. `Upper` appears only in a tt.rs test. The field doc "READ by the PVS bound cutoffs" is false.
evidence: probe (allow stripped) → `variant Upper is never constructed`, `field bound is never read`. `grep -c "Bound::Lower," tactics/search.rs` → 2.
callers:
  whole tree: `git grep -n "\.bound\b\|Bound::" -- crates/` → store sites only, no read. Every mantis-bridge `Bound` hit is pyo3's `Bound<'_, T>`, a different type
  Python/pyo3: tt is crate-internal (ProofTt is not re-exported)
Δlines: about −24 (tt.rs enum block 11 + field doc/allow/decl 4 + EMPTY 1 + store_loss_proof 1 + store_bound param/field 2 + search.rs args 2 + 3 test args)
witness: `cargo test -p mantis-search --lib tactics::tt`
depends: —  (lane B because store_bound's signature changes)

### S-A-RUST-1-05 | DEAD | C
subject: crates/mantis-search/src/mcts/backup.rs::MCTSTree::expand_and_backup_ls (the UNFRAMED batch expand)
claim: No production caller. The bridge's expand_and_backup_ls_graph and selfplay's search_drive both call `expand_and_backup_ls_at`. The TT-hit path uses `expand_and_backup_single_ls`, which stays.
evidence: probe → `method expand_and_backup_ls is never used`. Callers: crates/mantis-search/tests/r153_leg2_ls_target_mass.rs (the SECONDARY `Expand::Ls` arm) and crates/mantis-selfplay/tests/target_sign_integrity.rs; plus a Python error STRING, src/mantis/eval/worker.py ("the Rust expand_and_backup_ls) exists but is not wired").
callers:
  whole tree: `git grep -n -w expand_and_backup_ls -- .` → def, the 2 PZ tests, the worker.py message
  bridge: not exposed. Python: string only
Δlines: −26 (`sed -n 527,552p backup.rs | wc -l`)
witness: `cargo test -p mantis-search --test r153_leg2_ls_target_mass`; `cargo test -p mantis-selfplay --test target_sign_integrity`
depends: S-A-RUST-1-09  (lane C: both callers are PZ tests, and worker.py is a PZ file)

### S-A-RUST-1-06 | DEAD | B
subject: crates/mantis-search/src/mcts/backup.rs::{OMITTED_PRIOR_MASS_MICROS, OMITTED_PRIOR_EXPANSIONS, TOTAL_EXPANSIONS, omitted_prior_stats, take_omitted_prior_stats, record_omitted_prior_global} (+ re-exports in mcts/mod.rs, lib.rs)
claim: The process-wide omitted-prior totals feed exactly one consumer, the bridge pyfunctions mcts_omitted_prior_stats / take_mcts_omitted_prior_stats. Those have ZERO Python references (CALLERS §9). The per-tree counters (OmittedPriorStats) are the live measurement.
evidence: `git grep -n -i "omitted_prior" -- crates/mantis-selfplay crates/mantis-bridge src tools tests` → bridge/src/utils.rs + both .pyi twins only. repo_design.md amendment "5. The omitted-prior counters are PER-SEARCH" keeps the totals "because the bridge publishes them as the run-wide aggregate", a publisher with no reader.
callers:
  whole tree: the grep above + docs → repo_design (the contract sentence), 00_MAP, archive audits (prose)
  pyo3: bridge utils.rs registers both; Python `git grep -w mcts_omitted_prior_stats -- src tools tests '*.py'` → pyi only
  tests: search_kind_conformance.rs reads the METHOD `tree.omitted_prior_stats()`, not the free fn
Δlines: −38 in slice (backup.rs 39–64: 26 + 72–82: 11 + the call in record_omitted_prior: 1). Bridge and pyi lines belong to R3.
witness: tests/bridge/test_engine_stub_twins_agree.py (pyi twins); `cargo check --workspace`
depends: HANDOFF R3 (bridge pyfunctions + pyi)  (lane B because repo_design needs an amendment, per R9)

### S-A-RUST-1-07 | DEAD | B
subject: test-only `pub` surface: mcts/backup.rs::TopKPick.truncated (+ `#[allow(dead_code)]`), mcts/mod.rs::MCTSTree::{next_free_slot, take_omitted_prior, omitted_prior_stats, root_children_cap}, backup.rs::OmittedPriorStats::{read, take}, tactics/mod.rs::Outcome::negate, ProofResult.budget_exhausted, gumbel_mctx.rs::MctxRootState::simulation_index (pub, internal only)
claim: The probe with narrowed exports flags each of these as never used by any production path. Tests reach them: tests.rs, ls_prior_tests, pool_overflow.rs, search_kind_conformance.rs (PZ), and search.rs cfg(test) solve_3valued.
evidence: probe warnings `methods next_free_slot, take_omitted_prior, omitted_prior_stats, and root_children_cap are never used`; `field truncated is never read`; `methods read and take are never used`; `method negate is never used`; `fields nodes and budget_exhausted are never read` (`nodes` IS read by the bridge, a probe artefact).
callers: per-symbol `git grep -n -E "\.<m>\b|::<m>\b" -- crates` → test files only. None is in the bridge, so none is in Python.
Δlines: −6 production lines for `truncated` (doc/allow/field 4 + 2 initialisers). Its 6 test reads must be re-expressed through `children.len()`. The rest is visibility narrowing (0 lines), and the conformance-read accessors stay.
witness: `cargo test -p mantis-search --lib mcts::`
depends: —

### S-A-RUST-1-08 | DEAD | C
subject: crates/mantis-search/src/mcts/policy.rs::MCTSTree::get_policy (the dense temperature visit policy)
claim: The only Rust caller is the bridge pymethod MCTSTree.get_policy, whose only Python caller is a test (tests/bridge/test_mcts_inference_roundtrip.py). Production exports through get_policy_ls (selfplay) and get_top_visits / the Gumbel root (deploy head).
evidence: `git grep -n -E "\.get_policy\(|::get_policy\b" -- crates` → bridge/mcts.rs + policy.rs tests; `git grep -l -E "\.get_policy\(" -- src tools '*.py' | grep -v pyi` → the roundtrip test only.
callers:
  whole tree: `git grep -n "get_policy\b" -- docs/contracts docs/governance` → no contract row
  pyo3: registered (bridge mcts.rs::get_policy). Python: test-only
  in-src tests: policy.rs test_get_policy_{proportional_to_visits,argmax_temperature_zero}, test_policy_sums_to_one_after_search
Δlines: −41 (`sed -n 11,51p policy.rs | wc -l`), plus its three in-src tests and the bridge pymethod (R3)
witness: tests/bridge/test_mcts_inference_roundtrip.py (must drop its get_policy leg); `cargo test -p mantis-search --lib`
depends: HANDOFF R3  (lane C: policy.rs is PZ, and the _engine surface moves)

### S-A-RUST-1-09 | ONE-SHOT | C
subject: crates/mantis-search/tests/r153_target_mass.rs (the R153 leg-1 instrument)
claim: A spent measurement. It drives the DENSE expand (`expand_and_backup`) and then the ls export, a pairing no production path runs (selfplay uses ls_at + get_policy_ls; the deploy head uses dense + top-visits/Gumbel). Its sister leg-2 header says a dense-expand zero "is structural rather than a clearance". Each of its assertions is also made on the production path: no-drop ≤ 1e-6, abort 1 (> 361 legal) and abort 3 (determinism) by r153_leg2_ls_target_mass.rs (gnn_axis_v1 via ls_at, gnn_axis_r8 via ls), and sum-to-1 with off-window mass by target_export_stage1.rs::s1a_t1.
evidence: `sed -n 37,52p r153_target_mass.rs` (dense `expand_and_backup`) vs `sed -n 80,104p r153_leg2_ls_target_mass.rs` (ls_at / ls); leg-2 header lines "under the DENSE expand … a zero there is structural".
deliberate?: S-L-DUP K-D10 rates the R153 harnesses DELIBERATE (a pre-registered per-leg harness). This row is a retirement claim, not a merge.
Δlines: −243 (`wc -l`)
witness: NONE by construction (no other test reads this file). The frozen docs/design/archive/measurements/{PREREG,MEASUREMENT}_R153.md cite it, and gate 10 exempts docs/design.
depends: —  (lane C: crates/mantis-search/tests/** is on the PZ glob, and a ruling (R153) names the instrument)

### S-A-RUST-1-10 | TEST | C
subject: crates/mantis-search/tests/temperature_schedule.rs::{temperature_at_move_zero_is_one, temperature_at_threshold_equals_floor, temperature_past_threshold_stays_at_floor, temperature_matches_quarter_cosine_formula}
claim: These four assert values that the frozen CSV golden already pins for the same function at the SAME parameters (threshold 15, temp_min 0.05): cm 0 → 1.0, cm 15 → 0.05, cm 24 → 0.05, and cosine values at cm 1–12. Only the monotonicity test and the cm=14 above-floor test are unique. The header's "DEFERS to the config work package" note is stale.
evidence: `grep "^[0-9]*,15," tests/fixtures/search_golden/temperature/temperature_parity_golden.csv` → 12 rows (ply 0…48). temperature_parity_golden.rs drives every row through compute_move_temperature(ply_to_compound_move(ply), …).
Δlines: −53 (spans 22–30, 32–41, 43–53, 72–90 plus 4 separators)
witness: `cargo test -p mantis-search --test temperature_parity_golden`
depends: —

### S-A-RUST-1-11 | TEST | C
subject: Dirichlet tests, triplicated: mcts/dirichlet.rs::tests::{test_dirichlet_sums_to_one, test_dirichlet_non_negative}; mcts/policy.rs::tests::test_dirichlet_mixes_priors_correctly; tests/dirichlet_parity.rs::intermediate_ply_gate_matches_self_play_spec
claim: dirichlet_parity.rs::sample_dirichlet_sums_to_one_and_is_nonneg covers the first two (same α 0.3; it adds n ∈ {1,2,5,24,50} and a length check). dirichlet_parity.rs::apply_dirichlet_to_root_blends_linearly covers the policy.rs blend test on a real expanded root, and golden_tests.rs S4 pins both bit-exactly. The intermediate-ply test checks a formula re-typed locally in the test, not search_drive.rs's expression, so it has no production subject. The file header ("duplicated on both sides of the self-play `if gumbel_mcts` branch") is stale: there is one site, search_drive.rs, and no gumbel_mcts bool.
evidence: `grep -n "moves_remaining == 1 && .*ply" -r crates` → search_drive.rs (1 site) + the test's local fn
Δlines: about −49 (dirichlet.rs 41–63: 23; policy.rs test about 26). The ply test's deletion is not counted.
witness: `cargo test -p mantis-search --test dirichlet_parity`; golden_tests::test_golden_s4_puct_dirichlet_unchanged
depends: —  (lane C: policy.rs and tests/** are PZ)

### S-A-RUST-1-12 | TEST | C
subject: crates/mantis-search/src/mcts/kind.rs::tests::{the_config_spelling_round_trips, an_unknown_kind_is_refused_rather_than_defaulted, the_completed_q_target_is_the_kinds_own_answer}
claim: These restate search_kind_conformance.rs::the_config_spelling_round_trips_and_an_unknown_kind_is_refused (whose refusal list is a superset: it adds "gumbel ") and ::the_kind_is_the_only_authority_over_the_target. Only only_the_gumbel_kind_stores_a_sparse_row is unique.
evidence: `sed -n 68,104p kind.rs` vs `sed -n 128,164p tests/search_kind_conformance.rs`
Δlines: −28 (`sed -n 70,97p kind.rs | wc -l`)
witness: search_kind_conformance.rs (PZ suite-v2 section, kept)
depends: —  (lane C: SearchKind is a seam member)

### S-A-RUST-1-13 | DUP | B
subject: crates/mantis-search/src/mcts/backup.rs::MCTSTree::{expand_and_backup_single, expand_and_backup_single_ls_framed}
claim: The same four pre-check arms (terminal → backup tv; already expanded → quiesce + backup; check_win → CF-1 sign; empty legal set → draw) appear in both. The CF-1 sign comment is on one copy only.
evidence: `sed -n 326,361p` (33 non-blank lines) vs `sed -n 462,489p` (28 non-blank lines): arm-for-arm identical except the comments
deliberate?: not a seam, oracle or twin. Both feed finish_expansion, which the doc already calls "representation-agnostic"
Δlines: about −25 net (one `fn backup_if_not_expandable(&mut self, leaf, board, value) -> bool` of about 28 lines replaces 2 copies)
witness: `cargo test -p mantis-search --lib mcts::`; golden S4; crates/mantis-selfplay/tests/graph_child_parity.rs
depends: —  (hot path; behaviour-neutral by construction)

### S-A-RUST-1-14 | DUP | B
subject: crates/mantis-search/src/mcts/selection.rs::MCTSTree::{select_leaves, select_leaves_forced}
claim: The desync-unwind block, the overlap-skip block and the pending-push block are copied between the two, and the 3-line board-rewind loop appears 7 times.
evidence: `grep -c "while let Some(diff) = diffs.pop()" selection.rs` → 7
deliberate?: select_leaves_forced deliberately omits the TT fast path (commented). The shared blocks carry no such difference.
Δlines: about −12 net (7 × 3 lines → 7 calls + a 4-line helper)
witness: mcts/tests.rs desync/unwind tests; tests/bridge/test_select_leaves_forced.py
depends: —

### S-A-RUST-1-15 | DUP | C
subject: action_idx pack/unpack re-typed inline: mcts/policy.rs (9 decode pairs), selection.rs (1), backup.rs (3 encodes); outside the slice: bridge/mcts.rs (3), selfplay/runner/search_drive.rs (2), and test copies (target_export_stage1::coord_of and others)
claim: `(val >> 16) as i32 - 32768 / (val & 0xFFFF) as i32 - 32768` and its inverse are typed out more than 15 times. A `Node::cell()` / `pack_cell()` pair on node.rs would make it one authority.
evidence: `git grep -n -E ">> 16\) as i32 - 32768|\+ 32768\) as u32\) << 16|& 0xFFFF\) as i32 - 32768" -- crates | grep -v tests` → policy.rs 18 lines, backup.rs 3, selection.rs 2, bridge 3, selfplay 2
deliberate?: none stated. get_improved_policy_ls is FROZEN math, but the decode is its input scatter, not the math.
Δlines: about −9 in policy.rs net (9 two-line decodes → 9 one-liners; +5 for the helper in node.rs); small
witness: golden S4, mctx parity, target_export_stage1
depends: —  (lane C: policy.rs is PZ)

### S-A-RUST-1-16 | SIMPLIFY | B
subject: crates/mantis-search/src/lib.rs::{CRATE_NAME, tests::crate_name_pinned, tests::dag_deps_compile} + Cargo.toml [dependencies] mantis-encoding
claim: This is scaffold. The lib's only use of mantis_encoding is the `dag_deps_compile` test assert, so the encoding edge is a dev-only dependency in practice (three tests use `lookup_or_panic`). repo_design §2 states the edge "mantis-search → core, encoding".
evidence: `git grep -n mantis_encoding -- crates/mantis-search` → lib.rs test + 3 tests/*.rs only
Δlines: −15 if the scaffold goes (`sed -n 30,44p lib.rs`). But mantis-selfplay's lib test reads `mantis_search::CRATE_NAME`, and the pattern is workspace-wide (HANDOFF R2/R3). The dependency move is 0 lines.
witness: `cargo check --workspace --all-targets --locked`; make check.wasm (not affected)
depends: HANDOFF R2, R3  (lane B: needs a repo_design §2 amendment)

### S-A-RUST-1-17 | DOC | A
subject: non-PZ in-source doc drift: benches/mcts_bench.rs (`registry.toml:160-190`); tactics/tt.rs module doc (dangling duplicate fragment "//! fields; the algorithmic wins are all present."); tt.rs::Slot.bound doc ("READ by the PVS bound cutoffs"); selection.rs (pick_best_puct's "Single-pass argmax …" doc sits on `ForcedChildOutOfRange`)
claim: The cited line range does not exist (registry.toml is 120 lines), one module-doc line is an orphaned fragment, one doc states a false read, and one doc is attached to the wrong item.
evidence: `wc -l crates/mantis-encoding/src/registry.toml` → 120; `sed -n 19,21p tt.rs`; `sed -n 49,63p selection.rs`
Δlines: −1 (the orphan line); the rest are in-place rewords
witness: gate 14 comment ratchet (may only fall)
depends: S-A-RUST-1-04 (the Slot.bound doc goes with the field)

### S-A-RUST-1-18 | DOC | C
subject: PZ test headers/messages: r153_target_mass.rs and r153_leg2_ls_target_mass.rs cite `docs/design/measurements/{PREREG,MEASUREMENT}_R153.md` (now docs/design/archive/measurements/); `records.rs:468-479` line citations in 3 assert messages (r153 ×2, target_export_stage1); dirichlet_parity.rs header (`if gumbel_mcts`); temperature_schedule.rs's DEFERS note
claim: Paths are stale and line numbers are asserted (R192(e) derive-or-delete).
evidence: `git ls-files | grep R153` → docs/design/archive/measurements/* only; `git grep -n "records.rs:468-479" -- crates` → 3
Δlines: 0 (rewords)
witness: none (messages only)
depends: S-A-RUST-1-09, -10, -11

## DEFECTS
- mcts/selection.rs::MCTSTree::select_leaves_forced: the doc says it "takes the same validation" as set_forced_root_child, but it stores `self.forced_root_child = Some(child)` unchecked. A foreign index therefore descends into a node the root does not own, the exact hazard ForcedChildOutOfRange was added to refuse. It is reached from the bridge (Python deploy_head/ring_audit) and from selfplay's round_batch.
- mcts/dirichlet.rs::sample_dirichlet: `Gamma::new(..).expect("Dirichlet: invalid alpha")` sits on a production path (CLAUDE.md Rust style: a named error, never expect), and alpha > 0 is guarded only by debug_assert.
- mcts/tests.rs: two test names trip rustc non_snake_case (`…_is_an_ERR_not_a_panic`, `a_FULL_batch_…`). These are the only 2 warnings from `cargo check -p mantis-search --all-targets`.

## PARKED
- tactics/tt.rs::ProofTt::new_generation plus the generation compare: zero callers, kept in-source as a banked perf decision (a fresh 8 MiB ProofTt per `prove`). Not raised.
- tests/r153_*.rs and target_export_stage1.rs drive about 3 games × 128 plies × 50–150 sims × 2 encodings per run, the heaviest cargo-test cost in the slice.

## HANDOFF
- R3 (mantis-bridge): the pyfunctions mcts_omitted_prior_stats / take_mcts_omitted_prior_stats have zero Python refs (feeds -06). The MCTSTree pymethods get_policy (Python: test only, feeds -08), run_simulations_cpu_only, last_search_stats and root_raw_value have zero Python refs, and apply_dirichlet_to_root is referenced only as a string.
- R2 (mantis-selfplay): tests/target_sign_integrity.rs is the only non-R153 caller of expand_and_backup_ls (-05). The lib test reads mantis_search::CRATE_NAME (-16).
- S-L-DUP-03 already covers the r8_game_target_mass.rs / gumbel_sparse_row_kl.rs harness duplicate (r8_board, stub_policy, search: 71 lines, `sed -n 26,96p`). It is not repeated here.

## Not covered
- Per-test uniqueness inside mcts/tests.rs (1 520 lines) and the tactics/search.rs test block (about 1 590 lines): outlines were read, bodies were not compared test by test.
- pool_overflow.rs versus search_kind_conformance.rs::a_search_at_the_gumbel_ceiling_still_fits_the_pool; perspective_parity.rs versus the policy.rs tests; tactics/eval.rs tests; tt.rs tests beyond the Bound/generation ones.
- Frozen or PZ-by-design material was read only enough to exclude it: completed_q.rs, golden_tests.rs, parity_tests.rs, mctx_parity.rs, seq_halving.rs, search_kind_conformance.rs.
- No cargo test was run, so no row was probed green. The reviewer probes.
