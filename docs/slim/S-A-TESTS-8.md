# S-A-TESTS-8 — T8 tests/model (incl. the conformance suite)

scope: `tests/model/**` at HEAD e7c1d08 (slice tree identical to 69e1532), 32 files, 7 917 lines
(`git ls-files 'tests/model/**' | xargs wc -l`); 15 files / 4 710 lines are `tests/model/conformance/**` (PZ-1 row 3, PZ-2).
method: full static read of all 32 files; AST spans via `.venv/bin/python -c 'import ast…'`; `git grep -n -w` over the whole
tree per symbol; `uvx ruff check --isolated --select F401,F811,F841,B007 tests/model`; `uvx vulture tests/model
--min-confidence 60`; `npx --yes jscpd@4 --min-lines 5 --min-tokens 40 tests/model` (10 clones, 69 dup lines) and a
cross-dir jscpd run over tests/{model,train,selfplay,eval,util,encoding,diagnostics}; `pytest --collect-only -q -m ''
tests/model` → 11 collected, 26 collection errors, all `No module named 'torch'`; the 11 torch-free tests
(test_forward_batch_call_sites, test_one_amp_dtype_authority, test_arch_ban) run → 11 passed.
Already raised by S-L-SEAM and NOT repeated here: S-L-SEAM-04 (the two conformance `specs_for`), S-L-SEAM-06
(test_flatten_ban_p3 HexTacToeNet tautology), S-L-SEAM (a) A7/A8 (hand-typed net-name sets in
test_one_amp_dtype_authority / test_arch_ban). S-L-SEAM (b) rows are respected: the build_net isinstance chain,
registered_probes/registered_envelopes, GRAVES and the per-arch witness files stay; the findings below touch only
test-side walkers, helpers and individually implied assertions.

## Summary

- 14 findings. DEAD 2 (A 1, C 1) · TEST 2 (C 2) · DUP 8 (B 2, C 6) · DOC 2 (A 1, C 1). Lanes: A 2, B 2, C 10.
  Anything inside `tests/model/conformance/**` or a PZ-listed witness file is lane C by rule.
- Top 3 by Δ: S-A-TESTS-8-09 implied assertions -41 · S-A-TESTS-8-05 the real-wire batch recipe x5 ≈ -40 ·
  S-A-TESTS-8-02 dead conformance helpers -39.
- Gate 3: 09 and 03 delete 5 test functions in total. That is a test-floor move against a 250 margin
  (5112 collected vs floor 4862 per CALLERS §0). None of them has a tier_declaration.txt row (`grep tests/model
  tools/ci_gates/tier_declaration.txt` → rows 40–45 only, all kept).
- R8: no proposal takes a >300 file under 300. Checked: test_gnn_v2_witnesses 354 → ≥311 after 05/07/09;
  test_arch_selector 403 → ≈354; test_config_partition 490 → 481. The two files that shrink
  (test_legal_move_coverage_boundary 281, test_arch_reachability_and_graves 267) carry no header today.
- Archived-arch tests: the only HeXONet/ValueHead residue is the T11 grave fence plus its goods
  (`git grep -n -i "hexonet\|ValueHead\b" -- tests tools src ':!tests/fixtures'` → graves test + gine.py docstring). Keep it.
  The HexTacToeNet residue is S-L-SEAM-06 and S-L-SEAM A7/A8.
- Cross-slice builders: tests/train/conftest.py::make_tiny_arch is NOT a duplicate of the model-side tiny widths
  (hidden 16 vs 8, and seeded goldens `_V2_GOLDEN`/`small_gnn.pt` bind the model-side widths). The real
  cross-dir clone is `deterministic_algorithms` (11).

## Findings

### S-A-TESTS-8-01 | DEAD | A
subject: tests/model/test_arch_ban.py::_ARCH_ATTRS; tests/model/test_gine_bf16_drift.py::test_retired_max_form_rows_are_recorded_not_gating (loop variable `registered_abort`)
claim: `_ARCH_ATTRS` is a module constant that nothing reads (`_RE_ATTR` re-types the same four names inline), and `registered_abort` is an unpacked tuple element the loop body never uses.
evidence: `git grep -n -w _ARCH_ATTRS` → 1 hit (the def) · `uvx ruff check --isolated --select B007 tests/model` → `test_gine_bf16_drift.py:307:15: B007 Loop control variable registered_abort not used` · `uvx vulture` → same
callers: AST imports: `git grep -n -E "from test_arch_ban|import test_arch_ban"` → 0 (test modules imported by bare name are listed in CALLERS §8, and this one is not among them) · entry points: none in pyproject (CALLERS §1) · python -m / Makefile / shell / STATE: `git grep -n -w _ARCH_ATTRS -- docs Makefile tools '*.sh'` → 0 · subprocess strings: 0 · importlib/getattr: `git grep -n -E "[\"']_ARCH_ATTRS[\"']"` → 0 · conftest: none reads it · pyo3/config keys: n/a · gate paths: gate 14's census counts comments, not names · the `_GATING_ASSERT_CENSUS` AST pin reads only `assert` comparators, and the `4.0e-1` inside the tuple is not one, so dropping the element leaves the census unchanged.
Δlines: -1 (`grep -c _ARCH_ATTRS` = 1). registered_abort: 0 (an edit inside the line).
witness: tests/model/test_arch_ban.py (5 passed here); test_gine_bf16_drift.py::test_every_gating_row_still_asserts_its_registered_threshold [torch]
depends: —

### S-A-TESTS-8-02 | DEAD | C
subject: tests/model/conformance/test_legal_move_coverage_boundary.py::{NoWitnessConstructed, BoundaryNotWhereDerived, DensePartitionRefused}; test_arch_reachability_and_graves.py::consumers_of; test_config_partition_shared_vs_arch_scoped.py::leaf_present; test_leaf_forward_throughput_harness.py `Board` import
claim: Three refusal classes that are never raised, two helpers that are never called and one unused import. All are left over from the deleted dense arm and boundary derivation, and from the HexTacToeNet surfacing that R346(f) closed.
evidence: `uvx vulture tests/model --min-confidence 60` flags all five defs; `uvx ruff … --select F401` → `test_leaf_forward_throughput_harness.py:30:28 F401 mantis._engine.Board imported but unused`
callers: AST imports: `git grep -n -w <name>` over the whole tree → exactly 1 hit each (the def). The sibling bare-name imports are `from test_config_partition_shared_vs_arch_scoped import CONFIGS / live_leaf_paths` and `from test_arch_states_its_perf_floor import BUILD_SOURCE, arch_kinds_dispatched`; `git grep -n -E "from (test_[a-z_]+|_corpus) import" -- tests/model` → none names these · entry points: none · python -m / Makefile / shell / STATE / docs: `git grep -n -E "consumers_of|leaf_present|NoWitnessConstructed|BoundaryNotWhereDerived|DensePartitionRefused" -- docs Makefile tools '*.sh' '*.toml' '*.yaml'` → 0 · subprocess: 0 · importlib/getattr/monkeypatch strings: `git grep -n -E "[\"'](consumers_of|leaf_present)[\"']"` → 0 (the one monkeypatch string targets `arch_scoped_leaves`) · conftest: conformance/conftest.py defines only `derived` · pyo3 / config keys: n/a · gate tool paths: the tier census and marker census read decorators only, and none of these carry one.
Δlines: -39 = classes 3×(2 + 2 blank) = 12 + consumers_of 16 + 2 blank = 18 + leaf_present 7 + 2 blank = 9 (AST spans 48–57, 94–109, 197–203). `Board`: 0.
witness: the three host modules themselves [torch]; `git grep` above
depends: —

### S-A-TESTS-8-03 | TEST | C
subject: tests/model/conformance/test_arch_selector_makes_v2_selectable.py::test_the_kind_vocabulary_is_set_equal_to_build_nets_dispatch
claim: This test's assertion is `{name in ARCH_KINDS if "isinstance(arch, name)" in source} == set(ARCH_KINDS)`. That checks only that every kind has a branch. The same equality, in BOTH directions, is asserted by test_arch_reachability_and_graves.py::test_the_dispatch_census_and_the_arch_kind_registry_agree (AST, `set(pairs) == set(ARCH_KINDS)`). It is asserted a third time by test_arch_states_its_perf_floor.py::test_the_CHECKPOINT_LOADERS_arch_registry_matches_the_same_dispatch, because `src/mantis/train/checkpoints.py::_ARCH_KINDS = ARCH_KINDS` is an alias.
evidence: `sed -n 108,125p` (the substring scan) · `git grep -n "_ARCH_KINDS = ARCH_KINDS" -- src` → checkpoints.py:137
Δlines: -20 (AST span 108–125 = 18, + 2 blank). Test count -1 (floor move).
witness: the graves and perf_floor tests named above stay [torch]
depends: — (S-L-SEAM (b) row 2 cites this grep as one of the chain's census tests. The chain stays parsed by the two AST walkers.)

### S-A-TESTS-8-04 | DUP | C
subject: tests/model/conformance/test_arch_states_its_perf_floor.py::arch_kinds_dispatched vs test_arch_reachability_and_graves.py::dispatch_pairs (+ ::dispatch_census)
claim: Two AST walkers parse build_net's isinstance chain inside one suite. The memory-envelope module imports the perf-floor walker and says why in its own docstring: "two walkers over one dispatch is two authorities, and the one nobody looks at goes stale". Keep one walker that yields the kind→net pairs, accepts a source string (for the planted breaks) and handles Tuple targets. The fourth walker, tests/model/test_arch_v2_dispatch.py::test_the_V2_branch_PRECEDES_the_V1_branch_in_the_dispatch, is ORDERED and stays.
evidence: AST spans: arch_kinds_dispatched 77–96 (20), dispatch_pairs 66–86 (21), dispatch_census 89–91 (3)
deliberate?: The chain being parsed is the seam (S-L-SEAM (b)). Parsing it twice is not: neither module's docstring argues for a second walker, and the envelope's docstring argues against one.
Δlines: ≈ -15 (the 20-line span of the retired walker, minus about 5 lines the survivor gains for the source-string and Tuple forms; a bound, not derived)
witness: the planted-break tests in perf_floor (PB-T7a, negative control) and the graves census tests [torch]
depends: 03

### S-A-TESTS-8-05 | DUP | C
subject: the real-wire batch recipe — `_corpus.py::graph_wire_for` (push part), test_arch_states_its_memory_envelope.py::_gnn_batch, test_arch_states_its_perf_floor.py::_gnn_probe_arms.{wire,collate}, test_leaf_forward_throughput_harness.py::_tiny_graph_forward.build, tests/model/test_gnn_v2_witnesses.py::test_both_arches_FORWARD_on_a_real_wire_position
claim: The same HexgBuffer push → sample → collate(device="cpu") → stone-mask block appears 5 times. The stone mask is hand-rolled 4 times (`stone_mask[: int(batch.n_stones.sum())] = True`, correct only at B=1) even though production has src/mantis/selfplay/graph_collate.py::stone_mask_from_batch, which test_pmask_gather_parity already uses. Proposal: `_corpus.sampled_wire(enc, board)` and `_corpus.collated_cpu_batch(spec, wire)`, the latter keeping `device="cpu"`.
evidence: `git grep -n "board.current_player, board.moves_remaining, board.ply, True, 0.0, True, 8" -- tests` → exactly these 5 files · jscpd → 6 of the 10 tests/model clones are this block · `git grep -n "stone_mask\[: int(batch.n_stones.sum())\] = True" -- tests` → 4 · `sed -n` line counts: perf 211–229 = 19, leaf 636–653 = 18, envelope 238–261 = 24, corpus 161–169 = 9, witnesses 324–341 = 18
deliberate?: No. Each copy's comment "Stated, not sniffed" records the same one fact. RULING CONTACT: CARD-GATES-ON-CUDA-VENV / R349(a) names "the four collate sites in the model oracles" stating `device="cpu"`. The one shared site must keep it.
Δlines: ≈ -40 conformance-only. Removed: envelope 22 (24 minus a 2-line wrapper) + perf 17 + leaf 14 + corpus 6 = 59. Added: about 19 lines of helpers. The witnesses copy (-15 more) needs the helper at the tests/ root, because `_corpus` is not on sys.path for tests/model files; tests/_drivable.py shows a root helper is bare-importable from every subdir.
witness: every conformance tier that builds a batch; test_gnn_v2_witnesses.py [torch]
depends: —

### S-A-TESTS-8-06 | DUP | C
subject: test_arch_states_its_memory_envelope.py::check_envelope_manifest vs test_arch_states_its_perf_floor.py::check_floor_manifest
claim: Same shape: an empty refusal, `dispatched - registered` missing, `registered - dispatched` stray. Only the exception classes and messages differ. This is the companion of S-A-TESTS-8-04's merge of the two `specs_for` copies (see S-L-SEAM-04): one parametrised `check_dispatch_manifest` in `_corpus`.
evidence: AST spans: envelope 120–143 (24), floor 99–125 (27); `sed` shows the same three-branch body
deliberate?: Each arch must STATE its floor and its envelope (S-L-SEAM (b) row 6). That is about the registries. The checker has no such argument.
Δlines: ≈ -21 (the envelope copy 24 + 2 blank, minus about 5 lines of exception and message parameters; a bound)
witness: PB-T7b/T7c, PB-T8b and the EMPTY-census refusals in both modules [torch]
depends: S-L-SEAM-04

### S-A-TESTS-8-07 | DUP | C
subject: tests/model/test_gnn_v2_witnesses.py::_star_graph vs tests/model/conformance/test_arch_selector_makes_v2_selectable.py::_batch
claim: The same synthetic star graph (n_real nodes plus one dummy wired both ways, zero edge attrs, first n_stones rows stones) is built twice. `_readout_masks` in the witnesses file derives the same masks a third time.
evidence: jscpd → `test_gnn_v2_witnesses.py [37–46] ↔ test_arch_selector_makes_v2_selectable.py [231–240]` (9 lines, 162 tokens); AST spans 20 and 22
deliberate?: The two sides differ only in seeding (the selector seeds inside) and in the witnesses returning `legal_mask`.
Δlines: ≈ -18 (one 20-line copy, plus about 2 lines for the import and the seed call). Needs a tests/-root helper for the same sys.path reason as 05.
witness: W-A1/W-A3/W-C1 witnesses; the selector round-trip rows (`policy.shape[0] == 9`) [torch]
depends: 05 (same helper home)

### S-A-TESTS-8-08 | DUP | C
subject: test_arch_selector_makes_v2_selectable.py::{SelectorWentVacuous, _config_source, _graph_config_source} vs test_config_partition_shared_vs_arch_scoped.py::config_for
claim: Both scan `sorted(CONFIGS.glob("*.yaml"))` for the first config whose `identity.representation` matches. `_config_source` is only ever called as `_config_source("graph")` through a 2-line forwarder. The selector already imports `CONFIGS` and `live_leaf_paths` from the partition module.
evidence: `git grep -n -w "_config_source\|_graph_config_source\|SelectorWentVacuous\|config_for" -- tests` → the selector's only call is through `_graph_config_source` (3 sites); `config_for` has 4 sites
Δlines: -17 (lines 70–91 = 22, replaced by a 3-line `_graph_config_source` over `config_for` plus 2 blank). The refusal on "no graph config" becomes ArchVocabularyKeyUnplaced; the reviewer judges the message.
witness: the selector round-trip and row-honoured tests [torch]
depends: —

### S-A-TESTS-8-09 | TEST | C
subject: (a) test_arch_selector_makes_v2_selectable.py::test_the_selected_V2_arch_is_the_SIBLING_dataclass_and_not_V1; (b) test_arch_reachability_and_graves.py::test_every_shipped_config_selects_a_net_build_net_can_construct; (c) tests/model/test_gnn_v2_witnesses.py::test_the_V2_golden_is_NOT_the_V1_golden; (d) tests/model/test_arch_v2_dispatch.py::test_the_declared_arch_is_the_HANDLE_the_built_net_carries
claim: Each makes no assertion that other tests do not already make together.
(a) The GnnArchV2 round-trip row asserts `type(arch) is ARCH_KINDS["GnnArchV2"]` through the same selector, and test_arch_v2_dispatch asserts `not issubclass(GnnArchV2, GnnArch)`.
(b) The selector's test_every_shipped_config_selects_the_kind_its_own_row_declares asserts `type(arch) is ARCH_KINDS[…]` per config, and graves test 2 asserts dispatch keys == ARCH_KINDS. `nets_selected` stays live for test_RunConfig_cannot_select_a_buried_arch.
(c) W_ID1 asserts hash(v1) != hash(v2), and test_the_V2_golden_holds asserts hash(v2) == golden, so hash(v1) != golden is already implied.
(d) This is the same `net.arch = arch` line (build.py) that test_build_net_arch_handle.py pins with `is` plus the state-dict and registry checks. Parametrising that file over ARCH_KINDS instead would also cover GnnArchV2SoftPolicy, which neither file covers today.
evidence: AST spans (a) 326–335, (b) 176–185, (c) 309–313, (d) 70–77; the implying tests were read in full
deliberate?: S-L-SEAM (b) keeps the witness FILES. These are single rows inside them whose claim another row already proves.
Δlines: -41 = (10+2) + (10+2) + (5+2) + (8+2). Test count -4 (floor move; with (d) parametrised over 3 kinds the count rises by 2 instead).
witness: the implying tests named above [torch]
depends: —

### S-A-TESTS-8-10 | DUP | B
subject: tests/model/test_pmask_gather_parity.py::payload_fields (+ `_COLLATE` and its 4-line comment)
claim: The file's own comment says it re-implements tests/selfplay/conftest.py::{_payload_bank, collate_expectations, payload_fields} because a subdirectory conftest does not reach tests/model. Move the file to tests/selfplay/ (it drives collate_graph_batch and stone_mask_from_batch), or hoist the fixtures to tests/conftest.py.
evidence: `sed -n 30,50p` (the copy and its comment) vs `grep -n -A16 "def payload_fields" tests/selfplay/conftest.py`
deliberate?: No. The comment calls it a forced copy ("the two-line loader is duplicated"), not an independent oracle. Both read the same fixture files.
Δlines: -19 (fixture span 15 + comment 4). No test-count change.
witness: test_pmask_gather_parity.py (8 rows incl. the mutation rows) [torch]
depends: HANDOFF T5 (the conftest side)

### S-A-TESTS-8-11 | DUP | B
subject: tests/model/_bf16_parity.py::deterministic_algorithms vs tests/train/_microbatch_harness.py::deterministic_algorithms
claim: The two bodies are identical (a context manager for the process-global determinism mode plus a CUBLAS env restore). Only the model copy has a producer test (test_bf16_parity_nulldist.py::test_test_scope_determinism_does_not_leak_to_sibling_tests). Keep one copy at the tests/ root and re-import it in `_bf16_parity`, so `bp.deterministic_algorithms` in the PZ file keeps resolving unedited.
evidence: cross-dir jscpd → `_bf16_parity.py [68–84] ↔ _microbatch_harness.py [34–50]` (16 lines); AST spans 62–81 (20) and 27–47 (21); callers: nulldist 4 sites via `bp.`, test_graph_microbatch 3 sites via `H.`
deliberate?: Neither copy cites the other. The train docstring restates the same "TEST SCOPE ONLY" claim.
Δlines: ≈ -20 (one copy, plus about 1 import line)
witness: test_test_scope_determinism_does_not_leak_to_sibling_tests; tests/train/test_graph_microbatch.py [torch]
depends: HANDOFF T1

### S-A-TESTS-8-12 | DUP | C
subject: tests/model/test_bf16_parity_nulldist.py::_bf16_ulp vs the inline ulp in test_gine_bf16_drift.py::test_retired_max_form_rows_are_recorded_not_gating; the all-pairs bit-identity loop in test_mutation_green_cpu_exact_… vs test_mutation_green_cuda_exact_…
claim: The same bf16 spacing arithmetic appears twice, and the same 15-pair `torch.equal` loop appears twice. Both could live in `_bf16_parity`.
evidence: jscpd → nulldist `[405–410] ↔ [333–338]`; `sed -n 51,56p` (nulldist) vs `sed -n 330,332p` (drift)
Δlines: ≈ -7 (ulp -2, pairs loop about -5)
witness: both files [torch]; `_GATING_ASSERT_CENSUS` is unaffected because the asserted literals stay in place
depends: —

### S-A-TESTS-8-13 | DOC | C
subject: stale text in protected tests: (a) test_no_second_arch_kind_table.py::test_the_census_can_actually_SEE_a_table docstring "the vocabulary now has exactly two kinds"; (b) test_config_partition_shared_vs_arch_scoped.py::SHARED_DESPITE_THE_NAME["identity.arch_kind"] "absent until the run6 mint writes it"; (c) test_one_amp_dtype_authority.py::test_every_autocast_in_src_names_a_dtype message "not from `train.amp_dtype`"; (d) test_arch_reachability_and_graves.py module docstring "anything load-bearing is SURFACED with its consumers NAMED" (the surfacing helper is dead, 02); (e) test_gnn_v2_witnesses.py `widths["n_value_bins"] if "n_value_bins" in widths else 65` (a dead branch; the 65 transcribes N_VALUE_BINS)
claim: Each text disagrees with the tree.
evidence: (a) ARCH_KINDS has 3 keys (arch.py) · (b) the selector's test_the_selector_row… asserts that every production config carries the row · (c) `.venv/bin/python -c "…leaf_paths…"` → no `train.amp_dtype` leaf · (d) 02 · (e) `_WIDTHS` keys read in full
Δlines: ≈ 0
witness: none (text)
depends: 02

### S-A-TESTS-8-14 | DOC | A
subject: tests/model/test_gnn_v2_soft_policy.py module docstring
claim: It still claims "the two soft-policy kind tables agree", but 3d93447 (REVIEW-1 F4) deleted the second table and that test.
evidence: `git show 3d93447 -- tests/model/test_gnn_v2_soft_policy.py` → `-def test_the_schema_and_model_tables_of_soft_policy_kinds_agree`
Δlines: 0 (an edit inside the line)
witness: none
depends: —

## DEFECTS
- test_arch_reachability_and_graves.py::test_the_grave_guard_can_FIRE raises GraveDisturbed from its own inline code inside `pytest.raises`. It never drives test_a_GRAVE_stays_dead's check, so a broken fence stays green (only `net_classes` is exercised).
- test_no_second_arch_kind_table.py::test_the_census_can_actually_SEE_a_table never calls `_arch_keyed_dict_literals` (its root is hard-coded to src/mantis). It re-implements the predicate, so a walk that finds nothing still passes the self-test.
- tests/model/test_arch_v2_dispatch.py::test_every_arch_ROUND_TRIPS_through_the_checkpoint_serializer says "every arch" but parametrises `[_V1, _V2]`, so GnnArchV2SoftPolicy is absent. tests/train/test_checkpoint_conformance.py may cover it; not verified.
- The hand-typed net-name sets in test_one_amp_dtype_authority / test_arch_ban are already filed as S-L-SEAM A7/A8.

## PARKED
- test_arch_reachability_and_graves.py::test_a_GRAVE_stays_dead re-parses every .py under src/, tools/ and tests/ (748 files, 148 913 lines) once per grave, ×3. A session-scoped parse would do it once.

## HANDOFF
- T5: tests/selfplay/conftest.py::payload_fields / _payload_bank / collate_expectations are the other side of S-A-TESTS-8-10.
- T1: tests/train/_microbatch_harness.py::deterministic_algorithms is the other side of S-A-TESTS-8-11.
- T6: tests/eval/test_deploy_matched_hparam_coincidence.py asserts `amp_dtype_for("graph") is torch.bfloat16` verbatim. That duplicates tests/model/test_amp_dtype.py::test_graph_is_bf16_unconditionally, which is the dedicated LAW-06 pin; if either copy goes, it is the eval one.

## Not covered
- Nothing torch-dependent was executed: 26 of the 28 test modules fail collection here, and every claim about them is static.
- The value of the R181 bf16 oracle bank (test_gine_bf16_drift.py says of itself that it kills no F1 mutant on CPU) is a design question, not raised.
- Deliberate twins, not raised: the in-suite tier census (`_SLOW_TIER_MEMBERS`, marker_census) vs tools/ci_gates/tier_census.py + tier_declaration.txt, both declared twins and cross-checked by tests/tools/test_tier_census.py; and test_net_param_hash_promotion's stability control, which the golden implies but which is a PZ-1 pinning test.
- Merging test_amp_dtype.py into test_one_amp_dtype_authority.py was weighed and dropped: it would make the torch-free census torch-dependent.
- The six separate "walk src/**/*.py and ast.parse" loops across tests/model, each 3–5 lines, were not costed.

## Review
reviewer: fresh read-only agent (not the author); probes in throwaway worktrees, removed.
Reviewed at HEAD af4d37e. `git diff --stat 69e1532 HEAD -- tests src tools crates` is empty, so every subject is unchanged from the census base.
Roster guard: test_conformance_roster_guard.py compares the ENCODING registry roster (`_corpus.roster_names` vs `encoding.all_specs()`). It enumerates no refusal class or helper by name. `git grep -n -E "getmembers|__subclasses__|vars\(|globals\(\)" -- tests/model` → no enumeration of test-module names. So no dead subject here is roster-guarded.

| ID | verdict | lane | Δlines (probe-measured for lane A) | note |
|---|---|---|---|---|
| 01a `_ARCH_ATTRS` | CONFIRMED | A | −1 (probe diff) | probe green: collect 2289/167 (= baseline), test_arch_ban 5 passed |
| 01b `registered_abort` | PENDING-PROBE | A | 0 (probe diff: 3+/3−) | torch-bound: test_gine_bf16_drift errors at collection. The AST census was re-derived as unchanged |
| 02 | CONFIRMED | C | −39 (unprobed; lane C) | not roster-guarded. Only def hits outside docs/slim |
| 03 | CONFIRMED | C | −20 | implied by graves `set(pairs) == set(ARCH_KINDS)`. Test-floor move (tools/ci_gates/test_count_floor.txt). Touches an S-L-SEAM (b) row-2 census test |
| 04 | AMENDED | C | not derivable (the scout's −15 bound fails) | the two walkers recognise DIFFERENT shapes, so a merge changes what each census sees |
| 05 | CONFIRMED | C | ≈ −40 (bound) | makes CARDS' R349(a) closure text ("the four collate sites") stale. It can be annotated only |
| 06 | AMENDED | C | ≈ −10 (was −21) | the three per-tier refusal messages must survive as parameters |
| 07 | AMENDED | C | ≈ −13 standalone (−18 only if it shares 05's new root module) | the RNG draw order must be kept |
| 08 | AMENDED | C | −17 | config_for's refusal message is written for the partition tier and must be generalised |
| 09 | AMENDED | C | −31 (was −41), tests −3 (was −4) | (d) REFUTED: it is the ONLY V2 arch-handle pin |
| 10 | CONFIRMED | B | −19 | no path citation outside docs/slim |
| 11 | CONFIRMED | B | ≈ −20 | AST-identical bodies (sans docstring) |
| 12 | CONFIRMED | C | ≈ −7 (bound) | the CPU pairs loop carries one extra assert that the helper must keep |
| 13 | CONFIRMED | C | 0 | all five texts re-derived as stale |
| 14 | CONFIRMED | A | 0 | docstring clause only (DOC; no delete-probe required) |

### Per-finding notes
S-A-TESTS-8-01 — split verdict.
- 01a CONFIRMED. `git grep -n -w _ARCH_ATTRS` → only the def (test_arch_ban.py:19) plus docs/slim. `_RE_ATTR`/`_RE_HASATTR` re-type the names inline.
- 01a DELETE-PROBE, worktree scratchpad/wt/tests8-01 at HEAD, with the `-S` + isolating PYTHONPATH recipe:
  - `import mantis` resolves to the worktree src;
  - collect-only → `2289 tests collected, 167 errors` (= baseline);
  - `pytest tests/model/test_arch_ban.py` → 5 passed;
  - `git diff --stat` → `2 files changed, 3 insertions(+), 4 deletions(-)`.
- cargo check was not run. The subjects are Python test modules, and `git grep -E "test_arch_ban|test_gine_bf16_drift" -- crates tools Makefile` → 0, so no include_str!, bench or gate reaches them.
- 01b PENDING-PROBE (torch-bound). I dropped the tuple element and its two row literals. An AST re-run of `test_every_gating_row_still_asserts_its_registered_threshold`'s census gives `[0.001953125, 0.4]` both at HEAD and in the worktree: the 4.0e-1 comes from the separate `assert policy_null_max > 4.0e-1`. Collection is unchanged. The row itself cannot run without torch.

S-A-TESTS-8-02 — CONFIRMED (lane C by PZ-1).
- `git grep -n -w <name> -- ':!docs/slim'` → exactly the def for all five. The three classes are at test_legal_move_coverage_boundary.py:48/52/56.
- No dynamic use: no `__subclasses__`/getmembers anywhere in tests or tools, and conformance/conftest.py defines only `derived`.
- The `Board` import is used only in a docstring (harness:11/88), so F401 is real.

S-A-TESTS-8-03 — CONFIRMED.
- The selector's set comprehension is ⊆ ARCH_KINDS by construction, so its equality is one-directional: every kind has an `isinstance(arch, K)` substring.
- graves::test_the_dispatch_census_and_the_arch_kind_registry_agree asserts `set(pairs) == set(ARCH_KINDS)` over If-test isinstance nodes with an assigned-Call body. That is strictly stronger, except for the literal receiver name `arch` and the whitespace, which are not a claim.
- The `derived("t10.*")` keys have no consumer: `git grep -E "t10\.|t11\." -- ':!tests/model/conformance' ':!docs/slim'` → 0.
- No governance or tools citation of the test name. Floor file tools/ci_gates/test_count_floor.txt (4862) moves.

S-A-TESTS-8-04 — AMENDED (Δ not derivable; still lane C). The two walkers are not one reading of the chain:
- `perf_floor::arch_kinds_dispatched` collects EVERY `isinstance(_, Name|Tuple)` Call inside build_net, whatever the branch body.
- `graves::dispatch_pairs` records a kind only for an `ast.If` whose test is isinstance(_, Name) AND whose body holds an `x = Call(...)` Assign.
- The planted breaks (PB-T7a and the negative control) use `return B()` bodies (`sed -n 300,332p test_arch_states_its_perf_floor.py`), which `dispatch_pairs` would read as ZERO kinds. A merged walker must carry both arms, or the planted sources must be rewritten, and either changes which mutation reds which census.
- The envelope docstring's "two walkers … two authorities" is about re-walking inside the envelope module, which it avoids by importing. The scout's bound of the survivor gaining "≈5 lines" is unsupported.

S-A-TESTS-8-05 — CONFIRMED.
- `git grep -n "board.current_player, board.moves_remaining, board.ply, True, 0.0, True, 8" -- tests` → exactly the 5 files.
- `git grep -n -F "stone_mask[: int(batch.n_stones.sum())] = True" -- tests` → 4.
- `graph_collate.py::stone_mask_from_batch`'s docstring states the per-graph layout `[stones | legal | dummy]`. The prefix mask is therefore right only at B=1, and every site samples 1.
- Ruling contact is real: CARDS.md (the R349(a) closure) names "the four collate sites in the model oracles". That governance text goes stale and is corrected by annotation only.
- `_corpus` imports torch nowhere at module scope today. The shared collate helper must keep torch lazy. Moot for tier membership, because the roster guard already fails collection on torch.

S-A-TESTS-8-06 — AMENDED.
- The shape claim holds (read both bodies).
- The Δ does not. Of the envelope body's 18 lines, 6 are the three tier-specific messages, which state each tier's reason and interpolate `missing`/`stray`. They must be passed in, as templates or callables, alongside three exception classes.
- The floor checker is called at 6 sites and the envelope checker at 2, so thin per-module wrappers must stay. Net ≈ −10.
- If S-L-SEAM-02 lands, its review deletes PK3, one of the floor call sites.

S-A-TESTS-8-07 — AMENDED.
- The bodies are identical apart from seeding and the returned keys.
- `_batch` seeds INSIDE, right before `randn`, and the `_star_graph` callers seed before the call. A shared helper must take the seed as an optional argument so the draw stays bit-identical.
- A new tests/-root module costs its own `from __future__` / `import torch` / docstring. Standalone ≈ −13.

S-A-TESTS-8-08 — AMENDED.
- `grep -n -w` confirms: `_config_source` has only the `"graph"` call (via `_graph_config_source`, 3 sites: 98/374/396), and `SelectorWentVacuous` is raised once and matched by no `pytest.raises`.
- `config_for`'s refusal reads "…the cross-arch reachability of an arch-scoped key cannot be executed…". That would mis-describe a selector failure, so its message must be made tier-neutral (an in-place edit in the partition module, 0 Δ).

S-A-TESTS-8-09 — AMENDED.
- (a) implied: the round-trip row is parametrised over `ARCH_KINDS_BY_REPRESENTATION["graph"]` and asserts `type(arch) is ARCH_KINDS[arch_kind]` through `select_arch`. Together with test_arch_v2_dispatch's `not issubclass(GnnArchV2, GnnArch)`, that implies both isinstance asserts.
- (b) implied: the selector asserts `type(arch) is ARCH_KINDS[…]` per `discover_configs` file, and graves asserts `set(pairs) == set(ARCH_KINDS)`. Together they give `set(nets_selected()) == set(selected)`.
- (c) implied, by W_ID1 plus golden_holds.
- (d) REFUTED:
  - test_build_net_arch_handle.py::_archs builds only `gnn_axis_v1`/`gnn_axis_r8` and asserts `all(isinstance(a, GnnArch) …)`, so V1 only.
  - `git grep -n -E "\.arch is |net\.arch\b" -- tests` shows no other test pins `build_net(V2).arch is V2`, or its absence from the state dict.
  - A mutation that moves `net.arch = arch` inside the V1 branch reds only (d).
  - The scout's alternative (parametrise the handle file over ARCH_KINDS, then delete (d)) is a different, lane-B change that raises the count.
- Net −31, tests −3, floor move. There are no governance citations of any of the four names.

S-A-TESTS-8-10 — CONFIRMED (B).
- `sed -n 28,52p` shows a 3-line comment, `_COLLATE` and a 15-line fixture. The selfplay conftest side is session-scoped `_payload_bank`/`collate_expectations`/`payload_fields`.
- `git grep -l test_pmask_gather_parity -- ':!docs/slim'` → 0, so the file can move without breaking a citation.
- Hoisting to tests/conftest.py makes those session fixtures global. That is a design choice (B).

S-A-TESTS-8-11 — CONFIRMED (B).
- An AST dump of both `deterministic_algorithms`, docstrings stripped, compares equal (spans 63–81 and 28–47; both `@contextlib.contextmanager`).
- The docs/governance hits for the name are `torch.use_deterministic_algorithms`, not this symbol.
- Note: `_bf16_parity` backs the PZ file test_bf16_parity_nulldist.py (PZ list). Lane B holds only if that file stays unedited, as proposed.

S-A-TESTS-8-12 — CONFIRMED (C; nulldist is PZ).
- The ulp arithmetic is identical (nulldist `_bf16_ulp` vs drift's inline `quarter` block).
- The pair loops (nulldist 331–339 vs 403–410) differ: the CPU loop carries an extra `assert bp.median_form(...) == 0.0`, which a shared helper must keep.
- Drift's `assert ulp_p90 == 1.953125e-3` stays in place, so the census is unaffected.

S-A-TESTS-8-13 — CONFIRMED.
- (a) arch.py::ARCH_KINDS has 3 rows (96–101) against "exactly two kinds" (no_second_arch_kind_table:59).
- (b) partition:139–141 says "absent until the run6 mint writes it".
- (c) `git grep amp_dtype -- configs` → 0. There is no config leaf, and the message at test_one_amp_dtype_authority.py:64–65 names one.
- (d) follows from 02.
- (e) `widths = {**_WIDTHS, "in_dim", "edge_dim"}` never holds `n_value_bins`, and dist65.py::N_VALUE_BINS = 65.

S-A-TESTS-8-14 — CONFIRMED. The docstring still says "the two soft-policy kind tables agree", and `git show 3d93447 -- tests/model/test_gnn_v2_soft_policy.py` shows that test deleted. This is a docstring edit with 0 Δ; the file is torch-bound.

### Defect claims
- D1 grave self-test — CONFIRMED. `test_the_grave_guard_can_FIRE` re-types the `if name in classes: raise GraveDisturbed` branch inline inside `pytest.raises`. It never calls `test_a_GRAVE_stays_dead`'s body or a shared checker, so a broken guard (or the `dispatch_census()` and USED-names arms) stays green. `_names_used` alone has a real self-test (test_the_consumer_census_counts_USES_and_not_MENTIONS).
- D2 arch-kind-table self-test — CONFIRMED. `test_the_census_can_actually_SEE_a_table` parses its own planted file and calls only `_dict_key_strings`, restating the `>= _MIN_KEYS_TO_JUDGE` predicate. `_arch_keyed_dict_literals` walks the hard-coded `_SRC`, and the census test asserts only `not incomplete`, never that `found` is non-empty. A walk that finds nothing passes both.
- D3 SoftPolicy round-trip — CONFIRMED, and the scout's open question is now closed. test_arch_v2_dispatch.py:80 parametrises `[_V1, _V2]`. No test calls `_arch_to_dict`/`_arch_from_dict` on a GnnArchV2SoftPolicy: `git grep -E "_arch_(to|from)_dict" -- tests` → test_arch_v2_dispatch plus a message string in perf_floor. The checkpoint tests that mention soft policy set `"aux_soft_policy": None` (test_checkpoint_conformance.py:648, test_resume_wiring_integration.py:129, train/conftest.py:217). test_gnn_v2_soft_policy round-trips through the eval snapshot serializer, a different path (S-L-SEAM-01).

### Cross-check with S-L-SEAM
No double count.
- 06 depends on S-L-SEAM-04 (the `specs_for` pair) and does not re-raise it.
- S-L-SEAM-06 (flatten_ban) and A7/A8 are absent from this file's findings.
- 03 deletes one of the three census readers that S-L-SEAM (b) row 2 lists as reasons to KEEP the isinstance chain. The chain stays parsed by the two AST walkers, so that row's argument survives.

### Missed by the scout
None raised. The R8 re-checks hold: after all proposals, test_arch_selector stays > 300 (403 − 20 − 18 − 17 − 12 → 336) and test_gnn_v2_witnesses stays > 300 (354 − 15 − 18 − 7 → 314), so both keep their headers.

### Tally: raised 14 | confirmed 8 (01a counted within 01) | amended 5 (04, 06, 07, 08, 09) | refuted 0 whole (09(d) refuted inside 09) | pending 1 (01b) | architect 0
