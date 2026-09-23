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
