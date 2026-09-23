# S-A-TESTS-9 — T9 tests/fixtures

scope: `tests/fixtures/**` at HEAD e7c1d08 (tree identical to 69e1532 for this slice: `git diff --stat 69e1532 HEAD -- tests/fixtures` → empty), 64 files, 33 binary, 182 119 text lines;
method: `git ls-files`, `comm` of the manifest.toml rows against the tracked files, `sha256sum` over all 64 files, `git diff --numstat <empty-tree> HEAD`, `wc -c`,
per-file `git grep -l -F <basename>` plus Path-join / dir-name / glob scans over src tools tests crates docs Makefile,
Rust `include_str!` and runtime-path reads, `pytest --collect-only` per reader file, and a torch-free run of the readers that can run here.
Large data files were not opened. Only their headers and top-level keys were read.

## Summary

- Findings: 6. ONE-SHOT 3 (C 3), DOC 2 (C 1, B 1), TEST 1 (C 1). Lane A: 0. Everything with a real saving touches a sha-pinned row or a ruling, so it is lane C.
- Top 3 by Δ: -01 · 03 `graph_parity/wpa_positions.json` -124 772 lines (966 939 B) · -02 · 06 dense drain-arm fixtures ≥ -3 686 lines and -86 558 B binary (before the B-15 re-base) · -03 · 01 the dead O2 value-health bank -938 lines, -15 170 B binary (2 887 686 B in total).
- Manifest coherence: 45 `[[required]]` rows, 0 rows naming a missing file (`comm -23` → empty), and all 45 digests match (`pytest tests/test_fixtures_manifest.py` → 4 passed).
  - 19 files have no manifest.toml row: `manifest.toml` itself; the 16 `graph_parity/{inputs.bin, raw/*.bin ×14, wpa_positions.json}`, which are covered by F-rows in `graph_parity/manifest.tsv`; `silent_encoding_evasions.toml`, a live gate-11 corpus that is not a golden; and `search_golden/temperature/gen_temperature_parity_golden.py`.
- Duplicates: none. `sha256sum` over all 64 files, then `uniq -d` on the digests → empty.
  - The four 10 718 B `raw/case_0000[0-3].bin` blobs have the same size but distinct digests.
  - The two mctx readers (`crates/mantis-search/src/mcts/parity_tests.rs`, 8 tests on the completed-Q surfaces, and `crates/mantis-search/tests/mctx_parity.rs`, 3 tests on the seq-halving surfaces) pin different surfaces of one fixture. That is deliberate, not DUP.
- Gate 6 is unaffected: every finding is a deletion or an in-place edit. `artifact_gate.py` checks only A/R-status adds. The one M-status touch is the `graph_parity/manifest.tsv` re-pin in 03, which is 2.29 MB and below the 10 MB fixture ceiling.
  - After 01, `git ls-files '*.jsonl'` would be empty, because the two value_probes files are the only `*.jsonl` in the repo. The jsonl carve-out stays valid, just with nothing under it.
- Gate 16 is unaffected. The generator in 04 has one function-scope `write_text` without `encoding=` (a backlog site, not a violation). `MIN_FILES["tests"]=200` against about 254 measured.
- Gate 3: no finding deletes a test except 06, which is owned by the B-15 re-base. `tests/fixtures` collects 0 tests (`pytest --collect-only tests/fixtures` → "no tests collected").
- Floor to watch: `tests/test_line_endings.py::test_gitattributes_marks_every_byte_significant_path` asserts `len(paths) >= 40`, where paths = manifest rows + 1.
  - Now: 46. After 01+02: 41. After 01+02+06: 39, which is RED, so the floor must be lowered in the same commit (T7 owns it).

### File-level census (what reads each fixture)

`toml` = has a manifest.toml row · `tsv` = has an F-row in graph_parity/manifest.tsv · `[torch]` = the reader does not collect here (no torch) but exists and is collected on the box.

| fixture(s) | text lines / bytes | pin | live readers | generator |
|---|---|---|---|---|
| bf16_nulldist/measurement_raw_R181_NULLDIST.json | 1 / 976 398 | toml | tests/model/_bf16_parity.py (sha-verified on read) → test_bf16_parity_nulldist.py, test_gine_bf16_drift.py [torch] | off-tree box capture |
| board/board_replay_golden_v1.json | 1 / 3 021 473 | toml | crates/mantis-core/tests/golden_replay.rs (2 tests) | off-tree |
| eval_selfplay_parity/child_parity_v1.json, dispersed_r6_v1.json | 34+54 / 30 254+26 642 | toml | crates/mantis-selfplay/tests/graph_child_parity.rs; tests/eval/test_eval_selfplay_child_parity.py (+ dispersed: test_eval_value_channel.py, test_rung_seat_off_window.py) [torch] | graph_child_parity.rs call sequence |
| eval_selfplay_parity/target_parity{,_dispersed}_v1.json | 45+45 / 4 261+5 836 | toml | crates/mantis-selfplay/tests/target_export_parity.rs; tests/selfplay/test_target_export_parity.py (4 passed here) | — |
| graph_parity/manifest.tsv | 23 771 / 2 286 437 | toml | crates/mantis-graph/tests/{graph_parity.rs, fixture_selftest.rs, common/mod.rs} | migration workspace (off-tree) |
| graph_parity/inputs.bin | bin 871 174 | tsv | graph_parity.rs, benches/build_bench.rs | same |
| graph_parity/raw/case_*.bin ×14 | bin 1 591 254 | tsv | graph_parity.rs (`format!("raw/case_{:05}.bin")` over the `raw_subset` header), fixture_selftest.rs | same |
| graph_parity/wpa_positions.json | 124 769 / 966 939 | tsv | **only the F-row sha/size check** → 03 | same |
| ladder/recorded_stream.ndjson, recorded_two_games.ndjson | 8+8 / 1 551+1 064 | toml | tests/tools/test_ladder_client.py, test_ladder_session.py [torch via mantis.util.device] | — |
| manifest.toml | 331 / 15 728 | (the registry itself) | tests/test_fixtures_manifest.py, tests/test_line_endings.py | hand-kept |
| mctx_parity/mctx_parity_v1.json | 18 822 / 376 023 | toml | parity_tests.rs (8), tests/mctx_parity.rs (3) | tools/gen_mctx_parity_fixtures.py (manual, jax venv) |
| model_graves/hexonet_grave_v1.json | 158 / 3 080 | toml | tests/model/conformance/test_arch_reachability_and_graves.py [torch] | burial commit |
| replay/hexg_v1_golden.hexg, hexg_v2_golden.hexg | bin 148+156 | toml | crates/mantis-selfplay/tests/replay_hexg.rs (v1 = the refusal test) | — |
| search_golden/completed_q/golden_bits.txt | 22 / 15 090 | toml | `include_str!` golden_tests.rs::GOLDEN_BITS | — |
| search_golden/temperature/temperature_parity_golden.csv | 73 / 1 905 | toml | crates/mantis-search/tests/temperature_parity_golden.rs (Rust only) | gen_temperature_parity_golden.py → 04 |
| search_golden/temperature/gen_temperature_parity_golden.py | 54 / 2 249 | none | **no runner** → 04 | — |
| selfplay/collate/*.npz ×8 + collate_expectations.json | bin 320 784 + 1 507 | toml | tests/selfplay/conftest.py (_payload_bank, collated_golden, hotpath_golden, collate_expectations) → 11 selfplay test files + tests/model/test_pmask_gather_parity.py [torch] | old-side capture |
| selfplay/drain/graph_pushed.npz, drain_goldens.json | bin 1 453 + 4 855 | toml | conftest → test_pool_drain_parity.py [torch]; drain_goldens also test_drain_row_shape_parity.py (6 passed), test_selfplay_census.py (22 passed), test_game_complete_absence.py, test_pool_drain_arms.py | old-side capture |
| selfplay/drain/collect_data_input.npz, dense_pushed.npz | bin 49 644+36 914 | toml | conftest → test_pool_drain_parity.py, test_pool_drain_arms.py (the DENSE arm) → 06 | old-side capture |
| selfplay/instrumentation/pure_function_battery.json | 3 593 | toml | test_instrumentation.py [torch] | old-side capture |
| selfplay/pool/encoding_resolve.json, runner_config_goldens.json | 80+219 | toml | test_pool_encoding_resolve.py, test_pool_hparams.py [torch] | old-side capture |
| silent_encoding_evasions.toml | 286 | none | tests/tools/test_silent_encoding_gate.py (74 passed here) | hand-built corpus |
| train/legacy_payload_shapes.json, resume_goldens.json | 108+132 | toml | tests/train/conftest.py::{legacy_shapes, resume_goldens} → test_checkpoint_conformance.py, test_resume_semantics.py [torch]; resume_goldens is cited in docs/contracts/checkpoint_envelope.md (gate 10) | — |
| value_probes/dist65_golden.json | 2 089 | toml | tests/model/test_dist65.py [torch] | old-side capture |
| value_probes/forward/small_gnn.pt | bin 28 103 | toml | tests/model/test_forward_parity.py [torch] | new-side |
| value_probes/statedict_keys/gnn_axis_v1.txt | 50 | toml | tests/model/test_build_net.py::_golden_keys; tests/train/conftest.py::anchor_key_set [torch] | — |
| value_probes/{decoded_v.npz, metrics.json, probe_set_v1.jsonl, negatives_v1.jsonl} | 912 + bin 15 170 | toml | **NONE** → 01 | old-side capture |
| value_probes/CHANGELOG.md | 36 | toml | **NONE** → 02 | — |
| value_targets/lambda_return_golden_v1.json | 56 | toml | tests/model/conformance/test_value_target_codecs_land_unarmed.py [torch] | — |
| worker/*.bin ×3 | bin 474 207 | toml | worker_output_pin.rs (2 files), queue_fuse_pin.rs (1) | frozen hexo_rl capture |

## Findings

### S-A-TESTS-9-01 | ONE-SHOT | C
subject: tests/fixtures/value_probes/{decoded_v.npz, metrics.json, probe_set_v1.jsonl, negatives_v1.jsonl} + their 4 manifest.toml rows + the 6-line "CORRECTED IN PLACE (R311(c))" note
claim: The O2 value-health bank has no reader at HEAD. Its only consumer, tests/model/test_value_health.py, was deleted with the dense arm in 3dd20b4. What remains is the sha check and the CRLF census. The manifest KEEPS the rows by recorded choice ("a decision to revisit"), so deleting them is a ruling call.
evidence: `git grep -l -F <basename>` for each of the 4 names, excluding tests/fixtures and docs/slim → empty. `git grep -n value_probes` → only test_build_net.py (statedict_keys), test_dist65.py, test_forward_parity.py, train/conftest.py (statedict_keys). `git log --all --oneline -- tests/model/test_value_health.py` → 3dd20b4 "wip: grid path deletion checkpoint", 313f189.
callers:
- AST/string: `git grep -l -F` on each basename across the whole tree → empty. `git grep -n -E "probe_set|negatives_v1|decoded_v|value_probes/metrics"` → empty.
- glob/dir readers: `git grep -n -E "glob\(|rglob\(|iterdir\(|listdir|os\.walk" -- tests src tools` → no fixture-root glob.
- entry points / `python -m` / subprocess: none name a fixture (CALLERS §1, §4, §6; `git grep fixtures -- Makefile '*.sh' .github` → only a lint_gate.sh comment).
- importlib/getattr: n/a for data files. CALLERS §7 has no fixture loader.
- conftest: tests/{selfplay,train}/conftest.py read only other fixtures (both read in full).
- pyo3 / config keys: `git grep fixtures -- configs src/mantis/config tools/config_templates` → empty.
- gate tool paths: artifact_gate.py handles FIXTURES_PREFIX generically; silent_encoding_gate skips fixtures; manifest/line-ending tests read rows generically.
- STATE procedures: `git grep -n -E "value_probes|probe_set|negatives_v1|decoded_v" -- docs/governance/*.md` → empty.
- Rust: `git grep -n value_probes -- crates` → empty.
Δlines: -938. That is -912 fixture text lines (metrics.json 27 + probe_set_v1.jsonl 234 + negatives_v1.jsonl 651, from numstat), -20 for 4 rows × 5 lines, and -6 for the note. Also decoded_v.npz binary 15 170 B. Bytes removed in total: 2 887 686 (`wc -c`). The WP9 intro comment then needs an in-place trim (see 05(c)).
witness: tests/test_fixtures_manifest.py::test_every_required_fixture_present_with_matching_sha (reds if a file goes but its row stays); tests/test_line_endings.py floor ≥ 40 (46 → 42, still green)
depends: 02 (same package), 05

### S-A-TESTS-9-02 | DOC | C
subject: tests/fixtures/value_probes/CHANGELOG.md + its manifest.toml row
claim: This is a CHANGELOG inside fixtures with no reader.
- It lists `forward/small_{cnn_scalar,cnn_dist65,cnn_aux_chain,gnn}.pt`, but only small_gnn.pt exists.
- It describes the O2/O2a gates, whose test is deleted.
- Its "re-bump discipline" is enforced by nothing but the manifest's sha rows, which already carry provenance comments.
evidence: `git ls-files tests/fixtures/value_probes/forward` → small_gnn.pt only. `git grep -l -F CHANGELOG.md -- ':!tests/fixtures' ':!docs/slim'` → empty.
callers: same channels and commands as 01, with the basename `CHANGELOG.md` / `value_probes/CHANGELOG` → no reader. No docs cite the path. Gate 10 scans only Makefile/README/CLAUDE/docs/contracts/governance, and none of them has the string.
Δlines: -41 (36 by `wc -l`, plus a 5-line manifest row)
witness: tests/test_fixtures_manifest.py (the row must go in the same commit)
depends: 01 (if 01 is ruled keep, 02 can still go on its own; 01+02 → line-endings floor 41)

### S-A-TESTS-9-03 | ONE-SHOT | C
subject: tests/fixtures/graph_parity/wpa_positions.json (+ its F-row and the `corpus_file`/`corpus_sha256` header lines in graph_parity/manifest.tsv)
claim: The 124 769-line corpus is never parsed. inputs.bin already carries the positions, and the capture tool lives off-tree ("regenerable from the migration workspace"). Its only reader is the generic F-row sha/size loop in graph_parity.rs::manifest_and_fixture_files_valid.
- Deleting it re-edits manifest.tsv, which is sha-pinned in manifest.toml. That is a re-pin under grant.
- R334(d) names this fixture set ("Option (iii) rides lane C's golden recapture").
- This is the largest single text subject in the repo.
evidence:
- `git grep -n -i "wpa_positions\|corpus_file" -- ':!<itself>'` → .gitattributes comment, archive rulings, manifest.tsv lines 7/13.
- `git grep -n "verify_file_row\|m.files"` → graph_parity.rs (the loop over every F-row) and fixture_selftest.rs, which picks a raw blob, not the corpus.
- `grep -n corpus crates/mantis-graph/{tests,benches,src}` → only a build_bench.rs comment.
callers:
- Rust runtime paths: `git grep -n -E "value_probes|wpa_positions|corpus_file" -- crates` → empty. mantis-graph tests read only inputs.bin, manifest.tsv and `raw/case_{:05}.bin`.
- include_str!: none. Benches read inputs.bin only (build_bench.rs::load_positions).
- Python: `git grep -l wpa_positions -- '*.py'` → empty, which matches R334(d)'s own measured negative.
- Makefile/sh/STATE: empty (same commands as 01).
Δlines: -124 772. That is -124 769 by numstat (966 939 B by `wc -c`), plus -3 manifest.tsv lines (the F-row and 2 header lines). The graph_parity/manifest.tsv sha in manifest.toml is re-pinned at net 0. The .gitattributes line-5 comment needs a reword (D4).
witness: crates/mantis-graph/tests/graph_parity.rs::manifest_and_fixture_files_valid (reds if the file goes while the F-row stays); tests/test_fixtures_manifest.py (reds on the manifest.tsv re-pin until the row is updated)
depends: —

### S-A-TESTS-9-04 | ONE-SHOT | C
subject: tests/fixtures/search_golden/temperature/gen_temperature_parity_golden.py
claim: This generator lives inside fixtures and nothing runs it: no Makefile target, test, tool, CI step or STATE procedure. Its docstring's "the Python eval/bot path checks the same fixture" is false.
- The only reader of the CSV is the Rust test.
- The sha-pinned CSV's first line names this file ("see gen_temperature_parity_golden.py"). Deleting the generator leaves a dangling pointer inside a golden, and fixing that pointer means a re-pin.
- It is also a loose script (`#!`, `__main__`), which CLAUDE.md's "no loose script files" rule does not otherwise allow outside tools/. Its sibling generator is tools/gen_mctx_parity_fixtures.py.
evidence: `git grep -n -i "temperature_parity\|gen_temperature" -- ':!docs/slim'` → only crates/mantis-search/{src/temperature.rs, tests/temperature_parity_golden.rs}, the manifest row, the CSV header, and itself. `git log --all --oneline -S temperature_parity_golden -- 'tests/**/*.py'` → only the commit that added this generator, so no Python twin ever existed in tracked history.
callers:
- AST import: `git grep -n -w gen_temperature_parity_golden -- '*.py'` → itself.
- `python -m` / script-path: none (CALLERS §4 table; `git grep "gen_temperature" Makefile '*.sh' .github docs/governance` → empty).
- subprocess / importlib `spec_from_file_location`: `git grep -n -E "gen_temperature|spec_from_file_location\(.*fixtures"` → only the Rust doc comment.
- conftest/pytest: not collected (`pytest --collect-only tests/fixtures` → 0 tests).
- gate tools: comment_lint (gate 14) scans tests/ .py, so deleting it only lowers the docstring measures. Gate 15: 54 lines, no header.
Δlines: -54 (`wc -l`). Alternative lane B: keep the file and delete the one false sentence, net about 0.
witness: none reds on deletion. crates/mantis-search/tests/temperature_parity_golden.rs::rust_matches_temperature_parity_golden holds the CSV either way.
depends: —

### S-A-TESTS-9-05 | DOC | B
subject: tests/fixtures/manifest.toml (comments only; no row, no sha moves)
claim: Four comment claims disagree with the tree. The precedent for repairing them in place is the manifest's own "CORRECTED IN PLACE (R311(c))" note, and the derive-or-delete rule (R192(e)) covers the transcribed byte counts.
- (a) The WPMAIN block (5 lines) describes minted-config rows that are gone and a test, tests/config/test_minted_config_remint.py, that is deleted. The baseline dir it describes is now asserted ABSENT by tests/config/test_mint_header_roundtrip.py::test_the_frozen_wpmain_baseline_set_is_gone_and_stays_gone.
- (b) EVALDECODE says "9 494 B + 8 675 B = 18 169 B, inside the 64 KB budget". The files are 30 254 + 26 642 = 56 896 B, and the in-test budget is 131 072 (tests/eval/test_eval_selfplay_child_parity.py::_FIXTURE_BYTE_BUDGET).
- (c) WP9 says "probe_set_v1.jsonl (2.39 MB) is the sole gate-6 large-file exception (operator allowlist)". artifact_gate.py has no allowlist, and board/board_replay_golden_v1.json (3 021 473 B) and graph_parity/manifest.tsv (2 286 437 B) are also over 1 MB under the plain fixtures carve-out.
- (d) WP9 lists "constructed + promoted-anchor KEY SETS (O3/O3b)" and "forward drift goldens". Only statedict_keys/gnn_axis_v1.txt and forward/small_gnn.pt exist.
evidence: `git ls-files tests/config | grep -i remint` → only test_eval_config_remint.py. `wc -c` on the eval_selfplay_parity files. `grep -n _FIXTURE_BYTE_BUDGET tests/eval/test_eval_selfplay_child_parity.py` → 131072. `grep -n -i allow tools/ci_gates/artifact_gate.py` → none. `git ls-files tests/fixtures/value_probes` → 9 files.
Δlines: -5 for (a). (b)-(d) are in-place rewrites (net ≤ 0; drop the transcribed sizes rather than re-transcribe them). If 01 lands, (c) goes with it.
witness: none (comments). tests/test_fixtures_manifest.py stays green because no row changes.
depends: 01

### S-A-TESTS-9-06 | TEST | C
subject: tests/fixtures/selfplay/drain/{dense_pushed.npz, collect_data_input.npz} + the 4 dense variants of drain_goldens.json (dense_5s_crossed, dense_5s_not_crossed, dense_no_recent_buffer, dense_zero_rows)
claim: These are the fixtures of the dense drain arm, which has been dead in production since R346(f). They stay live only as the drain-parity suite's instrumentation oracle (CARDS B-15: "deleting it means re-basing six oracles on the graph variant and re-pinning the fixture"), so they go only with that ruling-named re-base.
evidence: src/mantis/selfplay/pool_drain.py takes `push_dense` only when `not pool._is_graph`. CARDS.md B-15 text is quoted above. The drain_goldens `variants` keys → 4 of 5 are dense. `git grep -l -w dense_pushed -- tests` → test_pool_drain_parity.py. `git grep -l -w collect_data_input` → test_pool_drain_parity.py, test_pool_drain_arms.py.
Δlines: ≥ -3 686 (drain_goldens.json redumped with the dense variants removed: 4 855 → 1 169). Also -86 558 B binary (`wc -c`: 36 914 + 49 644), and -10 for 2 manifest rows. This is offset by whatever graph-variant re-captures the re-base adds, so the net cannot be derived before B-15 is ruled.
witness: tests/selfplay/test_pool_drain_parity.py (dense_* tests), test_pool_drain_arms.py, tests/test_fixtures_manifest.py (re-pin of drain_goldens.json), tests/test_line_endings.py floor (with 01+02 → 39 < 40, RED: the floor moves)
depends: T5/C2 owner of `push_dense` (HANDOFF)

## DEFECTS
- none in tests/fixtures bytes. All 45 digests match, and every Rust/Python reader resolves its path. The stale claims are listed as findings 04 and 05.

## PARKED
- none

## HANDOFF
- D4 (.gitattributes): three problems.
  - "pinned by sha256 in tests/fixtures/manifest.toml (65 rows)" is wrong: `grep -c '^\[\[required\]\]'` → 45.
  - It cites `crates/mantis-search/src/mcts/golden_tests.rs:267` by line number.
  - The `crates/mantis-encoding/src/manifests.toml -text` rule points at a file that does not exist.
- R1 (crates/mantis-search/tests/temperature_parity_golden.rs header): "Pins that the Rust training-path temperature reproduces the SAME golden table that the Python eval/bot path checks … (or its Python twin) fires" is wrong, because no Python reader of the CSV exists.
- C2/T5 (src/mantis/selfplay/pool_push.py::push_dense, tests/selfplay/test_pool_drain_{parity,arms}.py): the B-15 re-base that 06 depends on.
- T7 (tests/test_line_endings.py): the `>= 40` row-census floor has to move if 01+02+06 all land.
- L2 (tools/gen_mctx_parity_fixtures.py): a manual one-shot generator that needs a jax venv, with 0 runners. It is the same shape as 04, and there it sits in tools/ as the recipe of record.

## Not covered
- The contents of the large data files (wpa_positions.json, manifest.tsv C-rows, mctx_parity_v1.json, board golden, npz/bin blobs): I did not read them, per the slice brief. Only headers and top-level keys were read.
- Whether the off-tree captures (the migration workspace, the old-side `wp/` banks) still exist: that is outside the repo.
- Rust readers were verified statically, and no cargo test was run (to save CPU). Torch-dependent Python readers were verified by static reference plus collection. They error here on `No module named 'torch'`, which is expected in this environment.
