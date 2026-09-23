# S-A-TOOLS-1 — L1: tools/ci_gates, tools/config_templates, Makefile, .github
scope: tools/ci_gates/** (21 files), tools/config_templates/dev.yaml, Makefile, .github/workflows/ci.yml (24 files, 6 043 lines by `git ls-files … | xargs wc -l`); method: full read of every file except the preflight pair (see Not covered), an AST top-level-symbol census of every gate `.py` (in-file loads plus `git grep -w` over tests/tools/src/docs/Makefile/.github), `npx --yes jscpd@4 --min-lines 5 --min-tokens 40` (0 verbatim clones), `uvx vulture tools/ci_gates --min-confidence 60` (2 hits, both false positives), torch-free live runs of gates 10/11/15/16, the tier census and `comment_lint.py --measure`, and file:line cites checked against HEAD.

## Summary
- 16 findings. By class: DOC 7, SIMPLIFY 4, CONFIG 2, DUP 2, PACK 1. By lane: A 1, B 1, C 14. Nearly everything is lane C because PZ's glob list protects `tools/ci_gates/**` and `tools/config_templates/**` whole.
- Top 3 by Δlines: S-A-TOOLS-1-07 (up to −367: narrative comments in the non-`.py` gate files, which gate 14 does not measure), S-A-TOOLS-1-15 (−258: ci.yml plus pytest_step_summary.py, the only tool that ci.yml alone calls; remote CI is suspended), S-A-TOOLS-1-10 (−40: gate 11's KNOWN_DEBT machinery, whose register a test pins empty).
- No DEAD gate script. Every `tools/ci_gates` file has an executed caller: run_all.sh, lint_gate.sh, test_count_gate.sh, preflight_mint.py (which loads preflight_mint_parent.py by path), or ci.yml (pytest_step_summary.py only). tests/test_meta_ci.py::test_every_ci_gate_script_is_invoked_by_ci_yaml enforces this.
- PACK verdicts: tools/ci_gates KEEP. tools/config_templates KEEP: it holds one file, but tools/mint_config.py, tools/config_diff.py and 6 test files resolve `config_templates/<t>.yaml`, and moving it saves 0 lines.
- Floors: `comment_lint.py --measure` equals comment_length_floor.txt on all 5 gated measures (3385/0/13079/1485/2676). Any accepted `.py` comment or docstring cut in this slice (03, 09–12) lowers that floor in the same commit. tier census: "79 deselected test(s), all declared" (99 lines, 79 rows).

## Findings

### S-A-TOOLS-1-01 | DOC | A
subject: Makefile (comment above `lint.rust`)
claim: The comment says "the seven non-smoke `[[bench]]` targets" and "the 28 floors in tools/bench_floors.toml". HEAD has 6 bench targets (5 non-smoke) and 23 floor rows. R346(f) deleted encode_bench, replay_sample_bench and their 5 floors, and this comment was not updated.
evidence: `git grep -n -A3 '^\[\[bench\]\]' -- 'crates/*/Cargo.toml' | grep name` -> smoke_bench, board_bench, build_bench, mcts_bench, graph_build_bench, queue_fuse_bench (6); `grep -c '^\[floor\.' tools/bench_floors.toml` -> 23; the bench_floors.toml header states "R346(f) DELETED five rows with the benches that produced them".
Δlines: 0 (reword in place; R192(e) derive-or-delete means drop both numbers rather than re-transcribe them)
witness: NONE for the comment. tests/test_meta_ci.py and tests/tools/test_local_gate_runner.py read Makefile recipes, not comments.
depends: —

### S-A-TOOLS-1-02 | DOC | C
subject: tools/ci_gates/run_all.sh (header and the gate-2b comment)
claim: The same stale count appears at three sites: "compiles exactly one of the eight", "seven of the eight bench targets" and "the seven non-smoke bench targets". The real figures are 6 and 5.
evidence: `grep -n -E "eight|seven" tools/ci_gates/run_all.sh` -> 3 comment lines; the bench count is the same as in 01.
Δlines: 0 (reword); see 07 for trimming the surrounding narrative
witness: NONE. tests/tools/test_local_gate_runner.py derives the bench count (`len(benches) >= 6`) and never reads the comment.
depends: —

### S-A-TOOLS-1-03 | DOC | C
subject: transcribed line cites and counts across the gate scripts (batched)
claim: These are line numbers and tallies written into comments and messages, and most no longer match the tree:
- tools/ci_gates/gate_01_fresh_sync.sh cites `test_surface.py:29,73`, `encoding.rs:213`, `_engine.pyi:650` and `registry_gate.sh:34`. All four are wrong at HEAD (31/75, 300, 563, 36), and the `_engine.pyi` twin at `src/mantis/_engine.pyi` goes unnamed.
- tools/ci_gates/preflight_mint_parent.py cites `monitor/supervise.py:39`; `RELAUNCH_BUDGET_EXIT_CODE` is at 44. It also cites `actor_sync.py:63`; `maybe_sync` starts at 58.
- tools/ci_gates/lint_gate.sh cites `preflight_mint.py:952`, a historical incident that now points at `main()`.
- tools/ci_gates/encoding_io_gate.py::EXEMPT grounds cite "tests/tools/conftest.py:3,15 -- 'editing it is an R43 event'". Commit 2649e0b removed that text, so the byte-frozen claim for tests/tools/test_preflight_mint.py no longer has an in-tree source.
- encoding_io_gate.py prints "this repo's 639 tracked text files" on failure. The tree has more than 1 070 tracked files.
- encoding_io_gate.py's MIN_FILES comment gives "(tools/ 13, tests/ 254)"; the live counts are 78/483.
- r8_header_gate.py gives "(src 148, tools 14, crates 134, tests 256)" and "137 over the cap, 151 justifications"; the live counts are 187/78/136/483 and 172/172.
- tools/ci_gates/preflight_mint.py has a truncated comment ("F-B1 closure: copy … into the child block and") and cites "25001 on run5/run6"; configs/run5.yaml is not in the tree.
evidence: `git grep -n -o -E "[A-Za-z0-9_./-]+\.(py|rs|md|toml|txt|pyi|sh|yml|yaml):[0-9]+…" -- tools/ci_gates …` -> 9 cites; per-cite `grep -n registry_sha_hex …`, `sed -n 36,42p src/mantis/monitor/supervise.py`; `git show 2649e0b -- tests/tools/conftest.py | grep R43` -> `-is an R43 event`; live counts from `r8.scan()` / `eio.scan()` loaded by spec_from_file_location.
Δlines: ≈0 (edit in place). Deleting the cite block in gate_01 (`sed -n 9,19p` = 11 lines) instead would be up to −11.
witness: NONE for the text. Gates 15 and 16 do not read their own comments.
depends: —

### S-A-TOOLS-1-04 | DOC | C
subject: tools/ci_gates/artifact_gate.py (module docstring)
claim: The docstring contradicts the code in three places:
- It says the base "defaults to $ARTIFACT_GATE_BASE, else HEAD~1" and that "an all-zeros/invalid base falls back to HEAD~1". The code tries `_WIDE_FALLBACKS = ("origin/dev", "dev", "HEAD~1")` and widens an empty range to the full tree.
- Rules (2) and (4) say "ADDED". The code sizes M files and the new side of R/C renames as well (B-6).
- It attributes the raised-ceiling rule to "(R8)". That rule is CLAUDE.md R7.
evidence: `sed -n 1,13p` against `_WIDE_FALLBACKS` and `changed.append((status[0], fields[i + 1], status[0] in ("A", "M")))`.
Δlines: 0
witness: NONE (docstring). tests/tools/test_artifact_gate.py tests behaviour.
depends: —

### S-A-TOOLS-1-05 | DOC | C
subject: comments that cite documents or quotes absent from the tree (batched)
claim: These comments point at text that does not exist in this repo:
- test_count_gate.sh quotes CLAUDE.md as saying "Main branch (you will usually use this for PRs): dev". CLAUDE.md has no such line. The same header lists branches `remediation` and `wppre-scratch`, which origin does not have.
- registry_gate.sh cites `migration_plan.md` and "(ported in WP3)".
- lint_gate.sh cites `CENSUS_LT §7/§5b` and `IMPL_NOTES_LT_PYRIGHT`.
- tools/config_templates/dev.yaml cites `DESIGN_P2.md` and `DEBT_DOSSIER`.
- ci.yml cites `CENSUS_LT` (its ci.yml half is lane B, see 14).
These are migration-workspace documents that were never tracked here, and gate 10 does not scan tools/.
evidence: `grep -n "Main branch" CLAUDE.md` -> none; `git branch -a` -> dev, perf-a4, perf-baseline-20260829, recal-mint-20260828, remint-*, run6-mint, wave3; `git ls-files | grep -c -i <name>` -> 0 for IMPL_NOTES_LT_PYRIGHT, CENSUS_LT, DESIGN_P2, DEBT_DOSSIER, migration_plan.
Δlines: ≈0 to small negative (delete the pointer, keep the fact)
witness: NONE
depends: 07

### S-A-TOOLS-1-06 | DOC | C
subject: tools/config_templates/dev.yaml (comments)
claim:
- An orphaned 3-line comment ("A fraction of each batch drawn from the bot corpus. Its sibling `bot_corpus_path` …") sits directly above `value_target`. No bot or corpus key exists among RunConfig's 160 leaves, so the key it described is gone.
- The fused-caps derivation names the retired `v6w25` as "the registry's WIDEST legal_move_radius (8)". Radius 8 is now `gnn_axis_r8`.
- The comments say "`null` … the two production configs mint it". All 4 production configs mint calibrated caps.
- "run5 overrides both members" refers to run5, and the header says "Carries every WP8+WP11-A field". Both are stale.
Minted configs do not copy template comments: configs/run10.yaml has 46 comment lines and none of them is template prose.
evidence: `.venv/bin/python -c "…leaf_paths(RunConfig)…"` -> 160 leaves, `[]` matching bot/corpus; `grep -n legal_move_radius crates/mantis-encoding/src/registry.toml` -> v1 6, r8 8; `grep -n -A2 fused_graph_caps configs/*.yaml` -> run6/run10 at 1373143/56645.
Δlines: −3 for the orphan (`sed -n 152,154p`); the rest are rewords
witness: NONE. tests/config/test_mint_and_diff.py and the others yaml-load the template, and comments do not survive yaml.safe_load.
depends: —

### S-A-TOOLS-1-07 | DOC | C
subject: narrative comment blocks in the non-`.py` slice files, which gate 14 does not measure
claim: comment_lint.py measures only `.py` and `.rs` (`SCOPES` × `in_scope`). The bash gates, ci.yml, the template, the Makefile and the floor/declaration text files carry 532 own-line comment lines, 367 of them beyond two per run (comment_lint's own measure). Much of this is history ("WHY THIS FILE WAS REWRITTEN", "THE WITNESS, measured 2026-09-03", "THE SECOND DEFECT"), which R316(e) bars and no ratchet sees.
evidence: The comment_lint `_excess` rule applied per file (excess-over-2 / comment lines): test_count_gate.sh 93/131, dev.yaml 82/121, lint_gate.sh 59/79, run_all.sh 44/56, ci.yml 34/58, tier_declaration.txt 18/20, gate_01 16/18, registry_gate.sh 12/19, Makefile 6/23, comment_length_floor.txt 3/7.
Δlines: up to −367 (an upper bound; the kept one-line facts reduce it). Because these file types are outside the ratchet, no floor file moves.
witness: NONE (outside gate 14's scope)
depends: — (lane C: a tree pass is what R316(e)/R346(f) reserve to a ruling. On contact is allowed, and 02/03/05/06 are contacts. The ci.yml share, 34, is operator-owned.)

### S-A-TOOLS-1-08 | CONFIG | C
subject: tools/ci_gates/check_tracked_refs.py::GENERATED_WHITELIST entries `"target/"` and `"dist/"`
claim: Neither entry can ever match. `TOKEN_RE` only produces tokens rooted at `src|tests|tools|configs|crates|docs|vendor`, and `token.startswith("target/")` or `("dist/")` is never true for such a token. Only `vendor/external` is reachable, with 3 tokens in scope.
evidence: A module loaded by spec_from_file_location run against every scope token -> `target/ UNREACHABLE`, `dist/ UNREACHABLE`, `vendor/external tokens in scope: 3`; `git grep -n GENERATED_WHITELIST -- tests tools` -> only the gate itself.
Δlines: 0 (tuple edit plus a docstring phrase)
witness: gate 10 (`python3 tools/ci_gates/check_tracked_refs.py`, rc 0 at HEAD); tests/tools/test_gate_vacuity.py
depends: —

### S-A-TOOLS-1-09 | SIMPLIFY | C
subject: tools/ci_gates/silent_encoding_gate.py::find_violations
claim: This is a test-only pure forwarder (`return scan()[0]`). Nothing calls it except tests/tools/test_silent_encoding_gate.py. That file's `monkeypatch.setattr(GATE, "find_violations", …)` in test_main_returns_nonzero_when_an_arm_is_present has no effect, because `main()` calls `scan()`.
evidence: `.venv/bin/python scratchpad/s1_ast.py tools/ci_gates/*.py` -> `find_violations in-file-loads=0 ext=['tests/tools/test_silent_encoding_gate.py']`; `grep -n find_violations tests/tools/test_silent_encoding_gate.py` -> lines 45 and 137.
Δlines: −5 (def span 278–280 plus 2 blank lines); the test edit (`GATE.scan()[0] == []`, drop 1 inert line) is T3's
witness: tests/tools/test_silent_encoding_gate.py::test_gate_is_green_on_the_current_tree
depends: —

### S-A-TOOLS-1-10 | SIMPLIFY | C
subject: tools/ci_gates/silent_encoding_gate.py::KNOWN_DEBT and its match, stale and report paths
claim: test_known_debt_register_is_empty asserts `GATE.KNOWN_DEBT == ()`, so a green tree can never reach the register-handling code in scan() and main(). The code is exercised only by two monkeypatched producer tests. The docstring retains it deliberately "for the next owned arm". That is the design choice to put to the architect.
evidence: `grep -n -E "KNOWN_DEBT|debt"` -> ranges 21–23, 112–115, 247–248, 260–271, 294–306, 316–317; `grep -n "KNOWN_DEBT ==" tests/tools/test_silent_encoding_gate.py`.
Δlines: −40 (36 lines of machinery by `sed -n <range>p | wc -l`; 327−36 = 291 ≤ 300, so the 4-line R8 header must also go)
witness: tests/tools/test_silent_encoding_gate.py (3 tests go: known_debt_register_is_empty, the stale-entry test, a_matching_debt_entry…). Test-floor move −3, lane B-shaped inside C.
depends: 09

### S-A-TOOLS-1-11 | SIMPLIFY | C
subject: tools/ci_gates/rule7_gate.py::EXEMPT and its blob-sha matching and stale paths
claim: The register ships empty (`()`), and as written it cannot work. tests/tools/test_rule7_gate.py requires a 64-hex sha256 per entry, while the gate compares against `git hash-object`, which is 40-hex SHA-1 in this sha1 repo. Any registered exemption would never match and would always red as stale (see DEFECTS). The choice is to delete the machinery or fix the hash; CLAUDE.md's gate-17 text describes the mechanism, hence lane C.
evidence: `git rev-parse --show-object-format` -> sha1; `git hash-object Makefile | awk '{print length}'` -> 40; test asserts `len(sha) == 64`; ranges 108–114, 480, 491–503, 514–521, 531–532, 545.
Δlines: −31 (32 lines by sed ranges, minus the 1 line `violations.extend(file_hits)` that replaces them)
witness: tests/tools/test_rule7_gate.py::test_exempt_register_ships_empty (−1 test if deleted); gate 17 self-test
depends: —

### S-A-TOOLS-1-12 | DUP | C
subject: tools/ci_gates/comment_lint.py::_rust_doc_excess vs ::_excess
claim: `_rust_doc_excess` repeats `_excess`'s run-length loop verbatim except for the cap (1 instead of `CAP`). The fix is a `cap` parameter on `_excess`.
evidence: AST spans `_excess` 153–161 and `_rust_doc_excess` 231–245; the trailing loop is 238–245 (8 lines).
deliberate?: No. It is one file, one measure family and one decision ("lines beyond N in an own-line run"). No seam, oracle or twin role.
Δlines: −7
witness: comment_lint self_test arm "rust doc run: want 2"; tests/tools/test_comment_lint.py
depends: —

### S-A-TOOLS-1-13 | DUP | C
subject: cross-gate helper near-duplicates (rule7_gate, encoding_io_gate, silent_encoding_gate, artifact_gate, comment_lint, test_count_gate.sh)
claim: These helpers repeat across gates:
- the escape-hatch walker: `_justified` ×2 (10+10 lines) and `_is_justified`+`_comment_res` (17)
- the "entries matched nothing" stale-register block ×3
- base-ref resolvers ×3 in Python plus 1 in bash
- the `--name-status -z` parser ×2
- `_git` ×2
- the "A gate that scans nothing finds nothing" floor message ×4
VERDICT KEEP. jscpd finds 0 verbatim clones. The comment-lead sets differ by language (rule7 `#/;-`, gate 16 `#`, gate 11 per-suffix with the `#[` exclusion), and the fallback directions differ on purpose (gate 6 widens to origin/dev, gate 17 to the full tree). Because tests load gates by spec_from_file_location and R5 forbids sys.path writes, a shared module would need a by-path loader in every gate, like preflight_mint's `_load_parent_half`. That costs about what it saves.
evidence: `npx --yes jscpd@4 --min-lines 5 --min-tokens 40 … tools/ci_gates …` -> "Found 0 clones"; AST spans from `scratchpad` span script.
deliberate?: The semantics diverge by scanned language and by leak-vs-artifact posture.
Δlines: ≈0 net (not recommended)
witness: each gate's producer test in tests/tools/
depends: —

### S-A-TOOLS-1-14 | CONFIG | B
subject: .github/workflows/ci.yml `on.push.branches` entry `'ci-repair/**'` and its 3-line comment
claim: The comment says the entry exists for repair branches and is "Removed when unused". Origin has no `ci-repair/*` branch. The gate-8 step name "(auto-arming; WP3 debt)" is also stale: the handshake has been ARMED since WP7, per registry_gate.sh. The workflow is operator-owned (R348(a)), so this goes to the ARCHITECT.
evidence: `git ls-remote --heads origin 'ci-repair/*'` -> nothing (8 heads total); `git grep -n ci-repair` -> ci.yml only; `git grep -n "WP3 debt"` -> ci.yml only.
Δlines: −3
witness: tests/test_meta_ci.py::test_ci_yaml_pins_tiers_and_gate_scripts (does not read triggers)
depends: —

### S-A-TOOLS-1-15 | PACK | C
subject: .github/workflows/ci.yml (141) + tools/ci_gates/pytest_step_summary.py (117)
claim: Remote CI is suspended by operator decision (R348(a)), so nothing runs either file. pytest_step_summary.py has exactly one caller, ci.yml, and no test. ci.yml survives as the parsed gate roster:
- tests/tools/test_local_gate_runner.py requires run_all.sh's `gate N:` set to equal ci.yml's.
- tests/test_meta_ci.py pins ci.yml's tier and gate commands.
- tests/tools/test_test_count_gate.py::test_no_ci_step_injects_a_count reads it.
The roster therefore lives in two files kept in step by a test. ARCHITECT question:
- (a) keep both as they are, or
- (b) make run_all.sh the one roster, delete both files and repoint the 3 test files.
If CI is ever re-enabled, it could call `run_all.sh --only "gate N"` per step.
evidence (DEAD-while-suspended checklist):
- AST imports: `git grep -n -F pytest_step_summary -- . ':!docs/slim'` -> ci.yml:74,79 only.
- entry points: none (CALLERS §1).
- `python -m`: none.
- subprocess strings: the same basename grep over tests/src/tools -> none.
- importlib/getattr/registry: `git grep -n -E "[\"']pytest_step_summary[\"']"` -> none.
- conftest: none.
- pyo3: n/a.
- config keys: n/a.
- gate tool paths: run_all.sh and Makefile -> none.
- STATE procedures: `grep -c "pytest_step_summary\|ci.yml" docs/governance/STATE.md` -> 0.
- ci.yml readers: `git grep -l -F ci.yml -- tests tools src` -> test_meta_ci.py, test_local_gate_runner.py, test_test_count_gate.py (+ two gate comments).
Δlines: −258 (`wc -l` of both). The test edits are T3/T7's: roughly 3 test functions in test_meta_ci plus the roster tests re-pointed, so a test-floor move.
witness: tests/test_meta_ci.py, tests/tools/test_local_gate_runner.py
depends: 14, 16

### S-A-TOOLS-1-16 | SIMPLIFY | C
subject: tools/ci_gates/test_count_gate.sh::resolve_ref arm 3 (the shallow `git fetch origin dev`) and its header notes
claim: By its own header, arm 3 is "a repair, not a design" for ci.yml's python-job checkout, which lacks `fetch-depth: 0`. Local runs resolve origin/dev or dev first. If 15(b) retires ci.yml, or if the workflow gains `fetch-depth: 0`, the arm and its notes are dead weight.
evidence: `sed -n 275,281p` (7), header items 30–35 (6) and 42–45 (4); ci.yml's python job `actions/checkout@v4` has no `fetch-depth` (only hygiene has `fetch-depth: 0`).
Δlines: −17
witness: tests/tools/test_test_count_gate.py::test_a_shallow_checkout_recovers_the_ref_by_fetching (−1 test, test-floor move)
depends: 15

## DEFECTS
- tools/ci_gates/rule7_gate.py::main compares EXEMPT's recorded sha against `git hash-object` (40-hex SHA-1 here), while tests/tools/test_rule7_gate.py demands a 64-hex sha256. Any gate-17 exemption can therefore never match and always reds as "matched nothing".
- tools/ci_gates/tier_census.py::_empty_declaration creates `NamedTemporaryFile(delete=False)` and never unlinks it. Every `--self-test` run leaks one temp file.
- Makefile `dashboard`/`viewer`/`analyzer` hardcode `uv run` where every other recipe uses `$(UV)`, so `make UV=…` does not reach them.
- tools/ci_gates/test_count_floor.txt holds 4862 against 5112 collected (STATE's R367 exit line and CALLERS §0; not re-measured here because torch is absent). Up to 250 tests could vanish while gate 3c stays green until someone ratchets the floor.

## PARKED
none

## HANDOFF
- D4 (root files): CLAUDE.md says "all 28 bench floors". tools/bench_floors.toml has 23 `[floor.*]` rows at HEAD.
- T3 (tests/tools): the test_local_gate_runner.py assertion message still says "the seven non-smoke bench targets" (5 at HEAD). test_silent_encoding_gate.py has an inert `monkeypatch.setattr(GATE, "find_violations", …)` (see 09). test_rule7_gate.py::test_exempt_register_ships_empty never asserts emptiness and pins the sha format that makes the gate unusable (see DEFECTS).
- L2 (tools/bench_floors.toml): the file is correct at 23. Only other files' counts are stale.

## Not covered
- tools/ci_gates/preflight_mint.py and preflight_mint_parent.py were read in part: the preflight_mint.py docstring, the re-export block, the audit/preflight/main paths, and the parent's first 200 lines. The rest was checked only by the AST symbol census (no unreferenced top-level name) and the re-export census, which tests/tools/test_preflight_parent_census.py pins. Frozen-oracle context: many internals are asserted by byte-frozen tests, so slimming there is lane C by default.
- Gates were not run where they need torch (7, 12, 13), node/pyright (14), cargo (2, 4, 5) or a fresh clone (1). ci.yml was not executed; remote CI is suspended.
- Whether test_preflight_mint.py is still byte-frozen (03's EXEMPT grounds) cannot be settled from the tree: history is grafted at 46c49d9.
