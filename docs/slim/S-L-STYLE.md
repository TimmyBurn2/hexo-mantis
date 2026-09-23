# S-L-STYLE — LENS: code style, whole tree

scope: tracked `.py`/`.rs` under src/, tools/, tests/, crates/ (884 files: src 187, tools 78, tests 483,
crates 136), measured against CLAUDE.md "Code style", R316(e) (frozen text in
docs/governance/archive/rulings_register.md) and R346(f). tests/fixtures data is out of scope (no source there).
method: all static, nothing edited, nothing torch-dependent.
- gate 14's comment lint in measure mode: `.venv/bin/python tools/ci_gates/comment_lint.py --measure`
- gates 15 and 16 in their normal report mode: `tools/ci_gates/r8_header_gate.py`, `tools/ci_gates/encoding_io_gate.py`
- a throwaway AST/tokenize census, `scratchpad/lstyle/census.py`, which writes `scratchpad/lstyle/census.json`.
  It imports the gates' own predicates (`comment_lint.rust_comment_spans`/`_own_line`/`_rust_doc_excess`,
  `encoding_io_gate.is_unsafe`/`_justified`, `r8_header_gate.MARKER_RE`), so its totals reconcile exactly with
  gate 14 (3385 / 13079 / 1485 / 2676).
- `.venv/bin/ruff check --isolated --target-version py311 --select D1,ANN,BLE,E722,PLC0415 --statistics --no-fix <scope>`,
  and `--preview --select DOC501` for the `Raises:` half. `--isolated` means pyproject.toml was not read or changed.
- Rust: `cargo clippy --workspace --lib --bins --locked -- -A clippy::all -A clippy::pedantic -W clippy::unwrap_used -W clippy::expect_used`
  with the shared CARGO_TARGET_DIR. Only `--lib --bins`, so `#[cfg(test)]`, tests/ and benches/ are not compiled.
  It finished in 14 s (cached), so the grep fallback was not needed.
- PZ split: the PZ.md glob list, compiled to regexes and matched per file (221 of 884 files are in PZ).

## Summary

### Counts by class

The unit is **lines** for classes 1–4 and **sites** for classes 5–10. "PZ" means the part that lies in PZ.md glob files.

| # | class | count at HEAD | where it sits | PZ | held by a gate? | fix |
|---|---|---|---|---|---|---|
| 1 | file-top banners | 0 banner-rule lines. 3 files / 6 lines of file-top prose comment outside the docstring (not R8, not licence): crates/mantis-search/src/lib.rs, tests/train/test_cluster_stat_wiring.py, crates/mantis-graph/benches/build_bench.rs | crates 4, tests 2 | 0 | rule-shape only: gate 14 `banner_comment_lines` 0/0. The prose blocks are unmeasured | JUDGEMENT (fold into the docstring or delete) |
| 2 | narrative comment runs > 2 lines | `comment_excess_lines` 3385 = Rust `///`/`//!` runs 1771 (moved to class 3) + R8 headers 330 (carve-out) + marker runs 127 (carve-out) + **class 2 proper 1157**, which is: narrative 745 in 446 runs, invariant-ish (MUST/NEVER/ALWAYS/pinned) 376 in 178 runs, invariant-tagged (invariant / SAFETY: / load-bearing) 36 in 18 runs | src 469, tools 68, tests 395, crates 225 | 627 of 1157 | gate 14 floor 3385, measured 3385: **zero slack** | JUDGEMENT |
| 3 | multi-line non-invariant comments and docstrings | (a) public docstring excess 11594 = 10778 prose + 816 section lines (Args/Returns/Raises…); tests hold 7264 of the prose. (b) private docstring excess 1485. (c) Rust doc excess 2676. (d) two-line own-line comment runs 1353 (not carve-out, not invariant-tagged) | (a) src 3832, tools 480, tests 7282; (b) src 687, tools 205, tests 593 | (a) 3062 prose / 417 section; (b) 796; (c) 1403; (d) 613 | (a)+(b) `docstring_excess_lines` 13079/13079; (b) 1485/1485; (c) 2676/2676, all at their floors. (d) is unmeasured | POLICY first (see S-L-STYLE-07/08), then JUDGEMENT |
| 4 | ruling numbers in comments and docstrings | 1165 lines (R12+ with or without a clause letter, LAW-nn, F-nn, F-816-n, ADJ-, RQ-, AUDIT-n, WP…, CARD-…): comments 338, Rust doc comments 91, docstrings 736. Also 111 hard-rule cites R1–R11 outside R8 headers (33 comment, 78 docstring) | comments: src 109, tools 26, tests 173, crates 30. Docstrings: src 167, tools 53, tests 516 | 344 | measured, NOT gated: `ruling_cite_comment_lines` 393 (a narrower regex over comments only), **up from 351** in CARD-STYLE-BACKLOG. Docstrings are never read by the lint | JUDGEMENT (the card kept 104 comment + 152 docstring cites as provenance) |
| 5 | missing public docstrings or `Raises:` | ruff D1: src 201 (D101 36, D102 147, D103 18) plus `__init__` 39 and magic methods 11; tools 98 (21/24/53) plus 19 and 5; tests: 155 public helpers. DOC501: 212 missing-exception entries in 182 documented functions (src 125, tools 57), 106 of them public (src 72, tools 34), and 37 of those already carry a `Raises:` that omits a type. Also 21 public functions (src 15, tools 6) that raise directly and have no docstring at all | see the previous cell | undocumented public symbols 153 of 308; raise without `Raises:` 31 of 89 | unmeasured | JUDGEMENT. Adds lines, so it is census only and not slimming work |
| 6 | missing type hints | public src/tools APIs: 2 functions without a return type (src/mantis/util/coordinates.py::axial_distance, and an `__init__` in src/mantis/config/resolve/encoding.py). All ruff ANN except ANN401: src 60 (the 58 others are private or nested), tools 14, tests 4946, of which 1969 are test functions or fixtures | tests dominate | — | unmeasured (pyright basic does not require annotations) | MECHANICAL for `-> None` (ruff offers 1174 unsafe fixes in tests). JUDGEMENT for parameter types. The rule itself is "new/changed code", so on contact |
| 7 | `except Exception` / bare `except` outside a top-level handler | 58 broad handlers (src 39, tools 7, tests 12), bare `except:` 0. By context: top-level (`main`/`__main__`) 7, re-raise 8, `.exception(` log 5, other (record/count/return) 38. 53 of 58 carry `# noqa: BLE001 — <reason>` | src/mantis/selfplay/inference_server.py 6 | 21 | ruff BLE001 (src+tools) and E722 held at 0 by gate 14, via reasoned `noqa`. The "top-level only, logs through logger.exception" half is unmeasured | JUDGEMENT |
| 8 | encoding-less text I/O | gate 16's own predicate run over the whole tree: tools 0; tests module scope 1 (the registered EXEMPT row); tests function scope 211 in 66 files (53 in tests/tools/test_preflight_mint_process.py); src 15 in 10 files, of which 2 are `os.open` false positives (src/mantis/train/bundle.py), so **13 real**. Outside the predicate: subprocess `text=True` without `encoding=` 86 (tests 73, tools 10, src 3) | — | src 6 real; tests-fn 13; subprocess 12 | gate 16 holds tools/ and tests/ module scope. src/, tests/ function scope and subprocess are unmeasured | MECHANICAL (`encoding="utf-8"`; gate 16's own message says UTF-8 is always right here) |
| 9 | non-top imports | ruff PLC0415 statements: src 100, tools 40, tests 638. AST split (src 107 / tools 40 / tests 663):<br>• src: first-party 60, `mantis._engine` 13, torch 17, **optional analysis deps 8 (matplotlib/rich/scipy: the permitted exception)**, stdlib 5, numpy/yaml/pydantic 4<br>• tools: first-party 29, torch 3, strix-venv externals 5, stdlib 3<br>• tests: first-party 444 (incl. `_engine` 38), test-local helpers 29, stdlib 149, torch 30, third-party 11<br>21 test imports repeat an existing module-top import exactly | tests/util/test_device.py 20, tests/bots/test_sealbot_vendored.py 19, tests/conftest.py 14 | 188 | unmeasured (gate 9 sees only top-level cycles) | stdlib/third-party subset (src 9, tools 3, tests 160) is MECHANICAL. First-party and torch are JUDGEMENT/POLICY |
| 10 | Rust `unwrap()`/`expect()` on production paths | 64 sites in 13 files (unwrap Option 23 / Result 6, expect Result 24 / Option 11). By kind:<br>• mutex/condvar-poison expects 23<br>• registry/spec post-validation 27 (18 bare `unwrap()` in crates/mantis-encoding/src/registry/parse.rs, 9 spec-field expects)<br>• fixed-width `try_into().unwrap()` 6 (crates/mantis-selfplay/src/replay/hexg/persist.rs)<br>• other 8 (`choose`/`min`/`max`/`get` on sets assumed non-empty, `Gamma::new`, `visit_capacity`) | selfplay 40, encoding 18, bridge 4, core 1, search 1 | 32 in 4 files | unmeasured (gate 2's `-D clippy::all` does not include the restriction lints) | JUDGEMENT (named error types change signatures) |

**Carve-outs, counted separately and never proposed:**
- R8 justification headers: 172. Gate 15 reads "172 file(s) over the 300-line cap, all justified; … none stating a count, 0 stale". 97 of them are comment-run headers (339 lines); the rest sit in module docstrings.
- Comment lines carrying a load-bearing marker (the families overlap):
  - pinned/golden 254
  - armed/minted/provenance 239
  - planted-break 38
  - licence/"port of" attribution 16
  - Rust `SAFETY:` 7
- Pragma comments (noqa, type: ignore, pyright:, encoding-gate escape, fmt): 256.

### Mechanical vs judgement

- **MECHANICAL:**
  - class 8: all 13 real src sites, 211 test function-scope sites, 86 subprocess sites
  - class 6: test `-> None` returns
  - class 9: the stdlib/third-party hoists (172 statements) and 21 duplicate test imports
- **JUDGEMENT:** classes 1, 2, 4, 5, 7 and 10, plus class 9's first-party/torch part.
- **POLICY before any work:** class 3 (three rules disagree, see S-L-STYLE-07/08), class 4 (gate it or not) and class 8's subprocess scope.

### Lanes

12 blocks:
- A: 1 (S-L-STYLE-01)
- B: 9 (S-L-STYLE-03, -04, -05, -07, -08, -09, -10, -11, -12)
- C: 2 (S-L-STYLE-02, -06)

Each block is a class-level batch, not a per-line list.

### Top 3 by Δlines

All are upper bounds and all depend on a policy call:
- S-L-STYLE-08: −10778, if public docstrings are trimmed to one line and section lines are kept.
- S-L-STYLE-07: −4161, private docstrings −1485 plus Rust doc comments −2676.
- S-L-STYLE-05 + -06: −745, narrative runs trimmed to the 2-line cap (−352 non-PZ, −393 PZ).

### Top 15 files, all classes combined

The score is excess comment/docstring lines plus sites. Units are mixed, so read the ranking as an ordering, not as a quantity. Test functions' missing docstrings and hints are excluded, because they are not APIs.

| # | file | score | dominant class | runner-up | PZ |
|---|---|---:|---|---|---|
| 1 | tests/tools/test_preflight_mint_process.py | 268 | C3 178 | C8 57 | - |
| 2 | src/mantis/run.py | 246 | C3 116 | C2 110 | PZ |
| 3 | src/mantis/train/coordinator/step.py | 188 | C3 156 | C2 14 | PZ |
| 4 | tools/ci_gates/preflight_mint.py | 170 | C3 139 | C2 15 | PZ |
| 5 | src/mantis/diagnostics/worker_sweep.py | 167 | C3 140 | C5 17 | - |
| 6 | src/mantis/config/armed_aborts.py | 160 | C3 133 | C2 22 | PZ |
| 7 | src/mantis/config/schema/core.py | 120 | C3 105 | C2 9 | PZ |
| 8 | src/mantis/train/lifecycle/heartbeat_watchdog.py | 115 | C3 96 | C2 8 | PZ |
| 9 | src/mantis/config/schema/train.py | 113 | C3 89 | C2 19 | PZ |
| 10 | tools/ci_gates/preflight_mint_parent.py | 110 | C3 85 | C5 13 | PZ |
| 11 | src/mantis/selfplay/graph_collate.py | 107 | C3 66 | C5 19 | PZ |
| 12 | src/mantis/monitor/supervise.py | 104 | C3 95 | C2 3 | - |
| 13 | tests/model/test_bf16_parity_nulldist.py | 103 | C3 98 | C2 4 | PZ |
| 14 | tests/model/test_gine_bf16_drift.py | 102 | C3 99 | C4 1 | - |
| 15 | src/mantis/train/lifecycle/signals.py | 102 | C3 94 | C2 5 | - |

Class 3 dominates every file. With class 3 excluded, the leaders are:
- src/mantis/run.py: 130 (C2 110)
- tests/tools/test_preflight_mint_process.py: 90 (C8 57)
- src/mantis/train/coordinator/config.py: 58 (C5 46)

### Per-class file concentration

| class | files | top files |
|---|---:|---|
| C2 | 241 | run.py 110, crates/mantis-graph/src/lib.rs 24, armed_aborts.py 22 |
| C3 private docstrings | 220 | coordinator/step.py 93, preflight_mint.py 63, run.py 62 |
| C3 Rust docs | 133 | queues/graph.rs 78, mcts/mod.rs 76, replay/hexg/mod.rs 68 |
| C4 | 462 | run.py 17, trainer/core.py 13 |
| C5 | 89 | coordinator/config.py 46 |
| C10 | 13 | registry/parse.rs 18, queues/graph.rs 14 |

### Blockers (standing rules that gate batch work)

1. **R316(e) / CLAUDE.md "Applied ON CONTACT, never as a cleanup pass."** Only R346(f)'s wave-2 pass was ever exempt. A slimming pass over classes 1–4 needs a ruling that orders it, the way R346(f) did.
2. **Gate 14 floors have zero slack.** Every comment or docstring trim must lower `tools/ci_gates/comment_length_floor.txt` in the same commit. That file is in PZ-6, and lowering is its permitted direction.
3. **The pyproject.toml comment records tests/ as an "oracle-write corpus… edit-averse by frozen-oracle discipline".** Tests hold 7282 of the public docstring excess and 211 of the encoding sites.
4. **R8 interplay (gate 15).** Trimming class-2 narrative takes 1 over-cap file to ≤ 300 lines. Also trimming private docstrings and Rust docs takes 4. Each such file must drop its header in the same commit.

## Findings

### S-L-STYLE-01 | SIMPLIFY | A
subject: src/mantis/data/{corpus_analysis,corpus_io,corpus_metrics,generate,human_seeding}.py (the encoding-less `open`/`read_text`/`write_text` calls)
claim: 7 text-mode I/O calls in non-PZ src/ omit `encoding=`, so they default to the platform codepage (CLAUDE.md: "the RULE is the whole tree"). Each is a one-token add.
evidence: `is_unsafe` from tools/ci_gates/encoding_io_gate.py run over `git ls-files 'src/*.py'` -> 15 hits in 10 files. The 7 non-PZ hits are corpus_analysis 2, corpus_io 2, corpus_metrics 1, generate 1, human_seeding 1.
Δlines: 0 (edit-in-place)
witness: NONE (gate 16 scans only tools/ and tests/ module scope)
depends: —

### S-L-STYLE-02 | SIMPLIFY | C
subject: src/mantis/eval/pipeline.py, src/mantis/eval/worker.py, src/mantis/train/anchor.py, src/mantis/encoding/audit_sections.py (encoding-less text I/O)
claim: 6 real encoding-less text I/O sites sit in PZ files. The 2 `os.open(path, flags)` hits in src/mantis/train/bundle.py are predicate false positives and are not sites (see DEFECTS).
evidence: the same scan -> eval/pipeline 2, eval/worker 2, train/anchor 1, encoding/audit_sections 1, train/bundle 2 (both `os.open`).
Δlines: 0 (edit-in-place)
witness: NONE
depends: —

### S-L-STYLE-03 | SIMPLIFY | B
subject: tests/** function-scope encoding-less text I/O
claim: 211 sites in 66 files, 13 of them in PZ test files, 53 in tests/tools/test_preflight_mint_process.py alone. Gate 16 declares these a "registered backlog, not a rule", so fixing them all is a scope decision. Widening gate 16 to function scope afterwards is a gate change.
evidence: census.py `c8_tests_function` = 211. The gate's docstring: "other `tests/` sites and all of `src/` are deliberately out of scope".
Δlines: 0 (edit-in-place)
witness: gate 16, if widened in the same commit
depends: —

### S-L-STYLE-04 | SIMPLIFY | B
subject: subprocess `run`/`check_output`/`Popen` with `text=True`/`universal_newlines` and no `encoding=`
claim: 86 sites (tests 73, tools 10 of which 8 are in tools/ci_gates [PZ-6], src 3) decode through the locale codec. Gate 16's predicate does not cover them. Whether they fall under "text-mode IO always passes encoding=" is a policy call; once decided, the fix is mechanical.
evidence: census.py `c8_subprocess_text_noenc` = 86. `git grep -n "text=True" -- 'tools/*.py' | grep -v encoding` -> 10 lines.
Δlines: 0 (edit-in-place)
witness: NONE
depends: —

### S-L-STYLE-05 | SIMPLIFY | B
subject: own-line comment runs longer than 2 lines in non-PZ files
claim: 205 runs carry no invariant or marker tag (352 lines past the cap), plus 84 invariant-ish runs (160 lines) to judge site by site. R346(f): "≤ 2 lines unless stating a non-obvious invariant; no narrative".
evidence: census.py per-run classification. Its total reconciles with `comment_lint.py --measure` `comment_excess_lines 3385`.
Δlines: ≤ −352 (census.py `c2_excess_narrative` over non-PZ files; trim-to-cap, the excess definition gate 14 uses)
witness: gate 14 (the floor must fall in the same commit)
depends: BLOCKER 1 (the on-contact rule)

### S-L-STYLE-06 | SIMPLIFY | C
subject: own-line comment runs longer than 2 lines in PZ files (src/mantis/run.py alone accounts for 110 of them)
claim: 241 untagged runs (393 lines past the cap), plus 94 invariant-ish runs (216 lines).
evidence: census.py, the PZ half of the same split.
Δlines: ≤ −393 (census.py `c2_excess_narrative` over PZ files)
witness: gate 14
depends: BLOCKER 1

### S-L-STYLE-07 | SIMPLIFY | B
subject: multi-line PRIVATE Python docstrings and multi-line Rust `///`/`//!` runs
claim: 1485 private docstring lines (non-PZ 689 / PZ 796) and 2676 Rust doc lines (non-PZ 1273 / PZ 1403) sit beyond the first line. R346(f) names only public APIs, and CARD-STYLE-BACKLOG leaves "whether a private symbol may carry a multi-line docstring" as the architect's call. Rust docs are the same question. Both measures are already gated at their floors.
evidence: `comment_lint.py --measure` -> `private_docstring_excess_lines 1485`, `rust_doc_excess_lines 2676` (floor file: 1485 / 2676).
Δlines: ≤ −4161 (1485 + 2676, if the call is "one line")
witness: gate 14
depends: an architect ruling (CARD-STYLE-BACKLOG); BLOCKER 1

### S-L-STYLE-08 | SIMPLIFY | B
subject: public docstrings and two-line comments, where three rules disagree
claim: The three rules:
- R346(f) says "one-line docstrings on public APIs".
- CLAUDE.md requires a `Raises:` section naming every catchable exception, which is multi-line by construction.
- CLAUDE.md's "ONE line" for comments is stricter than R346(f)'s "≤ 2".

What they apply to:
- 11594 public docstring lines beyond the first: 816 are section lines, 10778 are prose, and tests hold 7264 of the prose. Test functions are not APIs.
- 1353 two-line comment runs sit between the two comment rules.

evidence: census.py `c3_doc_excess_pub` 11594 (reconciles with 13079 − 1485), `c3_doc_sectionlines_pub` 816, `c3_tworun_runs` 1353.
Δlines: 0 until ruled; upper bound −10778 (census.py `c3_doc_prose_excess_pub`)
witness: gate 14 `docstring_excess_lines`
depends: a policy ruling reconciling R346(f) with CLAUDE.md's `Raises:` rule; BLOCKER 1

### S-L-STYLE-09 | SIMPLIFY | B
subject: ruling-number cites in comments and docstrings
claim: 1165 lines cite a ruling, law, finding or card ID: comments 338, Rust doc 91, docstrings 736. Gate 14's own count is printed but not gated, and it rose from 351 (CARD-STYLE-BACKLOG) to 393. Its regex only counts `R\d+` when a clause letter follows, and it never reads docstrings. The card kept 256 parenthetical cites as provenance, so a sed strip is wrong: this is site-by-site work, plus a call on whether to gate the count.
evidence: `comment_lint.py --measure` -> `ruling_cite_comment_lines 393`. census.py -> `c4_cite_comment` 338, `c4_cite_rustdoc` 91, `c4_cite_docstring` 736.
Δlines: 0 (edit-in-place; a cite-only line deletion cannot be separated without judgement)
witness: NONE (ungated)
depends: BLOCKER 1

### S-L-STYLE-10 | SIMPLIFY | B
subject: function-scope imports
claim: 21 test imports exactly repeat an existing module-top import; pyproject's F811 ignore records a "deliberate re-import idiom", so check each one. 172 stdlib/third-party inner imports (src 9, tools 3, tests 160) are hoistable. The first-party (src 73 incl. `_engine`, tools 29, tests 444+29) and torch (src 17, tools 3, tests 30) sites need two policy calls:
- whether torch counts as an "optional dep" under CLAUDE.md's one exception. It is a dependency group/extra, not a base dependency.
- what a hoist does to gate 9's top-level import DAG.

evidence: `ruff check --isolated --select PLC0415 --statistics` -> src 100 / tools 40 / tests 638. The census AST split is in the class table.
Δlines: ≤ −21 (the duplicate re-imports, 21 statements = 21 lines by AST span). Hoists are 0 (edit-in-place).
witness: gate 9 (tools/check_import_dag.py), for first-party hoists
depends: —

### S-L-STYLE-11 | SIMPLIFY | B
subject: crates/mantis-selfplay/src/{queues/graph.rs, runner/finalize.rs, runner/game.rs, runner/spawn.rs, replay/hexg/persist.rs}, crates/mantis-bridge/src/{inference.rs, runner.rs}, crates/mantis-core/src/board/moves.rs, crates/mantis-search/src/mcts/dirichlet.rs (the non-PZ unwrap/expect sites)
claim: 32 production `unwrap()`/`expect()` sites in 9 non-PZ files. By kind:
- lock/condvar-poison expects: 16
- `try_into` on fixed-width slices: 6 (a helper returning `[u8; N]` removes them)
- bridge spec-field expects after validation: 4
- other: 6 (`choose`/`min`/`max`/`get`/`Gamma::new`)

CLAUDE.md wants a named error type instead. That changes signatures, and the poison-expect idiom needs a stance first.
evidence: the clippy command in the method line -> 64 warnings, 32 of them outside PZ files.
Δlines: 0 (edit-in-place)
witness: NONE (restriction lints are not in gate 2)
depends: —

### S-L-STYLE-12 | SIMPLIFY | B
subject: the style classes no gate measures (a gate-coverage decision, no code change)
claim: Seven classes are unmeasured or only partly measured, so any fix can silently regrow:
- class 1: prose file-top blocks
- class 3(d): two-line runs
- class 4: gated neither for comments nor for docstrings
- classes 5 and 6: D1/DOC501/ANN are not in the ruff select
- class 8: src, test function scope, subprocess
- class 9: PLC0415
- class 10: `clippy::unwrap_used`/`expect_used`

R98 admits a rule only with a named defect class and a clean baseline, so each needs its own adoption call.
evidence: pyproject.toml `[tool.ruff.lint] select = ["E","F","W","I","UP","B","BLE","PLE"]`; the gate 16 docstring; the comment_lint docstring ("`ruling_cite_comment_lines` is measured and PRINTED but never gated").
Δlines: 0 (edit-in-place)
witness: the gates themselves
depends: S-L-STYLE-01..11

## DEFECTS
- tools/ci_gates/encoding_io_gate.py::is_unsafe treats `os.open(path, flags)` (a low-level fd open) as `Path.open`, so the 2 hits in src/mantis/train/bundle.py are false positives. This is latent: src/ is not scanned today, but it would trip if gate 16 were widened to src/.
- tools/ci_gates/comment_lint.py::_RULING misses bare `R\d{2,3}` cites and never reads docstrings, so `ruling_cite_comment_lines` (393) under-measures the class it names (1165 by the broad regex). It is by design ("measured, not gated"), but the printed figure is not the class size.

## PARKED
none

## HANDOFF
- R3 (S-A-RUST-3): `cargo clippy --lib` reports crates/mantis-bridge/src/encoding.rs::from_static as never used outside `#[cfg(test)]`, and the `InferenceBatcher` field `feature_len` in crates/mantis-bridge/src/inference.rs as never read (DEAD candidates). It also reports 2 pyo3 `HasAutomaticFromPyObject` deprecation warnings on `#[pyclass]` `InferenceBatcher` and `SelfPlayRunnerConfig`, which will break a future pyo3 bump.

## Not covered
- Invariant tagging in class 2 is a regex heuristic: "strict" = invariant|SAFETY:|load-bearing; "ish" = also MUST|NEVER|ALWAYS|pinned. The narrative/invariant split is an upper/lower bracket, not a verdict. Carve-out tagging (planted|golden|provenance|ARMED|licence|pragma) is also a heuristic.
- The docstring prose/section split counts everything from the first section header (`Args:`/`Returns:`/`Raises:`/…) to the end as section lines.
- DOC501 is a ruff preview rule. It sees only documented functions and only exceptions raised in the function body, not ones propagated from callees.
- Class 6 counts use my public/private rule (leading `_`, nesting in a def or private class). Ruff's ANN counts are given alongside for reproducibility.
- Rust: `--lib --bins` only. Unwraps in benches/, tests/ and `#[cfg(test)]` modules are excluded by construction. Build scripts and doctests were not linted. rustfmt conformance was not measured (it is on-contact and ungated by rule).
- pyright was not run. tools/strix_driver.py runs in an external venv; only its inner-import count is included.
- The per-line site lists are in scratchpad/lstyle/census.json, scratchpad/lstyle/clippy.txt and scratchpad/lstyle/doc501.txt. They are not reproduced here by design.
