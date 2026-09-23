# S-A-DOCS-3 — D3 docs/design/**
scope: docs/design/** (43 files, 12 314 lines: 5 top-level, archive/ 8, measurements/ 28, research/ 2);
method: per-file citer census (`git grep -n -F <stem>` over the whole tree, docs/slim excluded, each hit
classed as governance / contract / code / other design doc, and flagged when the hit is a full
`docs/design/...` path inside gate 10's scan scope); gate 10 scope read from
tools/ci_gates/check_tracked_refs.py (Makefile, README.md, CLAUDE.md, docs/contracts/*.md,
docs/governance/*.md non-recursive, RULINGS.md exempt, docs/design exempt); code consumers by
`git grep -n -E "design/measurements|[\"']design[\"']"` over tools tests src crates; path-token drift with
gate 10's own TOKEN_RE over each live design doc; `mantis.*` symbol drift by AST resolution against src/;
arch-class bases by AST over src/mantis/model/arch.py. Scratch scripts in the scratchpad (d3/).

## Summary
- Findings: 8. ONE-SHOT 3 (lane A 1, lane C 2); DOC 4 (lane A 1, lane C 3); PACK 1 (lane C).
- Top 3 by Δlines: S-A-DOCS-3-03 (-450), S-A-DOCS-3-02 (-144), S-A-DOCS-3-01 (-80). Slice total if all land: -674.
- Verdict: the slice is almost entirely STANDING RECORD. 39 of 43 files are cited by a ruling, a
  governance doc, a contract doc, a CLAUDE/README line or code; 32 of them are cited by full path in
  gate-10 scope (30) or only in RULINGS (2: A2_QUIESCENCE, FORCED_MOVE_CENSUS), counted from the census
  script's GATE10 rows, so a delete reds gate 10 or breaks a ruling citation that corrects only by
  annotation. One file is uncited (SHAKEDOWN7G). Five frozen archive records are cited only by the
  frozen archive index (and, for the R153 four, by two protected Rust test headers at a stale path).

Citer codes: R RULINGS, S STATE, C CARDS, F falsified, K docs/contracts, D other design doc,
X code/tests/vendor, A docs/audits, G10 = full path cited inside gate 10 scope (delete reds gate 10).

| path (docs/design/…) | lines | citers | verdict |
|---|---|---|---|
| repo_design.md | 1156 | CLAUDE, README (G10), K×3, R, S, C, A, X (tests/test_exit_code_table_census.py and tests/test_run_import_authority.py READ it) | KEEP, lane C (S-03 §3 drift: -04, -05) |
| analyzer_design.md | 565 | D (repo_design R363 amendment, SR2), X tools/analyzer/__init__.py docstring | KEEP; status line stale (-06) |
| observatory_design.md | 482 | S, C (G10), D | KEEP, lane C — STATE: "stays as DASH-2's record" (-07) |
| observatory_research.md | 405 | S (by name, same sentence), D | KEEP, lane C (-08) |
| eval_gate_memo_2026-09-15.md | 417 | K run_config_schema (G10), S (G10), X src/mantis/eval/sequential.py, D | KEEP, lane C |
| archive/README.md | 15 | frozen index (R355(f)) | KEEP, lane C; loses 6 lines under -02/-03 |
| archive/repo_design_contract5_amendments_v1_to_v8.md | 373 | D repo_design §4 (path) | KEEP, lane C |
| archive/measurements/MEASUREMENT_PERF3B_2026-09-11.md | 81 | C (G10 ×2) | KEEP, lane C |
| archive/measurements/{MEASUREMENT_R153,MEASUREMENT_R153_LEG2,PREREG_R153,PREREG_R153_LEG2}.md | 96+149+106+96 | archive README; each other; X crates/mantis-search/tests/r153_{target_mass,leg2_ls_target_mass}.rs headers (STALE pre-archive path) | DELETE, lane C (-03) |
| archive/measurements/RUN7_PREREG_2026-09-13.md | 141 | archive README only | DELETE, lane C (-02) |
| measurements/SHAKEDOWN7G_2026-09-14.md | 80 | NONE | DELETE, lane A (-01) |
| measurements/RUN7_STAMP2_2026-09-14.md | 83 | X vendor/patches/sealbot.patch (comment in the applied patch), D ×5 | KEEP (a code citer; dropping it means editing the pinned patch) |
| measurements/RUN8_PREREG_2026-09-17.md | 251 | R, S (G10), X tests/diagnostics/test_ring_audit.py READS it (`A.load_bands(Path(...))`) | KEEP, hard consumer |
| measurements/SEALBOT_TT_AB_2026-09-14.md | 115 | S, C (G10), X tools/dashboard/tier2.py, tests/tools/test_run_dashboard.py | KEEP |
| measurements/STRIX_RUN7_60K_2026-09-17.md | 94 | R, S (G10), X tools/dashboard/external.py, D | KEEP |
| measurements/FORCED_MOVE_CENSUS_2026-09-15.md | 493 | R, X src/mantis/diagnostics/ring_reader.py, D | KEEP |
| measurements/GAME_QUALITY_CENSUS_2026-09-14.md | 228 | K (G10), R, X src/mantis/diagnostics/tactics.py, D | KEEP |
| measurements/{EVAL_COST_2026-09-19, PERF_A4_2026-09-11, PROBE1_2026-09-21, STRENGTH_FRONTIER_1_2026-09-13} | 225, 659, 396, 202 | K (G10) + R + C/S/F (G10) | KEEP, lane C |
| measurements/{PERF3_2026-09-18, RUN9_PREREG_2026-09-19, PARAM_DISTANCE_2026-09-21, RUN10_PREREG_2026-09-21, RUN7_EVAL_COST_2026-09-15} | 262, 212, 87, 233, 99 | R + S + C (G10; PERF3/RUN9 also F); RUN10_PREREG is STATE's procedure pointer | KEEP, lane C |
| measurements/{PERF_INVESTIGATION_2026-09-11, MEASUREMENT_STARTPATH_2026-09-11, INVESTIGATION1_2026-09-15, SHAKEDOWN7_2026-09-14} | 592, 145, 138, 95 | F (G10) (+R/C) | KEEP, lane C |
| measurements/{GAME_QUALITY_2026-09-14, INVESTIGATION1_TROUGH_2026-09-13, LADDER_SHAKEDOWN_2026-09-19, RUN6_BLOCK_2026-09-12} | 497, 64, 100, 171 | R + C (G10) | KEEP, lane C |
| measurements/{MEASUREMENT_OC7_2026-09-11, STRIX_RUNG_2026-09-14, CPU_HEAD_PROFILE_2026-09-20} | 79, 60, 96 | C or S (G10) only (+D) | KEEP, lane C |
| measurements/A2_QUIESCENCE_FALSIFIER_2026-09-16.md | 77 | R (path), D | KEEP, lane C (ruling-cited) |
| research/STRENGTH_RESEARCH_2026-09-18.md | 686 | R, S, F (G10) | KEEP, lane C |
| research/STRENGTH_RESEARCH_2_2026-09-21.md | 1713 | R (grounds of a ruling), F (G10) | KEEP, lane C |

## Findings

### S-A-DOCS-3-01 | ONE-SHOT | A
subject: docs/design/measurements/SHAKEDOWN7G_2026-09-14.md
claim: a spent run7 shakedown-twin measurement record that nothing in the tree cites; its one-line result is already carried by the archived run7 prereg and the rulings of that day.
evidence: `git grep -n -i -E "shakedown ?7g|SHAKEDOWN7G" -- . ':!docs/slim'` -> only the file itself and docs/design/archive/measurements/RUN7_PREREG_2026-09-13.md (which names the RUN `shakedown7g`, not this doc)
callers:
  AST imports / entry points / `python -m` / subprocess / importlib / conftest / pyo3 / config keys: n/a for a .md; code readers searched with `git grep -n -E "design/measurements|[\"']measurements[\"']|[\"']design[\"']" -- tools tests src crates Makefile` -> no glob or reader of the directory, only named-file reads (RUN8_PREREG, repo_design) -> none for this file
  gate tool paths: tools/ci_gates/check_tracked_refs.py exempts docs/design and no in-scope file names the stem (stem grep above covers Makefile/README/CLAUDE/contracts/governance) -> none
  STATE procedures: `git grep -n -i shakedown7g -- docs/governance` -> none
  why these searches would find a caller: every citation shape in this repo (path, basename, stem, run name) contains the literal stem `SHAKEDOWN7G` or `shakedown7g`, searched case-insensitively over the whole tree
Δlines: -80 (wc -l docs/design/measurements/SHAKEDOWN7G_2026-09-14.md)
witness: NONE (gate 10 green either way; no test reads it)
depends: —

### S-A-DOCS-3-02 | ONE-SHOT | C
subject: docs/design/archive/measurements/RUN7_PREREG_2026-09-13.md (+ its bullet in docs/design/archive/README.md)
claim: run7's superseded pre-registration, cited only by the frozen archive index (the one other hit, INVESTIGATION1's relocation table, names it as an item to MOVE, not as grounds); lane C because R355(f) froze docs/design/archive/.
evidence: `git grep -n -E "RUN7_PREREG" -- . ':!docs/slim' ':!docs/design/archive'` -> docs/design/measurements/INVESTIGATION1_2026-09-15.md (relocation list row only); stem grep over governance -> 0
Δlines: -144 (wc -l = 141; archive/README.md bullet = 3 lines, `grep -n RUN7_PREREG docs/design/archive/README.md` + its continuation lines)
witness: NONE
depends: a ruling that lifts R355(f)'s "never edited" for docs/design/archive/ (the README must drop the bullet)

### S-A-DOCS-3-03 | ONE-SHOT | C
subject: docs/design/archive/measurements/{MEASUREMENT_R153,MEASUREMENT_R153_LEG2,PREREG_R153,PREREG_R153_LEG2}.md (+ README bullet)
claim: the R153 line-dispersal probe and its two preregs, superseded (the README says so: R346(f) deleted the grid path they measured); cited only by the frozen index, each other, and two protected Rust test headers that name a pre-R355(f) path which no longer exists.
evidence: stem grep -> archive README, archive siblings, INVESTIGATION1 relocation row; `git grep -n "design/measurements" -- crates` -> crates/mantis-search/tests/r153_target_mass.rs header "`docs/design/measurements/{PREREG,MEASUREMENT}_R153.md`", r153_leg2_ls_target_mass.rs header "committed verbatim under `docs/design/measurements/`"
Δlines: -450 (wc -l 96+149+106+96 = 447; README bullet 3 lines)
witness: NONE (docs/design exempt from gate 10; the Rust headers are comments)
depends: R355(f) lift as in -02; HANDOFF crates/mantis-search tests (whether r153_* still measure a live subject decides whether the headers are repointed or go with them)

### S-A-DOCS-3-04 | DOC | C
subject: docs/design/repo_design.md §3 "Representation extensibility"
claim: §3 says `build_net`'s dispatch order is load-bearing "because `GnnArchV2` is a subclass of `GnnArch`" — false at HEAD, and §3 never names the third kind `GnnArchV2SoftPolicy`; the code's own comment states the opposite rationale.
evidence: AST over src/mantis/model/arch.py -> `GnnArch [] ['dataclass(frozen=True)']`, `GnnArchV2 []`, `GnnArchV2SoftPolicy []` (no bases); `grep -c GnnArchV2SoftPolicy docs/design/repo_design.md` -> 0; src/mantis/model/build.py::build_net comment "if a kind is ever made a subclass of an earlier one, the earlier `isinstance` would silently build it" (order is defensive, not forced); confirms L-SEAM's report
Δlines: 0 (a correcting amendment; repo_design changes only by amendment commit, R9)
witness: NONE (no test parses §3's prose)
depends: —

### S-A-DOCS-3-05 | DOC | C
subject: docs/design/repo_design.md, AMENDMENT "`search.kind`: ONE search regime, read by ONE selector"
claim: the amendment still states in the present tense that `mantis.config.resolve.resolve_search_kind` "is the ONE selector"; the later R351(c) SPLIT amendment records it deleted, and the symbol does not resolve (src/mantis/config/resolve/search.py has resolve_selfplay_search_kind / resolve_deploy_search_kind).
evidence: AST resolution of every `mantis.*` token in repo_design -> 6 unresolved: resolve_search_kind, diagnostics.workspace_durability, util.mounts, eval.{bt,channel_health,ladder}; the last five sit in amendments that RECORD their deletion (correct); only resolve_search_kind is asserted live in an earlier amendment's text
Δlines: 0 (a one-line "superseded by R351(c)" note under R9)
witness: NONE
depends: —

### S-A-DOCS-3-06 | DOC | A
subject: docs/design/analyzer_design.md (Status line)
claim: the Status line says the code "is on branch `worktree-analyzer` awaiting the ruling number and the ff"; at HEAD tools/analyzer/ is tracked, `make analyzer` dispatches it, R363(d) admitted ANALYZER-1 and RULINGS records the landing commit — a non-canonical working doc that disagrees with the tree, repaired in place under R311(c).
evidence: `grep -n Status: docs/design/analyzer_design.md` -> the line above; `git ls-files tools/analyzer | head` -> tracked; RULINGS R363 "(d) ANALYZER-1 admitted under R9 as amended" and "the ANALYZER-1 landing"; tools/analyzer/__init__.py docstring cites this doc
Δlines: 0 (one line reworded)
witness: NONE
depends: —

### S-A-DOCS-3-07 | DOC | C
subject: docs/design/observatory_design.md
claim: KEEP. CALLERS' flag (`python -m mantis.dash` does not resolve) is NOT drift — the doc quotes R344(d)'s archived dispatcher note ("recorded, not decided") and itself concludes `src/mantis/dash/` "cannot be the home"; the real staleness is its Status line ("proposal … Nothing here is landed") against STATE item (5), which records the 2026-09-17 decision and the reader layer's retirement.
evidence: `sed -n 30,52p docs/design/observatory_design.md`; STATE "(5) OBSERVATORY — DECIDED 2026-09-17 under R355(f) … The design (`docs/design/observatory_design.md`, `observatory_research.md`) stays as DASH-2's record"; path-token scan -> 14 untracked tokens, all proposal layout (tools/observatory/…, src/mantis/dash/) inside a doc STATE keeps as a record
Δlines: 0 (optional one-line status pointer to STATE item (5))
witness: NONE
depends: —

### S-A-DOCS-3-08 | PACK | C
subject: docs/design/measurements/ (27 records besides -01), docs/design/research/, eval_gate_memo, observatory_research, archive/{README, contract5 chronicle, PERF3B}
claim: KEEP IN PLACE, no archive move: every file is a one-shot prereg or measurement record that a ruling, a governance doc, a contract doc or code cites by FULL PATH, so a move edits gate-10-scoped governance and contract paths and leaves RULINGS citations dangling (RULINGS corrects only by annotation, R9); three are READ by code (RUN8_PREREG by tests/diagnostics/test_ring_audit.py; repo_design by two tests).
evidence: citer census script (scratchpad d3/cite.py) -> per-file rows in the table above; G10 hits e.g. PERF_A4: run_config_schema.md, CARDS ×2, STATE ×3; PROBE1: run_config_schema.md, CARDS ×3, STATE ×2
Δlines: 0
witness: gate 10 (tools/ci_gates/check_tracked_refs.py) reds on any delete or move of a G10-cited file; tests/diagnostics/test_ring_audit.py reds on RUN8_PREREG
depends: —

## DEFECTS
- crates/mantis-search/tests/r153_target_mass.rs and r153_leg2_ls_target_mass.rs headers cite `docs/design/measurements/…R153…`, a path R355(f) moved to docs/design/archive/measurements/ (stale path in protected tests).
- docs/design/repo_design.md §3 states a class relation that is false at HEAD (see -04).

## PARKED
none

## HANDOFF
- crates/mantis-search tests slice: do r153_target_mass.rs / r153_leg2_ls_target_mass.rs still measure a live subject after R346(f) deleted the grid path? (decides -03's Rust-header side).
- tools/ci_gates slice (gate 17): 16 design docs carry 43 box-absolute working-directory paths (`git grep -c` over docs/design); gate 17's pattern set does not match that prefix. Rule-7 class, not a slimming item.

## Not covered
- The numbers inside the 28 measurement and 2 research records were not re-derived (spent-run records; the census asks who cites them, not whether they are right).
- repo_design.md was checked only by path tokens, `mantis.*` symbol resolution and §3; no section-by-section audit of its remaining prose against the tree.
- Foreign-repo path tokens in observatory_research.md and STRENGTH_RESEARCH_2026-09-18.md (34 each) cite OTHER projects' trees and were not treated as drift.
- Frozen archive records' internal claims (R355(f)).

## Review
reviewer: fresh read-only agent (not the author); probes in throwaway worktrees, removed
(one worktree, scratchpad/wt/rev-adocs3-01, at HEAD 5a698b5; removed, `git worktree list` shows no rev-adocs3 entry)

| ID | verdict | lane | Δlines (probe-measured for lane A) | note |
|---|---|---|---|---|
| S-A-DOCS-3-01 | CONFIRMED | A | -80 | zero citers by basename, path or run name; probe green (import, collect, cargo, gates 6/10/15/17, nearest tests) |
| S-A-DOCS-3-02 | CONFIRMED | C | -144 (informational probe) | freeze wording is the archive README's "never edited"; the archive-not-delete choice was the R355 forward's §3 list |
| S-A-DOCS-3-03 | REFUTED (as a delete) | C | 0 | the records are the prereg grounds of two LIVE `#[test]` oracles in PZ-protected crates/mantis-search/tests/**; repoint their headers instead |
| S-A-DOCS-3-04 | CONFIRMED | C | 0 | `class GnnArchV2:` has no base; §3 prose is false at HEAD |
| S-A-DOCS-3-05 | REFUTED | — | 0 | the successor R351(c) amendment already names "the `search.kind` amendment above" and records the deletion |
| S-A-DOCS-3-06 | CONFIRMED | A | 0 | mechanical R311(c) repair: every replacement fact is tree-verifiable |
| S-A-DOCS-3-07 | AMENDED | C | 0 | KEEP holds; only "proposal for a ruling" is stale — "Nothing here is landed" is TRUE at HEAD |
| S-A-DOCS-3-08 | AMENDED | C | 0 | KEEP holds; gate-10 witness overstated: 18 of 43 files are cited by full path in gate-10 scope, not 30/32 |

### Per-finding notes
S-A-DOCS-3-01 — CONFIRMED: `grep -rIn -i SHAKEDOWN7G_2026-09-14` over the whole working tree (excl. .git/.venv/target/docs/slim) -> 0 hits; `grep -rIli "shakedown.\?7g"` -> only the file itself and the archived RUN7_PREREG (run name); governance prose search for the twin -> R353 cites "the twin shakedown's trainer reading" via STATE at a commit, not this doc; no code reader globs docs/design (`grep -rn "docs/design|'design'|glob(*.md)"` over tools tests src crates Makefile -> only named reads of repo_design, RUN8_PREREG, analyzer_design/eval_gate_memo/SEALBOT_TT_AB docstrings, and gate 10's own glob of contracts/governance). PROBE: `git rm` -> 1 file, 80 deletions; `python -S -c "import mantis"` ok from the worktree's src; collect-only `-m ''` -> "2289 tests collected, 167 errors" (= baseline); `cargo check --workspace --all-targets --locked` -> Finished; gate 10 rc 0; gate 15 rc 0 (172/172, 0 stale); gate 6 `--base HEAD` rc 0 (degraded wide); gate 17 `--base HEAD` rc 0 (degraded to full tree, 1062 files) and `--full-tree` rc 0; nearest doc-reading tests (test_ring_audit, test_exit_code_table_census, test_comment_lint, test_rule7_gate) -> 117 passed (test_run_import_authority is one of the 167 torch-bound collection errors at HEAD too). The file also carries box-absolute working-directory paths (rule-7 class), so the delete removes some of the HANDOFF's 43.
S-A-DOCS-3-02 — CONFIRMED (lane C): whole-tree basename grep -> only archive/README.md and INVESTIGATION1's relocation row (an order to MOVE, carried out); prose grep for "run7 prereg/pre-registration" over governance/contracts/design/audits -> 0. Informational probe (same worktree, lane-C files also removed, README bullets dropped): diff 7 files, -674 = 80 + 141 + 447 + 6; gate 10 rc 0 (docs/design exempt). Nuance: R355(f)'s clause text is "The DELETE list (§3) is APPROVED as sized"; the "never edited" freeze lives only in the archive README, and the same §3 list chose ARCHIVE over DELETE for these files — reversing that is the architect's call, as the scout's depends says.
S-A-DOCS-3-03 — REFUTED as a delete: `grep -n "#[test]\|#[ignore"` -> r153_target_mass.rs::r153_characterize_exported_target_dropped_mass and r153_leg2_ls_target_mass.rs::r153_leg2_run5_exposure_through_production_expand are live, un-ignored tests ("A permanent regression oracle"); PZ.md lists crates/mantis-search/tests/** as protected; r153_leg2's own R8 header justifies the file as one unit so "the regression oracle and the prereg'd measurement cannot drift apart" — the four records are the frozen verdict rule of a live oracle, i.e. they have code citers. This answers the scout's HANDOFF: the headers should be REPOINTED to docs/design/archive/measurements/ (a protected comment edit, lane C), not deleted with the records. On the defect: gate 10 scans only Makefile/README/CLAUDE/contracts/governance `*.md` (read in check_tracked_refs.py SCOPE/GLOB_SCOPE); the gates that do read `.rs` (comment_lint, r8_header_gate, silent_encoding_gate) check comment length, headers and fallbacks, not paths; cargo check green -> the stale path reds NOTHING. The defect line stands as a DEFECT.
S-A-DOCS-3-04 — CONFIRMED: `grep -B2 "^class GnnArch"` src/mantis/model/arch.py -> three `@dataclass(frozen=True)` classes, `class GnnArchV2:` and `class GnnArchV2SoftPolicy:` with no base list; repo_design §3 says the order is load-bearing "because `GnnArchV2` is a subclass of `GnnArch`" and names only two dataclasses; build.py::build_net's comment states the order is a "second line of defence". Lane C (repo_design changes by amendment, R9).
S-A-DOCS-3-05 — REFUTED: `grep -n "resolve_search_kind\|R351(c)" docs/design/repo_design.md` -> the SPLIT amendment opens "The `search.kind` amendment above made the regime one top-level key" and states "`resolve_search_kind` is deleted"; `grep -c -i superseded docs/design/repo_design.md` -> 0: the doc's convention is supersession by a later amendment, already in place — the same shape the scout itself judged correct for the five other unresolved symbols. A back-pointer is the doc owner's taste, not drift.
S-A-DOCS-3-06 — CONFIRMED, mechanical: Status line reads "proposal for a ruling … on branch `worktree-analyzer` awaiting the ruling number and the ff"; `git ls-files tools/analyzer | wc -l` -> 13; Makefile has an `analyzer:` target; RULINGS R363 "(d) ANALYZER-1 admitted" and "the ANALYZER-1 landing (`02f64442`)". The doc is a design doc, not a register (repo_design's R363 amendment cites it as the design, which does not make it canonical), so R311(c) in-place repair applies and every replacement fact is derivable from the tree and RULINGS — no owner call needed. The line's trailing "No producer, no gate moved" was not re-derived; keep it unless the repairer checks it.
S-A-DOCS-3-07 — AMENDED: KEEP confirmed (`mantis.dash` flag is quoted R344(d) text the doc itself rejects). But STATE item (5) says the phase-1 reader layer was RETIRED from the tree, so the Status line's "Nothing here is landed; no code" is TRUE at HEAD; only "proposal for a ruling" is stale (it was DECIDED 2026-09-17: not adopted, kept as DASH-2's record). The optional pointer stays the doc owner's call (STATE keeps the doc as a record).
S-A-DOCS-3-08 — AMENDED: KEEP holds on ruling/governance citation grounds, but the witness is narrower than stated: `for f in $(git ls-files docs/design); grep -lF "$f"` over Makefile, README.md, CLAUDE.md, docs/contracts/*.md, docs/governance/*.md minus RULINGS -> 18 of 43 files cited by FULL path in gate-10 scope (scout: 30, "32 of them"). Gate 10's TOKEN_RE needs a `docs/…` prefix, so a basename citation does not red it, and a brace form only yields a directory-prefix token that passes.
Citation-table sample (8 rows): analyzer_design, observatory_research, archive MEASUREMENT_PERF3B, RUN7_STAMP2, A2_QUIESCENCE_FALSIFIER, CPU_HEAD_PROFILE, INVESTIGATION1_TROUGH, STRENGTH_RESEARCH_2. Exact for 5. Three carry a wrong "(G10)": CPU_HEAD_PROFILE (STATE cites basename only), INVESTIGATION1_TROUGH (CARDS basename only), STRENGTH_RESEARCH_2 (falsified basename only; also two design-doc citers the row omits). No KEEP verdict changes. Whole-tree scan for design docs with no citer outside docs/design/archive -> exactly SHAKEDOWN7G plus the archive records of -02/-03 (MEASUREMENT_R153 has the INVESTIGATION1 row; the R153 pair also has the Rust headers in brace/dir form).

### Missed by the scout (optional, max 5)
- NEW-1 | DOC | C | the HANDOFF's gate-17 claim holds: `rule7_gate.py --full-tree` rc 0 on the main tree with box-absolute paths present in design docs (e.g. -01's instrument line) -> the tracked pattern set misses that prefix; gate-17 slice, not a slimming item.

### Tally: raised 8 | confirmed 4 | amended 2 | refuted 2 | pending 0 | architect 0
