# S-A-DOCS-4 — D4: configs/, vendor/, root files
scope: configs/** (6), vendor/pins.toml, vendor/patches/sealbot.patch, .gitattributes, .gitignore, CLAUDE.md,
Cargo.lock, Cargo.toml, LICENSE, README.md, mise.toml, pyproject.toml, rust-toolchain.toml, rustfmt.toml,
uv.lock (20 files, 4764 lines by `wc -l`); method: `git grep`, `git ls-files`, `git check-attr`,
`git ls-files -i -c --exclude-standard`, `.venv/bin/ruff check --isolated --statistics`, `wc -l`, `grep -c`.
Tree: origin/dev 69e1532 plus docs/slim commits only (HEAD 831bb76; `git diff --stat 69e1532 HEAD` touches only docs/slim).

## Summary
- 12 findings. By class: DOC 8 (01, 02, 03, 05, 06, 09, 10, 12), CONFIG 3 (04, 08, 11), PACK 1 (07); SIMPLIFY, DUP, DEAD, TEST and ONE-SHOT 0.
- By lane: A 5 (03, 04, 05, 09, 10), B 1 (11), C 6 (01, 02, 06, 07, 08, 12).
- Top 3 by Δlines: S-A-DOCS-4-08 (-558 if the retired run configs go; 0 if they only leave the census),
  S-A-DOCS-4-12 (upper bound -202 comment lines), S-A-DOCS-4-07 (-120 in this slice: sealbot pin and patch).
- Clean (checked, no finding): every pyproject dependency has an importer (pydantic 41 files, yaml 32,
  matplotlib 1, rich 3, scipy 1, torch 145); both pytest markers are used (integration 21 files, slow 7);
  pyright `exclude` and the paths its comments cite exist; every ruff per-file-ignore code still has hits in tests/;
  both `[workspace.dependencies]` are inherited (pyo3 by mantis-bridge, criterion by 4 crates), and all
  6 members inherit `rust-version`; .gitignore has no ignored-but-tracked file; LICENSE needs nothing.

## Findings

### S-A-DOCS-4-01 | DOC | C
subject: CLAUDE.md (Build & test "all 28 bench floors"; "## Laws digest (full text: …)"; Map "RULINGS.md … R23 onward"; "## Deliberately absent")
claim: Four statements in the operator's instruction file disagree with the tree: the floor count, where the full law text lives, the scope of RULINGS.md's canonical entries, and the "ONE exception" display surface.
evidence: (a) `grep -c '^\[floor\.' tools/bench_floors.toml` -> 23 (CLAUDE.md says "all 28"). (b) the digest header points at `docs/governance/archive/laws.md`; `grep -n LAW-10 docs/governance/archive/laws.md` -> "LAW-10 Threat-probe criterion …" (live wording), while `grep -n LAW-10 docs/governance/LAWS.md` -> "LAW-10 DELETED by R347(d)". LAWS.md's own head says archive/laws.md holds "the pre-R346 wording", and CLAUDE.md's first paragraph and rule R9 already name LAWS.md as the law text. The digest should point there. (c) `sed -n 1,5p docs/governance/RULINGS.md` -> "From R346 these entries are **canonical**"; CLAUDE.md says "the canonical ruling entries, R23 onward". (d) CLAUDE.md "Deliberately absent" calls tools/run_dashboard.py "The ONE exception", but `grep -n AMENDMENT docs/design/repo_design.md` -> "AMENDMENT — R352(g), VIEWER-1: the game viewer is ADMITTED" (tools/game_viewer.py, `make viewer`) and "AMENDMENT — R363, ANALYZER-1 … ADMITTED on the viewer's terms plus one loopback socket" (tools/position_analyzer.py, `make analyzer`).
Δlines: 0 (four rewordings; the viewer/analyzer admission adds about 2 lines and the fixes elsewhere give back about 2)
witness: NONE for semantics. Gate 10 (tools/ci_gates/check_tracked_refs.py) only checks that the cited paths are tracked, and they all are.
depends: —

### S-A-DOCS-4-02 | DOC | C
subject: rust-toolchain.toml (header comment)
claim: The comment cites `tools/bench_floors.toml:11` as the attestation and says "all 28 production bench floors". Line 11 is part of the R346(f) row-deletion note, the attestation is the `[provenance]` `rustc` key, and 23 rows remain.
evidence: `grep -n rustc tools/bench_floors.toml` -> 6 (comment), 11 ("the rustc/CPU attestation above still holds", deletion note), 17 `rustc = "rustc 1.97.1 (8bab26f4f 2026-07-14)"`; `grep -c '^\[floor\.' tools/bench_floors.toml` -> 23. A line-number citation in a comment is the derive-or-delete class (R192(e)).
Δlines: 0 (cite `[provenance].rustc`, drop the count)
witness: NONE (tests/tools/test_bench_floors.py checks the floors file, not this comment)
depends: —
note: lane C only because PZ-6 names rust-toolchain.toml with bench_floors.toml (LAW-09). The fix touches the comment, never the channel.

### S-A-DOCS-4-03 | DOC | A
subject: mise.toml (header comment, "a Rust bump invalidates all 28 bench floors")
claim: The count is stale: 23 floor rows exist.
evidence: `grep -c '^\[floor\.' tools/bench_floors.toml` -> 23; `git grep -n -E "\b28\b.{0,30}(bench|floor)"` -> CLAUDE.md, Makefile, mise.toml, rust-toolchain.toml (the Makefile is L1's slice).
Δlines: 0 (drop the number: "invalidates the bench floors")
witness: NONE
depends: —

### S-A-DOCS-4-04 | CONFIG | A
subject: .gitattributes rule `crates/mantis-encoding/src/manifests.toml -text`
claim: The rule matches no path. manifests.toml was removed by AUDIT-1 F-36, and the line-ending pin test no longer lists it.
evidence: `git ls-files | grep -c manifests.toml` -> 0; `grep -n -A2 "EMBEDDED_BYTE_SIGNIFICANT *=" tests/test_line_endings.py` -> only "crates/mantis-encoding/src/registry.toml"; `grep -n manifests tests/test_line_endings.py` -> nothing.
callers: this is an attribute rule, not code, so no importer channel applies. Its only possible consumer is git itself, and `git ls-files` shows no path for it to match. No gate or test reads the rule text: `git grep -n -F "manifests.toml"` over tests/ and tools/ -> 0 hits.
Δlines: -1 (the one rule line; the comment above it, which names registry.toml only, stays)
witness: tests/test_line_endings.py::test_gitattributes_marks_every_byte_significant_path (stays green: registry.toml keeps `-text`, and `git check-attr -a crates/mantis-encoding/src/registry.toml` -> text: unset)
depends: —

### S-A-DOCS-4-05 | DOC | A
subject: .gitattributes (comment block)
claim: Four comment facts disagree with the tree: the fixture-manifest row count, two line-number citations (one already wrong) and the shell-script count.
evidence: "(65 rows)" but `grep -c '^\[\[required\]\]' tests/fixtures/manifest.toml` -> 45. "golden_tests.rs:267 include_str!s" but `grep -n include_str crates/mantis-search/src/mcts/golden_tests.rs` -> line 75. "registry/mod.rs:25" is correct today but is a transcribed position (R192(e)). "All five carry `#!/usr/bin/env bash`" but `git ls-files '*.sh' | wc -l` -> 8.
Δlines: 0 (reword: cite files and symbols, drop the counts); combine with 12 on contact
witness: NONE (tests/test_line_endings.py pins the rules, not the comments)
depends: 04

### S-A-DOCS-4-06 | DOC | C
subject: vendor/pins.toml [pins.sealbot] comment
claim: The 3-line "(AUDIT-1 F-52: this cited `(:17, :24)` …)" parenthetical was spliced into the middle of a sentence. It separates "tools/vendor_fetch.sh reads only `url`, `sha` and `patch`, so the" from "checkout is sha-driven", and it repeats "tests/tools/test_vendor_pins_sealbot.py is what …" two lines before the original sentence says it.
evidence: `grep -n -E "AUDIT-1|transcribed position|is what holds" vendor/pins.toml` -> lines 15–17 inside the sentence that starts at line 13.
Δlines: -3 (the parenthetical)
witness: NONE (tests/tools/test_vendor_pins_sealbot.py reads the keys, not the comment)
depends: moot if 07 lands
note: lane C only because the block belongs to the sealbot pack (07); as a standalone edit it is mechanical.

### S-A-DOCS-4-07 | PACK | C
subject: vendor sealbot pin: vendor/pins.toml [pins.sealbot] plus vendor/patches/sealbot.patch (and, outside this slice, its consumer chain)
claim: Since R362(c) deleted the sealbot rung, no production path reaches the sealbot adapter. The pin, the patch, the build script and the adapter are held live only by tests and by the fetch/build targets that exist to feed them.
evidence: `git grep -n -w resolve_bot -- src tools` -> the only calls pass the literal "random" (src/mantis/eval/worker.py, src/mantis/diagnostics/acceptance_witness.py). `git grep -n -E "resolve_bot\([^\"']" -- src tools` -> only the def. src/mantis/eval/worker.py raises "no candidate sims for opponent kind … the sealbot rung was deleted". `git grep -n -i sealbot -- configs` -> 0. tools/strength_frontier.py docstring: "the old default (`sealbot_d5`) went with the rung (R362(c))". RULINGS.md R362 heading: "the sealbot rung DELETED with its five consumers in one commit".
callers:
- AST imports: `git grep -n -E "bots\.sealbot|_sealbot_mod" -- src tools` -> src/mantis/bots/resolve.py (the adapter branch) and src/mantis/bots/strix.py (`from mantis.bots.sealbot import find_vendor_root`, a shared helper that must MOVE, not die).
- entry points: none (CALLERS §1).
- `python -m` / Makefile / shell: Makefile `vendor.sealbot` -> tools/vendor_build_sealbot.sh; tools/vendor_fetch.sh iterates every `[pins.*]` generically (`grep -n "for name, spec in pins.items"`), so it has no sealbot-specific code.
- subprocess: tests/bots/test_sealbot_vendored.py and tests/tools/test_vendor_build_sealbot.py run the build script; tests/tools/test_vendor_fetch_idempotent.py and tests/tools/test_vendor_pins_sealbot.py read the pin (`_pins()["sealbot"]`).
- registry-by-name: src/mantis/bots/resolve.py::_KNOWN_KINDS and src/mantis/config/resolve/nsims.py::_KNOWN_OPPONENTS still list "sealbot".
- config keys: no config names a sealbot opponent (`git grep -i sealbot -- configs` -> 0).
- STATE procedures: CALLERS §5 names only `make vendor` / `make vendor.strix` on the box, never vendor.sealbot.
- gates: tests/test_meta_ci.py pins the Makefile target set, which includes vendor.sealbot; tools/ci_gates/tier_declaration.txt has sealbot rows.
- dashboard: tools/dashboard/{hero,strength,tier2,tier3}.py read sealbot rows from EXISTING event records, so historical display is a live reader of the NAME only, not of the pin.
Δlines: -120 in this slice (`awk` span of [pins.sealbot] = 22; `wc -l vendor/patches/sealbot.patch` = 98). Removing the pack also removes the files handed off below, which are not counted here.
witness: tests/bots/test_sealbot_{adapter,resolve,vendored}.py, tests/tools/test_vendor_{pins_sealbot,build_sealbot,fetch_idempotent}.py, tests/test_meta_ci.py (Makefile target set). This is a test-floor move (gate 3c) plus tier_declaration.txt STALE rows.
depends: 06
lane: C. PZ-6 names src/mantis/bots/sealbot.py::BUILD_ABSENT_MARKER (R324), R362 is the ruling that kept the adapter, and LAW-15's "reproducible fixed-depth bar" is phrased around a sealbot depth. The ARCHITECT decides whether a fixed-depth external instrument must stay buildable.

### S-A-DOCS-4-08 | CONFIG | C
subject: configs/run6.yaml, configs/run7.yaml, configs/run8.yaml (retired runs, still PRODUCTION census members)
claim: Three finished runs' configs remain in the production census. That keeps gate 7, gate 12 and every census pin armed against them, and it forces src/mantis/config/armed_aborts.py to carry owner text that exists only because "configs/run6.yaml is a finished run's record and mints null truthfully".
evidence: PZ-4 census -> production = run10, run6, run7, run8. By-name readers, from `git grep -l -F "<name>.yaml" -- src tools tests`:
- run6: src/mantis/config/armed_aborts.py (3 manifest rows' grounds or owner text), src/mantis/eval/pipeline.py (comment), tools/strength_frontier.py (docstring), 55 test files.
- run7: tools/select_balanced_book.py (procedure docstring), plus 3 tests including STATE's witness test tests/selfplay/test_qsigma_rescale_reaches_target.py.
- run8: 6 tests (run8 is also the box's parent stamp; STATE names `RETIRED_STAMP_PATHS`).
- run10: 0 by-name readers (census only).
Precedent: R367 deleted configs/run9.yaml (whitelisted DISSOLVED in gate 10).
Δlines: 0 if they move to EXEMPT or an undiscovered archive; -558 if deleted (`wc -l` of the three = 558), plus re-pointing about 60 test files from run6 to run10 or dev_example (lane B work behind the C decision).
witness: tests/config/test_config_census.py, gate 7 (validate_configs.py), gate 12 (preflight_mint.py --audit-only), and the pins listed above.
depends: —
lane: C (PZ-4 minted configs; R1; R322/R327 frozen-file grant). ARCHITECT question: must a finished run's config stay in the PRODUCTION census, or does the census mean "runs that can still launch"?

### S-A-DOCS-4-09 | DOC | A
subject: README.md (STATUS paragraph and the AUDIT-1 F-52 HTML comment)
claim: The STATUS paragraph says "the run6 identity is being minted and no production training run has completed", but STATE.md records run8 run to 45k and run10 minted (R367). The HTML comment below it is a 3-line narrative about the previous wrong sentence (R316(e)).
evidence: `grep -n -i -E "run6 identity|first full run" README.md` -> the STATUS lines; `grep -n -o -E "run8[^.]{0,60}45k" docs/governance/STATE.md` -> "run8 runs to 45k …", "parent run8@45k"; `grep -n -E "<!--|-->" README.md` -> the comment spans 3 lines.
Δlines: -3 (the comment; the STATUS rewrite is line-neutral and should point at STATE.md rather than restate the run, so it cannot go stale again)
witness: gate 10 (check_tracked_refs.py scans README.md path tokens; the rewrite must cite tracked paths only)
depends: —

### S-A-DOCS-4-10 | DOC | A
subject: Cargo.toml [workspace.package] MSRV comment (`rust-version = "1.87"` grounds)
claim: Floor (1) lists `mantis-bridge/src/{board.rs, graph_contract.rs}` plus two benches as the `is_multiple_of` sites. board.rs now has none, and the call spans 7 files and 9 sites. The "AUDIT-1 F-52" paragraph that corrected the previous miscount ("five call sites in four files") is itself wrong again. The comment's own last sentence names the durable form: "the census is `git grep is_multiple_of`".
evidence: `git grep -c is_multiple_of -- crates` -> graph_contract.rs 2, d6_lossless.rs 1, graph_build_bench.rs 1, queue_fuse_bench.rs 1, runner/game.rs 1, tests/common/mod.rs 1, replay_sym_counter.rs 2; board.rs absent.
Δlines: -5. `grep -n` places the file list on lines 23–25 and the F-52 paragraph on 30–32; keep one line naming the census command.
witness: NONE (clippy::incompatible_msrv guards the VALUE, not the comment)
depends: —

### S-A-DOCS-4-11 | CONFIG | B
subject: Cargo.toml [profile.profiling]
claim: No command in the tree builds this profile. It is named only by docs/design/repo_design.md ("Profiling builds: release + debug symbols (`profiling` profile)").
evidence: `git grep -n -E "profile[ =.]*profiling|--profile" -- ':!Cargo.toml' ':!docs/governance/archive' ':!docs/slim'` -> 0; `git grep -n -w -i profiling -- Makefile tools docs/design/repo_design.md docs/governance/STATE.md CLAUDE.md crates` -> repo_design only.
Δlines: -4 (the profile's 4 lines per `grep -n`), plus a repo_design amendment line
witness: NONE
depends: —
note: an operator convenience for LAW-09 hotspot work. Deleting it is a design-contract amendment (R9), so the likely verdict is KEEP. Recorded for completeness.

### S-A-DOCS-4-12 | DOC | C
subject: root config-file comment blocks (.gitattributes, .gitignore, mise.toml, rust-toolchain.toml, rustfmt.toml, Cargo.toml, pyproject.toml, vendor/pins.toml)
claim: 202 of the 371 lines in these 8 files are comments. Most are file-top banners or narrative blocks (WHY THIS FILE EXISTS, AUDIT/WPCLEAN/R-number histories), the class R316(e) bans, and gate 14's comment ratchet does not measure them.
evidence: `grep -c -E '^\s*#'` / `wc -l`: .gitattributes 41/51, .gitignore 17/36, mise.toml 22/24, rust-toolchain.toml 18/22, rustfmt.toml 10/12, Cargo.toml 20/54, pyproject.toml 41/121, vendor/pins.toml 33/51.
Δlines: upper bound -202 (`grep -c`). The real trim is a per-line judgement: load-bearing markers (registry.toml `-text` grounds, the panic="unwind" rule, the R348(a) cuda-extra rule) stay.
witness: NONE (tests/test_line_endings.py pins .gitattributes RULES; tests/test_bare_pytest_tier.py pins pyproject addopts, not comments)
depends: 02, 03, 05, 06, 10 (apply on contact with those)
lane: C. R316(e) says comments are trimmed "ON CONTACT, never as a cleanup pass". A slimming pass over these files is contact only where another finding already edits the file.

## DEFECTS
- none (every disagreement found is documentation; no rule or pin in the slice misbehaves)

## PARKED
- none

## HANDOFF
- L1 (Makefile): the Makefile comment "the 28 floors" (23 rows) and "eight"/"seven non-smoke" benches (6 `[[bench]]` targets per CALLERS §11). Also `vendor.sealbot` if 07 lands.
- L2 (tools top level): tools/select_balanced_book.py's procedure docstring runs `strength_frontier.py --config configs/run7.yaml` (depends on 08); tools/vendor_build_sealbot.sh (07).
- R3 (crates): crates/mantis-bridge/pyproject.toml and crates/mantis-encoding/Cargo.toml (`toml` dep comment) still say manifests.toml is embedded or parsed; crates/mantis-encoding/src/lib.rs names it as removed, which is correct.
- C3 (src/mantis/bots, config/resolve/nsims.py): sealbot adapter, resolve.py::_KNOWN_KINDS, nsims.py::_KNOWN_OPPONENTS (07); move bots/sealbot.py::find_vendor_root out before any deletion, because bots/strix.py imports it.
- C3 (src/mantis/config/armed_aborts.py): the "finished run's record" owner text depends on 08.
- T3/T6: sealbot tests (07); run6-pinned tests (08).

## Not covered
- Cargo.lock / uv.lock: generated. Checked only that every DECLARED dependency has an importer or a user. I did not audit transitive lock entries, and did not run `uv lock --check` or `cargo update` (the brief forbids lock-touching commands).
- Cargo.toml MSRV floor (2) ("highest locked `rust-version` is wasip2 1.87.0"): not verified, because the crates.io API is unreachable from here.
- Config KEY liveness (LAW-08): configs are `extra="forbid"`, so every key is a schema leaf; that bijection is tests/config/test_every_key_has_consumer*.py over the schema (C2's slice). I did not re-derive it per file.
- The ruff per-file-ignore check used `--isolated` with the selected codes; `tail` cut the top rows of the statistics (274 hits total), so I001/F401/UP006-class counts are inferred as non-zero, not read off.
- Whether gate 14's comment_lint scans .toml files (assumed not, from its .py/.rs description in CLAUDE.md; not opened).
- Per-crate Cargo.toml files (R1–R3), .github/ (L1).
