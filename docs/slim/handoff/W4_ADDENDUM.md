# W4 ADDENDUM — the tools wave (read after WAVE_BRIEF.md)

Tools slimming (R368(d)/(e)/(k)) plus the two gate-learning orders R368(g) names. A deletion of dead code is
proven by the touched tools' tests + `git grep` of every removed name over src tools tests crates docs (excl.
docs/slim, archives) + gates 10/15 green before and after; a WITNESS change needs its planted break (brief).

## The three named orders (the wave's core)
1. **Gate 14 learns the two classes R368(g) names** (defect 23: `comment_lint.py::_RULING` misses bare cites and
   never reads docstrings). Two NEW GATED measures, baselined into `tools/ci_gates/comment_length_floor.txt` at
   their measured values (down-only), each with a self-test arm that reds when the measure is neutered:
   - `ruling_cite_lines` — lines carrying a ruling/card/finding token, counted over own-line comments (.py/.rs),
     Rust `///`/`//!` docs, PYTHON DOCSTRINGS (the missing half), and the text-format comments below. The token
     set gains bare `\bR\d{2,3}\b`; ONE-DIGIT bare `R\d` stays OUT deliberately (gate 15's justification headers
     are required to carry `R8`; gating one-digit tokens would set two gates against each other — state this in
     the docstring). The old ungated `ruling_cite_comment_lines` (narrower regex, comments only) is REPLACED by
     the new measure: delete the old key and its floor row in the same commit (derive-or-delete; a measure that
     under-measures its class is misinformation).
   - `textfile_comment_excess_lines` — lines beyond two in own-line `#` comment runs over the text-format scoped
     files: `.sh/.yml/.yaml/.toml/.txt` under tools/, the root Makefile, and `.github/workflows/*.yml` (the
     narrative runs gate 14 never saw; S-A-TOOLS-1-07). tests/ is EXCLUDED: its text-format files are fixture
     registers whose rows W5 legitimately adds (manifest.toml re-pins), and a ratchet that reds a granted
     re-pin is a gate against a ruling.
   - Riding the same named edit (disclosed): `_rust_doc_excess`'s run-length loop folds into `_excess` behind a
     `cap` parameter (TOOLS-1-12) — one implementation of "lines beyond N in an own-line run" in the file the
     ruling opens. CLAUDE.md's gate-14 description and lint_gate.sh's ratchet echo line name the two new
     measures / R368(g) in the same commit.
2. **Gate 16 widens to ZERO across the tree** (R368(g)): scan EVERY tracked `.py` file (src/, tools/, tests/,
   crates/, docs/, root), module scope AND function scope; fix every site (≈208 measured: 4 real in src/
   [eval/pipeline.py ×2, eval/worker.py ×2], ~202 in tests/ function scope, plus tools/ already zero). Add
   `encoding="utf-8"` — the repo is UTF-8 by gate 17's own account. `os.open` is a FALSE POSITIVE under the
   method-keyed table (flags, not mode; no encoding concept): teach the scanner to skip it by receiver, with a
   self-test arm. Keep the byte-frozen-oracle EXEMPT (tests/tools/test_preflight_mint.py) — reword its stale
   grounds (the conftest text it cites is gone) and the stale "639 tracked text files" failure message
   (derive-or-delete) in the same commit. CLAUDE.md's gate-16 text (both the CI-gates list and the Code style
   bullet's parenthetical) says the widened rule in the same commit.
3. **The sealbot VENDOR side** (R368(e)): delete `tools/vendor_build_sealbot.sh`, `vendor/patches/sealbot.patch`,
   the `[pins.sealbot]` block in `vendor/pins.toml`, the Makefile `vendor.sealbot` target + `.PHONY` entry,
   `tests/tools/test_vendor_build_sealbot.py`, `tests/tools/test_vendor_pins_sealbot.py`; update
   tests/test_meta_ci.py's pinned target set; drop CLAUDE.md's `make vendor.sealbot` sentence (the vendor
   bullet keeps `make vendor` and gains `make vendor.strix` if the prose named it); remove the sealbot-build
   prose from docs/contracts/eval_instrument.md (version bump per that doc's convention, doc + tests in ONE
   commit). Gate 10: a scanned doc still citing a deleted path goes to DISSOLVED_PATHS in
   tools/ci_gates/check_tracked_refs.py ONLY for R368(e) deletions; prefer rewording non-register docs.

## Rows and lanes (LEDGER L15–L19 + the TOOLS lane-C re-lanes under R368(b))
- L15 TOOLS-2-15 (A): `LadderClient.challenges`, `::list_bots` (+ test-only `::cancel`) — no caller; delete with
  the test line; `tests/tools/test_ladder_client.py::test_endpoint_strings_live_only_in_the_client_module` stays
  green (it is the endpoint pin).
- L16 TOOLS-2-12: DONE in W3 (84a4eef0, the sha256 leg) — record only.
- L17 TOOLS-1-01 (A): the Makefile `lint.rust` comment's "seven non-smoke" and "28 floors" counts go
  (derive-or-delete: no re-transcribed tallies). The Makefile is not under tools/ci_gates.
- L18 TOOLS-1-14 (B): **KEEP by R368(k)** — record.
- L19 TOOLS-2-07 (B): the five `_package` by-path loaders (game_viewer, position_analyzer, run_dashboard,
  probe1, ladder_bot) + the two test copies (tests/tools/test_probe1.py, tests/tools/conftest.py) become ONE
  helper. Home: `src/mantis/util/` (a public function the five tools + the tests conftest import; no sys.path
  write — spec_from_file_location + submodule_search_locations, the R5-legal mechanism). Docstring states the
  R5 ground in one line.
- L19 TOOLS-2-08 (B): tools/viewer/hexlogic.py's HEX_AXES/WIN_LENGTH/owner/win_line re-implementation reads the
  engine exports instead (`mantis._engine.HEX_AXES`, `WIN_LENGTH`, `find_winning_line`); the 4 hexlogic tests
  and the helper/fixture go; reader.py gains the replay→Board path it needs.
- L19 TOOLS-2-09 (B): the viewer's inline board renderer goes; board.js (the declared fork's surviving side)
  is the one renderer, inlined the way analyzer/html.py does it.
- L19 TOOLS-2-10 (B): `viewer/reader.py::_SHARD_RE`/`::_shard_paths` read a new public
  `mantis.monitor.game_record.run_shard_paths` that `iter_run_games` also uses (net ≈ 0 in src, one owner of
  the filename convention).
- Re-lanes recorded in PROGRESS (do NOT implement): TOOLS-1-02..06, 08, 09, 10, 11, 16 stay C — the ground for
  each is R368(b)'s "The same holds for … tools/ci_gates/**" (no ruling names those edits; config_templates is
  the WAVE_BRIEF's do-not-edit) — EXCEPT the stale text inside comment_lint/encoding_io_gate, which rides the
  named edits above. TOOLS-1-13 KEEP (jscpd 0, divergent semantics). TOOLS-1-15 + 16: KEEP/stay C by R368(k)
  (the workflow is the operator's switch). TOOLS-1-NEW-2: RESOLVED by R368(b) itself (protection binds the
  invariant; the glob question is moot — record). TOOLS-2-01/02/03/04 KEEP (certify stage of a live pipeline /
  falsified.md-named instrument / operator's pre-R362 rendering / fixture provenance). TOOLS-2-13/-14/-16/-17
  stay C (PZ homes: checkpoints.py loader + build_net; pipeline/aggregate; the PZ-4 minter; contract cite) —
  TOOLS-2-13's probe1/swa site died with W3's residue; re-derive the site list at contact.

## Hard limits (beyond the WAVE_BRIEF's)
- tools/ci_gates/** is protected (R368(b)): edit ONLY what R368 names — comment_lint.py (the measure order),
  encoding_io_gate.py (the widening), check_tracked_refs.py (DISSOLVED_PATHS for (e) deletions). Any other
  ci_gates edit is a HALT.
- `tools/config_templates/**`, tools/mint_config.py: do not edit (TOOLS-1-06 stays C).
- The floors: comment_length_floor.txt gets the two new measures at their measured baseline ONLY in the
  measure commit; test_count_floor.txt tracks the collected count (sealbot deletes 2 test files: report the
  drop, the dispatcher folds it).
- Do not touch configs/*.yaml values, schema keys, defaults (R368(h)); run10 safety (R368(i)) applies to any
  src/mantis/{config,train,model} touch (the L19/L10 helper additions are util/monitor — run the resolved check
  anyway after the monitor edit).

## Checks (each commit)
- The touched tool's tests + every test that greps a removed name; `git grep -n <name> -- src tools tests crates docs`.
- comment_lint GREEN (report measures), r8_header_gate on touched files, encoding_io_gate GREEN, gate 10, gate 12
  `--audit-only` (the sealbot Makefile edit moves the meta_ci pin), `ruff check .`, pyright if src/tools touched.
- For the gate edits: the gates' OWN self-tests (`comment_lint.py --self-test`, `encoding_io_gate.py --self-test`)
  red under a planted neutering (delete a measure arm / re-scope the scan) and green after — quote the red line.
- Collected count after each commit that adds/deletes tests.

## Report additions
- Per commit: rows, Δlines net, tests run, collected count, run10 MATCH where applicable.
- The re-lane table (ID | old | new | ground) for every lane-C TOOLS row.
- The measure baselines: the two new measures' measured values and the floor rows written.
