# W6 ADDENDUM — the docs + configs wave (read after WAVE_BRIEF.md)

Docs and configs, LEDGER L35–L38, the DOCS rows R368 re-lanes, 01_DEFECTS 40–51's stale text, and the
four named orders: R368(e)'s CARDS closed rows, RULINGS_ACTIVE.md and configs/run6–8, and R368(f)'s STATE
rewrite. Verified inventory: a read-only scout at the W5 tip (2026-09-24, dispatcher session). Verify each
row at contact; do not trust it blindly. Work the legs in the order below, one commit per independent file
or class, and integrate after each leg.

## Leg order

1. **Re-point every by-name config reader FIRST** (R368(e): "tests binding them re-point to the census").
   `git grep -n 'run[678]\.yaml' -- src tools tests` finds about 95 test files, 2 tools and 2 src files.
   Makefile and crates have none. Rules:
   - Pick the set by what the law covers (REVIEW-W5 #3):
     - A law over EVERY committed config (a sweep, an absence oracle, a pairing) reads
       `mantis.config.census.discovered_config_paths(repo_root)`, gate 7's own enumeration, which
       includes the exempt dev_example and smoke configs.
     - Only a production-only law reads `production_configs(repo_root)`.
     - Narrowing a sweep from the first set to the second is a witness narrowing (R368(c)).
     - A test that stands in for "a production config" parametrizes over the census, or takes the member it
       needs and says why. Precedents: 15c2b6e8, e4018c6a, 4300b691, c79b4e79.
     - No hard count of configs: W6's own deletion must not red a count.
   - A test that needs ONE concrete production config may bind `configs/run10.yaml` only if no census form
     expresses the assertion. Name the reason in the test. R367(a): no run-named symbol.
   - The run-named symbols go in the same commit (defect 50):
     - `RUN5 = …run6.yaml` in tests/config/test_armed_abort_cadence.py:44
     - `RUN5` in tests/config/test_config_discovery_authority.py. Keep its synthetic discovery paths.
     - `_RUN5` in tests/eval/test_deploy_matched_hparam_coincidence.py:22. This is a PZ-1 eval test:
       re-point the import or constant only; its assertions stay byte-unchanged.
     - `_PRODUCTION` in tests/train/test_monitor_liveness_arming.py:48
     - `RUN5 =` in tests/tools/test_preflight_mint_process.py:50. This is an R43/R310 frozen oracle.
       Re-point the path only, or HALT the row if the freeze pins the bytes.
   - Two-config comparisons need their own re-point: tests/selfplay/test_qsigma_rescale_reaches_target.py
     compares run7 with run8.
   - src and tools prose: src/mantis/config/armed_aborts.py:585/599/617 owner text,
     src/mantis/eval/pipeline.py:484 comment, tools/select_balanced_book.py:52/77 `BOX_PROCEDURE`,
     tools/strength_frontier.py:81 docstring, and tools/probe1's `--run-id` default "run8". Reword each
     generically; never name another run.
   - A PZ-1 pinning test is re-pointed by import or path only. If an assertion depends on a run6–8 VALUE
     and has no census form, that is a HALT for the row: stop it and record it.
2. **Delete configs/run6.yaml, run7.yaml and run8.yaml** once `git grep` over src, tools and tests is
   empty. Then run:
   - gate 7: `tools/ci_gates/validate_configs.py`
   - gate 12: `preflight_mint.py --audit-only`
   - `tests/config/test_config_census.py` and `test_config_discovery_authority`
   - run10 MATCH
   Gate 10 scans STATE.md and CARDS.md: fix their citations of the three paths in THIS commit, or add
   the paths to check_tracked_refs.py's `DISSOLVED_PATHS`. That is the self-expiring mechanism, and
   tools/ci_gates/** may take ONLY that row, as W4's addendum permitted. Leave docs/audits/** and
   docs/design/** history alone: gate 10 does not scan them.
3. **Retire docs/governance/archive/RULINGS_ACTIVE.md** (R368(e)):
   - First carry F-816-34/35/36's text into CARDS.md. The archive is its only source today.
   - Swap CARDS' other ~26 archive-coordinate cites (`A:`/`R:` lines and the "Reading the identifiers"
     section) for the ruling id that the same paragraph already names.
   - Then `git rm`, and confirm gate 10 is green.
   - The register annotations (R271, R346(e)) already landed with R368. Add none.
4. **CARDS.md sweep.** Co-land it with leg 3 or run it right after; same file.
   - R368(e): remove the closed rows. The scout counts ~33 closed, landed, spent, discharged or folded
     blocks, about 313 lines: the four R346 rows, REVIEW-1, PROBE-1, PERF-3, the sealbot cards, STAMP-FLOOR,
     BOX-VOLUME, TIER-HOST and others. Re-verify each card's state before cutting.
   - DOCS-1-04: repair the in-source-only markers section. Six of its seven listed markers are gone, and
     eleven real markers are missing.
   - DOCS-1-05: fix the headline drift.
     - CARD-OC7-OVERRUN still reads "BLOCKING".
     - "What holds run6" describes a discharged hold, and its `MAX_CHILDREN_PER_NODE = 192` should read 1024.
     - CARD-RUN9-STOP-LAW cites the deleted run9.yaml.
     - The CARD-PYRIGHT-STRICT line cite has drifted.
   - REVIEW-W4 note 3: refresh CARD-STYLE-BACKLOG's measure list against
     `tools/ci_gates/comment_length_floor.txt`. The seven gated measures are named there;
     `ruling_cite_comment_lines` is retired.
   - Defect 48: CARD-MINPIN's "the parity pin landed" is false. `aggregate_cluster_values_min` has zero
     hits, and both registry rows read `value_pool = "none"`. Repair the claim.
   - Defect 51: add CARD-MECHANISM-SWEEP's missing members, `diagnostics/worker_sweep.py::_sha256` and the
     `_Trainer` copies. Re-grep first: W5 folded most trainer doubles. Also record W5's residue on that card
     (see below).
5. **The mechanical batch.** Commit per file. All files are distinct and nothing in them has a witness
   beyond gates 10, 13 and 15.
   - CLAUDE.md:
     - The Comments bullet becomes R368(g)'s ONE rule: a comment or docstring states what the code cannot,
       in one line, more only for an invariant; no ruling, card or finding numbers except carve-out
       markers; one sanctioned pass over cites and narrative runs, which gate 14 now measures. The three
       old rules it names are REPLACED.
     - Remove the "28 bench floors" count (AQ-FLOORS28 / defect 46: derive at point of use; the true
       count is 23 `[floor.*]` tables).
     - The Laws digest points at docs/governance/LAWS.md.
     - "Deliberately absent" admits game_viewer and the analyzer alongside run_dashboard, per their
       amendments.
   - Remove the same count from mise.toml:22 and rust-toolchain.toml:6. Replace the latter's stale
     `bench_floors.toml:11` pin with `[provenance].rustc`.
   - tools/ci_gates/run_all.sh's "seven of eight" bench text is tools/ci_gates/**. Leave it C: no ruling
     names it.
   - Defect 43 / DOCS-3-04: repo_design §3 through an R9 amendment commit. GnnArchV2 has no base class,
     the dispatch order is defensive, and the third kind, GnnArchV2SoftPolicy, is named.
   - DOCS-4-11 (lane B): Cargo.toml's `[profile.profiling]`, which nothing builds. Delete it and its two
     repo_design mentions, or keep it with the ground recorded.
   - DOCS-3-01 (L35): delete docs/design/measurements/SHAKEDOWN7G_2026-09-14.md. It has no citer.
   - DOCS-4-04/-05 (L36/L37), .gitattributes:
     - Drop the dead manifests.toml rule.
     - The fixture-row count is derived, not transcribed.
     - Correct the `golden_tests.rs:267` pin.
     - The shell-script count is derived.
   - DOCS-3-06: analyzer_design.md's Status line.
   - DOCS-4-09: README's STATUS paragraph, and drop the AUDIT-1 F-52 HTML comment.
   - DOCS-4-10: Cargo.toml's MSRV `is_multiple_of` site list goes; keep its grep census line.
   - Register annotations, APPEND-ONLY and never an edit (R9):
     - falsified.md F-43: its line cites into run.py, test_pool_drain_parity.py and test_selfplay_census.py
       drifted, and W5 re-based the drain suite. Add a foot annotation naming the current symbols; prefer
       symbols over line numbers.
     - falsified.md F-04: the absent min pin (defect 48).
     - Defect 47: RULINGS.md's coverage note ("322 numbers", and "Four entries" that names five). Optional,
       annotation only.
6. **STATE.md rewrite LAST** (R368(f)), because it names the commits legs 1–5 leave:
   - Rewrite to current facts. Keep the header, the current-phase line and one provenance line.
   - Each dropped paragraph class names where it now lives: a ruling, a measurement record or a commit.
   - Box-local specifics leave, sanitized forward (R281(b), defect 42's broader class). History is not
     rewritten.
   - Fix the present-tense drift: "run7 is LIVE", the stated delta counts, and the stated comment-floor
     numbers. Derive them, never transcribe them.
   - Gate 17 (rule-7 patterns) must stay green on the new text.

## Hard limits (beyond WAVE_BRIEF's)
- configs/run10.yaml and dev_example/smoke configs are NOT edited (R368(h)/(i)). run6–8 are deleted, not
  edited.
- RULINGS.md, LAWS.md and falsified.md correct only by appended annotation. docs/design/archive/** is
  never edited (R355(f)); DOCS-3-02 stays C. docs/audits/** are point-in-time records: leave them.
- tools/ci_gates/**: ONLY a `DISSOLVED_PATHS` row in check_tracked_refs.py if a deleted path is still
  cited by a scanned doc (the addendum's permission), plus floor folds. NOTHING else.
- Gate 13: docs/contracts/run_config_schema.md's version table must keep parsing, with header version =
  max(row). DOCS-1-11..-14 (contract-doc drifts) are NOT ordered here and stay C unless a leg already
  touches the doc.

## Stays C / done (recorded grounds)
- Stay C: DOCS-1-06..-14 except as folded above; DOCS-3-02 (R355(f)); DOCS-3-07, DOCS-4-12 (on contact
  only); defect 41 (BYTE-FROZEN claims, 10 hits: TESTS scope, on contact); defect 49 (Rust header, on
  contact).
- DONE earlier: DOCS-4-06/-07 (W4 60755d15); DOCS-3-03's header half; defect 40 (0 hits after W5); defect
  44; defect 45's gate_01 pin.
- DOCS-1-08 (COMMS_STYLE.md) and DOCS-1-09 (AUDIT-3) have zero citers and no R368 order. Optional deletes,
  re-laneable to A under (b).

## W5 residue the docs wave carries onto cards (not code legs)
- Carded, from W5:
  - The graph drain goldens have no committed generator. The capture script was a scratch file; if kept,
    it should be a tools/ generator.
  - `ResolvedPoolEncoding.board_size`/`trunk_size`/`n_kept_planes` have no src reader but are golden-pinned.
  - A supervisor poll-sleep plant hangs `tests/monitor/test_supervisor.py` instead of redding it.
  - `tests/_determinism.py`'s context has no CPU witness (REVIEW-W5 #13).
  - REVIEW-W5 #11's small residues: the `disk_guard.keep_all` tombstone could become a "deliberately
    absent" row, and RegimeKey's `==` leg is not re-asserted.
  - The drawrate trio's private fakes are PZ-1, with differing semantics.
  - `test_coordinator_knobs_wiring._real_graph_ring` duplicates `filled_hexg`.
  - L-STYLE-04 (subprocess `text=True` without `encoding=`, 79 sites, needs a gate-16 widening ruling).
  - L-STYLE-10 (function-scope imports, on contact).
- For the OPERATOR, not a leg: the armed_aborts.py `terminal_eval_broken` RESIDUAL text (a PZ-6 manifest
  row) cites the non-existent tests/config/test_minted_config_remint.py. The live holder is
  tests/test_run_eval_enabled_authority.py::test_every_production_config_declares_eval_enabled_true
  (c79b4e79). A re-point needs a ruling that names the row.
- Pre-existing and W6's: `test_book_geometry_pairing.py::test_run6_is_not_one_of_them` reds when run6.yaml
  goes. Re-express it as a mechanism, or delete it with the file.
- Unverifiable Δ0 NEW rows: TESTS-3-NEW-2, L-SEAM-NEW-2/-NEW-3, L-STYLE-NEW-2. Their text lived in a
  removed scratchpad; record them as closed-unverifiable in PROGRESS.

## Checks (each commit)
Gates 10, 13, 15 and 17 on every docs commit, and gate 14 (comment_lint GREEN; floors only fall; CLAUDE.md
is unmeasured). On config and test commits add: the touched tests; `pytest --collect-only -q -m ''` (the
dispatcher folds the count); gates 7 and 12; run10 MATCH. Doc-only commits need no pytest.

## Report additions
Per commit: rows, Δlines net, gates run. The re-lane table (ID | old | new | ground) for every DOCS lane-C
row the wave touches.
