# S-A-TOOLS-2 — L2: tools/ top-level files, analyzer, dashboard, ladder (incl. vps/), probe1, viewer

scope: `git ls-files tools` minus tools/ci_gates/ and tools/config_templates/ (76 files, 10 546 lines by `wc -l`) at 69e1532.
method: per-file citation census (`git grep -l -F <basename>` over docs/tests/src/crates/Makefile/.github), STATE.md
code-span extraction, an intra-package import map, an AST def census (`git grep -o -w` per def, own/test/other split, in
the scratchpad as sA2/defs*.py), `uvx vulture` at confidence 0 over slice + src + tests, `npx jscpd@4` over the slice,
AST span derivation for every Δ, targeted collection (`pytest --collect-only -m ''` on test_probe1 and
test_audit_bootstrap_corpus, 4 + 38 tests; the audit's 38 run green without torch), gate 15 run (green, 0 stale).

## Summary

| class | A | B | C |
|---|---|---|---|
| ONE-SHOT | – | – | 3 (01, 02, 04) |
| DEAD | 1 (15) | – | 1 (03) |
| DUP | 1 (12) | 5 (07, 08, 09, 10, 11) | 3 (13, 14, 16) |
| DOC | – | – | 3 (05, 06, 18) |
| SIMPLIFY | – | – | 1 (17) |

18 findings. Top 3 by Δlines: **01** audit_bootstrap_corpus −1 793 (tool + tests); **02** probe1 spent subcommands −301
(−268 tools, −33 tests); **04** gen_mctx_parity_fixtures −227 (verdict KEEP; listed because the hint asked).
Verified live and NOT proposed (each resolved at HEAD): run_dashboard + dashboard/ (STATE dashboard refresh, R333(d)),
game_viewer + viewer/ (STATE, R352(g)), position_analyzer + analyzer/ (Makefile, R363), ladder_bot + ladder/ + vps/
(STATE, open CARD-LADDER-RUNG), mirror_pull (STATE puller), strix_follower + strength_frontier + strix_driver (STATE
follower; src/mantis/bots/strix.py), bench_server (STATE, R367(e) admission), config_diff + mint_config (mint, PZ-4),
check_import_dag (gate 9), hardcode_scan (loaded by src/mantis/encoding/audit.py), mint_opening_book (LAW-15 book
provenance, manifest.toml), select_balanced_book (book_v2 still carded), worker_sweep_plan.toml (R309(f)/(g); consumer
src/mantis/diagnostics/worker_sweep.py), vendor_*.sh (Makefile), bench_floors.toml (PZ-6).

## Findings

### S-A-TOOLS-2-01 | ONE-SHOT | C
subject: tools/audit_bootstrap_corpus.py (+ tests/tools/test_audit_bootstrap_corpus.py, tests/diagnostics/test_tool_absences.py::test_the_convention_audit_does_not_certify_over_zero_games)
claim: the R247 corpus-intake audit is spent — R279(b) CERTIFIED the corpus off its run ("exit 0, sha match, 8698/8698 distinct, winner convention replay-verified") and no open card, STATE line or procedure calls it again.
evidence: `git grep -n -F audit_bootstrap_corpus` -> only docs/governance/archive/{RULINGS_ACTIVE,rulings_register}.md (history), its own test, test_tool_absences.py A09, and self-strings; `grep -n -i corpus docs/governance/{CARDS,STATE}.md` -> no corpus card (one unrelated hit); RULINGS.md R279 "(b) The corpus is now CERTIFIED GROUNDS".
callers: AST imports: tool has no importers; `git grep -n audit_bootstrap_corpus -- src tools` -> self only.
  entry points: pyproject has no [project.scripts] (CALLERS §1) -> none.
  python -m / script-path: `git grep -n "audit_bootstrap_corpus" -- Makefile '*.sh' docs .github` -> archive only; STATE.md 0 hits.
  subprocess: `git grep -n -E "subprocess|Popen" -- tests | grep audit_bootstrap` -> none.
  importlib/by-path: tests/tools/test_audit_bootstrap_corpus.py + tests/diagnostics/test_tool_absences.py (spec_from_file_location) — the tests ARE the only callers.
  conftest: tests/tools/conftest.py loaders name dashboard/viewer/ladder/analyzer/puller only.
  pyo3: n/a (the tool imports no mantis module by design). config keys: none. gates: not in run_all.sh / lint_gate.sh / test_count_gate.sh.
  STATE procedures: CALLERS §5 table has no corpus row; STATE.md grep 0.
Δlines: −1 793 (`wc -l` tool 1 124 + test file 654; A09 test AST span 15). Collected tests −39 (38 + 1); 5 112 → 5 073 stays above the 4 862 floor, still a test-floor move.
witness: none reds except the deleted tests themselves; gate 10 unaffected (only archive/ cites the path, and archive/ is outside gate 10's non-recursive glob).
depends: — (T6 owns test_tool_absences.py; T3 owns the test file). Lane C: the tool is the R247/R279 certification instrument, named by path in the archive registers; deletion loses the re-audit ability for a new corpus drop.

### S-A-TOOLS-2-02 | ONE-SHOT | C
subject: tools/probe1/ subcommands `decompose`, `gap`, `proofs`, `swa` (proofs.py, swa.py, rings.py::{_share,decompose,decompose_rings,full_arm_rows,target_argmax,reconstructed}, readings.py::gap_table, cli.py::{cmd_decompose,cmd_gap,cmd_proofs,cmd_swa} + their parser blocks)
claim: PROBE-1 is READ and CARD-PROBE-1 SPENT (STATE, CARDS), and four of its six subcommands now have successors or are dead; only `netread` (named by RUN10_PREREG as the box's KL(prior‖target) reading on the 36k rings) and `spread` (CARD-ARCH-D6's symmetry-spread witness) remain procedures.
evidence: RUN10_PREREG_2026-09-21.md KL row: "the box reads it off the 36k rings with `tools/probe1.py netread`"; decompose -> src/mantis/diagnostics/ring_audit.py::per_mr_rows ("PROBE-1's decomposer as audit rows", prereg row) computes the same `one_hot_share_full_mr<k>` + `tail_only_full`; gap -> the in-run held-out witness src/mantis/train/heldout.py (train.heldout_gap, R366(c)); proofs -> P-B2 DEAD, F-53 filed (STATE); swa -> the SWA net `run8swa_00051000_ae478dee.ckpt` BUILT (STATE), row 7 plays it through strength_frontier, not probe1.
callers: (subcommand-level) the only callers of cmd_decompose/gap/proofs/swa are cli.py's parser (`git grep -n -w -e cmd_gap -e cmd_proofs -e cmd_swa -e cmd_decompose` -> cli.py only); proofs.py/swa.py/decompose* have no importers outside cli.py and test_probe1.py (import map); docs cite them only in the measurement record PROBE1_2026-09-21.md (docs/design, gate-10 exempt).
Δlines: −268 tools (AST spans 105 + `wc -l` proofs.py 80 + swa.py 58 + parser blocks 25) and −33 tests (3 of test_probe1.py's 4 tests, AST spans).
witness: tests/tools/test_probe1.py (the kept netread/spread paths); gate 10 unaffected (STATE cites tools/probe1.py, tools/probe1/, tests/tools/test_probe1.py — all stay).
depends: — . Removing `gap` also removes the R10 `--run-id default="run8"` (DEFECTS). Lane C: R365(b)-ordered instrument, STATE/CARDS/RULINGS-named.

### S-A-TOOLS-2-03 | DEAD | C
subject: the retired sealbot-rung / ladder-file surface in tools/dashboard: ladder.py (whole), strength.py::{RUNG_RETIRED_NOTE,sealbot_readings_present,primary_rung,_ci}, tier2.py::{SEALBOT_TT_NOTE,strength}, reader.py::{Record.rungs,_load_ladder}, plus tier3.py::rounds' wr_sealbot columns / eval_channel_health / eval_rung_* tables, the `eval_rung_skipped` _WATCH entry and cli.py `--ladder-state`
claim: every input these read lost its producer with R362(c) (sealbot rung, eval_ladder_state.json, eval_channel_health, wr_sealbot, eval_rung_*); on any post-R362 record they only draw the RUNG_RETIRED_NOTE gap, so they live only for pre-R362 records.
evidence: `git grep -n -E "[\"']<name>[\"']" -- src` -> 0 for eval_channel_health, eval_rung_activated, eval_rung_graduated, eval_rung_skipped, eval_ladder_zero_game_round, wr_sealbot; strength.py's own comment "R362(c): the sealbot rung, its ladder file and `wr_sealbot` have no producer at HEAD"; contract v33 / run_config_schema.md: `mantis.eval.bt` and `mantis.eval.channel_health` went with them.
callers: AST: ladder_chart <- tier2.py only; rung_series/primary_rung <- hero.py, html.py, tier2/tier3 (in-package). python -m/script: Makefile `dashboard` passes `--ladder-state L` (L1). subprocess/importlib: run_dashboard.py loads dashboard.cli by name (whole package). conftest: tests/tools/conftest.py::dashboard fixture. STATE: dashboard refresh runs the whole page on the current (post-R362) run's record. config keys/pyo3/gates: none.
Δlines: ≤ −124 (`wc -l` ladder.py 64 + AST spans 60), before the hero.py/tier3.py/cli.py edits and the tests' ladder fixtures; dashboard/external.py::external_chart then stops being a near-copy of ladder_chart (jscpd: 6-line clone).
witness: tests/tools/test_run_dashboard.py, test_dashboard_strength.py, test_dashboard_reader.py (ladder fixtures) — they red until re-pointed.
depends: L1 (Makefile `dashboard` LADDER_STATE), T3 (dashboard tests). Lane C: R333(d)-admitted dashboard; keeping pre-R362 record rendering is an operator choice.

### S-A-TOOLS-2-04 | ONE-SHOT | C
subject: tools/gen_mctx_parity_fixtures.py
claim: a spent one-shot generator ("Run it ONCE: the fixture is the frozen artefact") — but it is the provenance of the sha-pinned mctx_parity_v1.json; verdict KEEP.
evidence: tests/fixtures/manifest.toml row `mctx_parity/mctx_parity_v1.json` comment "Minted by running google-deepmind/mctx itself (tools/gen_mctx_parity_fixtures.py)"; crates/mantis-search/tests/mctx_parity.rs (docstring + panic message) and crates/mantis-search/src/mcts/parity_tests.rs cite it; jax/mctx are not repo deps so nothing executes it.
callers: none executable (by design); cited by 3 files as provenance.
Δlines: −227 (`wc -l`) if deleted — NOT recommended: deletion loses the fixture's reproducibility and needs edits to the manifest comment (frozen tests/fixtures/**) and two Rust files.
witness: NONE (nothing runs it).
depends: —

### S-A-TOOLS-2-05 | DOC | C
subject: the "28 bench floors" claim in CLAUDE.md (Build & test), rust-toolchain.toml header, mise.toml comment, Makefile `lint.rust` comment
claim: tools/bench_floors.toml carries 23 floor rows, not 28 — R346(f) deleted five (its own header says so); rust-toolchain.toml also cites `tools/bench_floors.toml:11` for the rustc attestation, which is line 17.
evidence: `grep -c '^\[floor\.' tools/bench_floors.toml` -> 23; its header: "R346(f) DELETED five rows"; `git grep -n -E "all 28|28 floors|28 bench"` -> CLAUDE.md, Makefile, mise.toml, rust-toolchain.toml; `grep -n '^rustc' tools/bench_floors.toml` -> 17.
Δlines: 0 (text edits). Per R8/R192(e) derive-or-delete, dropping the count beats replacing it.
witness: NONE.
depends: — (Makefile is L1's; the root files are raised here at the dispatcher's request). Lane C: PZ-6 names bench_floors.toml + rust-toolchain.toml.

### S-A-TOOLS-2-06 | DOC | C
subject: CLAUDE.md "Deliberately absent" — "Display surfaces (web dashboard, viewer, TUI) … The ONE exception, admitted by R333(d) … tools/run_dashboard.py … no server"
claim: the tree holds two more admitted display surfaces: the game viewer (tools/game_viewer.py + tools/viewer/, repo_design AMENDMENT R352(g) VIEWER-1) and the position analyzer, a loopback ThreadingHTTPServer (tools/position_analyzer.py + tools/analyzer/, AMENDMENT R363 ANALYZER-1).
evidence: `git grep -n "AMENDMENT — R352(g)\|AMENDMENT — R363" docs/design/repo_design.md` -> both; `grep -n ThreadingHTTPServer tools/analyzer/serve.py` -> hit; Makefile has `viewer` and `analyzer` targets.
Δlines: 0 (text edit).
witness: NONE.
depends: — . Lane C: the operator's instruction file restating three rulings' scope.

### S-A-TOOLS-2-07 | DUP | B
subject: tools/{game_viewer,position_analyzer,run_dashboard,probe1}.py::_package and tools/ladder_bot.py::_load_ladder
claim: five copies of the same load-a-tools-package-by-path loader (spec_from_file_location + submodule_search_locations + sys.modules), differing only in the package name; a sixth and seventh live in tests (tests/tools/test_probe1.py::_package, tests/tools/conftest.py::load_tools_package).
evidence: `git grep -n submodule_search_locations` -> the 5 tools files + 2 tests files.
deliberate?: not a seam, oracle or twin — each docstring gives the same R5 reason (no sys.path write); the duplication is of the mechanism, not a second authority.
Δlines: ≈ −51 (AST spans 13+13+12+13+12 = 63, less one ~12-line shared helper; the shims' call lines stay).
witness: tests/tools/test_run_dashboard.py, test_game_viewer.py, test_analyzer_*.py, test_ladder_bot.py, test_probe1.py (all load through these).
depends: T3 (the two test copies). Lane B: where the helper lives is a design choice — tools/ is not a package, so it is either a src API (e.g. mantis.util) serving tools or a documented exception.

### S-A-TOOLS-2-08 | DUP | B
subject: tools/viewer/hexlogic.py (HEX_AXES, WIN_LENGTH, owner, win_line)
claim: transcribes engine facts the bridge already exports — `mantis._engine.HEX_AXES`, `WIN_LENGTH` (used that way by src/mantis/selfplay/{graph_collate,instrumentation}.py) and `Board.find_winning_line()` (used by tools/analyzer/position.py) — a second implementation of the six-in-a-row line (R10 "one implementation per thing").
evidence: `git grep -n "HEX_AXES\|WIN_LENGTH\|find_winning_line" -- src tools` -> hexlogic.py literals vs the _engine imports above; hexlogic's own comments cite `mantis_core::board::state::HEX_AXES` and `::WIN_LENGTH`.
deliberate?: not stated as an oracle or twin; the viewer already imports mantis (viewer/reader.py -> mantis.monitor.game_record), so the engine is available.
Δlines: −35 (`wc -l` hexlogic.py) −26 tests (AST spans of the 4 hexlogic tests + helper/fixture in tests/tools/test_game_viewer.py), plus a replay-to-Board in reader.py.
witness: tests/tools/test_game_viewer.py (index/win-line fields).
depends: T3. Lane B: the viewer takes an engine dependency and 4 tests go (test-floor move).

### S-A-TOOLS-2-09 | DUP | B
subject: tools/viewer/html.py::_HEAD/_SCRIPT board renderer vs tools/analyzer/web/board.js + analyzer.css
claim: board.js is a declared fork ("forked from tools/viewer/html.py") of the viewer's inline hex renderer (hexPts, X/Y, owner, the cell grid, stones, last-two rings, win cells, heat) and analyzer.css repeats the viewer's tokens and board classes; the viewer could inline board.js the way analyzer/html.py does.
evidence: board.js line 1 comment; derived shared text: 4 verbatim CSS lines and 9 board-render JS lines in viewer/html.py (scratchpad script over both files).
deliberate?: no — a fork, not a seam; test_analyzer_html.py pins token NAMES, not a second copy.
Δlines: ≈ −13 (4 CSS + 9 JS lines), less the scene/inline hook; the viewer's dashed "fast" stone needs a HexBoard scene field.
witness: tests/tools/test_game_viewer.py, test_analyzer_html.py.
depends: — . Lane B: board.js grows one scene field; which page owns the shared file.

### S-A-TOOLS-2-10 | DUP | B
subject: tools/viewer/reader.py::_SHARD_RE, ::_shard_paths
claim: re-implements src/mantis/monitor/game_record.py::_SHARD_RE and the (segment, hour) shard ordering inside ::iter_run_games (the filename convention's ONE owner).
evidence: both regexes are `^games_(?P<run>.+)_seg(?P<seg>\d+)_(?P<hour>\d{10})\.jsonl$`; both sort `(int(seg), hour, path)`.
deliberate?: no — the viewer needs the path list (per-shard data files), which game_record does not expose; that is the missing seam, not a twin.
Δlines: ≈ −9 (AST span 8 + the regex line), game_record gains a public `run_shard_paths` that iter_run_games also uses (≈ net 0 there).
witness: tests/tools/test_game_viewer.py; tests/monitor game_record tests.
depends: owner of src/mantis/monitor (a new public function). Lane B.

### S-A-TOOLS-2-11 | DUP | B
subject: tools/mirror_pull.py::closed_shards vs src/mantis/diagnostics/mirror_receipts.py::first_closed_shard
claim: two parsers of the same `games_<run>_index.jsonl` `shard_closed` rows (the writer is game_record.py); first_closed_shard is `closed_shards(...)[0]`.
evidence: `git grep -n shard_closed -- src tools` -> the writer + these two readers, same loop body (skip blank, json.loads, skip JSONDecodeError, `record == "shard_closed" and shard`).
deliberate?: no stated reason.
Δlines: ≈ −14 (first_closed_shard AST span 19 -> ~5-line wrapper; closed_shards 16 moves to src beside index_filename).
witness: tests/tools puller tests (tests/tools/conftest.py::_load_puller), tests/diagnostics mirror_receipts tests.
depends: owner of src/mantis/diagnostics. Lane B.

### S-A-TOOLS-2-12 | DUP | A
subject: tools/strix_follower.py::_sha256; tools/select_balanced_book.py::main's inline `hashlib.sha256(out.read_bytes())`
claim: both re-derive the file hash that src/mantis/util/hashing.py::sha256_file ("the one sha256_file the bundle, the receipts and the puller share") computes; mirror_pull.py and probe1/swa.py already use it.
evidence: `git grep -n -E "hashlib\.sha256|sha256_file" -- tools` (slice) -> these two plus audit_bootstrap_corpus (deliberately mantis-free, excluded).
deliberate?: no; identical hex digest (streamed vs whole-file read).
Δlines: −4 (strix_follower: 2-line def + 2 blank lines, `import hashlib` swapped for the util import; select_balanced_book net 0).
witness: tests/tools/test_strix_follower.py (sidecar checkpoint_sha256), tests/tools/test_select_balanced_book.py.
depends: —

### S-A-TOOLS-2-13 | DUP | C
subject: the stamp -> net rebuild in 6 tools: analyzer/engines.py::MantisEngine.__init__, bench_server.py::_load_net, ladder/backends.py (MantisBackend.__init__), probe1/nets.py::load_net, probe1/swa.py::average_checkpoints, strength_frontier.py::_snapshot_from_checkpoint
claim: each repeats "arch is None -> refuse; build_net(ck.metadata.arch); load deploy_state or model_state", with five different error types; analyzer and ladder additionally repeat the LocalInferenceEngine composition from the config's graph knobs.
evidence: `git grep -n -E "metadata\.arch is None|resolves no arch" -- tools` -> 6 sites (24 lines of check+build+load).
deliberate?: the learner-vs-deploy choice is deliberate per site (commented), the rest is not.
Δlines: ≈ −8 (24 site lines -> 6 call lines + a ~10-line helper taking `weights="deploy"|"learner"`).
witness: tests/tools/test_analyzer_engines.py, test_ladder_backends.py, test_bench_server.py, test_strength_frontier*.
depends: 02 (removes the swa.py site). Lane C: the helper's natural home is src/mantis/train/checkpoints.py or src/mantis/model/ (both PZ).

### S-A-TOOLS-2-14 | DUP | C
subject: tools/strength_frontier.py::base_round_spec, ::pair_readout (+ ::_dedupe_key, ::_candidate_outcome)
claim: base_round_spec is a second RoundSpec composition "as `mantis.run` composes it" parallel to src/mantis/eval/pipeline.py's (every new RoundSpec field must be threaded twice), and pair_readout re-derives the opening-pair grouping of src/mantis/eval/aggregate.py::pair_units with its own seat-qualified dedupe key.
evidence: `git grep -n "RoundSpec(" -- src tools` -> pipeline.py + strength_frontier.py only; docstring "The run's eval-pipeline seam as `mantis.run` composes it".
deliberate?: partly — the frontier reads game-record rows (`colors.candidate`), not result records (`candidate_color`), so the key differs by schema.
Δlines: AST spans base_round_spec 33, pair_readout 30, _dedupe_key 3, _candidate_outcome 6; a shared composer would net ≈ −20 (not derived beyond the spans).
witness: tests/tools/test_strength_frontier*.py; tests/eval/test_gate_pair_statistics.py.
depends: — . Lane C: pipeline.py and aggregate.py (gate pair statistics) are PZ.

### S-A-TOOLS-2-15 | DEAD | A
subject: tools/ladder/client.py::LadderClient.challenges, ::list_bots (and ::cancel, test-only)
claim: no caller: challenges (`GET /api/bot/challenges`) and list_bots (`GET /api/bots`) are never called; cancel is called only by tests/tools/test_ladder_client.py.
evidence: `git grep -n -E "\.(challenges|list_bots)\b|[\"'](challenges|list_bots)[\"']" -- src tools tests Makefile docs/contracts` -> 0; vulture (conf 0, slice+src+tests) flags exactly these two; `\.cancel\(` -> test_ladder_client.py only.
callers: AST imports: LadderClient used by session.py/ladder_bot.py, neither calls these. entry points: none. python -m/script: ladder_bot.py `--challenge` path uses `.challenge()`, not these. subprocess: none. importlib/getattr: `git grep -n getattr -- tools/ladder tools/ladder_bot.py` -> 0. conftest: tests/tools/conftest.py::ladder fixture loads the package only. pyo3/config/gates: n/a. STATE: ladder procedure runs ladder_bot.py (BUILD.md smoke uses `GET /api/bots?online=1` by hand, not the method).
Δlines: −6 (AST spans 3 + 3); cancel −3 more plus one test line (lane B if taken).
witness: NONE — `git grep -n -E '/api/bot/challenges"|/api/bots' -- tests` -> 0, so no endpoint pin lists these two.
depends: —

### S-A-TOOLS-2-16 | DUP | C
subject: tools/mint_config.py::main — the `--set` loop and the `--mint-row` loop
claim: both loops repeat the same malformed-check / partition / `_delta_line` under HeaderRenderError / assign / append block (jscpd: 9-line clone).
evidence: `npx jscpd@4 --min-lines 5 tools` -> mint_config.py 218-227 vs 244-253.
deliberate?: the parent resolution differs (template vs schema default); the tail does not.
Δlines: ≈ −6 (the shared tail into one helper).
witness: tests/config/test_mint_row.py, test_mint_and_diff.py, test_mint_header_roundtrip.py.
depends: — . Lane C: PZ-4 minter.

### S-A-TOOLS-2-17 | SIMPLIFY | C
subject: tools/run_dashboard.py::SIZE_CAP_BYTES, ::MIRROR_LAG_WARN_BUNDLES (re-exports)
claim: pure forwarders of dashboard/cli.py::SIZE_CAP_BYTES and dashboard/health.py::MIRROR_LAG_WARN_BUNDLES; SIZE_CAP_BYTES's only reader is a test, MIRROR_LAG_WARN_BUNDLES's is a test plus a docs/contracts/event_manifest.md citation by the shim's name.
evidence: `git grep -n "SIZE_CAP_BYTES\|MIRROR_LAG_WARN_BUNDLES"` -> tests/tools/test_run_dashboard.py (shim.*), event_manifest.md (`tools/run_dashboard.py::MIRROR_LAG_WARN_BUNDLES`), the definitions.
Δlines: −4 (two assignments, the `_health` import, the provenance comment).
witness: tests/tools/test_run_dashboard.py.
depends: T3; a docs/contracts edit (gate-13-adjacent contract doc). Lane C: contract-doc citation + R333(d).

### S-A-TOOLS-2-18 | DOC | C
subject: tools/ladder/vps/README.md
claim: its one line restates ladder_bot.py's module docstring and BUILD.md's pointers; it could fold into BUILD.md.
evidence: `wc -l` -> 1; `git grep -n -F "ladder/vps"` -> CARDS.md CARD-LADDER-RUNG ("carries the unit file, the CPU build recipe and the README"), LADDER_SHAKEDOWN doc.
Δlines: −1.
witness: NONE.
depends: — . Lane C: an open card names the README.

## DEFECTS
- tools/probe1/cli.py (`gap` parser): `--run-id` defaults to `"run8"` — a run name in a tool (R10) and a code-side default; goes away with 02.
- tools/select_balanced_book.py::BOX_PROCEDURE hard-codes `configs/run7.yaml` in the tool's run-book text (R10: no runN in a tool).
- tools/hardcode_scan.py::_registry_targets asks RegistrySpec for `n_chain_planes`, which it lacks (`hasattr` false at HEAD); `getattr(..., None)` hides the dead field name.
- tools/dashboard/tier3.py::rounds labels `wr_sealbot` as the round's "wr" column; on a post-R362 record the column is always "—" while the round's own GSPRT/gate fields are not shown there (part of 03).

## PARKED
- tools/strix_follower.py::_sha256 reads the whole checkpoint into memory (`read_bytes`) per sidecar; sha256_file streams (fixed as a side effect of 12).

## HANDOFF
- L1: Makefile `lint.rust` comment says "the seven non-smoke `[[bench]]` targets" — 6 `[[bench]]` exist (5 non-smoke; `git grep -c "\[\[bench\]\]" -- 'crates/*/Cargo.toml'` sums to 6); the same comment carries "28 floors" (05). Makefile `dashboard` LADDER_STATE rides 03.
- T3: tests/tools/test_probe1.py::_package is another copy of the 07 loader; tests for 01/02/03/08/15/17 live there.
- T6: tests/diagnostics/test_tool_absences.py A09 depends on 01.
- src slices (diagnostics, train/coordinator, eval): private names with tools consumers — mantis.diagnostics.ring_reader::_read_header and mantis.train.coordinator.dispatch::_build_graph_parts (probe1/nets.py), mantis.eval.worker::_graph_expand_fn (analyzer/engines.py), mantis.bots.strix::_pin (strix_follower.py); a private rename reds tools, not the owning module's tests.

## Not covered
- tools/analyzer/web/analyzer.js and the dashboard panels beyond event-name producer checks (no per-field producer audit of every panel).
- hardcode_scan.py's rule/allowlist tables, line by line (only the target list was checked).
- tools/strix_driver.py internals (runs inside the vendored strix venv, which is not built here).
- Nothing torch-dependent was run: probe1/analyzer/ladder-mantis/bench_server paths were checked by grep and AST only; only the torch-free audit tests were executed.
- The tools themselves were not run end to end.
