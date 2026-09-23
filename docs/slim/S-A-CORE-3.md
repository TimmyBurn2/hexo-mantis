# S-A-CORE-3 — slice C3

scope: src/mantis/eval/**, arena/**, bots/**, diagnostics/**, monitor/**, util/**, deploy/** (59 files, 12 167 lines at 69e1532);
method: AST import graph of every slice module (a throwaway script in the scratchpad: from-imports, relative imports, `import a.b`) over src/tools/tests;
top-level symbol census (every def, class and assign, counted by word over the whole tracked tree, bucketed src/tools/tests/docs); class-method census
(attribute and string tokens); `uvx vulture src/mantis tools [tests] --min-confidence 60` (leads only); `git grep -n -w` over the whole tree for every
DEAD claim; CALLERS.md §1–§11 and PZ.md read and ticked. torch is not importable here, so this is static analysis only and runs no probes.
Torch-free import probe: `.venv/bin/python -c "import <mod>"` per module.

## Summary

19 findings. By class: DEAD 9 · SIMPLIFY 3 · PACK 3 · DOC 2 · DUP 2. By lane: A 3 · B 5 · C 11.
Top 3 by Δlines:
1. S-A-CORE-3-11 sealbot adapter PACK (lane C, needs an operator ruling): −374 src (bots/sealbot.py) −46 resolver lines, −1 504 test lines.
2. S-A-CORE-3-01 package `__init__` re-export blocks with no importers (lane B): −141.
3. S-A-CORE-3-05 test-only coordinate helpers (lane B): −42 src, −17 collected test items.

No ONE-SHOT found under diagnostics/: all 10 modules have a live caller (S-A-CORE-3-18).

## Findings

### S-A-CORE-3-01 | SIMPLIFY | B
subject: src/mantis/arena/__init__.py, src/mantis/eval/__init__.py, src/mantis/monitor/__init__.py (the re-export blocks)
claim: No file imports any re-exported name through these three packages. Every consumer imports the submodule directly. The blocks' only effect is an import side effect: `arena/__init__` → `deploy_head` → `util.device` pulls torch into modules that are otherwise torch-free.
evidence: `git grep -n -E "from mantis\.(eval|arena|monitor) import"` → only submodule imports (`from mantis.eval import worker|pipeline|snapshot`, `from mantis.monitor import supervise`), and none of mantis.arena; an attribute scan `git grep -ohE "mantis\.(eval|arena|monitor|bots)\.[A-Za-z_]+"` over py/md/toml/yaml/sh → every non-submodule hit is `mantis.eval.{ladder,bt,channel_health}` (deleted-module prose) or `mantis.monitor.__path__/__name__` (test_monitor_census walk); probe `.venv/bin/python -c "import mantis.arena.regime"` / `mantis.eval.sequential` → `No module named 'torch'` via `arena/__init__.py` → `deploy_head.py` → `util/device.py`.
callers: AST imports: the scratchpad import-graph script → mantis.arena 0, mantis.eval 0, mantis.monitor 0 package-level name importers · entry points: none (CALLERS §1) · python -m: none of the packages · subprocess/`-c` strings: `git grep -E "[\"'](import mantis|from mantis[a-zA-Z_.]* import)" tests` → only `from mantis.monitor import supervise` (a submodule) · importlib/getattr: monitor/manifest.py resolves `mantis.monitor.{heartbeat,sink}` (submodules) · conftest: none · pyo3: n/a · config keys: n/a · gates: gate 9 check_import_dag counts only edges, and removing them cannot create a cycle · STATE: none.
Δlines: −141 (non-docstring lines derived by AST: arena 29, eval 54, monitor 58). The monitor docstring (HEADLESS ONLY) stays.
witness: tests/monitor/test_monitor_census.py::test_monitor_imports_without_torch_subprocess (still imports mantis.monitor); gate 9; the full default tier (import-order side effects).
depends: — (bots/__init__.py is excluded: tests/bots/test_protocol.py imports `BotProtocol, RandomBot, RungUnresolvable, resolve_bot` from the package)

### S-A-CORE-3-02 | SIMPLIFY | C
subject: src/mantis/eval/errors.py (re-exports of BookError, MixedRegimeError, RungUnresolvable; the "one import surface" docstring)
claim: Only eval/__init__.py consumes the three foreign-error re-exports. Every other raiser and catcher imports them from arena.books, arena.regime or bots.protocol.
evidence: `git grep -n -A3 "from mantis.eval.errors import" -- src tools tests | grep -E "BookError|MixedRegimeError|RungUnresolvable"` → only src/mantis/eval/__init__.py; `git grep -n "errors\.(BookError|MixedRegimeError|RungUnresolvable)"` → 0.
Δlines: −8 (3 import lines, 3 `__all__` entries, 2 docstring lines; counted from the file).
witness: the default tier.
depends: S-A-CORE-3-01. Lane C because eval/errors.py is in the PZ glob (PZ-6 EvalBrokenReason). The edit leaves EvalBrokenReason unchanged.

### S-A-CORE-3-03 | DEAD | C
subject: src/mantis/eval/errors.py::EvalBrokenError
claim: Nothing raises, catches, tests or documents it. EvalBrokenReason is the live authority.
evidence: `git grep -n -w EvalBrokenError` → 4 hits, all its def plus the `__all__`/`__init__` re-exports; string form `git grep -nE "[\"']EvalBrokenError[\"']"` → 2 (both `__all__`).
callers: AST imports: only eval/__init__.py (re-export, S-01) · entry points: none · python -m: n/a (a class) · subprocess: none (the whole-tree -w grep above covers string-embedded code) · importlib/getattr: none (the string grep) · conftest: none · pyo3: n/a · config keys: none · gate tool paths: none · STATE/docs: `git grep -w EvalBrokenError -- docs` → none outside docs/governance/archive.
Δlines: −2 (AST span of the class) and −2 export entries.
witness: NONE
depends: S-A-CORE-3-01 (it is re-exported there). Lane C because of the PZ glob on errors.py.

### S-A-CORE-3-04 | DEAD | C
subject: src/mantis/diagnostics/worker_sweep.py::RULED_DETERMINISM_BAND_PCT + tests/diagnostics/test_worker_sweep_determinism.py::test_the_bands_constant_is_SUPERSEDED_but_still_pinned_for_history
claim: The constant says of itself that it "gates nothing now", and it is "kept only because old reports cite it". No tracked report cites it. Its one test asserts the constant equals its own literal (`== 1.0`), so it pins nothing.
evidence: `git grep -n -w RULED_DETERMINISM_BAND_PCT` → the def plus that test, and nothing in docs/ at all.
callers: AST: the test only · entry points: none · python -m: the module's own CLI never reads it (`grep -n RULED_DETERMINISM src/mantis/diagnostics/worker_sweep.py` → def only) · subprocess: none · getattr/string: `git grep -nE "[\"']RULED_DETERMINISM_BAND_PCT[\"']"` → 0 · conftest: none · pyo3: n/a · config keys: not a key (tools/worker_sweep_plan.toml does not name it, per the whole-tree grep) · gates: none · STATE: none.
Δlines: −3 src (1 line + 2 comment lines), −4 test (one test function). Gate 3c floor: −1 collected item (test-floor move).
witness: NONE (the only test is the tautology itself)
depends: —. Lane C: worker_sweep is ruling-named (R308 orders "WORKER-SWEEP as PHASE W"; R338 ratifies its halt).

### S-A-CORE-3-05 | DEAD | B
subject: src/mantis/util/coordinates.py::flat_to_axial, ::axial_to_flat, ::cell_to_flat
claim: No production or tool caller, so all three are test-only. The module's one live export is `axial_distance` (selfplay/instrumentation.py).
evidence: `git grep -n -w -E "axial_to_flat|flat_to_axial|cell_to_flat" -- src tools crates docs` → only coordinates.py itself plus the util/__init__.py docstring example; tests: tests/util/test_coordinates.py only.
callers: AST imports: tests/util/test_coordinates.py only · entry points: none · python -m: none · subprocess/`-c` strings: the whole-tree -w grep covers them → 0 · importlib/getattr: `git grep -nE "[\"'](flat_to_axial|axial_to_flat|cell_to_flat)[\"']"` → 0 · conftest: none · pyo3: n/a (the Rust `from_flat`/`to_flat` are not these) · config keys: none · gates: none · STATE: none.
Δlines: −42 (AST spans 14+10+18). Tests: `pytest --collect-only -m '' tests/util/test_coordinates.py | grep -cE "flat_to_axial|axial_to_flat|cell_to_flat"` → 17 of 31 items go (test-floor move, gate 3c). The module docstring's Rust-mirror sentence and the util/__init__ example move with them.
witness: tests/util/test_coordinates.py (the remaining axial_distance rows)
depends: S-A-CORE-3-17 (the util/__init__ docstring)

### S-A-CORE-3-06 | DEAD | A
subject: src/mantis/monitor/game_record.py::TERMINATIONS
claim: A "declared vocabulary" constant that nothing validates against or reads.
evidence: `git grep -n -w TERMINATIONS` → 1 hit (its def); string form → 0.
callers: AST: 0 · entry points: none · python -m: none · subprocess: none · getattr/string: 0 · conftest: none · pyo3: n/a · config keys: none · gates: none · STATE/docs: docs/contracts/game_record.md lists the same tokens in prose but does not cite the symbol (it is not in the -w hits).
Δlines: −3 (1 line + 2 comment lines).
witness: NONE
depends: —

### S-A-CORE-3-07 | DEAD | A
subject: src/mantis/util/mirror_receipts.py::is_receipt
claim: Defined and exported, never called. tools/mirror_pull.py filters on RECEIPT_SUFFIX directly.
evidence: `git grep -n -w is_receipt` → 2 (the def and `__all__`); string form → 1 (`__all__`).
callers: AST: 0 · entry points: none · python -m: none · subprocess/rsync filters: tools/mirror_pull.py uses `RECEIPT_SUFFIX`, not this · getattr: 0 · conftest: none · pyo3: n/a · config keys: none · gates: none · STATE puller procedure: tools/mirror_pull.py (checked above).
Δlines: −3 (2-line def + blank) and −1 `__all__` token.
witness: NONE
depends: —

### S-A-CORE-3-08 | DEAD | B
subject: src/mantis/arena/regime.py::RegimeKey.from_canonical (+ ::_N_FIELDS, whose only reader it is)
claim: Nothing parses a canonical RegimeKey back. The inverse exists for one round-trip test.
evidence: `git grep -n -w from_canonical` → regime.py + tests/arena/test_regime_key.py::test_canonical_roundtrip.
callers: AST: that test only · entry points: none · python -m: none · subprocess: none · getattr/string: 0 · conftest: none · pyo3: n/a · config keys: none · gates: none · STATE: none · tools (dashboard/viewer/analyzer): the whole-tree grep → 0.
Δlines: −12 (AST span) −1 (`_N_FIELDS`).
witness: tests/arena/test_regime_key.py (the test must be rewritten, or it drops 1 item; test-floor move)
depends: —

### S-A-CORE-3-09 | DEAD | B
subject: src/mantis/bots/strix.py::StrixBot.__init__ (`self._sims`, `self._m_actions`, kwarg `m_actions`)
claim: Both attributes are written and never read: the sims and m_actions travel in the driver's `load` request, not through the bot. No caller passes `m_actions=`.
evidence: `git grep -n -w _m_actions` → 1 (the assignment); `git grep -n "\._sims\b" -- src/mantis/bots` → the assignment only; `git grep -n "StrixBot(" -- src tools tests` → 10 sites, and none passes m_actions.
callers: AST/attribute: 0 readers · entry points: none · python -m: none · subprocess: the driver JSON carries `m_actions` from DEFAULT_M_ACTIONS (load_request), not from the bot · getattr/string: `"m_actions"` strings are request keys in tools/ladder/backends.py and tools/analyzer/strix.py, not this attribute · conftest: none · pyo3: n/a · config keys: none · gates: none · STATE strix follower: goes through resolve_strix → StrixBot(sims=…), still no reader.
Δlines: −3 (2 assignments + the `m_actions` param). Dropping the `sims` parameter itself is a signature change touching 10 call sites (tools/ladder/backends.py included), so lane B.
witness: tests/bots/test_strix_adapter.py
depends: —. The file also carries a CARDED item (see 14).

### S-A-CORE-3-10 | DEAD | C
subject: src/mantis/bots/resolve.py::SKIP_REASON_MARKERS (+ its comment "Consumed by `mantis.eval.pipeline`'s in-run skip-class counter")
claim: The consumer its comment names (eval_rung_skip_class) was deleted with the sealbot rung by R362(c). Nothing reads the table now, so the comment is false and the table is dead.
evidence: `git grep -n SKIP_REASON_MARKERS` → resolve.py (def, `__all__`), docs/design/repo_design.md §947 ("the adapter seam — `bots/protocol.py`, `resolve_bot`, `SKIP_REASON_MARKERS` — is untouched"), 00_MAP; docs/contracts/event_manifest.md and eval_instrument.md record the counter as deleted by R362(c).
callers: AST: 0 · entry points: none · python -m: none · subprocess: none · getattr/string: `[\"']SKIP_REASON_MARKERS[\"']` → 1 (`__all__`) · conftest: none · pyo3: n/a · config keys: none · gates: none · STATE: none · tests: 0 (the symbol census's tests bucket).
Δlines: −6 (the dict) −1 (`__all__`) −2 (the comment).
witness: NONE
depends: S-A-CORE-3-11. Lane C because repo_design names it as a seam member; deleting it needs a repo_design amendment (R9).

### S-A-CORE-3-11 | PACK | C
subject: sealbot adapter — src/mantis/bots/sealbot.py; resolve.py::_resolve_sealbot, ::_NO_DEPTH_REASON, ::_R326_EXCLUDED_SEALBOT_DEPTHS, ::_R139_SKIP_MARKER, the "sealbot" entry of ::_KNOWN_KINDS; tests/bots/test_sealbot_{adapter,resolve,vendored}.py, tests/tools/test_vendor_{build,pins}_sealbot.py; Makefile `vendor.sealbot`; tools/vendor_build_sealbot.sh; vendor/pins.toml [pins.sealbot] + vendor/patches/sealbot.patch
claim: Since R362(c) no production path resolves `sealbot`. The live callers are `resolve_bot("random", …)` in eval/worker.py and acceptance_witness.py, the strix RungJob built in tools/strength_frontier.py, and `_play_rung_block` with `rung_jobs=[]` in production. STATE records the choice explicitly: "the sealbot ADAPTER stays as a vendored opponent with no production caller".
evidence: `git grep -n -i sealbot -- src tools configs Makefile vendor` → resolve/sealbot/strix (`find_vendor_root` import), nsims.py `_KNOWN_OPPONENTS`, and prose. The tools/dashboard readers handle pre-R362 records only. docs/governance/STATE.md (R362 deletion paragraph) states the kept-adapter posture.
callers: resolve_bot's only `kind` literals in src/tools are "random" (worker.py ×2, acceptance_witness.py) and the RungJob.bot values built in tools/strength_frontier.py (`STRIX`) · python -m / entry points / subprocess: none name sealbot · conftest: none · configs: `git grep -n sealbot -- configs` → 0 · STATE procedures: none run it.
Verdict: KEEP until the operator rules. It is ruled-kept (the R362 landing record), so this is not a scout call. If retired: strix.py's `find_vendor_root` import has to move first (strix.py imports it from sealbot.py), config/resolve/nsims.py `_KNOWN_OPPONENTS` changes (slice C2), and tier_declaration.txt rows 22–24 go.
Δlines: if retired, −374 (wc -l bots/sealbot.py) −46 resolver lines (AST spans 36+4+5+1) −1 504 test lines (`wc -l` over the 5 test files; test-floor move); plus the vendor script/patch/pin.
witness: tests/bots/test_sealbot_*.py, tests/tools/test_vendor_*_sealbot.py, gate 10 (Makefile/doc references to the build script)
depends: S-A-CORE-3-10

### S-A-CORE-3-12 | DEAD | C
subject: the GRID representation arm — src/mantis/eval/worker.py::build_candidate_player (`representation == "grid"` branch) and the grid prose in ::_assert_policy_pool_implemented / ::_assert_value_pool_implemented; src/mantis/arena/deploy_head.py (`InferFn`, `infer_fn=`, the exactly-one guard, the dense tail of `_evaluate`); the "`None` is the GRID arm" comments in eval/pipeline.py, eval/rounds.py, diagnostics/acceptance_witness.py::_arm_engine
claim: No registered encoding can be grid. The Rust registry parser refuses "grid" by name (crates/mantis-encoding/src/spec/mod.rs) and both registered specs are graph, so production never reaches the branch. By R10 this is "a compatibility shim for a state the tree no longer has".
evidence: `.venv/bin/python -c "from mantis.encoding import all_specs; …"` → `gnn_axis_r8 graph`, `gnn_axis_v1 graph`; registry.toml header: "'grid' is REFUSED by name"; `git grep -n "infer_fn=" -- src tools` → only worker.py's grid branch; tools/analyzer and tools/ladder call build_candidate_player on graph specs.
callers: src/tools: worker.py grid branch only · tests: 11 `infer_fn=` sites (tests/arena/test_deploy_head*.py ×4 files, tests/eval/test_eval_selfplay_child_parity.py) use the grid arm as a cheap search stub · entry points / python -m / subprocess / getattr / conftest / pyo3 / config keys / gates / STATE: none reach a grid spec (the RunConfig schema still types `representation: Literal["grid","graph"]` — slice C2).
Δlines: worker.py branch −7 (the grid `if` block of build_candidate_player, whose AST span is 29) plus prose; deploy_head ≈ −18 (the `_evaluate` span of 12 → 2, the exactly-one guard of 7 → 2, `InferFn`, the param and the attribute). Exact figures are owed after the tests are re-stubbed.
witness: tests/arena/test_deploy_head*.py, tests/eval/test_eval_selfplay_child_parity.py (the exactly-one test), tests/eval/test_leaf_build_threads_wiring.py::test_the_grid_arm_is_the_serial_width
depends: —. Lane C because eval/worker.py is protected (PZ-1: strength_floor and 1-in-1 collate). The deploy_head half alone would be lane B, since the arena tests need re-stubbing onto `expand_fn`.

### S-A-CORE-3-13 | DOC | C
subject: src/mantis/monitor/rules.py::WARN_RULE_INPUTS["selfplay_entropy_collapse"], ::check_selfplay_entropy_collapse
claim: The key rules.py calls "canonical", `selfplay_model_entropy_batch`, has NO producer anywhere. The one it calls "legacy", `policy_entropy_selfplay`, is the LIVE producer (src/mantis/train/events.py, trainer/core.py; R355(e)). The labels are inverted, and the canonical-key lookup is a dead first arm.
evidence: `git grep -n selfplay_model_entropy_batch` → rules.py + tests/monitor/* only; `git grep -n policy_entropy_selfplay -- src` → train/events.py, train/trainer/core.py, rules.py.
Δlines: −1 (the dead key) plus a docstring reword. tests/monitor/test_rules.py feeds the dead key in 5 asserts, so those move to the live key.
witness: tests/monitor/test_rules.py, tests/monitor/test_phantom_rule_is_visible.py
depends: —. Lane C because rules.py is protected (PZ-1: draw-rate abort, finite-gradient guard).

### S-A-CORE-3-14 | DUP | C
subject: sha256 and git provenance helpers — CARDED: src/mantis/bots/strix.py::verify_checkpoint_sha and src/mantis/diagnostics/fusion_calibrate.py::_sha256 (`hashlib.sha256(path.read_bytes())`, CARD-MECHANISM-SWEEP). UNCARDED: src/mantis/diagnostics/worker_sweep.py::_sha256 (the same idiom), fusion_calibrate.py::_git_head and worker_sweep.py::_git beside src/mantis/util/git.py::head_sha / ::is_dirty
claim: `util.hashing.sha256_file` and `util.git` are the one implementation of each. Four local re-implementations remain in the slice, and CARD-MECHANISM-SWEEP lists only two of them.
evidence: `grep -n "hashlib.sha256(.*read_bytes" -r src/mantis/{bots,diagnostics}` → strix.py, fusion_calibrate.py, worker_sweep.py; CARDS.md CARD-MECHANISM-SWEEP names strix.py, fusion_calibrate.py, encoding/__init__.py, but not worker_sweep.py.
deliberate?: not a seam or oracle twin. BUT the git helpers are not byte-equivalent. worker_sweep runs `git status --porcelain`, which counts UNTRACKED files; util.git.is_dirty uses `--untracked-files=no`. util.git also has a 2 s timeout where fusion_calibrate has 10 s. The merge is a behaviour choice (lane B at least).
Δlines: −4 (fusion_calibrate `_sha256` span) −8 (worker_sweep `_sha256`) −9 (`_git_head`) −6 (`_git`), less any import lines added.
witness: tests/diagnostics/test_fusion_calibrate_*.py, tests/diagnostics/test_worker_sweep_*.py
depends: —. Lane C: carded, applied ON CONTACT only (card text), and worker_sweep is ruling-named.

### S-A-CORE-3-15 | DUP | C
subject: src/mantis/diagnostics/worker_sweep.py::GIB, ::_gib vs src/mantis/diagnostics/eval_child_memory.py::GIB, ::_fmt_gib
claim: The two definitions are byte-identical (`1024 ** 3`; `"unmeasured" if value is None else f"{value / GIB:.4f} GiB"`), and worker_sweep already imports eval_child_memory.
evidence: `git grep -n -E "1024 \*\* 3" -- src tools` → exactly these two; `grep -A2 "def _gib\|def _fmt_gib"` → identical bodies.
deliberate?: no. worker_sweep's header says the stopping rule is "imported, not re-written" from eval_child_memory, and the formatter is the same instrument's unit.
Δlines: −4 (GIB 1 + `_gib` 2 + blank 1), plus 1 import line changed.
witness: tests/diagnostics/test_worker_sweep_*.py, tests/diagnostics/test_eval_child_memory*.py
depends: —. Lane C because worker_sweep is ruling-named.

### S-A-CORE-3-16 | SIMPLIFY | B
subject: src/mantis/diagnostics/mirror_receipts.py `__all__` forwarding of mantis.train.bundle_receipts names (bundle_member_paths, unreceipted_bundle_steps, unreceipted_members)
claim: A pure forwarder. tools/mirror_pull.py imports `bundle_member_paths` through it while importing `mantis.train.bundle_receipts` directly on the next line. Tests reach `D.stamped_checkpoints` and `D.unreceipted_bundle_steps` through the module attribute.
evidence: `sed -n 18,21p tools/mirror_pull.py` → both imports; `git grep -n "D\.\(unreceipted\|bundle_member\|stamped\)" tests` → tests/diagnostics/test_mirror_receipts.py, tests/tools/test_mirror_pull.py, tests/tools/test_preflight_start_halts.py.
Δlines: −3 (`__all__` tokens), with 3 test files and 1 tool import retargeted.
witness: tests/diagnostics/test_mirror_receipts.py, tests/tools/test_mirror_pull.py
depends: —

### S-A-CORE-3-17 | DOC | A
subject: src/mantis/bots/__init__.py, src/mantis/diagnostics/__init__.py, src/mantis/util/__init__.py (package docstrings)
claim: bots: "at HEAD `resolve_bot` raises `RungUnresolvable` for all three (0/6 ladder-rung census verdict)" is false, because random and strix resolve. diagnostics: "skeleton, port lands with its work package" is stale, because 10 modules have landed. util: a 5-line narrative on the R289(q) cpu_budget relocation, the kind of narrative block R316(e) forbids, and its usage example names `axial_to_flat` (S-05).
evidence: `cat` of the three files; resolve.py returns factories for "random" and "strix" (read at HEAD).
Δlines: −5 (the util narrative) and two one-line rewords. Gate 14's docstring_excess can only fall.
witness: gate 14 (comment ratchet)
depends: S-A-CORE-3-05

### S-A-CORE-3-18 | PACK | C
subject: src/mantis/diagnostics/
claim: Verdict KEEP ALL. No spent one-time probe: every module has a live caller on a channel this census ticked.
evidence (per module, from the AST graph plus CALLERS §4/§5): acceptance_witness — PZ-2 R327 witness, tests/diagnostics/test_acceptance_witness.py · cuda_build_guard — tools/ci_gates/preflight_mint.py import · eval_child_memory — worker_sweep import + `python -m` in docs/contracts/event_manifest.md · f816_37_rate_bar — PZ-1, tools/dashboard/health.py import (its CLI `main` has no `python -m` citation, CALLERS §4; protected, left) · fusion_calibrate — `python -m` named by config/armed_aborts.py and resolve/fused_graph_caps.py error text + the dev template · mirror_receipts — tools/ci_gates/preflight_mint.py + tools/mirror_pull.py (puller procedure) · ring_audit / ring_reader / tactics — STATE witness/ring procedures + tools/probe1, tools/analyzer imports · worker_sweep — ruling-named (R308 Phase W, R338) + tools/worker_sweep_plan.toml.
Δlines: 0
witness: n/a
depends: —

### S-A-CORE-3-19 | PACK | C
subject: src/mantis/deploy/
claim: Verdict KEEP. It is a 1-line reserved-empty package, named RESERVED by CLAUDE.md ("src/mantis/deploy/ is reserved-empty until post-cutover").
evidence: `wc -l src/mantis/deploy/__init__.py` → 1; no importers (the AST graph).
Δlines: 0
witness: n/a
depends: —

## DEFECTS
- src/mantis/monitor/config.py::MonitorConfig carries a literal default for every field, duplicating config/schema/monitor.py's defaults, against R1 ("a default lives only in the schema field"). src/ is held by the single-construction census (tests/config/test_monitor_config_single_authority.py), but 61 bare `MonitorConfig()` in tests read these literals as a second authority.
- src/mantis/bots/__init__.py docstring asserts that every rung is unresolvable at HEAD. That is false (S-17).
- src/mantis/monitor/rules.py calls the live `policy_entropy_selfplay` "legacy" and the producer-less key "canonical" (S-13).

## PARKED
- none

## HANDOFF
- C2: src/mantis/config/schema/core.py `representation: Literal["grid","graph"]` and the selfplay/hparams grid refusal are the schema half of the grid residue (S-12); config/resolve/nsims.py `_KNOWN_OPPONENTS` carries "sealbot" (S-11).
- T7: tests/monitor/test_phantom_rule_is_visible.py docstring says `policy_entropy_selfplay` has no producer, stale since R355(e) (S-13).
- T6: the tautological test tests/diagnostics/test_worker_sweep_determinism.py::test_the_bands_constant_is_SUPERSEDED_but_still_pinned_for_history (S-04).
- L2: tools/dashboard/{strength,tier2,tier3,hero}.py keep pre-R362 sealbot readers. STATE records the stated-gap rendering as landed, so check it before calling them dead.
- C1: src/mantis/encoding/__init__.py is the third carded `hashlib.sha256(path.read_bytes())` site (S-14).

## Not covered
- Line-level reads of the five largest files (eval/pipeline.py 911, diagnostics/worker_sweep.py 1560, eval/worker.py 781, diagnostics/fusion_calibrate.py 720, monitor/supervise.py 603). The symbol and method census covered them only down to top-level defs and class methods, so nested helpers and in-body dead branches were not examined.
- producer_manifest.yaml row liveness (its verifier needs torch here) and the monitor/manifest.py internals.
- No probe was run: torch is absent. Every lane-A claim still needs the reviewer's probe.
- eval/worker.py::_model_sims_for_kind's "random" branch: it is reached only by test-built RungJobs; noted, not claimed.
- The frozen book data (arena/books/*.json, manifest.toml): out of scope under LAW-15 and not examined.

## Review
reviewer: fresh read-only agent (not the author); probes in throwaway worktrees, removed
src/ tools/ tests/ are byte-identical between 69e1532 and the reviewed tip (`git diff --stat 69e1532 HEAD -- src tools tests crates configs Makefile` → empty).

| ID | verdict | lane | Δlines (probe-measured for lane A) | note |
|---|---|---|---|---|
| S-A-CORE-3-01 | CONFIRMED | B | −141 (AST, not probed: lane B) | only package-level importer in the tree is tests/bots/test_protocol.py (mantis.bots, excluded) |
| S-A-CORE-3-02 | CONFIRMED | C | −8 | the AST shows eval/__init__.py as the only importer of the 3 foreign errors from eval.errors |
| S-A-CORE-3-03 | CONFIRMED | C | −2 span, −2 tokens | producer_manifest row names EvalBrokenReason, not this |
| S-A-CORE-3-04 | CONFIRMED | C | −3 src, −4 test | RULINGS.md names the worker sweep (R308(f)); test-floor move |
| S-A-CORE-3-05 | CONFIRMED | B | −42 | spans 14/10/18 re-derived; no `_engine` parity pin in the test, so not an oracle |
| S-A-CORE-3-06 | AMENDED (Δ) | A | −5 (probe) | probe green; the 2 blank separators go with it |
| S-A-CORE-3-07 | AMENDED (Δ) | A | −4 (probe) | probe green; the `__all__` line is rewritten (net 0) |
| S-A-CORE-3-08 | CONFIRMED | B | −12 −1 | def span 48–59 re-derived |
| S-A-CORE-3-09 | CONFIRMED | B | −3 | 11 `StrixBot(` sites, not 10; none passes `m_actions` |
| S-A-CORE-3-10 | CONFIRMED | C | −9 | repo_design.md:947 is the one live citation |
| S-A-CORE-3-11 | ARCHITECT | C | −374 −46 −1 504 (if retired) | retire the adapter STATE keeps? |
| S-A-CORE-3-12 | CONFIRMED | C | owed after re-stub | Rust refuses "grid" by name; one `infer_fn=` call in src |
| S-A-CORE-3-13 | CONFIRMED (real, behaviour-neutral) | C | −1 + reword | LAW-07-shaped phantom input; see note |
| S-A-CORE-3-14 | CONFIRMED | C | −27 less imports | card omits worker_sweep; the git flags differ |
| S-A-CORE-3-15 | CONFIRMED | C | −4 | bodies byte-identical; worker_sweep already imports eval_child_memory |
| S-A-CORE-3-16 | AMENDED (subject, Δ) | B | −1 import + `__all__` rewrap | only `unreceipted_bundle_steps` is a pure forward |
| S-A-CORE-3-17 | CONFIRMED | A (DOC, not a deletion class; not probed) | −5 + 2 rewords | comment floor file moves in the same commit |
| S-A-CORE-3-18 | CONFIRMED | C | 0 | every module has a `-m` or import caller |
| S-A-CORE-3-19 | CONFIRMED | C | 0 | reserved-empty, named in CLAUDE.md |

### Per-finding notes
S-A-CORE-3-01 — CONFIRMED: an AST walk of every tracked .py (ImportFrom on the 7 slice packages naming a non-submodule, plus `mantis.<pkg>.<Name>` attribute chains) → only `tests/bots/test_protocol.py mantis.bots [BotProtocol, RandomBot, RungUnresolvable, resolve_bot]`; alias/import_module grep → only tests/test_run_composition.py's substring census over train/**, which is unaffected. The torch side effect was re-run: `import mantis.arena.regime` and `mantis.eval.sequential` → ModuleNotFoundError torch; `mantis.monitor.rules` imports. Non-docstring lines by AST: 29/54/58 = 141. Lane B holds because it is an import-order behaviour change.
S-A-CORE-3-02 — CONFIRMED: an AST scan of ImportFrom(mantis.eval.errors | mantis.eval) for BookError/MixedRegimeError/RungUnresolvable → only src/mantis/eval/__init__.py; no `X.BookError`-style attribute use anywhere.
S-A-CORE-3-03 — CONFIRMED: `git grep -n -w EvalBrokenError` → 4 hits (def and `__all__` in errors.py, 2 in eval/__init__); AST span 50–51. The producer_manifest.yaml eval-broken row's `also:` names `EvalBrokenReason`, so the manifest is unaffected.
S-A-CORE-3-04 — CONFIRMED: `git grep -n -w RULED_DETERMINISM_BAND_PCT` → def + tests/diagnostics/test_worker_sweep_determinism.py (asserts `== 1.0`), no docs. RULINGS.md cites the worker sweep under R308(f), so lane C stands. Deleting the test is a gate-3 floor move.
S-A-CORE-3-05 — CONFIRMED: AST spans re-derived (flat_to_axial 14, axial_to_flat 10, cell_to_flat 18). tests/util/test_coordinates.py imports no `_engine`, so the "mirror the Rust from_flat/to_flat" docstring has no parity pin and these are not an oracle twin.
S-A-CORE-3-06 — AMENDED Δ −3 → −5: DELETE-PROBE (scratchpad worktree, `-S` + isolating PYTHONPATH). Deleted the 2-line comment, the tuple and the 2 blank separators → `import mantis` OK; collect-only `2289 tests collected, 167 errors`, which equals the baseline; `cargo check --workspace --all-targets --locked` Finished. The nearest tests (tests/monitor/test_game_record.py, tests/diagnostics/test_mirror_receipts.py, tests/tools/test_mirror_pull.py, tests/tools/test_ladder_receipt.py) show per-node outcomes IDENTICAL to HEAD under the same recipe (`-rA` lists diffed; 22 pass; every failure/error is `No module named 'torch'` at HEAD too). Gates: comment_lint GREEN, r8_header_gate 0 stale, ruff clean. docs/contracts/game_record.md lists the tokens but does not cite the symbol.
S-A-CORE-3-07 — AMENDED Δ −3 → −4: same probe and worktree (the two subjects are independent). `git diff --stat` for the batch: 2 files, +1 −10 = net −9. The `__all__` line is rewritten in place (net 0). `git grep -n -w is_receipt` → def + `__all__` only.
S-A-CORE-3-08 — CONFIRMED: `git grep -n -w -E "from_canonical|_N_FIELDS"` → regime.py + tests/arena/test_regime_key.py:31 only.
S-A-CORE-3-09 — CONFIRMED (evidence corrected): `git grep -n -E "\b_sims\b|\.m_actions\b"` over src/tools/tests (strix_driver excluded) → only the strix.py:95 assignment. `StrixBot(` → 11 sites (7 test_strix_adapter, 2 test_strix_net_only_cell, tools/ladder/backends.py, strix.py's resolver), none passes `m_actions`. The research doc hit for `_m_actions` is an external `sprt_mcts_m_actions`, not this.
S-A-CORE-3-10 — CONFIRMED: `git grep -n -w SKIP_REASON_MARKERS` → resolve.py (def, `__all__`), repo_design.md:947, 00_MAP, and an archived register entry (history). No src/tools/tests reader.
S-A-CORE-3-11 — ARCHITECT: STATE.md:327–328 reads "the sealbot ADAPTER stays as a vendored opponent with no production caller". `git grep -n -i -E "[\"']sealbot"` in src/tools → only resolve.py and sealbot.py; tools/select_balanced_book.py and tools/strength_frontier.py mention it only as the deleted R362(c) cell; strix.py imports `find_vendor_root` from sealbot.py (confirmed). wc: 374 src, 1 504 test lines; tier_declaration rows 22–24 confirmed. The scout missed two things. CLAUDE.md itself names `make vendor.sealbot` as a per-checkout build step. PZ lists sealbot.py::BUILD_ABSENT_MARKER (R324) as a protected marker. Retirement therefore also edits CLAUDE.md and needs a ruling. Question: does the operator retire the adapter that the R362 landing record keeps?
S-A-CORE-3-12 — CONFIRMED: crates/mantis-encoding/src/spec/mod.rs refuses `"grid"` BY NAME (R346(f)); registry.toml header "Registered set (2 entries): gnn_axis_v1, gnn_axis_r8"; `git grep -n "infer_fn=" -- src tools` → deploy_head.py docstring and error text plus eval/worker.py:277 (the grid branch) only.
S-A-CORE-3-13 — CONFIRMED, the defect is REAL but behaviour-neutral. `git grep -n -c selfplay_model_entropy_batch` → rules.py 3, tests/monitor/test_rules.py 5, tests/monitor/test_phantom_rule_is_visible.py 2. No producer exists in src/crates/tools/configs, and producer_manifest.yaml's warn.training_step_alerts row names only the `training_step` literal. `policy_entropy_selfplay` is produced (train/events.py:253, trainer/core.py:499). The docstring's "canonical"/"legacy" labels are therefore inverted. Behaviour is still correct: `payload.get(dead, payload.get(live))` falls through, and `rule_input_absent` is `all(... is None)`, so the dead key neither suppresses the rule nor inflates WARN_RULE_SKIPS. It is a LAW-07-shaped declared input with no producer, and it stays lane C (rules.py PZ-1). tests/monitor/test_phantom_rule_is_visible.py:6–7 still says the live key "has none either", which is stale (the T7 handoff is correct).
S-A-CORE-3-14 — CONFIRMED: CARDS.md CARD-MECHANISM-SWEEP names bots/strix.py, diagnostics/fusion_calibrate.py, encoding/__init__.py, but not worker_sweep.py. util/git.py:28 runs `status --porcelain --untracked-files=no` while worker_sweep.py:1112 runs a bare `status --porcelain`, so the merge is a behaviour choice. AST spans 8+6 (worker_sweep) and 4+9 (fusion_calibrate) re-derived.
S-A-CORE-3-15 — CONFIRMED: `grep -A2` shows the `GIB = 1024 ** 3` lines and the `_gib`/`_fmt_gib` bodies are character-identical; worker_sweep.py:57 already imports from eval_child_memory.
S-A-CORE-3-16 — AMENDED (subject, Δ): diagnostics/mirror_receipts.py imports `bundle_member_paths`, `stamped_checkpoints`, `unreceipted_bundle_steps`, `unreceipted_members` from train.bundle_receipts. It USES `unreceipted_members` (:66), `bundle_member_paths` (:73) and `stamped_checkpoints` (:76) itself, so only `unreceipted_bundle_steps` is a pure forward. Dropping names from `__all__` alone breaks no `D.x` attribute access. The real edit is −1 import line (`unreceipted_bundle_steps`) plus the retargets (tests reaching `D.unreceipted_bundle_steps`, and mirror_pull.py's `bundle_member_paths`). It stays lane B.
S-A-CORE-3-17 — CONFIRMED: resolve.py::resolve_bot returns a RandomBot factory for "random" and delegates "strix" to `resolve_strix`, so the bots/__init__ "raises RungUnresolvable for all three" is false. The diagnostics one-liner is stale. No test pins any package `__doc__` (`git grep -E "(util|bots|diagnostics|monitor)\.__doc__|HEADLESS ONLY" -- tests` → 0). comment_lint on the probe tree measured docstring_excess 13079 at floor 13079. A fall prints a "ratchet the floor down in this commit" note, so tools/ci_gates/comment_length_floor.txt moves with it. Doc rewords are not a deletion class, so no probe was run.
S-A-CORE-3-18 — CONFIRMED: `git grep -h -o "python -m mantis\.diagnostics\.[a-z_]+"` → fusion_calibrate 5, eval_child_memory 3, worker_sweep 2, ring_reader 2, ring_audit 1, mirror_receipts 1, cuda_build_guard 1, acceptance_witness 1 (preregs under docs/design/measurements name ring_audit/ring_reader). f816_37_rate_bar (PZ-1) and tactics are reached by import (tools/dashboard/health.py, diagnostics/ring_audit.py:19). STATE.md's only `-m` is `mantis.run`.
S-A-CORE-3-19 — CONFIRMED: CLAUDE.md "Deliberately absent" names deploy/ reserved-empty.
DEFECTS (scout) — the MonitorConfig R1 duplicate-default defect stands (monitor/config.py carries literal field defaults), but the count is 53 bare `MonitorConfig()` in tests (`git grep -o "MonitorConfig()" -- tests | wc -l`), not 61.

### Missed by the scout (optional, max 5)
NEW-1 | DEAD/CONFIG | C — src/mantis/eval/aggregate.py::should_escalate (4-line span) has no src/tools caller (`git grep -n -w should_escalate` → aggregate.py, eval/__init__ re-export, tests/eval/test_gate_parity.py only), yet aggregate.py's docstring says the gate "calls" it. Its threshold, config/schema/core.py `screen_confirm_lo`, has no src reader outside the schema (`git grep -n screen_confirm_lo -- src` → schema only). That is a LAW-08 check for C2. Lane C: aggregate.py is in the PZ glob and the function is a parity-pinned port.

### Tally: raised 19 | confirmed 15 | amended 3 | refuted 0 | pending 0 | architect 1 (+1 NEW)
