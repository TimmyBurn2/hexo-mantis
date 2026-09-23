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
