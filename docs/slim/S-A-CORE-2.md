# S-A-CORE-2 — slice C2
scope: src/mantis/selfplay/**, src/mantis/data/**, src/mantis/config/**, src/mantis/env/**, src/mantis/*.py + src/mantis/_engine.pyi (79 files, 13 937 lines at 69e1532); method: `uvx vulture src/mantis tools tests --min-confidence 60` (leads, 35 in-slice hits, mostly pydantic validators/fields = false positives); an AST census of every top-level def/assign in the slice with `git grep -w` hit counts split src/tools/tests/docs (scratchpad c2/census.py, c2/census.tsv); an AST scan of class methods and `self.X` attributes with no non-test reader (scratchpad c2/methods.py); `.venv/bin/python -m pytest -q -p no:cacheprovider tests/config/test_every_key_has_consumer.py tests/config/test_every_key_has_consumer_p2.py tests/config/test_consumer_citation_arrows.py` -> 19 passed; `leaf_paths(RunConfig)` (160 leaves) and a single-value-Literal scan over them; runtime `dir(mantis._engine)` against an AST parse of src/mantis/_engine.pyi. torch is not importable here, so nothing that imports torch was run. The analysis is static.

## Summary
- By class and lane: DEAD 12 (A 4, B 2, C 6) · PACK 2 (B 2) · SIMPLIFY 4 (B 1, C 3) · CONFIG 1 (C 1) · DUP 2 (A 1, B 1) · DOC 2 (A 2). Total 23: lane A 7, lane B 6, lane C 10.
- Top 3 by Δlines: S-A-CORE-2-01 data/ legacy corpus pack, -2066 · S-A-CORE-2-15 selfplay/__init__ re-exports, -107 · S-A-CORE-2-16 one-line resolver forwarders, up to -99.
- Lane C comes from the PZ glob, which lists src/mantis/run.py, the selfplay pool, inference_server and graph_collate files, and the schema core. It also comes from ruling-named items and contract-doc citations (gate 13). Where a lane-C finding does not touch the protected symbol itself, its block says so.

## Findings

### S-A-CORE-2-01 | PACK | B
subject: src/mantis/data/ (every module except bootstrap_encode.py and a slimmed __init__.py): _log.py, corpus_analysis.py, corpus_io.py, corpus_metrics.py, corpus_reporter.py, generate.py, human_seeding.py, loss_counters.py, pipeline_metrics.py, sources/{__init__,base,human}.py
claim: This is the pre-graph human/bot corpus analysis pipeline. Nothing in src/ or tools/ imports it. The live BC corpus producer, bootstrap_encode.py, imports none of it (only mantis.encoding + _engine). generate.py::generate_bot_games has zero callers anywhere. Tests are the only other importers.
evidence: `git grep -n -E "mantis\.data|from \.\.?data" -- src tools ':!src/mantis/data'` -> only two error-message strings in train/pretrain/graph_route.py naming `python -m mantis.data.bootstrap_encode`; `git grep -n -w generate_bot_games` -> def + its own log label only; `sed` of data/__init__.py -> re-exports corpus_io/pipeline_metrics/sources, and census.tsv puts every consumer of those in tests/data/*. corpus_analysis.py keeps a "Re-export back-compat" import (`analyse_ply_coverage  # noqa: F401`), a shim for external callers that do not exist (R10).
callers:
  AST imports: census.tsv + `git grep` of `mantis.data.<mod>` for each module -> importers are tests/data/{test_corpus_io,test_data_loss_counters,test_sources_metrics}.py and tests/selfplay/test_game_complete_absence.py only
  pyproject entry points: none exist (CALLERS §1). pyproject.toml's `analysis` extra carries a COMMENT naming corpus_metrics/corpus_reporter/corpus_analysis as the reason for matplotlib/rich
  python -m: `git grep -n -E "python[0-9.]* -m mantis.data.corpus_analysis"` -> no citation anywhere (CALLERS §4: "self only"). It is runnable but no procedure, doc or Makefile names it
  subprocess: CALLERS §6 subprocess table -> no data/ module except bootstrap_encode (git rev-parse)
  importlib/getattr/registry: CALLERS §7 -> none. The `data.generate.*` strings are counter labels, not lookups
  conftest: CALLERS §8 fixture list -> none import data/ (tests/data/_frozen_games.py is a bare-name helper for bootstrap tests)
  pyo3: n/a (Python-only modules)
  config keys: no schema leaf is read here (`grep` of leaves.txt names in data/ -> none)
  gate tool paths: run_all.sh / lint_gate.sh name no data/ module
  STATE/box procedures: `git grep -n -i -E "corpus_analysis|corpus_metrics|generate_bot_games|human_seeding|pipeline_metrics|corpus_io" -- docs` -> none outside the frozen archive
  PATH-STRING CONSUMER (the blocker): tools/audit_bootstrap_corpus.py cites `src/mantis/data/sources/human.py:115-119` as `"floor_source"` in its emitted record, and generate.py:147-150 in a message. Deleting human.py needs that provenance re-pointed.
Δlines: -2066 (`wc -l` of the 12 files). data/__init__.py also shrinks from 34 lines to its bootstrap mention. Tests: 25 functions in 3 files (test_corpus_io 6, test_data_loss_counters 12, test_sources_metrics 7) plus test_game_complete_absence::test_the_data_loss_counters_sibling_is_deliberately_UNCHANGED. That is a gate-3c test-floor move. repo_design §1 ("data/ corpus generation, IO, metrics, augmentation LUTs"), the CLAUDE.md map and the pyproject comment also move.
witness: tests/data/* (deleted with it); gate 9 import-DAG
depends: —

### S-A-CORE-2-02 | PACK | B
subject: src/mantis/env/ (__init__.py, game_state.py::GameState)
claim: The only production importer of GameState is data/generate.py, which is dead in -01. With -01 the package has no live caller.
evidence: `git grep -n -w -E "GameState|game_state|mantis\.env" -- src tools crates docs` -> data/generate.py (3 sites). tools/strix_driver.py's `GameState` is hexo_rs's, not ours. train/axis_distribution.py only mentions it in comments.
callers:
  AST imports: the grep above + tests -> tests/env/test_game_state.py, tests/util/test_b4_history_len_sot.py
  entry points / python -m / subprocess: none (CALLERS §1, §4, §6)
  importlib/getattr: none. tests/selfplay/test_pool_surface.py::ALLOWED_MANTIS_ROOTS names "mantis.env" as an allowed import root (a set literal, not a caller)
  conftest: none
  pyo3: consumes `_engine.Board` and exports nothing
  config keys: none
  gate tool paths: none. tools/hardcode_scan.py lists `game_state` as a scan-location label (a string)
  STATE/box: none
Δlines: -84 (`wc -l src/mantis/env/*.py`); 11 tests in tests/env/test_game_state.py plus the env arm of test_b4_history_len_sot (test-floor move). repo_design §1 and the CLAUDE.md package list name env/.
witness: tests/env/test_game_state.py
depends: S-A-CORE-2-01

### S-A-CORE-2-03 | DEAD | C
subject: src/mantis/selfplay/inference_server.py::InferenceServer: the grid `else:` arm of `__init__`, ::submit_and_wait, ::infer, and the grid-only attributes `_feature_len`, `_shape`, `_board_size`, `_h2d_staging`, `_traced_model`
claim: `self._is_graph` is always True, so the dense arm and the dense single-state API are unreachable. Even if the dense arm were reached, `amp_dtype_for("grid")` raises two statements later.
evidence: hparams.py::is_graph_representation returns True for "graph" and RAISES RepresentationMismatch for "grid" or anything unknown (read in full). config/resolve/amp.py::resolve_amp_dtype raises ValueError for any representation except "graph". `git grep -n -E "submit_and_wait\b|\.infer\(" -- src tools tests` -> only the method's own `infer` wrapper and a census row. The attribute scan found `_h2d_staging` NO READS, and `_board_size` read only by tests.
callers:
  AST imports/attrs: the grep above; `git grep -n -E "_shape\b|_feature_len|_traced_model|_h2d_staging" -- src tests` -> only inference_server.py itself
  entry points / python -m / subprocess / importlib: none (CALLERS §1, §4, §6, §7). producer_manifest.yaml names this module for other symbols only
  conftest: none
  pyo3: `InferenceBatcher(feature_len=…, policy_len=…)` is the dense ctor form, used only inside the dead arm
  config keys / gates / STATE: none
Δlines: about -67 (AST/sed spans: else-arm 425-452 = 28, submit_and_wait 32, infer 2, the five dead graph-arm attribute lines). The `"graph" if self._is_graph else "grid"` ternaries in the snapshot collapse on the same lines.
witness: tests/selfplay/test_selfplay_census.py `_Q6_TABLE` row "InferenceServer.submit_and_wait / load_state_dict_safe", which must be edited.
depends: —  (lane C: the file is in the PZ-1 glob for the 1-in-1 collate and dump-on-fire members. None of those members is touched.)

### S-A-CORE-2-04 | DEAD | B
subject: src/mantis/selfplay/inference_local.py::LocalInferenceEngine.infer, ::infer_batch, ::_infer_batch_graph, ::infer_ls
claim: No production path reaches these. The eval worker's graph arm calls `infer_batch_ls`. `_infer_batch_graph`'s own docstring says "NO PRODUCTION CONSUMER REACHES THIS METHOD". The one other production reference, `infer_fn=engine.infer`, sits in eval/worker.py::build_candidate_player's `spec.representation == "grid"` arm, which is unreachable: the registry holds only graph encodings, and this class raises on grid at construction.
evidence: `git grep -n -E "\.infer_batch\(|\.infer\(|_infer_batch_graph|\.infer_ls\(" -- src tools tests` -> eval/worker.py:252 (infer_batch_ls, live) and :277 (grid arm); everything else is tests. `git grep -n -w infer_ls -- tools` -> none.
callers:
  AST imports: the grep above
  entry points / python -m / subprocess: none
  importlib/getattr: none. DeployHeadPlayer receives the bound method only in the dead grid arm
  conftest: none
  pyo3/config/gates/STATE: none
Δlines: -50 (AST spans 5+12+26+7). Test-floor move: tests/selfplay/test_inference_local.py loses 3 functions (the infer_batch ones), tests/selfplay/test_selfplay_census.py loses 2 `_Q6_TABLE` rows, and tests/eval/test_eval_selfplay_child_parity.py::test_infer_ls_is_the_same_refusal_predicate_as_infer_batch_ls plus tests/tools/test_analyzer_engines.py are re-pointed to infer_batch_ls.
witness: the tests named above
depends: HANDOFF C3 (the eval/worker.py grid arm)

### S-A-CORE-2-05 | DEAD | C
subject: src/mantis/selfplay/utils.py (whole module: ::get_temperature, ::quarter_cosine_temperature)
claim: Both names are referenced only by the selfplay/__init__ re-export. The "legacy alias" `temperature_threshold_ply` branch in get_temperature has no config source either. The module docstring claims "the Python bot paths use" it. None do.
evidence: `git grep -n -w -E "get_temperature|quarter_cosine_temperature|selfplay\.utils"` -> utils.py, selfplay/__init__.py, CARDS.md (B-15 "get_temperature (re-export only)"), and the fixture generator docstring
callers:
  AST imports: the grep above (whole tree)
  entry points / python -m / subprocess / importlib / conftest / pyo3 / config / gates / STATE: none (CALLERS §1-§11 list it nowhere). tests/fixtures/search_golden/temperature/gen_temperature_parity_golden.py mentions it in prose only
Δlines: -73 (`wc -l`) plus -3 re-export lines in selfplay/__init__.py
witness: NONE (no test imports it)
depends: — (lane C: CARDS.md B-15 names get_temperature "on contact")

### S-A-CORE-2-06 | DEAD | A
subject: src/mantis/selfplay/hparams.py::build_runner_config (the `else:` dense-dims arm)
claim: `is_graph_representation(spec)` never returns False, so the arm that computes a non-zero feat_len/chain_len is unreachable. The `trunk_size` local is used only there.
evidence: `sed -n 233,241p` -> `if is_graph_representation(spec): dims = PoolDims(0, 0, …) else: dims = PoolDims(n_kept_planes*trunk*trunk, 6*trunk*trunk, …)`. The function's own body is read in full above.
callers:
  AST: the arm is intra-function, so no external caller applies. `git grep -n "is_graph_representation" -- src tests` -> no monkeypatching of it anywhere
  other channels: n/a (a branch, not a symbol)
Δlines: -7 (`sed -n 235,240p` = 6, plus the `trunk_size` local). The check stays as a bare call.
witness: tests/selfplay/test_pool_hparams_arms.py::test_pool_dims_derivation_golden (expects PoolDims(0, 0, 362) on both graph encodings)
depends: —

### S-A-CORE-2-07 | DEAD | A
subject: src/mantis/selfplay/buffers.py::_RAW_FOR (and its `HexgBuffer` import), ::BufferKindMismatch
claim: `_RAW_FOR` is defined and never read. `BufferKindMismatch` is never raised anywhere. Two tests import it and never use it.
evidence: census.tsv -> `_RAW_FOR` self1 (definition only); `grep -n "HexgBuffer" buffers.py` -> import + `_RAW_FOR` + a comment; `git grep -n "BufferKindMismatch(" -- src tools` -> only the class line; `grep -n BufferKindMismatch tests/selfplay/test_buffer_facade.py tests/selfplay/test_pool_surface.py` -> import lines only
callers:
  AST imports: `git grep -n -w -E "_RAW_FOR|BufferKindMismatch" -- src tools tests` -> selfplay/__init__ re-export + the two unused test imports
  entry points / python -m / subprocess: none
  importlib/getattr: none. CALLERS §7 lists `_RAW_FOR` as a name-keyed table, but nothing indexes it
  conftest / pyo3 / config / gates / STATE: none
Δlines: -11 (the `_RAW_FOR` comment and definition 3, the import 1, the class 3 plus 2 separating blanks, and 2 re-export lines in selfplay/__init__). The 2 test import lines are edited in place.
witness: tests/selfplay/test_buffer_facade.py (import line)
depends: —

### S-A-CORE-2-08 | DEAD | A
subject: src/mantis/selfplay/pool_hooks.py::RunnerStats.runner_encoding
claim: This is a compatibility shim for a state the tree no longer has (R10). Its own comment reads "Vestigial `None` slot, kept so kwarg constructions do not break". runner_stats() never sets it and no constructor passes it.
evidence: `grep -rn runner_encoding src tests` -> the field + tests/selfplay/test_pool_surface.py::RUNNER_STATS_FIELDS. `git grep -n -E "RunnerStats\(" -- src tools tests` -> only runner_stats(), which omits it. `git grep runner_encoding -- docs tools tests/fixtures` -> none.
callers:
  AST / attr: the greps above. No `.runner_encoding` read anywhere
  entry points / python -m / subprocess: none
  importlib/getattr: pool_hooks' getattr-by-string counters (CALLERS §7) do not include it
  conftest / pyo3 / config / gates / STATE: none. It is not serialised (`asdict` sites are listed by grep, and none is on RunnerStats)
Δlines: -3 (a 2-line comment + the field), plus the one set-literal entry in test_pool_surface.py
witness: tests/selfplay/test_pool_surface.py (the exact RUNNER_STATS_FIELDS set)
depends: —

### S-A-CORE-2-09 | DEAD | A
subject: src/mantis/_engine.pyi: `MY_STONE_PLANE`, `OPP_STONE_PLANE`, `MOVES_REMAINING_PLANE`, `PLY_PARITY_PLANE`
claim: The stub declares four constants the compiled module does not export (they were retired with the grid path).
evidence: runtime `hasattr(mantis._engine, n)` -> only HEX_AXES and WIN_LENGTH are present. An AST diff of every pyi top-level name, class and method against the runtime -> these four are the ONLY missing names.
callers:
  AST imports: `git grep -n -w -E "MY_STONE_PLANE|OPP_STONE_PLANE|MOVES_REMAINING_PLANE|PLY_PARITY_PLANE"` -> the two pyi twins, tools/hardcode_scan.py (regex/label strings, not reads), and one test docstring
  pyo3: the Rust `add_*` registrations export none of them (the runtime check above)
  other channels: none (a stub is never imported)
Δlines: -4 per twin, plus the owner-comment lines 583-584 trimmed
witness: tests/bridge/test_engine_stub_twins_agree.py (the twins must change in one commit)
depends: HANDOFF R3 (crates/mantis-bridge/python/mantis/_engine.pyi, the wheel twin)

### S-A-CORE-2-10 | DEAD | C
subject: src/mantis/selfplay/pool.py::WorkerPool.latest_replay_path → pool_hooks.py::latest_replay_path, ::RecorderLike.latest_replay_path, ::NullRecorder.latest_replay_path
claim: No production code asks the pool for a replay path, and the one production recorder answers None by design (monitor/game_recorder.py docstring: "returns `None` and that is not an oversight").
evidence: `git grep -n -w latest_replay_path -- src tools tests` -> only the definitions, the forwarder, the Protocol member and tests (test_pool_lifecycle asserts, test_pool_surface FROZEN_METHODS + H-01, and 3 test fakes)
callers:
  AST / attr: the grep above; `git grep -n -E "\.latest_replay_path\b" -- src tools` -> only the forwarders
  entry points / python -m / subprocess / importlib: none. producer_manifest.yaml does not name it
  conftest: none
  pyo3/config/gates/STATE: none. The dashboard reads events, not this method
Δlines: -9 in slice (AST spans 3+3+1+2), plus the monitor side (HANDOFF C3)
witness: tests/selfplay/test_pool_surface.py FROZEN_METHODS (an exact set), tests/selfplay/test_pool_lifecycle.py
depends: — (lane C: pool.py is in the PZ glob; the frozen-surface test edit is also a design choice)

### S-A-CORE-2-11 | DEAD | C
subject: src/mantis/selfplay/graph_collate.py::WIN_AXES
claim: This is a derived copy of `_engine.HEX_AXES` that nothing in src/ or tools/ reads. Its one test asserts that it equals the table it is built from.
evidence: `git grep -l -w WIN_AXES -- src tools` -> graph_collate.py (definition + `__all__`), selfplay/__init__ (re-export), tools/hardcode_scan.py (regex string). The test is tests/encoding/test_geometry_crosses_the_ffi.py::test_every_python_axis_table_is_the_engines.
callers:
  AST imports: the grep above
  entry points / subprocess / importlib / conftest / pyo3 / config / gates / STATE: none
Δlines: -4 (the comment and definition), -1 `__all__`, -1 re-export. One assert line in the test is edited; no function is removed.
witness: tests/encoding/test_geometry_crosses_the_ffi.py
depends: — (lane C: graph_collate.py is PZ-1/PZ-2)

### S-A-CORE-2-12 | SIMPLIFY | C
subject: src/mantis/config/resolve/encoding.py (::reconcile_encoding's stamp arms, ::EncodingConflictError, ::EncodingResolution, ::UNSPECIFIED, ::normalize_declared, ::normalize_stamp) + config/emit.py::_SOURCE_REMAP
claim: The only production call is `reconcile_encoding(cfg.identity.encoding, None)` in emit.py::resolve_config. The declared value is a required schema key and the stamp is always None, so only the "variant wins" arm runs. normalize_declared and normalize_stamp are test-only. The conflict, checkpoint and absent arms cannot fire in production.
evidence: `git grep -n -w -E "normalize_declared|normalize_stamp|reconcile_encoding|EncodingConflictError|AbsentEncodingError" -- src tools tests` -> emit.py:96 is the only production call; everything else is re-exports (config/__init__, resolve/__init__) and tests/config/test_resolve_encoding.py (8 tests)
Δlines: at least -40 (AST spans: EncodingConflictError 16, EncodingResolution 12, UNSPECIFIED plus its comment 4, normalize_* 8). The module is 104 lines (`wc -l`).
witness: tests/config/test_resolve_encoding.py; gate 13
depends: — (lane C: docs/contracts/run_config_schema.md cites "mantis.config.resolve.encoding (`AbsentEncodingError`)", and the contract doc is PZ-2)

### S-A-CORE-2-13 | DEAD | C
subject: src/mantis/config/resolve/nsims.py::resolve_eval_model_sims (+ ::_KNOWN_OPPONENTS)
claim: Its one production caller, bots/resolve.py::resolve_bot, DISCARDS the return value. The call runs only after `kind in _KNOWN_KINDS` (the same three names as `_KNOWN_OPPONENTS`) and only when `opponent_sims is not None`, so it can neither raise nor change anything. The CONSUMER_REGISTRY entry for eval.random_model_sims therefore cites a no-op. The key's real readers are eval/pipeline.py -> RoundSpec -> eval/worker.py.
evidence: `sed -n 93,110p src/mantis/bots/resolve.py`; `grep -n _KNOWN_KINDS src/mantis/bots/resolve.py` -> `("random", "sealbot", "strix")`; `git grep -n random_model_sims -- src tools` -> pipeline.py:490, worker.py:242/335/538…
callers:
  AST imports: `git grep -n -w resolve_eval_model_sims -- src tools` -> bots/resolve.py:104 (the no-op), plus re-exports and a schema docstring
  entry points / python -m / subprocess / importlib / conftest / pyo3 / gates / STATE: none
  config keys: cited by CONSUMER_REGISTRY (both twins) and test_regime_parity{,_p2}; tests/config/test_resolve_nsims.py has 5 tests
Δlines: -29 (`wc -l`), -2 at the call site, -4 re-export lines
witness: tests/config/test_every_key_has_consumer*.py::test_every_registry_string_names_symbols_that_exist (the strings must be re-pointed); tests/config/test_resolve_nsims.py
depends: HANDOFF C3 (the bots/resolve.py call site) (lane C: run_config_schema.md names "mantis.config.resolve.nsims" twice, gate 13)

### S-A-CORE-2-14 | CONFIG | C
subject: schema key `train.value_target` (src/mantis/config/schema/train.py `value_target: Literal["pure_outcome_z"]`)
claim: This is the only single-value Literal among the 160 leaves. Its sole consumer is TrainHParams.from_config's `!= "pure_outcome_z"` assertion, which the loader makes unreachable, and `TrainHParams.value_target` is stored but never read. LAW-08: a key with no live consumer.
evidence: a single-Literal scan over `leaf_paths(RunConfig)` -> only `train.value_target`; `git grep -n -w value_target -- src tools` -> schema, the TrainHParams field + assert (trainer/core.py), the checkpoints.py literal and the template; `git grep -n "\.value_target"` -> no reader
Δlines: about -14 (the schema field, the dataclass field, a 2-line assert, one row in each of 6 minted configs, the template and the checkpoints literal); both CONSUMER_REGISTRY rows also go
witness: tests/config/test_every_key_has_consumer*.py (the bijection); gate 7
depends: — (lane C: touching minted rows is a HALT under R322/R327 and needs a frozen-file grant; the contract doc and the leaf count 160 move too)

### S-A-CORE-2-15 | SIMPLIFY | B
subject: src/mantis/selfplay/__init__.py (the 70-plus re-exports and `__all__`)
claim: Not one consumer in src/, tools/ or tests/ imports a SYMBOL from the `mantis.selfplay` package. All of them import submodules (`mantis.selfplay.pool`, `from mantis.selfplay import pool_drain`, …). The docstring's layout also names a `worker` module that does not exist.
evidence: `git grep -n -E "from mantis\.selfplay import \(?[A-Za-z_]" -- src tools tests` filtered to non-submodule names -> none; `ls src/mantis/selfplay/` -> no worker.py
Δlines: -107 (`wc -l` 125, keeping an 18-line docstring)
witness: gate 9 (the import DAG); every selfplay test at import
depends: — (lane B: the __init__ import order may be masking a submodule cycle, and that cannot be probed here without torch)

### S-A-CORE-2-16 | SIMPLIFY | C
subject: one-line forwarders: config/resolve/actor_sync.py::resolve_actor_sync_cadence (`return int(train_section.actor_sync_cadence_steps)`), config/resolve/run_length.py::resolve_max_train_steps (same shape), config/resolve/leaf_build_threads.py::resolve_leaf_build_threads (`return resolve_sample_threads(...)`), and run.py's second-layer wrappers ::_resolve_monitor_cfg and ::_resolve_actor_sync_cadence_steps (each forwards to a forwarder; their docstrings describe arms that are already retired)
claim: Each is a pure forwarder with one reader, or a forwarder of one.
evidence: files read in full; `git grep -n -w` on each name -> run.py (+ preflight_mint_parent.py and schema/train.py for run_length; three tools for leaf_build_threads) plus tests (run.py wrappers: test_run_strict_composition, test_actor_lag_wiring_live, test_actor_sync_real_config)
Δlines: run.py -14 (AST spans 7+7). The resolver modules come to -85 (`wc -l` 20+27+38) if they are dissolved.
witness: tests/config/test_every_key_has_consumer*.py (CONSUMER_REGISTRY strings cite resolve_actor_sync_cadence and resolve_max_train_steps); the three run.py tests
depends: — (lane C: run_config_schema.md lists `actor_sync` and `run_length` in the "one resolver per regime knob" row (gate 13); run.py is in the PZ glob. The one-read-site rule is deliberate architecture, so this is a design choice, not a mechanical cut.)

### S-A-CORE-2-17 | SIMPLIFY | C
subject: src/mantis/run.py::_lazy_save_anchor, ::_lazy_guarded_load
claim: Both lazily import `mantis.train.anchor`, which run.py already imports at top level (`from mantis.train.anchor import canonical_anchor_path`). They can be passed directly as `save_anchor=` and `guarded_load=`.
evidence: `grep -n "mantis.train.anchor" src/mantis/run.py` -> the top-level import at line 81 plus the two lazy imports; `git grep -n -E "save_best_model_atomic|_guarded_load_state_dict" -- tests src` -> no test monkeypatches the late-bound name
Δlines: -8 (AST spans 4+4 and their blank lines; the top import grows by 2 names, about +2)
witness: tests/test_run_composition.py (compose_run)
depends: — (lane C: run.py is in the PZ-1 glob via _restore_resume_state, which this does not touch)

### S-A-CORE-2-18 | DUP | A
subject: src/mantis/config/resolve/__init__.py `__all__`
claim: Four names are listed twice: SEARCH_KINDS, MissingSearchKindError, resolve_deploy_search_kind and resolve_selfplay_search_kind.
evidence: a Counter over the `__all__` string entries -> those 4 appear twice
deliberate?: no. The two blocks are verbatim copies.
Δlines: -4
witness: tests/encoding/test_no_dead_resolver_export.py-style export pins (none pin duplicates)
depends: —

### S-A-CORE-2-19 | DOC | A
subject: src/mantis/selfplay/hparams.py::InferenceHParams docstring ("same R1-exception as `SelfPlayHParams`"), the stray `# diagnostics ns` comment, and the code-side defaults on SelfPlayHParams/InferenceHParams fields
claim: tests/config/test_docstring_debt_discharge.py records that the R1-exception is DISCHARGED, so the docstring is stale. Every field default is dead: the only construction is `from_config`, which passes every field.
evidence: `git grep -n -E "SelfPlayHParams\(|InferenceHParams\(" -- src tools tests` -> 0 direct constructions
Δlines: -1 (the stray comment). Dropping the `= default` values is line-neutral, keeps R1 honest, and makes `kw_only` the only ordering tool left.
witness: tests/selfplay/test_pool_hparams*.py
depends: —

### S-A-CORE-2-20 | DOC | A
subject: src/mantis/selfplay/inference_local.py::LocalInferenceEngine class docstring and ::close docstring
claim: The docstrings say "Wrap a grid or graph net…", "Dense (grid): build (K, C, trunk, trunk) tensors…", "GRID callers pass `None`" and "a no-op for a dense engine". The class refuses grid at construction through is_graph_representation.
evidence: file read in full (lines 18-33 and 103)
Δlines: about -3
witness: NONE
depends: S-A-CORE-2-04 (same file)

### S-A-CORE-2-21 | DUP | B
subject: src/mantis/config/preflight_stamp.py::_flat_leaves vs tools/config_diff.py::_flatten
claim: Both are the same recursive dotted-key dict flattener (`f"{prefix}.{k}" if prefix else k`). R10 asks for one implementation per thing.
evidence: `git grep -n -E "f\"\{prefix\}\.\{"` -> these two plus schema/leaves.py (a schema walker, different input); both bodies read
deliberate?: neither states a reason. tools/ may import src (config_diff already imports mantis.config.loader).
Δlines: -8 (AST span of `_flatten`), with `_flat_leaves` made public
witness: tests/config/test_config_diff_from_header.py, tests/config/test_mint_and_diff.py
depends: HANDOFF (the tools slice owns config_diff.py)

### S-A-CORE-2-22 | DEAD | B
subject: src/mantis/config/resolve/allocator_posture.py::AllocatorPostureSpec.required_env
claim: A production class carries a helper with no production reader. Its only user builds a child env in a test.
evidence: `grep -rn required_env src tests tools` -> the definition + tests/tools/test_preflight_mint_process.py:547. The frozen archive RULINGS_ACTIVE mentions it historically.
callers:
  AST / attr: the grep above
  entry points / python -m / subprocess: preflight_mint.py builds its env without it (grep -> none)
  importlib / conftest / pyo3 / config / gates / STATE: none
Δlines: -11 (AST span)
witness: tests/tools/test_preflight_mint_process.py (would inline the env)
depends: —

### S-A-CORE-2-23 | DEAD | C
subject: the dense self-play arm: selfplay/pool_drain.py::run_stats_loop `else:` (collect_data + push_dense), selfplay/pool_push.py::push_dense, hparams.py::PoolDims dense members, and pool.py's `_feat_len`/`_chain_len`/`_pol_len`/`_board_size`
claim: This is unreachable for the same reason as -03 (`pool._is_graph` is always True). It is CARDED: CARDS.md B-15 keeps push_dense as the drain-parity instrumentation oracle, and `tests/fixtures/selfplay/pool/runner_config_goldens.json` pins `_board_size`. Reported as carded, not proposed.
evidence: the attribute scan -> `_chain_len`, `_pol_len` and `_board_size` are read only by tests; CARDS.md "the dense push arm (`push_dense`, B-15 — dead in production since R346(f) …)"
Δlines: not derived (a ruling-gated re-base of six oracles)
witness: tests/selfplay/test_pool_drain_parity.py and the drain goldens
depends: — (lane C: CARDS.md B-15 plus PZ-2 goldens)

## DEFECTS
- src/mantis/data/corpus_metrics.py runs `REPORT_DIR.mkdir(parents=True, exist_ok=True)` at IMPORT time, so importing it (tests do) creates `reports/corpus_analysis` in the CWD. data/__init__.py's docstring names this side effect.
- src/mantis/selfplay/inference_server.py::submit_and_wait would reshape to `self._shape`, which is None on the only live (graph) arm, so it would also fail if anyone called it (see -03).

## PARKED
- src/mantis/selfplay/__init__.py eagerly imports every submodule (torch included) whenever any `mantis.selfplay.<mod>` is imported (see -15).

## HANDOFF
- C3: eval/worker.py::build_candidate_player's `representation == "grid"` arm is unreachable (see -04); monitor/game_recorder.py::GameRecorder.latest_replay_path is the tail of -10; bots/resolve.py:104 is the no-op call site of -13; monitor/config.py::MonitorConfig carries code-side defaults (R1) on every field; util/constants.py::HISTORY_LEN loses its only src consumer with -02.
- C1: train/axis_distribution.py comments cite `env.game_state._HEX_AXES`, which does not exist; train/checkpoints.py's full-config literal carries the `value_target` row of -14 (CARD-MECHANISM-SWEEP "nine full-config literal dicts").
- R3: crates/mantis-bridge/python/mantis/_engine.pyi is the twin edit of -09.
- tools slice: tools/config_diff.py::_flatten (-21); tools/audit_bootstrap_corpus.py path-cites data/sources/human.py (-01 blocker).
- tests slices: tests/data/test_no_identity_blind_board.py's docstring says data/corpus_metrics.py "went with the grid path", but the file exists; tests/fixtures/search_golden/temperature/gen_temperature_parity_golden.py says "the Python eval/bot path checks the same fixture", but no Python test reads it (see -05).

## Not covered
- src/mantis/config/armed_aborts.py::MANIFEST (420 of 1166 lines): the gate-12 rows were not audited row by row. They are lane C by ruling.
- config/schema/core.py cross-field validators: all are pydantic-decorated and live by decorator. The ARCH_SCOPED_KEYS / `refuse_outside_its_arch` grid/graph partition is a PZ-2 seam member and was deliberately not proposed.
- run.py::compose_run (442 lines): only its first ~170 lines were read for dead branches.
- graph_collate.py, collate_dump.py, instrumentation.py and pool_push.py (PZ) were only symbol-scanned, not read in full.
- The 160 config leaves were spot-checked, not traced one by one: value_target, draw_reward/ply_cap_value, monitor axis/heartbeat/supervisor, eval worker_device/random_floor_games/concurrency/random_model_sims, total_steps/scheduler_t_max. Rust-side consumption of selfplay.* keys belongs to the R slices.
- Nothing that imports torch was executed, so none of the lane-A items has a runtime probe here.
