# S-A-CORE-1 — slice C1 (src/mantis/train, model, encoding)
scope: src/mantis/train/**, src/mantis/model/**, src/mantis/encoding/** (55 files, 12 029 lines at 69e1532).
method: static only (torch not importable). Lead generator `uvx vulture src/mantis tools tests --min-confidence 60`;
an AST census of every top-level def/class/const/method in the slice with a whole-tree `git grep -n -w <name>` count, split
code / `__init__` / tests / docs / own file (scratchpad `c1/defs2.py` -> `c1/refs2.tsv`, 603 rows); an AST
package-import census (`from <pkg> import X` over every tracked .py); an AST unused-parameter scan; an AST
text-IO-without-`encoding=` scan; protocol-member access census over `coordinator/config.py` + `events.py`; targeted reads.
`.venv/bin/python -c "from mantis.config.schema import RunConfig, leaf_paths; ..."` for config-key checks.

Checklist shorthand used below (each DEAD row states which channels returned what):
- **grep** = `git grep -n -w <name>` over the WHOLE tree (src, tools, tests, crates, docs, configs, Makefile, *.sh,
  *.toml, *.yaml). It matches from-imports, `mod.X` attribute access, `__init__` re-exports, string-embedded
  `python -c` code, subprocess argv strings and getattr/registry string keys of the bare name, so one search covers
  the AST, subprocess, getattr/registry and conftest channels for a Python symbol.
- **str** = `git grep -n -E "[\"']<name>[\"']"` (string-keyed lookups).
- **ep/-m/gates** = CALLERS.md §1-4 re-read: no `[project.scripts]`/entry-points; the slice's only `-m` targets are
  `mantis.encoding` (-> audit.main), `mantis.train.pretrain` (-> cli.pretrain), `mantis.train.lifecycle.arm_exec`;
  run_all.sh / Makefile name no slice symbol.
- **STATE** = `grep -n <name> docs/governance/STATE.md` (STATE names only `mantis.train.mixing._steps_budget` from this slice).
- **pyo3** = not applicable (pure-Python symbols; none is an `_engine` name).
- **cfg** = `leaf_paths(RunConfig)` filtered for the name -> no leaf, and CONSUMER_REGISTRY strings grepped.

## Summary
32 findings. By class: DEAD 16, SIMPLIFY 10, DUP 4, DOC 2, CONFIG 0, TEST 0, ONE-SHOT 0, PACK 0 (02 is DEAD with a
contract-doc half). By lane: A 6, B 4, C 22 (the slice is mostly protected: model/** and encoding/** are PZ-2
globs, and 16 of the train files are PZ-1/PZ-2/PZ-6 paths, so most of the slice is lane C).
Already carded and NOT re-raised as new (CARD-STYLE-BACKLOG B-14/B-15, CARD-MECHANISM-SWEEP): the F1 defer path
(`declared_keys`/`declared_lr`, `apply_config_overrides_f1`, `RESUME_DIRECTIVE_KEYS`), the `get_temperature`
re-export, `collate_graph_batch(device=None)`, the two segment softmaxes, the double `torch.load`, and
`hashlib.sha256(path.read_bytes())` in `encoding/__init__.py::_registry_sha_handshake`.
Top 3 by Δlines: S-A-CORE-1-24 (-58, the queued-dead resolver cluster), S-A-CORE-1-07 (-50+, try_save_buffer chain),
S-A-CORE-1-19 (≈ -50, pretrain dense-arm flag surface).

## Findings

### S-A-CORE-1-01 | DEAD | C
subject: src/mantis/train/losses.py::compute_chain_loss, ::chain_target_fire_rate
claim: the grid-era chain-plane Huber loss and its LAW-18 fire-rate helper have no caller anywhere; no GNN has a chain head.
evidence: vulture -> "unused function 'compute_chain_loss' / 'chain_target_fire_rate' (60%)"; `git grep -n -E "chain_target_fire_rate|compute_chain_loss"` -> only the two defs.
callers: grep -> 1 hit each (the def). str -> 0. ep/-m/gates -> none. STATE -> 0. pyo3 n/a. cfg -> `leaf_paths` has no `*chain*` leaf; `aux_chain_weight` exists only in a cli.py help string and event_manifest.md. conftest -> 0 (tests/tools/test_tier_census.py carries the synthetic tuple `("tests/train/test_losses.py","test_chain_loss_math_is_smooth_l1","slow")`, a string fixture for a file that does not exist, not a caller).
Δlines: -38 (AST spans 21 + 17) plus the module docstring's "and the chain head".
witness: NONE (no test imports either).
depends: 02 (the `loss_chain` row is its reader's shadow). Lane C because losses.py carries the finite-gradient guard (PZ-1).

### S-A-CORE-1-02 | DEAD | C
subject: src/mantis/train/events.py::emit_training_step_event (10 producer-less rows) + docs/contracts/event_manifest.md training_step roster
claim: `loss_aux`, `loss_ownership`, `loss_threat`, `loss_chain`, `avg_sigma`, `policy_entropy_pretrain`, `policy_entropy_recent`, `n_rows_policy_loss`, `n_rows_total`, `value_accuracy` read `loss_info` keys no trainer tail produces, so they are None on every emission; the contract doc still says "the two trainer tails … `value_accuracy` (dense only) and `loss_chain` (dense, when `aux_chain_weight > 0`)" and lists an `aux_chain_loss` event with no emitter.
evidence: per key `git grep -n -E "[\"']<k>[\"']" -- src tools | grep -v measured` -> 0 for opp_reply_loss, ownership_loss, threat_loss, chain_loss, avg_sigma, policy_entropy_pretrain, policy_entropy_recent, n_rows_policy_loss, n_rows_total, value_accuracy (only `policy_entropy_selfplay` and `policy_target_entropy` are produced, trainer/core.py). `git grep -n aux_chain_loss` -> event_manifest.md + an error-message string in tests/test_run_composition.py.
callers: readers of the payload fields: `git grep -n -w <field> -- tools src docs/contracts tests` -> only event_manifest.md and tests/train/test_training_step_absence.py::ABSENCE_CAPABLE (the dashboard reads none of them). str/cfg as above -> 0.
Δlines: -10 (the ten dict rows, `grep -c`), plus the doc roster and the test's ABSENCE_CAPABLE tuple.
witness: tests/train/test_training_step_absence.py (pins the tuple; must be edited in the same commit).
depends: 01. Lane C: event_manifest.md is a PZ-2 contract doc; events.py is a PZ glob.

### S-A-CORE-1-03 | DEAD | C
subject: src/mantis/train/events.py::emit_training_step_event params `qfire_delta`, `early_game_probe`, `trainer_model`, `solver_deltas`; ::EARLY_GAME_ENTROPY_WARN_THRESHOLD
claim: the one production caller (coordinator/step.py::StepCoordinator._emit_training_step) passes `None` for qfire_delta and nothing for the other three, so the early-game-probe block and the solver merge never run.
evidence: `git grep -n -A8 "emit_training_step_event(" -- src` -> one call, args `(self._train_step, loss_info, None, sink)`; `git grep -n -E "early_game_probe|EARLY_GAME_ENTROPY|trainer_model=" -- . ':!src/mantis/train/events.py'` -> 0.
callers: grep -> tests call with 4 positionals only (test_training_step_absence.py); `solver_deltas` is pinned as a PREMISE by tests/train/test_target_counter_events.py::test_the_target_integrity_parameter_has_no_default ("queues it (`Q-O-SOLVERDELTAS`)", a queue row absent from docs/). str/cfg -> 0.
Δlines: -13 (the probe block, sed range) -4 params -1 constant; `quiescence_fires_per_step` stays (schema-stable None) or becomes a literal None.
witness: test_target_counter_events.py (premise must be re-pointed, its docstring says "never deleted").
depends: 02. Lane C (events.py PZ glob, test premise ruling-adjacent).

### S-A-CORE-1-04 | DEAD | C
subject: src/mantis/train/events.py::emit_axis_distribution params `baseline`, `tb_writer` and the tensorboard block; coordinator/step.py `getattr(self.subsystems, "axis_baseline"|"tb_writer", None)`
claim: production's `subsystems` is `SimpleNamespace(gpu_monitor=None, disk_guard=...)` (src/mantis/run.py), so `tb_writer` is always None and `baseline` always `{}`; no test passes either.
evidence: `git grep -n -E "tb_writer|axis_baseline|log_step" -- tests` -> 0; `git grep -n tb_writer -- src` -> events.py + the one step.py getattr.
callers: grep -> as above; str -> the two getattr strings only; cfg -> no tensorboard key.
Δlines: -13 (tb block, `sed -n '/if tb_writer is not None:/,/axis_distribution_tb_failed/p' | wc -l`) -2 params -2 getattr lines.
witness: NONE found.
depends: 06. Lane C (events.py, step.py PZ).

### S-A-CORE-1-05 | SIMPLIFY | C
subject: src/mantis/train/events.py::emit_iteration_complete_event params `config`, `mcts_config`, `capacity`, `w_pre`
claim: the first three are never read in the body; `w_pre` is the literal `0.0` at the only caller (step.py::_emit_iteration_complete), so `iteration_complete.corpus_selfplay_frac` is a constant 1.0 published as if measured (the mixing it measured was deleted by R346(f)).
evidence: AST unused-param scan -> `emit_iteration_complete_event: ['config', 'mcts_config', 'capacity']`; vulture -> "unused variable 'mcts_config' (100%)"; `git grep -n corpus_selfplay_frac` -> events.py + two RULINGS quotes, no dashboard/contract reader.
deliberate?: n/a.
Δlines: -5 (params + `w_pre = 0.0`) at the call site and def; 3 test callers drop kwargs (test_augment_sym_counter, test_quiescence_fires_producer, test_rates_are_measured).
witness: those three tests.
depends: —. Lane C (event field removal is a contract change; events.py PZ glob).

### S-A-CORE-1-06 | DEAD | C
subject: src/mantis/train/coordinator/step.py::StepCoordinator.__init__ params `pretrained_buffer`, `bot_buffer`, `bufs`, `train_cfg`, `batch_size_cfg`, `iterations`, `mixing_cfg`, `recent_buffer`
claim: the first six are stored and never read; `mixing_cfg` is read only for a `buffer_persist_path` production never sets (run.py passes `{}`); `recent_buffer` is always None in production and only reaches dispatch.py's refusal ("the graph route takes no dense recent_buffer").
evidence: `git grep -n -E "\.(pretrained_buffer|bufs|train_cfg|mixing_cfg)\b" -- src tools tests` -> only step.py:254 (`mixing_cfg.get`); bot_buffer / batch_size_cfg reads -> 0; run.py passes `pretrained_buffer=None, recent_buffer=None, bufs=None, train_cfg={}, mixing_cfg={}`.
callers: 24 construction sites (`git grep -l "StepCoordinator(" -- tests src | wc -l`); tests/config/test_coordinator_knobs_wiring.py passes `pretrained_buffer=`/`bot_buffer=`/`mixing_cfg=` with no observable effect; SEAM_MATRIX row `(step_mod, ("buffer","pretrained_buffer","bot_buffer"), …)` names the aliases.
Δlines: ≈ -16 in step.py (15 grep hits on these names in step.py) plus the kwargs at 24 sites and the dispatch `recent_buffer` parameter chain.
witness: tests/train/test_trainer_seam_conformance.py (SEAM_MATRIX), test_train_step_dispatch.py (the recent_buffer refusal test).
depends: 07. Lane C (step.py PZ-1, SEAM_MATRIX PZ-2).

### S-A-CORE-1-07 | DEAD | C
subject: src/mantis/train/buffer_persist.py::try_save_buffer, ::buffer_save_errors_total; coordinator/config.py::RecentBufferLike; the `monitor_gates.buffer_save_errors_total` field in step.py
claim: `try_save_buffer` has no production caller, so the counter it feeds is published as a permanent 0 in every `monitor_gates` event; `RecentBufferLike` is policed only through SEAM_MATRIX's `persist_mod` rows that read this function.
evidence: `git grep -n -w try_save_buffer -- src tools` -> def + a comment in buffer_persist.py only; `leaf_paths` has no `*persist*`/`*buffer*` leaf (the `buffer_persist` switch it reads is not a config key).
callers: grep -> tests/train/test_buffer_persist_counter.py (5 tests), test_survivability.py (2 call sites); str -> `"buffer_persist"` only inside the function; STATE -> 0; `canonical_buffer_path` stays (run.py, step.py).
Δlines: -40 (AST span) -1 counter -10 RecentBufferLike (AST) -1 monitor_gates row; test-floor move (test_buffer_persist_counter.py = 5 tests; tools/ci_gates/test_count_floor.txt ratchet-down needed).
witness: test_buffer_persist_counter.py, SEAM_MATRIX persist rows in test_trainer_seam_conformance.py, test_no_phantom_seam_member.py (RecentBufferLike would become an orphan if left).
depends: 06. Lane C (step.py PZ-1, protocols PZ-2).

### S-A-CORE-1-08 | DEAD | C
subject: src/mantis/train/coordinator/config.py::TrainerLike.train_step_from_tensors, ::GridRouteBufferLike (+ `sample_batch_with_pos`), their coordinator/__init__.py re-exports
claim: phantom seam members for the deleted grid route: no production trainer defines `train_step_from_tensors`, no engine buffer defines `sample_batch_with_pos`, nothing in src/tools calls either.
evidence: protocol-member census -> "TrainerLike.train_step_from_tensors: 0 src/tools accesses", "GridRouteBufferLike.sample_batch_with_pos: 0"; `grep -n "def train_step_from_tensors" src/mantis/train/trainer/core.py` -> none; `git grep sample_batch_with_pos` -> config.py + one test stub.
callers: grep -> 13 test files define a `train_step_from_tensors` stub (HANDOFF); SEAM_MATRIX dispatch row lists GridRouteBufferLike in its union.
Δlines: -1 member -5 class (AST) -2 re-export lines; plus stub methods in 13 test files (test-slice work).
witness: test_trainer_seam_conformance.py, test_no_phantom_seam_member.py.
depends: 14. Lane C (PZ-2 trainer/server protocols).

### S-A-CORE-1-09 | DEAD | C
subject: src/mantis/train/coordinator/config.py::promotion_capable_rounds
claim: "Surfaced at launch" per its docstring, but no src/tools code calls it; it survives only as a re-export and as a citation inside two CONSUMER_REGISTRY strings.
evidence: `git grep -n -w promotion_capable_rounds` -> config.py def, coordinator/__init__.py (2), tests/config/test_every_key_has_consumer{,_p2}.py (`"… round boundary (+ promotion_capable_rounds)"`).
callers: grep as above; str -> 0; ep/-m/gates -> 0; STATE -> 0; cfg -> the `train.eval_interval` registry string cites it, so deletion reds `test_every_registry_string_names_symbols_that_exist` unless the string is trimmed in the same commit.
Δlines: -8 (AST) -2 re-export; two registry strings shortened.
witness: tests/config/test_every_key_has_consumer.py, _p2.py.
depends: 14. Lane C (config.py PZ glob; the registry claim is a consumer contract).

### S-A-CORE-1-10 | DOC | C
subject: src/mantis/train/coordinator/dispatch.py (R8 header, module docstring, ::run_declared_train_step docstring); step.py comment naming `train_step_from_tensors`
claim: the header justifies >300 lines by "the graph arm and the grid arm have to be read together"; the docstrings route "grid to `train_step_from_tensors`", cite `_grid_step` and "four FROZEN grid coordinators" — none exist.
evidence: `git grep -n -E "train_step_from_tensors|_grid_step" -- src` -> dispatch.py:9, :114, step.py:930, config.py (08).
Δlines: 0 net (the file stays 343 > 300, so the R8 header is REWORDED, not dropped — gate 15).
witness: gate 15 (header must still state a reason, no count).
depends: 08. Lane C (dispatch.py PZ-1 dump path).

### S-A-CORE-1-11 | SIMPLIFY | A
subject: src/mantis/train/__init__.py (re-exports of EventSink, NullEventSink, emit_via + 7 lifecycle names)
claim: no module imports any of the ten names from `mantis.train`; every consumer imports the submodule.
evidence: AST package-import census -> `mantis.train {'checkpoints','bundle','events','orchestrator','resume_state'}` (submodules only); `git grep -n -E "mantis\.train\.(EventSink|…|SELFPLAY_STALL_EXIT_CODE)"` -> 0; docs/yaml/toml/sh -> 0.
callers: grep of each bare name finds only its defining module, lifecycle/__init__.py and tests importing from the submodule. The docstring is also stale ("`aux_decode`", "land in later slices"; `git grep aux_decode -- src tests tools` -> this docstring only).
Δlines: -25 (37 lines, docstring ends line 12; everything after it).
witness: gate 9 (import DAG), gate 14 pyright.
depends: 12. Probe: `import mantis.train` no longer imports lifecycle eagerly; no test relies on it (`git grep "import mantis.train\b"` -> one doctored string in tests/selfplay/test_pool_surface.py for a DAG check).

### S-A-CORE-1-12 | SIMPLIFY | A
subject: src/mantis/train/lifecycle/__init__.py (8 re-exports)
claim: the only importer of the package-level names is train/__init__.py (11); every other site imports `mantis.train.lifecycle.<module>`.
evidence: AST census -> `mantis.train.lifecycle {` the 7 names ×1 (train/__init__.py) `, 'signals': 5}`.
Δlines: -22 (33 lines, docstring ends line 11). The docstring's two-watchdog note can stay.
witness: gate 9, gate 14.
depends: 11.

### S-A-CORE-1-13 | SIMPLIFY | A
subject: src/mantis/train/trainer/__init__.py (re-exports Trainer, TrainHParams, build_param_groups)
claim: zero package-level importers; the only two importers take `core`.
evidence: AST census -> `mantis.train.trainer {'core': 2}`; `git grep -n -E "mantis\.train\.trainer\.(Trainer|TrainHParams|build_param_groups)"` -> 0.
Δlines: -9 (10 lines, docstring is line 1).
witness: gate 9, gate 14.
depends: —.

### S-A-CORE-1-14 | SIMPLIFY | A
subject: src/mantis/train/coordinator/__init__.py (14 re-exports)
claim: only `StepCoordinator` is imported through the package (tests/train/test_heldout_gap.py); the other 13 names have no package-level importer, and the docstring narrates retired items (`bot_refresh.py` "DEFINITE KILL — NOT created", `run_until_stopped` RETIRED).
evidence: AST census -> `mantis.train.coordinator {'drain':13,'dispatch':2,'StepCoordinator':1,'config':1,'step':2}`; test_no_phantom_seam_member.py reads `getattr(pkg, "__all__", ())`, so an absent `__all__` stays green.
Δlines: -33 (45 lines, docstring ends line 10; keep one import line + blank).
witness: test_no_phantom_seam_member.py::test_the_retired_seam_members_stay_retired, test_heldout_gap.py.
depends: 08, 09 (both shrink this list anyway).

### S-A-CORE-1-15 | SIMPLIFY | A
subject: src/mantis/train/pretrain/__init__.py::pretrain
claim: a lazy forwarder to `cli.pretrain` that nothing calls; `python -m mantis.train.pretrain` enters through `__main__.py`, which imports `cli.pretrain` directly.
evidence: `git grep -n -E "from mantis\.train\.pretrain import|pretrain\.pretrain\(|mantis\.train\.pretrain\.pretrain\b"` -> only imports of `graph_route` and `cli` (tests).
callers: grep -> 0; -m -> `__main__.py` bypasses it; STATE -> 0; str -> 0.
Δlines: -11 (AST span 9 + `__all__` + blank).
witness: NONE.
depends: —.

### S-A-CORE-1-16 | DEAD | A
subject: src/mantis/train/ema.py::DEFAULT_UPDATE_EVERY
claim: an unreferenced constant (the value is the required schema key `train.ema.update_every`).
evidence: vulture -> "unused variable 'DEFAULT_UPDATE_EVERY' (60%)"; `git grep -n -w DEFAULT_UPDATE_EVERY` -> the def only.
callers: grep 1 (def); str 0; cfg -> `train.ema.update_every` is read by `resolve_ema_config`; STATE 0.
Δlines: -1.
witness: NONE.
depends: —.

### S-A-CORE-1-17 | SIMPLIFY | C
subject: src/mantis/train/ema.py::build_ema_model
claim: a one-line forwarder to `EmaModel(model, decay=decay)`; callers are trainer/core.py and one test.
evidence: `git grep -n -E "build_ema_model|EmaModel\("` -> ema.py, trainer/core.py:225, tests/train/test_ema_lever_is_reachable.py.
Δlines: -3 (AST span) net ≈ -4 with the blank line.
witness: test_ema_lever_is_reachable.py.
depends: —. Lane C only because the call site is trainer/core.py (PZ-1).

### S-A-CORE-1-18 | DEAD | B
subject: src/mantis/train/pretrain/cli.py — the local `config` dict in ::pretrain, ::training_terms, ::SHADOWED_TRAIN_KEYS
claim: `config = {"encoding", "in_channels", **training_terms(...), filters, res_blocks}` was the dense `BootstrapTrainer`'s input and is never read after it is built; the graph route reads `run_config.model_dump()`. `training_terms` exists only to feed it, and `SHADOWED_TRAIN_KEYS` is read only by a test.
evidence: `sed -n '/config: dict = {/,/res_blocks/p' cli.py | wc -l` -> 9; no later `config` name in the function body (read 150-229); `git grep -n -w SHADOWED_TRAIN_KEYS` -> def + tests/train/test_pretrain_cli_states_no_training_knob.py.
callers: grep -> training_terms: cli.py + that test (4 uses); -m -> `python -m mantis.train.pretrain` reaches pretrain() but not the dict's readers (there are none).
Δlines: -9 -17 (AST) -3 (constant + comment); test module test_pretrain_cli_states_no_training_knob.py loses the tests that pin `training_terms` (test-floor move).
witness: test_pretrain_cli_states_no_training_knob.py.
depends: 19.

### S-A-CORE-1-19 | SIMPLIFY | B
subject: src/mantis/train/pretrain/cli.py argparse flags `--filters --res-blocks --resume --lr-peak --eta-min --freeze-trunk-entry --unfreeze-blocks --inference-out --label-smoothing` + ::DEFAULT_LABEL_SMOOTHING; graph_route.py::DENSE_ARM_FLAGS, ::refuse_dense_arm_flags, run_graph_pretrain(dense_arm_flags)
claim: nine dense-arm flags are parsed only so they can be refused; deleting them keeps the outcome (argparse refuses unknown flags, exit 2) and removes the refusal machinery. `--resume` also feeds `_resolve_encoding_name`, whose resume branch then ends in the same refusal.
evidence: graph_route.py docstring: "Refused rather than ignored"; `git grep -l -E "dense_arm_flags|refuse_dense_arm_flags|DENSE_ARM_FLAGS" -- tests` -> tests/train/test_bc_graph_reroute.py.
Δlines: ≈ -13 -25 (AST) -11 dict at the call -1 param, -18 argparse lines, -1 constant ≈ -69; graph_route.py goes 315 -> under 300, so its R8 header must be DROPPED (gate 15 third rule).
witness: test_bc_graph_reroute.py (error type changes from GraphPretrainError to SystemExit(2)); test_bc_pretrain_cli_stopflags.py.
depends: 18. Lane B: the refusal message is a design choice.

### S-A-CORE-1-20 | DUP | B
subject: src/mantis/train/pretrain/graph_route.py::_assert_launch_pin (inline chunked sha256)
claim: a 4-line chunked-hash loop identical in result to mantis.util.hashing::sha256_file, which the same package's sibling train/heldout.py already uses; not among CARD-MECHANISM-SWEEP's three named sites.
evidence: `sed -n 94,100p graph_route.py` -> `digest = hashlib.sha256()` / `for chunk in iter(lambda: fh.read(1 << 20), b"")`.
deliberate?: no second-authority grounds stated; the only difference is chunk size (1 MiB vs util's `_HASH_CHUNK`), which does not change the digest.
Δlines: -4 (and the `hashlib` import).
witness: tests/train/test_bc_graph_reroute.py (the three monkeypatched `_CORPUS_SHA_PINS` cases).
depends: 21.

### S-A-CORE-1-21 | DUP | C
subject: src/mantis/encoding/audit_sections.py::_sha256_of_file
claim: a private chunked-sha256 helper identical in result to mantis.util.hashing::sha256_file (3 call sites in _section_corpora).
evidence: `grep -n _sha256_of_file audit_sections.py` -> def + 3 calls.
deliberate?: none stated; audit.py already depends on mantis.util.
Δlines: -9 (AST).
witness: tests/encoding/test_audit_cli.py.
depends: 20. Lane C (encoding/** PZ glob).

### S-A-CORE-1-22 | DUP | C
subject: src/mantis/train/resume_state.py::write_resume_state; train/anchor.py::_write_provenance_sidecar
claim: both hand-roll temp-write + `os.replace` beside bundle.py::atomic_write (the one atomic writer checkpoints.py already uses).
evidence: `git grep -n -E "os\.replace|\.replace\(" -- src/mantis/train` -> bundle.py:67, resume_state.py:268, anchor.py:141/142/156/232.
deliberate?: NOT behaviour-identical: atomic_write adds a unique temp name, temp cleanup on failure and a directory fsync; the sidecar writer has no fsync at all. A merge is a (small) durability change, so it needs a ruling-grade choice.
Δlines: ≈ -6 (resume_state) -3 (anchor).
witness: tests/train/test_resume_state.py, test_anchor.py.
depends: —. Lane C (PZ-1 resume bundle; anchor.py PZ-1/PZ-6).

### S-A-CORE-1-23 | DUP | C
subject: src/mantis/train/heldout.py::HeldoutSlice.read vs src/mantis/train/pretrain/heldout.py::HeldOutMonitor.evaluate
claim: the same loop, N × `run_declared_eval_step`, summing policy/value losses and dividing by N; they differ only in who picks N and in HeldoutSlice's reseed.
deliberate?: the two witnesses are separate rulings (R328(d) BC stop, R366(c)/R367 gap witness); a shared helper would add a function (net Δ small).
Δlines: ≈ -8 net (derived by eye from the two ~12-line loops; not measured by tool — treat as an estimate for the reviewer to derive).
witness: tests/train/test_heldout_gap.py, test_bc_heldout_stop.py.
depends: —. Lane C (heldout.py R367-ratified).

### S-A-CORE-1-24 | DEAD | C
subject: src/mantis/encoding/resolvers.py::resolve_arch, ::ArchSpec, ::expand_auto_paths, ::resolve_anchor_path, ::_ANCHOR_PATHS, ::_AUTO (+ 4 re-exports in encoding/__init__.py)
claim: `resolve_arch` is the named, queued dead export (R154 conditioned; R162 held it as an R20 dense surface); `expand_auto_paths` roots the transitively-dead ADJ-WP12R-19 cluster, and `_ANCHOR_PATHS` is now `{}` so `resolve_anchor_path` can only raise. The grid path is gone (R346(f)), which is the R20 grounds' subject.
evidence: tests/encoding/test_no_dead_resolver_export.py::_QUEUED_DEAD_EXPORTS = {"resolve_arch","expand_auto_paths"}; ::test_the_queued_dead_exports_are_still_dead green by construction; `git grep -n -w ArchSpec` -> encoding/__init__.py + resolvers.py only.
callers: grep -> resolve_arch/expand_auto_paths: re-exports + that test + RULINGS/archive; str -> `"corpus_npz"`/`"bootstrap_anchor"` keys only inside expand_auto_paths; cfg -> neither `corpus_npz` nor `bootstrap_anchor` is a leaf.
Δlines: -58 (AST 9+9+13+23+1+1, plus 2 for ShapeMismatchError in 25) and the 4 re-export pairs (-8); the test's queue rows and two anti-rot tests go (test-floor move). resolvers.py stays > 300 (458 -> ≈ 390): its R8 header's "corpus/anchor/held-out registries" must be reworded.
witness: test_no_dead_resolver_export.py.
depends: 25, 27. Lane C (ruling-named: R154/R162, ADJ-WP12R-19).

### S-A-CORE-1-25 | DEAD | C
subject: src/mantis/encoding/resolvers.py::ShapeMismatchError
claim: an exception class nothing raises, catches or imports.
evidence: vulture -> "unused class 'ShapeMismatchError' (60%)"; `git grep -n -w ShapeMismatchError -- src tools tests crates` -> the def only (5 docs hits are docs/slim and archive).
callers: grep 1; str 0; ep 0; STATE 0.
Δlines: -2 (AST) -2 blank.
witness: NONE.
depends: —. Lane C (encoding/** PZ glob).

### S-A-CORE-1-26 | SIMPLIFY | C
subject: src/mantis/encoding/_probes.py (one constant, `GNN_GRAPH_MARKER_KEY`)
claim: a 10-line module whose single src consumer is resolvers.py; the "any future detector" rationale has no second tenant.
evidence: `git grep -n -E "_probes|GNN_GRAPH_MARKER_KEY" -- src tools tests` -> resolvers.py:19 and tests/encoding/test_r8_identity.py.
Δlines: -9 (10-line file, the constant moves into resolvers.py with the test import re-pointed).
witness: test_r8_identity.py.
depends: —. Lane C (PZ-2 detector).

### S-A-CORE-1-27 | DEAD | C
subject: src/mantis/encoding/__init__.py::handshake_ran, ::handshake_skipped, ::_skipped (and the append in _registry_sha_handshake)
claim: exposed "so a composition root can publish it into the event stream", but no composition root reads them; the only reader is tests/monitor/test_phantom_rule_is_visible.py.
evidence: `git grep -n -E "handshake_ran|handshake_skipped" -- src tools` -> encoding/__init__.py only.
callers: grep -> that test; str 0; STATE 0; event_manifest.md -> no field.
Δlines: -3 -1 -2 (AST) and the `_skipped.append` arm (-2).
witness: test_phantom_rule_is_visible.py::test_the_skip_is_readable_as_STATE_not_only_as_a_log_line.
depends: —. Lane C (PZ-2 `_registry_sha_handshake`); the alternative is to wire it into run_boot_identity, which is a feature, out of scope.

### S-A-CORE-1-28 | DEAD | C
subject: src/mantis/train/anchor.py::_BOOTSTRAP_ANCHOR_CANDIDATES, ::_OPTIONAL_HEAD_PREFIXES
claim: the fallback list names two CWD-relative dense `.pt` files (`bootstrap_model_v6.pt`, `_v7full.pt`) that a graph run skips on encoding mismatch, and production reaches it because loop.py passes no `bootstrap_candidates`; the optional-prefix tuple lists CNN aux heads (opp_reply_conv, value_var, ownership_head, threat_head, chain_head, ply_index_head, input_channel_index) that no GNN state dict has.
evidence: `git grep -n bootstrap_candidates -- src tools` -> anchor.py only (loop.py:73's call omits it); `git grep -n bootstrap_model_v -- src` -> anchor.py only.
callers: grep as above; tests/train/test_anchor.py passes `bootstrap_candidates=()` explicitly (10 sites).
Δlines: -4 and -10 (AST) plus a duplicated comment block above `CANONICAL_ANCHOR_FILENAME` (2 lines).
witness: test_anchor.py.
depends: —. Lane C (anchor.py PZ-1 launch pin file). Emptying the list changes behaviour only if a matching file exists in CWD.

### S-A-CORE-1-29 | DEAD | C
subject: src/mantis/model/arch.py::Representation; trainer/core.py `Trainer.ckpt_had_value_fc2_bins`; `TrainHParams.draw_reward`, `.ply_cap_value`, `.value_target`, `.policy_target`
claim: an unused type alias; an attribute set to False and never read; four hparams fields never read off the instance (validation reads the raw `train` dict, self-play reads its own copies).
evidence: vulture -> "unused variable 'Representation'", "unused attribute 'ckpt_had_value_fc2_bins'", "unused variable 'draw_reward'"; `git grep -n -E "(hp|hparams|self|trainer)\.(draw_reward|value_target|policy_target|ckpt_had_value_fc2_bins)\b" -- src tools` -> 0 (`hp.ply_cap_value` hits are selfplay/hparams.py's own dataclass).
callers: grep -> fixtures construct TrainHParams with those four fields (tests/train/_microbatch_harness.py, conftest `make_full_train_hparams`).
Δlines: -1 -1 -4 (fields); fixture kwargs drop.
witness: tests/train/test_trainer_*; config consumer registry (the keys stay live through selfplay/hparams.py).
depends: —. Lane C (model/** PZ-2; core.py PZ-1).

### S-A-CORE-1-30 | SIMPLIFY | C
subject: unused parameters on protected signatures: checkpoints.py::load_checkpoint(device); anchor.py::resolve_anchor(anchor_state, sink); trainer/core.py::Trainer.eval_step_from_graph_batch(batch_composition), ::Trainer.save_checkpoint(loss_info), ::Trainer.load_checkpoint(checkpoint_dir); orchestrator.py::build_resume_config_overrides(baked_config); step.py::StepCoordinator._emit_training_step(cfg)
claim: each parameter is accepted and never read in the body.
evidence: AST unused-param scan (scratchpad) -> exactly these nine names (plus signals.py::_stop(sig, frame), which the signal API requires and is excluded).
Δlines: ≈ -9 signature slots plus call-site kwargs (not derived per site).
witness: test_checkpoint_conformance.py, test_resume_goldens users (build_resume_config_overrides is golden-pinned), TrainerLike.save_checkpoint in SEAM_MATRIX.
depends: —. Lane C (LAW-12 one loader, PZ-1 resume, PZ-6 golden row).

### S-A-CORE-1-31 | DEAD | B
subject: src/mantis/train/lifecycle/watchdog.py::DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC (+ both package re-exports)
claim: a code-side default constant with no code reader (the timeout comes from `StepCoordinatorConfig.selfplay_stall_timeout_sec`); only tests/train/test_lifecycle_contract.py reads it, as a helper default.
evidence: `git grep -n -w DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC` -> def, 4 re-export lines, 2 test lines.
callers: grep as above; str 0; cfg -> the live value is the config's; STATE 0.
Δlines: -1 (+ -4 re-export lines if 11/12 do not already take them).
witness: test_lifecycle_contract.py (its `# noqa: F401 — pinned default` import).
depends: 11, 12.

### S-A-CORE-1-32 | DOC | C
subject: stale grid/dense-era wording in slice docstrings: resolvers.py module docstring + ::detect_encoding_from_state_dict ("ONE deterministic shape fallback … over the registered grid set"); registry.py::_load ("`compat.py`, `resolvers.py`, and `audit_sections.py` all import it" — `compat.py` does not exist); model/__init__.py ("nets (GNN + CNN)"); arch.py::arch_from_spec_and_config message ("expected 'grid' or 'graph'"); train/emit.py ("`mantis/monitor/` is EMPTY until WP13", "Injected everywhere until WP13"); encoding/audit_sections.py `_CORPUS_FILENAME_HEURISTIC` rows `v6w25`/`v6_live2_ls` and the `"v6"` default
claim: each names a structure the tree no longer has.
evidence: `git grep -n -i -E "\bgrid\b|\bdense\b|\bCNN\b|\bv6\b" -- src/mantis/{train,model,encoding}` (read hit by hit); `git ls-files | grep compat.py` -> 0.
Δlines: ≈ 0 (rewording); the heuristic rows -2.
witness: NONE (tests/encoding/test_audit_cli.py for the heuristic rows).
depends: 24. Lane C (PZ globs).

## DEFECTS
- src/mantis/train/pretrain/cli.py: `--corpus-npz` is parsed and never read and is NOT in DENSE_ARM_FLAGS, so it is silently ignored (the exact class `refuse_dense_arm_flags` exists to refuse); `--no-compile` is likewise ignored, by stated choice.
- src/mantis/train/anchor.py::_write_provenance_sidecar `tmp.write_text(...)` and src/mantis/encoding/audit_sections.py::_section_corpora `sidecar.read_text()` pass no `encoding=` (CLAUDE.md code style: the rule is the whole tree; gate 16 only enforces tools/ and tests/).
- `monitor_gates.buffer_save_errors_total` is published every gate tick from a counter no production path can increment (LAW-07 class; see 07).
- `training_step` carries ten always-None fields, and docs/contracts/event_manifest.md claims producers (a "dense" tail, `aux_chain_weight`, an `aux_chain_loss` event) that do not exist (see 02).
- `iteration_complete.corpus_selfplay_frac` is a constant 1.0 published as a measurement (see 05).
- `TrainerLike` is `@runtime_checkable` and declares `train_step_from_tensors`, which the production `Trainer` does not define, so `isinstance(trainer, TrainerLike)` is False for the real trainer (no call site does that check today).
- coordinator/config.py::promotion_capable_rounds docstring says "Surfaced at launch"; nothing calls it, and two CONSUMER_REGISTRY strings cite it as part of `train.eval_interval`'s consumer path.

## PARKED
none.

## HANDOFF
- Test slices: 13 test files define a `train_step_from_tensors` stub for the phantom TrainerLike member (08): tests/_drivable.py, tests/test_run_strict_composition.py, tests/train/test_{abort_exit_signal,clean_stop_save,cluster_stat_wiring,coordinator_gates,eval_result_routing,gate_interval_decoupling,inference_seam_events,iteration_complete_decoupling,target_counter_events,train_step_dispatch}.py (`git grep -c train_step_from_tensors -- tests | wc -l` -> 13); tests/test_run_strict_composition.py also stubs `sample_batch_with_pos`.
- C2 (src/mantis/run.py): the comment above the StepCoordinator construction cites `train/batch_assembly.py`, which does not exist (`git ls-files | grep -c batch_assembly` -> 0; tests/encoding/test_resolvers_mapping_no_fallback.py's docstring cites it too); run.py is the caller side of 06 (`train_cfg={}, mixing_cfg={}, pretrained_buffer=None, bufs=None, SimpleNamespace(gpu_monitor=None, …)`).
- C2 (src/mantis/selfplay/pool.py `recent_buffer` attribute, pool_push.py's `recent_buffer.push`): the producer end of the dense recent-buffer chain in 06/07.

## Not covered
- checkpoints.py (1277), step.py (1099), heartbeat_watchdog.py (591), signals.py (390), trainer/core.py (673) were covered by the AST reference/unused-parameter census only, not a line-by-line read: dead BRANCHES inside live functions there are unexamined.
- `checkpoints.py::strip_and_restamp` (174 lines) has no src/tools caller and no CLI, but LAWS.md LAW-12 names it as the one sanctioned encoding-change path and STATE/CARDS record operator use; deliberately not proposed.
- model/value_targets.py (codecs landed unarmed, R322/R323) and the conformance-pinned model internals were not assessed for slimming (PZ-2 tenants).
- The encoding audit CLI (audit.py + audit_sections.py, 834 lines) is live (`python -m mantis.encoding audit`, test_audit_cli.py); its dense-era sections (§3 scans `.npz`, the corpus graph rings are `.hexg`) were only noted (32), not sized as a PACK.
- No probe or targeted test was run (torch not importable; the reviewer probes). Δlines for 06, 19, 23 and 30 are partly estimated and marked as such.
