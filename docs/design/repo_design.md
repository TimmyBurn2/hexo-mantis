# MANTIS — REPOSITORY DESIGN (v1, founding contract)

AlphaZero-style self-play bot for Hex Tac Toe (6-in-a-row, unbounded hex grid, 2-stone
compound turns). Rust engine (cargo workspace) + Python training/eval (uv, src-layout),
PyO3 bridge. GNN-first, representation-extensible. This document is the structural
contract of the repository; CLAUDE.md carries the operating laws; docs/registers/ carries
law text and the falsified register. Deviations from this document require an amendment
commit to this file — never a silent drift.

Design provenance: every structural rule below exists to make a named, previously
observed bug class unrepresentable. The bug-class evidence lives in the private
migration/records archive; the rules stand on their own here.

---

## 1. Layout

```
hexo-mantis/
├── Cargo.toml                  # workspace: resolver=2; release: lto=fat, cg-units=1,
│                               #   panic="unwind", strip="symbols"; NO target-cpu=native
├── pyproject.toml              # uv project root: mantis package (src-layout);
│                               #   [tool.uv.workspace] members = ["crates/mantis-bridge"]
├── uv.lock
├── Makefile                    # thin: build / test / test.integration / bench /
│                               #   bench.baseline / check.wasm / vendor / clean
├── crates/
│   ├── mantis-core/            # board, hex geometry, rules, Ply/Turn vocabulary types
│   ├── mantis-graph/           # dep-free axis-graph builder; native + wasm32 targets
│   ├── mantis-encoding/        # registry.toml + spec + validators + dense encode kernels
│   ├── mantis-search/          # MCTS (PUCT + Gumbel), completed-Q, tactics solver
│   ├── mantis-selfplay/        # runner, worker loop, inference queues, replay buffers
│   └── mantis-bridge/          # ALL PyO3; maturin; builds mantis._engine (abi3 release)
├── src/mantis/                 # the one Python package (installed, src-layout)
│   ├── encoding/               # delegating shim over _engine registry + resolvers + audit
│   ├── config/                 # schema (pydantic, extra=forbid) + per-knob resolvers
│   ├── env/                    # thin GameState wrapper over _engine
│   ├── data/                   # corpus generation, IO, metrics, augmentation LUTs
│   ├── model/                  # nets (GNN + CNN), dist65 value codec, build_net authority
│   ├── train/                  # trainer, step coordinator, lifecycle (signals/watchdog/
│   │                           #   disk guard/shutdown-save), pretrain, checkpoint IO (ONE)
│   ├── selfplay/               # pool over Rust runner, inference server, graph collate
│   ├── eval/                   # pipeline, promotion gate, round-robin, BT
│   ├── arena/                  # EVALFAIR instrument (deploy-matched, paired books)
│   ├── bots/                   # BotProtocol + all bot wrappers incl. community API client
│   ├── monitor/                # HEADLESS ONLY: event emit, producer manifest, alert rules
│   ├── diagnostics/            # single-definition production-importable readouts
│   ├── util/                   # small helpers (device, coordinates, cpu budget)
│   └── deploy/                 # RESERVED: ONNX export + torch-parity check (post-cutover)
├── tests/                      # SINGLE collection root; mirrors src/mantis + crates
│   └── fixtures/               # frozen oracle banks (books, goldens, probe npz, positions)
├── configs/                    # explicit COMPLETE schema-validated configs; no variants-
│                               #   on-base; minted via tools/mint_config.py
├── docs/
│   ├── design/                 # this file + subsystem designs
│   ├── contracts/              # versioned seam contracts (see §4)
│   └── registers/              # laws, falsified register (curated)
├── tools/                      # dev-only: mint_config, hardcode_scan, bench floors data
├── vendor/                     # pins.toml (url+sha+patch) + `make vendor` fetcher; no
│                               #   submodules, no loose weights
└── .github/workflows/ci.yml
```

Display surfaces (web dashboard, game viewer, TUI monitors) are deliberately absent.
The event-manifest + schema'd JSONL channels (§4.7) are the stable contract any future
display builds against. `src/mantis/deploy/` is reserved, empty until post-cutover.

## 2. Dependency DAG (one-way; enforced by crate/package deps, checked in CI)

Rust:
```
mantis-core      → (nothing in-workspace)
mantis-graph     → (nothing; dep-free, wasm32-clean)
mantis-encoding  → core, graph          # pins graph consts into registry validation
mantis-search    → core, encoding      # LegalSetPolicy + gumbel live HERE, not in selfplay
mantis-selfplay  → core, encoding, graph, search
mantis-bridge    → all of the above    # the ONLY crate that knows pyo3 exists
```
Rules: no `#[pyclass]`/`#[pymethods]`/`#[pyfunction]` outside mantis-bridge (core crates
compile without pyo3). No core API takes or returns a Python wrapper type. `Board` takes
plain geometry values; registry-spec → values resolution happens in encoding/selfplay,
never inside core.

Python (import direction, top-level imports only; lazy imports need a stated reason):
```
util, diagnostics            → (leaves)
encoding                     → _engine only
env                          → encoding
config                       → encoding, util
data                         → encoding, env, util
model                        → encoding, config          # dist65 codec lives HERE
bots                         → encoding, env, model
selfplay                     → encoding, env, model, config, util, monitor(events only)
monitor                      → encoding, util            # headless; imports NO torch
train                        → all above except eval     # eval reached via injected callable
eval, arena                  → all above except train's internals; checkpoint IO via the
                               ONE loader exposed from train.checkpoints
run                          → train, eval, monitor, config, selfplay   # the composition
                               root AND the launcher (WP11-A §a.4/§c.6; WPMAIN); a
                               top-level module ABOVE both train and eval; NOTHING imports
                               `mantis.run` — it is a source-only DAG node. WPMAIN added
                               NO new edge class: `build_run_collaborators` calls
                               `mantis.train.orchestrator.init_trainer` and
                               `mantis.selfplay.pool.WorkerPool`, both already inside this
                               row, and the ONE lazy import (`mantis._engine`, in
                               `_select_buffer`) keeps its stated DAG reason. ADDITIVE: this row registers the new
                               node; it does not weaken the train↛eval ban above, which
                               stays verbatim (census-tested,
                               tests/test_run_composition.py::
                               test_no_train_module_imports_eval_even_lazily).
```
The historical training↔eval / model↔training / bots↔bootstrap cycles are dissolved by
relocation (protocol→bots, dist65→model, eval hooks injected). CI runs an import-cycle
check (tools/check_import_dag.py) — a new top-level cycle fails the build.

## 3. Representation extensibility (the registry-kind axis)

- `crates/mantis-encoding/registry.toml` is the single source of truth for encoding and
  shape. Every entry carries `representation` (required, no default): `"grid" | "graph"`
  today; the set is extensible. Unknown TOML keys are a parse ERROR; missing required
  keys are a parse error; the validator collects ALL errors before reporting.
- Rust: representation is a closed enum on `RegistrySpec`; dense-only or graph-only code
  paths are unreachable for the other kind (match exhaustiveness, no `_ =>` on the kind).
- Python: `mantis.encoding.lookup(name)` returns the `_engine` spec (delegation, single
  parser). Model/buffer/batcher construction dispatches through ONE authority per layer
  (`model.build.build_net(arch)`, buffer facade, batcher ctor). Reading arch attributes off
  live `nn.Module` instances is banned — arch metadata travels on declared dataclasses
  (`model.arch.CnnArch` / `model.arch.GnnArch`), which `build_net` consumes; a live-module
  representation sniff (the former `model_representation`) is DELETED and grep-gate-banned
  (a census test proves it stays absent). The discriminator, stated so the prose stops being
  broader than the gate: what is banned is *deriving* arch metadata from a live module's
  structure (a representation sniff, or reading arch hyperparameters off an `nn.Module`);
  *carrying* the declared dataclass instance itself as a handle is the convention, not a
  breach of it — `build_net` attaches `net.arch = arch` and `eval/snapshot.py` reads it back,
  which is the arch travelling with the model exactly as this section prescribes.
- No dense-by-default anywhere: an absent representation is an error, never `"grid"`.
- The compiled module exposes `registry_sha()` (sha256 of the embedded TOML).
  `mantis.encoding` hashes the on-disk TOML at import in dev/test and hard-errors on
  mismatch — a stale extension cannot silently serve a stale registry.

## 4. Seam contracts (each has docs/contracts/<name>.md with: version, owner module,
   who-asserts-what-where table, pinning-test list)

| # | contract | ver | one-line summary |
|---|---|---|---|
| 1 | registry | v1 | TOML schema + validator invariants + audit CLI exit codes (0/1/2) |
| 2 | dense wire | v1 | fixed `[n, feature_len]` f32 batches; strides spec-derived; shape-checked both sides |
| 3 | graph wire (ragged) | v1 | block-diagonal GraphWire; 18 assertions, 15+ named errors; single-read `take()`; −1 off-window sentinel travels; NO fixed-width fallback |
| 4 | checkpoint envelope | v2 | see §6 |
| 5 | run config schema | v4 | pydantic models, extra=forbid; schema_version key in every file |
| 6 | replay persist | HEXB v9 / HEXG v1 | magics, versioned headers, wire-signature cross-load law, loud cross-format rejection |
| 7 | event manifest | v1 | every panel AND every headless gate input cites a live producer; mutation self-test proves the checker bites |
| 8 | community bot API | bot-api v1 | vendored OpenAPI 3.1 spec + BKE-notation round-trip suite |
| 9 | eval instrument | v1 | deploy-matched argmax head, frozen sha-pinned paired opening books, per-pair bootstrap CI, eff_n = trajectory-hash-distinct games |
| 10 | mint preflight report | preflight-mint-v1 | the mint preflight's evidence JSON: always written (LAW-14); mode, verdict and mint TIER derived from what the run DID, never from what it intended |

Contract changes bump the version and update the contract doc + its tests in the same
commit. The PyO3 seam stays thin flat arrays (marshaling is a measured cost); per-field
copies are single-read by contract.

### AMENDMENT — contract #5 v1 → v2, with the doc half of the same-commit clause DEFERRED

**WPAX Phase S (card CARD-SMOKE-SEAM), ADJ-09 Option B.** Recorded here rather than left as
silent drift (R9).

1. **The bump.** `train.max_train_steps` is a NEW **required** field on `RunConfig` — the
   run-length authority, resolved to `StepCoordinatorConfig.stop_step`, which is the real
   stop condition. This is an *incompatible* change: a config file lacking the key fails to
   load. That is strictly more than the precedent the contract doc's own status line records
   for WP8 (*"additive founding growth … no key changed incompatibly; the config's own
   `schema_version` stays 1"*), so contract #5's row above moves **v1 → v2**. The config
   files' own `schema_version:` key is a *file-format* pin and is unchanged at `1`; it is
   not this contract's version.
2. **The doc half is OWED and BLOCKED, and the owner is WP14.**
   `docs/contracts/run_config_schema.md` must gain the key and the bump. Phase S is barred
   from touching that file (rule R11: it carries another work package's uncommitted work —
   an unrecoverable loss if disturbed), so the same-commit clause above is **half-kept**: the
   version bumped here, the contract doc deferred to **WP14**. Stated outright rather than
   left to be discovered.
3. **run5-mint checklist item.** Before the next `configs/run5.yaml` mint, check that
   `docs/contracts/run_config_schema.md` carries `train.max_train_steps`. A stale contract doc
   *for the very config being minted* is what misleads at mint time, so the owed update is
   recorded where the minter will read it.
4. **Nothing enforces contract #5's doc.** MEASURED at this commit: `grep -rn
   "run_config_schema" .` over the whole repo returns **no hits at all outside this
   amendment** — no test, no tool, no Makefile target, no CI gate names the contract file
   (not even the file itself). Two consequences: the drift
   recorded above **cannot make any gate lie** (nothing reads the doc, so nothing can report
   green over it), and this clause was **already weaker than it reads** — kept by manual
   discipline alone. That missing handshake is its own defect, recorded for WP14 / WP-R
   rather than fixed here.

**Precedence, recorded so it is not re-derived.** R11 protects another WP's uncommitted work
(unrecoverable); this clause's doc half protects a document with no consumer (recoverable
staleness). The reading applied here is that **R11 yields nothing and the doc clause defers**.
That is a dispatcher reading pending operator ratification, not a settled rule.

### AMENDMENT — contract #5 v2 → v3, same shape, same deferred doc half

**WPAX Phase D (card CARD-DRAWRATE-KEY), R65 as re-scoped by R80.** Recorded here rather than
left as silent drift (R9); the S-4 amendment above is the precedent this follows verbatim.

1. **The bump.** `train.draw_rate_abort` is a NEW **required** field on `RunConfig` — a nested
   block (`threshold` `gt=0, le=1`, `min_step` `ge=1`, `N_pool_min` `ge=1`) or `null`, which
   is the EXPLICIT disarmed posture. Incompatible for the same
   reason S-4's was: a config lacking the key fails to load. Contract #5's row moves
   **v2 → v3**; the config files' own `schema_version:` file-format pin is unchanged at `1`.
   A second cross-field rule joins `RunConfig`'s validators: `train.draw_rate_abort.min_step`
   must be `< train.max_train_steps`, the twin of the actor-lag rule and the same defect class
   ("armed in the config, absent in effect").

   **WPMINT Phase DS amendment (operator ruling R92), same contract row.** The block's third
   key was `min_samples` (`ge=1, le=DRAW_RATE_WINDOW`) and is now `N_pool_min` (`ge=1`),
   because the gated STATISTIC changed: it is the pooled count-weighted rate
   `Σ draws / Σ completed` over the union of worker windows, and insufficient evidence
   (`Σ completed < N_pool_min`) is a NO OBSERVATION rather than a healthy `0.0`. Two
   validators replace the retired `le=` bound, one at each end of the same defect class:
   `RunConfig._draw_rate_evidence_bar_within_configured_capacity` (`N_pool_min <= DRAW_RATE_WINDOW *
   selfplay.n_workers` — a cross-SECTION rule, which is why it cannot be a field bound) and
   `DrawRateAbortConfig._one_drawn_game_cannot_fire_the_abort` (`1/N_pool_min < threshold`).
   The block stays required and `null` stays the explicit disarmed posture, so the contract
   row does not move again.
2. **The doc half is OWED and BLOCKED, and the owner is still WP14.** R11 bars this phase from
   `docs/contracts/run_config_schema.md` exactly as it barred Phase S, so the same-commit
   clause is half-kept again: version bumped here, contract doc deferred. The run5-mint
   checklist item above now covers **two** owed keys — `train.max_train_steps` and
   `train.draw_rate_abort`.
3. **Measured consequence, recorded because it moves an operator-facing number.** With run5
   armed at `min_step: 25000`, the mint preflight's minimum legal `--burst-steps` for
   `configs/run5.yaml` moves from **101 to 25001**: the burst override shortens
   `train.max_train_steps`, and every fire-floor in the config must stay inside the run. The
   tool now enumerates each binding rule with its own floor rather than reporting only the
   maximum. See IMPL_NOTES_D's STOP-2.
4. **The 101→25001 tension is resolved by DISCLOSURE, not by a shorter burst** (WPMINT Phase B
   / CARD-D-BURST-FLOOR). The floor cannot be shrunk — `min_step` is a run5 armed value and is
   mint-prereg-only (R82/R85) — and a shorter burst that pretended to cover the draw-rate axis
   is barred by R64. The preflight's evidence report therefore carries a `tier` block naming
   which mint tier the burst it ACCEPTED belongs to (`none` / `sync_lag` / `full`, derived from
   `_burst_floors`) and, in words, what that tier does NOT prove. **Both `sync_lag` and `full`
   are required for a mint, and `full` covers `sync_lag`** — one green `full` run discharges
   both. This deviates from the two-SEPARATE-RUNS shape the card presumed, on a measured
   ground: a `PRODUCTION_CONFIGS` row must arm `draw_rate_collapse` (gate 12 assertion (c)), an
   armed row puts `min_step + 1` in the floor set, and the override refuses anything below the
   max at rc 11 — so on a production config tier `sync_lag` is UNREACHABLE, and the only route
   to it is disarming the row the mint exists to arm. Measured, HEAD: run5's floor is 25001 and
   every other `configs/` entry's is 101, because only run5 arms the abort. Also measured: no
   burst of any length has ever run here — the boot child dies at TD-4 before `compose_run` —
   so `covered` is `[]` and both tiers stay OWED on every report the tool can currently write.
   The cost of the `full` tier is a published LOWER BOUND (`>= 1041.5 s` from WP10's 41.66
   ms/train-step floor) whose missing term — game-bound self-play generation for `>= 25001`
   completed games on one worker — is named rather than estimated.

### AMENDMENT — contract #5 v3 → v4: the step-coordinator knobs are CONFIG

**WPMINT Phase K-B (card `CARD-COORD-KNOBS`), R78 as clarified by R80, method bound by R93.**
Recorded here rather than left as silent drift (R9); the S-4 and Phase D amendments above are
the precedent this follows verbatim.

1. **The bump.** Twenty NEW **required** leaves on `RunConfig`, so contract #5's row moves
   **v3 → v4**; the config files' own `schema_version:` file-format pin is unchanged at `1`.
   Nineteen are flat `train.*` keys — `eval_interval`, `log_interval`, `buffer_save_interval`,
   `min_buf_size`, `replay_capacity`, `replay_capacity_schedule`, `training_steps_per_game`,
   `max_train_burst`, `batch_size`, `augment`, `recency_weight`, `mixing_initial_w`,
   `mixing_min_w`, `mixing_decay_steps`, `hard_gn_threshold`, `hard_gn_min_steps`,
   `terminal_eval_enabled`, `bot_batch_share`, `selfplay_stall_timeout_sec` — read by ONE
   resolver, `mantis.config.resolve.coordinator.resolve_coordinator_knobs`. The twentieth is
   `train.draw_rate_abort.consec`, which joins the abort family's block because R80 says its
   terms travel together. Incompatible for the same reason S-4's and Phase D's were: a config
   lacking any of them fails to load. `mantis.run._step_coordinator_config` now holds **zero
   literals** — every `StepCoordinatorConfig` field arrives from a resolver.

   **They are FLAT `train.*` keys, not a `train.coordinator` block, and that is a decision.**
   `train.step_coordinator.*` was RULED AGAINST at Phase D (§2, pinned verbatim by
   `tests/config/test_drawrate_arming_authority.py`) on the ground that a config block named
   after a dataclass is named after the wrong thing; these are training hyperparameters, and
   the coordinator is the object that reads them, not the fact they express. An OPTIONAL block
   was independently barred: `_leaf_paths` counts a `Block | None` as one leaf, so ~20 keys
   would have bought a cheap registry count by hiding from the LAW-08 bijection that justifies
   them.

   **Three names differ from their runtime fields, each for a measured reason.**
   `buffer_save_interval` -> `checkpoint_interval` (the coordinator's is the REPLAY-BUFFER
   save cadence; `train.checkpoint_interval` is the already-authored TRAINER cadence, and two
   config keys with one spelling is the duplicated-authority class R1 exists to kill);
   `replay_capacity` -> `capacity` and `replay_capacity_schedule` -> `buffer_schedule` (a bare
   `train.capacity` names nothing on its own).

2. **SIX coordinator fields are DELETED, not authored** (adjudication call K-a):
   `composition_interval`, `value_probe_interval`, `soft_ew_threshold`, `soft_ew_min_pts`,
   `instrumentation_enabled`, `bot_corpus_path` had no reader anywhere in `src/`, re-verified
   at HEAD by grep AND by recording every attribute read on a live `StepCoordinatorConfig`
   across the whole test tier. R1 requires a live consumer per key, so typing them in would
   have CREATED the violation the card meant to close.

3. **One value moves, and it is a correction, not a change.** `train.batch_size` is minted at
   **256**, not the dead field's `8`: WPMINT Phase K-A measured that
   `coordinator/step.py::_run_training_step` read
   `train_cfg.get("batch_size", full_config.get("batch_size", 256))`, that both lookups miss on
   the production path, and that the run's real batch size was therefore the literal `256`.
   Every other authored value is the value the code already used, proven by rebuilding
   `StepCoordinatorConfig` from all five minted configs at the parent commit and comparing
   field for field.

4. **`grad_norm_hard_abort` joins the armed-abort manifest as a DEFERRED row**
   (adjudication call K-c), with a new `Mechanism.CONFIG_THRESHOLD_BELOW_CEILING` and a
   `ceiling_path` naming `monitor.alert_grad_norm_max`. DEFERRED because a REQUIRED row would
   gate run5's mint on a grad-norm threshold nobody pre-registered — the class R84 refused.
   The mechanism is upper-bounded because `CONFIG_THRESHOLD_GT_ZERO` reads the shipped `1e9`
   as ARMED, which is "armed in the config, absent in effect".

5. **R78's first design question is ANSWERED YES and implemented.** The mint preflight's
   evidence JSON now carries a `coordinator` block — the resolved knobs, drain caps,
   `stop_step` and draw-rate terms, DERIVED from the shipped resolvers and never restated —
   in both AUDIT and PREFLIGHT mode. It witnesses the config -> resolver -> builder seam and
   explicitly NOT a child that was handed something else.

6. **The doc half is OWED and BLOCKED, and the owner is still WP14 / Phase W.** R11 bars this
   phase from `docs/contracts/run_config_schema.md` exactly as it barred Phase S and Phase D,
   so the same-commit clause is half-kept a third time: version bumped here, contract doc
   deferred. The run5-mint checklist item now covers **three** owed entries —
   `train.max_train_steps`, `train.draw_rate_abort` (including `consec`) and the nineteen
   `train.*` coordinator knobs.

### AMENDMENT — the deferred doc half is DISCHARGED; contract #10 is added

**WPMINT Phase W (WP14), operator rulings R66 and R91.** The three amendments above each
half-kept the same-commit clause and deferred the doc to WP14. This is that commit.

1. **`docs/contracts/run_config_schema.md` is committed and current at v4.** It carries the
   version history v1 -> v4, the run-length key, the draw-rate block including `consec` and the
   `min_samples` -> `N_pool_min` swap, the nineteen coordinator knobs with their three
   schema-vs-runtime renames, the `ReplayCapacityStage` sub-block, and every cross-field
   validator by name. The run5-mint checklist item raised in the v2 amendment is **discharged**:
   all three owed entries are in the doc.
2. **The curation was TRUTH-CHECKED, not merely extended.** Every claim inherited from the WP8
   working copy was grepped against the tree and a claim that could not be verified was removed
   or corrected rather than softened. Five were FALSE at HEAD: the status line's "checkpoints
   land in WP10" (WP10 landed at `b29f0bc`), an assertion row calling the legal-move radius
   schedule a REQUIRED field (the field and its resolver module are RETIRED), the resolver list
   naming `radius`, an emit payload described as 9 knobs (it is 8), and a regime-parity row
   whose radius arm now asserts the field's ABSENCE. The corrections are in the doc; the
   evidence is in `wp/WPMINT/IMPL_NOTES_W.md`.
3. **R11 is discharged and it never had repo-side enforcement.** Re-measured at this commit:
   `git grep R11` over `tools/ src/ tests/ Makefile .github/ docs/` returns only this file's
   own prose (plus one unrelated `§f-R11` fixture id). The rule lived entirely in the migration
   workspace, so "every gate asserting the exception drops it" was a no-op in the tree — the
   deferral was kept by discipline alone, exactly as the v2 amendment's item 4 predicted. That
   item's measured claim (*"`grep -rn "run_config_schema" .` returns no hits outside this
   amendment"*) is now **superseded**: item 4 below is its remedy.
4. **CI gate 13 closes the missing handshake** (R91's design question, answered by BUILDING).
   `tools/ci_gates/contract_doc_gate.py` refuses a contract doc that cites a config key or a
   `mantis.*` symbol the shipped schema does not have, or that states a leaf-key-path count
   other than the live one. It follows the gate-12 pattern strictly: every check is answered by
   importing `RunConfig` and the module tree, so a schema change ALONE reds it — pinned by a
   mutation that adds a leaf to `RunConfig` and leaves the doc untouched. The "deliberately
   absent" section is checked in REVERSE rather than exempted, so a retired key returning is
   also a red. Its one transcription — the leaf walker — is self-defending: the two
   consumer-registry copies assert `170` against the same schema, so a walker that diverged
   here would disagree with the doc and red this gate.
5. **`docs/contracts/checkpoint_envelope.md` is filled** (R11's second named item): status
   SKELETON -> LIVE, with the v2 filename/payload/metadata shape, the eleven assertions and
   their pinning tests. The subsystem landed at WP10; only the contract half was outstanding.
6. **Contract #10 is ADDED — the mint preflight evidence report** (Phase B's F-B3). The
   artifact has carried the version string `preflight-mint-v1` since WPAX and Phase B added a
   top-level `tier` field to it, while nothing under `docs/contracts/` described it. A
   versioned artifact with no contract is a version string that means nothing, so it is
   written rather than deferred a fourth time. It documents the shape, the always-written rule,
   the derived-from-what-the-run-DID discipline and the measured F-B1 gap.
7. **What this amendment does NOT close.** Gate 13 is a CITATION check: a claim the doc omits
   entirely is invisible to it, and it does not read prose for truth. Stated here rather than
   left for a reader to discover, because a gate whose real reach is narrower than its name is
   the class §9 keeps closing.


## 5. Config system

- Every config file is explicit and complete. There is NO inheritance, NO base-merge, NO
  code-side default for any config value. Loader = `yaml.safe_load` → schema validate.
  Missing key = hard error. Unknown key = hard error (kills the silently-disabled-
  opponent class). `extra="forbid"` on every model.
- A default lives in exactly one place: the schema field. Duplicated default authorities
  (code + yaml) are structurally impossible because code has no defaults.
- Copy-drift antidote (the cost of explicit-complete): `tools/mint_config.py` generates a
  complete config from a named template + an explicit delta mapping, stamping the delta
  into the file header; a one-key-diff assert tool verifies two configs differ exactly
  where claimed. Minted output is committed; the template is not a runtime input.
- One resolver per regime knob (sims, temperature, radius, corpus, amp dtype, …); eval
  reads the same resolver seam self-play does. Unknown knob consumer = ValueError.
- Identity keys have no terminal defaults: absent encoding/representation is an error.
- Rust crossing: `SelfPlayRunnerConfig` is a versioned builder struct; field set pinned
  by a byte-equivalence test; every field maps to exactly one runner slot.
- Every config key must have a live consumer (test-enforced); dead knobs are deleted
  with their freeze-tests in one commit.

## 6. Checkpoint envelope v2 (ONE format, ONE loader)

- Filename: `{run_id}_{step:08d}_{sha8}.ckpt` — run-id + content-hash in the name; a
  provenance check at load re-verifies both. Cross-lineage same-step collisions are
  structurally impossible.
- Payload: `schema_version=2`, `kind: "full"|"weights"`, `model_state`, (full only:)
  `optimizer_state`, `scaler_state`, `scheduler_state`, `config` (complete snapshot,
  schema-validated on write AND read), `metadata` = { `encoding_name` (required),
  `run_id`, `step`, `commit_sha`, `created_utc`, `arch`, `corpus_sha256?` }.
- Stamps are written once at creation and are IMMUTABLE — no re-stamping from a loaded
  config, ever. An artifact that cannot be stamped cannot be written (save fails loud;
  quarantine path if the run must survive).
- Exactly one loader, shared by train / eval / bots. `torch.load(weights_only=True)` on
  every surface. Declared-encoding is an assertion (mismatch raises); decode-override is
  a deliberate, loudly-logged cross-decode; both together is an error.
- Resume precedence: launch config wins EXCEPT the checkpoint-owned key set (encoding
  pins, arch, optimizer/scheduler state) — the set is a single frozen constant with its
  own test. The sanctioned encoding-change path is a weights-only strip gated on
  wire-signature equality.
- Config form is ONE shape (encoding as string); no string↔dict normalization funnel.

## 7. Build & packaging

- uv + src-layout; `mantis` installed from day one; zero `sys.path` writes anywhere; no
  directory named `tests` below the root `tests/`. Entry points are `python -m mantis.*`
  or console scripts — no loose script files as launch surface.
- maturin builds `crates/mantis-bridge` as `mantis._engine` (underscore-private, re-
  exported by the package); `uv sync` builds it; abi3 (py311) for release wheels.
- Workspace release profile: `panic = "unwind"` — Rust panics cross the FFI boundary as
  `PanicException`, never a process abort. lto=fat, codegen-units=1, strip=symbols; a
  `profiling` profile keeps debug symbols.
- No `target-cpu=native` in committed config; `make build.native` sets it via env for
  local perf work. Built artifacts are portable by default.
- `mantis-graph` is dep-free and wasm32-clean; `make check.wasm` (cargo check
  --target wasm32-unknown-unknown) must stay green from day one.
- Vendoring: `vendor/pins.toml` (url + commit sha + optional tracked patch) + one
  `make vendor` fetcher into gitignored `vendor/external/`. Single mechanism; features
  depending on unfetched vendors skip loudly with the fetch command named.

## 8. Testing doctrine

- Single collection root `tests/`, mirroring `src/mantis` and `crates/`. Behavior-named
  tests; contracts over implementation; private-attr assertions only as documented
  wiring pins paired to an invariant file.
- Tiers: default (fast unit), `integration` (runs in CI, includes at least one launch-
  path smoke), `slow` (on-demand). A meta-test asserts every `integration`-marked file
  is reachable from a CI/make target.
- Fixtures: a fixtures-manifest test FAILS (not skips) when the canonical dev fixture
  set is absent — suite shrinkage is loud. Large optional artifacts may skip but are
  counted and reported.
- Fixture regimes derive from production configs (`production_config()` fixture +
  explicit per-test deltas); one regime-parity test per LAW knob (graph amp dtype, sims,
  encoding, radius schedule) asserts suite default == production default.
- Census pins (grep-gate tests) guard classes the type system can't reach; each names
  its bug class and its triage protocol.
- Rust: inline unit tests near code; invariant pins and cross-language goldens under
  `crates/*/tests/`; proptest for board invariants; goldens are f32-bit-exact where the
  contract is numeric identity.

## 9. CI gates (PR → dev, and dev → main; main is merge-gated only)

1. Fresh-clone `uv sync` (builds the extension) — clone-and-run is the product.
2. `cargo test --workspace` and `cargo clippy` (pedantic=warn baseline).
3. `pytest` default tier, then `integration` tier. Collected-test count must be
   non-decreasing vs main.
4. `make check.wasm` green.
5. Bench smoke (short run; regressions surface at merge, not at launch).
6. Artifact rejection: any diff touching `reports/ checkpoints/ logs/ benchmarks/`, or
   adding files >1 MB or `*.jsonl` outside `tests/fixtures/`, fails.
7. Every file in `configs/` schema-validates.
8. Registry sha handshake + `python -m mantis.encoding audit` exit 0.
9. Import-DAG check (no new top-level cycles).
10. No Makefile/doc reference to a path absent from `git ls-files`.
11. No silent encoding-fallback arms — an absent encoding RAISES, it is never defaulted
    into existence (LAW-11, LAW-05). Grep gate over `src/` and `crates/`
    (`tools/ci_gates/silent_encoding_gate.py`). A site that is not a fallback is justified
    in place; a real arm that cannot be closed yet is registered with a named owner, never
    hidden in the justification hatch.
12. Armed-abort manifest audit — every `required` row of the manifest
    (`src/mantis/config/armed_aborts.py`) must be ARMED in every config the manifest binds
    (`PRODUCTION_CONFIGS`). Read through the real loader; no boot, no burst, no GPU
    (`tools/ci_gates/preflight_mint.py --audit-only`). A `deferred` row is printed loudly on
    every run, is tamper-evident through a pinned source literal, and does not gate. The
    same tool's full mint preflight — a real `compose_run` boot plus a bounded burst,
    asserting sync cadence and lag transport — is MANUAL and is invoked by no CI step. Its
    evidence report states which MINT TIER the accepted burst was and what that tier does not
    prove (§4 item 4); a tier that could not be run stays OWED rather than reading as optional.
13. Contract-doc drift — `docs/contracts/run_config_schema.md` may not cite a config key or a
    `mantis.*` symbol the shipped schema does not have, and its stated leaf-key-path count must
    equal the live one (`tools/ci_gates/contract_doc_gate.py`). Every check is answered by
    importing `RunConfig` and the module tree, never by consulting a transcribed key list, so a
    schema change alone reds it. The "deliberately absent" section is checked in REVERSE — a
    retired key that comes back reds the gate — rather than exempted. Trigger demonstrated in
    both directions by `tests/tools/test_contract_doc_gate.py`.
14. Curated lint/type gate (`tools/ci_gates/lint_gate.sh`, WPCLEAN Phase LG / R98) — the
    ruff select in `pyproject.toml` and pyright's basic src+tools surface are held at ZERO.
    Adoption law (R98): a rule enters the select only with a named in-repo defect class it
    would have caught (the mapping lives in the gate's header — F601's ca237d2 incident, the
    3.12-under-3.11 floor class, PLE0303) and only after being burned to zero — no gate over
    a dirty baseline. Exclusions are enumerated with grounds in `pyproject.toml`, never
    ambient (E501 dispositioned NEVER; tests/ style-exempt under frozen-oracle edit-aversion
    with the adopted defect classes still live there; pyright strict CARDED adopt-later).
    The trigger self-tests on every run (`--self-test`: one planted violation per arm).

## 10. Performance doctrine (design constraint, not a pass)

- Division of labor: Rust owns every per-position / per-record / per-leaf loop (board,
  legal moves, MCTS, graph build, contract validation on marshaled arrays). Python
  orchestrates and trains — vectorized numpy/torch only; a Python-level per-item loop on
  a hot path is a review-blocking defect.
- Graph input is built once per evaluated leaf, co-located with the NN forward. No
  search-time incremental graph deltas (falsified; see the register).
- amp policy: graph path = bf16 (law, pinned in code and by a regime-parity test).
- Profile first (flamegraph / py-spy; DHAT for allocation-rate hunting — allocation
  churn in hot loops is the first suspect; capacity-reserve fixes beat clever
  algorithms). Profiling builds: release + debug symbols (`profiling` profile).
- One optimization = pre-registered hotspot list + expected gain bracket + abort
  threshold, one change = one commit = one IQR-gated bench, parity oracles re-run after
  every hot-path change. Measure end-to-end steps/hr, not just the microbench. A
  measured structural floor is a finding, not a failure.
- Bench floors live in ONE machine-readable data file (`tools/bench_floors.toml`)
  consumed by both the bench script and the perf doc; per-host baselines are generated
  locally (`make bench.baseline`). No host's numbers are baked into code, defaults, or
  prose gates.
- `n_workers` defaults derive from `os.cpu_count()` via a documented formula; configs
  override only for stated experimental reasons.

## 11. Run-safety core (headless monitor + lifecycle)

- Event emit (JSONL) + producer manifest: every consumer binding — display OR headless
  gate input — must cite a live producer; the contract test includes a mutation
  self-test. No gate input without a producer test.
- Persistence failures are run-fatal by default: event-sink and buffer save/restore
  errors increment `persist_errors_total`, and the watchdog aborts on it. `except
  Exception: pass` is lint-banned; optional effects go through a `best_effort()` wrapper
  that requires a counter.
- Log identity: rotation on resume; a JSONL file never spans two run segments. The segment
  is claimed ATOMICALLY (`O_CREAT|O_EXCL` + bounded re-scan), so concurrent starts cannot
  share one file, and `run_id` is validated where the filename is built.
- Lifecycle is one subsystem: SIGINT/SIGTERM → save-then-exit (second signal force-
  exits), self-play stall watchdog (always armed), disk guard. Contract-tested.
- Livelock-proof watchdog (WP13-A): the pipeline stages emit heartbeats — train step,
  inference dispatch, selfplay drain — into a monotonic in-process registry; an
  INDEPENDENT watchdog thread (never driven from the step path) fires on per-source
  heartbeat staleness or on `persist_errors_total > 0`: loud event → time-bounded
  snapshot to the distinct `.watchdog` path (never the canonical resume buffer) →
  `os._exit` with a distinct code (42 stall/livelock, 43 persist-fatal). Every optional
  effect in the fire path carries a hard time budget: a hung snapshot may delay the exit,
  never suppress it. The in-loop games-progress tick watchdog is KEPT as a complement — it
  catches live-but-unproductive loops the heartbeat signal cannot see. A clean shutdown
  SWAPS the per-source deadlines for one bounded close-out deadline — teardown legitimately
  stops the heartbeats and legitimately runs long, but it is never unbounded, because an
  unbounded teardown is invisible to BOTH levels (the file `seq` keeps advancing, so the
  supervisor reads a wedged child as healthy). The persist-fatal fire is never disarmed.
  A source the composition root did not declare as wired, and which has never beaten, is a
  WIRING gap, not a wedge: it is reported loudly and never fires — an omitted heartbeat
  kwarg must not kill a healthy run.
- Supervisor liveness: the watchdog thread mirrors heartbeats to an atomically-replaced
  heartbeat file carrying a monotonic `seq`; the host-neutral out-of-process supervisor
  (`python -m mantis.monitor.supervise`) relaunches the child on exit 42 or on frozen
  `seq` (the watchdog-thread-starvation backstop: SIGTERM, grace, SIGKILL), stops loud
  on 0/43/other codes and on relaunch-budget exhaustion. Staleness is measured by seq
  progression on the supervisor's own monotonic clock — never file mtime, never wall
  clock.
- Process exit-code family (the supervisor-readable contract). One number per outcome, and
  the whole band is reserved by the run's own machinery — no tool of this repo may emit one
  as its own diagnosis:

  | rc | constant | authority | delivery |
  |---|---|---|---|
  | 42 | `WATCHDOG_STALL_EXIT_CODE` (= `lifecycle.watchdog.SELFPLAY_STALL_EXIT_CODE`) | `monitor/heartbeat.py` | `os._exit` from the watchdog thread |
  | 43 | `PERSIST_FATAL_EXIT_CODE` | `monitor/heartbeat.py` | `os._exit` from the watchdog thread |
  | 44 | `RELAUNCH_BUDGET_EXIT_CODE` | `monitor/supervise.py` | supervisor return — never a child's |
  | 45 | `ACTOR_LAG_EXIT_CODE` | `monitor/heartbeat.py` | `os._exit` from the watchdog thread |
  | 46 | `DRAW_RATE_COLLAPSE_EXIT_CODE` | `monitor/heartbeat.py` | **cooperative** — see below |
  | 47 | `DISK_SPACE_EXHAUSTED_EXIT_CODE` | `monitor/heartbeat.py` | **cooperative** — see below |
  | 48 | `TERMINAL_EVAL_BROKEN_EXIT_CODE` | `monitor/heartbeat.py` | **cooperative** — see below |

  46 (WPMINT Phase X, CARD-ABORT-EXIT / R84) deviates from the family on DELIVERY, and the
  deviation is the point. The draw-rate collapse abort stops the run by
  `StepCoordinator._fire_hard_abort` setting `shutdown.running = False` and RETURNING, so the
  loop unwinds through `close_out`, the terminal-eval drain and the shutdown checkpoint; an
  `os._exit(46)` would discard all three and contradict LAW-16 (save-then-exit), making the
  registration strictly worse than the gap it closes. **Parity is taken in the registry and in
  the supervisor's reading of the rc; delivery stays cooperative.** What carries the signal
  out of the loop is `ShutdownState.abort_rule` — the rule NAME the fire records, `None` on
  every clean stop — and a process boundary maps it to a number through
  `mantis.config.armed_aborts.exit_code_for_abort`, which reads the manifest row and never
  branches on the rule's identity. A rule with no manifest row resolves to `None` and NO code
  is invented for it. The number is written in exactly one place per authority: the constant,
  and the manifest row that imports it.

  47 (WPMAIN RED-TEAM RT-2 / R132) is the SECOND cooperative member, and it closes the same
  gap one leg further down LAW-16. `DiskGuard.check_once`'s critical arm SIGTERMs its own pid;
  with WPMAIN's handlers finally live that is save-then-exit — but `install_signal_handlers`
  writes `shutdown_save`/`running` and never `abort_rule`, which had exactly ONE writer in all
  of `src/`, so `mantis.run.main` read `rule is None` and returned **0**. A run the disk guard
  killed reported success and the supervisor above relaunched into the same full volume. The
  registration is the R84 shape verbatim: `mantis.config.armed_aborts.MANIFEST` gains a
  REQUIRED `disk_space_exhausted` row whose arming surface is `monitor.disk_guard.fail_gb`
  (minted on all six configs, `gt=0` in the schema, so a validated config arms it by
  construction — the row's job is to go RED the day the block is made nullable and the guard
  quietly disappears again), and its `exit_code` imports the constant.

  **The seam, because it is the part a reader will otherwise re-derive wrong.** The guard runs
  on its own thread and may not name the rule: the name is a manifest row's, and `mantis.train`
  does not import `mantis.config.armed_aborts` (the rule-name carrier's whole point). So the
  guard publishes a FACT — `DiskGuard.critical_fired`, latched — and `mantis.run.compose_run`'s
  teardown, which already owns `disk_guard.stop()`, reads that latch AFTER the guard thread is
  joined and records the rule through `ShutdownState.record_abort`. Consequences: no thread but
  the main one ever writes the run's stop state; the rule name has one spelling
  (`armed_aborts.DISK_SPACE_ABORT_RULE`, imported by the root); and `record_abort` is now THE
  writer of `abort_rule` for BOTH fire paths, set-once, first fire wins — the invariant
  `ShutdownState`'s docstring always claimed, enforced by the carrier instead of by two call
  sites agreeing to be careful.

  The LATCH is a second defect closed in the same arm (RT-2b): the guard polls every
  `interval_sec` (minted 60 s) on a condition that does not clear itself, so it supplied the
  SECOND press of LAW-16's two-press force-exit ITSELF — `sys.exit(1)` from a signal handler,
  mid-save, against `close_out`'s 14400 s drain caps. The two-press force-exit is the
  OPERATOR's affordance and it stays theirs; the `disk_alert` critical event still fires every
  tick, because the condition persists and an observer must see that.

  NOT covered by 47, stated because a supervisor depends on the difference: a signal the
  process did not send itself — an operator's SIGTERM, a supervisor's own stop — still resolves
  to **0**, since nothing records a rule for it. R132's scope is the guard, and whether a
  deliberate operator stop is a clean stop is a judgement no ruling has taken.

  48 (WP12-R Phase O / R152) is the THIRD cooperative member, and it discharges R133's
  measured caveat **"rc 0 does not certify eval health"**. `drain.close_out` computed the
  terminal round's result, routed it, and then DISCARDED the return value one frame below
  `ShutdownState` — the only object that can carry an outcome to `main` — while `promote.py`'s
  refusal to promote a broken round was the ONLY production consumer of broken-ness anywhere
  in `src/`. So a run whose terminal battery was killed, whose worker returned garbage or
  whose ladder state never reached disk exited **0**, and the supervisor above recorded a
  clean finish (LAW-15: no promotion decision = deliverable incomplete). The registration is
  the same R84 shape: a REQUIRED `terminal_eval_broken` manifest row whose arming surface is
  `train.terminal_eval_enabled` (a REQUIRED typed bool minted `true` on all six committed
  configs, so gate 12 is green on arrival and the row's job is the drift it makes loud — a
  production config minted with the terminal eval OFF is a red gate rather than a silent
  posture), and an `exit_code` that imports the constant.

  **The seam.** `mantis.eval.errors.EvalBrokenReason` is the ONE authority for WHY a round
  broke — seven `StrEnum` members, one per censused failure route, wire spellings unchanged
  from the bare literals they replace. The round result carries `eval_broken_reason` (`None`
  IS the clean state; the old `eval_broken: bool` and `error` fields are DELETED, not
  defaulted — two fields for one fact were two authorities that could disagree, R79).
  `drain.run_terminal_eval` latches that value on the coordinator, set-once, through the ONE
  writer in `src/` — reachable only from the one function that passes `ignore_stride=True`,
  which is what keeps R133's mid-run/terminal split STRUCTURAL rather than conditional. The
  latch stores a `str`, never the enum, because `mantis.train` may not import the eval
  package; the composition root RE-PARSES it through the enum before naming the rule, so an
  unregistered spelling is a loud `ValueError` at the process boundary rather than a silent
  rc 0. That read sits AFTER the disk-guard read in the same teardown, so `record_abort`'s
  first-fire-wins keeps the ROOT CAUSE: a disk-full run whose terminal eval then breaks
  reports 47, and a draw-rate collapse keeps 46.

  ONE number for SEVEN reason classes, on the record: this family is one number per OUTCOME
  and puts CAUSES in the payload (rc 45 covers every actor-lag fire), so the seven stay
  pairwise-distinguishable in the ONE channel — on the `eval_broken` event's `reason` and on
  the round result's `eval_broken_reason` — and never at the rc. A supervisor reading only the
  rc sees "terminal eval degraded" and not which break; seven codes nobody pre-registered
  would be six inventions, the class R84 refused.

  NOT covered by 48, disclosed rather than implied: an exception raised BEFORE or DURING the
  terminal eval (the staleness disarm, the flush, `on_drained`, or `write_model_snapshot` /
  `_spawn_worker` inside the round, which sit outside `_finalize_round`'s catch-all) leaves
  the latch unset and exits **rc 1** — loud, never a silent 0, but indistinguishable from a
  composition wall. Authoring an "the epilogue raised" code is `Q-RT-RC1-COLLISION`'s subject,
  not R152's.

  DISCHARGED (WPMAIN, CARD-RUN-MAIN). The OWED clause used to read: "`run_until_stopped` has
  no caller in `src/` and `mantis.run.main()` is smoke-grade, so the only production-posture
  process boundary that reads the resolver today is the mint preflight's boot child; when a
  production launcher lands it must read the SAME resolver rather than re-deriving the
  mapping." The launcher landed. `mantis.run.main` — `--config PATH --out-dir PATH`, both
  required, no code-side default anywhere on the surface — builds the collaborators, composes
  the run, drives the live loop and reads THIS resolver: `abort_rule is None` -> 0, a rule
  with an authored code -> that code, a rule with none -> `UnregisteredAbortExitError` naming
  the rule. Two process boundaries now, one resolver, and no number written at either.
  `StepCoordinator.run_until_stopped` is DELETED (R121(c)): a bare
  `while self.shutdown.running: self.step()` with no final save and no bound, zero callers
  and zero test references — wiring it would have forked LAW-16's save-then-exit into a
  second driver, and a differently-named "production entry" that nothing enters is a false
  affordance (R73/R116). `mantis.train.loop.run_training_loop` is THE loop.

## 12. Strength-claim + eval discipline

- Deploy-matched eval (argmax Gumbel-greedy head) is the DEFAULT promotion bar; a
  missing deploy decision blocks promotion, never falls back to a proxy regime.
- Strength claims ship protocol + n + eff_n (distinct games by trajectory hash) +
  per-side compute. Opening books are versioned, sha-pinned, paired; CI on pairs is a
  bootstrap percentile.
- Promotion-gate eval runs subprocess-isolated (own CUDA context; sidecar-JSON result
  contract, never stderr).
