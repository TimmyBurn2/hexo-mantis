# MANTIS — REPOSITORY DESIGN (v1, founding contract)

AlphaZero-style self-play bot for Hex Tac Toe (6-in-a-row, unbounded hex grid, 2-stone
compound turns). Rust engine (cargo workspace) + Python training/eval (uv, src-layout),
PyO3 bridge. GNN-first, representation-extensible. This document is the structural
contract of the repository; CLAUDE.md carries the operating laws; docs/governance/ carries
law text, ruling texts, live state and the falsified register. Deviations from this document
require an amendment commit to this file — never a silent drift.

AMENDMENT (R346, 2026-09-09): `docs/registers/` is dissolved. `laws.md` and the two rulings
registers are frozen under `docs/governance/archive/`; the live governance files are
`docs/governance/{LAWS,STATE,RULINGS,CARDS,falsified}.md`.

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
│   ├── mantis-core/            # board, hex geometry, rules, the Ply vocabulary type
│   ├── mantis-graph/           # dep-free axis-graph builder; native + wasm32 targets
│   ├── mantis-encoding/        # registry.toml + spec + validators
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
│   ├── bots/                   # BotProtocol + the bot wrappers (sealbot, strix, random)
│   ├── monitor/                # HEADLESS ONLY: event emit, producer manifest, alert rules
│   ├── diagnostics/            # single-definition production-importable readouts
│   ├── util/                   # small helpers (device, coordinates, determinism)
│   └── deploy/                 # RESERVED: ONNX export + torch-parity check (post-cutover)
├── tests/                      # SINGLE collection root; mirrors src/mantis + crates
│   └── fixtures/               # frozen oracle banks (books, goldens, probe npz, positions)
├── configs/                    # explicit COMPLETE schema-validated configs; no variants-
│                               #   on-base; minted via tools/mint_config.py
├── docs/
│   ├── design/                 # this file + subsystem designs
│   ├── contracts/              # versioned seam contracts (see §4)
│   └── governance/             # LAWS, STATE, RULINGS, CARDS, falsified register
│       └── archive/            # FROZEN pre-R346 registers; never edited, never current
├── tools/                      # dev-only: mint_config, hardcode_scan, bench floors data
├── vendor/                     # pins.toml (url+sha+patch) + `make vendor` fetcher; no
│                               #   submodules, no loose weights
└── .github/workflows/ci.yml
```

Display surfaces (web dashboard, game viewer, TUI monitors) are deliberately absent.
The event-manifest + schema'd JSONL channels (§4.7) are the stable contract any future
display builds against — and `tools/run_dashboard.py` is the ONE thing admitted to build
against them (the amendment at the foot of this file). `src/mantis/deploy/` is reserved,
empty until post-cutover.

## 2. Dependency DAG (one-way; enforced by crate/package deps, checked in CI)

Rust:
```
mantis-core      → (nothing in-workspace)
mantis-graph     → (nothing; dep-free, wasm32-clean)
mantis-encoding  → core, graph          # pins graph consts into registry validation
mantis-search    → core                # LegalSetPolicy + gumbel HERE; encoding dev-dep only (SLIM-FIX)
mantis-selfplay  → core, encoding, graph, search
mantis-bridge    → all of the above    # the ONLY crate that knows pyo3 exists
```
Rules: no `#[pyclass]`/`#[pymethods]`/`#[pyfunction]` outside mantis-bridge (core crates
compile without pyo3). No core API takes or returns a Python wrapper type. `Board` takes
plain geometry values; registry-spec → values resolution happens in encoding/selfplay,
never inside core.

Python (import direction, top-level imports only; lazy imports need a stated reason):
```
util                         → (leaf)
diagnostics                  → SINK, not leaf: may import anything BELOW `run`; NOTHING imports
                               it. AMENDED (R9) by WORKER-SWEEP, R309(g). The old row read
                               "util, diagnostics -> (leaves)" and was ALREADY FALSE at that
                               commit, in two places: `diagnostics/fusion_calibrate.py` imports
                               `config.resolve.allocator_posture`, `config.loader`, `encoding`,
                               `model`, `selfplay.graph_collate` and `_engine`, and
                               `diagnostics/eval_child_memory.py` imports `eval.child_memory`.
                               CI gate 9 checks CYCLES ONLY (its own docstring), so nothing ever
                               saw the drift. This row is corrected rather than newly broken, and
                               the correction is the honest description of what the layer IS: a
                               production-importable readout that must measure the SAME program
                               the run executes — `fusion_calibrate.py`'s own module docstring
                               gives the reason ("a calibration that skips the softmax or the D2H
                               measures a different program"), and `diagnostics/worker_sweep.py`
                               inherits it, reaching `selfplay.pool` for exactly that reason.
                               WHAT ENFORCES IT, since gate 9 does not: the direction. Nothing
                               may import `mantis.diagnostics`, which is what keeps the sink a
                               sink and keeps every cycle through it unrepresentable — census
                               test `tests/test_import_layers.py::
                               test_nothing_outside_diagnostics_imports_mantis_diagnostics`. The
                               narrower ban that matters for the re-sit — `worker_sweep` may not
                               reach `mantis.train` or `mantis.run` at ALL — is a separate,
                               stronger census (`tests/diagnostics/
                               test_worker_sweep_reachability.py`), because R309(g) requires that
                               no trainer step be executable before the mint.
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
                               row; `mantis._engine` is imported at the top like any other
                               edge (R367 fix leg, F5.5: `selfplay.pool` already imported it
                               unconditionally, so the former lazy sites guarded nothing). ADDITIVE: this row registers the new
                               node; it does not weaken the train↛eval ban above, which
                               stays verbatim (census-tested,
                               tests/test_run_composition.py::
                               test_no_train_module_imports_eval_even_lazily).
```
The historical training↔eval / model↔training / bots↔bootstrap cycles are dissolved by
relocation (protocol→bots, dist65→model, eval hooks injected). CI runs an import-cycle
check (tools/check_import_dag.py) — a new top-level cycle fails the build.

## 3. Representation extensibility (the registry-kind axis)

- `crates/mantis-encoding/src/registry.toml` is the single source of truth for encoding and
  shape. Every entry carries `representation` (required, no default): `"graph"`
  today; the set is extensible. Unknown TOML keys are a parse ERROR; missing required
  keys are a parse error; the validator collects ALL errors before reporting.
- Rust: representation is a closed enum on `RegistrySpec`; dense-only or graph-only code
  paths are unreachable for the other kind (match exhaustiveness, no `_ =>` on the kind).
- Python: `mantis.encoding.lookup(name)` returns the `_engine` spec (delegation, single
  parser). Model/buffer/batcher construction dispatches through ONE authority per layer
  (`model.build.build_net(arch)`, buffer facade, batcher ctor). Reading arch attributes off
  live `nn.Module` instances is banned — arch metadata travels on declared dataclasses
  (`model.arch.GnnArch` / `model.arch.GnnArchV2`), which
  `build_net` consumes — and its dispatch ORDER is load-bearing, `GnnArchV2` first,
  because `GnnArchV2` is a subclass of `GnnArch`; a live-module
  representation sniff (the former `model_representation`) is DELETED and grep-gate-banned
  (a census test proves it stays absent). The discriminator, stated so the prose stops being
  broader than the gate: what is banned is *deriving* arch metadata from a live module's
  structure (a representation sniff, or reading arch hyperparameters off an `nn.Module`);
  *carrying* the declared dataclass instance itself as a handle is the convention, not a
  breach of it — `build_net` attaches `net.arch = arch` and `eval/snapshot.py` reads it back,
  which is the arch travelling with the model exactly as this section prescribes.
- No dense-by-default anywhere: an absent representation is an error, and `"grid"` is
  refused BY NAME (R346(f)), never resolved to the one remaining variant.
- The compiled module exposes `registry_sha()` (sha256 of the embedded TOML).
  `mantis.encoding` hashes the on-disk TOML at import in dev/test and hard-errors on
  mismatch — a stale extension cannot silently serve a stale registry.

## 4. Seam contracts (each has docs/contracts/<name>.md with: version, owner module,
   who-asserts-what-where table, pinning-test list)

| # | contract | ver | one-line summary |
|---|---|---|---|
| 1 | registry | v1 | TOML schema + validator invariants + audit CLI exit codes (0/1/2) |
| 2 | dense wire | RETIRED | deleted with the grid path (R346(f)); `archive/grid-path` carries it |
| 3 | graph wire (ragged) | v1 | block-diagonal GraphWire; the structural assertions and named errors are ENUMERATED in the contract, not counted here (AUDIT-1 F-52); single-read `take()`; −1 off-window sentinel travels; NO fixed-width fallback |
| 4 | checkpoint envelope | v2 | see §6 |
| 5 | run config schema | see the contract doc's version table | pydantic models, extra=forbid; schema_version key in every file. The version is the CONTRACT DOC's own table, whose last row is the authority — this cell read v8 while `docs/contracts/run_config_schema.md` had reached v13, and v13 while it had reached v32, so the cell now DEFERS to the doc; gate 13 now asserts the doc's header equals its own max row (AUDIT-1 F-52) |
| 6 | replay persist | HEXG v2 | magic, versioned header, slot-geometry guard, two-pass atomic load |
| 7 | event manifest | v1 | every panel AND every headless gate input cites a live producer; mutation self-test proves the checker bites |
| 9 | eval instrument | v1 | deploy-matched argmax head, frozen sha-pinned paired opening books, per-pair bootstrap CI, eff_n = trajectory-hash-distinct games |
| 10 | mint preflight report | preflight-mint-v1 | the mint preflight's evidence JSON: always written (LAW-14); mode, verdict and mint TIER derived from what the run DID, never from what it intended |
| 11 | game record | game-record-v1 | every game a run plays, one JSON object per line, sharded by (run, segment, hour) with an index; the SEGMENT is the writer and the hour is the window, so a resume never appends into a stopped process's file |

Contract changes bump the version and update the contract doc + its tests in the same
commit. The PyO3 seam stays thin flat arrays (marshaling is a measured cost); per-field
copies are single-read by contract.

### AMENDMENTS — contract #5 v1 → v8 (ARCHIVED)

The six AMENDMENT sections that chronicled contract #5's growth from v1 to v8 and the discharge of
the "deferred doc half" moved verbatim to
`docs/design/archive/repo_design_contract5_amendments_v1_to_v8.md` on 2026-09-17 (R355(f), REPAIR-A4
step 11). Structural facts they carried stay in force here: contract #10 was ADDED by the fourth of
them (row 10 above), `train.buffer_save_interval` was DELETED (v5 → v6), `monitor.gate_interval` was
ADDED (v7 → v8). `docs/contracts/run_config_schema.md` is the version authority.


## 5. Config system

- Every config file is explicit and complete about what the RUN DECIDES. There is NO
  inheritance, NO base-merge, and no value has a second authority. Loader = `yaml.safe_load` →
  schema validate. Unknown key = hard error (kills the silently-disabled-opponent class).
  `extra="forbid"` on every model. A missing key is a hard error unless the schema itself
  declares the key omittable — see the amendment below.
- A default lives in exactly one place: the schema field. Duplicated default authorities
  (code + yaml) are structurally impossible because no CALL SITE has a default.

  <!-- AMENDMENT (R347 / CONFIG-1, 2026-09-10): the two bullets above used to read "NO
       code-side default for any config value" and "code has no defaults", which contradicted
       the schema-field sentence between them and, from CONFIG-1 on, would be false. What
       changed: the OPERATIONAL CONSTANTS — a watchdog deadline, a poll interval, a
       subprocess-join bound, a disk threshold, a queue cap, a diagnostic switch — now carry a
       schema default and are absent from the YAML. 23 leaves moved; `configs/run6.yaml` went
       from 190 keys to 127. What did NOT change, and is the whole of why this is R1 rather
       than an exception to it: the default lives in the schema FIELD and nowhere else, no
       call site carries a fallback, and every ARMING key stays required — `gate_interval`,
       the actor-lag pair, `supervisor_kill_grace_sec` and the WR/axis warn family are all
       still mandatory, because that is where R1's silently-disabled-opponent reason bites.
       The set is DECLARED, with per-row grounds, in
       `mantis.config.schema.core.OPERATIONAL_DEFAULT_KEYS` — the one authority, checked in
       both directions by `tests/config/test_schema.py::test_o16_all_fields_required_no_code_side_defaults`
       so an undeclared default and a stale declaration are each a red. A run that wants
       another value MINTS the row (`mint_config.py --mint-row`), which is the same deliberate
       act and is visible in the stamped header; a defaulted BLOCK is minted whole.
       THE PROVENANCE HALF, which is what makes the shrink safe: `build_run_collaborators`
       writes the COMPLETE resolved config (`resolved_config.yaml`, every leaf including the
       defaulted ones) into the run directory before any collaborator is built, and re-reads
       it through `RunConfig` so the record is strict on its own bytes. A schema default
       revised later therefore cannot rewrite what an old run is recorded as having used. -->
- Copy-drift antidote (the cost of explicit-complete): `tools/mint_config.py` generates a
  complete config from a named template + an explicit delta mapping, stamping the delta
  into the file header; a one-key-diff assert tool verifies two configs differ exactly
  where claimed. Minted output is committed; the template is not a runtime input.
- One resolver per regime knob (sims, temperature, corpus, amp dtype, …); eval
  <!-- AUDIT-1 F-52: `radius` was in this list and there is NO radius resolver. The schema
       declares `legal_move_radius` DELIBERATELY ABSENT (`config/schema/selfplay.py`): radius
       is a registry IDENTITY key, which is the whole premise of the run6 re-mint, so a
       config-level resolver for it would be a consumer-less knob (R1/LAW-08). -->
  reads the same resolver seam self-play does. Unknown knob consumer = ValueError.
- Identity keys have no terminal defaults: absent encoding/representation is an error. `identity.arch_kind`
  is the ONE optional identity leaf (R330(e), amendment 2026-09-02): the arch-selector row,
  absent in every committed config until run6's mint writes it (R323(b)); an absent row resolves
  to the representation's pinned incumbent kind and a present one is refused by name if unknown.
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

  <!-- AMENDMENT (R366(b) / CARD-SERVER-OWNED-COPY, 2026-09-21): the payload gains ONE optional
       key, `ema_state` — the EMA shadow, the same key set as `model_state`, written only when
       `train.ema.enabled` is true (an EMA-off stamp carries no key, never a null). It is the
       DEPLOY net: `deploy_state(ck)` hands a deploy reader (the frontier cell, the ladder bot)
       the shadow when present and `model_state` otherwise, saying which; a resume restores it
       into the trainer's `EmaModel`; an EMA-on resume from a shadowless stamp re-seeds and emits
       `ema_shadow_reseeded`. What made it safe: the inference server serves its OWN copy of the
       declared arch (`mantis.selfplay.pool.served_copy`), written by ActorSync from the LEARNER's
       weights and never read by the learner, so the actors serve the learner while deploy, gate
       and follower read the shadow (`docs/contracts/checkpoint_envelope.md`). -->
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
  encoding) asserts suite default == production default.
  <!-- AUDIT-1 F-52: `radius schedule` was named here too. `RadiusStage` /
       `resolve_radius_from_schedule` are formally retired and the schema field is gone,
       so there is no radius arm for a regime-parity test to assert. -->
- Census pins (grep-gate tests) guard classes the type system can't reach; each names
  its bug class and its triage protocol.
- AMENDMENT (R367(a)/(b), 2026-09-21), the DESIGN STANDARD: code and tests are keyed to
  mechanisms, never to a run — no `runN` in a symbol, test, pin or tool; "production configs" is
  a CENSUS (discovered `configs/` minus the exempt set, each exemption carrying its reason as
  data), never a list edited per mint; one implementation per thing; a comment states what code
  cannot; no compatibility shim for a state the tree no longer has. Every implementation leg
  ends with a fresh read-only review against this bullet plus correctness (budget, determinism,
  seam contracts, LAW-07 breaks); the report lives under `docs/audits/`; findings are fixed
  before merge; the dispatcher does not review its own leg.
- Rust: inline unit tests near code; invariant pins and cross-language goldens under
  `crates/*/tests/`; proptest for board invariants; goldens are f32-bit-exact where the
  contract is numeric identity.

## 9. CI gates (PR → dev; `dev` is the only long-lived branch)

<!-- AUDIT-1 F-52: this heading read "PR → dev, and dev → main; main is merge-gated only".
     There is no `main` branch — `tools/ci_gates/test_count_gate.sh` sets `MAIN_BRANCH="dev"`
     and `dev` is what every gate compares against. The enumeration below listed gates 1-14;
     15, 16 and 17 exist (R8 headers, encoding-less text I/O, rule-7 host content) and are
     described in CLAUDE.md, which is the roster of record. Item 8's audit sub-check is
     described here as live; it is an `echo ... DEFERRED` and only the sha HANDSHAKE
     sub-check is armed. -->

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
    (`src/mantis/config/armed_aborts.py`) must be ARMED in every production config — the
    census, `mantis.config.census.production_configs`: every file under `configs/` minus the
    `EXEMPT_CONFIGS` rows, R367(a) — and every armed row must still be ABLE TO FIRE inside that config's
    own run. Read through the real loader; no boot, no burst, no GPU
    (`tools/ci_gates/preflight_mint.py --audit-only`). The second clause is R251 / ADJ-D22 and
    it is not a refinement of the first: `monitor.gate_interval: 1000000000` produces zero gate
    boundaries, so an armed `train.draw_rate_abort` is never evaluated — armed in the config,
    absent in effect — and the audit read it green because it never read the interval at all.
    `ge=1` bans one spelling of "never gate" and permits every larger one. So each armed row's
    EARLIEST POSSIBLE FIRE STEP is computed from that config's own cadence keys, declared per
    row as `ArmedAbort.cadence` + `cadence_paths` and derived from the code that evaluates the
    row, and a row whose value exceeds `armed_aborts.EARLIEST_FIRE_FRACTION *
    train.max_train_steps` fails at the same rc as a disarmed one. The fraction is a module
    constant beside the rows and deliberately NOT a config key — a config that could set its
    own audit fraction could relax its own audit (ADJ-D20's class relocated). A large interval
    is never a sanctioned disarm; the one sanctioned spelling stays the explicit R56-style
    deferred row. A `deferred` row is printed loudly on
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
    R9 AMENDMENT (R346(f), CLEANUP WAVE 2): gate 14 gained a third arm, the COMMENT
    RATCHET (`tools/ci_gates/comment_lint.py`, floor in
    `tools/ci_gates/comment_length_floor.txt`, producer `tests/tools/test_comment_lint.py`).
    It measures three counts over tracked `.py`/`.rs` under `src/`, `tools/`, `crates/`,
    `tests/` — lines beyond two in a run of own-line comments, banner lines, and lines
    beyond the first in a docstring — and enforces DIRECTION, not a cap: a measure may not
    exceed the committed floor, and the floor may not be raised. A cap was rejected because
    R346(f) itself permits a longer block that states a non-obvious invariant, so a cap
    needs an exemption list, and an exemption list nobody maintains is how a lint dies. The
    ruling-citation count is measured and printed but NOT gated: gate 15 requires the token
    `R8` in every oversized file's justification, so gating it would set two gates against
    each other. The arm runs before ruff and pyright so a pyright REFUSAL (rc 2, a host
    condition) cannot suppress the comment measures. It adds NO gate number — the set is
    still seventeen, and gate 18 stays reserved for RQ-21's `freeze_verify`.

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
- `n_workers` is a REQUIRED minted key with NO default anywhere (R1: a default lives only
  in the schema field, and this field has none). <!-- AUDIT-1 F-52: this row read
  "defaults derive from `os.cpu_count()` via a documented formula"; no such formula
  exists and no code-side default does either. `diagnostics/worker_sweep.py` is how the
  value is CHOSEN — measured on the box, picked by the knee rule, minted by the
  operator. -->; configs
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
  | 49 | `POLICY_LOSS_TROUGH_EXIT_CODE` | `monitor/heartbeat.py` | **cooperative** — the R350(b)(iv) trough halt, fired through `_fire_hard_abort` like 46 |
  | 50 | `PLY_CAP_ATTRACTOR_EXIT_CODE` | `monitor/heartbeat.py` | **cooperative** — the R352(c) ply-cap attractor halt, fired through `_fire_hard_abort` like 46 |

  Codes OUTSIDE the reserved band that a supervisor may nonetheless read from its child. They
  are not run-outcome diagnoses — the 42–50 band stays reserved for those — and none is minted
  here: each is either a pre-existing lifecycle constant or a POSIX convention.

  | rc | constant | authority | delivery |
  |---|---|---|---|
  | 71 | `PARENT_VANISHED_EXIT_CODE` | `monitor/heartbeat.py` | `os._exit` from `train/lifecycle/signals.py`'s entry-point arming gate (and from `arm_parent_death_signal`'s captured-ppid race check) |
  | 127 | `EXEC_FAILED_EXIT_CODE` (POSIX "command not found") | `train/lifecycle/arm_exec.py` | `os._exit` when the arming trampoline cannot `execvp` the child command |

  71 predates F-816-19 (`eae0fc4`) but was reachable only from an mp eval worker and the
  preflight boot child; F-816-19 put an `os._exit(71)` inside `mantis.run.main`'s arming gate,
  which makes it a code the SUPERVISOR reads. It lands in `_on_child_exit`'s named
  `child_parent_vanished` arm: propagated, never relaunched — a child whose supervisor was
  already gone has nothing to be relaunched into. It sits outside 42–50 deliberately: the band
  is the run's own diagnosis of its own work, and 71 says the run never began. (It is also
  `EX_OSERR` in `sysexits.h`; cosmetic, no collision.) Its constant is DEFINED in
  `monitor/heartbeat.py`, beside the rest of the family, and imported by
  `train/lifecycle/signals.py` — a code that both the supervisor and the run must name has to
  sit below the `monitor -> train` cut, for the same reason the parent-death env stamp does.

  A supervisor stopped by a signal it CAUGHT appears in neither table, and that is a decision
  rather than an omission: it forwards SIGTERM, waits `supervisor_kill_grace_sec`, escalates,
  and then dies OF that signal (`SIG_DFL` + re-raise) unless the child's own rc carries a
  diagnosis, in which case that rc is propagated. Its waiter therefore reads "terminated by
  SIGTERM", which is the truth and needs no number. The run itself still resolves to 0 on such
  a stop, exactly as the paragraph below already states.

  Neither table is read by any CI gate — gate 13's `DEFAULT_DOC` is `run_config_schema.md` and
  nothing else — so `tests/test_exit_code_table_census.py` derives the constant set from the
  source by AST and compares it against the tables in BOTH directions. Derived, never
  transcribed (R192(e)/G-DFIX-4), and carrying its own mutation self-test.

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

- Deploy-matched eval is the DEFAULT promotion bar; a missing deploy decision blocks
  promotion, never falls back to a proxy regime. **The bar runs the RUN'S OWN SEARCH** —
  see the `search.kind` amendment at the foot of this file; the PUCT-with-transformed-Q-root
  hybrid this row used to describe is deleted.
- Strength claims ship protocol + n + eff_n (distinct games by trajectory hash) +
  per-side compute. Opening books are versioned, sha-pinned, paired; CI on pairs is a
  bootstrap percentile.
- Promotion-gate eval runs subprocess-isolated (own CUDA context; sidecar-JSON result
  contract, never stderr).

---

### AMENDMENT — an OFFLINE, FILE-BASED run report is admitted; every display surface stays absent

**R333(d), REPAIR-3 Leg 4.** §1 says display surfaces are deliberately absent and names the
§4.7 JSONL channels as the contract any future display builds against. A run dashboard is
ordered. Under R9 that is a deviation from this file and it lands as an amendment, in the same
commit as the tool, rather than as drift.

1. **What is admitted, narrowly.** `tools/run_dashboard.py` — one command, an EXISTING run
   record in, one self-contained HTML file out. It adds no producer, opens no socket, runs no
   server, and has no connection to a live run. It reads the §4.7 JSONL stream and, when given
   one, the run's `eval_ladder_state.json`. It is dev-only tooling under `tools/`, which is
   where §1's tree already puts `mint_config` and `hardcode_scan`.
2. **What stays absent, unchanged.** The web dashboard, the game viewer and the TUI monitor
   (the game viewer is admitted later by the R352(g) amendment below, on this amendment's terms).
   The distinction is not size, it is COUPLING: an absent surface is one that would have to
   watch a run, and everything on that list would. A report generated after the fact from an
   artifact is the same shape as the preflight report (contract #9) and the sweep report —
   neither of which was ever read as a display surface.
3. **The rule the tool carries, which is why it is admissible at all.** A panel whose producer
   does not exist at HEAD is declared BANKED and drawn as a stated gap naming the producer that
   would fill it; a panel whose producer exists but whose series is empty in a given record is
   drawn as an ABSENCE naming the event. Neither is ever a zero, an empty axis, or a flat line.
   The tool refuses an event-less record rather than rendering a clean-looking empty page, and
   its `--self-test` drives both refusals plus the two shapes that would smuggle a number in
   (a chart drawn from no series; a bare zero with no absence label). "Absent is not zero"
   (class 7, the P1 packet's whole subject) applied to pixels.
4. **Measured at landing, and stated so the gap is on the record rather than in the code:**
   of the nine panels R333(d) names, **SEVEN are LIVE and TWO are BANKED** — *average
   sims/move* (`SelfPlayHParams.effective_sims_per_move` is derived in-process to BILL
   `sims_per_sec` and is emitted by nothing; the quotient of two differently-windowed rates is
   not the quantity) and *held-out loss* (`HeldOutMonitor.counters()` is LAW-18-shaped but sits
   on the BC pretrain path and reports through a logger line, never through the sink).
5. **§4.7 is unchanged.** The contract this reads through is the one that was already there;
   nothing about the event manifest moves, and no row is added to it.

---

### AMENDMENT — R352(g), VIEWER-1: the game viewer is ADMITTED on the dashboard's terms

**R352(g).** §1 and the R333(d) amendment above keep the game viewer on the "stays absent"
list. R352(g) orders it, over the GAME-RECORD-1 shards, "served from the mirror". Under R9 that
is a deviation from this file and it lands as an amendment in the same commit as the tool.

1. **What is admitted, narrowly.** `tools/game_viewer.py` (`make viewer RUNS="id=dir …" OUT=…`;
   the implementation is `tools/viewer/`) — one command, EXISTING game-record shards in
   (contract #11, `<run>/logs/games/`), one static directory out: an `index.html` carrying the
   light per-game index and one `data/<run>/<shard>.js` per shard, loaded by the page on demand.
   It adds no producer, opens no socket, runs no server and has no connection to a live run;
   "served from the mirror" means any file server over the output directory (or `file://`).
   It is dev-only tooling under `tools/`, beside the dashboard, and it reads the shards through
   `mantis.monitor.game_record.read_shard`, the contract's own reader.
2. **What stays absent, unchanged.** The web dashboard and the TUI monitor; the coupling rule
   of the R333(d) amendment is untouched — this tool watches nothing.
3. **The rule the tool carries.** Absent is not zero: the self-play channel records no
   per-position search stats (contract #11), and the page states that gap in words on every
   self-play game rather than drawing an empty heatmap; a games directory with no game of the
   run is REFUSED, never rendered as an empty list. Facts the page derives — the owner of a ply
   (the ply-to-turn mapping) and the six-in-a-row through the completing stone (`HEX_AXES`,
   `WIN_LENGTH`) — are derived in Python and checked against every record's own `result` and
   `termination` at build time; a disagreement is printed as a FINDING and carried on the page.
4. **Measured at landing:** run6's 38 988 games (35 287 self-play, 3 701 eval-channel games
   with search stats) and shakedown7's 2 421 — 41 409 — build in one page; the owner mapping
   and the derived line agree with every decisive record of both runs (0 findings).
5. **Contract #11 is unchanged.** Nothing about the game record moves; no field is added.

---

### AMENDMENT — R356(d): the dashboard reads a THIRD input, the strix follower's sidecars

**R356(d).** The R333(d) amendment above says the dashboard "reads the §4.7 JSONL stream and,
when given one, the run's `eval_ladder_state.json`". R356(d) orders an external-points panel —
the strix series with CIs, unit and regime labelled, the gap to strix as a number — and the
strix readings are NOT in the event stream or the ladder file: they are OFFLINE cells on frozen
checkpoints (`STRIX_RUN7_60K_2026-09-17.md` §C said so of the 60k series). Under R9 the new
input is a deviation from the amendment's text and lands here in the same commit as the code.

1. **What is admitted, narrowly.** `--external-points <dir|file>`, repeatable (`make dashboard …
   EXTERNAL="<run>/checkpoints <parent>/checkpoints"`): the `<ckpt>.strix256.json` /
   `.strix512.json` sidecars `tools/strix_follower.py` writes beside a run's checkpoints
   (contract #9's strix paragraph names their fields) — repeatable because a parent's bridge
   cell lives beside the parent, in another run's directory. Read at render time like the
   ladder file; omitted, the panel is a stated gap naming the producer. Still no server, no
   socket, no producer, no connection to a live run.
2. **The rule the panel carries.** Each (run, unit) is its own series and each unit (ours
   PUCT-256 vs strix 256; ours PUCT-512 vs strix 128) its own instrument, never merged — another
   run's point sits on that run's step axis and says so; every point carries the regime
   it was read in (CONTENDED / IDLE, from every run's heartbeat under the runs root at cell start); the y axis and the
   legend name the unit; the gap to strix is printed as a number (pp below parity, and the Elo
   it implies) for the latest point of every unit, not shown as a colour. A failed cell
   (`.failed.json`) is named in the panel's note and drawn as nothing.
3. **Throughput units (the same clause).** The rate charts leave "Data economy" for a
   "Throughput" panel where every label carries its unit and its producer: games / h, plies / h
   (Σ `game_complete.moves` per window ÷ the window's own wall — plies, LAW-03), turns / h
   (`positions_per_hour`, compound turns), leaves / s (`sims_per_sec`, the pool's bill of
   positions × effective sims per move — stated as billed, not served), steps / h. "Data economy"
   keeps the buffer fill and the replay-ratio gap.
4. **§4.7 is unchanged.** No event row is added or moved; the sidecar is an artifact of a
   dev-only tool, contract #9 describes it, and nothing in the run reads it.

---

### AMENDMENT — contract #11 ADDED: the game record

**R344(b).** A run's games are now WRITTEN. Under §4's own clause — *"each has
docs/contracts/<name>.md"* and *"contract changes … update the contract doc + its tests in the
same commit"* — a new seam contract lands as a table row and this note, in the commit that adds
it, rather than as an eleventh doc nobody's index names (R9).

1. **What it is.** `docs/contracts/game_record.md`, owned by `mantis.monitor.game_record`.
   Every game on all four channels — self-play, promotion, external rung, random floor — is
   written to `<out_dir>/logs/games/` as one JSON object per line, from step 0. It is a RUN
   RECORD, the same kind of artifact as the event stream and the preflight report (contract
   #10), and it is not a display surface: nothing here renders, serves, or watches.
2. **Why it is not §4.7's JSONL stream wearing a new name.** `game_complete` already carries a
   self-play game's move list into the event channel and continues to. That stream is keyed by
   TIME and interleaves forty event kinds, which cannot answer *"show me game 1 837"* without
   reading the whole run. This store is keyed by GAME and carries an index over its shards.
   **§4.7 is unchanged and no row is added to the event manifest** — `persist_errors_total`
   here feeds no gate, and contract #7 is headless-gate scope by its own text.
3. **The one law it inherits rather than re-invents.** Shards are claimed `O_CREAT|O_EXCL` and
   keyed on (run, SEGMENT, hour), which is `monitor/sink.py`'s law — *"no JSONL file ever spans
   two run segments"* — applied to a second writer. A run writes from two kinds of process (the
   trainer continuously, each eval round's child for its own lifetime), so an hour-only key
   would let a resume append into a stopped process's file.
4. **What is ABSENT and stated as a gap, not a zero.** Per-position search stats are LIVE on the
   eval channels and have **no producer** on the self-play channel: the visit distribution
   exists in the engine and reaches the replay ring, but every row is pushed `game_id=-1` by
   construction, so no position can be attributed to a game without an engine change on the hot
   drain path (LAW-09). The field is OMITTED from self-play records — never written as an empty
   list — and a test asserts the omission, so the day a producer lands the row reds.
   `CARD-GAME-RECORD-SELFPLAY-STATS`.
5. **§1's "deliberately absent" list does NOT move.** No display surface is admitted here. The
   game VIEWER R344(d) orders is a separate act that owes its own amendment, and this one does
   not pre-authorise it.

---

### AMENDMENT — `search.kind`: ONE search regime, read by ONE selector

**GUMBEL-2, wave 1.** §2's Rust DAG row for `mantis-search` says *"MCTS (PUCT + Gumbel)"* and
§12 described the promotion bar as *"PUCT with transformed-Q root selection … the tree is
PUCT and no Sequential Halving runs"*. Both moved. Under R9 that lands as an amendment in the
commit that moves them, rather than as drift.

1. **The regime is one key in its own top-level section.** `search.kind ∈ {puct, gumbel}`
   REPLACES four leaves, all deleted: `selfplay.gumbel_mcts`, `selfplay.gumbel_variant`,
   `selfplay.completed_q_values` and `train.completed_q_values`. A fifth,
   `selfplay.gumbel_root_counts`, is deleted without replacement — the root's own evaluation
   is charged against `n_simulations` on both arms, structurally, so `N` means `N leaves` and
   no config can falsify a fixed-node claim. In Rust the key is ONE closed `SearchKind` enum
   with no `Default`. Contract #5 moves v17 → v18; `docs/contracts/run_config_schema.md`
   carries the row, and unlike this file's two earlier bumps the doc half is NOT deferred.

2. **`search` is TOP-LEVEL and not a `selfplay` key**, for `eval_enabled`'s recorded grounds:
   it is a root-composition fact spanning more than one section's surface. Self-play searches
   with it and `mantis.arena.deploy_head` searches with it, and LAW-15's deploy-matched claim
   is precisely that the two are the SAME search.

3. **The deploy hybrid is deleted.** `DeployHeadPlayer` ran a PUCT tree whose ROOT pick was
   the Gumbel scoring function with its noise term set to zero — a third algorithm that
   appeared in no config, so the bar was matched to nothing. It now takes `search_kind` as a
   REQUIRED argument, hands it to the SAME `MCTSTree::configure_search` the worker calls, and
   plays each kind's own move rule: the most-visited child under `puct`, Sequential Halving's
   answer under `gumbel`. Its Gumbel draw is SEEDED, so the bar stays a reproducible
   instrument (LAW-15). `mantis.config.resolve.resolve_search_kind` is the ONE selector both
   sides read, and `tests/eval/test_search_kind_is_one_selector.py` is the source census that
   a second reader cannot pass.

4. **§2's DAG row reads the same and means something narrower.** `mantis-search` still
   carries PUCT and Gumbel; what is gone is the third arm and the lattice of flags. No crate
   moves, no import edge changes, and `pyo3` still lives only in the bridge.

5. **STANDING BLOCKER, recorded here because it is the reason a clause of the ruling could
   not land.** `search.kind: gumbel` on the GRAPH representation is refused at MINT and at
   BOOT by `replay::hexg::derived_visit_capacity`: that kind exports the completed-Q improved
   policy over the FULL legal set, so a row's support is the legal set — which is NOT a
   constant (355 median, 8 142 maximum at radius 8) and which no sims regime bounds, while
   the HEXG record's visit slot is derived from the sims regime. The graph lineage therefore
   cannot record a completed-Q target today. Closing it needs a MINTED visit-slot bound and a
   ruling on what a row that overruns it does; the ring cost is the subject (at 8 bytes a
   slot, 8 192 slots is ~65 KB per row against today's ~252 B).

6. **SECOND STANDING BLOCKER, raised in review and NOT fixed here.** `load_checkpoint`
   schema-validates the checkpoint's EMBEDDED config through the LIVE `RunConfig`, so every
   required-with-no-default field addition breaks every artifact written before it. This
   branch makes that WORSE by one error class: a pre-branch config now both carries keys
   `extra="forbid"` rejects and lacks `search.kind`, so it fails on both halves. **A
   pre-`search.kind` checkpoint therefore REFUSES to load, and the sanctioned recovery
   HOLDS** — `strip_and_restamp` (LAW-12's one weights-only path) never reads the embedded
   config, re-synthesises one from the live schema, and its output loads clean carrying
   `search: {kind: puct}`. `tests/train/test_pre_search_kind_checkpoint.py` asserts both
   halves. The ROOT defect — validating a HISTORICAL RECORD against today's schema — is not
   fixed here: it moves §6's *"schema-validated on write AND read"* and retires T-CK-04,
   which is a checkpoint-contract change (contract #4, LAW-12) belonging to a ruling.

---

### AMENDMENT — the SPARSE Gumbel row, and the two constants it unblocked

**GUMBEL-3, wave 2 (R347).** This moves contract #6's HEXG version, closes the STANDING
BLOCKER the GUMBEL-2 amendment recorded above, and moves two search-pool constants. Under R9
that lands as an amendment in the commit that moves them, rather than as drift.

1. **The HEXG record is v2, and a v1 file is REFUSED by name.** The record gained one field,
   the per-row tail mass α, written after `weight` — so every byte of a v1 record from that
   offset on means something else under the v2 reader, and the payload is self-consistent
   under both readings. Only the version field can see that, so the version check is the
   mechanism and not a formality. There is no in-place upgrade: a v1 ring is REGENERATED.
   Contract #6 moves HEXG v1 → v2; the frozen v1 byte-golden is KEPT as the refusal fixture.

2. **The blocker of §5 above is CLOSED.** `search.kind: gumbel` on the GRAPH representation
   is no longer refused. The refusal stood on the row having to carry the exported target's
   support, and R347(a) rules that it does not: under Sequential Halving only the
   `selfplay.gumbel_m` sampled candidates are ever visited, so the completed-Q target is
   EXACT on those m entries and, on every unvisited legal action, is the recording prior
   times ONE scalar — every unvisited child completes to the same mixed value. The row
   therefore stores m explicit `(action, target)` entries plus α, the trainer rebuilds the
   tail as `α × its own DETACHED current prior renormalized over the remaining legal set`,
   and the visit-slot bound is the MINTED `gumbel_m` rather than anything derived. The ring
   cost the blocker named is inverted: the row SHRINKS. What replaces the refusal is an m past
   `HEXG_GUMBEL_M_MAX`, refused at MINT and at BOOT by the same one authority.

3. **A THIRD refusal is added on that arm, and it is new.** The tail carries no per-cell
   shape, so a lever that injects target mass by cell after the export would put mass in the
   tail that the prior cannot reproduce. `selfplay.forced_win_policy_enabled` and
   `selfplay.solver_enabled` are therefore refused at BOOT alongside a graph Gumbel run.
   Both are `false` in every committed config, so nothing armed moves.

4. **`MAX_CHILDREN_PER_NODE` 192 → 1024 and `MAX_NODES` 1M → 4M (R347(c)).** No structure
   moves and no crate changes; what moves is the armed-sims ceiling the schema bounds every
   sims knob against, and the pre-allocated per-worker pool. Both ceilings are DERIVED from
   the two constants and are re-derived automatically; the numbers they now take are printed
   by `crates/mantis-selfplay/tests/audit1_named_errors.rs` rather than transcribed anywhere.

5. **The omitted-prior counters are PER-SEARCH.** They were process-global statics, so a
   bracketed measurement around one search admitted every other search in the process — a
   wrong number rather than a flaky one, and the conformance suite held a serialising mutex
   to work around it. The counters now live on `MCTSTree`. AMENDED (SLIM-FIX, R368(d)): the
   bridge no longer exports the process-wide totals — no Python reader ever consumed them —
   and mantis-search's statics that fed them are deleted, so the per-search set is the only one.

---

### AMENDMENT — the GRID/DENSE representation is DELETED (R346(f))

R346(f) deletes the grid/dense path and every parked, unmeasured lever that hung off it. The
tag `archive/grid-path` is the recoverable record; nothing below is reconstructible from this
document and nothing needs to be.

1. **The registry loses three rows.** `v6`, `v6w25` and `v6_live2_ls` are gone;
   `crates/mantis-encoding/src/registry.toml` registers `gnn_axis_v1` and `gnn_axis_r8`, both
   `representation = "graph"`. §3's "`"grid" | "graph"` today; the set is extensible" now reads
   `"graph"`, and the axis it names is unchanged: the key stays REQUIRED with no default, and
   `Representation::parse` REFUSES `"grid"` BY NAME rather than treating it as merely unknown —
   a stale row says what happened instead of falling through to the one remaining variant.
   The enum stays a closed type with one member for exactly that reason (LAW-11).

2. **The crates lose their dense halves.** `mantis-encoding` loses `encode/` (the dense plane
   kernels and the four plane-index constants). `mantis-selfplay` loses `queues::dense`, the
   HEXB ring (`replay::{storage,push,push_config,sample,persist}` and the `ReplayBuffer` type),
   `runner::rotate`, the dense recorder and its K histogram, `finalize_game`, the dense
   aggregators in `records.rs`, and the `SymTables` scatter machinery — `replay::sym` keeps
   `rotate_axial` / `N_SYMS`, which are the GRAPH D6 primitives. `mantis-core` loses
   `board/state/cluster.rs`. `mantis-bridge` loses the `ReplayBuffer` pyclass, the dense
   `InferenceBatcher` methods, `collect_data`, `apply_symmetries_batch`, `Board.to_tensor`,
   `Board.get_cluster_views` and `MCTSTree.expand_and_backup_ls`. §1's crate lines drop the
   "dense encode kernels" clause.

3. **The Python package loses its dense modules.** `model/cnn.py`, `model/cnn_heads.py`,
   `data/replay.py`, `data/replay_v6w25.py`, `data/augment.py`, `encoding/compat.py`,
   `train/recency_buffer.py`, `train/aux_decode.py`, `train/batch_assembly.py`,
   `train/pretrain/{dataset,trainer,freeze,validate}.py`. `model.arch.CnnArch` is deleted, so
   §3's arch triple becomes a pair (`GnnArch` / `GnnArchV2`) and `ModelArch` is their union.
   `env/game_state.py` survives as identity + history: `to_tensor()` and the 18-plane source
   layout went with the encode kernels.

4. **Contract #2 (dense wire) is RETIRED** and contract #6 becomes `HEXG v2` alone; both
   contract docs are amended in place. The cross-format magic law it carried has no second
   format to reject.

5. **Parked levers die with their keys.** solver-in-loop, forced-win one-hot injection, ZOI
   move filtering, trap-corpus seeding, the pretrained-corpus mixing schedule, the bot-batch
   share, every aux/entropy loss weight, the dense record rotation and the inference
   trace/compile/perf fields. `crates/mantis-search/src/tactics/` — the bounded minimax prover
   itself — is KEPT: falsified.md F-38 and F-39 were MEASURED with it and F-38's verdict names
   search-in-the-loop as the earned lever, so deleting it would delete the instrument those
   rows were measured with. What died is its self-play visit-injection hook, which was never
   armed and never measured.

6. **The ladder keeps one rung.** `kraken` and `strix` are deleted from `LadderRung.bot`, from
   `resolve_bot`'s known set and from `_KNOWN_OPPONENTS`, and `eval.kraken_model_sims` /
   `eval.strix_model_sims` go with them (their only consumer was the route to a permanent
   refusal). `sealbot_d5` is the minted rung; the adapter seam — `bots/protocol.py`,
   `resolve_bot` — is untouched, so a new rung is a row plus a factory (`SKIP_REASON_MARKERS`
   left with the sealbot adapter, R368(e)).

7. **`configs/` keeps three files, not two, and the third is stated.** `run6.yaml` and
   `smoke_preflight_armed.yaml` are what R346(f) names; `dev_example.yaml` is KEPT on LAW-07
   grounds and its `EXEMPT_CONFIGS` row says so — ADJ-13 N-3 makes it gate 12's M1 mutation
   row, the only real committed config that demonstrates the audit going RED on the real
   `configs/` tree, and with run5, the shakedown and the plain smoke gone it is the only
   DISARMED config left. All three are re-minted from `tools/config_templates/dev.yaml`, so
   their `# delta:` headers replay. `configs/run5.yaml` left the tree with
   `docs/contracts/eval_decision_run5.md`, whose drift gate derived every expectation from it;
   the two durable properties of that document — a one-lineage Bradley-Terry fit, and eff_n
   bounded by the openings on a deterministic rung — are folded into contract #9.

8. **Two instruments are retired rather than widened, and both say so in place.**
   `tests/config/test_minted_config_remint.py` diffed the live configs against a byte-frozen
   `b482243` baseline and tolerated exactly one ruled deletion; a mass deletion is not that
   shape, and the file's own §1 argues that re-cutting the baseline makes its directory name
   false and turns every assertion vacuous. It and its fixture are deleted, and
   `test_mint_header_roundtrip.py` carries the tombstone. `tools/bench_floors.toml` loses five
   floors with the two benches that produced them; no surviving floor moved, so the rustc/CPU
   attestation still holds and this is a deletion rather than a re-baseline.

### AMENDMENT — one OPERATOR-SIDE tool is admitted under `tools/`: the mirror puller (R349(b))

**R349(b), landed at `8e307af1`.** Recorded here rather than left as silent drift (R9).

1. **What changed.** `tools/mirror_pull.py` runs on the OPERATOR'S machine, not on a dev box
   and not on the box it mirrors: each cycle it rsyncs a run directory down from an rsync
   spec (an ssh alias plus a path, or a plain directory for the in-tree loop), verifies every
   complete bundle against its own manifest and every closed shard against the index's
   recorded size, writes `<artifact>.receipt.json` beside each verified copy and rsyncs the
   receipts back up. The `tools/` row's "dev-only" is therefore widened by exactly this one
   entry; it names no host (gate 17), and its transport is `rsync` alone.
2. **What it replaced.** R347(d)'s volume arm — `mantis.diagnostics.workspace_durability`,
   `mantis.util.mounts`, the planted-table seam and `PreflightStampPlantedTableError` — is
   DELETED: no host on offer has a volume, and a halt no host can pass gates nothing. The
   receipt primitives live in `mantis.util.mirror_receipts` (a leaf, so `mantis.config` can
   read the stamp's verdict), the artifact enumeration in `mantis.diagnostics.mirror_receipts`,
   and the run's own lag reading in `mantis.train.bundle_receipts` (the coordinator publishes
   it and cannot import diagnostics without a package cycle).
3. **Where it bites.** The preflight's rc 16 is `PreflightMirrorReceiptsError`, decided AFTER
   the boot on the boot's own bundle — or, since a clean completion writes a checkpoint and no
   bundle (R137's third leg), on its stamped checkpoint of record — and first shard; `mantis.run` refuses a stamp whose
   workspace verdict is not `MIRRORED`; `resume_state_persisted.unreceipted_bundles` feeds
   the dashboard's two-interval warning and halts nothing. `sha256_file` moved to
   `mantis.util.hashing` so the bundle, the receipts and the puller share one hash.

---

### AMENDMENT — `search.kind` SPLIT: `selfplay.search.kind` and `deploy.search.kind` (R351(c))

**R351(c), 2026-09-13.** The `search.kind` amendment above made the regime one top-level key so
that the eval head and the self-play workers would run the SAME search by construction (its
clause 2). run6 showed what that construction costs: every screen, round and promotion was read
through the TRAINING search's head, and that head was the failure (R351(a), `falsified.md` F-50).
Under R9 this lands as an amendment in the commit that moves it.

1. **Two keys, two homes.** The top-level `search` section is DELETED. `selfplay.search.kind`
   (inside `SelfplayConfig`) is what the workers run and what a stored ring's rows MEAN;
   `deploy.search.kind` (the new top-level `DeployConfig`, one leaf) is what the promotion bar
   and every ladder rung play. Both are `SearchConfig`, REQUIRED, no default. Contract #5 moves
   v26 → v27 and the doc half is not deferred. Twelve top-level sections still: one out, one in.

2. **Clause 2 above is REVERSED, not repaired.** LAW-15's deploy-matched claim is that the bar
   plays what will be DEPLOYED; nothing in it says the training search must be that search. The
   two may differ by construction — run7 mints PUCT deploy under either self-play kind. The σ
   (`selfplay.{c_visit, c_scale, q_rescale}`) stays ONE key set both heads read (R351(b)).

3. **One reader PER KEY.** `mantis.config.resolve.resolve_selfplay_search_kind` feeds
   `SelfPlayHParams`, the pool, the HEXG capacity derivation, the policy-target validator and
   the checkpoint's target-semantics guard; `mantis.config.resolve.resolve_deploy_search_kind`
   feeds `build_eval_pipeline` and the diagnostics that build a deploy head. The pre-split
   `resolve_search_kind` is deleted; `tests/eval/test_search_kind_is_one_selector.py` now
   pins that the self-play wire never calls the deploy resolver and vice versa.

4. **What follows which key.** `train.policy_target` and the resume guard follow the SELF-PLAY
   kind (the deploy kind plays games nobody trains on, so a resume may re-take it). The
   node-pool ceiling checks the self-play sims under the self-play kind and the eval sims under
   the deploy kind. No crate moves, no import edge changes.

### AMENDMENT — R355(f), the approved delete list: contract #8 is DELETED, two profiling harnesses go

**2026-09-17, REPAIR-A4 step 11.** Contract #8 (community bot API, `bot-api v1 (SKELETON)`) is
deleted: the row said "NEITHER EXISTS" since AUDIT-1 F-52, nothing under `src/`, `tests/` or
`vendor/` ever carried an `openapi` or `bke` token, and an intention with no seam is not a contract.
Row 8 is removed rather than renumbered; the numbering keeps the gap so every older cite of "#8"
still names what it named. `tools/profile_eval.sh`, `tools/profile_selfplay.sh` and
`tools/perf_prereg_skeleton.md` — a run5-era harness pair never run, one reading a config key that
does not exist, one citing a `plan/` directory that was never tracked — are deleted under the
same clause. `bots/` in §1 now names the three shipped wrappers.

### AMENDMENT — R362(c): the sealbot rung is DELETED; the gate is an instrument; the strix cell is the external scale

**R362(c), 2026-09-19.** Under R9 a structural deletion lands as an amendment in the commit that
makes it, never as drift. The grounds are `docs/design/measurements/EVAL_COST_2026-09-19.md`:
promotion selects nothing in the self-play loop (the actor runs the learner's weights on a
2-step cadence; `best_model` feeds only the next anchor, the resume and the next parent), and
the sealbot rung read 0.657 ± 0.045 flat over run7's 23 rounds while the strix series swung
0.045 → 0.170 → 0.111, at 34 % of run7's 50 h of eval wall, separating no promoted round from a
rejected one.

1. **The eval round is three phases, not four.** The strength-floor probe, the gate block (the
   internal comparator and parent selector) and the random floor. The `eval.ladder` block,
   `eval.sealbot_model_sims` and `eval.rung_concurrency` leave the schema (contract #5 v33);
   `mantis.eval.ladder` (the activation/graduation/calibration state), `mantis.eval.bt` (the
   Bradley-Terry fit) and `mantis.eval.channel_health` are deleted; `eval_ladder_state.json` is
   written by nothing; the resume sidecar's `last_p_hat` is read by nothing (a pre-R362 sidecar
   carrying it is tolerated as provenance, unbumped).
2. **The five consumers go with their producer (R4/LAW-07).** `sealbot_wr_abort` (a DEFERRED
   armed-abort row — with it the `EVAL_ROUND` sample clock and `Cadence.EVAL_ROUND_CONSEC`, which
   served that row alone), `sealbot_wr_warn` and the coordinator's `on_eval_round_complete` gate
   (§11's "sealbot-WR gate" is that method; a completed round now routes straight to its
   promotion decision through `drain._route_eval_result`), the nine `monitor.wr_*` keys and
   their A/B/C predicates in `monitor/rules.py`, `eval_channel_health`, and `wr_sealbot` on
   `eval_round_complete`. The rung-skip channels (`eval_rung_skipped`, `eval_rung_skip_class`)
   were the production rung's and go too.
3. **The rung machinery survives for ONE consumer, the strix cell.** `RoundSpec.rung_jobs`,
   `RungJob` (now carrying its own pair-bootstrap terms, since no config block does) and
   `worker._play_rung_block` are how `tools/strength_frontier.py` and `tools/strix_follower.py`
   play the external scale (R352(e), R356(a)); a production round carries `rung_jobs: []` and a
   cell names its opponent (`strix`, or a snapshot through the gate) — the tool's old default
   opponent, `sealbot_d5`, is gone. The sealbot ADAPTER (`bots/sealbot`) is DELETED by R368(e),
   `find_vendor_root` moved to `bots/strix` first; its vendored build and pin go under the same
   ruling.
4. **What the stream gains in the same commit.** `eval_round_complete.gate` carries the gate's
   rule fields (`rule`, `pairs_played`, `stopped`, `llr`, `wr_confirm`, `n_pooled`, `promoted`,
   `wall_sec` — the gate BLOCK's own wall, beside the row's round wall), `null` when no gate ran
   (CARD-EVAL-GATE-FIELDS-IN-STREAM; contract #7). A stream reader can now see how a round
   stopped and what it cost without the spool.
5. **The dashboard (the R333(d) amendment above) keeps its `--ladder-state` input for pre-R362
   records** and draws the rung as a STATED GAP on a record that carries no reading — never as a
   zero; the run's external scale is the strix points panel (R356(d)).
6. **The stamp loader widens its tolerance to nested retired paths.** `RETIRED_STAMP_SECTIONS`
   (top-level only) becomes `RETIRED_STAMP_PATHS` carrying the dotted paths above, because run9's
   warm start is a run8 checkpoint whose stamp carries every one of them (contract #4).

---

### AMENDMENT — LADDER-1 (CARD-LADDER-RUNG): a SECOND operator-side tool is admitted under `tools/`, the ladder bot

**The LADDER-1 packet, 2026-09-19; the card is `docs/governance/CARDS.md` CARD-LADDER-RUNG.** Recorded
here rather than left as silent drift (R9), beside the R349(b) amendment whose "exactly this one
entry" it widens to two.

1. **What it is.** `tools/ladder_bot.py` with the `tools/ladder/` package runs on the operator's
   VPS, one process per registered bot on a HeXO server's bot API (the Hexo-Bot-Api contract,
   verified against the DEPLOYED server, which is ahead of the spec file): hold the NDJSON stream
   (presence and availability ARE the connection), answer every move request through ONE backend
   behind one seam — `mantis`, the deploy head (`mantis.arena.deploy_head`) on a named checkpoint
   under the run config's own `deploy.search.kind`, `eval.gate.deploy_sims`, σ and batching; or
   `strix`, the pinned rung through its own driver, solver ON, noise off — and write one receipt
   per game under `receipts/<net_hash[:8]>/`, keyed like the follower's sidecars. Endpoint strings
   live in `tools/ladder/client.py` and nowhere else (pinned by test).
2. **Posture A.** Ladder games are EVAL. The package imports no ring writer, and a test pins the
   absence; nothing a ladder game produces reaches self-play data.
3. **What it is NOT.** Not a series, not a promotion input, not in any prereg: the receipts are
   a shakedown's until a ruling (R363 or later) reads one 288-game cell beside a follower cell on
   the same checkpoint and admits the rung as an instrument. §12's strength discipline is
   untouched — a ladder result is a different unit (server openings, the server's clock), labelled
   as such wherever it is read.
4. **The two head counters it added.** `DeployHeadPlayer.last_sims` and `StrixBot.last_sims`
   (each head's own count of the leaves the last search spent), so the receipt's budget column is
   read off the head, never tallied by the caller.
5. **The unit, fixed by R363(c) (2026-09-20).** `book_v1_s20260625_p4` PAIRED — the follower's
   unit — with opening index = match index: pair `m` (games `2m` and `2m+1`, colours swapped by the
   challenger's alternating `firstPlayer`) plays opening `m` in the book's file order, a CONVENTION
   BETWEEN OUR TWO BOTS because the server's challenge carries no opening field. Both bots
   translate the opening onto the server's auto-placed origin and play its prefix unsearched
   (`tools/ladder/openings.py`; the receipt is schema v2 with the opening and `book_stones` per
   move; `--replay` re-derives the forced stones from the receipt's opening). §3's "server
   openings" is thereby retired as a unit difference; the server's clock and the host's CPU remain.
   The admission test is ONE 288-game IDLE cell beside a follower cell on the same checkpoint. The
   64-sim `--preset play` R363 §0(5) allows is the ladder tool's own row, labelled on every receipt,
   never a unit reading.
---

### AMENDMENT — R363, ANALYZER-1: an interactive position analyzer is ADMITTED on the viewer's terms plus one loopback socket

**R363.** §1 keeps display surfaces absent because each would WATCH A RUN: the R333(d)
criterion is coupling, not sockets, and every tool admitted so far happened to need neither.
ANALYZER-1 is a position analyzer: a position in (typed, pasted, or a deep link), one or more
engines' readings out. It needs a process, so it is the first admitted tool that LISTENS on a socket
(the ladder bot above holds an outbound stream);
under R9 it lands as an amendment in the same commit as the server. The design and its review
are `docs/design/analyzer_design.md`.

1. **What is admitted, narrowly.** `tools/position_analyzer.py` (`make analyzer CHECKPOINTS=…
   [STRIX=1]`; the implementation is `tools/analyzer/`) — a stdlib `ThreadingHTTPServer` bound
   to loopback by default (`--bind 0.0.0.0` is documented as unsafe, never the default), serving
   ONE page and answering `POST /analyze` by running the run's own `DeployHeadPlayer` on stamped
   checkpoints (immutable, R3) and, opted in, the vendored strix pin through its driver. It
   watches no run, reads no run record, holds no file open, polls nothing, adds no producer and
   writes no artifact. It is dev-only tooling under `tools/`, beside the dashboard and the viewer.
2. **What stays absent, unchanged.** The web dashboard and the TUI monitor; the coupling rule of
   the R333(d) amendment is untouched. DASH-2 (a server over the run record carrying the viewer)
   is neither discharged nor engaged by this.
3. **The rule the tool carries.** Every number on the page is derived in Python from a named
   engine call and carries its derivation; the NET's value is read through a quiescence-off tree
   and the HEAD's value is the deploy head's own, its quiescence override included and counted;
   values are for the side to move, stated once; a search not yet run is a STATE with its elapsed
   time, never a stale number; a panel not requested is a stated cost, never a blank; a position
   the engine refuses is refused by name; strix's gaps are stated on its card. Absent is not zero,
   applied to a live engine.
4. **Measured at landing** (the operator's CPU host, `run8_00018000_98fe0e8c`, 302 466 params, 8
   torch threads): engine load 50 ms; `MCTSTree()` 45–50 ms; the net's read through the cached
   tree 20–37 ms; search 64 sims 1.3 s, 256 sims 4.3 s. The design's §0 table (default threads)
   reads 32 / 128 / 256 / 512 sims = 1.7 / 3.3 / 5.6 / 11.7 s.
5. **Contracts #11 and §4.7 are unchanged.** Nothing is emitted; nothing is read from a record.

The ruling number is R363(d) (2026-09-20), which admitted the tool on these terms.
