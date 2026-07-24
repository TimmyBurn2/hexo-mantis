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
                               root (WP11-A §a.4/§c.6); a top-level module ABOVE both
                               train and eval; NOTHING imports `mantis.run` — it is a
                               source-only DAG node. ADDITIVE: this row registers the new
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
  (a census test proves it stays absent).
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
| 5 | run config schema | v1 | pydantic models, extra=forbid; schema_version key in every file |
| 6 | replay persist | HEXB v9 / HEXG v1 | magics, versioned headers, wire-signature cross-load law, loud cross-format rejection |
| 7 | event manifest | v1 | every panel AND every headless gate input cites a live producer; mutation self-test proves the checker bites |
| 8 | community bot API | bot-api v1 | vendored OpenAPI 3.1 spec + BKE-notation round-trip suite |
| 9 | eval instrument | v1 | deploy-matched argmax head, frozen sha-pinned paired opening books, per-pair bootstrap CI, eff_n = trajectory-hash-distinct games |

Contract changes bump the version and update the contract doc + its tests in the same
commit. The PyO3 seam stays thin flat arrays (marshaling is a measured cost); per-field
copies are single-read by contract.

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

## 12. Strength-claim + eval discipline

- Deploy-matched eval (argmax Gumbel-greedy head) is the DEFAULT promotion bar; a
  missing deploy decision blocks promotion, never falls back to a proxy regime.
- Strength claims ship protocol + n + eff_n (distinct games by trajectory hash) +
  per-side compute. Opening books are versioned, sha-pinned, paired; CI on pairs is a
  bootstrap percentile.
- Promotion-gate eval runs subprocess-isolated (own CUDA context; sidecar-JSON result
  contract, never stderr).
