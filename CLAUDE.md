# CLAUDE.md — mantis

mantis is an AlphaZero-style self-play bot for Hex Tac Toe: hex grid, 6-in-a-row to win,
compound 2-stone turns, unbounded board. Rust engine (cargo workspace) + Python
training/eval (uv, src-layout), PyO3 bridge, GNN-first. Read docs/design/repo_design.md
(the structural contract) before structural work. Read docs/registers/falsified.md
before proposing ANY optimization or experiment. Law text: docs/registers/laws.md.

## Map

- Cargo.toml + pyproject.toml — cargo workspace and uv project root (src-layout).
- crates/mantis-core — board, hex geometry, rules, Ply/Turn vocabulary types.
- crates/mantis-graph — dep-free axis-graph builder (native + wasm32), sits below
  mantis-encoding in the DAG; `make check.wasm` targets it (and mantis-encoding if it
  becomes wasm-targeted).
- crates/mantis-encoding — registry.toml + spec + validators + dense encode kernels.
- crates/mantis-search — MCTS (PUCT + Gumbel), completed-Q, tactics solver.
- crates/mantis-selfplay — runner, worker loop, inference queues, replay buffers.
- crates/mantis-bridge — ALL PyO3 lives here; maturin builds mantis._engine (abi3).
- src/mantis/ — the ONE Python package: encoding, config, env, data, model, train,
  selfplay, eval, arena, bots, monitor (HEADLESS ONLY), diagnostics, util, deploy
  (RESERVED, empty until post-cutover).
- tests/ — SINGLE collection root, mirrors src/mantis + crates; tests/fixtures carries
  the fixtures manifest. configs/ — minted, complete, schema-validated. docs/ — design +
  contracts + registers. tools/ — dev-only tooling + CI gate scripts. vendor/ —
  pins.toml + `make vendor` fetcher.

## Hard rules

1. **R1 config.** Every config file is explicit + complete; schema `extra="forbid"`;
   missing key = error, unknown key = error; NO code-side defaults — a default lives only
   in the schema field; configs are minted (tools/mint_config.py), never hand-varied;
   identity keys (encoding/representation) have no terminal defaults; every config key
   has a live consumer.
   Reason: kills the silently-disabled-opponent and duplicated-default-authority classes.
2. **R2 build/FFI.** The extension profile uses `panic = "unwind"` — panics cross the FFI
   as PanicException, never a process abort; no `target-cpu=native` in committed build
   config — native builds go through `make build.native` (env-only).
   Reason: a process abort loses runs; host-pinned artifacts are not portable.
3. **R3 artifacts.** Checkpoint stamps are written once, immutable; artifact filenames
   carry run-id + content hash; an artifact that cannot be stamped cannot be written.
   Reason: re-stamping and unstamped saves destroyed provenance once (see LAW-12).
4. **R4 gates.** No gate/monitor input without a producer test; every registered encoding
   has a live consumer.
   Reason: a phantom gate input once armed an abort chain no producer ever fed (LAW-07).
5. **R5 tests + imports.** Single collection root `tests/`; no package named `tests` below
   it; ZERO `sys.path` mutation anywhere in the repo.
   Reason: collection shadowing and path hacks made failures unreproducible before.
6. **R6 FFI surface.** No `#[pyclass]`/`#[pymethods]`/`#[pyfunction]` outside
   crates/mantis-bridge; core crates compile without pyo3.
   Reason: keeps every crate but one Python-free and the DAG one-way.
7. **R7 artifact hygiene.** reports/, checkpoints/, logs/, benchmarks/ are never tracked;
   files >1 MB and `*.jsonl` only under tests/fixtures/, and even there a 10 MB per-file
   ceiling holds — the carve-out is a raised ceiling, not an exemption (CI gate 6 enforces).
   Reason: run outputs in git history are unremovable and poison clones.
8. **R8 file size.** 300-line soft cap; exceeding is fine WITH a one-line justification
   at the top of the file.
   Reason: keeps the audit greppable; unjustified growth hides structure drift.
9. **R9 registers.** docs/registers/falsified.md is read-before-optimizing;
   docs/registers/laws.md governs; deviations from docs/design/repo_design.md require an
   amendment commit, never silent drift.
   Reason: re-litigating falsified work and silent contract drift burned weeks before.

## Laws digest (full text: docs/registers/laws.md)

- LAW-01 prime directive — context first, measurement mandatory.
- LAW-02 re-validation discipline — never drop a driver on an un-re-validated prior.
- LAW-03 measurement-unit — verify turn-vs-ply and the completing cell before framing.
- LAW-04 effective-n — strength CIs count DISTINCT games (trajectory-hash dedupe).
- LAW-05 falsified-register-first — read the register before proposing experiments.
- LAW-06 bf16-graph — graph-path autocast dtype is bf16, pinned by parity test.
- LAW-07 producer-test — every gate/monitor input cites a live producer + mutation self-test.
- LAW-08 live-consumer — every config key / registered encoding has a live consumer.
- LAW-09 bench discipline — prereg hotspots, one change = one commit = one IQR-gated bench.
- LAW-10 threat-probe criterion — C1–C3 gate checkpoints; anchor-matched baselines.
- LAW-11 identity-keys — no dense-by-default; absent encoding/representation = error.
- LAW-12 checkpoint-stamp — stamps immutable; one loader; weights-only strip is the one path.
- LAW-13 FFI/build — panic="unwind" across FFI; no target-cpu in committed config.
- LAW-14 persistence-fatal — persistence failures are run-fatal; no silent excepts.
- LAW-15 eval-instrument — deploy-matched promotion bar; reproducible fixed-depth bars.
- LAW-16 lifecycle — signals save-then-exit; stall watchdog always armed; disk guard.
- LAW-17 structure — zero sys.path writes; one tests/ root; pyo3 only in the bridge.
- LAW-18 in-run observability — a lever under test logs its own fire-rate in-run.

## Build & test

- `uv sync` — the ONE bootstrap: builds mantis._engine via maturin, installs everything.
- `make build` — alias for `uv sync`. `make build.native` — local perf build (env-only
  native flags; artifacts host-specific, never distributed).
- `make test` — pytest default tier + `cargo test --workspace --locked`.
- `make test.integration` — the CI integration tier (`-m integration`).
- `make bench` / `make bench.baseline` — criterion smoke bench (baseline saves locally).
- `make check.wasm` — mantis-graph must stay wasm32-clean.
- `make vendor` — fetch vendor pins. `make clean` — cargo clean + dist removal.
- Entry points are `python -m mantis.*` or console scripts — no loose script files.
- Python floor is 3.11 (CI pins 3.11; local interpreters may be newer).

## CI gates (all locally runnable — run them before pushing)

1. Fresh-clone `uv sync` builds the extension (tools/ci_gates/gate_01_fresh_sync.sh).
2. `cargo test --workspace --locked` + clippy (pedantic=warn baseline, `-D clippy::all`).
3. pytest default tier; integration tier; collected-test count non-decreasing
   (tools/ci_gates/test_count_gate.sh vs the committed floor file).
4. `make check.wasm` green.
5. Bench smoke (`make bench`, trivial criterion bench).
6. Artifact rejection (tools/ci_gates/artifact_gate.py): artifact dirs, >1 MB adds,
   >10 MB fixture adds, stray `*.jsonl`.
7. Every configs/ file schema-validates (tools/ci_gates/validate_configs.py; empty = fail).
8. Registry sha handshake + audit (tools/ci_gates/registry_gate.sh — auto-arming stub
   until the registry ports; its trigger is self-tested).
9. Import-DAG check (tools/check_import_dag.py — no top-level cycles).
10. No Makefile/doc reference to untracked paths (tools/ci_gates/check_tracked_refs.py).
11. No silent encoding-fallback arms (tools/ci_gates/silent_encoding_gate.py) — an absent
    encoding raises, never defaults (LAW-11/LAW-05).
12. Armed-abort manifest audit (tools/ci_gates/preflight_mint.py --audit-only) — every
    `required` row of src/mantis/config/armed_aborts.py is armed in every production
    config; deferred rows print loud and do not gate. The same tool's full mint preflight
    (a real boot + burst) is MANUAL, invoked by no CI step.
13. Contract-doc drift (tools/ci_gates/contract_doc_gate.py) — docs/contracts/run_config_schema.md
    may not cite a config key or a `mantis.*` symbol the shipped schema lacks, and its stated
    leaf-count must equal the live one. Every check is derived from `RunConfig` itself, never a
    transcribed key list; the "deliberately absent" section is checked in REVERSE.

Every gate's check logic is a repo-local script or make target under tools/ — nothing
lives only in workflow YAML.

## Deliberately absent

- Display surfaces (web dashboard, viewer, TUI): the event-manifest JSONL contract
  (docs/contracts/event_manifest.md) is what any future display builds against.
- src/mantis/deploy/ is reserved-empty until post-cutover.
- Vendoring only via vendor/pins.toml + `make vendor` — no submodules, no loose weights.
- No requirements.txt (uv.lock is the lock), no setup script (`uv sync` is bootstrap).
