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
- crates/mantis-encoding — crates/mantis-encoding/src/registry.toml (THE encoding registry,
  single source of truth) + spec + validators + dense encode kernels. Cite that path WHOLE,
  `src/` segment included: this line used to say a bare `registry.toml`, and a landed
  ruling was twice mis-cited from it to a crate-root path that does not exist (R309(b),
  ANNOTATION 7 — which carries the full account, and names the wrong string so this file
  does not have to).
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
8. **R8 file size.** 300-line soft cap; exceeding is fine WITH a justification in the
   file's opening comment or module docstring saying WHY the file is one unit. It states a
   reason, NEVER a line count: a transcribed tally must be re-edited on every edit, will
   eventually be wrong, and is then read as evidence (ratified G-DFIX-4 / R192(e),
   derive-or-delete). Sizes are derived by `wc -l`, never asserted. Gate 15 enforces both
   halves — the justification is present, and it states no count.
   Reason: keeps the audit greppable; unjustified growth hides structure drift.
9. **R9 registers.** docs/registers/falsified.md is read-before-optimizing;
   docs/registers/laws.md governs; deviations from docs/design/repo_design.md require an
   amendment commit, never silent drift. A NON-canonical working doc that disagrees with
   verified repo state is repaired in place by whoever finds it, noted in one line, no loop
   (R311(c)); register text still corrects only by annotation.
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

## Code style

- **Python.** Type hints on all new/changed code. Public APIs carry a docstring, and every
  catchable exception is named in a `Raises:` section. Imports at top of file — lazy-loading
  an optional dep is the ONE exception. Text-mode IO always passes `encoding=` (gate 16
  enforces tools/ and tests/ module scope; the RULE is the whole tree).
  Catch specific exceptions; bare `except Exception:` only in a top-level handler, which
  logs through `logger.exception` and does NOT repeat the exception in the message.
- **Comments (R316(e), operator direction).** Comments only where needed; public APIs carry
  docstrings. NO file-top banner comments and NO narrative comment blocks — a needed comment
  states its non-obvious fact in ONE line. CARVE-OUT: load-bearing in-source markers are
  mechanism, not commentary, and STAY — pinned bands, planted-break markers, armed-value
  provenance, licence-required attribution. Applied ON CONTACT, never as a cleanup pass.
- **Rust.** No `unwrap()`/`expect()` on production paths — fail-loud means a NAMED error type
  that propagates, never a panic (R2/LAW-13 is about what crosses the FFI; this is about not
  reaching for the panic in the first place). `expect()` is fine in tests and in startup
  invariants when its message names the invariant. clippy rides gate 2 (`-D clippy::all`);
  rustfmt is house style and is NOT gated anywhere — run it yourself, do not assume a gate
  caught it.

## Build & test

- `uv sync` — the ONE bootstrap: builds mantis._engine via maturin, installs everything.
- `make build` — alias for `uv sync`. `make build.native` — local perf build (env-only
  native flags; artifacts host-specific, never distributed).
- `make test` — pytest default tier + `cargo test --workspace --locked`.
- `make test.integration` — the CI integration tier (`-m integration`).
- A bare `pytest` IS the default tier: pyproject's `addopts` carries
  `-m 'not integration and not slow'`, a later `-m` overrides it, and `-m ''` runs or counts
  the whole tree (gate 3c does). The header prints `TIER:` on every run — read it, never
  assume the tier from the command typed (R330(g); the superset is ~35 min, the tier ~3).
- Cadence (R311(b)): targeted tests, smallest relevant first, while iterating; the FULL local
  gate set at leg exit and before any push, never per edit. **Remote CI is SUSPENDED by
  operator decision** until the operator re-enables it — no push or merge waits on it, and
  local green is the gate. Doc/governance-only commits need no gates at all. The accepted cost
  is on the record: gate 1's fresh-clone `uv sync` is the one check no local run reproduces.
  This sets WHEN gates run, never WHAT they check.
- `make bench` / `make bench.baseline` — criterion smoke bench (baseline saves locally).
- `make check.wasm` — mantis-graph must stay wasm32-clean.
- `make vendor` — fetch vendor pins. `make vendor.sealbot` — build the fetched sealbot
  extension (verifies the pinned sha and the applied patch first; `vendor/external/` is
  gitignored, so this is PER-CHECKOUT state every clone and the box must re-run).
  `make clean` — cargo clean + dist removal.
- Entry points are `python -m mantis.*` or console scripts — no loose script files.
- Python floor is 3.11 (CI pins 3.11; local interpreters may be newer).
- Rust toolchain is PINNED by `rust-toolchain.toml` (channel 1.97.1 + clippy, rustfmt,
  wasm32-unknown-unknown). rustup honours it automatically — no `rustup default`, no setup
  step, and it provisions the components and target on first use. The channel matches the
  rustc attested in `tools/bench_floors.toml`, so changing it invalidates all 28 bench
  floors: a bump is a perf-host event, not a local one. Without rustup the file is inert,
  and the `rust-version = "1.87"` MSRV in `[workspace.package]` is what refuses the build.
- Node is PINNED by `mise.toml` (`node = "26.7.0"`) and exists for ONE consumer: gate 14's
  pyright, which is a shim over node. Same contract as `rust-toolchain.toml` — committed,
  portable, auto-provisioned, no global config touched — but none of its weight: a node bump
  invalidates no bench floor, because nothing in `src/` or `crates/` touches node. Without mise
  the file is inert and gate 14 REFUSES loudly (rc 2) rather than reporting an unmeasured green.
- Commits are ONE line: `type(scope): what changed and why it matters`. Informative, not
  bloated — no body paragraphs, no trailing register/dispatch dumps, no multi-line footers.
  If the change needs more explanation than one line, it is more than one commit.

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
8. Registry sha handshake + audit (tools/ci_gates/registry_gate.sh — the handshake
   sub-check is LIVE and ARMED with its own LAW-07 mutation self-test; only the audit
   exit-0 sub-check is deferred, to the cutover battery).
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
14. Curated lint/type gate (tools/ci_gates/lint_gate.sh; `make lint`) — the pyproject ruff
    select + pyright (basic, src+tools) held at ZERO. A rule is adopted only with a named
    in-repo defect class AND a clean baseline (R98); exclusions are enumerated with grounds
    in pyproject.toml; the trigger self-tests every run.
15. R8 justification headers (tools/ci_gates/r8_header_gate.py) — every `.py`/`.rs` file over
    300 lines under src/, tools/, crates/, tests/ carries a justification, and NO justification
    states a line count. The second half is the load-bearing one: 47 headers stated a tally,
    at least 8 were already wrong, and run.py claimed 867 against 1024. A stale count is
    misinformation a future reader trusts (SF-7). Line counts are derived, never asserted.
16. Encoding-less text I/O (tools/ci_gates/encoding_io_gate.py) — `open`/`read_text`/
    `write_text` without `encoding=` default to the platform codepage, so they raise
    UnicodeDecodeError on any non-UTF-8 locale. ZERO in tools/ (these are the gates
    themselves); ZERO at MODULE scope in tests/ (a module-scope failure is collection-fatal
    and takes down the whole tier). Binary mode is correctly exempt; exemptions are
    self-expiring. Function-scope tests/ sites are a registered backlog, not a rule.
17. Rule-7 host content (tools/ci_gates/rule7_gate.py) — box specifics live in the migration
    workspace, never here. Absolute home paths, ssh invocation/config, `user@host`, IPv4,
    detached-run and provider names, over files added/modified vs `--base` (plus `--full-tree`).
    Rule 7 was memory-enforced until a scan found 101 committed box paths in a fixture that had
    been public since it landed. Operator-identifying terms are DELIBERATELY not in the tracked
    register — that would make the gate the leak; they go in an untracked local supplement, so
    the tracked half is a floor, not a ceiling. RFC-reserved domains and loopback/unspecified
    IPv4 are carved out IN the patterns; exemptions carry grounds + a blob sha and self-expire.

Every gate's check logic is a repo-local script or make target under tools/ — nothing
lives only in workflow YAML.

## Deliberately absent

- Display surfaces (web dashboard, viewer, TUI): the event-manifest JSONL contract
  (docs/contracts/event_manifest.md) is what any future display builds against.
- src/mantis/deploy/ is reserved-empty until post-cutover.
- Vendoring only via vendor/pins.toml + `make vendor` — no submodules, no loose weights.
- No requirements.txt (uv.lock is the lock), no setup script (`uv sync` is bootstrap).
