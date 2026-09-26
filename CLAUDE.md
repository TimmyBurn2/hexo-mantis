# CLAUDE.md — mantis

mantis is an AlphaZero-style self-play bot for Hex Tac Toe (hex grid, 6-in-a-row, compound
2-stone turns, unbounded board): a Rust cargo workspace, Python training/eval (uv, src-layout),
one PyO3 bridge, GNN-first. This file holds what an agent must know that no gate enforces. Each
rule is stated once, here or in docs/governance/LAWS.md.

## Read first

- docs/governance/LAWS.md — the laws and the protected set with its pinning tests. They govern.
- docs/governance/falsified.md — the falsified register; LAW-05 says when it must be read.
- docs/governance/STATE.md — where the run is; docs/governance/CARDS.md — the open work;
  docs/governance/RULINGS.md — the rulings, newest first (grep it; nobody reads it whole).
- docs/design/repo_design.md — the structural contract, before any structural work.

## Map

- crates/: mantis-core (board, geometry, rules, Ply), mantis-graph (dep-free axis-graph builder,
  wasm32-clean), mantis-encoding, mantis-search (MCTS, Gumbel, tactics solver), mantis-selfplay
  (runner, inference queues, replay), mantis-bridge (ALL PyO3; maturin builds mantis._engine).
- The encoding registry is crates/mantis-encoding/src/registry.toml. Cite that path whole, `src/`
  included: a bare `registry.toml` was twice mis-cited to a crate-root path that does not exist.
- src/mantis/ is the one Python package (monitor is headless; deploy/ is reserved-empty until
  post-cutover). tests/ is the one collection root. configs/ are minted. tools/ holds dev tooling
  and the gate scripts. vendor/ is pins.toml plus `make vendor`.

## Rules no gate enforces

- Configs are minted with tools/mint_config.py, never hand-edited.
- Design standard: code and tests are keyed to mechanisms, never to a run (no `runN` in a symbol,
  test, pin or tool); production pins are a census of configs/, never a hand-kept list; one
  implementation per thing; no compatibility shim for a state the tree no longer has.
- Review gate: every implementation leg ends with a FRESH read-only review agent against the
  design standard plus correctness (budget, determinism, seam contracts, LAW-07 breaks). Its
  findings are fixed before merge, its report is a local record outside the public tree, and the
  dispatcher never reviews its own leg. A leg that skips either half is not done.
- Registers: a non-canonical working doc that disagrees with verified repo state is repaired in
  place by whoever finds it, noted in one line. RULINGS, LAWS and falsified correct only by
  annotation. A deviation from repo_design.md takes an amendment commit, never silent drift.
- Price law: renting, stopping, destroying or re-speccing the box is the operator's act. A session
  recommends it with the cost and the alternative stated, and never orders or performs it.

## Code

- Python: type hints on new or changed code; a public API has a docstring whose `Raises:` names
  every catchable exception; imports at the top of the file, a lazily loaded optional dep the one
  exception; a top-level `except Exception:` handler logs through `logger.exception` and does not
  repeat the exception in its message.
- Comments: a comment or docstring states what the code cannot, in one line, more only for an
  invariant; no ruling, card or finding numbers outside carve-out markers (pinned bands,
  planted-break markers, armed-value provenance, licence attribution). Gate 14 ratchets the
  measures; docstring length and `Raises:` are fixed on contact.
- Rust: no `unwrap()`/`expect()` on production paths; fail loud through a named error type that
  propagates. `expect()` is for tests and for startup invariants whose message names the
  invariant. `rustfmt <file>` also formats that file's `mod` children and `cargo fmt` (even
  with `-p`) a whole crate; format a touched file through stdin, as gate 18 checks it.

## Build, test, gates, commits

- `uv sync` (`make build`) is the one bootstrap. On the box, and on a CUDA desktop that gates (faster, and
  it runs the CUDA-gated rows), use `make build.cuda`: a bare `uv sync` silently swaps torch for the CPU wheel.
- A bare `pytest` is the default tier; `-m integration` and `-m slow` select the other two, and
  the `TIER:` header line says which ran. Read it rather than assuming.
- Cadence: targeted tests while iterating; `make gates` (tools/ci_gates/run_all.sh, whose row
  labels carry the gate numbers) at leg exit and before any push; `make gates.exit`, which adds
  the slow tier, at a packet exit. Doc/governance-only commits need no gates. Remote CI is
  suspended by operator decision, so the local set is the gate; gate 1's fresh-clone sync is the
  one check it does not reproduce.
- Toolchains are pinned (rust-toolchain.toml, mise.toml) and provision themselves. The Rust
  channel is attested in tools/bench_floors.toml, so a bump invalidates every bench floor: it is
  a perf-host event.
- vendor/external/ is per-checkout: every clone and the box re-run `make vendor` and
  `make vendor.strix`.
- Commits are ONE line, `type(scope): what changed and why it matters`: no body and no trailer
  of any kind, a tool's default attribution block included. Gate 19 reds a violation only at
  exit, when every commit after it must be rewritten, so write them right the first time.

## Deliberately absent

- Display surfaces. A display builds against docs/contracts/event_manifest.md. Three ruled
  exceptions exist, and none connects to a live run or adds a producer: `make dashboard` (an
  offline report from a run record), `make viewer` (a static page from game-record shards) and
  `make analyzer` (a loopback server over stamped checkpoints). A panel with no producer is drawn
  as a stated gap, never as a zero.
- Submodules and loose weights (vendoring is vendor/pins.toml only), requirements.txt (uv.lock
  is the lock) and setup scripts (`uv sync` is the bootstrap).
