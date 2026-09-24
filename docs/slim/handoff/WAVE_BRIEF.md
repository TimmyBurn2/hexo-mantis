# WAVE BRIEF — rules every SLIM-FIX (R368) implementation agent follows

Read, in your worktree: CLAUDE.md (hard rules, code style, commits); docs/governance/RULINGS.md entry
`### R368` (the ruling you work under — especially (b) protection, (c) correctness first, (g) comments,
(h) out of scope, (i) run10 safety); docs/slim/01_DEFECTS.md (the defect list); docs/slim/00_MAP.md §3
(the protected set PZ-1..PZ-6) and §4 (the live-caller checklist); the scout file(s) named for your rows.

## Your workspace
- You work ONLY in your own git worktree (path given in your prompt), on its branch. It has its OWN venv:
  run python as `<wt>/.venv/bin/python` (e.g. `.venv/bin/python -m pytest ... -p no:cacheprovider`), from
  inside the worktree. Never `uv sync` / `uv run` (they try to rebuild). Export
  `XDG_STATE_HOME=$TMPDIR/xdg UV_CACHE_DIR=$TMPDIR/uvcache` for every test run.
- NEVER touch the main checkout (the repo root) or another agent's worktree.
- If you change Rust in a crate the Python extension links (mantis-core/-encoding/-graph/-search/-selfplay/
  -bridge) and need Python tests to see it, rebuild the extension into YOUR venv with
  `cd <wt> && uv sync --reinstall-package mantis-engine` — this ONE command may run outside the sandbox (it
  writes uv's cache under ~/.cache/uv; set dangerouslyDisableSandbox for it and nothing else). Then check
  `.venv/bin/python -c "import mantis._engine"`.
- This host is the operator's desktop with a long gate sweep and other agents running: targeted tests only,
  never the full pytest tier, never `cargo test --workspace` in full (use `-p <crate>` and `--test <name>`).

## What "done" means for a defect (R368(c))
1. Repair it at the cause, smallest change that makes the witness/gate TRUE.
2. PROVE it by a planted break: plant the defect (or a mutation of the repaired code that re-creates the hole),
   run the repaired witness/gate, show it goes RED (quote the one failing line), revert the plant, show GREEN.
   A witness that stays green under its plant is a HALT: stop that row and report it.
3. A repair may EXTEND a protected witness (add rows/assertions), NEVER narrow one.
4. Record the planted-break command and the red line in your report (not in the commit body).

## Hard limits (a breach is a HALT — stop the row and report, do not work around)
- run10 safety (R368(i)): no change to any configs/*.yaml, schema keys or defaults, the checkpoint/stamp format,
  or trainer / search / eval NUMERICS. Adding an observability counter or event field is not numerics.
- Protected symbols (00_MAP §3 PZ-1 implementing symbols and pinning tests, PZ-2 seam members, PZ-6 named
  code): edit one only as far as the defect requires and name it in your report; never weaken a check.
- tests/model/conformance/**: only defect 10's repair and stale text.
- Minted configs, tools/mint_config.py, tools/config_templates/**: do not edit.
- Do NOT edit tools/ci_gates/test_count_floor.txt or tools/ci_gates/comment_length_floor.txt — the dispatcher
  folds floor moves into your commits at integration. Report the measures instead.
- Never `git add -A` / `git add .` (the sandbox mounts dotfiles into the tree). `git add <paths>` only.

## Style (the checks that WILL red if ignored)
- Comments (R368(g)): a comment or docstring states what the code cannot, ONE line; more only for an invariant.
  No ruling/card/finding numbers in new comments except carve-out markers (pinned bands, planted-break markers,
  armed-value provenance, licence attribution). Public APIs get a one-line docstring; name every catchable
  exception inline (`Raises: X.`).
- Run `.venv/bin/python tools/ci_gates/comment_lint.py` before each commit: it must print GREEN. If it says a
  measure fell ("ratchet the floor down"), report the new numbers; if it REDs because a measure rose, shorten
  your comments/docstrings until it is green.
- R8 (gate 15, `tools/ci_gates/r8_header_gate.py`): a .py/.rs file over 300 lines needs a justification header
  that states a reason and never a line count; a file at or under 300 lines carries none. Re-run it on touched files.
- Gate 16: every text-mode open/read_text/write_text passes `encoding=`.
- Python: type hints; imports at top; catch specific exceptions.
- Rust: no unwrap()/expect() on production paths (a named error type that propagates); run `rustfmt --edition 2021`
  on the files you touched ONLY (never `cargo fmt` — it sweeps the crate); `cargo clippy -p <crate> --all-targets
  --locked -- -D clippy::all` green for crates you touched.
- If you add a test with a skip/skipif/slow/integration/importorskip mark, declare it in
  tools/ci_gates/tier_declaration.txt (gate 3c's tier census reds otherwise) and say so.
- A contract doc under docs/contracts/ that you change: its version bump, the doc and its tests move in ONE commit.
- A monitor/gate input you add needs a LIVE producer and a producer test (LAW-07); see
  src/mantis/monitor/producer_manifest.yaml and its test for the existing rows.

## Commits
- One commit per defect (or per tight class if two defects share a mechanism), on your branch.
- Message: ONE line, `type(scope): what changed and why it matters`. NO body, NO trailers, NO Co-Authored-By.
- Before each commit: the touched modules' tests green, `comment_lint.py` green, `r8_header_gate.py` green,
  `.venv/bin/python -m pytest --collect-only -q -m '' -p no:cacheprovider | tail -1` (note the count).

## Your final report (return it as your last message, compact)
Per defect: ID | commit sha | files | planted break (what; command; the red line) | witness now green (command) |
collected count after | comment_lint measures if changed | notes (refuted-at-contact with evidence / HALT with reason).
Then: anything you deliberately left for a later wave, and any shared file you edited (tier_declaration.txt etc.).
