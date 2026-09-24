# W2 ADDENDUM — the Rust slimming wave (read after WAVE_BRIEF.md)

Slimming now (R368(d)/(e)/(k)), no longer defect repair: the brief's "planted break" rule applies only where you
touch a WITNESS (a deleted helper's callers must still be tested); a deletion of dead code is proven by the
crate's tests + clippy + (for anything pyo3-visible) the Python bridge tests, green before and after.
Slimming NEVER grows lines net (per commit and per wave).

## Rows and lanes
- Your rows are listed in your prompt with their census IDs. The ledger (docs/slim/LEDGER.md §6 table and §5 queue)
  gives each row's lane; the scout file (docs/slim/S-A-RUST-N.md, S-L-DUP.md) gives subject/claim/Δ/witness and the
  `## Review` section amends. docs/slim/REPROBE.md is irrelevant for Rust.
- LANE A/B rows: implement as the review amended them.
- LANE C rows: RE-LANE under R368(b) FIRST, and record the decision per row (→A / →B / still C) with its ground:
  * protection binds the INVARIANT: the 11 LAWS invariants' implementing symbols + pinning tests (00_MAP §3 PZ-1);
    for Rust these are served-sims (runner/search_drive.rs::run_mcts_search's PUCT clamp, seq_halving.rs::
    considered_visits_sequence, the `max_sims_per_search` counter, tests/served_sims_exact.rs), arena legality
    (bridge board.rs::is_legal), 1-in-1 collate (bridge graph_contract.rs verify_edge_geometry, mantis-graph
    lib.rs::verify_contract). Seam members (PZ-2: SearchKind, MAX_CHILDREN_PER_NODE, MAX_ARMED_SIMS*, the encoding
    registry's listed symbols, graph wire, goldens, search_kind_conformance.rs) and ruling-named code the same.
    Anything ELSE in the same file changes under the normal legs, with the file's pinning tests green before and after.
  * a ruling that named code protects it only while it still directs that code to exist (e.g. R153's
    characterize-first instrument: its measurement ruled long ago — if the invariant it measured keeps another live
    witness, the instrument is spent; name that other witness);
  * a real freeze is a manifest row, a test/gate that reds on the edit, or a ruling naming the file; a header calling
    itself frozen is a claim (correct it on contact). The completed-Q golden (golden_bits.txt), graph_parity.rs byte
    parity, golden_tests.rs's seeded RNG const, the mctx/temperature parity goldens and the oracle-bank feature tests
    ARE real freezes on NUMERICS/BYTES: dead-code removal around them is fine, any output change is not;
  * R368(d) ORDERS grid/dense residue gone (arms, flags, stubs, pins) and duplicated helpers made one; R368(k):
    RUST-2-12 and RUST-2-13 DELETE; RUST-3-16: ONE stub at the path the built wheel installs, checked against the
    runtime surface; TESTS-5-27: a test-only pyo3 export stays only while its Python test is the sole coverage of
    that binding; RUST-1-13/-14 are CARDED (hot path) — do not implement.
  * R368(h) OUT: hot-path refactors in search, graph build and collate (inline decode dedup, loop restructuring) —
    dead-code REMOVAL there is in. A row that is a hot-path refactor stays C: say "C (R368(h) hot path)".
- The UNWRAP/EXPECT CLASS (R368(g) last bullet) for your crates: classify every `unwrap()`/`expect(` outside
  `#[cfg(test)]`/tests/benches as (i) production path, (ii) startup invariant with an expect message naming the
  invariant (allowed by CLAUDE.md), (iii) hot-loop site. Fix (i) with a named error that propagates (behaviour for
  valid inputs unchanged; goldens green). Do NOT change (iii): list it for a card with its file:function and why it
  is hot. Report the classification table (counts + the (i) sites fixed + the (iii) list).

## run10 safety (R368(i)) — hard
- search, self-play and eval NUMERICS bit-identical: every golden/parity test green (mantis-search golden_tests,
  parity_tests, tests/{mctx_parity,temperature_parity_golden,search_kind_conformance}.rs; mantis-selfplay
  tests/{graph_child_parity,queue_fuse_pin,replay_hexg,target_export_parity,worker_output_pin,served_sims_exact}.rs;
  mantis-graph tests/graph_parity.rs; mantis-core tests/golden_replay.rs).
- no change to the hexg ring format, the checkpoint/stamp format, or anything the Python config schema reads.

## Checks before each commit / at the end
- `cargo test -p <crate> --locked` for the crates you touched: targeted test targets while iterating, the FULL crate
  suite ONCE at the end (release where the crate's tests are slow — look at how gate 2a runs: plain
  `cargo test --workspace --locked`; the long characterization tests exist, budget for them).
- `cargo clippy -p <crate> --all-targets --locked -- -D clippy::all`; `cargo build --workspace --all-targets --locked`
  0 warnings; `make check.wasm` if mantis-graph changed; rustfmt on touched files only.
- anything pyo3-visible (bridge exports, `_engine.pyi`): rebuild the extension (brief) and run tests/bridge and every
  Python test that greps the removed name (`git grep -n <name> -- src tools tests`), plus
  tests/bridge/test_engine_stub_matches_runtime.py (the stub must equal the runtime surface).
- a deleted Python test lowers the collected count: report it (the dispatcher moves the floor); a deleted test that
  is named in tools/ci_gates/tier_declaration.txt must have its row removed in the same commit.
- gate 10 (`python3 tools/ci_gates/check_tracked_refs.py`): a deleted file cited in a scanned doc (docs/governance/*.md
  except RULINGS.md, docs/contracts/*.md, README, CLAUDE.md, Makefile) reds; either reword the doc (non-register) or
  add the path to `DISSOLVED_PATHS` in tools/ci_gates/check_tracked_refs.py (the self-expiring whitelist) — R368(e)
  deletions are the grant for that list only.
- One commit per finding class (group rows of one class in one crate), one-line messages, no trailers.

## Report additions
- The re-lane table: ID | old lane | new lane | ground.
- The unwrap/expect classification table.
- Per commit: rows, Δlines (`git diff --numstat` net, must be ≤ 0 for slimming), tests run.
