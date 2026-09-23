# S-A-RUST-3 — slice R3 (mantis-core, mantis-encoding, mantis-graph, mantis-bridge)
scope: crates/mantis-core/**, crates/mantis-encoding/**, crates/mantis-graph/**, crates/mantis-bridge/** (incl. python/mantis/_engine.pyi); 47 files at 69e1532.
method: `git grep -n -w` per symbol over the whole tree; runtime `dir(mantis._engine.*)` from the built `.so` (built 2026-09-23 from HEAD) diffed against an AST parse of the wheel stub; `cargo check --workspace --all-targets --locked` (shared target dir); `cargo check -p mantis-graph --target wasm32-unknown-unknown --locked` (green); a DELETION PROBE in a `git archive HEAD` copy under scratchpad/r3probe (findings 01-07 + 12 applied together, then `cargo check --workspace --all-targets --offline` → Finished, no new error; the one new warning is `unused import: hex_distance` in core moves.rs, a one-token edit). Δlines are `git diff --no-index --numstat` of the probe against HEAD, or `sed -n`/`awk` spans of the subject, or `wc -l`.

## Summary
- 18 findings. By class: DEAD 11, TEST 1, SIMPLIFY 1, DOC 2, DUP 3. By lane: A 5 (04, 07, 12, 14, 15), B 6 (03, 05, 06, 13, 16, 18), C 7 (01, 02, 08, 09, 10, 11, 17).
- Top 3 by Δ: S-A-RUST-3-16 (-665, one of the two `_engine.pyi` twins plus the twin test; lane B), S-A-RUST-3-01 (-338, `Board.get_threats` and the viewer scanner it drags; lane C by file), S-A-RUST-3-07 (-127, `Board::get_clusters` and friends in core; lane A).
- Of the CALLERS.md §9 candidates, all 10 are confirmed dead from Python: mcts_omitted_prior_stats and take_mcts_omitted_prior_stats (03), Board.terminal_value_to_move, Board.threat_moves and Board.get_threats (01, 02), RegistrySpec.builder_impl_required, MCTSTree.last_search_stats and MCTSTree.run_simulations_cpu_only (04), MCTSTree.root_raw_value (05) and SelfPlayRunner.positions_dropped (06). Board.set_legal_move_radius is also dead from Python (02). The stub drift is confirmed (15 + DEFECTS).

### Common sweep (SW), which every DEAD row below reports per channel
- SW-a AST/attr (Python): `git grep -n -w <X> -- src tools tests` (this also catches `obj.X` attribute access and `from … import X`).
- SW-b whole tree: `git grep -n -w <X>` with no pathspec. It covers docs/, docs/governance (STATE procedures), Makefile, *.sh, *.toml, *.yaml, .github, crates/, and the string-embedded `python -c` code in tests/ and gates.
- SW-c dynamic: `git grep -n -E "getattr\((_engine|engine|e|board|tree|runner|spec|self\._runner)\b" -- '*.py'` and `git grep -n -E "dir\((spec|board|tree|runner|_engine)\)|getmembers|vars\((spec|board)\)" -- src tools tests`. Every hit is a fixed literal: the 13 runner counters in pool_hooks.py, resolvers.py::_SCATTERED_KEYS_TO_FIELD, test_inv22_spec_parity.py::_REQUIRED_FIELDS/_REQUIRED_DERIVED, test_surface.py lists and test_runner_derived_means.py means. None of them name a candidate, and no code enumerates an engine object's attributes.
- SW-d config: `.venv/bin/python -c "from mantis.config.schema import RunConfig, leaf_paths; …"` (160 leaves) and a grep over test_every_key_has_consumer{,_p2}.py, producer_manifest.yaml, armed_aborts.py, docs/contracts/, bench_floors.toml and tier_declaration.txt. Zero hits for every candidate. The one exception is `root_raw_value` in run_config_schema.md v32, which names the Rust tree method on the runner path (see 05).
- SW-e entry points: there are none (CALLERS §1). A pyo3 method is never an entry point.
- SW-f stub parsers: tests/config/test_no_bridge_default_shadows_a_config_key.py reads both stubs for signature DEFAULTS, and none of the candidates carry one. tests/bridge/test_engine_stub_twins_agree.py needs both twins edited identically. tests/train/test_cluster_stat_wiring.py and tests/selfplay/test_drain_row_shape_parity.py read unrelated stub members.

## Findings

### S-A-RUST-3-01 | DEAD | C
subject: crates/mantis-bridge/src/board.rs::PyBoard::get_threats + board.rs::threats (inlined "Threat-viewer scanner" mod + its 4 unit tests) + test get_threats_surfaces_open_line; stub lines in both _engine.pyi
claim: `Board.get_threats` has no caller in src/tools/tests. The file's own comment calls the scanner "Viewer only", but no viewer calls it.
evidence: `git grep -n -w get_threats -- ':!tests/fixtures'` -> only board.rs, both .pyi and docs/design/analyzer_design.md (prose listing it as part of the "tactical oracle"); `git grep -n -i threat -- tools src/mantis` -> no get_threats/threat use in tools/analyzer or tools/viewer
callers:
- AST: SW-a -> 0 .py hits (only the two .pyi)
- entry points: SW-e -> n/a
- python -m/Makefile/sh/docs/STATE: SW-b -> analyzer_design.md prose only (design doc, not a procedure); 0 in docs/governance
- subprocess strings: SW-b -> 0
- importlib/getattr/registry: SW-c -> 0
- conftest: SW-b covers tests/**/conftest.py -> 0
- pyo3 export: runtime `dir(_engine.Board)` lists it; Python refs 0
- config keys: SW-d -> 0
- gate tool paths: SW-b over tools/ci_gates -> 0; SW-f -> no default, twins edited together
- STATE/box procedures: SW-b -> 0 (the analyzer procedure runs tools/position_analyzer.py, which never calls it)
- Rust: `git grep -w get_threats -- crates` -> board.rs only (the Rust `mantis_core` has no get_threats; the scanner was inlined here)
Δlines: -338 (probe numstat hunks in board.rs: pymethod -16, `mod threats` -297, test -23; stubs -1 ×2)
witness: cargo gate 2 (the probe compiled clean); tests/bridge/test_engine_stub_twins_agree.py
depends: 16 (a stub collapse halves the stub edit)
note: lane C only because the PZ glob lists board.rs whole (arena legality `is_legal`, which this does not touch). docs/design/analyzer_design.md then needs a one-line repair (R311(c)). board.rs stays over 300 lines, so its R8 header stays.

### S-A-RUST-3-02 | DEAD | C
subject: crates/mantis-bridge/src/board.rs::PyBoard::{terminal_value_to_move, threat_moves, set_legal_move_radius} + test radius_guard_fires_when_bound; stub lines in both _engine.pyi
claim: three thin bridge wrappers over mantis-core methods have no Python caller. The core methods stay live (mantis-search tactics, golden_replay.rs).
evidence: `git grep -n -w <X> -- ':!tests/fixtures'` -> terminal_value_to_move: crates (core, search, golden_replay.rs) + pyi; threat_moves: crates + pyi; set_legal_move_radius: crates + pyi + tests/selfplay/test_radius_chain_removed.py, which runs `grep -rln set_legal_move_radius crates/` and still passes on core's own method
callers:
- AST: SW-a -> 0 .py call sites (the golden fixture JSON field `terminal_value_to_move` is read by Rust golden_replay.rs against the CORE method)
- entry points: SW-e -> n/a
- python -m/Makefile/sh/docs/STATE: SW-b -> analyzer_design.md lists threat_moves as prose; 0 in governance
- subprocess strings: SW-b -> only the crates/ grep in test_radius_chain_removed.py (still satisfied by core)
- importlib/getattr: SW-c -> 0
- conftest: -> 0
- pyo3 export: runtime lists all three; Python refs 0
- config keys: SW-d -> 0
- gate tool paths: -> 0; SW-f -> no defaults
- STATE procedures: -> 0
- Rust: the bridge methods have no Rust caller beyond their own test (board.rs::tests::radius_guard_fires_when_bound)
Δlines: -38 (probe hunks: -5, -5, -11, test -11; stubs -3 ×2)
witness: cargo gate 2; tests/selfplay/test_radius_chain_removed.py::test_set_legal_move_radius_positive_control_survives (its docstring names the bridge sibling and needs a word changed)
depends: 01 (same file)

### S-A-RUST-3-03 | DEAD | B
subject: crates/mantis-bridge/src/utils.rs::{mcts_omitted_prior_stats, take_mcts_omitted_prior_stats} + their register lines; stub lines in both _engine.pyi
claim: the process-wide omitted-prior counters are exported but never read from Python. Only the per-tree `MCTSTree::omitted_prior_stats` has readers, and those are Rust tests.
evidence: `git grep -n -E "omitted_prior_stats|omitted_prior"` -> bridge utils.rs, both .pyi, mantis-search (backup.rs globals; mod.rs per-tree; tests); docs: repo_design.md AMENDMENT item 5 "the process-wide totals stay … because the bridge publishes them as the run-wide aggregate", and nothing consumes that aggregate
callers:
- AST: SW-a -> 0 .py hits
- entry points: SW-e -> n/a
- python -m/Makefile/sh/docs/STATE: SW-b -> repo_design.md prose only; 0 in governance/STATE
- subprocess strings: -> 0
- importlib/getattr: SW-c -> 0 (no `getattr(_engine, <computed>)` outside test_surface.py's fixed lists, which omit these)
- conftest: -> 0
- pyo3 export: in runtime `dir(mantis._engine)`; Python refs 0
- config keys: SW-d -> 0
- gate tool paths: -> 0
- STATE procedures: -> 0 (no monitor or dashboard event carries it: `git grep -n omitted -- src tools` -> 0)
Δlines: -20 (probe numstat utils.rs +1 -17 → -16, the import line trimmed; stubs -2 ×2)
witness: tests/bridge/test_surface.py does not list them; cargo gate 2
depends: —
note: lane B because repo_design.md (the structural contract) states that the bridge publishes the aggregate, so removing it needs an amendment commit (R9). After this, mantis-search's `omitted_prior_stats` / `take_omitted_prior_stats` / `record_omitted_prior_global` statics lose their only non-test consumer (HANDOFF R1).

### S-A-RUST-3-04 | DEAD | A
subject: crates/mantis-bridge/src/mcts.rs::PyMCTSTree::{last_search_stats, run_simulations_cpu_only} + test cpu_only_simulations_visit_root; crates/mantis-bridge/src/encoding.rs::PyRegistrySpec::builder_impl_required; stub lines in both _engine.pyi
claim: three bridge members have no Python caller. The Rust-side methods stay live (search_drive.rs reads last_search_stats; mcts_bench.rs drives run_simulations_cpu_only; validate.rs checks the builder_impl_required field).
evidence: `git grep -n -w <X>` -> last_search_stats: mantis-search mod.rs/tests.rs, selfplay search_drive.rs, bridge, pyi; run_simulations_cpu_only: search mod.rs ("has no production caller"), mcts_bench.rs, bridge + its test, pyi; builder_impl_required: registry.toml, encoding parse/spec/validate + tests, bridge getter, pyi
callers:
- AST: SW-a -> 0 .py hits for all three
- entry points: SW-e -> n/a
- python -m/Makefile/sh/docs/STATE: SW-b -> 0 outside crates/pyi
- subprocess strings: -> 0
- importlib/getattr: SW-c -> builder_impl_required is absent from test_inv22 _REQUIRED_FIELDS/_REQUIRED_DERIVED and from resolvers._SCATTERED_KEYS_TO_FIELD
- conftest: -> 0
- pyo3 export: runtime lists all three; Python refs 0
- config keys: SW-d -> 0
- gate tool paths: -> 0; SW-f -> none has a default
- STATE procedures: -> 0
Δlines: -33 (probe hunks mcts.rs -6, -5, test -9; encoding.rs -5; stubs (1+1+2) ×2)
witness: cargo gate 2 (probe green); tests/bridge/test_engine_stub_twins_agree.py
depends: —

### S-A-RUST-3-05 | DEAD | B
subject: crates/mantis-bridge/src/mcts.rs::PyMCTSTree::root_raw_value; stub lines in both _engine.pyi
claim: the Python export has no caller. The R349(c) reconstruction its doc cites reads `root_raw` from game records, which the RUNNER fills from the Rust tree (search_drive.rs `Some(tree.root_raw_value())`), not from this pymethod.
evidence: `git grep -n -w root_raw_value` -> mantis-search (mod.rs, policy.rs, tests), selfplay search_drive.rs, bridge, pyi, docs/design/analyzer_design.md ("`root_raw_value()` is never read"), run_config_schema.md v32 (names the runner-path read); `git grep -n -w root_raw -- src` -> monitor/game_record.py only
callers:
- AST: SW-a -> 0 .py hits
- entry points: SW-e -> n/a
- python -m/Makefile/sh/docs/STATE: SW-b -> design/contract prose only, both about the Rust tree read
- subprocess strings: -> 0
- importlib/getattr: SW-c -> 0
- conftest: -> 0
- pyo3 export: runtime lists it; Python refs 0
- config keys: SW-d -> run_config_schema.md v32 cites the name for the Rust consumer chain `play_one_move` (gate 13 resolves `mantis.*` symbols and config keys, not Rust method names)
- gate tool paths: -> 0
- STATE procedures: -> 0
Δlines: -7 (probe hunk -5; stubs -1 ×2)
witness: cargo gate 2; gate 13 (tools/ci_gates/contract_doc_gate.py) should stay green because the v32 row names a Rust method; the reviewer should confirm
depends: 04 (same file)
note: lane B because the export's own doc ties it to a ruling (R349(c)). The reviewer should confirm that the reconstruction only reads records.

### S-A-RUST-3-06 | DEAD | B
subject: crates/mantis-bridge/src/runner.rs::PySelfPlayRunner::positions_dropped; stub lines in both _engine.pyi
claim: the drop-oldest counter getter has no Python reader. pool_hooks.py reads 13 runner counters by literal name, and this is not one of them.
evidence: `git grep -n -w positions_dropped -- src tools tests` -> 0 .py hits (pyi only); src/mantis/selfplay/pool_hooks.py getattr list (games_completed … worker_panics) omits it
callers:
- AST: SW-a -> 0
- entry points: SW-e -> n/a
- python -m/Makefile/sh/docs/STATE: SW-b -> crates + pyi only
- subprocess strings: -> 0
- importlib/getattr: SW-c -> pool_hooks.py literal list and test_runner_derived_means.py DERIVED_MEANS omit it
- conftest: -> 0
- pyo3 export: runtime lists it; Python refs 0
- config keys: SW-d -> 0
- gate tool paths / producer manifest: `git grep -n positions_dropped -- src/mantis/monitor tools` -> 0
- STATE procedures: -> 0 (no dashboard event reads it)
Δlines: -8 (probe hunk -4; stubs -2 ×2)
witness: cargo gate 2
depends: —
note: lane B because this is a design choice. The counter is the only witness of self-play data loss (drop-oldest), so the alternative to deleting it is surfacing it, which would be a feature and is out of scope here. If deleted, the selfplay atomic chain (runner/{mod,stats,finalize,game,spawn}.rs) loses its only reader (HANDOFF R2).

### S-A-RUST-3-07 | DEAD | A
subject: crates/mantis-core/src/board/moves.rs::Board::get_clusters + ClusterScratch + CLUSTER_SCRATCH_TLS (+ `use std::cell::RefCell`); crates/mantis-core/src/board/state/core.rs::Board::{set_cluster_threshold, set_cluster_window_size, get_cell}; board/mod.rs tests cluster_threshold_splits_at_distance_six + the trailing cluster assert in the radius test
claim: `get_clusters` exists only for `get_cluster_views()`, which was deleted with the dense path (R346(f)). Its only callers now are two in-crate tests. The two setters and `get_cell` (a duplicate of `Board::get`) have zero callers anywhere.
evidence: `git grep -n "get_cluster_views"` -> doc comments only (golden_replay.rs: "went with the dense path (R346(f))"); `git grep -n -E "get_clusters|set_cluster_(threshold|window_size)|get_cell\b" -- crates` -> definitions + mod.rs tests only; src/mantis/selfplay/instrumentation.py docstrings cite "the engine's `get_clusters`" (prose)
callers:
- AST: SW-a -> 0 (not exported over pyo3; the instrumentation.py hits are docstrings)
- entry points: SW-e -> n/a
- python -m/Makefile/sh/docs/STATE: SW-b -> repo_design.md names get_cluster_views as deleted; 0 in governance
- subprocess strings: -> 0
- importlib/getattr: -> n/a (Rust)
- conftest: -> 0
- pyo3 export: not exported (no bridge wrapper)
- config keys: SW-d -> 0
- gate tool paths / bench floors: `grep -c cluster tools/bench_floors.toml` -> 0; `git grep -w get_clusters -- crates/*/benches` -> 0
- STATE procedures: -> 0
- Rust extras: no include_str!, no [[bench]]/[[test]] users, no trait impl; cross-crate `git grep -l -w` -> none
Δlines: -127 (probe numstat: moves.rs -79, board/mod.rs -30, state/core.rs -18)
witness: cargo gate 2 (probe green); `cargo test -p mantis-core` (2 cluster tests leave the cargo test count; gate 3c counts pytest only)
depends: —
note: all three files stay over 300 lines, so their R8 headers stay. A follow-up is left to lane B: with get_clusters gone, `BoardGeometry.cluster_threshold` feeds nothing in core, but its removal would touch the PZ golden fixture's `geometry` object (board_replay_golden_v1.json).

### S-A-RUST-3-08 | DEAD | C
subject: crates/mantis-encoding/src/spec/mod.rs::RegistrySpec::{kept_slot_of, cur_stone_slot, opp_stone_slot, history_planes, turn_phase_planes, wire_signature, half}
claim: dense plane-slot accessors (v6 history and turn-phase planes, "the 4-plane live set") have zero callers in any crate or in Python. The grid path they served is deleted, and neither graph row keeps planes.
evidence: `git grep -n -E "\b(cur_stone_slot|opp_stone_slot|history_planes|turn_phase_planes|wire_signature|kept_slot_of)\b"` -> definitions only; `half` (RegistrySpec::half) -> no `.half()` call in crates (word hits are unrelated: the `half` crate, prose)
callers:
- AST: SW-a -> 0 (not exported over pyo3; PyRegistrySpec has no such getters)
- entry points / python -m / subprocess / conftest / STATE: SW-b -> 0 outside spec/mod.rs
- importlib/getattr: -> n/a
- pyo3 export: none
- config keys: SW-d -> 0
- gate tool paths: registry_gate.sh hashes registry.toml only -> unaffected
- Rust extras: cross-crate `git grep -w` -> 0; no trait impl
Δlines: -80 (spec/mod.rs spans: kept_slot_of…wire_signature `sed -n 216,288p` = 73; half 7)
witness: cargo gate 2; crates/mantis-encoding/tests/{registry_census,axis_pin}.rs
depends: —
note: lane C because mantis-encoding/** is PZ (the R321 encoding registry member), even though the registry data and sha do not change.

### S-A-RUST-3-09 | DEAD | C
subject: crates/mantis-core/src/ply.rs::Turn (+ Ply::turn and the Turn tests)
claim: the `Turn` newtype and `Ply::turn()` have no caller outside ply.rs's own tests. Only `Ply` is used.
evidence: `git grep -n -E "\.turn\(\)|\bTurn\b" -- crates ':!crates/mantis-core/src/ply.rs'` -> the lib.rs re-export + prose only; docs: CLAUDE.md map ("Ply/Turn vocabulary types"), repo_design.md (tree line + "`Ply::turn`'s mapping" is re-derived in PYTHON by the viewer)
callers:
- AST: SW-a -> 0 (not exported over pyo3)
- entry points / python -m / subprocess / conftest / STATE: SW-b -> CLAUDE.md + repo_design prose only
- importlib/getattr: -> n/a
- pyo3 export: none
- config keys: SW-d -> 0
- gate tool paths: -> 0
- Rust extras: cross-crate -> 0; `Turn` has derives only
Δlines: -71 (ply.rs 120 lines; Turn struct, Ply::turn, impl Turn, turn_mapping_derived_from_engine, the Turn asserts in ply_turn_round_trips_and_next and repr_transparent_size_pin, by `sed -n` spans)
witness: cargo gate 2
depends: —
note: lane C because CLAUDE.md's map and repo_design name it as vocabulary (LAW-03 measurement unit), so removing it edits the structural contract.

### S-A-RUST-3-10 | TEST | C
subject: crates/mantis-core/src/lib.rs::CRATE_NAME + tests::crate_name_pinned; crates/mantis-encoding/src/lib.rs::CRATE_NAME + tests::{crate_name_pinned, dag_deps_compile} + the F-45 narrative doc above it
claim: a WP0 scaffolding marker. Each test asserts that a const equals its own literal, and `dag_deps_compile` proves only that the deps link, which every other use already proves.
evidence: `git grep -n CRATE_NAME` -> the 5 crate definitions + self/downstream tautology asserts only (graph lib.rs, search lib.rs, selfplay lib.rs)
callers:
- AST/pyo3/config/STATE/gates: SW-a..SW-d -> 0 outside crates
- Rust: downstream asserts in mantis-search/src/lib.rs and mantis-selfplay/src/lib.rs (other slices), mantis-graph lib.rs (frozen)
Δlines: -31 in slice (core lib.rs const + tests = 9; encoding lib.rs lines 28-49 = 22)
witness: NONE (tautologies)
depends: —
note: lane C: encoding is PZ and graph lib.rs is frozen. The search/selfplay copies are HANDOFF R1/R2, and deleting only some copies breaks their asserts.

### S-A-RUST-3-11 | DEAD | C
subject: crates/mantis-graph/src/lib.rs::parallelism_hint + the `native` feature (Cargo.toml [features] default/native + the crate-level `cfg_attr(not(feature="native"), allow(dead_code))`)
claim: the `native` feature gates exactly one item, `parallelism_hint`, and nothing calls it.
evidence: `git grep -n -E "parallelism_hint|\"native\""` -> graph lib.rs def + cfg_attr, graph Cargo.toml only; wasm check green at HEAD (`cargo check -p mantis-graph --target wasm32-unknown-unknown --locked` → Finished)
callers:
- AST/pyo3/config/STATE/gates: SW-a..SW-d -> 0
- Rust: cross-crate `git grep -l -w parallelism_hint` -> Cargo.toml comment only; no crate enables `features = ["native"]`
Δlines: -14 (lib.rs fn 7 + cfg_attr 1; Cargo.toml `sed -n 16,21p` = 6)
witness: gate 4 (make check.wasm), gate 2
depends: —
note: lane C because graph lib.rs is FROZEN (byte-parity port, PZ-3). The Cargo.toml half would lose its only purpose.

### S-A-RUST-3-12 | DEAD | A
subject: crates/mantis-bridge/Cargo.toml `half = "2"` + numpy `features = ["half"]` (+ the comment naming "f16-bit ReplayBuffer push borrows")
claim: the `half` dependency existed for the deleted dense `ReplayBuffer` f16 push, and the bridge no longer uses f16.
evidence: `grep -n -E "\bhalf\b|f16" crates/mantis-bridge/src/*.rs` -> prose only; probe with both removed: `cargo check -p mantis-bridge --all-targets --offline` (forced recheck) → Finished; Cargo.lock diff = 2 lines (`"half",` under numpy and mantis-bridge)
callers:
- Rust: 0 uses of `half::` or `f16` in bridge src
- Python: numpy float16 arrays from the bridge -> `git grep -n float16 -- src/mantis` shows no bridge-returned f16 array (the bridge returns f32/i32/i64/i8/u8 arrays only, per inference.rs/hexg.rs/mcts.rs `into_pyarray` sites)
- other channels: n/a for a dependency
Δlines: -3 (probe numstat Cargo.toml +1 -2; Cargo.lock -2) plus a 2-line comment trim
witness: gate 2 (`--locked` catches a lockfile mismatch), gate 1 fresh sync
depends: —

### S-A-RUST-3-13 | SIMPLIFY | B
subject: crates/mantis-bridge/src/inference.rs::PyInferenceBatcher spec-less ("grid") residue: `feature_len` field, `pool_size` kwarg, ::graph_params else-arm, ::require_graph + its 6 call sites, the spec-less ctor arms; tests no_spec_no_lens_errors, explicit_lens_without_spec_construct, a_specless_batcher_rejects_graph_seam_methods
claim: every registered encoding is graph (`Representation` has one member), so `graph_params`' else-arm is unreachable. `feature_len` is never read (cargo: "field `feature_len` is never read"). `pool_size` is "accepted for signature compat but inert" and no caller passes it. A non-graph batcher is reachable only by passing explicit widths without a spec.
evidence: `cargo check` warning inference.rs `feature_len`; `git grep -n pool_size -- src tools tests crates` -> ctor + pyi only; `grep -c "self.require_graph()?;" inference.rs` -> 6; Python spec-less callers: src/mantis/selfplay/inference_server.py non-graph else-branch (`InferenceBatcher(feature_len=…, policy_len=…)`) and tests/bridge/test_pyclass_roundtrips.py
deliberate?: —
Δlines: ≥ -40 (require_graph fn 9 + 6 call lines; the 3 spec-less tests 8+5+9; graph_params else 3; feature_len field/param/init 3 + 2 test asserts; pool_size 2 rs + 1 pyi ×2), exact figure set by the ctor design
witness: cargo gate 2; tests/bridge/test_pyclass_roundtrips.py; tests/selfplay/test_inference_server.py
depends: HANDOFF (the Python grid else-branch in inference_server.py must go first)
note: lane B because it is a signature change on the pyo3 face (`encoding_spec` becomes required).

### S-A-RUST-3-14 | DOC | A
subject: stale in-slice docs: crates/mantis-bridge/pyproject.toml [tool.uv] comment; crates/mantis-bridge/src/lib.rs module + #[pymodule] docs; both _engine.pyi headers; crates/mantis-bridge/src/inference.rs R8 header + module doc; crates/mantis-bridge/src/mcts.rs v6 comments; crates/mantis-core/src/board/state/mod.rs module doc
claim: each of these disagrees with the tree:
- (a) The pyproject comment says `registry.toml` AND `manifests.toml` are include_str!-ed. manifests.toml is gone (AUDIT-1 F-36; `git ls-files | grep -c manifests.toml` → 0). It also cites `golden_tests.rs:267` and `mcts/mod.rs:271`, while the include_str! is at golden_tests.rs:75 and the mod decl at mod.rs:384 (`grep -n`).
- (b) lib.rs says "10 pyclasses + 3 free fns + 3 module fns". Runtime has 10 pyclasses, 2 exceptions, 10 free fns, 3 module fns and 3 constants (`dir(mantis._engine)` = 28). The utils register comment omits armed_sims.
- (c) The pyi headers say "11 pyclasses + 4 free fns + 3 module fns + WireAlreadyConsumed".
- (d) The inference.rs header says "22-method" InferenceBatcher; runtime has 16 public members. Its module doc also says "remap over the WP6 `DenseQueue`", and `git grep DenseQueue -- crates` → only this line.
- (e) mcts.rs says "MCTSTree path is v6-only today" and "n_actions for a v6 board".
- (f) state/mod.rs describes a `cluster` sub-file (`get_cluster_views`, `get_threat_anchors`) that is gone, and an "encoding crate" encoder that is also gone (R346(f)).
evidence: as listed; `ls crates/mantis-core/src/board/state/` → core.rs, mod.rs
Δlines: -5 (state/mod.rs cluster bullet + encoder note); the rest are in-place rewrites (0)
witness: NONE (prose). Gate 15 is unaffected (the inference.rs header states a reason; "22-method" is not a line count)
depends: 03, 04 (utils/lib docs change with them)

### S-A-RUST-3-15 | DOC | A
subject: crates/mantis-bridge/python/mantis/_engine.pyi (and its twin src/mantis/_engine.pyi) `MY_STONE_PLANE, OPP_STONE_PLANE, MOVES_REMAINING_PLANE, PLY_PARITY_PLANE` + the owner comment naming `mantis_encoding::encode::`
claim: the stubs declare four module constants that the extension does not export, retired with the dense path. The comment names a module, `mantis_encoding::encode`, that does not exist.
evidence: AST-parse of the wheel stub vs `dir(mantis._engine)` → "pyi-only top: MOVES_REMAINING_PLANE, MY_STONE_PLANE, OPP_STONE_PLANE, PLY_PARITY_PLANE"; `git grep -n -E "MY_STONE_PLANE|OPP_STONE_PLANE|MOVES_REMAINING_PLANE|PLY_PARITY_PLANE"` → the two stubs, tools/hardcode_scan.py (name table), tests/encoding/test_geometry_crosses_the_ffi.py (docstring). No code reads `_engine.<X>`, which would raise at runtime.
Δlines: -8 (4 decl lines ×2 stubs; the comment lines are rewritten in place)
witness: tests/bridge/test_engine_stub_twins_agree.py (edit both); pyright gate 14 (a stub removal can only remove false positives)
depends: 16
note: the opposite drift (4 runtime members missing from the stubs) is a DEFECT, listed below, because fixing it adds lines.

### S-A-RUST-3-16 | DUP | B
subject: crates/mantis-bridge/python/mantis/_engine.pyi ⇄ src/mantis/_engine.pyi (+ tests/bridge/test_engine_stub_twins_agree.py)
claim: two byte-identical 593/594-line transcriptions of one FFI surface, held together by a 3-test twin guard. Every stub edit is paid twice (findings 01-06 and 15 each touch both).
evidence: `diff <(sed '1,/^"""$/d' src/mantis/_engine.pyi) <(sed '1,/^"""$/d' crates/mantis-bridge/python/mantis/_engine.pyi)` → empty; `wc -l` 594 / 593 / twin test 71
deliberate?: yes, today. The src copy exists because "a regular package shadows namespace portions", so pyright (include src+tools) never sees the wheel copy. The wheel copy ships typing to wheel installs and keeps `python/mantis/` tracked so maturin's `python-source` exists on a fresh clone. Collapsing needs a MEASURED pyright resolution (e.g. `stubPath`/`extraPaths` onto crates/mantis-bridge/python, or a non-.pyi placeholder in python/mantis/), plus a matching edit to tests/train/test_cluster_stat_wiring.py and tests/config/test_no_bridge_default_shadows_a_config_key.py, which read both.
Δlines: -665 (drop src/mantis/_engine.pyi 594 + the twin test 71), or -593 on the other side plus a placeholder
witness: gate 14 (pyright), tests/selfplay/test_drain_row_shape_parity.py (reads the src stub path)
depends: —
note: test-floor move. The twin test holds 3 tests (`grep -c "^def test_"` → 3), so gate 3c's floor moves by 3 through tools/ci_gates/test_count_ratchet_down.txt. src/mantis/_engine.pyi is in another slice; this is raised here because the wheel copy is the Rust-owned half.

### S-A-RUST-3-17 | DUP | C
subject: crates/mantis-graph/tests/common/mod.rs hand-rolled SHA-256 (K table, sha256, sha256_hex) + crates/mantis-graph/tests/fixture_selftest.rs::sha256_fips_vectors
claim: a test-only FIPS-180-4 re-implementation duplicates the `sha2` crate, which is already locked in the workspace (mantis-core dev-dep, mantis-encoding dep). The crate's no-dependency rule covers only `[dependencies]`, and Cargo.toml says dev-deps "are never linked into the lib".
evidence: `awk` span `// SHA-256 (FIPS` … end of sha256_hex = 89 lines; `grep -n sha2 crates/*/Cargo.toml` → core dev-deps, encoding deps; the wasm check builds the lib only (dev-deps are not compiled for `cargo check -p mantis-graph --target wasm32`)
deliberate?: the header says "single shared, dep-free fixture-verification module", which describes the lib's rule and not a stated need for dev-dep freedom. The FIPS self-test exists only to validate the hand-roll.
Δlines: -104 (89 + FIPS test 16, +1 dev-dep line)
witness: crates/mantis-graph/tests/graph_parity.rs (manifest sha rows), fixture_selftest.rs mutation tests
depends: —
note: lane C because crates/mantis-graph/tests/** is PZ (goldens). The header's R8 justification would need rewording if the file stays over 300 (674 − 89 = 585, so it stays).

### S-A-RUST-3-18 | DUP | B
subject: crates/mantis-core/benches/smoke_bench.rs::board_with_n_stones ⇄ benches/board_bench.rs::board_with_n_stones; inline splitmix64 restatements in core tests (board/moves.rs tests, ply.rs tests, tests/empty_board_legal_pin.rs, tests/miri_cache.rs::splitmix; zobrist.rs has the const fn)
claim: an identical 8-line bench helper appears twice (smoke_bench's own doc says it is "restated"), and the 4-line splitmix64 step appears inline four times in core tests.
evidence: `diff` of the two helpers → identical bodies; `git grep -c 0x9e3779b97f4a7c15 -- crates/mantis-core` → moves.rs 1, zobrist.rs 2, ply.rs 1, empty_board_legal_pin.rs 1, miri_cache.rs 1
deliberate?: smoke_bench is the gate-5 1-second cut and may stay self-contained on purpose. A shared helper module via `#[path]` (the pattern build_bench.rs already uses) keeps bench group/fn names, so the 23 tools/bench_floors.toml rows are untouched.
Δlines: -6 (bench helper: -8 +2 `#[path]` lines); splitmix sharing is net ≈ 0 across src-unit and integration tests
witness: gate 5 (make bench); tests/tools/test_bench_floors.py
depends: —

## DEFECTS
- Stub drift, runtime→stub: `_engine.pyi` (both twins) OMITS RegistrySpec.{cluster_threshold, cluster_window_size} and SelfPlayRunner.{inference_failures_total, max_sims_per_search}, which exist at runtime and are read from Python (pool_hooks.py reads inference_failures_total, for example). pyright cannot see them. test_engine_stub_twins_agree.py compares the twins to each other and never to `dir(mantis._engine)`.
- crates/mantis-core/tests/miri_cache.rs::miri_nested_shared_borrows: `drop(g1)` drops a REFERENCE, which does nothing (rustc warning). The test believes it ends g1's borrow before reading g2, but that "proof" step is a no-op.
- pyo3 deprecation warnings: `#[pyclass] + Clone` on InferenceBatcher (inference.rs) and SelfPlayRunnerConfig (runner.rs) rely on the automatic `FromPyObject`, which is changing to opt-in. A pyo3 bump will break or change them (RegistrySpec already says `from_py_object`).
- crates/mantis-bridge/src/encoding.rs::PyRegistrySpec::from_static is flagged by cargo as never used in the lib build (test-only callers); it wants `#[cfg(test)]`.
- crates/mantis-bridge/pyproject.toml cache-key comment line cites are wrong (golden_tests.rs:267 and mod.rs:271, versus 75 and 384); see 14(a).
- tools/ci_gates/gate_01_fresh_sync.sh cites `crates/mantis-bridge/python/mantis/_engine.pyi:650` (the file has 593 lines) and `encoding.rs:213`. These are stale line pins in a gate comment (out of slice; see HANDOFF).

## PARKED
- none. The one perf-shaped item, `get_clusters`' O(n²) BFS, is moot if 07 lands.

## HANDOFF
- R1 (crates/mantis-search): after 03, `omitted_prior_stats`/`take_omitted_prior_stats`/`record_omitted_prior_global` (mcts/backup.rs process statics) have no non-test reader. The CRATE_NAME tautology asserts in search lib.rs belong to the 10 family. Non-snake-case test fn warnings in mcts/tests.rs.
- R2 (crates/mantis-selfplay): after 06, the `positions_dropped` atomic chain (runner/{mod,stats,finalize,game,spawn}.rs) has no reader. Cross-crate duplicated test helpers: `drive` ×7, `spawn_producer` ×6, `splitmix64` ×5, `spawn_graph_producer` ×4, `wide_board` ×4 (selfplay tests + search_kind_conformance.rs), found by a name census of crates/*/tests + benches. CRATE_NAME asserts in selfplay lib.rs.
- Python selfplay slice: src/mantis/selfplay/inference_server.py's non-graph else-branch (explicit `feature_len`/`policy_len` InferenceBatcher) is grid residue that blocks 13. instrumentation.py docstrings cite the engine's `get_clusters` (07). src/mantis/_engine.pyi mirrors every stub edit here (01-06, 15, 16).
- tools slice: tools/hardcode_scan.py name table lists the retired *_PLANE constants (15). tools/audit_bootstrap_corpus.py cites `state/core.rs:51-55`-style line numbers. tools/ci_gates/gate_01_fresh_sync.sh has stale line pins (PZ: tools/ci_gates/**).
- docs slice: docs/design/analyzer_design.md lists Board.get_threats/threat_moves as the analyzer's oracle (01, 02).

## Not covered
- The frozen mantis-graph/src/lib.rs was not audited item by item beyond pub-item cross-crate use (all pub items are used except BASE_DIM, which is internal-only, `parallelism_hint` (11) and `CRATE_NAME` (10)).
- Bridge members used ONLY by tests are not raised, because the brief's DEAD class needs no live caller and tests are callers. There are about 25 of them: e.g. Board.first_winning_move, MCTSTree.{get_policy, forced_root_child, search_sigma}, InferenceBatcher.{spawn_mock_graph_games, policy_len_py, representation_py, graph_max_in_flight}, SelfPlayRunner.{get_win_stats, set_model_version}, HexgBuffer.{game_id_at, get_buffer_stats} (list produced by the runtime-vs-src/tools census). A later pass may judge them.
- No `cargo test` was run (CPU budget). Evidence is `cargo check --all-targets`, the wasm check and the scratch deletion probe. No pyright run: a collapsed stub (16) is unmeasured.
- Line-level review of inference.rs, hexg.rs, graph_contract.rs, runner.rs internals beyond their pyo3 surface; board_bench.rs group names vs tools/bench_floors.toml (unchanged by any finding here).
- `threat_moves_ref` is `pub` but test-only (a visibility nit, Δ 0, not raised).

## Review
reviewer: fresh read-only agent (not the author); probes in throwaway worktrees, removed
One batch worktree under scratchpad/wt (04 + 07 + 12 + 15 applied together; each subject is independent). Recipe: `cargo check --workspace --all-targets --locked` → Finished, and the warning set is exactly the pre-existing one (no new unused-import or dead-code warning). `cargo test -p mantis-core --locked` → all green (60 + 7 integration binaries). `cargo test -p mantis-bridge --locked --lib` → 54 passed. `cargo clippy -p mantis-core -p mantis-bridge --all-targets --locked -- -D clippy::all` → Finished. `cargo check -p mantis-graph --target wasm32-unknown-unknown --locked` (the Makefile check.wasm line) → Finished. ENGINE REBUILT in the worktree (`cargo build -p mantis-bridge --features extension-module --locked`, .so copied into the worktree's bridge python dir), and the probe imported THAT .so, which lacked the deleted members. Python probe under `-S` + isolated PYTHONPATH: `import mantis` ok; `--collect-only -m ''` → "2289 tests collected, 167 errors", the same as HEAD. Nearest non-torch tests: tests/bridge + test_no_bridge_default_shadows_a_config_key + test_inv22_spec_parity + test_drain_row_shape_parity + test_hardcode_scanner_sees_the_graph_era → 108 passed. Torch-bound (collection error at HEAD too, so they cannot witness here): test_geometry_crosses_the_ffi, test_radius_chain_removed, test_cluster_stat_wiring. gate 15 → "172 over the cap, all justified, 0 stale". Probe `git diff --numstat` in total: +3 −189.

| ID | verdict | lane | Δlines (probe-measured for lane A) | note |
|---|---|---|---|---|
| 01 | CONFIRMED | C | -338 (scout; lane C, not probed) | board.rs is in the PZ list (arena `is_legal`) |
| 02 | CONFIRMED | C | -38 (scout) | its witness test_radius_chain_removed is torch-bound here |
| 03 | CONFIRMED | B | -20 (scout) | repo_design.md states that the bridge publishes the aggregate |
| 04 | AMENDED (Δ) | A | **-37** | probe green with the rebuilt engine; the scout's -33 left out the trailing blank lines |
| 05 | CONFIRMED | B | -7 (scout) | the Rust tree method stays, so the gate-13 row keeps its referent |
| 06 | CONFIRMED | B | -8 (scout) | pool_hooks.py has 0 reads |
| 07 | AMENDED (Δ, note) | A | **-137** | probe green; a PZ-file docstring cites get_clusters (see note) |
| 08 | CONFIRMED | C | -80 (scout) | mantis-encoding/** is PZ |
| 09 | AMENDED (lane) | B | -71 (scout) | no PZ row and no ruling names Turn/ply.rs; it is a contract amendment (repo_design + CLAUDE.md map) |
| 10 | CONFIRMED | C | -31 (scout) | a cross-crate family; encoding lib.rs asserts core's and graph's consts |
| 11 | CONFIRMED | C | -14 (scout) | graph lib.rs is frozen (PZ-3) |
| 12 | CONFIRMED | A | **-4** | the lock was hand-edited and `--locked` passes, so it is consistent |
| 13 | CONFIRMED | B | ≥-40 (scout) | `feature_len` never-read warning reproduced |
| 14 | CONFIRMED | A | -5 (prose, compile-neutral; not probed) | every quoted stale string re-found |
| 15 | CONFIRMED | A | **-8** | probe green (twin test edited on both sides) |
| 16 | AMENDED (evidence, ARCHITECT) | B | -665 is NOT reachable as stated | both collapse directions measured red (see note) |
| 17 | CONFIRMED | C | -104 (scout) | graph tests/** is PZ; sha2 is locked in core dev-deps and encoding deps |
| 18 | CONFIRMED | B | -6 (scout) | the helpers are byte-identical (8 lines) |

### Per-finding notes
S-A-RUST-3-01 — CONFIRMED: `grep -c -E "fn get_threats|mod threats" board.rs` → 4; `git grep -l -w get_threats -- '*.py' '*.sh' '*.toml' '*.yaml' Makefile docs/governance tests/config docs/contracts` → 0. board.rs is 751 lines and stays over 300 after the cut, so the R8 header stays. PZ.md lists crates/mantis-bridge/src/board.rs, so this is lane C.
S-A-RUST-3-02 — CONFIRMED: `git grep -n "fn set_legal_move_radius\|fn threat_moves\b\|fn terminal_value_to_move" -- crates` → the bridge wrappers plus the core defs (core.rs, moves.rs), so the core methods stay live. The only .py hit is test_radius_chain_removed.py (a crates/ grep satisfied by core). That witness is torch-bound here (a HEAD collection error), so it is unverifiable in this environment.
S-A-RUST-3-03 — CONFIRMED: `git grep -n -i -E "process-wide|run-wide" docs/design/repo_design.md` → the amendment text "the process-wide totals stay … because the bridge publishes them as the run-wide aggregate". A contract edit is needed, so lane B stands. The only registrations are the utils.rs `add_function` lines.
S-A-RUST-3-04 — AMENDED Δ -37 (mcts.rs -23 [last_search_stats doc+fn+blank 7, run_simulations_cpu_only 6, test 10], encoding.rs -6, stubs -4 ×2). Probe green with the REBUILT engine: `'last_search_stats' in dir(e.MCTSTree)` → False, and the same for run_simulations_cpu_only and builder_impl_required. test_inv22_spec_parity and test_surface pass, so no string lookup reaches them. `git grep -l -w <X> -- docs/governance tests/config docs/contracts '*.py'` → 0 for all three. The Rust test cpu_only_simulations_visit_root leaves the cargo count, which gate 3 does not count. The pytest count is unchanged at 2289.
S-A-RUST-3-05 — CONFIRMED lane B: mcts.rs's own doc names R349(c) ("exposed so R349(c)'s reconstruction can replay"). But `git grep -w root_raw_value -- docs/governance` → 0, so the ruling does not name the symbol (not C). `git grep -w root_raw -- src tools` → only monitor/game_record.py (the record writer), which fits the claim that reconstruction reads records. mantis-search keeps `root_raw_value`, so the run_config_schema.md v32 citation keeps a referent.
S-A-RUST-3-06 — CONFIRMED: `grep -c positions_dropped src/mantis/selfplay/pool_hooks.py` → 0. The lane-B reasoning (the only data-loss witness) stands.
S-A-RUST-3-07 — AMENDED Δ -137 (moves.rs -85 +1 [the import loses hex_distance, and nothing else in the file uses it: `grep -n hex_distance moves.rs` → only the get_clusters body], core.rs -21, board/mod.rs -32). The probe is green: `cargo test -p mantis-core` passes, clippy is clean, and no new warning appears. All three files stay over 300 (885 / 641 / 518) and gate 15 is green. The scout missed one thing: src/mantis/selfplay/instrumentation.py::_components and its winner-components docstring name "the engine's `get_clusters`" as their connectivity convention (`git grep -n get_clusters -- src`). instrumentation.py is a PZ member (draw-rate row). The Rust delete does not touch it and no test pins the pair (so it is not an oracle), but it leaves two docstrings citing a vanished symbol. Their repair is a lane-C one-liner. The Rust delete itself stays lane A.
S-A-RUST-3-08 — CONFIRMED: `git grep -c -w -E "kept_slot_of|cur_stone_slot|…|wire_signature"` → spec/mod.rs 8 (definitions only). mantis-encoding/** is PZ.
S-A-RUST-3-09 — AMENDED lane C→B: `grep -c -E "ply\.rs|LAW-03" PZ.md` → 0, and `git grep -w Turn -- docs/governance` → 0. The only citations are CLAUDE.md's map line and repo_design.md's tree line. That makes this a structural-contract amendment, which is lane B by the same test the scout applied to 03. No ruling and no PZ member is involved.
S-A-RUST-3-10 — CONFIRMED: `git grep -n CRATE_NAME -- crates | grep -v "pub const"` → the encoding lib.rs asserts on `mantis_core::`/`mantis_graph::CRATE_NAME`, search and selfplay assert on upstream consts, and graph lib.rs:assert is frozen. The family cannot be cut piecemeal.
S-A-RUST-3-11 — CONFIRMED: the graph Cargo.toml `[features]` comment says native-only items are target-gated and that wasm passes plain. lib.rs is frozen (PZ-3).
S-A-RUST-3-12 — CONFIRMED Δ -4 (Cargo.toml +2 −4 including the comment trim; Cargo.lock −2: the `"half",` lines under mantis-bridge and numpy). The `half` package itself stays in the lock (mantis-selfplay keeps it). `--locked` check, tests and the rebuilt extension are all green, so no bridge symbol needs numpy's `half` feature.
S-A-RUST-3-13 — CONFIRMED B: my own `cargo check` shows "field `feature_len` is never read" (inference.rs).
S-A-RUST-3-14 — CONFIRMED: every quoted stale string was re-found with `grep -n`: lib.rs "10 pyclasses … 3 free fns"; inference.rs "22-method" and "WP6 `DenseQueue`"; state/mod.rs "`cluster` — … `get_cluster_views`" and "tensor-encoder sub-file lives in the encoding crate"; mcts.rs "v6-only today" and "for a v6 board"; bridge pyproject.toml "`manifests.toml`". A count of methods or pyclasses is not a line count, so gate 15 is unaffected.
S-A-RUST-3-15 — CONFIRMED Δ -8: the probe removed the four decls from both stubs. test_engine_stub_twins_agree and test_hardcode_scanner_sees_the_graph_era pass, and `git grep -l -w PLY_PARITY_PLANE -- '*.py'` → only tools/hardcode_scan.py (a name table, not an `_engine.` read). pyright was not run gate-wide: removing a stub symbol that no .py references cannot add a diagnostic.
S-A-RUST-3-16 — AMENDED (evidence; lane B stands; ARCHITECT). I measured both directions.
(a) Delete src/mantis/_engine.pyi, keep the wheel copy: pyright 1.1.408 with the interpreter's sys.path ISOLATED to the worktree's src + bridge python dir + site-packages (via an `-S` shim) gives `Import "mantis._engine" could not be resolved (reportMissingImports)` and every type Unknown. With the src copy restored, Board, str and MCTSTree resolve. So the src header's "a regular package shadows namespace portions" holds. CAUTION for whoever lands this: an UNisolated run looks green, because the venv's editable .pth resolves the stub through the MAIN checkout's src copy.
(b) Delete the wheel copy, keep src: the wheel copy is the ONLY tracked file under crates/mantis-bridge/python (`git ls-files`). A fresh-clone-equivalent `uvx maturin build --profile dev` → "python-source is set to `python` but the directory does not exist".
So either direction adds structure: a pyright resolution setting (unmeasured) or a python-source placeholder. The -665/-593 figures are gross, not net. Other paths that move with it: .gitignore, bridge pyproject cache-key `python/**/*.pyi`, gate_01_fresh_sync.sh comment, the twin test (gate 3 floor −3), test_no_bridge_default_shadows_a_config_key and test_drain_row_shape_parity (they read the stub paths).
S-A-RUST-3-17 — CONFIRMED: `grep -n sha2 crates/*/Cargo.toml` → core dev-deps and encoding deps. graph tests/** is PZ (goldens).
S-A-RUST-3-18 — CONFIRMED: an awk extract of both `board_with_n_stones` bodies, then `diff` → identical, 8 lines. The `#[path = "../tests/common/mod.rs"]` precedent exists in graph build_bench.rs and selfplay queue_fuse_bench.rs.

### Missed by the scout (optional, max 5)
NEW-1 | DEFECT | — : root pyproject `[tool.hatch.build.targets.wheel] packages = ["src/mantis"]` ships `mantis/_engine.pyi` from the root wheel, AND the maturin engine wheel ships `mantis/_engine.pyi` from python-source. A non-editable install therefore has two distributions owning one installed path. This is latent today (editable only), and it bears on 16's direction.
NEW-2 | DOC | C : src/mantis/selfplay/instrumentation.py::_components (and the winner-components docstring) cite the engine's `get_clusters` as the connectivity convention. They go stale with 07. The file is a PZ member, so this is lane C; Δ 0 (in-place rewording).

### Tally: raised 18 | confirmed 14 | amended 4 (04, 07, 09, 16) | refuted 0 | pending 0 | architect 0 (one ARCHITECT question attached to 16)
Lane-A Δ probe-measured: 04 -37, 07 -137, 12 -4, 15 -8 = **-186** (probe numstat +3 −189); 14's -5 is prose and was not probed.
ARCHITECT (16): which side keeps the one `_engine.pyi`? Keeping the wheel copy needs a pyright import-root setting; keeping the src copy needs a maturin python-source placeholder or layout change.
