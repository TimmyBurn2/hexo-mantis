# Contract: registry

- version: v1
- owner: crate mantis-encoding + mantis.encoding
- status: v1 — ported in WP3; Filesystem/torch/npz audit sections + Python CLI + sha handshake ported in WP7
- WP7: `_engine.{all_specs,registry_sha,registry_sha_hex}` bindings + the import-time registry-sha handshake landed; FS/torch/npz audit sections + the `python -m mantis.encoding audit` CLI port over the WP3 Rust backend (§1/§6). No schema/version change (v1 semantics unchanged).
- R368 (SLIM-FIX): `_probes.py` folds into `resolvers.py` (the graph marker key) and the anchor-path table is deleted. No schema/version change.
- DENSE-1 L1 (2026-10-18): schema v6 — `hexcanvas` joins the representation enum with its canvas keys, and `empty_edges` is scoped to graph rows. The contract's semantics are otherwise unchanged (v1).

## Summary
TOML schema + validator invariants + audit backend exit codes (0/1/2).

## Schema
`crates/mantis-encoding/src/registry.toml` is the single source of truth for encoding and
shape. Each `[encodings.<name>]` table carries `representation` (REQUIRED, no default),
spelled `"graph"` or `"hexcanvas"` (a closed enum; `"grid"` is REFUSED BY NAME since the grid
path's deletion, R346(f), and `"dense"` is refused by name too, the hex-canvas kind being
`"hexcanvas"`, `spec/mod.rs`), plus its shape fields and `n_chain_planes` (= 6 on every row, a
relic with no consumer on either representation). Unknown TOML keys are a parse error; missing
required keys are a parse error; the validator collects ALL errors before reporting. `empty_edges`
(REQUIRED on a graph row since schema v5, `"kept"` or `"pruned"`, no default; REFUSED on a
`hexcanvas` row since v6) says whether the builder emits the axis edges joining two empty cells;
under `"pruned"` the builder's producer verify refuses any such edge (`PrunedEmptyEdge`).

A `hexcanvas` row (schema v6, contract #12) carries the reply frame exactly as the graph rows do
(`board_size` 19, `trunk_size` 19, `policy_logit_count` 362 with the pass slot, `sym_table_id`
`size_19`: `LegalSetPolicy`'s window and overflow map, the one search-facing reply), its
`legal_move_radius` and `win_length` (the game's rule, as on a graph row), `n_planes` 6 with a
`plane_layout` equal to the builder's `CANVAS_PLANES`, `kept_plane_indices` = 0..n_planes and
`n_source_planes` = n_planes (no slice), and five canvas keys: `canvas_side_step` (> 0),
`canvas_min_side` (a multiple of the step that a lone stone's fence, 2·radius + 1, fits),
`canvas_max_side` (S_max: a multiple of the step, at least the minimum; far mode past it),
`far_halo` (≥ 1, below the radius) and `far_max_side` (S_far: a multiple of the step above
S_max; the bound of the root rule the serving path builds). It is a single window with one canvas a
leaf (`is_multi_window` false, `k_max` 1, `trunk_size` 19). The canvas keys are refused on a graph
row, and the graph-only keys (`node_feat_dim`, `edge_feat_dim`, `graph_radius`, `win_axes`,
`contract_version`, `builder_impl_required`, `empty_edges`) on a hexcanvas row. Every branch of the
canvas arm has its refusal row in the census (`every_canvas_validator_branch_refuses_its_row_by_name`).

The registered set is pruned to its live consumers — every entry is named by ≥ 1 config or ≥ 1
anchor artifact — with two stated exceptions, each a landed capability no config names until a
run adopts it: `gnn_axis_r8_pruned` (ENCODE-1, R388(f); not a lever, R389(a)) and `hexcanvas_r8`
(DENSE-1 L1, R394(c); its consumers are built in DENSE-1's later legs). The encoding-consumer check is the
registry census test (`crates/mantis-encoding/tests/registry_census.rs`, which pins the registered
set by name, so a row lands or leaves only with its census edit); gate 11 is unchanged and refuses
silent encoding fallbacks (R389(a), LAW-08's annotation). Repaired 2026-10-08.

## Who asserts what where
- `registry/parse.rs` — per-field parse + unknown-key reject + `representation` required
  (absent representation is an error, never a default).
- `spec/validate.rs` — cross-field, representation-gated invariants; `n_chain_planes ≥ 1`;
  graph fields checked against `mantis-graph`'s graph constants, the canvas plane layout against
  its `CANVAS_PLANES`.
- `registry/mod.rs` — load-time validation + `registry_sha()` (sha256 of the embedded
  TOML; in dev/test `mantis.encoding` hashes the on-disk TOML and hard-errors on mismatch,
  so a stale extension cannot serve a stale registry).
- Python side (`mantis.encoding`): `audit.py` + `audit_sections.py` are THE encoding audit —
  registry census (§1), the filesystem/torch/npz sections, and the cross-table sha
  reconciliation (§6); severity maps to exit codes 0 = info, 1 = warn, 2 = error. `resolvers.py`
  is THE one encoding-resolution authority (agreement-or-raise on dual-shape configs, R104),
  the unified state-dict detector AND the one artifact-pin authority (corpus / held-out); `__main__.py` is the `python -m mantis.encoding audit` CLI.

**TWO RUST SECOND-AUTHORITIES WERE DELETED HERE (AUDIT-1 F-36 / F-45, R332).** `manifests.rs`
+ `manifests.toml` carried corpus/anchor/held-out pins beside the resolver dicts that actually
enforce them, and `audit.rs` was a "reference" port of the census and cross-table beside the
live CLI. Neither was reached by anything but its own test — no bridge export wrapped either —
and the manifests pair had DRIFTED from the live dicts. One authority per invariant, and it is
the reachable one. `tests/encoding/test_one_artifact_pin_authority.py` keeps the class closed.

## Pinning tests
`crates/mantis-encoding/tests/{registry_census.rs, axis_pin.rs}` (Rust) and
`tests/encoding/test_one_artifact_pin_authority.py` + `tests/encoding/test_one_config_parser.py`
(Python).

## Audit trail
WPCLEAN Phase RES (2026-07-29): the carried "Rust-crate audit, not a docs pass" card was
executed — every file, pinning test, exit-code mapping, sha mechanism and the pruned-set
claim above was re-verified against HEAD and found TRUE; the drift found was staleness only
(a WP7 future-tense line, the missing Python-side rows above, the unnamed audit-core file),
fixed in this revision. Grounds: mantis-migration wp/WPCLEAN/GROUND_RES.md §7.
