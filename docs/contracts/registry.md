# Contract: registry

- version: v1
- owner: crate mantis-encoding + mantis.encoding
- status: v1 — ported in WP3; Filesystem/torch/npz audit sections + Python CLI + sha handshake ported in WP7
- WP7: `_engine.{all_specs,registry_sha,registry_sha_hex}` bindings + the import-time registry-sha handshake landed; FS/torch/npz audit sections + the `python -m mantis.encoding audit` CLI port over the WP3 Rust backend (§1/§6). No schema/version change (v1 semantics unchanged).

## Summary
TOML schema + validator invariants + audit backend exit codes (0/1/2).

## Schema
`crates/mantis-encoding/src/registry.toml` is the single source of truth for encoding and
shape. Each `[encodings.<name>]` table carries `representation` (REQUIRED, no default),
spelled `"grid" | "graph"`, plus its shape fields and `n_chain_planes` (≥ 1). Unknown TOML
keys are a parse error; missing required keys are a parse error; the validator collects ALL
errors before reporting. The registered set is pruned to its live consumers — every entry
is named by ≥ 1 config or ≥ 1 anchor artifact.

## Who asserts what where
- `registry/parse.rs` — per-field parse + unknown-key reject + `representation` required
  (absent representation is an error, never a default).
- `spec/validate.rs` — cross-field, grid/graph-gated invariants; `n_chain_planes ≥ 1`;
  graph fields checked against `mantis-graph` constants.
- `registry/mod.rs` — load-time validation + `registry_sha()` (sha256 of the embedded
  TOML; in dev/test `mantis.encoding` hashes the on-disk TOML and hard-errors on mismatch,
  so a stale extension cannot serve a stale registry).
- `manifests.rs` — corpus/anchor pin loud-parse (same discipline; sibling `manifests.toml`,
  which keeps `registry.toml` pure encoding-shape).
- `crates/mantis-encoding/src/audit.rs` (the audit backend's pure-data core) — registry census (§1) + cross-table
  sha reconciliation (§6); severity maps to exit codes 0 = info, 1 = warn, 2 = error.
- Python side (`mantis.encoding`): `audit.py` + `audit_sections.py` carry the
  filesystem/torch/npz audit sections over the Rust core; `_probes.py` holds their probe
  helpers; `resolvers.py` is THE one encoding-resolution authority (agreement-or-raise on
  dual-shape configs, R104); `__main__.py` is the `python -m mantis.encoding audit` CLI.

## Pinning tests
`crates/mantis-encoding/tests/{registry_census.rs, manifests.rs, audit_backend.rs}`.

## Audit trail
WPCLEAN Phase RES (2026-07-29): the carried "Rust-crate audit, not a docs pass" card was
executed — every file, pinning test, exit-code mapping, sha mechanism and the pruned-set
claim above was re-verified against HEAD and found TRUE; the drift found was staleness only
(a WP7 future-tense line, the missing Python-side rows above, the unnamed audit-core file),
fixed in this revision. Grounds: mantis-migration wp/WPCLEAN/GROUND_RES.md §7.
