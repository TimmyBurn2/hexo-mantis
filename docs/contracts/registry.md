# Contract: registry

- version: v1
- owner: crate mantis-encoding + mantis.encoding
- status: v1 — ported in WP3

## Summary
TOML schema + validator invariants + audit backend exit codes (0/1/2).

## Schema
`crates/mantis-encoding/registry.toml` is the single source of truth for encoding and
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
- audit backend — registry census (§1) + cross-table sha reconciliation (§6); severity maps
  to exit codes 0 = info, 1 = warn, 2 = error. (Filesystem/torch/npz sections land in WP7.)

## Pinning tests
`crates/mantis-encoding/tests/{registry_census.rs, manifests.rs, audit_backend.rs}`.
