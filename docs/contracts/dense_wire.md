# Contract: dense wire

- version: v1
- owner: crate mantis-encoding / mantis-bridge
- status: v1 — kernels ported in WP3 (bridge marshaling lands in WP7)

## Summary
Fixed `[n, feature_len]` f32 batches; strides SPEC-DERIVED; shape-checked both sides.

## Strides
The state / chain / aux / policy stride formulae are derived from `RegistrySpec` fields
(never hardcoded); the byte layout is identical to the pre-migration kernels and is
parity-gated (floats byte-exact, no tolerance).

## Who asserts what where
- `encode/{state,chain}.rs` — the dense encode kernels (free functions over the core
  `Board`); byte layout per the spec-derived strides.
- `spec/mod.rs` — the stride accessors derived from `RegistrySpec`.

## Pinning tests
`crates/mantis-encoding/tests/{registry_census.rs (spec-derived strides), encode_parity.rs
(byte layout, 29 goldens)}`.
