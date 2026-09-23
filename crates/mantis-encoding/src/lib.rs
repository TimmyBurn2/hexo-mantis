//! mantis-encoding: registry.toml + spec + validators.
//!
//! ARTIFACT PINS ARE NOT HERE (AUDIT-1 F-36). A `manifests.rs` + `manifests.toml` pair used
//! to carry corpus/anchor/held-out pins beside `mantis.encoding.resolvers`' dicts. It was
//! reached by nothing but its own test — no bridge export wrapped it — while the resolver
//! dicts were the live authority, and the two had DRIFTED (the TOML declared anchor paths for
//! the two graph rows that the live dict does not have, and the hold-out check that can
//! actually fire is the Python one). Two implementations of one invariant, one of them
//! unreachable, is zero. `mantis.encoding.resolvers` is THE authority.
//!
//! DAG: depends on mantis-core (Board, geometry, Ply) + mantis-graph (builder
//! schema constants for the graph-encoding validator) ONLY. No pyo3 (the bridge
//! crate owns all FFI); no reach-through into search/selfplay/replay.
//!
//! The `representation` identity key is REQUIRED (absent → error, never a default — LAW-11)
//! and `"grid"` is refused by name; the registered set is the graph entries in
//! `registry.toml`; `registry_sha()` is the runtime handshake primitive the bridge
//! re-exports. The dense encode kernels went with the grid path (R346(f)).

pub mod registry;
pub mod spec;

pub use registry::{
    all_specs, lookup, lookup_or_panic, parse_encoding_toml, registry_sha, registry_sha_hex,
};
pub use spec::{PolicyPool, RegistrySpec, Representation, ValuePool};
