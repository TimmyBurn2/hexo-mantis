//! mantis-encoding: registry.toml + spec + validators.
//!
//! Artifact pins are NOT here: `mantis.encoding.resolvers` is THE authority for corpus, anchor
//! and held-out pins, and a second copy in this crate would be an unreachable twin.
//!
//! DAG: depends on mantis-core (Board, geometry, Ply) + mantis-graph (builder
//! schema constants for the graph-encoding validator) ONLY. No pyo3 (the bridge
//! crate owns all FFI); no reach-through into search/selfplay/replay.
//!
//! The `representation` identity key is REQUIRED (absent → error, never a default)
//! and `"grid"` is refused by name; the registered set is the graph entries in
//! `registry.toml`; `registry_sha()` is the runtime handshake primitive the bridge
//! re-exports.

pub mod registry;
pub mod spec;

pub use registry::{
    all_specs, lookup, lookup_or_panic, parse_encoding_toml, registry_sha, registry_sha_hex,
};
pub use spec::{PolicyPool, RegistrySpec, Representation, ValuePool};
