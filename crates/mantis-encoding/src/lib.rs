//! mantis-encoding: registry.toml + spec + validators + dense encode kernels.
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
//! The `representation` identity key is REQUIRED (absent → error, never a
//! grid/dense default — LAW-11); the pruned registered set is the 4 entries in
//! `registry.toml`; `registry_sha()` is the runtime handshake primitive the
//! bridge re-exports.

pub mod encode;
pub mod registry;
pub mod spec;

pub use encode::{
    encode_chain_planes, encode_planes_to_buffer, encode_state_to_buffer,
    encode_state_to_buffer_channels, to_planes, to_planes_channels, MOVES_REMAINING_PLANE,
    MY_STONE_PLANE, OPP_STONE_PLANE, PLY_PARITY_PLANE,
};
pub use registry::{
    all_specs, lookup, lookup_or_panic, parse_encoding_toml, registry_sha, registry_sha_hex,
};
pub use spec::{PolicyPool, RegistrySpec, Representation, ValuePool};

/// THE ENCODING AUDIT IS NOT HERE EITHER (AUDIT-1 F-45). `audit.rs` was a Rust "reference"
/// port of the Python audit's §1 census and §6 cross-table, labelled as such, reached by
/// nothing but its own test and wrapped by no bridge export — a second implementation of the
/// invariant `mantis.encoding.audit` (the live CLI, the one gate 8 defers to) enforces, with
/// no cross-language parity check between them. The live one is kept.
///
/// Crate identity pin (WP0 DAG marker, retained).
pub const CRATE_NAME: &str = "mantis-encoding";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_pinned() {
        assert_eq!(super::CRATE_NAME, "mantis-encoding");
    }

    #[test]
    fn dag_deps_compile() {
        assert_eq!(mantis_core::CRATE_NAME, "mantis-core");
        assert_eq!(mantis_graph::CRATE_NAME, "mantis-graph");
    }
}
