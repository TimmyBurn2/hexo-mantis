//! mantis-encoding: registry.toml + spec + validators + dense encode kernels +
//! corpus/anchor manifests + the registry-shape audit backend.
//!
//! DAG: depends on mantis-core (Board, geometry, Ply) + mantis-graph (builder
//! schema constants for the graph-encoding validator) ONLY. No pyo3 (the bridge
//! crate owns all FFI); no reach-through into search/selfplay/replay.
//!
//! The `representation` identity key is REQUIRED (absent → error, never a
//! grid/dense default — LAW-11); the pruned registered set is the 4 entries in
//! `registry.toml`; `registry_sha()` is the runtime handshake primitive the
//! bridge re-exports.

pub mod audit;
pub mod encode;
pub mod manifests;
pub mod registry;
pub mod spec;

pub use encode::{
    encode_chain_planes, encode_planes_to_buffer, encode_state_to_buffer,
    encode_state_to_buffer_channels, to_planes, to_planes_channels, MOVES_REMAINING_PLANE,
    MY_STONE_PLANE, OPP_STONE_PLANE, PLY_PARITY_PLANE,
};
pub use manifests::{
    anchor_path, assert_not_heldout, corpus_path, corpus_sha_pin, held_out_shas,
};
pub use registry::{
    all_specs, lookup, lookup_or_panic, parse_encoding_toml, registry_sha, registry_sha_hex,
};
pub use spec::{PolicyPool, RegistrySpec, Representation, ValuePool};

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
