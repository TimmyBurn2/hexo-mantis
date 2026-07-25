//! mantis-selfplay: runner, worker loop, inference queues, replay buffers.
//! WP0 scaffold: compiles empty; selfplay port lands with its work package.

pub mod queues;
pub mod records;
pub mod replay;
pub mod runner;

// CARD-MINPIN: re-exported solely so the bit-for-bit parity pin
// (tests/min_value_aggregation_pin.rs) can reach the one named min-aggregation home.
pub use runner::search_drive::aggregate_cluster_values_min;

pub const CRATE_NAME: &str = "mantis-selfplay";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_pinned() {
        assert_eq!(super::CRATE_NAME, "mantis-selfplay");
    }

    #[test]
    fn dag_deps_compile() {
        assert_eq!(mantis_core::CRATE_NAME, "mantis-core");
        assert_eq!(mantis_encoding::CRATE_NAME, "mantis-encoding");
        assert_eq!(mantis_graph::CRATE_NAME, "mantis-graph");
        assert_eq!(mantis_search::CRATE_NAME, "mantis-search");
    }
}
