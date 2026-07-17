//! mantis-selfplay: runner, worker loop, inference queues, replay buffers.
//! WP0 scaffold: compiles empty; selfplay port lands with its work package.

pub mod replay;

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
