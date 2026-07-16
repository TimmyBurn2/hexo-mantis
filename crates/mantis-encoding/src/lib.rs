//! mantis-encoding: registry.toml + spec + validators + dense encode kernels.
//! WP0 scaffold: compiles empty; registry port lands with its work package.

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
