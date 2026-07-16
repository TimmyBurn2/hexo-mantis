//! mantis-search: MCTS (PUCT + Gumbel), completed-Q, tactics solver.
//! WP0 scaffold: compiles empty; search port lands with its work package.

pub const CRATE_NAME: &str = "mantis-search";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_pinned() {
        assert_eq!(super::CRATE_NAME, "mantis-search");
    }

    #[test]
    fn dag_deps_compile() {
        assert_eq!(mantis_core::CRATE_NAME, "mantis-core");
        assert_eq!(mantis_encoding::CRATE_NAME, "mantis-encoding");
    }
}
