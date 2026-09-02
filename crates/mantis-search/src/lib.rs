//! mantis-search: MCTS (PUCT + Gumbel), completed-Q, tactics solver.
//!
//! Depends on `mantis-core` (board, geometry, rules, Ply) and `mantis-encoding`
//! only (repo DAG). The MCTS takes `n_actions` as a caller parameter and the
//! Board carries plain geometry, so no encoding/spec resolution happens here.

// The ported doc comments use markdown list-continuation lines that clippy's
// `doc_lazy_continuation` (a rendering nicety) flags; suppressed crate-wide to
// keep the verbatim doc structure. Logic-affecting lints are handled per-site.
#![allow(clippy::doc_lazy_continuation)]

pub mod legal_set;
pub mod mcts;
pub mod tactics;
pub mod temperature;

pub use legal_set::{is_covered, LegalSetPolicy};
pub use mcts::gumbel::GumbelSearchState;
pub use mcts::{
    pool_overflow_count, take_pool_overflow_count, CachedPolicy, MCTSTree, Node, TTEntry,
    MAX_ARMED_SIMS, MAX_CHILDREN_PER_NODE, MAX_NODES, VIRTUAL_LOSS_PENALTY,
};
pub use tactics::{Budget, Outcome, ProofResult, TacticalConfig, TacticalSolver};
pub use temperature::{compute_move_temperature, ply_to_compound_move};

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
