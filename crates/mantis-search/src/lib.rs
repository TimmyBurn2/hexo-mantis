//! mantis-search: MCTS (PUCT or Gumbel — one closed `SearchKind`), completed-Q,
//! tactics solver.
//!
//! Depends on `mantis-core` only (repo DAG; `mantis-encoding` is a test dependency). The MCTS
//! takes `n_actions` as a caller parameter and the Board carries plain geometry, so no
//! encoding/spec resolution happens here.

// The ported doc comments use markdown list-continuation lines that clippy's
// `doc_lazy_continuation` (a rendering nicety) flags; suppressed crate-wide to
// keep the verbatim doc structure. Logic-affecting lints are handled per-site.
#![allow(clippy::doc_lazy_continuation)]

pub mod legal_set;
pub mod mcts;
pub mod tactics;
pub mod temperature;

pub use legal_set::LegalSetPolicy;
pub use mcts::gumbel_mctx::MctxRootState;
pub use mcts::kind::SearchKind;
pub use mcts::QSigma;
pub use mcts::{
    pool_overflow_count, take_pool_overflow_count, CachedPolicy, MCTSTree, Node, TTEntry,
    MAX_ARMED_SIMS, MAX_ARMED_SIMS_GUMBEL, MAX_CHILDREN_PER_NODE, MAX_NODES, MAX_ROOT_CHILDREN,
    VIRTUAL_LOSS_PENALTY,
};
pub use tactics::{Budget, Outcome, ProofResult, TacticalConfig, TacticalSolver};
pub use temperature::{compute_move_temperature, ply_to_compound_move};
