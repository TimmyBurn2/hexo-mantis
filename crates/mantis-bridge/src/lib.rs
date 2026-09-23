//! mantis-bridge: ALL PyO3 lives here. Builds the `mantis._engine` extension module.
//!
//! Each module ships a `pub(crate) fn register(m)` that adds its pyclasses, free fns and
//! constants onto `mantis._engine`; the surface is pinned by `_engine.pyi` and its runtime test.
use pyo3::prelude::*;

mod board;
mod encoding;
mod graph_contract;
mod hexg;
mod inference;
mod mcts;
mod runner;
mod tactics;
mod utils;

/// Compiled mantis engine bridge (PyO3). All Python-facing Rust lives here.
#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    board::register(m)?; // Board
    encoding::register(m)?; // RegistrySpec + all_specs/registry_sha/registry_sha_hex
    graph_contract::register(m)?; // verify_edge_geometry
    mcts::register(m)?; // MCTSTree + SelectionDesync
    tactics::register(m)?; // TacticalSolver
    utils::register(m)?; // pool-overflow counters, armed-sims ceilings, graph_row_outcome
    inference::register(m)?; // InferenceBatcher + GraphWire + WireAlreadyConsumed
    runner::register(m)?; // SelfPlayRunnerConfig + SelfPlayRunner
    hexg::register(m)?; // HexgBuffer + GraphTargets
    Ok(())
}
