//! mantis-bridge: ALL PyO3 lives here. Builds the `mantis._engine` extension module.
//!
//! WP7 ASM assembly: each module (`board, encoding, mcts, tactics, utils,
//! graph_contract, inference, runner, buffer, hexg`) ships a `pub(crate) fn
//! register(m)` that adds its pyclasses / free fns onto the `_engine` module; the
//! `#[pymodule]` below calls them in the OLD `engine/src/lib.rs:39` order so the full
//! surface — 11 pyclasses (`Board, RegistrySpec, MCTSTree, TacticalSolver,
//! InferenceBatcher, GraphWire, SelfPlayRunnerConfig, SelfPlayRunner, ReplayBuffer,
//! HexgBuffer, GraphTargets`) + 4 free fns (`verify_edge_geometry,
//! apply_symmetries_batch, mcts_pool_overflow_count, take_mcts_pool_overflow_count`) +
//! 3 module fns (`all_specs, registry_sha, registry_sha_hex`) + the
//! `WireAlreadyConsumed` exception — resolves on `mantis._engine`.
use pyo3::prelude::*;

mod board;
mod encoding;
mod mcts;
mod tactics;
mod utils;
mod graph_contract;
mod runner;
mod inference;
mod buffer;
mod hexg;

/// Compiled mantis engine bridge (PyO3). All Python-facing Rust lives here.
///
/// Registration order ports OLD `engine/src/lib.rs:39`: `board`, `encoding` (class +
/// the 3 NEW-BUILD module fns), `graph_contract`, `mcts`, `tactics`, `utils`, then the
/// selfplay-facing classes (`InferenceBatcher`, `GraphWire`, `SelfPlayRunnerConfig`,
/// `SelfPlayRunner`, `ReplayBuffer`, `HexgBuffer`, `GraphTargets`).
#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    board::register(m)?; // Board
    encoding::register(m)?; // RegistrySpec + all_specs/registry_sha/registry_sha_hex
    graph_contract::register(m)?; // verify_edge_geometry
    mcts::register(m)?; // MCTSTree
    tactics::register(m)?; // TacticalSolver
    utils::register(m)?; // apply_symmetries_batch + (take_)mcts_pool_overflow_count
    inference::register(m)?; // InferenceBatcher + GraphWire + WireAlreadyConsumed
    runner::register(m)?; // SelfPlayRunnerConfig + SelfPlayRunner
    buffer::register(m)?; // ReplayBuffer
    hexg::register(m)?; // HexgBuffer + GraphTargets
    Ok(())
}
