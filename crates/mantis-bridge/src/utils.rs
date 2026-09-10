//! Python-visible utilities: MCTS pool-overflow accessors, the armed-sims ceilings and the
//! graph row-outcome helper.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_search::{
    omitted_prior_stats, pool_overflow_count, take_omitted_prior_stats, take_pool_overflow_count,
};
use mantis_selfplay::records::finalize_graph_outcome;

/// Read the process-wide MCTS pool-overflow counter without resetting.
///
/// The counter is global and is incremented immediately before the hard panic overflow causes,
/// so a live process never observes a nonzero value from its own work.
#[pyfunction]
pub(crate) fn mcts_pool_overflow_count() -> u64 {
    pool_overflow_count()
}

/// Atomically read-and-reset the pool-overflow counter, returning the previous value.
#[pyfunction]
pub(crate) fn take_mcts_pool_overflow_count() -> u64 {
    take_pool_overflow_count()
}

/// `(omitted_prior_mass_micros, expansions_that_omitted, total_expansions)` — how much prior
/// mass the Top-K child cap drops, the quantity that says whether the cap costs the search.
///
/// Mass is fixed-point (x 1e6): there is no atomic f32, and a threaded float sum would not be
/// reproducible even if there were.
#[pyfunction]
pub(crate) fn mcts_omitted_prior_stats() -> (u64, u64, u64) {
    omitted_prior_stats()
}

/// Atomically read-and-reset all three, to bracket a measurement window.
#[pyfunction]
pub(crate) fn take_mcts_omitted_prior_stats() -> (u64, u64, u64) {
    take_omitted_prior_stats()
}

/// The `(outcome, value_valid)` a graph training row carries, from THE authority — exposed so a
/// bootstrap corpus encoder cannot transcribe the sign convention a second time in Python.
///
/// `winner` is `1` / `-1` for a decided game and `0` for none; `terminal_reason` is the
/// runner's own code and `2` is the ply-cap branch.
///
/// # Errors
/// `ValueError` if `rec_player` or `winner` is outside its declared set — refused rather than
/// coerced, since a silently-mapped player is a value target with the wrong sign.
#[pyfunction]
#[pyo3(signature = (rec_player, winner, terminal_reason, ply_cap_value, draw_reward))]
pub(crate) fn graph_row_outcome(
    rec_player: i8,
    winner: i8,
    terminal_reason: u8,
    ply_cap_value: f32,
    draw_reward: f32,
) -> PyResult<(f32, u8)> {
    if rec_player != 1 && rec_player != -1 {
        return Err(PyValueError::new_err(format!(
            "graph_row_outcome: rec_player {rec_player} out of range (expected +1 / -1)"
        )));
    }
    let winner_player = match winner {
        1 => Some(mantis_core::Player::One),
        -1 => Some(mantis_core::Player::Two),
        0 => None,
        other => {
            return Err(PyValueError::new_err(format!(
                "graph_row_outcome: winner {other} out of range (expected +1 / -1 / 0)"
            )))
        }
    };
    Ok(finalize_graph_outcome(
        rec_player,
        winner_player,
        terminal_reason,
        ply_cap_value,
        draw_reward,
    ))
}

/// The largest sim budget the MCTS node pool can serve: `MAX_NODES / (4 * MAX_CHILDREN_PER_NODE)`.
///
/// Exported so the config schema bounds `n_simulations` against the engine's own derivation
/// rather than a Python-side literal that would be a second authority.
#[pyfunction]
pub fn mcts_max_armed_sims() -> usize {
    mantis_search::MAX_ARMED_SIMS
}

/// The same bound under `search.kind: gumbel`, which spends `MAX_ROOT_CHILDREN` slots on its
/// root instead of `MAX_CHILDREN_PER_NODE` and therefore has a lower ceiling.
///
/// Exported so the schema can refuse an over-budget Gumbel config at MINT rather than at boot.
#[pyfunction]
pub fn mcts_max_armed_sims_gumbel() -> usize {
    mantis_search::MAX_ARMED_SIMS_GUMBEL
}

/// Register the utility free fns into `_engine`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcts_max_armed_sims, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_max_armed_sims_gumbel, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_pool_overflow_count, m)?)?;
    m.add_function(wrap_pyfunction!(take_mcts_pool_overflow_count, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_omitted_prior_stats, m)?)?;
    m.add_function(wrap_pyfunction!(take_mcts_omitted_prior_stats, m)?)?;
    m.add_function(wrap_pyfunction!(graph_row_outcome, m)?)?;
    Ok(())
}
