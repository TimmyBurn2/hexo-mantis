//! Module-level Python-visible utility functions: MCTS pool-overflow accessors, the armed-sims
//! ceilings and the graph row-outcome helper. The batched dense symmetry scatter went with the
//! grid path (R346(f)).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_search::{
    omitted_prior_stats, pool_overflow_count, take_omitted_prior_stats, take_pool_overflow_count,
};
use mantis_selfplay::records::finalize_graph_outcome;

/// Read the process-wide MCTS pool-overflow counter without resetting.
///
/// Pool overflow is a hard panic — the counter is incremented immediately
/// before the panic inside the MCTS node allocator. A live process therefore
/// never observes a nonzero value from its own work. Non-zero reads at startup
/// indicate a previous-life event (a test fixture with a hand-crafted small
/// pool, or a config that drove the worker outside MCTS `MAX_NODES`' design
/// envelope) carried across the symbol surface, not a silent terminal-value
/// fabrication.
///
/// The counter is global (all trees across all worker threads share it).
/// Bench harnesses use the take-counterpart (`take_mcts_pool_overflow_count`)
/// to bracket measurement windows and reject contaminated runs.
#[pyfunction]
pub(crate) fn mcts_pool_overflow_count() -> u64 {
    pool_overflow_count()
}

/// Atomically read-and-reset the pool-overflow counter. Returns the
/// previous value. Used by the bench harness to bracket per-run
/// measurement windows and detect contamination.
#[pyfunction]
pub(crate) fn take_mcts_pool_overflow_count() -> u64 {
    take_pool_overflow_count()
}

/// R345(b)(5) — `(omitted_prior_mass_micros, expansions_that_omitted, total_expansions)`.
///
/// The Top-K cap keeps the highest-prior `MAX_CHILDREN_PER_NODE` children of a leaf and drops
/// the rest. `topk_truncated` said only that SOMETHING was dropped, which at radius 8 is true
/// on essentially every ply and therefore carries no information. This says how much PRIOR
/// MASS went with it — the quantity that decides whether the cap costs the search anything,
/// and the one a decision to raise it has to be argued against.
///
/// Mass is fixed-point (x 1e6): there is no atomic f32, and a float sum across worker threads
/// would not be reproducible even if there were.
#[pyfunction]
pub(crate) fn mcts_omitted_prior_stats() -> (u64, u64, u64) {
    omitted_prior_stats()
}

/// Atomically read-and-reset all three, to bracket a measurement window.
#[pyfunction]
pub(crate) fn take_mcts_omitted_prior_stats() -> (u64, u64, u64) {
    take_omitted_prior_stats()
}

/// The `(outcome, value_valid)` a graph training row carries, from THE authority.
///
/// NIGHTRUN-1 Leg 3. A bootstrap corpus encoder has to stamp the same value target
/// self-play stamps, and the rule — `+1` iff the winner IS this row's player, `-1` if not,
/// the ply-cap value with `value_valid = 0` on a truncation, the draw reward otherwise — is
/// `mantis_selfplay::records::finalize_graph_outcome`. Exposing it is what stops a second
/// Python transcription of a sign convention whose §178 split is already pinned by a Rust
/// test; a transcription would agree today and drift the first time the split moves.
///
/// `winner` is `1` / `-1` for a decided game and `0` for none — the `Option<Player>` the
/// Rust side takes, in the shape a corpus record actually carries. `terminal_reason` is the
/// runner's own code and `2` is the ply-cap branch.
///
/// # Errors
/// `ValueError` if `rec_player` or `winner` is outside its declared set. Refused rather
/// than coerced: a silently-mapped player is a value target with the wrong sign, which no
/// downstream check can see.
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

/// The largest sim budget the MCTS node pool can serve (AUDIT-1 F-21).
///
/// Exported so the config schema can bound `n_simulations` and its siblings against the
/// ENGINE'S own derivation rather than a re-typed number. `MAX_ARMED_SIMS` is
/// `MAX_NODES / (4 * MAX_CHILDREN_PER_NODE)`, and both constants live in `mantis-search`; a
/// literal on the Python side would be a second authority for a bound only the pool knows.
#[pyfunction]
pub fn mcts_max_armed_sims() -> usize {
    mantis_search::MAX_ARMED_SIMS
}

/// The same bound under `search.kind: gumbel`, which spends `MAX_ROOT_CHILDREN` slots on
/// its root instead of `MAX_CHILDREN_PER_NODE` and therefore has a lower ceiling.
///
/// Exported for the same reason as its sibling: the schema must be able to refuse an
/// over-budget Gumbel config at MINT rather than at boot (R255/ADJ-D34's inversion), and it
/// cannot do that against a number it does not have.
#[pyfunction]
pub fn mcts_max_armed_sims_gumbel() -> usize {
    mantis_search::MAX_ARMED_SIMS_GUMBEL
}

/// Register the utility free fns into `_engine`. Called by Slice ASM.
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
