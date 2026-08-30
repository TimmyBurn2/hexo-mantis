//! Module-level Python-visible utility functions:
//! batched symmetry scatter + MCTS pool-overflow accessors.
//!
//! `apply_symmetries_batch` exposes the exact Rust D6 scatter kernel used by the
//! ReplayBuffer sampling path (and `Board.to_tensor()`) so Python callers
//! (pretrain collate, parity tests) can never diverge (Q13 parity). The pool-
//! overflow accessors read the process-wide MCTS node-allocator counter.
//!
//! Thread-local `SymTables` avoids per-call allocation.

use numpy::{IntoPyArray, PyArray4, PyArrayMethods, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_core::board::BOARD_SIZE;
use mantis_search::{pool_overflow_count, take_pool_overflow_count};
use mantis_selfplay::records::finalize_graph_outcome;
use mantis_selfplay::replay::sample::apply_symmetry_state;
use mantis_selfplay::replay::sym::{SymTables, N_SYMS};

// ── Symmetry + chain-plane bindings (Q13 pretrain parity) ────────────────────
//
// State planes do not permute under hex dihedral symmetry — only cell
// coordinates do — so a single scatter table applies to any plane count.
// Thread-local SymTables avoids per-call allocation.

thread_local! {
    static SYM_TABLES_TLS: SymTables = SymTables::new();
}

/// Batched hex-dihedral symmetry scatter.
///
/// Plane-count-generic: any positive `C` works (8 for HEXB v6 buffer planes,
/// 18 for legacy inference / corpus tensors). State planes do not permute
/// under hex dihedral symmetry — only cell coordinates do — so a single
/// scatter table applies to any plane count.
///
/// Args:
///     states:      (N, C, 19, 19) float32 numpy array.
///     sym_indices: (N,) integer sym_idx per state, values in [0, 12).
///
/// Returns a newly-allocated (N, C, 19, 19) float32 numpy array.
#[pyfunction]
pub(crate) fn apply_symmetries_batch<'py>(
    py: Python<'py>,
    states: PyReadonlyArray4<'py, f32>,
    sym_indices: Vec<usize>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let shape = states.shape();
    if shape.len() != 4 || shape[2] != BOARD_SIZE || shape[3] != BOARD_SIZE {
        return Err(PyValueError::new_err(format!(
            "expected states shape (N, C, {BOARD_SIZE}, {BOARD_SIZE}); got {shape:?}"
        )));
    }
    let n = shape[0];
    let n_planes = shape[1];
    if sym_indices.len() != n {
        return Err(PyValueError::new_err(format!(
            "sym_indices length {} != batch size {}",
            sym_indices.len(),
            n
        )));
    }
    for (i, &s) in sym_indices.iter().enumerate() {
        if s >= N_SYMS {
            return Err(PyValueError::new_err(format!(
                "sym_indices[{i}] = {s} out of range (expected 0..{N_SYMS})"
            )));
        }
    }
    let stride = n_planes * BOARD_SIZE * BOARD_SIZE;
    let src = states.as_slice()?;
    let mut dst = vec![0.0f32; n * stride];
    SYM_TABLES_TLS.with(|tables| {
        for b in 0..n {
            let src_b = &src[b * stride..(b + 1) * stride];
            let dst_b = &mut dst[b * stride..(b + 1) * stride];
            apply_symmetry_state::<f32>(src_b, dst_b, sym_indices[b], tables);
        }
    });
    dst.into_pyarray(py)
        .reshape([n, n_planes, BOARD_SIZE, BOARD_SIZE])
}

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

/// Register the utility free fns into `_engine`. Called by Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_symmetries_batch, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_pool_overflow_count, m)?)?;
    m.add_function(wrap_pyfunction!(take_mcts_pool_overflow_count, m)?)?;
    m.add_function(wrap_pyfunction!(graph_row_outcome, m)?)?;
    Ok(())
}
