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

/// Register the three utility free fns into `_engine`. Called by Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_symmetries_batch, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_pool_overflow_count, m)?)?;
    m.add_function(wrap_pyfunction!(take_mcts_pool_overflow_count, m)?)?;
    Ok(())
}
