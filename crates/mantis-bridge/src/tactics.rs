//! Python-visible `TacticalSolver` wrapper over `mantis_search::TacticalSolver`.
//!
//! Binds the native in-window-offense tactical proof solver so an offline
//! harness can drive it with no new perf build, and a future solver-probe DI
//! hook can swap in this engine.
//!
//! # `prove` contract
//! `prove(board, depth, node_budget) -> (result:int, line:list[(q,r)], nodes:int)`
//! where `result` is WIN=1 / LOSS=-1 / UNKNOWN=0. A `result == 1` (WIN) yields
//! a principal variation whose `line[0]` is the move to play and `line[1]` (if
//! present) is the cached 2nd stone of the turn.
//!
//! No Board is held — `TacticalSolver` is `Send + Sync` (POD config), so the
//! pyclass takes the default derive (no `unsendable`). F-42: `module = "mantis._engine"`.

use pyo3::prelude::*;

use mantis_search::{TacticalConfig, TacticalSolver};

use crate::board::PyBoard;

/// Native in-window-offense tactical proof solver.
///
/// `window_half`: in-window offense guard — a WIN whose played move is
/// cheb-distance > `window_half` from the window center is suppressed
/// (downgraded to UNKNOWN); `None` disables the guard. Default `9` (19-window
/// single-window band). `cand_cap`: threat-guided candidate cap (default 40).
#[pyclass(name = "TacticalSolver", module = "mantis._engine")]
pub struct PyTacticalSolver {
    inner: TacticalSolver,
}

#[pymethods]
impl PyTacticalSolver {
    #[new]
    #[pyo3(signature = (window_half = Some(9), cand_cap = 40, neighbor_dist = None))]
    pub fn new(window_half: Option<i32>, cand_cap: usize, neighbor_dist: Option<i32>) -> Self {
        PyTacticalSolver {
            inner: TacticalSolver::new(TacticalConfig {
                cand_cap,
                window_half,
                neighbor_dist,
            }),
        }
    }

    /// Prove the side-to-move at `board` within `depth` plies and `node_budget`
    /// board expansions.
    ///
    /// Returns `(result, line, nodes)`:
    ///   - `result`: 1 = WIN (side-to-move has a proven forced win),
    ///     -1 = LOSS, 0 = UNKNOWN (unresolved / off-window-suppressed).
    ///   - `line`: principal variation as `[(q, r), ...]` — populated for WIN;
    ///     `line[0]` is the move to play, `line[1]` (if present) is the 2nd
    ///     stone of a 2-stone forcing turn.
    ///   - `nodes`: board expansions charged (the honesty axis vs the deploy
    ///     search). NET-FREE: the value head is never read inside the proof.
    #[pyo3(signature = (board, depth, node_budget))]
    pub fn prove(
        &self,
        board: &PyBoard,
        depth: u32,
        node_budget: u64,
    ) -> (i32, Vec<(i32, i32)>, u64) {
        let res = self.inner.prove(board.inner_ref(), depth, node_budget);
        (res.result.to_i32(), res.line, res.nodes)
    }
}

/// Register the `TacticalSolver` pyclass into `_engine`. Called by Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTacticalSolver>()?;
    Ok(())
}
