// Exceeds the 300-line soft cap (R8): the full PyMCTSTree pymethod surface ports as one
// line-auditable unit with its tests.
//! Python-visible PUCT `MCTSTree` wrapper over `mantis_search::MCTSTree`.
//!
//! `unsendable`: the tree embeds a `Board` (Send + !Sync), so single-thread Python ownership is
//! the synchronization. `pending_boards` and `forced_root_child` are bridge-side mirrors of
//! `pub(crate)` tree state, reset on `new_game`/`reset` to match the tree's own reset.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use mantis_core::board::BOARD_SIZE;
use mantis_core::Board;
use mantis_search::{LegalSetPolicy, MCTSTree, MctxRootState, SearchKind};

use crate::board::PyBoard;

pyo3::create_exception!(
    mantis_engine,
    SelectionDesync,
    pyo3::exceptions::PyRuntimeError,
    "Raised when the PUCT descent selects a child whose action_idx decodes to a cell the \
     board refuses -- the tree and the board have desynchronised (AUDIT-1 F-02)."
);

/// Per-root-child info returned by `get_root_children_info`:
/// `((q, r), pool_idx, prior, visits, q_value)`.
type RootChildInfo = ((i32, i32), u32, f32, u32, f32);

/// Single-threaded PUCT MCTS tree exposed to Python: `new_game(board)`, then per simulation
/// `select_leaves(n)` followed by `expand_and_backup(policies, values)`, then `get_policy()`.
#[pyclass(name = "MCTSTree", module = "mantis._engine", unsendable)]
pub struct PyMCTSTree {
    inner: MCTSTree,
    board_size: usize,
    /// Bridge-held clones of the last `select_leaves` boards — the tree's `pending` is
    /// `pub(crate)`.
    pending_boards: Vec<Board>,
    /// Bridge mirror of the tree's `pub(crate)` `forced_root_child`, kept in lockstep.
    forced_root_child: Option<u32>,
    /// The Gumbel root state of the search in progress. HELD BY THE TREE rather than handed to
    /// Python: the state is only meaningful against the tree it was drawn over, and a
    /// Python-side handle could outlive a `new_game`. Cleared by `new_game`/`reset`.
    gumbel_root: Option<MctxRootState>,
}

#[pymethods]
impl PyMCTSTree {
    /// Args: `c_puct` exploration constant (1.5); `virtual_loss` fixed penalty (1.0);
    ///     `fpu_reduction` KataGo dynamic FPU base (0.25), an unvisited child's FPU being
    ///     `parent_q - fpu_reduction * sqrt(explored_mass)`; `quiescence_enabled` (True) and
    ///     `quiescence_blend_2` (0.3) for the proven forced win/loss override.
    #[new]
    #[pyo3(signature = (c_puct = 1.5, virtual_loss = 1.0, fpu_reduction = 0.25, quiescence_enabled = true, quiescence_blend_2 = 0.3))]
    pub fn new(
        c_puct: f32,
        virtual_loss: f32,
        fpu_reduction: f32,
        quiescence_enabled: bool,
        quiescence_blend_2: f32,
    ) -> Self {
        let mut inner = MCTSTree::new_full(c_puct, virtual_loss, fpu_reduction);
        inner.configure_quiescence(quiescence_enabled, quiescence_blend_2);
        PyMCTSTree {
            inner,
            board_size: BOARD_SIZE,
            pending_boards: Vec::new(),
            forced_root_child: None,
            gumbel_root: None,
        }
    }

    /// Select the search kind once per player, through the SAME setter the self-play worker
    /// calls — what makes "the bar searches the way the run searched" a construction.
    /// `c_visit`/`c_scale` are the config's required keys; nothing here defaults them.
    ///
    /// # Errors
    /// `ValueError` — `kind` is not a search kind this build knows. REFUSED, never defaulted.
    pub fn configure_search(&mut self, kind: &str, c_visit: f32, c_scale: f32) -> PyResult<()> {
        let parsed = SearchKind::from_config_str(kind).ok_or_else(|| {
            PyValueError::new_err(format!(
                "search.kind={kind:?} is not a known search kind (expected \"puct\" or \
                 \"gumbel\")"
            ))
        })?;
        self.inner.configure_search(parsed, c_visit, c_scale);
        self.gumbel_root = None;
        Ok(())
    }

    /// The search kind this tree runs, as its config spelling.
    #[getter]
    pub fn search_kind(&self) -> &'static str {
        self.inner.search_kind().as_config_str()
    }

    /// Draw this search's Gumbel root state over the EXPANDED root, from an explicit seed —
    /// required rather than a thread RNG, because a promotion bar has to replay. `m` is Mctx's
    /// `max_num_considered_actions`, `budget` the simulations descending from the root.
    ///
    /// # Errors
    /// `RuntimeError` — the root is not expanded, so there is nothing to draw over.
    pub fn gumbel_root_begin(&mut self, m: usize, budget: usize, seed: u64) -> PyResult<()> {
        if self.inner.root_n_children() == 0 {
            return Err(PyRuntimeError::new_err(
                "gumbel_root_begin: the root is not expanded — evaluate the root leaf first \
                 (the root's own evaluation is charged against the budget)",
            ));
        }
        self.gumbel_root = Some(MctxRootState::new_seeded(&self.inner, m, budget, seed));
        Ok(())
    }

    /// The root child this simulation must descend into, as a pool index, or `None` when
    /// the schedule is exhausted.
    ///
    /// # Errors
    /// `RuntimeError` — no root state has been drawn for this search.
    pub fn gumbel_root_select(&self, c_visit: f32, c_scale: f32) -> PyResult<Option<u32>> {
        let state = self.gumbel_root.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("gumbel_root_select: call gumbel_root_begin first")
        })?;
        Ok(state.select(&self.inner, c_visit, c_scale))
    }

    /// Sequential Halving's answer: the highest-scoring of the MOST-VISITED root children,
    /// as an axial `(q, r)`. `None` when the root has no children.
    ///
    /// # Errors
    /// `RuntimeError` — no root state has been drawn for this search.
    pub fn gumbel_root_best_move(
        &self,
        c_visit: f32,
        c_scale: f32,
    ) -> PyResult<Option<(i32, i32)>> {
        let state = self.gumbel_root.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("gumbel_root_best_move: call gumbel_root_begin first")
        })?;
        Ok(state
            .best_action(&self.inner, c_visit, c_scale)
            .map(|pool_idx| {
                let val = self.inner.pool[pool_idx as usize].action_idx;
                ((val >> 16) as i32 - 32768, (val & 0xFFFF) as i32 - 32768)
            }))
    }

    /// Total quiescence value overrides/blends since last `new_game()`.
    #[getter]
    pub fn get_quiescence_fire_count(&self) -> u64 {
        self.inner
            .quiescence_fire_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Search statistics since the last `new_game()`, as `(mean_depth, root_concentration)`:
    /// average leaf depth across all simulations, and max child visits / total root visits in
    /// [0.0, 1.0]. Both 0.0 before any simulations; call after search completes, not during.
    pub fn last_search_stats(&self) -> (f32, f32) {
        self.inner.last_search_stats()
    }

    /// Reset the tree for a new game starting from `board`, re-using the pre-allocated pool.
    pub fn new_game(&mut self, board: &PyBoard) {
        self.board_size = BOARD_SIZE;
        self.pending_boards.clear();
        self.forced_root_child = None;
        self.gumbel_root = None;
        self.inner.new_game(board.inner_ref().clone());
    }

    /// Select up to `n` distinct leaves for evaluation, one Board per unique leaf; always call
    /// `expand_and_backup` with the same number of results before the next call.
    ///
    /// Raises:
    ///     SelectionDesync: tree and board disagree about what has been played — once a
    ///         `PanicException` recovered from by matching its message text.
    pub fn select_leaves(&mut self, py: Python<'_>, n: usize) -> PyResult<Vec<Py<PyBoard>>> {
        let boards = py
            .detach(|| self.inner.select_leaves(n))
            .map_err(|desync| SelectionDesync::new_err(desync.to_string()))?;
        // Keep our own clones for the ls path (the tree's `pending` is pub(crate)).
        self.pending_boards = boards.clone();
        boards
            .into_iter()
            .map(|b| Py::new(py, PyBoard::from_inner(b)))
            .collect()
    }

    /// Expand leaves and backup values from the last `select_leaves` call: `policies` is one
    /// vector per leaf of length `board_size * board_size + 1`, `values` one scalar in [-1, 1]
    /// per leaf from the leaf's current player.
    ///
    /// Raises:
    ///     ValueError: the lengths do not match the preceding `select_leaves` list. The inner
    ///         call took `min(pending, policies, values)` and DROPPED the rest, so a short
    ///         batch silently expanded fewer leaves.
    pub fn expand_and_backup(
        &mut self,
        py: Python<'_>,
        policies: Vec<Vec<f32>>,
        values: Vec<f32>,
    ) -> PyResult<()> {
        let pending = self.pending_boards.len();
        if policies.len() != pending || values.len() != pending {
            return Err(PyValueError::new_err(format!(
                "expand_and_backup: the last select_leaves returned {pending} leaf/leaves, \
                 but got {} policy row(s) and {} value(s). A short batch used to expand the \
                 leading leaves and silently drop the rest",
                policies.len(),
                values.len()
            )));
        }
        py.detach(|| self.inner.expand_and_backup(&policies, &values));
        Ok(())
    }

    /// Graph counterpart of `expand_and_backup`: the deploy Gumbel-SH head expands children over
    /// the FULL legal set, the action space the net trains under in self-play.
    ///
    /// Args: `policies` per-leaf dense prob vectors of `policy_stride`; `overflows` per-leaf
    ///     off-window `((q, r), p)` entries; `values` per-leaf scalars; `centers` the BUILDER's
    ///     own `(cq, cr)` per leaf, threaded from the Python decode because the builder is the
    ///     one authority for the frame; `policy_stride` and `trunk_sz` the dense half's shape.
    #[pyo3(signature = (policies, overflows, values, centers, policy_stride, trunk_sz))]
    #[allow(clippy::too_many_arguments)] // Python-facing signature (7 params incl. py)
    #[allow(clippy::type_complexity)]
    pub fn expand_and_backup_ls_graph(
        &mut self,
        py: Python<'_>,
        policies: Vec<Vec<f32>>,
        overflows: Vec<Vec<((i32, i32), f32)>>,
        values: Vec<f32>,
        centers: Vec<(i32, i32)>,
        policy_stride: usize,
        trunk_sz: i32,
    ) -> PyResult<()> {
        // The four arity conjuncts, checked separately so a flip names the one that failed.
        let n_pending = self.pending_boards.len();
        for (label, len) in [
            ("policies", policies.len()),
            ("overflows", overflows.len()),
            ("values", values.len()),
            ("centers", centers.len()),
        ] {
            if len != n_pending {
                return Err(PyValueError::new_err(format!(
                    "expand_and_backup_ls_graph: {label} has {len} entries but there are \
                     {n_pending} pending leaves (the inner expand takes the MIN and would \
                     silently expand fewer)"
                )));
            }
        }

        let mut ls_vec: Vec<LegalSetPolicy> = Vec::with_capacity(n_pending);
        for (i, board) in self.pending_boards.iter().enumerate() {
            // C-4: a dense half of the wrong width is a silent wrong-width decode, and it
            // must be loud on the graph seam.
            if policies[i].len() != policy_stride {
                return Err(PyValueError::new_err(format!(
                    "expand_and_backup_ls_graph leaf {i}: dense half has {} slots != \
                     policy_stride={policy_stride}",
                    policies[i].len()
                )));
            }
            // The producer's builder centre against the leaf board's own. Expected
            // always-equal, so this is a pairing/drift tripwire, not a correction.
            let board_center = board.window_center();
            if board_center != centers[i] {
                return Err(PyValueError::new_err(format!(
                    "expand_and_backup_ls_graph leaf {i}: builder window_center {:?} != \
                     board.window_center() {board_center:?} (coord/slot drift — the priors \
                     would be read in a different frame from the one they were baked in)",
                    centers[i]
                )));
            }
            // Mirrors the self-play always-on assert and the CNN sibling.
            if board.cluster_window_size() as i32 != trunk_sz {
                return Err(PyValueError::new_err(format!(
                    "expand_and_backup_ls_graph leaf {i}: trunk_sz={trunk_sz} != \
                     board.cluster_window_size()={}",
                    board.cluster_window_size()
                )));
            }
            let mut ls = LegalSetPolicy {
                dense: policies[i].clone(),
                ..LegalSetPolicy::default()
            };
            for &(coord, prob) in &overflows[i] {
                ls.overflow.insert(coord, prob);
            }
            ls_vec.push(ls);
        }

        py.detach(|| {
            self.inner
                .expand_and_backup_ls_at(&ls_vec, &values, &centers, trunk_sz)
        });
        Ok(())
    }

    /// The visit-count policy at the root, as a list of length `board_size * board_size + 1`.
    /// `temperature` 0 is argmax; `board_size` defaults to the size from the last `new_game`.
    #[pyo3(signature = (temperature = 1.0, board_size = None))]
    pub fn get_policy<'py>(
        &self,
        py: Python<'py>,
        temperature: f32,
        board_size: Option<usize>,
    ) -> Bound<'py, PyArray1<f32>> {
        let bs = board_size.unwrap_or(self.board_size);
        // The inner API takes `n_actions` (= policy_stride). The Python-side
        // MCTSTree path is v6-only today, so bs²+1 is correct. Zero-copy return.
        let n_actions = bs * bs + 1;
        self.inner
            .get_policy(temperature, n_actions)
            .into_pyarray(py)
    }

    /// Total visit count at the root (= number of simulations run).
    pub fn root_visits(&self) -> u32 {
        self.inner.root_visits()
    }

    /// Reset the tree to its root state (for benchmarking / reuse).
    pub fn reset(&mut self) {
        self.forced_root_child = None;
        self.inner.reset();
    }

    /// Run `n` simulations using uniform priors and value=0 (no neural network).
    /// Used for CPU-only MCTS throughput benchmarking.
    pub fn run_simulations_cpu_only(&mut self, n: usize) {
        self.inner.run_simulations_cpu_only(n);
    }

    /// Mix Dirichlet noise into the root node's priors (self-play only), after the first
    /// `expand_and_backup` has expanded the root. `noise` is a list of floats of length
    /// `root_n_children()`; `epsilon` the mixing weight (default 0.25 per AlphaZero).
    #[pyo3(signature = (noise, epsilon = 0.25))]
    pub fn apply_dirichlet_to_root(&mut self, noise: Vec<f32>, epsilon: f32) {
        self.inner.apply_dirichlet_to_root(&noise, epsilon);
    }

    /// Number of children at the root (0 if not yet expanded) — the noise vector length for
    /// `apply_dirichlet_to_root`.
    pub fn root_n_children(&self) -> usize {
        self.inner.root_n_children()
    }

    /// Top-N children of root by visit count, as `((q, r), visits, prior, q_value)` sorted by
    /// visits descending. `(q, r)` is a raw axial tuple; Python callers format it.
    pub fn get_top_visits(&self, n: usize) -> Vec<((i32, i32), u32, f32, f32)> {
        self.inner.get_top_visits(n)
    }

    /// Value estimate at root from perspective of player to move.
    pub fn root_value(&self) -> f32 {
        self.inner.root_value()
    }

    /// Get/set forced root child for Gumbel Sequential Halving: a child pool index restricts
    /// `select_leaves` to that subtree, `None` restores normal PUCT selection.
    #[getter]
    pub fn forced_root_child(&self) -> Option<u32> {
        self.forced_root_child
    }

    #[setter]
    ///
    /// Raises:
    ///     ValueError: `val` is not one of the ROOT's children. The store was unchecked, so an
    ///         index at or beyond `MAX_NODES` index-panicked and any other foreign index
    ///         descended into a node the root does not own.
    pub fn set_forced_root_child(&mut self, val: Option<u32>) -> PyResult<()> {
        self.inner
            .set_forced_root_child(val)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.forced_root_child = val;
        Ok(())
    }

    /// `((q, r), pool_idx, prior, visits, q_value)` for each root child, for the policy viewer's
    /// Gumbel Sequential Halving. `(q, r)` is a raw axial tuple.
    pub fn get_root_children_info(&self) -> Vec<RootChildInfo> {
        let children = self.inner.get_root_children_info();
        let q_sign: f32 = if self.inner.pool[0].moves_remaining == 1 {
            -1.0
        } else {
            1.0
        };
        children
            .into_iter()
            .map(|(pool_idx, prior)| {
                let child = &self.inner.pool[pool_idx as usize];
                let visits = child.n_visits;
                let q_value = if visits > 0 {
                    q_sign * child.w_value / visits as f32
                } else {
                    0.0
                };
                let val = child.action_idx;
                let aq = (val >> 16) as i32 - 32768;
                let ar = (val & 0xFFFF) as i32 - 32768;
                ((aq, ar), pool_idx, prior, visits, q_value)
            })
            .collect()
    }

    /// Improved policy targets from Gumbel completed Q-values (Danihelka et al., ICLR 2022),
    /// for the policy viewer's Gumbel-mode overlay.
    #[pyo3(signature = (board_size = None, c_visit = 50.0, c_scale = 1.0))]
    pub fn get_improved_policy<'py>(
        &self,
        py: Python<'py>,
        board_size: Option<usize>,
        c_visit: f32,
        c_scale: f32,
    ) -> Bound<'py, PyArray1<f32>> {
        let bs = board_size.unwrap_or(self.board_size);
        let n_actions = bs * bs + 1;
        self.inner
            .get_improved_policy(n_actions, c_visit, c_scale)
            .into_pyarray(py)
    }
}

/// Register the `MCTSTree` pyclass into `_engine`. Called by Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMCTSTree>()?;
    // The named face of the desync, registered so a Python caller can name it in an `except`
    // clause — the point of retiring the `PanicException` matched by message text.
    m.add("SelectionDesync", m.py().get_type::<SelectionDesync>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_root_child_mirror_round_trips() {
        // The setter validates against the ROOT's child range, so the round-trip needs a root
        // that HAS children and an index that is one of them.
        let mut t = PyMCTSTree::new(1.5, 1.0, 0.25, true, 0.3);
        assert_eq!(t.forced_root_child(), None);
        assert!(
            t.set_forced_root_child(Some(7)).is_err(),
            "an unexpanded root owns no child 7"
        );

        let seed = PyBoard::new();
        t.new_game(&seed);
        Python::attach(|py| {
            let _leaves = t.select_leaves(py, 1).expect("a fresh root selects itself");
            let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
            t.expand_and_backup(py, vec![vec![1.0 / n_actions as f32; n_actions]], vec![0.0])
                .expect("one policy for one leaf");
        });
        let first = t.inner.pool[0].first_child;
        t.set_forced_root_child(Some(first))
            .expect("the root's own first child");
        assert_eq!(t.forced_root_child(), Some(first));
        // new_game resets the mirror (matches the tree's internal reset).
        let board = PyBoard::new();
        t.new_game(&board);
        assert_eq!(t.forced_root_child(), None);
    }

    #[test]
    fn cpu_only_simulations_visit_root() {
        let mut t = PyMCTSTree::new(1.5, 1.0, 0.0, true, 0.3);
        t.run_simulations_cpu_only(16);
        assert!(
            t.root_visits() > 0,
            "cpu-only sims must accumulate root visits"
        );
    }

    #[test]
    fn quiescence_fire_count_starts_zero() {
        let t = PyMCTSTree::new(1.5, 1.0, 0.25, true, 0.3);
        assert_eq!(t.get_quiescence_fire_count(), 0);
    }

    /// Numpy-free GIL round-trip: `select_leaves` caches the leaf boards for the ls path, and
    /// the GIL-release `expand_and_backup` accumulates root visits.
    #[test]
    fn select_and_expand_round_trip_under_gil() {
        Python::initialize();
        Python::attach(|py| {
            let mut t = PyMCTSTree::new(1.5, 1.0, 0.25, false, 0.3);
            let board = PyBoard::new();
            t.new_game(&board);
            let leaves = t.select_leaves(py, 1).expect("select");
            assert_eq!(leaves.len(), 1);
            assert_eq!(
                t.pending_boards.len(),
                1,
                "bridge caches leaf boards for the ls path"
            );
            // n_actions for a v6 board = 19*19+1 = 362.
            let policies = vec![vec![1.0f32 / 362.0; 362]];
            let values = vec![0.0f32];
            t.expand_and_backup(py, policies, values).expect("expand");
            assert!(t.root_visits() >= 1);
        });
    }
}
