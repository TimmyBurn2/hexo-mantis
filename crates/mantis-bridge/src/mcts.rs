// Exceeds the 300-line soft cap (R8): the full PyMCTSTree pymethod surface
// (ctor-compose, the GIL-release select/expand/expand_and_backup_ls +
// expand_and_backup_ls_graph, policy getters, viewer accessors) ports as one
// line-auditable unit with its tests.
//! Python-visible PUCT `MCTSTree` wrapper over `mantis_search::MCTSTree`.
//!
//! `unsendable` (LOCKED #3): the tree embeds a `Board` (Send + !Sync), so
//! single-thread Python ownership is the synchronization. F-42:
//! `module = "mantis._engine"`.
//!
//! Two new-side adaptations, both bridge-internal (no cross-crate seam):
//! - `pending_boards`: the new `MCTSTree.pending` is `pub(crate)` (unreadable
//!   from the bridge), so the wrapper keeps its OWN clones of the leaf boards
//!   returned by `select_leaves` and drives the legal-set aggregation from them
//!   (behaviour-exact: same leaf boards, centers recomputed in Rust).
//! - `forced_root_child`: the new field is `pub(crate)` with a pub setter but NO
//!   pub getter, so the wrapper mirrors it (reset to `None` on `new_game`/`reset`,
//!   matching the tree's internal reset).

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1};

use mantis_core::Board;
use mantis_core::board::BOARD_SIZE;
use mantis_search::{LegalSetPolicy, MCTSTree};
use mantis_selfplay::records;

use crate::board::PyBoard;

/// Per-root-child info returned by `get_root_children_info`:
/// `((q, r), pool_idx, prior, visits, q_value)`. Used by the policy viewer
/// to drive Gumbel Sequential Halving from Python.
type RootChildInfo = ((i32, i32), u32, f32, u32, f32);

/// Single-threaded PUCT MCTS tree exposed to Python.
///
/// Usage (Python):
///
/// ```python
/// tree = MCTSTree(c_puct=1.5)
/// tree.new_game(board)
/// for _ in range(n_simulations):
///     boards = tree.select_leaves(1)
///     policies = [[...]]   # list of float lists, length = board_size^2 + 1
///     values   = [0.5]     # list of scalars
///     tree.expand_and_backup(policies, values)
/// policy = tree.get_policy(temperature=1.0, board_size=9)
/// visits = tree.root_visits()
/// ```
#[pyclass(name = "MCTSTree", module = "mantis._engine", unsendable)]
pub struct PyMCTSTree {
    inner: MCTSTree,
    board_size: usize,
    /// Bridge-held clones of the leaf boards from the last `select_leaves` — the
    /// substitute for the tree's `pub(crate)` `pending` used by the ls path.
    pending_boards: Vec<Board>,
    /// Bridge mirror of the tree's `pub(crate)` `forced_root_child` (pub setter,
    /// no pub getter). Kept in lockstep on set / new_game / reset.
    forced_root_child: Option<u32>,
}

#[pymethods]
impl PyMCTSTree {
    /// Args:
    ///     c_puct: exploration constant (default 1.5).
    ///     virtual_loss: fixed penalty (default 1.0).
    ///     fpu_reduction: KataGo-style dynamic FPU base (default 0.25).
    ///         FPU for unvisited children = parent_q - fpu_reduction * sqrt(explored_mass).
    ///         Set to 0.0 to disable (classical Q=0 for unvisited).
    ///     quiescence_enabled: override leaf value when forced win/loss is proven (default True).
    ///     quiescence_blend_2: blend amount for the 2-winning-moves case (default 0.3).
    #[new]
    #[pyo3(signature = (c_puct = 1.5, virtual_loss = 1.0, fpu_reduction = 0.25, quiescence_enabled = true, quiescence_blend_2 = 0.3))]
    pub fn new(c_puct: f32, virtual_loss: f32, fpu_reduction: f32, quiescence_enabled: bool, quiescence_blend_2: f32) -> Self {
        let mut inner = MCTSTree::new_full(c_puct, virtual_loss, fpu_reduction);
        inner.configure_quiescence(quiescence_enabled, quiescence_blend_2);
        PyMCTSTree {
            inner,
            board_size: BOARD_SIZE,
            pending_boards: Vec::new(),
            forced_root_child: None,
        }
    }

    /// Total quiescence value overrides/blends since last `new_game()`.
    #[getter]
    pub fn get_quiescence_fire_count(&self) -> u64 {
        self.inner.quiescence_fire_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Search statistics accumulated since the last `new_game()`.
    ///
    /// Returns `(mean_depth, root_concentration)`:
    /// - `mean_depth`: average leaf depth across all simulations this game/search
    /// - `root_concentration`: max child visits / total root visits ∈ [0.0, 1.0]
    ///
    /// Both 0.0 before any simulations. Call after search completes, not during.
    pub fn last_search_stats(&self) -> (f32, f32) {
        self.inner.last_search_stats()
    }

    /// Reset the tree for a new game starting from `board`.
    ///
    /// This re-uses the pre-allocated pool — no heap allocation.
    pub fn new_game(&mut self, board: &PyBoard) {
        self.board_size = BOARD_SIZE;
        self.pending_boards.clear();
        self.forced_root_child = None;
        self.inner.new_game(board.inner_ref().clone());
    }

    /// Select up to `n` distinct leaves for neural-network evaluation.
    ///
    /// Returns a list of Board objects (one per unique leaf).
    /// Always call `expand_and_backup` with the same number of results
    /// before the next call to `select_leaves`.
    pub fn select_leaves(&mut self, py: Python<'_>, n: usize) -> PyResult<Vec<Py<PyBoard>>> {
        let boards = py.detach(|| self.inner.select_leaves(n));
        // Keep our own clones for the ls path (the tree's `pending` is pub(crate)).
        self.pending_boards = boards.clone();
        boards
            .into_iter()
            .map(|b| Py::new(py, PyBoard::from_inner(b)))
            .collect()
    }

    /// Expand leaves and backup values from the last `select_leaves` call.
    ///
    /// Args:
    ///     policies: list of policy vectors (one per leaf).
    ///               Each vector has length `board_size * board_size + 1`.
    ///     values:   list of scalar values in [-1, 1] (one per leaf),
    ///               from the current player's perspective at that leaf.
    pub fn expand_and_backup(
        &mut self,
        py: Python<'_>,
        policies: Vec<Vec<f32>>,
        values: Vec<f32>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.expand_and_backup(&policies, &values));
        Ok(())
    }

    /// Legal-set (multi-window no-drop) counterpart of `expand_and_backup`.
    /// Productionizes the off-window decoding fix: the deploy Gumbel-SH head
    /// expands children over the FULL legal set (off-global-window cells COVERED
    /// by a cluster get a child), the action space the net already trains under
    /// in self-play.
    ///
    /// Mirrors the self-play worker aggregation EXACTLY: for each pending leaf,
    /// slice its `K` RAW per-cluster prob vectors + values, **recompute the
    /// cluster centers in Rust** via `get_cluster_views()` (the self-play
    /// center-order contract — never trust a Python-supplied order), **min-pool
    /// the values** (parity with selfplay `min_v`), build the ragged
    /// `LegalSetPolicy` via `records::aggregate_policy_ls`, then expand+backup via
    /// the inner `expand_and_backup_ls`.
    ///
    /// Args:
    ///     policies: FLAT list of per-cluster prob vectors (one per cluster, each
    ///               length `policy_stride`, = exp(log_policy); NO scatter-max, NO
    ///               drop, NO min-pool — Rust pools). Total len == sum(leaf_k).
    ///     values:   FLAT list of per-cluster scalar values, same leaf-major order.
    ///     leaf_k:   `K` (cluster count) per leaf; order aligned with the boards
    ///               returned by the preceding `select_leaves` (== pending order).
    ///     policy_stride:  action-space size (= encoding `policy_logit_count`).
    ///     has_pass_slot:  whether the last slot is the (dead) pass slot.
    ///     trunk_sz:       cluster window side length; cross-checked against each
    ///                     leaf board's `cluster_window_size()`.
    #[pyo3(signature = (policies, values, leaf_k, policy_stride, has_pass_slot, trunk_sz))]
    #[allow(clippy::too_many_arguments)] // ported Python-facing signature (8 params)
    pub fn expand_and_backup_ls(
        &mut self,
        py: Python<'_>,
        policies: Vec<Vec<f32>>,
        values: Vec<f32>,
        leaf_k: Vec<usize>,
        policy_stride: usize,
        has_pass_slot: bool,
        trunk_sz: i32,
    ) -> PyResult<()> {
        // K alignment: the flat per-cluster policy/value counts must equal sum(K).
        let total_k: usize = leaf_k.iter().sum();
        if total_k != policies.len() || total_k != values.len() {
            return Err(PyValueError::new_err(format!(
                "expand_and_backup_ls: K misalignment sum(leaf_k)={total_k} \
                 policies={} values={}",
                policies.len(),
                values.len()
            )));
        }
        if self.pending_boards.len() != leaf_k.len() {
            return Err(PyValueError::new_err(format!(
                "expand_and_backup_ls: pending leaves {} != leaf_k {}",
                self.pending_boards.len(),
                leaf_k.len()
            )));
        }

        // Build the ragged ls priors + min-pooled values from an IMMUTABLE read of
        // the bridge-held pending boards (centers RECOMPUTED in Rust — the
        // self-play center-order contract).
        let mut ls_vec: Vec<LegalSetPolicy> = Vec::with_capacity(leaf_k.len());
        let mut min_vals: Vec<f32> = Vec::with_capacity(leaf_k.len());
        {
            let mut curr = 0usize;
            for (i, board) in self.pending_boards.iter().enumerate() {
                let k = leaf_k[i];
                let (_views, centers) = board.get_cluster_views();
                if centers.len() != k {
                    return Err(PyValueError::new_err(format!(
                        "expand_and_backup_ls leaf {i}: Rust K={} != Python leaf_k={k} \
                         (get_cluster_views center-order contract violated)",
                        centers.len()
                    )));
                }
                if board.cluster_window_size() as i32 != trunk_sz {
                    return Err(PyValueError::new_err(format!(
                        "expand_and_backup_ls leaf {i}: trunk_sz={trunk_sz} != \
                         board.cluster_window_size()={}",
                        board.cluster_window_size()
                    )));
                }
                let leaf_policies = &policies[curr..curr + k];
                let leaf_values = &values[curr..curr + k];
                // min-pool values (selfplay parity: worst window = leaf value).
                let mut min_v = leaf_values[0];
                for &v in leaf_values {
                    if v < min_v {
                        min_v = v;
                    }
                }
                ls_vec.push(records::aggregate_policy_ls(
                    policy_stride,
                    has_pass_slot,
                    trunk_sz,
                    board,
                    &centers,
                    leaf_policies,
                ));
                min_vals.push(min_v);
                curr += k;
            }
        }

        py.detach(|| self.inner.expand_and_backup_ls(&ls_vec, &min_vals));
        Ok(())
    }

    /// GRAPH legal-set counterpart of `expand_and_backup` (WP12-R Phase
    /// EVALDECODE, operator ruling R138: eval consumes what the shared producer
    /// already returns, and SELF-PLAY SEMANTICS IS THE AUTHORITY).
    ///
    /// This adds NO search logic. It rebuilds the `LegalSetPolicy` the producer
    /// (`assemble_ls_from_gnn_probs`, via `submit_graphs_and_wait_ls`) already
    /// assembled — dense half plus the ragged off-window overflow — and calls the
    /// SAME `expand_and_backup_ls_at` self-play calls (`search_drive.rs:421`). At
    /// HEAD the eval leg kept only the dense half and expanded through the dense
    /// rule, so 53.2% of legal moves could not become children at all; the two
    /// halves of that fix are inseparable, because the ls path floors an
    /// uncovered coord at `1/min(n_legal,192)` where the dense path scores it 0.
    ///
    /// The overflow half is rebuilt into a MAP, never scanned in wire order: it
    /// crosses the FFI as a `Vec` materialised from `FxHashMap` iteration, so its
    /// order is an artifact (D-22, pinned by the P-1d oracle).
    ///
    /// Args:
    ///     policies:  dense half per pending leaf, each of length `policy_stride`.
    ///     overflows: ragged off-window half per pending leaf, `((q,r), prob)`.
    ///     values:    scalar value per pending leaf.
    ///     centers:   the BUILDER's window centre per pending leaf, as returned by
    ///                `InferenceBatcher.submit_graphs_and_wait_ls`. Cross-checked
    ///                against each pending board's own `window_center()`.
    ///     policy_stride: action-space size (= encoding `policy_logit_count`).
    ///     trunk_sz:      window side length; cross-checked against each leaf
    ///                    board's `cluster_window_size()`.
    ///
    /// Every guard below is ALWAYS-ON, not a `debug_assert`: the inner
    /// `expand_and_backup_ls_at` takes the MIN of every input length and silently
    /// expands fewer leaves, and the frame invariant it carries is a stripped
    /// `debug_assert_eq!` inert in the release `.so` production actually runs.
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
        // C-1a-d: the four arity conjuncts, each checked separately so a flip of
        // one names the one that failed.
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
            // C-4: a dense half of the wrong width is the v6w25 class of silent
            // wrong-width decode, and it must be loud on the graph seam too.
            if policies[i].len() != policy_stride {
                return Err(PyValueError::new_err(format!(
                    "expand_and_backup_ls_graph leaf {i}: dense half has {} slots != \
                     policy_stride={policy_stride}",
                    policies[i].len()
                )));
            }
            // C-2 (D-7): the producer's builder centre against the leaf board's own.
            // Expected always-equal — both are the bbox midpoint over the same stones
            // — so this is a pairing/drift tripwire, not a correction.
            let board_center = board.window_center();
            if board_center != centers[i] {
                return Err(PyValueError::new_err(format!(
                    "expand_and_backup_ls_graph leaf {i}: builder window_center {:?} != \
                     board.window_center() {board_center:?} (coord/slot drift — the priors \
                     would be read in a different frame from the one they were baked in)",
                    centers[i]
                )));
            }
            // C-3 (D-8): mirrors the self-play always-on assert at
            // `search_drive.rs:415-419` and the CNN sibling at `mcts.rs:216-222`.
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

    /// Return the visit-count policy at the root.
    ///
    /// Args:
    ///     temperature: sampling temperature (0 = argmax).
    ///     board_size:  spatial dimension (default: size from last `new_game`).
    ///
    /// Returns a list of length `board_size * board_size + 1`.
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
        self.inner.get_policy(temperature, n_actions).into_pyarray(py)
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

    /// Mix Dirichlet noise into the root node's priors (self-play only).
    ///
    /// Call after the first expand_and_backup (which expands the root).
    /// On the Python side, generate `noise` with:
    ///     noise = np.random.dirichlet([alpha] * tree.root_n_children()).tolist()
    ///
    /// Args:
    ///     noise:   list of floats, length == root_n_children().
    ///     epsilon: mixing weight (default 0.25 per AlphaZero).
    #[pyo3(signature = (noise, epsilon = 0.25))]
    pub fn apply_dirichlet_to_root(&mut self, noise: Vec<f32>, epsilon: f32) {
        self.inner.apply_dirichlet_to_root(&noise, epsilon);
    }

    /// Number of children at the root (0 if not yet expanded).
    /// Use this to determine the noise vector length before calling
    /// apply_dirichlet_to_root.
    pub fn root_n_children(&self) -> usize {
        self.inner.root_n_children()
    }

    /// Top-N children of root by visit count.
    /// Returns list of ((q, r), visits, prior, q_value) sorted by visits descending.
    /// `(q, r)` is a raw axial tuple; Python callers format at the call site.
    pub fn get_top_visits(&self, n: usize) -> Vec<((i32, i32), u32, f32, f32)> {
        self.inner.get_top_visits(n)
    }

    /// Value estimate at root from perspective of player to move.
    pub fn root_value(&self) -> f32 {
        self.inner.root_value()
    }

    // ── Policy viewer accessors ──────────────────────────────────────────────

    /// Get/set forced root child for Gumbel Sequential Halving.
    /// Set to a child pool index to restrict select_leaves to that subtree.
    /// Set to None to restore normal PUCT selection.
    #[getter]
    pub fn forced_root_child(&self) -> Option<u32> {
        self.forced_root_child
    }

    #[setter]
    pub fn set_forced_root_child(&mut self, val: Option<u32>) {
        self.inner.set_forced_root_child(val);
        self.forced_root_child = val;
    }

    /// Returns list of ((q, r), pool_idx, prior, visits, q_value) for each root child.
    /// Used by the policy viewer to drive Gumbel Sequential Halving from Python.
    /// `(q, r)` is a raw axial tuple; Python callers format at the call site.
    pub fn get_root_children_info(&self) -> Vec<RootChildInfo> {
        let children = self.inner.get_root_children_info();
        let q_sign: f32 = if self.inner.pool[0].moves_remaining == 1 { -1.0 } else { 1.0 };
        children.into_iter().map(|(pool_idx, prior)| {
            let child = &self.inner.pool[pool_idx as usize];
            let visits = child.n_visits;
            let q_value = if visits > 0 { q_sign * child.w_value / visits as f32 } else { 0.0 };
            let val = child.action_idx;
            let aq = (val >> 16) as i32 - 32768;
            let ar = (val & 0xFFFF) as i32 - 32768;
            ((aq, ar), pool_idx, prior, visits, q_value)
        }).collect()
    }

    /// Compute improved policy targets using Gumbel completed Q-values
    /// (Danihelka et al., ICLR 2022). Used by the policy viewer for
    /// Gumbel-mode analysis overlay.
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_root_child_mirror_round_trips() {
        let mut t = PyMCTSTree::new(1.5, 1.0, 0.25, true, 0.3);
        assert_eq!(t.forced_root_child(), None);
        t.set_forced_root_child(Some(7));
        assert_eq!(t.forced_root_child(), Some(7));
        // new_game resets the mirror (matches the tree's internal reset).
        let board = PyBoard::new();
        t.new_game(&board);
        assert_eq!(t.forced_root_child(), None);
    }

    #[test]
    fn cpu_only_simulations_visit_root() {
        let mut t = PyMCTSTree::new(1.5, 1.0, 0.0, true, 0.3);
        t.run_simulations_cpu_only(16);
        assert!(t.root_visits() > 0, "cpu-only sims must accumulate root visits");
    }

    #[test]
    fn quiescence_fire_count_starts_zero() {
        let t = PyMCTSTree::new(1.5, 1.0, 0.25, true, 0.3);
        assert_eq!(t.get_quiescence_fire_count(), 0);
    }

    /// Numpy-free GIL round-trip: `select_leaves` caches the leaf boards for the
    /// ls path, and the GIL-release `expand_and_backup` accumulates root visits.
    /// (The numpy-marshaling legs — get_policy/get_improved_policy — are pinned by
    /// the Python-side O20 tests, post-ASM.)
    #[test]
    fn select_and_expand_round_trip_under_gil() {
        Python::initialize();
        Python::attach(|py| {
            let mut t = PyMCTSTree::new(1.5, 1.0, 0.25, false, 0.3);
            let board = PyBoard::new();
            t.new_game(&board);
            let leaves = t.select_leaves(py, 1).expect("select");
            assert_eq!(leaves.len(), 1);
            assert_eq!(t.pending_boards.len(), 1, "bridge caches leaf boards for the ls path");
            // n_actions for a v6 board = 19*19+1 = 362.
            let policies = vec![vec![1.0f32 / 362.0; 362]];
            let values = vec![0.0f32];
            t.expand_and_backup(py, policies, values).expect("expand");
            assert!(t.root_visits() >= 1);
        });
    }
}
