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
use pyo3::types::{PyBool, PyDict};

use mantis_core::board::BOARD_SIZE;
use mantis_core::Board;
use mantis_search::mcts::{
    AuditConfig, AuditMode, ForcedSelectionError, TacticsConfig, TacticsError,
};
use mantis_search::{
    LegalSetPolicy, MCTSTree, MctxRootState, QSigma, SearchKind, VIRTUAL_LOSS_PENALTY,
};

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
/// `select_leaves(n)` followed by `expand_and_backup(policies, values)`, then a policy read.
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
    /// The config's `selfplay.mcts` constants, all required (`mantis.config.resolve.puct` reads
    /// them): `fpu_reduction` is the KataGo dynamic FPU base, an unvisited child's FPU being
    /// `parent_q - fpu_reduction * sqrt(explored_mass)`; the virtual loss is self-play's constant.
    #[new]
    #[pyo3(signature = (*, c_puct, fpu_reduction, quiescence_enabled, quiescence_blend_2))]
    pub fn new(
        c_puct: f32,
        fpu_reduction: f32,
        quiescence_enabled: bool,
        quiescence_blend_2: f32,
    ) -> Self {
        let mut inner = MCTSTree::new_full(c_puct, VIRTUAL_LOSS_PENALTY, fpu_reduction);
        inner.configure_quiescence(quiescence_enabled, quiescence_blend_2);
        PyMCTSTree {
            inner,
            board_size: BOARD_SIZE,
            pending_boards: Vec::new(),
            forced_root_child: None,
            gumbel_root: None,
        }
    }

    /// Score pending children in the choosing parent's frame (the deploy head); self-play's trees keep the old one.
    pub fn configure_pending_loss_frame(&mut self, chooser: bool) {
        self.inner.configure_pending_loss_frame(chooser);
    }

    #[getter]
    pub fn pending_loss_chooser_frame(&self) -> bool {
        self.inner.pending_loss_chooser_frame()
    }

    /// Select the search kind and σ once per player, through the SAME setter the self-play
    /// worker calls; the root calls below read the σ set here rather than taking their own.
    ///
    /// # Errors
    /// `ValueError` — `kind` is not a search kind this build knows. REFUSED, never defaulted.
    pub fn configure_search(
        &mut self,
        kind: &str,
        c_visit: f32,
        c_scale: f32,
        q_rescale: bool,
    ) -> PyResult<()> {
        let parsed = SearchKind::from_config_str(kind).ok_or_else(|| {
            PyValueError::new_err(format!(
                "search.kind={kind:?} is not a known search kind (expected \"puct\" or \
                 \"gumbel\")"
            ))
        })?;
        self.inner.configure_search(
            parsed,
            QSigma {
                c_visit,
                c_scale,
                rescale: q_rescale,
            },
        );
        self.gumbel_root = None;
        Ok(())
    }

    /// The search kind this tree runs, as its config spelling.
    #[getter]
    pub fn search_kind(&self) -> &'static str {
        self.inner.search_kind().as_config_str()
    }

    /// The σ this tree runs, as `(c_visit, c_scale, q_rescale)` — readable so a head can be
    /// checked against the run's config rather than trusted.
    #[getter]
    pub fn search_sigma(&self) -> (f32, f32, bool) {
        let sigma = self.inner.q_sigma();
        (sigma.c_visit, sigma.c_scale, sigma.rescale)
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
    pub fn gumbel_root_select(&self) -> PyResult<Option<u32>> {
        let state = self.gumbel_root.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("gumbel_root_select: call gumbel_root_begin first")
        })?;
        Ok(state.select(&self.inner, self.inner.q_sigma()))
    }

    /// Sequential Halving's answer: the highest-scoring of the MOST-VISITED root children,
    /// as an axial `(q, r)`. `None` when the root has no children.
    ///
    /// # Errors
    /// `RuntimeError` — no root state has been drawn for this search.
    pub fn gumbel_root_best_move(&self) -> PyResult<Option<(i32, i32)>> {
        let state = self.gumbel_root.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("gumbel_root_best_move: call gumbel_root_begin first")
        })?;
        Ok(state
            .best_action(&self.inner, self.inner.q_sigma())
            .map(|pool_idx| self.inner.pool[pool_idx as usize].cell()))
    }

    /// Total quiescence value overrides/blends since last `new_game()`.
    #[getter]
    pub fn get_quiescence_fire_count(&self) -> u64 {
        self.inner
            .quiescence_fire_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset the tree for `board`, re-using the pool. Raises: ValueError when armed tactics meet a radius below 5.
    pub fn new_game(&mut self, board: &PyBoard) -> PyResult<()> {
        self.inner
            .check_tactics_board(board.inner_ref())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.board_size = BOARD_SIZE;
        self.pending_boards.clear();
        self.forced_root_child = None;
        self.gumbel_root = None;
        self.inner.new_game(board.inner_ref().clone());
        Ok(())
    }

    /// Arm the resolved tactics block or disarm with None. Raises: TypeError (not a dict), ValueError (a bad key or leaf).
    pub fn configure_tactics(&mut self, block: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let config = block.map(tactics_config_of).transpose()?;
        self.inner.configure_tactics(config);
        Ok(())
    }

    /// Whether a tactics block is armed.
    #[getter]
    pub fn tactics_armed(&self) -> bool {
        self.inner.tactics_config().is_some()
    }

    /// Descents the last `select_leaves` / `select_leaves_forced` call backed up inline, no board returned.
    pub fn last_inline_descents(&self) -> usize {
        self.inner.last_inline_descents()
    }

    /// Descents the last `select_leaves` call backed up from the TT, no board returned.
    pub fn last_tt_hits(&self) -> usize {
        self.inner.last_tt_hits()
    }

    /// This search's tactics rows by name; every row 0 with tactics off.
    pub fn tactics_counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, value) in self.inner.tactics_counters().rows() {
            d.set_item(name, value)?;
        }
        Ok(d)
    }

    /// Before the search: the decided root stone (an illegal one counted), or None to search. Raises: RuntimeError.
    pub fn root_offence(&mut self, py: Python<'_>) -> PyResult<Option<(i32, i32)>> {
        match py.detach(|| self.inner.root_offence()) {
            Ok(stone) => Ok(stone),
            Err(TacticsError::ProofStoneIllegal { .. }) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// After the search: `chosen`, or the armed audit's substitute, walking this kind's own ranking of the root.
    pub fn root_audit(&mut self, py: Python<'_>, chosen: (i32, i32)) -> (i32, i32) {
        let order = self.inner.audit_order(self.gumbel_root.as_ref());
        py.detach(|| self.inner.root_audit(chosen, order.as_deref()))
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

    /// `select_leaves` until `network` net leaves are queued or `descents` spent. Raises: SelectionDesync, as it.
    pub fn select_leaves_filled(
        &mut self,
        py: Python<'_>,
        network: usize,
        descents: usize,
    ) -> PyResult<Vec<Py<PyBoard>>> {
        let boards = py
            .detach(|| self.inner.select_leaves_filled(network, descents))
            .map_err(|desync| SelectionDesync::new_err(desync.to_string()))?;
        self.pending_boards = boards.clone();
        boards
            .into_iter()
            .map(|b| Py::new(py, PyBoard::from_inner(b)))
            .collect()
    }

    /// This search's PUCT select rows: `(calls, calls ended at an overlap, network leaves queued)`.
    pub fn select_counters(&self) -> (u64, u64, u64) {
        let c = self.inner.select_counters();
        (c.calls, c.overlaps, c.network_leaves)
    }

    /// One leaf per FORCED root child with no transposition fast path (an expanded leaf is
    /// returned and re-backed-up). Raises ValueError for a child the root does not own.
    pub fn select_leaves_forced(
        &mut self,
        py: Python<'_>,
        children: Vec<u32>,
    ) -> PyResult<Vec<Py<PyBoard>>> {
        let boards = py
            .detach(|| self.inner.select_leaves_forced(&children))
            .map_err(|err| match err {
                ForcedSelectionError::OutOfRange(err) => PyValueError::new_err(err.to_string()),
                ForcedSelectionError::Desync(err) => SelectionDesync::new_err(err.to_string()),
            })?;
        self.forced_root_child = None;
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

    /// Total visit count at the root (= number of simulations run).
    pub fn root_visits(&self) -> u32 {
        self.inner.root_visits()
    }

    /// Reset the tree to its root state (for benchmarking / reuse).
    pub fn reset(&mut self) {
        self.forced_root_child = None;
        self.inner.reset();
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
                let (aq, ar) = child.cell();
                ((aq, ar), pool_idx, prior, visits, q_value)
            })
            .collect()
    }

    /// Improved policy targets from Gumbel completed Q-values (Danihelka et al., ICLR 2022),
    /// for the policy viewer's Gumbel-mode overlay, under the σ `configure_search` set.
    #[pyo3(signature = (board_size = None))]
    pub fn get_improved_policy<'py>(
        &self,
        py: Python<'py>,
        board_size: Option<usize>,
    ) -> Bound<'py, PyArray1<f32>> {
        let bs = board_size.unwrap_or(self.board_size);
        let n_actions = bs * bs + 1;
        self.inner
            .get_improved_policy(n_actions, self.inner.q_sigma())
            .into_pyarray(py)
    }
}

/// The one tactics kind this build runs: the turn-level strictly forcing solver.
const TACTICS_KIND: &str = "strict_turn";

/// Refuse a dict whose keys are not exactly `want`, naming the difference.
fn exact_keys(d: &Bound<'_, PyDict>, what: &str, want: &[&str]) -> PyResult<()> {
    let mut got: Vec<String> = d
        .keys()
        .iter()
        .map(|k| {
            k.extract::<String>()
                .map_err(|_| PyValueError::new_err(format!("{what}: a key is not a string")))
        })
        .collect::<PyResult<_>>()?;
    got.sort();
    let mut expected: Vec<String> = want.iter().map(|s| (*s).to_string()).collect();
    expected.sort();
    if got != expected {
        return Err(PyValueError::new_err(format!(
            "{what}: keys {got:?}, expected exactly {expected:?}"
        )));
    }
    Ok(())
}

/// An integer leaf of `d` within `[lo, hi]`, refused by name otherwise.
fn int_leaf(d: &Bound<'_, PyDict>, what: &str, key: &str, lo: i64, hi: i64) -> PyResult<i64> {
    let value = d
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("{what}: {key} is missing")))?;
    // A bool is an int to Python; a block that says `True` for a count is refused, not read as 1.
    let n: i64 = (!value.is_instance_of::<PyBool>())
        .then(|| value.extract().ok())
        .flatten()
        .ok_or_else(|| PyValueError::new_err(format!("{what}: {key} is not an integer")))?;
    if !(lo..=hi).contains(&n) {
        return Err(PyValueError::new_err(format!(
            "{what}: {key}={n} is outside [{lo}, {hi}]"
        )));
    }
    Ok(n)
}

/// A string leaf of `d`, refused by name when absent or not a string.
fn str_leaf(d: &Bound<'_, PyDict>, what: &str, key: &str) -> PyResult<String> {
    d.get_item(key)?
        .and_then(|v| v.extract::<String>().ok())
        .ok_or_else(|| PyValueError::new_err(format!("{what}: {key} is not a string")))
}

/// The block's `kind`, `leaf_*`, `root_*` and `audit` (None, or `turns nodes k m total_nodes mode`), each checked.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // each leaf is range-checked first
pub(crate) fn tactics_config_of(d: &Bound<'_, PyDict>) -> PyResult<TacticsConfig> {
    let what = "tactics block";
    exact_keys(
        d,
        what,
        &[
            "kind",
            "leaf_turns",
            "leaf_nodes",
            "root_turns",
            "root_nodes",
            "audit",
        ],
    )?;
    let kind = str_leaf(d, what, "kind")?;
    if kind != TACTICS_KIND {
        return Err(PyValueError::new_err(format!(
            "{what}: kind {kind:?} is not one this build runs ({TACTICS_KIND:?})"
        )));
    }
    let audit = match d.get_item("audit")? {
        Some(a) if !a.is_none() => {
            let a = a
                .cast_into::<PyDict>()
                .map_err(|_| PyValueError::new_err(format!("{what}: audit is not a dict")))?;
            let what = "tactics audit";
            exact_keys(
                &a,
                what,
                &["turns", "nodes", "k", "m", "total_nodes", "mode"],
            )?;
            let mode = str_leaf(&a, what, "mode")?;
            let mode = match mode.as_str() {
                "hold" => AuditMode::Hold,
                "inverted" => AuditMode::Inverted,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "{what}: mode {other:?} is neither \"hold\" nor \"inverted\""
                    )))
                }
            };
            Some(AuditConfig {
                turns: int_leaf(&a, what, "turns", 1, 40)? as u8,
                nodes: int_leaf(&a, what, "nodes", 1, i64::MAX)? as u64,
                k: int_leaf(&a, what, "k", 1, 1024)? as u32,
                m: int_leaf(&a, what, "m", 1, 1024)? as u32,
                total_nodes: int_leaf(&a, what, "total_nodes", 1, i64::MAX)? as u64,
                mode,
            })
        }
        _ => None,
    };
    Ok(TacticsConfig {
        leaf_turns: int_leaf(d, what, "leaf_turns", 1, 40)? as u8,
        leaf_nodes: int_leaf(d, what, "leaf_nodes", 0, i64::MAX)? as u64,
        root_turns: int_leaf(d, what, "root_turns", 1, 40)? as u8,
        root_nodes: int_leaf(d, what, "root_nodes", 0, i64::MAX)? as u64,
        audit,
    })
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
        let mut t = PyMCTSTree::new(1.5, 0.25, true, 0.3);
        assert_eq!(t.forced_root_child(), None);
        assert!(
            t.set_forced_root_child(Some(7)).is_err(),
            "an unexpanded root owns no child 7"
        );

        let seed = PyBoard::new();
        t.new_game(&seed).expect("an unarmed tree takes any board");
        Python::initialize();
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
        t.new_game(&board).expect("an unarmed tree takes any board");
        assert_eq!(t.forced_root_child(), None);
    }

    #[test]
    fn quiescence_fire_count_starts_zero() {
        let t = PyMCTSTree::new(1.5, 0.25, true, 0.3);
        assert_eq!(t.get_quiescence_fire_count(), 0);
    }

    /// Numpy-free GIL round-trip: `select_leaves` caches the leaf boards for the ls path, and
    /// the GIL-release `expand_and_backup` accumulates root visits.
    #[test]
    fn select_and_expand_round_trip_under_gil() {
        Python::initialize();
        Python::attach(|py| {
            let mut t = PyMCTSTree::new(1.5, 0.25, false, 0.3);
            let board = PyBoard::new();
            t.new_game(&board).expect("an unarmed tree takes any board");
            let leaves = t.select_leaves(py, 1).expect("select");
            assert_eq!(leaves.len(), 1);
            assert_eq!(
                t.pending_boards.len(),
                1,
                "bridge caches leaf boards for the ls path"
            );
            // n_actions for the 19×19 window = 19*19+1 = 362.
            let policies = vec![vec![1.0f32 / 362.0; 362]];
            let values = vec![0.0f32];
            t.expand_and_backup(py, policies, values).expect("expand");
            assert!(t.root_visits() >= 1);
        });
    }
}
