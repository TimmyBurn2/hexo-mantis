//! R8 justify: PUCT descent and the two errors it can produce are one unit. `SelectionDesync`
//! and `ForcedChildOutOfRange` describe states only this traversal can reach, and an error type
//! declared elsewhere drifts from the code that raises it.
//! PUCT selection and tree traversal.

use super::{CachedPolicy, MCTSTree};
use fxhash::FxHashSet;
use mantis_core::board::{Board, MoveDiff};

/// The selected child's stored `action_idx` decoded to a cell the board cannot play.
///
/// `apply_move` errs only on OCCUPANCY, so this means the tree and the board have
/// desynchronised. An `Err` rather than a panic, so every caller latches a named reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionDesync {
    /// Pool index of the node whose child was selected.
    pub node: u32,
    /// The decoded axial column the board refused.
    pub q: i32,
    /// The decoded axial row the board refused.
    pub r: i32,
}

impl std::fmt::Display for SelectionDesync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SelectionDesync: the PUCT descent selected a child of node {} whose action_idx \
             decodes to ({}, {}), which the board refuses. `Board::apply_move` errs only on \
             occupancy, so the tree and the board have desynchronised — the search cannot be \
             continued or exported as if it had run.",
            self.node, self.q, self.r
        )
    }
}

impl std::error::Error for SelectionDesync {}

/// A forced root-child index the root does not own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForcedChildOutOfRange {
    /// The index that was offered.
    pub child: u32,
    /// The root's first child slot.
    pub first_child: u32,
    /// How many children the root has.
    pub n_children: u16,
}

impl std::fmt::Display for ForcedChildOutOfRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ForcedChildOutOfRange: {} is not a child of the root, whose {} child slot(s) \
             start at {}. Forcing a foreign index descends into a node the root does not own \
             — an uninitialised slot decodes to the cell (32767, 32767), which an unbounded \
             board accepts, so the search would proceed into a subtree belonging to nothing.",
            self.child, self.n_children, self.first_child
        )
    }
}

impl std::error::Error for ForcedChildOutOfRange {}

/// Why `select_leaves_forced` refused: a foreign forced index, or a desynchronised descent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedSelectionError {
    /// A forced index the root does not own; refused before any descent touches the tree.
    OutOfRange(ForcedChildOutOfRange),
    /// A descent selected a child the board refuses.
    Desync(SelectionDesync),
}

impl std::fmt::Display for ForcedSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfRange(err) => err.fmt(f),
            Self::Desync(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for ForcedSelectionError {}

impl From<SelectionDesync> for ForcedSelectionError {
    fn from(err: SelectionDesync) -> Self {
        Self::Desync(err)
    }
}

/// Single-pass argmax over `[first..first+n_ch)` by PUCT score, each child scored once and
/// `parent_n.sqrt()` once per level; a NaN score never displaces the running best.
#[inline]
fn pick_best_puct(
    tree: &MCTSTree,
    first: usize,
    n_ch: usize,
    parent_idx: u32,
    parent_n: f32,
    fpu_value: f32,
) -> u32 {
    debug_assert!(n_ch > 0, "pick_best_puct called on a node with no children");
    // Strict `>` matches `max_by` Greater semantics: the first equal score wins and a NaN
    // comparison preserves the running best.
    let sqrt_parent_n = parent_n.sqrt();
    let mut best_idx: u32 = first as u32;
    let mut best_score: f32 = tree.puct_score(best_idx, parent_idx, sqrt_parent_n, fpu_value);
    for i in (first + 1)..(first + n_ch) {
        let score = tree.puct_score(i as u32, parent_idx, sqrt_parent_n, fpu_value);
        // Strict `>` matches `max_by` Greater semantics: first equal score
        // wins, NaN comparisons preserve the running best (Equal fallback).
        if score
            .partial_cmp(&best_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            == std::cmp::Ordering::Greater
        {
            best_idx = i as u32;
            best_score = score;
        }
    }
    best_idx
}

impl MCTSTree {
    /// PUCT score for `child_idx`, from `parent_idx`'s player perspective.
    ///
    /// `fpu_value` is the pre-computed first-play urgency for unvisited children, ignored for
    /// visited ones. `sqrt_parent_n` is `parent.n_visits + parent.virtual_loss_count` already
    /// passed through `sqrt`, hoisted out of the per-child caller loop.
    #[inline]
    pub(crate) fn puct_score(
        &self,
        child_idx: u32,
        parent_idx: u32,
        sqrt_parent_n: f32,
        fpu_value: f32,
    ) -> f32 {
        let child = &self.pool[child_idx as usize];
        let parent = &self.pool[parent_idx as usize];

        let q = if child.n_visits == 0 && child.virtual_loss_count == 0 {
            // Unvisited: `fpu_value` is computed from the parent's own Q, so it is ALREADY in the
            // parent's to-move perspective and, unlike a visited child's Q, is never negated.
            fpu_value
        } else if parent.moves_remaining == 1 {
            -child.q_value_vl(self.virtual_loss)
        } else {
            child.q_value_vl(self.virtual_loss)
        };

        let u = self.c_puct * child.prior * sqrt_parent_n
            / (1.0 + child.n_visits as f32 + child.virtual_loss_count as f32);
        q + u
    }

    /// Walk the tree via PUCT until an unexpanded (or terminal) leaf is found, applying virtual
    /// loss to every node on the path. Returns `(leaf_node_index, leaf_depth)`.
    ///
    /// # Errors
    /// `SelectionDesync` — the tree and the board disagree about what has been played.
    pub(crate) fn select_one_leaf(
        &mut self,
        board: &mut Board,
        diffs: &mut Vec<MoveDiff>,
    ) -> Result<(u32, u32), SelectionDesync> {
        let mut cur: u32 = 0;
        let mut depth = 0;
        loop {
            self.pool[cur as usize].virtual_loss_count += 1;

            let node = &self.pool[cur as usize];
            if node.is_terminal || !node.is_expanded() {
                return Ok((cur, depth));
            }

            let parent_n = (node.n_visits + node.virtual_loss_count) as f32;
            let first = node.first_child as usize;
            let n_ch = node.n_children as usize;

            // KataGo-style dynamic FPU for unvisited children: `parent_q - fpu_reduction *
            // sqrt(sum of visited children's priors)`, which collapses to 0.0 at `fpu_reduction == 0.0`.
            let fpu_value = if self.fpu_reduction > 0.0 {
                let parent_q = if node.n_visits > 0 {
                    node.w_value / node.n_visits as f32
                } else {
                    0.0
                };
                let explored_mass: f32 = (first..first + n_ch)
                    .filter(|&i| {
                        let c = &self.pool[i];
                        c.n_visits > 0 || c.virtual_loss_count > 0
                    })
                    .map(|i| self.pool[i].prior)
                    .sum();
                // §F1: `parent_q - fpu_reduction * sqrt(mass)` → fused FMA.
                (-self.fpu_reduction).mul_add(explored_mass.sqrt(), parent_q)
            } else {
                0.0
            };

            // At the root a set `forced_root_child` (Sequential Halving) is descended directly;
            // otherwise the root selects by PUCT.
            let best = if cur == 0 {
                if let Some(forced) = self.forced_root_child {
                    forced
                } else {
                    pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value)
                }
            } else if self.kind == crate::mcts::SearchKind::Gumbel {
                // Gumbel interior: improved policy with the visit-count correction. `None` only on
                // a childless node, excluded above, so the PUCT fallback is unreachable.
                self.pick_best_mctx_interior(cur)
                    .unwrap_or_else(|| pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value))
            } else {
                pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value)
            };

            let (q, r) = self.pool[best as usize].cell();

            let diff = board
                .apply_move_tracked(q, r)
                .map_err(|_| SelectionDesync { node: cur, q, r })?;
            diffs.push(diff);

            cur = best;
            depth += 1;
        }
    }

    /// Mctx `gumbel_muzero_interior_action_selection`: pick the child maximising
    /// `softmax(log_prior + completed_q) - visits / (1 + sum_visits)`. The subtracted term makes
    /// repeated argmaxes APPROXIMATE the improved policy's visitation frequencies rather than
    /// piling every visit on one child; without it the interior of the tree stops sampling.
    /// Returns `None` when the node has no children. Allocates `completed_q` per call.
    #[allow(clippy::cast_possible_truncation)] // j indexes children, itself a u16 count
    pub(crate) fn pick_best_mctx_interior(&self, node_idx: u32) -> Option<u32> {
        let completed = self.node_completed_qvalues(node_idx, self.q_sigma);
        if completed.is_empty() {
            return None;
        }
        let node = &self.pool[node_idx as usize];
        let first = node.first_child as usize;
        let n = completed.len();
        let priors: Vec<f32> = (0..n).map(|j| self.pool[first + j].prior).collect();
        let visits: Vec<u32> = (0..n).map(|j| self.pool[first + j].n_visits).collect();
        let scores = super::completed_q::mctx_interior_argmax_input(&priors, &completed, &visits);

        let mut best: Option<(usize, f32)> = None;
        for (j, &score) in scores.iter().enumerate() {
            // Strict `>` — the first of equal scores wins, matching `argmax`.
            if best.is_none_or(|(_, b)| score > b) {
                best = Some((j, score));
            }
        }
        best.map(|(j, _)| (first + j) as u32)
    }

    /// Select up to `n` distinct leaves for evaluation.
    ///
    /// # Errors
    /// `SelectionDesync` — a selected child's `action_idx` decodes to a cell the board refuses.
    /// Virtual loss applied on the failing descent is UNWOUND before returning, so a caller that
    /// recovers does not leave the tree permanently penalising the path it walked.
    pub fn select_leaves(&mut self, n: usize) -> Result<Vec<Board>, SelectionDesync> {
        self.pending.clear();
        let mut boards = Vec::with_capacity(n);
        // O(1) overlap dedup on leaf pool indices; the set lives only for this call.
        let mut pending_ids: FxHashSet<u32> = FxHashSet::default();
        pending_ids.reserve(n);
        let mut board = self.root_board.clone();
        let mut diffs: Vec<MoveDiff> = Vec::with_capacity(32);

        let mut i = 0;
        let mut attempts = 0;
        let max_attempts = n * 4;

        while i < n && attempts < max_attempts {
            attempts += 1;
            diffs.clear();
            let (leaf_idx, leaf_depth) = match self.select_one_leaf(&mut board, &mut diffs) {
                Ok(pair) => pair,
                Err(desync) => {
                    // Unwind before propagating, or the path keeps a virtual loss for a walk that
                    // produced no leaf; the board rewinds too, so `root_board` holds for a retry.
                    self.undo_virtual_loss(desync.node);
                    while let Some(diff) = diffs.pop() {
                        board.undo_move(diff);
                    }
                    return Err(desync);
                }
            };
            self.depth_accum += leaf_depth as u64;
            self.sim_count += 1;

            if pending_ids.contains(&leaf_idx) {
                self.undo_virtual_loss(leaf_idx);
                while let Some(diff) = diffs.pop() {
                    board.undo_move(diff);
                }
                continue;
            }

            // An `Arc` refcount bump, not a 1448 B copy; the TT borrow drops before the
            // `&mut self` expand. Dispatch on the dense vs ragged legal-set variant.
            let cached = self
                .transposition_table
                .get(&board.zobrist_hash)
                .map(|e| (e.policy.clone(), e.value));
            if let Some((policy, value)) = cached {
                match policy {
                    CachedPolicy::Dense(p) => {
                        self.expand_and_backup_single(leaf_idx, &board, &p, value)
                    }
                    CachedPolicy::Ls(ls) => {
                        self.expand_and_backup_single_ls(leaf_idx, &board, &ls, value)
                    }
                }
                while let Some(diff) = diffs.pop() {
                    board.undo_move(diff);
                }
                continue;
            }

            // `pending` owns the fully-replayed leaf `Board`, so expansion never re-replays from
            // `root_board`; this sibling of the NN-input board skips the `legal_cache` copy.
            boards.push(board.clone());
            self.pending.push((leaf_idx, board.clone()));
            pending_ids.insert(leaf_idx);

            while let Some(diff) = diffs.pop() {
                board.undo_move(diff);
            }
            i += 1;
        }

        debug_assert_eq!(board.zobrist_hash, self.root_board.zobrist_hash);
        debug_assert_eq!(board.ply, self.root_board.ply);

        Ok(boards)
    }

    /// Select ONE leaf under each of `forced` root children, in one call.
    ///
    /// THE GUMBEL ANALOGUE OF `leaf_batch_size`. Sequential Halving visits every candidate at the
    /// current considered level before the level advances, so the SET of root children a round
    /// touches is fixed when the round starts: issuing their descents together visits exactly the
    /// children a one-at-a-time loop would, leaving the root visit counts identical. It buys one
    /// inference round trip per round.
    ///
    /// NO VIRTUAL LOSS IS NEEDED ACROSS CANDIDATES: each forced child roots a DISJOINT subtree.
    /// A forced child yielding an already-pending leaf is SKIPPED as an overlap, since that can
    /// only happen through a transposition. `forced_root_child` is CLEARED on every exit.
    ///
    /// # Errors
    /// `ForcedSelectionError::OutOfRange` — an index the root does not own, checked for EVERY entry
    /// before any descent; `::Desync` — a selected child's cell is one the board refuses.
    pub fn select_leaves_forced(
        &mut self,
        forced: &[u32],
    ) -> Result<Vec<Board>, ForcedSelectionError> {
        for &child in forced {
            if let Err(e) = self.check_forced_root_child(child) {
                self.forced_root_child = None;
                return Err(ForcedSelectionError::OutOfRange(e));
            }
        }
        self.pending.clear();
        let mut boards = Vec::with_capacity(forced.len());
        let mut pending_ids: FxHashSet<u32> = FxHashSet::default();
        pending_ids.reserve(forced.len());
        let mut board = self.root_board.clone();
        let mut diffs: Vec<MoveDiff> = Vec::with_capacity(32);

        for &child in forced {
            self.forced_root_child = Some(child);
            diffs.clear();
            let (leaf_idx, leaf_depth) = match self.select_one_leaf(&mut board, &mut diffs) {
                Ok(pair) => pair,
                Err(desync) => {
                    // Unwind this descent before propagating, exactly as `select_leaves` does: a
                    // virtual loss left applied would permanently penalise nodes for a leafless walk.
                    self.undo_virtual_loss(desync.node);
                    while let Some(diff) = diffs.pop() {
                        board.undo_move(diff);
                    }
                    self.forced_root_child = None;
                    return Err(desync.into());
                }
            };
            self.depth_accum += leaf_depth as u64;
            self.sim_count += 1;

            if pending_ids.contains(&leaf_idx) {
                self.undo_virtual_loss(leaf_idx);
                while let Some(diff) = diffs.pop() {
                    board.undo_move(diff);
                }
                continue;
            }

            // No TT fast path here: a Gumbel round needs an exact leaf count per round trip,
            // which `select_leaves`' uncounted TT-hit expansions would break.
            boards.push(board.clone());
            self.pending.push((leaf_idx, board.clone()));
            pending_ids.insert(leaf_idx);

            while let Some(diff) = diffs.pop() {
                board.undo_move(diff);
            }
        }
        self.forced_root_child = None;

        debug_assert_eq!(board.zobrist_hash, self.root_board.zobrist_hash);
        debug_assert_eq!(board.ply, self.root_board.ply);

        Ok(boards)
    }

    /// Reverse virtual loss on all nodes from `node_idx` to the root.
    pub(crate) fn undo_virtual_loss(&mut self, mut node_idx: u32) {
        loop {
            let node = &mut self.pool[node_idx as usize];
            if node.virtual_loss_count > 0 {
                node.virtual_loss_count -= 1;
            }
            let parent = node.parent;
            if parent == u32::MAX {
                break;
            }
            node_idx = parent;
        }
    }
}
