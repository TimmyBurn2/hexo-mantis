//! R8 justify: PUCT descent and the two errors it can produce are one unit. `SelectionDesync`
//! and `ForcedChildOutOfRange` describe states only this traversal can reach, and their
//! messages explain what the descent did — a reader who hits either needs the walk and the
//! refusal side by side, and an error type declared elsewhere drifts from the code that
//! raises it.
//! PUCT selection and tree traversal.

use mantis_core::board::{Board, MoveDiff};
use fxhash::FxHashSet;
use super::{CachedPolicy, MCTSTree};

/// The selected child's stored `action_idx` decoded to a cell the board cannot play.
///
/// AUDIT-1 F-02. This was `.expect("selected move should always be legal")`, and it FIRED in
/// production: `src/mantis/selfplay/worker.py` carried a `BaseException` handler matching
/// `"cell already occupied"` — the exact text `Board::apply_move` returns — to restart the
/// tree at root whenever it happened. `Board::apply_move` errs only on OCCUPANCY, never on
/// radius, so the panic means a child's `action_idx` decoded to a cell already on the board:
/// the tree and the board have desynchronised.
///
/// It matters that this is an `Err` and not a panic on three counts. On the Rust self-play
/// arm the unwind is caught by `runner::spawn::guard_worker`, which increments
/// `worker_panics` and sets `running = false` — a RUN HALT with no reason in the fatal-defect
/// latch, so R275(b)'s instrument never sees it. On the eval and deploy arms it surfaced as a
/// `PanicException` and was recovered from only by a string match, and only when the batch
/// was larger than one. And a panic crossing the FFI is convertible only because the profile
/// sets `panic = "unwind"` (R2/LAW-13) — a guarantee about the worst case, not a design.
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

/// `set_forced_root_child` was given an index that is not one of the ROOT's children.
///
/// AUDIT-1 F-02, second trigger. See `MCTSTree::set_forced_root_child` for why an unchecked
/// store is two distinct defects rather than one.
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

/// §P8: single-pass argmax over `[first..first+n_ch)` by PUCT score, computing
/// each child's score exactly once. Replaces the prior `.max_by()` closures
/// that re-evaluated `puct_score(a)` and `puct_score(b)` for every comparator
/// pair (`2·(K-1)` scores per descent level for K=192 children). Tie-break
/// follows `partial_cmp(...).unwrap_or(Equal)` semantics — a NaN score never
/// displaces the running best (Equal → keep current).
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
    // §P1: `parent_n.sqrt()` is loop-invariant across all K≤192 children of one
    // node — hoist it out of the per-child `puct_score` call so the sqrtf is
    // evaluated once per descent level, not K times.
    let sqrt_parent_n = parent_n.sqrt();
    let mut best_idx: u32 = first as u32;
    let mut best_score: f32 = tree.puct_score(best_idx, parent_idx, sqrt_parent_n, fpu_value);
    for i in (first + 1)..(first + n_ch) {
        let score = tree.puct_score(i as u32, parent_idx, sqrt_parent_n, fpu_value);
        // Strict `>` matches `max_by` Greater semantics: first equal score
        // wins, NaN comparisons preserve the running best (Equal fallback).
        if score.partial_cmp(&best_score).unwrap_or(std::cmp::Ordering::Equal)
            == std::cmp::Ordering::Greater
        {
            best_idx = i as u32;
            best_score = score;
        }
    }
    best_idx
}

impl MCTSTree {
    /// PUCT score for `child_idx`, evaluated from `parent_idx`'s player perspective.
    ///
    /// `fpu_value`: pre-computed first-play urgency value for unvisited children.
    /// For visited children this is ignored — their actual Q is used instead.
    ///
    /// `sqrt_parent_n`: `parent.n_visits + parent.virtual_loss_count` already
    /// passed through `sqrt`. §P1 hoists this loop-invariant out of the
    /// per-child caller loop so it is computed once per descent level.
    #[inline]
    pub(crate) fn puct_score(&self, child_idx: u32, parent_idx: u32, sqrt_parent_n: f32, fpu_value: f32) -> f32 {
        let child  = &self.pool[child_idx  as usize];
        let parent = &self.pool[parent_idx as usize];

        let q = if child.n_visits == 0 && child.virtual_loss_count == 0 {
            // Unvisited node: use dynamic FPU value.
            // CF-6 (Phase 6): fpu_value is computed by the caller from the
            // parent's own Q (parent.w_value / parent.n_visits), so it is ALREADY
            // in the parent's to-move perspective — the same perspective the
            // visited branches below resolve to. The visited child's stored Q is
            // in the CHILD's perspective and must be negated when the turn flips
            // (parent.moves_remaining == 1); fpu_value needs no such negation
            // because it never left the parent's perspective. Sign-consistent at
            // both mr==1 and mr==2 (verified, no bug). Pinned by the FPU-sign test.
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

    /// Walk the tree via PUCT until an unexpanded (or terminal) leaf is found.
    /// Applies virtual loss to every node on the path.
    /// Returns `(leaf_node_index, leaf_depth)`, or `SelectionDesync` when a selected child's
    /// `action_idx` decodes to a cell the board refuses (AUDIT-1 F-02).
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
                if depth > self.max_depth_observed {
                    self.max_depth_observed = depth;
                }
                return Ok((cur, depth));
            }

            let parent_n = (node.n_visits + node.virtual_loss_count) as f32;
            let first    = node.first_child as usize;
            let n_ch     = node.n_children  as usize;

            // KataGo-style dynamic FPU: value estimate for unvisited children.
            // explored_mass = sum of priors for all children that have been visited.
            // fpu_value = parent_q - fpu_reduction * sqrt(explored_mass)
            // When fpu_reduction == 0.0 this collapses to 0.0 (legacy behaviour).
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

            // Gumbel MCTS root mechanism: at root (cur==0) with a `forced_root_child`
            // set, descend directly to it (Sequential Halving forces sims into a
            // specific candidate subtree). Otherwise — and at every interior node —
            // selection is PUCT (§P8 single-pass argmax). The `forced_root_child`
            // override is Gumbel's ROOT mechanism and is orthogonal to interior
            // selection (which is always PUCT).
            let best = if cur == 0 {
                if let Some(forced) = self.forced_root_child {
                    forced
                } else {
                    pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value)
                }
            } else {
                pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value)
            };

            let val = self.pool[best as usize].action_idx;
            let q = (val >> 16) as i32 - 32768;
            let r = (val & 0xFFFF) as i32 - 32768;

            let diff = board
                .apply_move_tracked(q, r)
                .map_err(|_| SelectionDesync { node: cur, q, r })?;
            diffs.push(diff);

            cur = best;
            depth += 1;
        }
    }

    /// Select up to `n` distinct leaves for evaluation.
    /// # Errors
    /// `SelectionDesync` — a selected child's `action_idx` decodes to a cell the board
    /// refuses. AUDIT-1 F-02: this was an `expect` that fired in production and halted the run
    /// through `guard_worker` with no reason latched. Virtual loss applied on the failing
    /// descent is UNWOUND before returning, so a caller that recovers does not leave the tree
    /// permanently penalising the path it walked.
    pub fn select_leaves(&mut self, n: usize) -> Result<Vec<Board>, SelectionDesync> {
        self.pending.clear();
        let mut boards = Vec::with_capacity(n);
        // §P36: O(1) overlap dedup via FxHashSet<u32> on leaf pool indices.
        // Prior code scanned `self.pending` linearly (`.iter().any(...)`) for
        // every selected leaf — O(N²) in batch size. Set lives only for the
        // duration of this call; capacity matches batch hint to avoid rehash.
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
                    // Unwind this descent before propagating: the nodes on the path already
                    // took their virtual loss, and leaving it applied would permanently
                    // penalise them for a walk that never produced a leaf. The board is
                    // rewound too, so `self.root_board` invariants hold for any retry.
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
                self.selection_overlap_count += 1;
                while let Some(diff) = diffs.pop() {
                    board.undo_move(diff);
                }
                continue;
            }

            // §P7: TT-hit clone of 1448 B policy vector eliminated via
            // `Arc::clone` (refcount bump). `expand_and_backup_single` reads
            // the policy through `&[f32]`, so we dereference the Arc once at
            // the call site. Value is `Copy`. Reading `entry` then dropping
            // the borrow before the `&mut self` call satisfies the borrow
            // checker because `expand_and_backup_single` touches `self.pool`
            // / `self.transposition_table` (insert is a no-op for re-hits)
            // disjointly from the read-only fetch.
            // Clone the cached policy (Arc refcount bump) + value, dropping the
            // immutable TT borrow before the `&mut self` expand call. Dispatch on
            // the dense vs ragged legal-set variant.
            let cached = self
                .transposition_table
                .get(&board.zobrist_hash)
                .map(|e| (e.policy.clone(), e.value));
            if let Some((policy, value)) = cached {
                match policy {
                    CachedPolicy::Dense(p) => self.expand_and_backup_single(leaf_idx, &board, &p, value),
                    CachedPolicy::Ls(ls) => self.expand_and_backup_single_ls(leaf_idx, &board, &ls, value),
                }
                while let Some(diff) = diffs.pop() {
                    board.undo_move(diff);
                }
                continue;
            }

            // §P6+§P9: pending now owns the fully-replayed leaf `Board`
            // instead of a `Vec<MoveDiff>`. `expand_and_backup` no longer
            // clones `root_board` + replays `apply_move(q, r)` per leaf —
            // saving ~depth board mutations per leaf (avg depth ~30 ×
            // leaf_batch=8 = ~240 mutations/sim). `boards.push(board.clone())`
            // still produces the NN-input board; the pending clone is a
            // sibling: `Board::Clone` skips `legal_cache` copy, so the per-leaf
            // cost is small relative to the eliminated re-walk.
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
