// Exceeds the 300-line soft cap: expansion (dense + legal-set pick_topk),
// quiescence, finish_expansion, backup, and the pool-overflow path port together.
//! Expansion and backup for the MCTS tree.

use super::node::{CachedPolicy, Node};
use super::{MCTSTree, MAX_CHILDREN_PER_NODE};
use crate::legal_set::LegalSetPolicy;
use fxhash::FxHashSet;
use mantis_core::board::{Board, WIN_LENGTH};
use std::sync::atomic::{AtomicU64, Ordering};

/// One expansion's Top-K pick: the children, whether the cap truncated, and the PRIOR MASS the
/// truncation dropped. The mass is RETURNED rather than recorded into a static, so its
/// measurement window has one owner instead of every search in the process.
pub(crate) struct TopKPick {
    pub children: Vec<((i32, i32), f32)>,
    /// Read by the pickers' own oracles only: a truncation of zero-prior children costs nothing
    /// and this bool cannot say so, so production asks the counters.
    #[allow(dead_code)]
    pub truncated: bool,
    pub dropped_prior_mass: f32,
}

/// Process-wide counter of pool-overflow events. Reads 0 in production; non-zero means a
/// hand-crafted small pool or a config outside the design envelope. On overflow the expansion
/// PANICS rather than fabricate a terminal value, incrementing this first for attribution.
pub static POOL_OVERFLOW_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read-and-reset the global overflow counter atomically, returning the previous value.
pub fn take_pool_overflow_count() -> u64 {
    POOL_OVERFLOW_COUNT.swap(0, Ordering::Relaxed)
}

/// Read the current overflow count without resetting, for a running tally.
pub fn pool_overflow_count() -> u64 {
    POOL_OVERFLOW_COUNT.load(Ordering::Relaxed)
}

/// The OMITTED PRIOR MASS the Top-K cap has dropped, and how many expansions dropped any, in
/// fixed point (x 1e6) because there is no atomic f32 and a cross-thread float sum would not be
/// reproducible. MASS AND NOT A COUNT: `topk_truncated` is true on essentially every ply at
/// radius 8 (measured: the legal set exceeds 192 on 98% of plies even under clustered play), so
/// what decides whether the cap costs anything is how much PRIOR the dropped moves held.
pub static OMITTED_PRIOR_MASS_MICROS: AtomicU64 = AtomicU64::new(0);
pub static OMITTED_PRIOR_EXPANSIONS: AtomicU64 = AtomicU64::new(0);
pub static TOTAL_EXPANSIONS: AtomicU64 = AtomicU64::new(0);

/// `(omitted_mass_micros, expansions_that_omitted, total_expansions)`, read without reset.
pub fn omitted_prior_stats() -> (u64, u64, u64) {
    (
        OMITTED_PRIOR_MASS_MICROS.load(Ordering::Relaxed),
        OMITTED_PRIOR_EXPANSIONS.load(Ordering::Relaxed),
        TOTAL_EXPANSIONS.load(Ordering::Relaxed),
    )
}

/// Read-and-reset all three, for a bracketed measurement window.
pub fn take_omitted_prior_stats() -> (u64, u64, u64) {
    (
        OMITTED_PRIOR_MASS_MICROS.swap(0, Ordering::Relaxed),
        OMITTED_PRIOR_EXPANSIONS.swap(0, Ordering::Relaxed),
        TOTAL_EXPANSIONS.swap(0, Ordering::Relaxed),
    )
}

/// Fixed-point encoding of a prior mass, shared by the per-tree and process-wide counters.
#[inline]
#[must_use]
pub(crate) fn mass_micros(mass: f32) -> u64 {
    (f64::from(mass) * 1e6) as u64
}

/// Accumulate one expansion's dropped prior into the PROCESS-WIDE totals, called only from
/// `MCTSTree::record_omitted_prior` so an expansion cannot be in one set and missed in the other.
#[inline]
fn record_omitted_prior_global(dropped_mass: f32) {
    TOTAL_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
    if dropped_mass > 0.0 {
        OMITTED_PRIOR_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
        OMITTED_PRIOR_MASS_MICROS.fetch_add(mass_micros(dropped_mass), Ordering::Relaxed);
    }
}

/// Pick up to `MAX_CHILDREN_PER_NODE` children for a leaf expansion, returning `chosen` and
/// whether the Top-K cap truncated.
///
/// Children are ALWAYS ordered by `(prior desc, window_flat_idx asc)`, independent of
/// `FxHashSet` iteration order — a hashbrown table-layout artifact that leaked into
/// `pick_best_puct`'s "first equal score wins" tie-breaking, where a capacity-reserve once
/// shifted search silently (mcts_mean_depth 3.4 -> 2.5 from the bootstrap anchor).
///
/// Out-of-window cells get sort prior `0.0` on the slow path and the `1/n_ch` fallback on the
/// fast one. `trunk_sz`/`half` are the NN-input frame geometry: 19 / 9, or 25 / 12.
#[inline]
pub(crate) fn pick_topk_children(
    legal_moves: &FxHashSet<(i32, i32)>,
    cq: i32,
    cr: i32,
    policy: &[f32],
    trunk_sz: i32,
    half: i32,
    cap: usize,
) -> TopKPick {
    let n_legal = legal_moves.len();
    let n_ch = n_legal.min(cap);

    // One canonical path for every node size: the previous `n_legal <= K` fast path emitted
    // children in raw hash order. O(K log K) is negligible beside the per-leaf NN forward.
    let mut all: Vec<((i32, i32), f32, usize, u32)> = legal_moves
        .iter()
        .map(|&(q, r)| {
            let flat = Board::window_flat_idx_at_geom(q, r, cq, cr, trunk_sz, half);
            let sort_prior = if flat < policy.len() {
                policy[flat]
            } else {
                0.0
            };
            // The packed (q, r) key is the FINAL tie-break: `flat` is `usize::MAX` for EVERY
            // off-window cell, so `sort_unstable` would leave two of them unordered.
            let key = (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF);
            ((q, r), sort_prior, flat, key)
        })
        .collect();

    all.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    let dropped_prior_mass: f32 = all
        .iter()
        .skip(cap)
        .map(|&(_, sort_prior, _, _)| sort_prior)
        .sum();
    all.truncate(cap);

    let mut chosen: Vec<((i32, i32), f32)> = Vec::with_capacity(all.len());
    for ((q, r), _sort_prior, flat, _key) in all {
        let prior = if flat < policy.len() {
            policy[flat]
        } else {
            1.0 / n_ch as f32
        };
        chosen.push(((q, r), prior));
    }

    TopKPick {
        children: chosen,
        truncated: n_legal > cap,
        dropped_prior_mass,
    }
}

/// Legal-set counterpart of `pick_topk_children`: priors are read from the ragged `ls` BY COORD
/// (in-window from `ls.dense`, covered off-window from `ls.overflow`, else the `1/n_ch` floor)
/// and truncated by TRUE prior, tie-broken on the packed (q,r) key. `cq`/`cr` are the leaf's
/// global window centre, the one `ls.dense` was indexed with.
#[inline]
pub(crate) fn pick_topk_children_ls(
    legal_moves: &FxHashSet<(i32, i32)>,
    cq: i32,
    cr: i32,
    ls: &LegalSetPolicy,
    trunk_sz: i32,
    half: i32,
    cap: usize,
) -> TopKPick {
    let n_legal = legal_moves.len();
    let n_ch = n_legal.min(cap);
    let floor = 1.0 / n_ch as f32;

    let mut all: Vec<((i32, i32), f32, u32)> = legal_moves
        .iter()
        .map(|&(q, r)| {
            let prior = ls.get(q, r, cq, cr, trunk_sz, half, floor);
            // packed (q,r) — unique, total-orderable, deterministic tiebreak.
            let key = (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF);
            ((q, r), prior, key)
        })
        .collect();

    all.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });
    let dropped_prior_mass: f32 = all.iter().skip(cap).map(|&(_, prior, _)| prior).sum();
    all.truncate(cap);

    let chosen: Vec<((i32, i32), f32)> = all
        .into_iter()
        .map(|((q, r), prior, _)| ((q, r), prior))
        .collect();
    TopKPick {
        children: chosen,
        truncated: n_legal > cap,
        dropped_prior_mass,
    }
}

/// The per-search omitted-prior counters, one set per `MCTSTree`, in the statics' fixed point.
#[derive(Debug, Default)]
pub struct OmittedPriorStats {
    mass_micros: AtomicU64,
    omitting_expansions: AtomicU64,
    total_expansions: AtomicU64,
}

impl OmittedPriorStats {
    /// `(omitted_mass_micros, expansions_that_omitted, total_expansions)`, without reset.
    #[must_use]
    pub fn read(&self) -> (u64, u64, u64) {
        (
            self.mass_micros.load(Ordering::Relaxed),
            self.omitting_expansions.load(Ordering::Relaxed),
            self.total_expansions.load(Ordering::Relaxed),
        )
    }

    /// Read-and-reset all three — the measurement bracket.
    pub fn take(&self) -> (u64, u64, u64) {
        (
            self.mass_micros.swap(0, Ordering::Relaxed),
            self.omitting_expansions.swap(0, Ordering::Relaxed),
            self.total_expansions.swap(0, Ordering::Relaxed),
        )
    }

    /// Zero all three, for a lifecycle boundary that is not a measurement.
    pub fn reset(&self) {
        self.mass_micros.store(0, Ordering::Relaxed);
        self.omitting_expansions.store(0, Ordering::Relaxed);
        self.total_expansions.store(0, Ordering::Relaxed);
    }

    fn record(&self, dropped_mass: f32) {
        self.total_expansions.fetch_add(1, Ordering::Relaxed);
        if dropped_mass > 0.0 {
            self.omitting_expansions.fetch_add(1, Ordering::Relaxed);
            self.mass_micros
                .fetch_add(mass_micros(dropped_mass), Ordering::Relaxed);
        }
    }
}

impl MCTSTree {
    /// Count one expansion's dropped prior into both this search's counters and the totals.
    pub(crate) fn record_omitted_prior(&self, dropped_mass: f32) {
        self.omitted_prior.record(dropped_mass);
        record_omitted_prior_global(dropped_mass);
    }

    /// Apply quiescence correction to a NN value at a non-terminal leaf.
    ///
    /// Each turn places 2 stones, so the opponent blocks at most 2 winning cells per response:
    /// ≥3 winning moves is a forced win (+1.0), ≥3 for the opponent a forced loss (-1.0), and
    /// the unproven 2-move case blends toward the boundary by `quiescence_blend_2`. Value
    /// correction ONLY — the NN policy still drives expansion.
    #[inline]
    pub(crate) fn apply_quiescence(&self, board: &Board, value: f32) -> f32 {
        if !self.quiescence_enabled {
            return value;
        }

        // Tier 1 (free): the ply gate. P1 first reaches 5 stones at ply 8 and P2 at ply 9, so
        // below 8 half-moves `count_winning_moves` is necessarily 0.
        if board.ply.index() < 8 {
            return value;
        }

        // Tier 2: a winning move needs ≥5 consecutive stones, so skip the O(legal_moves) count.
        let current_player = board.current_player;
        let opponent = current_player.other();
        // A winning move needs a run of WIN_LENGTH - 1 (C1: no bare 5).
        let current_may_threat = board.has_player_long_run(current_player, WIN_LENGTH - 1);
        let opponent_may_threat = board.has_player_long_run(opponent, WIN_LENGTH - 1);

        if !current_may_threat && !opponent_may_threat {
            return value;
        }

        // One `fetch_add` at function end rather than four sites; no reader loads it mid-call.
        let mut fired: u64 = 0;
        let current_wins = if current_may_threat {
            board.count_winning_moves(current_player)
        } else {
            0
        };
        let result = if current_wins >= 3 {
            fired = 1;
            1.0
        } else {
            let opponent_wins = if opponent_may_threat {
                board.count_winning_moves(opponent)
            } else {
                0
            };
            if opponent_wins >= 3 {
                fired = 1;
                -1.0
            } else if current_wins == 2 {
                fired = 1;
                (value + self.quiescence_blend_2).min(1.0)
            } else if opponent_wins == 2 {
                fired = 1;
                (value - self.quiescence_blend_2).max(-1.0)
            } else {
                value
            }
        };
        if fired > 0 {
            self.quiescence_fire_count
                .fetch_add(fired, std::sync::atomic::Ordering::Relaxed);
        }
        result
    }

    /// Children `leaf_idx` may expand: the dialect's root cap at the root, per-node elsewhere.
    #[inline]
    fn expansion_cap(&self, leaf_idx: u32) -> usize {
        if leaf_idx == 0 {
            self.root_children_cap
        } else {
            MAX_CHILDREN_PER_NODE
        }
    }

    /// Expand a single leaf node and backup its value.
    pub(crate) fn expand_and_backup_single(
        &mut self,
        leaf_idx: u32,
        board: &Board,
        policy: &[f32],
        value: f32,
    ) {
        if self.pool[leaf_idx as usize].is_terminal {
            let tv = self.pool[leaf_idx as usize].terminal_value;
            self.backup(leaf_idx, tv);
            return;
        }
        if self.pool[leaf_idx as usize].is_expanded() {
            // TT-hit: already expanded, but still quiesce so repeated TT values are corrected.
            let corrected = self.apply_quiescence(board, value);
            self.backup(leaf_idx, corrected);
            return;
        }

        if board.check_win() {
            // CF-1: the terminal sign comes from the leaf's side-to-move — `mr==1` means the
            // winner is still to move (+1.0), `mr==2` that the player flipped to the loser.
            let tv = if board.moves_remaining == 1 {
                1.0
            } else {
                -1.0
            };
            self.pool[leaf_idx as usize].is_terminal = true;
            self.pool[leaf_idx as usize].terminal_value = tv;
            self.backup(leaf_idx, tv);
            return;
        }

        let legal_moves = board.legal_moves_set();
        if legal_moves.is_empty() {
            self.pool[leaf_idx as usize].is_terminal = true;
            self.pool[leaf_idx as usize].terminal_value = 0.0;
            self.backup(leaf_idx, 0.0);
            return;
        }

        // Top-K cap on leaf children; `trunk_sz` is Board's cached `cluster_window_size`.
        let (cq, cr) = board.window_center();
        let trunk_sz = board.cluster_window_size() as i32;
        let half = (trunk_sz - 1) / 2;
        // The ROOT's cap is the dialect's; `leaf_idx == 0` IS the root, since slot 0 is never
        // reallocated.
        let cap = self.expansion_cap(leaf_idx);
        let pick = pick_topk_children(legal_moves, cq, cr, policy, trunk_sz, half, cap);
        self.record_omitted_prior(pick.dropped_prior_mass);
        self.finish_expansion(leaf_idx, board, pick.children, value);
    }

    /// Shared tail of `expand_and_backup_single`[`_ls`]: materialise the children, quiesce and
    /// backup. Representation-agnostic, so both paths share it.
    fn finish_expansion(
        &mut self,
        leaf_idx: u32,
        board: &Board,
        chosen: Vec<((i32, i32), f32)>,
        value: f32,
    ) {
        let n_ch = chosen.len();
        let first_child = self.next_free;

        if first_child as usize + n_ch > self.pool.len() {
            // Unreachable in production: count for attribution, then panic — never fabricate a
            // terminal value, which silently corrupts training targets.
            POOL_OVERFLOW_COUNT.fetch_add(1, Ordering::Relaxed);
            panic!(
                "MCTS pool overflow: next_free={} n_ch={} pool_len={} K={}. \
                 Pool sizing assumption violated — increase MAX_NODES or \
                 reduce n_simulations × leaf_batch.",
                first_child,
                n_ch,
                self.pool.len(),
                MAX_CHILDREN_PER_NODE
            );
        }
        self.next_free += n_ch as u32;

        let leaf_mr = self.pool[leaf_idx as usize].moves_remaining;
        let child_mr: u8 = if leaf_mr == 1 { 2 } else { 1 };

        self.pool[leaf_idx as usize].first_child = first_child;
        self.pool[leaf_idx as usize].n_children = n_ch as u16;

        for (j, &((q, r), prior)) in chosen.iter().enumerate() {
            let ci = first_child as usize + j;
            let action_encoded = (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF);

            self.pool[ci] = Node {
                parent: leaf_idx,
                action_idx: action_encoded,
                n_visits: 0,
                w_value: 0.0,
                prior,
                first_child: u32::MAX,
                n_children: 0,
                moves_remaining: child_mr,
                is_terminal: false,
                terminal_value: 0.0,
                virtual_loss_count: 0,
            };
        }

        let corrected = self.apply_quiescence(board, value);
        // Mctx's `raw_values[node]`, captured HERE because `backup` folds it into the mean.
        if let Some(slot) = self.raw_values.get_mut(leaf_idx as usize) {
            *slot = corrected;
        }
        self.backup(leaf_idx, corrected);
    }

    /// Legal-set counterpart of `expand_and_backup_single`: identical pre-checks, priors from
    /// the ragged `ls` by coord.
    pub(crate) fn expand_and_backup_single_ls(
        &mut self,
        leaf_idx: u32,
        board: &Board,
        ls: &LegalSetPolicy,
        value: f32,
    ) {
        // Board-frame variant: read the priors in the frame the producer indexed `ls.dense` with.
        let (cq, cr) = board.window_center();
        let trunk_sz = board.cluster_window_size() as i32;
        self.expand_and_backup_single_ls_framed(leaf_idx, board, ls, value, cq, cr, trunk_sz);
    }

    /// Frame-explicit `expand_and_backup_single_ls`: the caller supplies the window centre and
    /// `trunk_sz` that `ls.dense` was BAKED against, so the read frame is the SAME object the
    /// slots were baked with rather than a coincident re-derivation.
    #[allow(clippy::too_many_arguments)] // frame (cq, cr, trunk_sz) is passed by value on the expand path (a struct bundle would re-pack per leaf)
    pub(crate) fn expand_and_backup_single_ls_framed(
        &mut self,
        leaf_idx: u32,
        board: &Board,
        ls: &LegalSetPolicy,
        value: f32,
        cq: i32,
        cr: i32,
        trunk_sz: i32,
    ) {
        if self.pool[leaf_idx as usize].is_terminal {
            let tv = self.pool[leaf_idx as usize].terminal_value;
            self.backup(leaf_idx, tv);
            return;
        }
        if self.pool[leaf_idx as usize].is_expanded() {
            let corrected = self.apply_quiescence(board, value);
            self.backup(leaf_idx, corrected);
            return;
        }
        if board.check_win() {
            let tv = if board.moves_remaining == 1 {
                1.0
            } else {
                -1.0
            };
            self.pool[leaf_idx as usize].is_terminal = true;
            self.pool[leaf_idx as usize].terminal_value = tv;
            self.backup(leaf_idx, tv);
            return;
        }
        let legal_moves = board.legal_moves_set();
        if legal_moves.is_empty() {
            self.pool[leaf_idx as usize].is_terminal = true;
            self.pool[leaf_idx as usize].terminal_value = 0.0;
            self.backup(leaf_idx, 0.0);
            return;
        }
        let half = (trunk_sz - 1) / 2;
        let cap = self.expansion_cap(leaf_idx);
        let pick = pick_topk_children_ls(legal_moves, cq, cr, ls, trunk_sz, half, cap);
        self.record_omitted_prior(pick.dropped_prior_mass);
        self.finish_expansion(leaf_idx, board, pick.children, value);
    }

    /// Expand all pending leaves and backup values to the root.
    pub fn expand_and_backup(&mut self, policies: &[Vec<f32>], values: &[f32]) {
        // `pending` owns the leaf `Board`, so each leaf board is consumed without a re-walk.
        let pending: Vec<(u32, Board)> = std::mem::take(&mut self.pending);
        let n = pending.len().min(policies.len()).min(values.len());
        // `pending` is already `mem::take`n, so any leaf past `n` is DROPPED carrying the
        // virtual loss `select_one_leaf` added. Returning it degrades the BATCH, not the TREE.
        for (leaf_idx, _board) in &pending[n..] {
            self.undo_virtual_loss(*leaf_idx);
        }

        // One `Arc<Vec<f32>>` per first-touch insertion, cheaper than a per-hit clone.
        for i in 0..n {
            let (leaf_idx, board) = &pending[i];
            let policy = &policies[i];
            let value = values[i];

            self.transposition_table.insert(
                board.zobrist_hash,
                super::node::TTEntry {
                    policy: CachedPolicy::Dense(std::sync::Arc::new(policy.clone())),
                    value,
                },
            );

            self.expand_and_backup_single(*leaf_idx, board, policy, value);
        }
    }

    /// Legal-set counterpart of `expand_and_backup`, caching the ragged policy in the TT.
    pub fn expand_and_backup_ls(&mut self, policies: &[LegalSetPolicy], values: &[f32]) {
        let pending: Vec<(u32, Board)> = std::mem::take(&mut self.pending);
        let n = pending.len().min(policies.len()).min(values.len());
        // `pending` is already `mem::take`n, so any leaf past `n` is DROPPED carrying the
        // virtual loss `select_one_leaf` added. Returning it degrades the BATCH, not the TREE.
        for (leaf_idx, _board) in &pending[n..] {
            self.undo_virtual_loss(*leaf_idx);
        }
        for i in 0..n {
            let (leaf_idx, board) = &pending[i];
            let ls = &policies[i];
            let value = values[i];

            self.transposition_table.insert(
                board.zobrist_hash,
                super::node::TTEntry {
                    policy: CachedPolicy::Ls(std::sync::Arc::new(ls.clone())),
                    value,
                },
            );

            self.expand_and_backup_single_ls(*leaf_idx, board, ls, value);
        }
    }

    /// Frame-explicit `expand_and_backup_ls` for the graph seam: `centers[i]` is the builder's
    /// centre that `policies[i]` baked against, and must line up with `self.pending` order.
    pub fn expand_and_backup_ls_at(
        &mut self,
        policies: &[LegalSetPolicy],
        values: &[f32],
        centers: &[(i32, i32)],
        trunk_sz: i32,
    ) {
        let pending: Vec<(u32, Board)> = std::mem::take(&mut self.pending);
        let n = pending
            .len()
            .min(policies.len())
            .min(values.len())
            .min(centers.len());
        for i in 0..n {
            let (leaf_idx, board) = &pending[i];
            let ls = &policies[i];
            let value = values[i];
            let (cq, cr) = centers[i];
            // Guard the centre invariant the TT-hit re-read path also relies on.
            debug_assert_eq!(
                board.window_center(),
                (cq, cr),
                "builder window_center != Board::window_center (coord/slot drift)"
            );

            self.transposition_table.insert(
                board.zobrist_hash,
                super::node::TTEntry {
                    policy: CachedPolicy::Ls(std::sync::Arc::new(ls.clone())),
                    value,
                },
            );

            self.expand_and_backup_single_ls_framed(*leaf_idx, board, ls, value, cq, cr, trunk_sz);
        }
    }

    /// Propagate `value` from `node_idx` to the root (negamax with VL reversal).
    pub(crate) fn backup(&mut self, mut node_idx: u32, mut value: f32) {
        loop {
            let node = &mut self.pool[node_idx as usize];
            node.n_visits += 1;
            node.w_value += value;
            if node.virtual_loss_count > 0 {
                node.virtual_loss_count -= 1;
            }

            let parent = node.parent;
            if parent == u32::MAX {
                break;
            }
            if self.pool[parent as usize].moves_remaining == 1 {
                value = -value;
            }
            node_idx = parent;
        }
    }
}

#[cfg(test)]
mod ls_prior_tests {
    //! pick_topk_children_ls reads priors by coord (in-window dense / off-window overflow).
    use super::*;

    #[test]
    fn test_pick_topk_children_ls_reads_dense_and_overflow() {
        // In-window cells read ls.dense, off-window (28,0) reads ls.overflow, sorted by prior.
        let mut legal: FxHashSet<(i32, i32)> = FxHashSet::default();
        legal.insert((0, 0)); // wq=9,wr=9 → flat 9*19+9 = 180 (in-window)
        legal.insert((1, 0)); // wq=10,wr=9 → flat 10*19+9 = 199 (in-window)
        legal.insert((28, 0)); // wq=37 ≥ 19 → off-window (usize::MAX)

        let mut dense = vec![0.0f32; 19 * 19 + 1];
        dense[180] = 0.2;
        dense[199] = 0.3;
        let mut overflow: fxhash::FxHashMap<(i32, i32), f32> = fxhash::FxHashMap::default();
        overflow.insert((28, 0), 0.5);
        let ls = LegalSetPolicy { dense, overflow };

        let pick = pick_topk_children_ls(&legal, 0, 0, &ls, 19, 9, MAX_CHILDREN_PER_NODE);
        let chosen = pick.children;
        assert!(!pick.truncated);
        assert_eq!(chosen.len(), 3);
        // sorted by prior desc: (28,0)=0.5 (overflow), (1,0)=0.3, (0,0)=0.2 (dense)
        assert_eq!(
            chosen[0],
            ((28, 0), 0.5),
            "off-window prior read from overflow, ranks first"
        );
        assert_eq!(chosen[1], ((1, 0), 0.3));
        assert_eq!(chosen[2], ((0, 0), 0.2));
    }
}
