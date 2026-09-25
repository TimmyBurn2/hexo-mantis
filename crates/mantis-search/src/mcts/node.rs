//! Node, TTEntry, and pool constants for the MCTS tree.

use crate::legal_set::LegalSetPolicy;
use std::sync::Arc;

/// Pre-allocated pool size per worker.
///
/// The per-worker footprint is `size_of::<Node>()` times this, plus an `f32` vector of the
/// same length under `SearchKind::Gumbel` (`MCTSTree::raw_values`); host = that × workers.
pub const MAX_NODES: usize = 4_000_000;

/// Virtual-loss penalty applied per unresolved selection.
pub const VIRTUAL_LOSS_PENALTY: f32 = 1.0;

/// Cached MCTS prior for a board state — either the dense scatter_max policy
/// (the existing path, byte-identical) or the ragged legal-set policy. One run
/// uses exactly one variant (selected by the encoding's `policy_pool`), so the
/// TT never mixes them within a tree.
#[derive(Clone)]
pub enum CachedPolicy {
    Dense(Arc<Vec<f32>>),
    Ls(Arc<LegalSetPolicy>),
}

/// Cached Neural Network evaluation for a board state.
///
/// The policy is wrapped in `Arc` so TT-hit reads in `select_leaves` are
/// refcount bumps instead of a 1448 B (362 floats × 4 B) Vec clone per hit.
/// Insertion still allocates a fresh `Arc` once per first-touch.
#[derive(Clone)]
pub struct TTEntry {
    pub policy: CachedPolicy,
    pub value: f32,
}

/// Pack an axial cell into a node's `action_idx`: each coordinate offset by 32768 into one u16 half.
#[inline]
#[must_use]
pub fn pack_cell(q: i32, r: i32) -> u32 {
    (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF)
}

/// One node in the MCTS tree.
#[derive(Clone, Copy)]
pub struct Node {
    pub parent: u32,
    pub action_idx: u32,
    pub n_visits: u32,
    pub w_value: f32,
    pub prior: f32,
    pub first_child: u32,
    pub n_children: u16,
    pub moves_remaining: u8,
    pub is_terminal: bool,
    pub terminal_value: f32,
    pub virtual_loss_count: u32,
}

impl Node {
    pub fn uninit() -> Self {
        Node {
            parent: u32::MAX,
            action_idx: u32::MAX,
            n_visits: 0,
            w_value: 0.0,
            prior: 0.0,
            first_child: u32::MAX,
            n_children: 0,
            moves_remaining: 1,
            is_terminal: false,
            terminal_value: 0.0,
            virtual_loss_count: 0,
        }
    }

    /// Mean value Q(s,a) adjusted for outstanding virtual losses.
    #[inline]
    pub fn q_value_vl(&self, penalty: f32) -> f32 {
        let effective_n = self.n_visits + self.virtual_loss_count;
        if effective_n == 0 {
            0.0
        } else {
            let total_penalty = self.virtual_loss_count as f32 * penalty;
            (self.w_value - total_penalty) / effective_n as f32
        }
    }

    #[inline]
    pub fn is_expanded(&self) -> bool {
        self.first_child != u32::MAX
    }

    /// The axial cell `action_idx` encodes, the inverse of `pack_cell`.
    #[inline]
    #[must_use]
    pub fn cell(&self) -> (i32, i32) {
        (
            (self.action_idx >> 16) as i32 - 32768,
            (self.action_idx & 0xFFFF) as i32 - 32768,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{pack_cell, Node};

    #[test]
    fn pack_cell_round_trips_through_node_cell() {
        for &(q, r) in &[(0, 0), (-1, 1), (28, -9), (-32768, 32767), (32767, -32768)] {
            let node = Node {
                action_idx: pack_cell(q, r),
                ..Node::uninit()
            };
            assert_eq!(node.cell(), (q, r));
        }
    }

    #[test]
    fn pack_cell_keeps_the_offset_layout() {
        // The layout is stored in trees and sort keys, so the bit pattern itself is pinned.
        assert_eq!(pack_cell(0, 0), 0x8000_8000);
        assert_eq!(pack_cell(-32768, -32768), 0);
        assert_eq!(pack_cell(1, -1), 0x8001_7FFF);
    }
}
