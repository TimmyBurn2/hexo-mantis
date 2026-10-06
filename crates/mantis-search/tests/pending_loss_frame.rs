//! The pending-loss frame reaches a batched PUCT search only: one leaf at a time plays alike in either frame.

#[allow(dead_code)]
mod common;

use common::{compact_prior, replay};
use mantis_search::MCTSTree;

/// `sims` PUCT descents in rounds of `batch`, values varying by leaf; the root's children's visits.
fn root_visits(chooser_frame: bool, batch: usize, sims: usize) -> Vec<u32> {
    // Player Two to place its second stone: the root is a last-stone parent.
    let board = replay(&[(0, 0), (1, 0)]);
    let mut tree = MCTSTree::new(1.5);
    tree.configure_quiescence(false, 0.0);
    tree.configure_pending_loss_frame(chooser_frame);
    tree.new_game(board);
    let mut spent = 0usize;
    while spent < sims {
        let leaves = tree
            .select_leaves(batch.min(sims - spent))
            .expect("the board replays every selected cell");
        let counted = leaves.len() + tree.last_inline_descents() + tree.last_tt_hits();
        if !leaves.is_empty() {
            let policies: Vec<Vec<f32>> = leaves.iter().map(compact_prior).collect();
            let values: Vec<f32> = (0..leaves.len())
                .map(|i| 0.4 - 0.1 * ((spent + i) % 9) as f32)
                .collect();
            tree.expand_and_backup(&policies, &values);
        }
        assert!(counted > 0, "a select call spent nothing");
        spent += counted;
    }
    let root = &tree.pool[0];
    (root.first_child..root.first_child + u32::from(root.n_children))
        .map(|i| tree.pool[i as usize].n_visits)
        .collect()
}

#[test]
fn one_leaf_at_a_time_plays_the_same_search_in_either_frame() {
    assert_eq!(root_visits(false, 1, 200), root_visits(true, 1, 200));
}

#[test]
fn a_batched_search_under_a_last_stone_root_reads_the_frame() {
    assert_ne!(
        root_visits(false, 8, 200),
        root_visits(true, 8, 200),
        "the frame never reached the search, so the gate pins nothing"
    );
}
