//! A driver-sized pool: `with_pool` allocates what it is asked, `new_full` stays self-play's `MAX_NODES`.

use mantis_core::board::{Board, BOARD_SIZE};
use mantis_search::{
    armed_sims_for, pool_overflow_count, MCTSTree, MAX_ARMED_SIMS, MAX_CHILDREN_PER_NODE,
    MAX_NODES, VIRTUAL_LOSS_PENALTY,
};

#[test]
fn new_full_keeps_self_plays_pool_and_with_pool_takes_its_own() {
    // PLANTED BREAK: any other size in `new_full` moves self-play's pool and reds this.
    assert_eq!(
        MCTSTree::new_full(1.5, VIRTUAL_LOSS_PENALTY, 0.25)
            .pool
            .len(),
        MAX_NODES
    );
    assert_eq!(
        MCTSTree::with_pool(1.5, VIRTUAL_LOSS_PENALTY, 0.25, 4096 * 32)
            .pool
            .len(),
        4096 * 32
    );
    assert_eq!(MAX_ARMED_SIMS, armed_sims_for(MAX_NODES));
    assert_eq!(
        MAX_ARMED_SIMS, 976,
        "self-play's armed ceiling keeps its value"
    );
    assert_eq!(armed_sims_for(4 * MAX_CHILDREN_PER_NODE * 32), 32);
}

#[test]
fn a_pool_sized_for_its_budget_serves_that_budget_without_overflow() {
    let sims = 32;
    let mut tree = MCTSTree::with_pool(
        1.5,
        VIRTUAL_LOSS_PENALTY,
        0.25,
        4 * MAX_CHILDREN_PER_NODE * sims,
    );
    let uniform = vec![1.0_f32 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
    let before = pool_overflow_count();
    let mut board = Board::new();
    for _ in 0..12 {
        tree.new_game(board.clone());
        let mut done = 0;
        while done < sims {
            let leaves = tree
                .select_leaves(8.min(sims - done))
                .expect("no desync in this fixture");
            if leaves.is_empty() {
                break;
            }
            tree.expand_and_backup(
                &vec![uniform.clone(); leaves.len()],
                &vec![0.0; leaves.len()],
            );
            done += leaves.len();
        }
        let (q, r) = tree
            .get_top_visits(1)
            .first()
            .expect("a searched root has a child")
            .0;
        board
            .apply_move(q, r)
            .expect("the most visited child is legal");
    }
    assert_eq!(
        pool_overflow_count(),
        before,
        "the sized pool never overflowed"
    );
}
