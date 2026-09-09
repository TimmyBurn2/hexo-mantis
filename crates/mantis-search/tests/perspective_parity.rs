//! Regression tests for the child-Q perspective flip in get_improved_policy,
//! root_completed_qvalues, and get_top_visits.
//!
//! Each node stores w_value in its own player-to-move perspective (backup.rs
//! negamax). When root has moves_remaining==1, children belong to the opponent,
//! so their w_value must be negated before use as Q targets.
//! puct_score already handled this; three sibling sites did not, inverting
//! training targets at ~50% of positions.

use mantis_core::board::{Board, BOARD_SIZE};
use mantis_search::MCTSTree;

/// Build a tree with one visited child (n_visits=1, w_value=child_value).
/// The board determines root.moves_remaining.
/// Returns (tree, flat_action_idx_of_visited_child).
fn build_tree_visit_one_child(board: Board, child_value: f32) -> (MCTSTree, usize) {
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];

    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board);

    // Sim 1: expand root (leaf = root itself; value 0.0 so root.w_value starts neutral)
    let _leaves = tree
        .select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    tree.expand_and_backup(std::slice::from_ref(&uniform), &[0.0]);

    // Sim 2: descend to one child, expand it, backup child_value
    let _leaves = tree
        .select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    tree.expand_and_backup(std::slice::from_ref(&uniform), &[child_value]);

    // find the visited child via get_top_visits
    let top = tree.get_top_visits(1);
    assert_eq!(top.len(), 1, "exactly one child should have visits");
    let ((q, r), visits, _prior, _q) = &top[0];
    assert_eq!(*visits, 1, "visited child should have n_visits=1");

    // coord is a `(i32, i32)` axial tuple — no string parsing.
    let flat = tree.root_board.window_flat_idx(*q, *r);
    assert!(flat < n_actions, "visited coord must be in window");

    (tree, flat)
}

mod perspective_parity {
    use super::*;

    /// Compare log(policy[flat]) between mr=2 and mr=1 trees.
    /// Both visited children have w_value=+0.8.
    /// After the fix: mr=2 assigns higher log-probability to its visited child
    /// than mr=1 does (because mr=1 must negate the opponent-perspective Q).
    #[test]
    fn test_improved_policy_flips_q_at_intermediate_ply() {
        // mr=2: apply one move to Board::new() (moves_remaining 1→2)
        let mut board_mr2 = Board::new();
        board_mr2
            .apply_move(0, 0)
            .expect("(0,0) must be legal on fresh board");
        assert_eq!(board_mr2.moves_remaining, 2);

        let board_mr1 = Board::new(); // moves_remaining==1
        assert_eq!(board_mr1.moves_remaining, 1);

        let (tree_mr2, flat_mr2) = build_tree_visit_one_child(board_mr2, 0.8);
        let (tree_mr1, flat_mr1) = build_tree_visit_one_child(board_mr1, 0.8);

        let c_visit = 50.0_f32;
        let c_scale = 1.0_f32;

        // get_improved_policy takes n_actions (= policy_stride). A 19-window
        // encoding has a pass slot, so n_actions = bs²+1.
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let policy_mr2 = tree_mr2.get_improved_policy(n_actions, c_visit, c_scale);
        let policy_mr1 = tree_mr1.get_improved_policy(n_actions, c_visit, c_scale);

        // log(policy) ≈ logit - log(Z). Visited-child logit differs by ±σ*q.
        // After fix: diff >> 2*0.8*c_scale*max_n (actual ≈ 20 at c_visit=50).
        let log_p_mr2 = policy_mr2[flat_mr2].ln();
        let log_p_mr1 = policy_mr1[flat_mr1].ln();
        let diff = log_p_mr2 - log_p_mr1;

        let min_bound = 2.0 * 0.8 * c_scale * 1.0_f32; // max_n=1
        assert!(
            diff > min_bound,
            "log_policy(mr2)={log_p_mr2:.4} − log_policy(mr1)={log_p_mr1:.4} = {diff:.4}; expected > {min_bound}"
        );
    }

    /// `root_completed_qvalues` must negate the child Q when root_mr==1.
    ///
    /// REPLACES the deleted `GumbelSearchState::score` arm of this file. The legacy
    /// dialect's per-candidate score is gone; the surviving surface that reads a child's
    /// `w_value` and has to put it in ROOT perspective is the completion, which both the
    /// root selector and the exported target run through. Same property, live subject.
    #[test]
    fn test_completed_qvalues_flip_at_intermediate_ply() {
        // mr==2: the children are the root's own, no negation.
        let mut board_mr2 = Board::new();
        board_mr2
            .apply_move(0, 0)
            .expect("(0,0) must be legal on fresh board");
        assert_eq!(board_mr2.moves_remaining, 2);
        let (tree_mr2, _flat2) = build_tree_visit_one_child(board_mr2, 0.8);

        // mr==1: the children belong to the opponent, so +0.8 must read as −0.8.
        let board_mr1 = Board::new();
        assert_eq!(board_mr1.moves_remaining, 1);
        let (tree_mr1, _flat1) = build_tree_visit_one_child(board_mr1, 0.8);

        // The completion min-max rescales, so compare the VISITED child's rank rather
        // than its raw value: at mr==2 the visited child holds the max completed value,
        // at mr==1 the minimum. A missing flip puts it at the same end of both.
        let rank = |tree: &MCTSTree| -> (usize, usize) {
            let completed = tree.root_completed_qvalues(50.0, 0.1);
            let visited = tree
                .get_top_visits(1)
                .first()
                .map(|&(coord, _, _, _)| coord)
                .expect("one visited child");
            let info = tree.get_root_children_info();
            let idx = info
                .iter()
                .position(|&(pool_idx, _)| {
                    let val = tree.pool[pool_idx as usize].action_idx;
                    let q = (val >> 16) as i32 - 32768;
                    let r = (val & 0xFFFF) as i32 - 32768;
                    (q, r) == visited
                })
                .expect("the visited child is one of the root children");
            let strictly_above = completed.iter().filter(|&&v| v > completed[idx]).count();
            (strictly_above, completed.len())
        };

        let (above_mr2, n2) = rank(&tree_mr2);
        let (above_mr1, n1) = rank(&tree_mr1);
        assert_eq!(
            above_mr2, 0,
            "at mr=2 the +0.8 child must top the completion"
        );
        assert!(
            above_mr1 > 0,
            "at mr=1 the +0.8 child belongs to the OPPONENT and must NOT top the \
             completion ({above_mr1} of {n1} above it; mr=2 had {above_mr2} of {n2})"
        );
    }

    /// get_top_visits must return Q in root's perspective.
    /// root.moves_remaining==1, child w_value=+0.8 → returned q_value must be -0.8.
    #[test]
    fn test_get_top_visits_returns_root_perspective_q() {
        let board = Board::new(); // moves_remaining==1
        assert_eq!(board.moves_remaining, 1);

        let (tree, _flat) = build_tree_visit_one_child(board, 0.8);

        let top = tree.get_top_visits(1);
        assert_eq!(top.len(), 1);
        let (_coord, visits, _prior, q_returned) = top[0];

        assert_eq!(visits, 1);
        assert!(
            (q_returned - (-0.8_f32)).abs() < 1e-5,
            "root-perspective q should be −0.8, got {q_returned}"
        );
    }
}
