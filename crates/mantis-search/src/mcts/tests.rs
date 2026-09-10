// Exceeds the 300-line soft cap: the shared mcts unit suite and its setup helpers port as one
// in-src module because they reach pool/next_free internals.
//
// VERBATIM-ported unit fixtures — the action-index encodings, the range-membership assertion
// and the `&vec![0.0; n]` calls — trip cosmetic style lints that are suppressed below.
#![allow(
    clippy::identity_op,
    clippy::manual_range_contains,
    clippy::useless_vec
)]

use super::*;
use mantis_core::board::{Cell, Player};

pub(super) fn setup_two_child_tree(c_puct: f32) -> (MCTSTree, u32, u32) {
    let mut tree = MCTSTree::new(c_puct);
    tree.pool[0].moves_remaining = 2;

    let first_child = 1u32;
    tree.next_free = 3;
    tree.pool[0].first_child = first_child;
    tree.pool[0].n_children = 2;

    let action_a = ((0u32 + 32768) << 16) | (0u32 + 32768);
    let action_b = ((0u32 + 32768) << 16) | (1u32 + 32768);

    tree.pool[1] = Node {
        parent: 0,
        action_idx: action_a,
        n_visits: 0,
        w_value: 0.0,
        prior: 0.7,
        first_child: u32::MAX,
        n_children: 0,
        moves_remaining: 1,
        is_terminal: false,
        terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.pool[2] = Node {
        parent: 0,
        action_idx: action_b,
        n_visits: 0,
        w_value: 0.0,
        prior: 0.3,
        first_child: u32::MAX,
        n_children: 0,
        moves_remaining: 1,
        is_terminal: false,
        terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    (tree, 1, 2)
}

#[test]
fn test_puct_prefers_higher_prior_when_unvisited() {
    let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
    tree.pool[0].n_visits = 1;
    let score_a = tree.puct_score(child_a, 0, 1.0, 0.0);
    let score_b = tree.puct_score(child_b, 0, 1.0, 0.0);
    assert!(
        score_a > score_b,
        "child with prior 0.7 should score higher than 0.3: {score_a:.4} vs {score_b:.4}"
    );
}

#[test]
fn test_puct_visits_reduce_exploration() {
    let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
    tree.pool[0].n_visits = 100;
    tree.pool[child_a as usize].n_visits = 99;
    tree.pool[child_a as usize].w_value = 0.0;

    // child_a is visited, child_b is not, so `fpu_value` applies: at 0.0 the unvisited child
    // still looks like Q=0 but has the higher U.
    let score_a = tree.puct_score(child_a, 0, 100.0, 0.0);
    let score_b = tree.puct_score(child_b, 0, 100.0, 0.0);
    assert!(
        score_b > score_a,
        "less-visited child_b should be preferred: {score_b:.4} vs {score_a:.4}"
    );
}

#[test]
fn test_backup_single_value_reaches_root() {
    let mut tree = MCTSTree::new(1.5);
    tree.pool[0].moves_remaining = 1;
    tree.pool[1] = Node {
        parent: 0,
        action_idx: (32768u32 << 16) | 32768u32,
        n_visits: 0,
        w_value: 0.0,
        prior: 1.0,
        first_child: u32::MAX,
        n_children: 0,
        moves_remaining: 2,
        is_terminal: false,
        terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.pool[2] = Node {
        parent: 1,
        action_idx: (32768u32 << 16) | 32769u32,
        n_visits: 0,
        w_value: 0.0,
        prior: 1.0,
        first_child: u32::MAX,
        n_children: 0,
        moves_remaining: 1,
        is_terminal: false,
        terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.next_free = 3;

    tree.backup(2, 1.0);

    assert_eq!(tree.pool[2].w_value, 1.0);
    assert_eq!(tree.pool[2].n_visits, 1);
    assert_eq!(tree.pool[1].w_value, 1.0, "child should not flip (mr=2)");
    assert_eq!(tree.pool[1].n_visits, 1);
    assert_eq!(tree.pool[0].w_value, -1.0, "root should flip (mr=1)");
    assert_eq!(tree.pool[0].n_visits, 1);
}

#[test]
fn test_backup_negamax_player_change() {
    let mut tree = MCTSTree::new(1.5);
    tree.pool[0].moves_remaining = 1;
    tree.pool[1] = Node {
        parent: 0,
        action_idx: (32768u32 << 16) | 32768u32,
        n_visits: 0,
        w_value: 0.0,
        prior: 1.0,
        first_child: u32::MAX,
        n_children: 0,
        moves_remaining: 2,
        is_terminal: false,
        terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.next_free = 2;

    tree.backup(1, 0.6);

    assert!((tree.pool[1].w_value - 0.6).abs() < 1e-6, "child w = 0.6");
    assert!(
        (tree.pool[0].w_value - (-0.6)).abs() < 1e-6,
        "root w should be -0.6, got {}",
        tree.pool[0].w_value
    );
}

#[test]
fn test_select_leaves_returns_root_when_empty() {
    let mut tree = MCTSTree::new(1.5);
    let board = Board::new();
    tree.new_game(board.clone());

    let leaves = tree
        .select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    assert_eq!(leaves.len(), 1);
    assert_eq!(leaves[0].ply, board.ply);
    assert_eq!(leaves[0].moves_remaining, board.moves_remaining);
}

#[test]
fn test_expand_and_backup_creates_children() {
    let mut tree = MCTSTree::new(1.5);
    let board = Board::new();
    tree.new_game(board);

    let leaves = tree
        .select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    let n_legal = leaves[0].legal_move_count();

    let policy = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
    tree.expand_and_backup(&[policy], &[0.0]);

    let root = &tree.pool[0];
    assert!(root.is_expanded());
    assert_eq!(root.n_children as usize, n_legal);
    assert_eq!(root.n_visits, 1);
}

#[test]
fn test_full_search_runs_n_simulations() {
    let mut tree = MCTSTree::new(1.5);
    let board = Board::new();
    tree.new_game(board);

    let n_sims = 20;
    let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];

    for _ in 0..n_sims {
        let leaves = tree
            .select_leaves(1)
            .expect("select_leaves: no desync in this fixture");
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values: Vec<f32> = (0..n).map(|_| 0.0).collect();
        tree.expand_and_backup(&policies, &values);
    }

    assert_eq!(tree.root_visits(), n_sims as u32);
}

#[test]
fn test_virtual_loss_applied_during_select() {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());
    let _leaves = tree
        .select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    assert_eq!(tree.pool[0].virtual_loss_count, 1);
}

#[test]
fn test_virtual_loss_reversed_after_backup() {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());

    let leaves = tree
        .select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    let n = leaves.len();
    let uniform = vec![1.0 / (BOARD_SIZE * BOARD_SIZE + 1) as f32; BOARD_SIZE * BOARD_SIZE + 1];
    let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
    tree.expand_and_backup(&policies, &vec![0.0; n]);

    for i in 0..tree.next_free as usize {
        assert_eq!(
            tree.pool[i].virtual_loss_count, 0,
            "node {i} should have virtual_loss_count=0 after backup"
        );
    }
}

#[test]
fn test_virtual_loss_causes_path_divergence() {
    let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
    tree.pool[0].n_visits = 1;

    let batch = tree
        .select_leaves(2)
        .expect("select_leaves: no desync in this fixture");
    assert_eq!(batch.len(), 2);

    let dummy = vec![0.5f32; BOARD_SIZE * BOARD_SIZE + 1];
    let policies: Vec<Vec<f32>> = (0..2).map(|_| dummy.clone()).collect();
    tree.expand_and_backup(&policies, &vec![0.0; 2]);

    assert_eq!(tree.pool[0].virtual_loss_count, 0);
    assert_eq!(tree.pool[child_a as usize].virtual_loss_count, 0);
    assert_eq!(tree.pool[child_b as usize].virtual_loss_count, 0);
}

#[test]
fn test_virtual_loss_q_adjustment() {
    let node = Node {
        parent: u32::MAX,
        action_idx: (32768u32 << 16) | 32768u32,
        n_visits: 4,
        w_value: 2.0,
        prior: 0.5,
        first_child: u32::MAX,
        n_children: 0,
        moves_remaining: 1,
        is_terminal: false,
        terminal_value: 0.0,
        virtual_loss_count: 2,
    };
    let q = node.q_value_vl(VIRTUAL_LOSS_PENALTY);
    assert!(q.abs() < 1e-6, "Q should be 0.0: got {q}");
}

#[test]
fn test_dynamic_fpu_reduces_unvisited_q() {
    let (mut tree, child_a, child_b) = setup_two_child_tree(1.5);
    tree.fpu_reduction = 0.25;
    tree.pool[0].n_visits = 10;
    tree.pool[0].w_value = 3.0; // parent Q = 0.3

    tree.pool[child_a as usize].n_visits = 1;
    tree.pool[child_a as usize].w_value = 0.2;

    // child_b is unvisited, so explored_mass = 0.7, reduction = 0.25*sqrt(0.7) ~ 0.209 and
    // fpu_value = parent_q(0.3) - 0.209 ~ 0.091.
    let explored_mass: f32 = 0.7;
    let expected_fpu = (3.0f32 / 10.0) - 0.25 * explored_mass.sqrt();

    let score_b_fpu = tree.puct_score(child_b, 0, 10.0, expected_fpu);
    let score_b_zero_fpu = tree.puct_score(child_b, 0, 10.0, 0.0);
    // With parent_q > 0 the FPU value sits below 0.3 but above 0.0, so dynamic FPU raises the
    // score against the legacy Q=0 baseline.
    assert!(
        score_b_fpu > score_b_zero_fpu || expected_fpu < 0.0,
        "dynamic FPU should raise unvisited score when parent_q > 0: \
         fpu_score={score_b_fpu:.4} vs zero_fpu={score_b_zero_fpu:.4} (fpu={expected_fpu:.4})"
    );
}

#[test]
fn test_quiescence_overrides_value_for_3_winning_moves() {
    // P1 has >= 3 winning moves, so the value is overridden to 1.0.
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    // Three winning cells for P1: q=-1 and q=5 off (0,0)..(4,0), and q=19 off (20,0)..(24,0).
    let mut stones: Vec<((i32, i32), Cell)> = Vec::new();
    for q in 0..5i32 {
        stones.push(((q, 0), Cell::P1));
    }
    // Block west end of first threat so it has only 1 winning cell
    stones.push(((-1, 0), Cell::P2));
    // Second threat (unblocked both ends → 2 winning cells: q=19 and q=25)
    for q in 20..25i32 {
        stones.push(((q, 0), Cell::P1));
    }
    // ply must be ≥ 8 so the early-game ply gate does not short-circuit.
    let board = Board::from_stones(&stones, Player::One, 1, 20, None);

    let wins = board.count_winning_moves(Player::One);
    assert!(wins >= 3, "expected ≥3 winning moves for P1, got {wins}");

    let corrected = tree.apply_quiescence(&board, 0.0);
    assert_eq!(
        corrected, 1.0,
        "quiescence should override to 1.0 for 3+ winning moves"
    );
}

#[test]
fn test_quiescence_overrides_value_for_3_opponent_winning_moves() {
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    // Current player is P1 but P2 has 3 winning moves, so the value is -1.0.
    let mut stones: Vec<((i32, i32), Cell)> = Vec::new();
    for q in 0..5i32 {
        stones.push(((q, 0), Cell::P2));
    }
    stones.push(((-1, 0), Cell::P1)); // block west end
    for q in 20..25i32 {
        stones.push(((q, 0), Cell::P2));
    }
    // ply must be ≥ 8 so the early-game ply gate does not short-circuit.
    let board = Board::from_stones(&stones, Player::One, 1, 20, None);

    let opp_wins = board.count_winning_moves(Player::Two);
    assert!(
        opp_wins >= 3,
        "expected ≥3 winning moves for P2, got {opp_wins}"
    );

    let corrected = tree.apply_quiescence(&board, 0.0);
    assert_eq!(
        corrected, -1.0,
        "quiescence should override to -1.0 when opponent has 3+ winning moves"
    );
}

#[test]
fn test_quiescence_blend_for_2_winning_moves() {
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    let stones: Vec<((i32, i32), Cell)> = (0..5i32).map(|q| ((q, 0), Cell::P1)).collect();
    // ply must be ≥ 8 so the early-game ply gate does not short-circuit.
    let board = Board::from_stones(&stones, Player::One, 1, 10, None);

    let wins = board.count_winning_moves(Player::One);
    assert_eq!(
        wins, 2,
        "unblocked 5-in-a-row should have exactly 2 winning moves"
    );

    let nn_value = 0.5f32;
    let corrected = tree.apply_quiescence(&board, nn_value);
    let expected = (nn_value + 0.3).min(1.0);
    assert!(
        (corrected - expected).abs() < 1e-6,
        "blend for 2 winning moves: expected {expected}, got {corrected}"
    );
}

#[test]
fn test_quiescence_disabled_does_not_change_value() {
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = false;

    let stones: Vec<((i32, i32), Cell)> = (0..5i32).map(|q| ((q, 0), Cell::P1)).collect();
    let board = Board::from_stones(&stones, Player::One, 1, 0, None);

    let nn_value = 0.42f32;
    let corrected = tree.apply_quiescence(&board, nn_value);
    assert_eq!(
        corrected, nn_value,
        "disabled quiescence must not change value"
    );
}

#[test]
fn test_quiescence_fire_count_increments_and_resets() {
    let mut tree = MCTSTree::new_full(1.5, 0.0, 0.0);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    let mut stones: Vec<((i32, i32), Cell)> = Vec::new();
    for q in 0..5i32 {
        stones.push(((q, 0), Cell::P1));
    }
    stones.push(((-1, 0), Cell::P2)); // block west end of first threat
    for q in 20..25i32 {
        stones.push(((q, 0), Cell::P1));
    }
    let board = Board::from_stones(&stones, Player::One, 1, 20, None);

    let wins = board.count_winning_moves(Player::One);
    assert!(wins >= 3, "expected ≥3 winning moves for P1, got {wins}");

    assert_eq!(tree.quiescence_fire_count.load(Ordering::Relaxed), 0);

    let result = tree.apply_quiescence(&board, 0.5);
    assert_eq!(result, 1.0, "forced win should override to 1.0");
    assert_eq!(
        tree.quiescence_fire_count.load(Ordering::Relaxed),
        1,
        "counter should be 1 after one firing call"
    );

    tree.apply_quiescence(&board, 0.5);
    assert_eq!(
        tree.quiescence_fire_count.load(Ordering::Relaxed),
        2,
        "counter should accumulate across calls"
    );

    tree.new_game(Board::new());
    assert_eq!(
        tree.quiescence_fire_count.load(Ordering::Relaxed),
        0,
        "counter should reset to 0 after new_game()"
    );
}

#[test]
fn test_quiescence_no_override_in_early_game() {
    let mut tree = MCTSTree::new(1.5);
    tree.quiescence_enabled = true;
    tree.quiescence_blend_2 = 0.3;

    let board = Board::new();
    let nn_value = 0.123f32;
    let corrected = tree.apply_quiescence(&board, nn_value);
    assert_eq!(
        corrected, nn_value,
        "early game should not trigger quiescence"
    );
}

#[test]
fn test_no_forced_win_short_circuit_in_expansion() {
    let stones: Vec<((i32, i32), Cell)> = (0..4i32).map(|r| ((0, r), Cell::P1)).collect();
    let board = Board::from_stones(&stones, Player::One, 1, 7, None);

    assert!(
        !board.check_win(),
        "test setup: board should not be a terminal win"
    );

    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board);

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform_policy = vec![1.0 / n_actions as f32; n_actions];
    let nn_value = 0.5;

    tree.expand_and_backup(&[uniform_policy], &[nn_value]);

    let root = &tree.pool[0];
    assert!(
        !root.is_terminal,
        "root must NOT be marked terminal for a forced-win formation"
    );
}

// CF-1: compound-turn terminal sign. The leaf's side-to-move (== `board.moves_remaining`)
// decides the sign of a `check_win` leaf's terminal value, NOT a hardcoded -1.0: a turn-final
// win leaves `mr==2` (loser to move) -> -1.0, while a stone-1 win keeps the player and leaves
// `mr==1` (winner to move) -> +1.0. The pre-fix hardcode scored the stone-1 win as -1.0 and
// dragged its parent's Q toward a loss, so PUCT avoided completing on the first stone. These
// exercise `expand_and_backup_single` on a non-legal-cadence fixture from `Board::from_stones`.

/// Build a P1 6-in-a-row along the E/W axis with `last_move` on the line.
/// `mr`/`player` are set by the caller to model stone-1 vs stone-2 wins.
fn make_stone1_win_board(mr: u8, player: Player) -> Board {
    let six: Vec<((i32, i32), Cell)> = (0..6i32).map(|q| ((q, 0), Cell::P1)).collect();
    let board = Board::from_stones(&six, player, mr, 11, Some((5, 0)));
    assert!(
        board.check_win(),
        "test setup: board must be a terminal win"
    );
    board
}

/// Attach a single leaf child to the root and run its terminal backup.
/// Returns (leaf_terminal_value, parent_backed_up_w).
fn run_terminal_leaf(parent_mr: u8, leaf_mr: u8, board: &Board) -> (f32, f32) {
    let mut tree = MCTSTree::new(1.5);
    tree.pool[0].moves_remaining = parent_mr;
    tree.pool[0].first_child = 1;
    tree.pool[0].n_children = 1;
    tree.next_free = 2;
    tree.pool[1] = Node {
        parent: 0,
        action_idx: (32768u32 << 16) | 32773u32,
        n_visits: 0,
        w_value: 0.0,
        prior: 1.0,
        first_child: u32::MAX,
        n_children: 0,
        moves_remaining: leaf_mr,
        is_terminal: false,
        terminal_value: 0.0,
        virtual_loss_count: 0,
    };
    tree.expand_and_backup_single(1, board, &[], 0.0);
    (tree.pool[1].terminal_value, tree.pool[0].w_value)
}

/// Case A — stone-1 win: leaf `mr==1` from an `mr==2` parent. The terminal value must be +1.0,
/// and since the parent does not flip, the winning child drags its Q toward +1. FAILS on the
/// pre-fix hardcoded -1.0.
#[test]
fn test_cf1_stone1_win_scored_as_win() {
    let board = make_stone1_win_board(1, Player::One);
    let (leaf_tv, parent_w) = run_terminal_leaf(2, 1, &board);
    assert_eq!(
        leaf_tv, 1.0,
        "stone-1 win leaf (mr==1, winner to move) must score +1.0, not -1.0"
    );
    assert_eq!(
        parent_w, 1.0,
        "winning stone-1 child must back up +1.0 to its mr==2 parent \
         (policy target points at the winning move, not filler-first)"
    );
}

/// Case B — stone-2 / turn-final win: leaf `mr==2` from an `mr==1` parent. The terminal value
/// stays -1.0 and the negamax flip turns it into +1.0 for the mover, proving the leaf-side
/// derivation does not regress the case the hardcode handled correctly.
#[test]
fn test_cf1_stone2_win_still_scored_as_loss_to_mover() {
    let board = make_stone1_win_board(2, Player::Two);
    let (leaf_tv, parent_w) = run_terminal_leaf(1, 2, &board);
    assert_eq!(
        leaf_tv, -1.0,
        "turn-final win leaf (mr==2, loser to move) must stay -1.0"
    );
    assert_eq!(
        parent_w, 1.0,
        "negamax flip at the mr==1 parent turns the -1.0 leaf into +1.0 \
         for the player who completed the line"
    );
}

// CF-6: FPU sign consistency, pinning a verified no-bug invariant in `puct_score`. A VISITED
// child's stored Q is mr-negated (at `mr==1` the child is the other player, at `mr==2` the
// same), while an UNVISITED child's `fpu_value` arrives ALREADY in the parent's to-move frame
// and is NEVER negated. `c_puct=0.0` zeroes the U term so `puct_score == q`; flipping either
// expected sign fails.

#[test]
fn test_cf6_fpu_sign_consistent_with_visited_child_at_both_mr() {
    let (mut tree, c1, c2) = setup_two_child_tree(0.0);
    let sqrt_n = 2.0_f32.sqrt();
    const FPU: f32 = -0.3;

    // c1: visited child with a decisive own-frame Q, no virtual loss, so q_value_vl = w/n.
    tree.pool[c1 as usize].n_visits = 4;
    tree.pool[c1 as usize].w_value = 2.0; // own-frame Q = 2.0 / 4 = 0.5
    let own_q = tree.pool[c1 as usize].q_value_vl(tree.virtual_loss);
    assert!((own_q - 0.5).abs() < 1e-6, "precondition: own_q = {own_q}");
    assert_eq!(tree.pool[c2 as usize].n_visits, 0);

    assert_eq!(tree.pool[0].moves_remaining, 2);
    let q_visited_mr2 = tree.puct_score(c1, 0, sqrt_n, FPU);
    assert!(
        (q_visited_mr2 - 0.5).abs() < 1e-6,
        "mr2 visited q = {q_visited_mr2}, want +0.5 (not negated)"
    );
    let q_unvisited_mr2 = tree.puct_score(c2, 0, sqrt_n, FPU);
    assert!(
        (q_unvisited_mr2 - FPU).abs() < 1e-6,
        "mr2 unvisited q = {q_unvisited_mr2}, want fpu_value {FPU}"
    );

    tree.pool[0].moves_remaining = 1;
    let q_visited_mr1 = tree.puct_score(c1, 0, sqrt_n, FPU);
    assert!(
        (q_visited_mr1 - (-0.5)).abs() < 1e-6,
        "mr1 visited q = {q_visited_mr1}, want -0.5 (negated into parent frame)"
    );
    let q_unvisited_mr1 = tree.puct_score(c2, 0, sqrt_n, FPU);
    assert!(
        (q_unvisited_mr1 - FPU).abs() < 1e-6,
        "mr1 unvisited q = {q_unvisited_mr1}, want fpu_value {FPU} (UNCHANGED by mr)"
    );
}

pub(super) fn setup_expanded_root() -> MCTSTree {
    let mut tree = MCTSTree::new(1.5);
    let board = Board::new();
    tree.new_game(board);

    let _leaves = tree
        .select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let policy = vec![1.0 / n_actions as f32; n_actions];
    tree.expand_and_backup(&[policy], &[0.0]);
    tree
}

#[test]
fn test_wp6_driver_setters_roundtrip() {
    // The two narrow public setters for the separate-crate selfplay driver; the in-crate test
    // reads the `pub(crate)` fields directly. The setter validates against the ROOT's child
    // range, so this round-trip needs a root that HAS children.
    let mut tree = setup_expanded_root();
    let first = tree.pool[0].first_child;
    tree.set_forced_root_child(Some(first))
        .expect("the root's own first child is in range");
    assert_eq!(
        tree.forced_root_child,
        Some(first),
        "set_forced_root_child(Some(first_child)) must set the field"
    );
    tree.set_forced_root_child(None)
        .expect("clearing is always in range");
    assert_eq!(
        tree.forced_root_child, None,
        "set_forced_root_child(None) must clear the field"
    );

    tree.configure_quiescence(false, 0.7);
    assert!(
        !tree.quiescence_enabled,
        "configure_quiescence must set enabled=false"
    );
    assert_eq!(
        tree.quiescence_blend_2, 0.7,
        "configure_quiescence must set blend_2=0.7"
    );

    tree.configure_quiescence(true, 0.3);
    assert!(
        tree.quiescence_enabled,
        "configure_quiescence must set enabled=true"
    );
    assert_eq!(
        tree.quiescence_blend_2, 0.3,
        "configure_quiescence must set blend_2=0.3"
    );
}

#[test]
fn test_forced_root_child_selection() {
    let mut tree = setup_expanded_root();
    let root = &tree.pool[0];
    assert!(root.is_expanded());
    let first = root.first_child;
    let n_ch = root.n_children as usize;
    assert!(n_ch >= 2, "need at least 2 children");

    let target_child = first + 1;
    tree.forced_root_child = Some(target_child);

    let n_sims = 10;
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0 / n_actions as f32; n_actions];
    for _ in 0..n_sims {
        let leaves = tree
            .select_leaves(1)
            .expect("select_leaves: no desync in this fixture");
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values = vec![0.0f32; n];
        tree.expand_and_backup(&policies, &values);
    }

    let forced_visits = tree.pool[target_child as usize].n_visits;
    assert!(
        forced_visits >= n_sims as u32 - 1,
        "forced child should have >= {} visits, got {}",
        n_sims - 1,
        forced_visits
    );

    let other_visits = tree.pool[first as usize].n_visits;
    assert_eq!(
        other_visits, 0,
        "non-forced child should have 0 visits, got {other_visits}"
    );

    tree.forced_root_child = None;
}

#[test]
fn test_forced_root_none_uses_puct() {
    let mut tree = setup_expanded_root();
    tree.forced_root_child = None;

    let n_sims = 20;
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0 / n_actions as f32; n_actions];
    for _ in 0..n_sims {
        let leaves = tree
            .select_leaves(1)
            .expect("select_leaves: no desync in this fixture");
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values = vec![0.0f32; n];
        tree.expand_and_backup(&policies, &values);
    }

    let first = tree.pool[0].first_child as usize;
    let n_ch = tree.pool[0].n_children as usize;
    let visited_count = (first..first + n_ch)
        .filter(|&i| tree.pool[i].n_visits > 0)
        .count();
    assert!(
        visited_count >= 2,
        "PUCT should visit multiple children, only {visited_count} visited"
    );
}

#[test]
fn test_gumbel_disabled_no_behavior_change() {
    // With `forced_root_child` None the behaviour is pre-Gumbel: two runs must agree.
    let run_search = || -> Vec<u32> {
        let mut tree = MCTSTree::new(1.5);
        let board = Board::new();
        tree.new_game(board);
        tree.forced_root_child = None; // explicitly None

        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform = vec![1.0 / n_actions as f32; n_actions];
        for _ in 0..10 {
            let leaves = tree
                .select_leaves(1)
                .expect("select_leaves: no desync in this fixture");
            let n = leaves.len();
            let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
            let values = vec![0.0f32; n];
            tree.expand_and_backup(&policies, &values);
        }

        let first = tree.pool[0].first_child as usize;
        let n_ch = tree.pool[0].n_children as usize;
        (first..first + n_ch)
            .map(|i| tree.pool[i].n_visits)
            .collect()
    };

    let visits_a = run_search();
    let visits_b = run_search();
    assert_eq!(
        visits_a, visits_b,
        "search with forced_root_child=None should be deterministic"
    );
}

#[test]
fn test_nonroot_uses_puct_when_root_forced() {
    // Non-root selection still uses PUCT even when root selection is forced.
    let mut tree = setup_expanded_root();
    let first_child = tree.pool[0].first_child;
    tree.forced_root_child = Some(first_child);

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0 / n_actions as f32; n_actions];

    for _ in 0..30 {
        let leaves = tree
            .select_leaves(1)
            .expect("select_leaves: no desync in this fixture");
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values = vec![0.0f32; n];
        tree.expand_and_backup(&policies, &values);
    }

    let fc = &tree.pool[first_child as usize];
    if fc.is_expanded() && fc.n_children > 1 {
        let gc_first = fc.first_child as usize;
        let gc_n = fc.n_children as usize;
        let gc_visited = (gc_first..gc_first + gc_n)
            .filter(|&i| tree.pool[i].n_visits > 0)
            .count();
        assert!(
            gc_visited >= 2,
            "PUCT at non-root should visit multiple grandchildren, got {gc_visited}"
        );
    }
    // If forced child isn't expanded (e.g., terminal), the test still passes.
    tree.forced_root_child = None;
}

#[test]
fn test_last_search_stats_bounds_after_sims() {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());

    let n_sims = 10;
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0 / n_actions as f32; n_actions];
    for _ in 0..n_sims {
        let leaves = tree
            .select_leaves(1)
            .expect("select_leaves: no desync in this fixture");
        let n = leaves.len();
        let policies: Vec<Vec<f32>> = (0..n).map(|_| uniform.clone()).collect();
        let values = vec![0.0f32; n];
        tree.expand_and_backup(&policies, &values);
    }

    let (mean_depth, root_concentration) = tree.last_search_stats();
    assert!(
        mean_depth >= 0.0,
        "mean_depth must be >= 0.0, got {mean_depth}"
    );
    assert!(
        root_concentration >= 0.0 && root_concentration <= 1.0,
        "root_concentration must be in [0.0, 1.0], got {root_concentration}"
    );
}

#[test]
fn omitted_prior_mass_is_the_tail_the_cap_dropped() {
    // The cap's own `topk_truncated` flag says only that SOMETHING was dropped, which at
    // radius 8 is true on essentially every ply — measured 3009 of 3010 expansions on a driven
    // game — so it carries no information. This pins the summed PRIOR of what was thrown away.
    use super::backup::pick_topk_children;
    use fxhash::FxHashSet;
    use mantis_core::board::HALF;

    // The cap is the picker's own PARAMETER and this fixture uses a LOCAL: the window holds
    // 361 cells, so a fixture sized off `MAX_CHILDREN_PER_NODE` stops being constructible the
    // moment that constant passes the window.
    const CAP: usize = 64;

    let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
    'fill: for q in -HALF..=HALF {
        for r in -HALF..=HALF {
            cells.insert((q, r));
            if cells.len() == CAP + 8 {
                break 'fill;
            }
        }
    }
    assert_eq!(cells.len(), CAP + 8);

    // A UNIFORM policy over the whole window: every kept child and every dropped child holds
    // the same prior, so the dropped mass is exactly `8 * p` and is hand-checkable.
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let p = 1.0f32 / n_actions as f32;
    let policy = vec![p; n_actions];

    let pick = pick_topk_children(&cells, 0, 0, &policy, BOARD_SIZE as i32, HALF, CAP);

    assert!(
        pick.truncated,
        "the fixture must exceed the cap for this to measure anything"
    );
    assert_eq!(pick.children.len(), CAP);
    let expected = 8.0 * p;
    println!(
        "dropped prior mass {} against the 8 uniform children's {expected}",
        pick.dropped_prior_mass
    );
    assert!(
        (pick.dropped_prior_mass - expected).abs() <= 1e-6,
        "dropped mass {} != the 8 uniform children's {expected}",
        pick.dropped_prior_mass
    );

    // The counter half, on a TREE — the quantity above is what a search accumulates, and the
    // bracket is the tree's own, so a second search in this process cannot enter it.
    let tree = MCTSTree::new(1.5);
    let _ = tree.take_omitted_prior();
    tree.record_omitted_prior(pick.dropped_prior_mass);
    let (mass_micros, omitted_expansions, total_expansions) = tree.take_omitted_prior();
    assert_eq!(total_expansions, 1, "one call must count as one expansion");
    assert_eq!(
        omitted_expansions, 1,
        "the truncating call was not counted as omitting"
    );
    let expected_micros = (f64::from(expected) * 1e6) as u64;
    assert!(
        mass_micros.abs_diff(expected_micros) <= 8,
        "counted {mass_micros} micros against {expected_micros} (fixed-point rounding \
         allows one micro per child)"
    );
    assert_eq!(
        tree.take_omitted_prior(),
        (0, 0, 0),
        "the bracket must be read-and-RESET, or two consecutive searches share one window"
    );
}

#[test]
fn an_untruncated_expansion_records_no_omitted_mass() {
    // The mutation half: a counter that always fires reports a cap cost on a node that has
    // fewer legal moves than the cap, which is most of the early game.
    use super::backup::pick_topk_children;
    use fxhash::FxHashSet;
    use mantis_core::board::HALF;

    let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
    for q in -2..=2 {
        for r in -2..=2 {
            cells.insert((q, r));
        }
    }
    assert!(cells.len() < MAX_CHILDREN_PER_NODE);

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let policy = vec![1.0f32 / n_actions as f32; n_actions];

    let pick = pick_topk_children(
        &cells,
        0,
        0,
        &policy,
        BOARD_SIZE as i32,
        HALF,
        MAX_CHILDREN_PER_NODE,
    );
    let tree = MCTSTree::new(1.5);
    let _ = tree.take_omitted_prior();
    tree.record_omitted_prior(pick.dropped_prior_mass);
    let (mass_micros, omitted_expansions, total_expansions) = tree.take_omitted_prior();

    assert!(!pick.truncated);
    assert_eq!(
        total_expansions, 1,
        "an untruncated expansion must still be counted"
    );
    assert_eq!(
        omitted_expansions, 0,
        "an untruncated expansion reported an omission"
    );
    assert_eq!(
        mass_micros, 0,
        "an untruncated expansion reported dropped mass"
    );
}

#[test]
fn test_topk_truncates_at_the_supplied_cap() {
    use super::backup::pick_topk_children;
    use fxhash::FxHashSet;
    use mantis_core::board::HALF;

    // A LOCAL cap, not `MAX_CHILDREN_PER_NODE`: the 361-cell window bounds the fixture, and
    // the production constant does not.
    const CAP: usize = 128;

    // 600 unique cells: 200 in-window with high priors, 400 out-of-window at sort prior 0.0,
    // so top-CAP is drawn from the in-window set.
    let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
    'iw: for q in -HALF..=HALF {
        for r in -HALF..=HALF {
            cells.insert((q, r));
            if cells.len() == 200 {
                break 'iw;
            }
        }
    }
    'ow: for q in 30..=60 {
        for r in 30..=60 {
            cells.insert((q, r));
            if cells.len() == 600 {
                break 'ow;
            }
        }
    }
    assert_eq!(cells.len(), 600, "test setup must produce 600 cells");

    // Strictly increasing prior with flat_idx gives every in-window cell a unique prior.
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let policy: Vec<f32> = (0..n_actions)
        .map(|i| (i + 1) as f32 / n_actions as f32)
        .collect();

    let pick = pick_topk_children(&cells, 0, 0, &policy, BOARD_SIZE as i32, HALF, CAP);
    let chosen = pick.children;
    assert!(pick.truncated, "600 > CAP must report truncation");
    assert_eq!(
        chosen.len(),
        CAP,
        "chosen must equal the cap, got {}",
        chosen.len()
    );

    // Out-of-window cells sort at 0.0, so top-CAP is entirely in-window.
    for &((q, r), prior) in &chosen {
        let flat = Board::window_flat_idx_at(q, r, 0, 0);
        assert!(
            flat < n_actions,
            "top-K must be drawn from in-window cells, got flat={flat} for ({q},{r})"
        );
        assert!(
            prior > 0.0,
            "in-window cell prior must be >0 for this fixture, got {prior}"
        );
    }

    // Priors monotonic in flat_idx, so the chosen Vec's priors must be non-increasing.
    for w in chosen.windows(2) {
        assert!(
            w[0].1 >= w[1].1,
            "priors must be non-increasing: {} then {}",
            w[0].1,
            w[1].1
        );
    }
}

#[test]
fn test_topk_tie_break_by_flat_idx() {
    use super::backup::pick_topk_children;
    use fxhash::FxHashSet;
    use mantis_core::board::HALF;

    // CAP + 1 cells inside the window at identical priors, so exactly one is dropped and the
    // flat_idx-ascending tie-break drops the largest flat_idx. CAP is a local for the sibling's
    // reason: the window bounds the fixture, the production constant does not.
    const CAP: usize = 128;
    let target = CAP + 1;
    let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
    let mut flats_inserted: Vec<usize> = Vec::new();
    'outer: for q in -HALF..=HALF {
        for r in -HALF..=HALF {
            let flat = Board::window_flat_idx_at(q, r, 0, 0);
            cells.insert((q, r));
            flats_inserted.push(flat);
            if cells.len() == target {
                break 'outer;
            }
        }
    }
    assert_eq!(cells.len(), target);

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform_high = vec![0.5_f32; n_actions];

    let pick = pick_topk_children(&cells, 0, 0, &uniform_high, BOARD_SIZE as i32, HALF, CAP);
    let chosen = pick.children;
    assert!(pick.truncated);
    assert_eq!(chosen.len(), CAP);

    let chosen_flats: std::collections::HashSet<usize> = chosen
        .iter()
        .map(|&((q, r), _)| Board::window_flat_idx_at(q, r, 0, 0))
        .collect();

    let max_flat = *flats_inserted.iter().max().unwrap();
    assert!(
        !chosen_flats.contains(&max_flat),
        "highest flat_idx must be the dropped cell under tie (max_flat={max_flat})"
    );
    assert_eq!(chosen_flats.len(), CAP);
}

#[test]
fn test_topk_fast_path_keeps_all_when_under_cap() {
    use super::backup::pick_topk_children;
    use fxhash::FxHashSet;
    use mantis_core::board::HALF;

    // 50 cells with K=192 takes the fast path: every cell appears and `sort_used` is false.
    let mut cells: FxHashSet<(i32, i32)> = FxHashSet::default();
    'outer: for q in -3..=4 {
        for r in -3..=4 {
            cells.insert((q, r));
            if cells.len() == 50 {
                break 'outer;
            }
        }
    }
    assert_eq!(cells.len(), 50);

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let policy = vec![1.0 / n_actions as f32; n_actions];

    let pick = pick_topk_children(
        &cells,
        0,
        0,
        &policy,
        BOARD_SIZE as i32,
        HALF,
        MAX_CHILDREN_PER_NODE,
    );
    let (chosen, sort_used) = (pick.children, pick.truncated);
    assert!(!sort_used, "fast path expected when n_legal <= K");
    assert_eq!(chosen.len(), 50);

    let chosen_set: std::collections::HashSet<(i32, i32)> =
        chosen.iter().map(|&(coord, _)| coord).collect();
    assert_eq!(
        chosen_set,
        cells.iter().copied().collect(),
        "fast path must include every legal move"
    );
}

#[test]
fn test_topk_child_order_independent_of_hashset_capacity() {
    // Regression guard: `pick_topk_children` must emit a canonical order independent of the
    // `FxHashSet`'s capacity or iteration order. A `legal_moves_set` capacity-reserve changed
    // the hashbrown layout, and the `n_legal <= K` path leaked it into MCTS.
    use super::backup::pick_topk_children;
    use fxhash::FxHashSet;
    use mantis_core::board::HALF;

    let coords: Vec<(i32, i32)> = (-3..=3)
        .flat_map(|q| (-3..=3).map(move |r| (q, r)))
        .collect();

    // Same elements at deliberately different capacities, so the two sets iterate differently.
    let mut set_small: FxHashSet<(i32, i32)> = FxHashSet::default();
    for &c in &coords {
        set_small.insert(c);
    }
    let mut set_large: FxHashSet<(i32, i32)> = FxHashSet::default();
    set_large.reserve(4096);
    for &c in &coords {
        set_large.insert(c);
    }
    assert_eq!(
        set_small, set_large,
        "the two sets must hold identical moves"
    );

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let mut policy = vec![0.0f32; n_actions];
    for (i, p) in policy.iter_mut().enumerate() {
        *p = ((i % 17) as f32) * 0.013;
    }

    let chosen_small = pick_topk_children(
        &set_small,
        0,
        0,
        &policy,
        BOARD_SIZE as i32,
        HALF,
        MAX_CHILDREN_PER_NODE,
    );
    let chosen_large = pick_topk_children(
        &set_large,
        0,
        0,
        &policy,
        BOARD_SIZE as i32,
        HALF,
        MAX_CHILDREN_PER_NODE,
    );

    assert_eq!(
        chosen_small.children, chosen_large.children,
        "pick_topk_children child ORDER must be independent of FxHashSet \
         capacity / iteration order (see backup.rs fn-doc)"
    );
}

// The desync is a NAMED error, and the forced child is bounded. `select_one_leaf` used
// `expect("selected move should always be legal")`, and `Board::apply_move` errs ONLY on
// occupancy, so it fired exactly when a child's stored `action_idx` decoded to an occupied
// cell — tree and board desynchronised. It fired in production: `selfplay/worker.py` matched
// the panic's message text to restart the tree at root.

/// A tree whose root has ONE child pointing at a cell the root board already holds: the stone
/// is played BEFORE `new_game` and the child's `action_idx` is overwritten to decode back to
/// it — exactly the state the production `expect` fired on.
fn desynchronised_root() -> MCTSTree {
    let mut tree = MCTSTree::new(1.5);
    let mut board = Board::new();
    board
        .apply_move(0, 0)
        .expect("an empty board accepts (0, 0)");
    tree.new_game(board);
    let _leaves = tree
        .select_leaves(1)
        .expect("the fresh root selects itself");
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    tree.expand_and_backup(&[vec![1.0 / n_actions as f32; n_actions]], &[0.0]);

    let first = tree.pool[0].first_child as usize;
    tree.pool[first].action_idx = (32768u32 << 16) | 32768u32; // decodes to (0, 0)
    tree.pool[0].n_children = 1; // the descent has no other child to take
    tree
}

#[test]
fn a_child_pointing_at_an_occupied_cell_is_an_ERR_not_a_panic() {
    let mut tree = desynchronised_root();
    let err = tree
        .select_leaves(1)
        .expect_err("a desynchronised child must not be a panic");
    assert_eq!(err.q, 0, "the error names the cell the board refused");
    assert_eq!(err.r, 0);
    assert_eq!(err.node, 0, "…and the node whose child was selected");
    assert!(err.to_string().contains("desynchronised"), "{err}");
}

#[test]
fn the_failing_descent_leaves_no_virtual_loss_behind() {
    // A propagated Err must not permanently penalise the path it walked: those nodes already
    // took their virtual loss, and a recovering caller would search a biased tree.
    let mut tree = desynchronised_root();
    let before = tree.pool[0].virtual_loss_count;
    let _ = tree.select_leaves(1);
    assert_eq!(
        tree.pool[0].virtual_loss_count, before,
        "the root kept the virtual loss from a descent that produced no leaf"
    );
}

#[test]
fn a_healthy_tree_still_selects_leaves() {
    // The control. The repair must not turn a working search into a refusal.
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());
    let leaves = tree.select_leaves(1).expect("a fresh tree has no desync");
    assert_eq!(leaves.len(), 1);
}

#[test]
fn a_forced_root_child_outside_the_roots_range_is_refused() {
    // The second trigger. `u32::MAX` index-panicked on the next descent, and any other foreign
    // index descended into a node the root does not own — an uninitialised slot's
    // `action_idx = u32::MAX` decodes to (32767, 32767), which an UNBOUNDED board accepts, so
    // that arm produced neither a panic nor an error.
    let mut tree = setup_expanded_root();
    let err = tree
        .set_forced_root_child(Some(u32::MAX))
        .expect_err("u32::MAX is not a child");
    assert_eq!(err.child, u32::MAX);
    assert!(err.to_string().contains("not a child of the root"), "{err}");

    let first = tree.pool[0].first_child;
    let past_end = first + u32::from(tree.pool[0].n_children);
    assert!(
        tree.set_forced_root_child(Some(past_end)).is_err(),
        "one past the last child is still not a child"
    );
    assert!(
        tree.set_forced_root_child(Some(first)).is_ok(),
        "the first child IS in range"
    );
    assert!(
        tree.set_forced_root_child(None).is_ok(),
        "clearing is always allowed"
    );
}

#[test]
fn a_root_with_no_children_accepts_no_forced_child_at_all() {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(Board::new());
    assert_eq!(
        tree.pool[0].n_children, 0,
        "an unexpanded root owns nothing"
    );
    assert!(tree.set_forced_root_child(Some(0)).is_err());
    assert!(tree.set_forced_root_child(Some(1)).is_err());
    assert!(tree.set_forced_root_child(None).is_ok());
}

// A short batch degrades the BATCH, not the TREE. `expand_and_backup` took
// `n = pending.len().min(policies.len()).min(values.len())` and dropped the rest of an already
// `mem::take`n `pending`, so every node on those descents kept a virtual loss nothing would
// ever back up — permanently depressing their PUCT score, on the deploy-strength path.

#[test]
fn a_short_policy_batch_gives_the_dropped_leaves_their_virtual_loss_back() {
    // The fixture the virtual-loss divergence row uses, and the one that reliably yields TWO
    // distinct leaves in one batch.
    let (mut tree, _child_a, _child_b) = setup_two_child_tree(1.5);
    tree.pool[0].n_visits = 1;
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;

    let leaves = tree.select_leaves(2).expect("no desync");
    assert_eq!(leaves.len(), 2, "the fixture needs two distinct leaves");
    let dropped: Vec<u32> = tree.pending.iter().skip(1).map(|(idx, _)| *idx).collect();
    assert_eq!(dropped.len(), 1);

    tree.expand_and_backup(&[vec![1.0 / n_actions as f32; n_actions]], &[0.0]);

    for idx in dropped {
        assert_eq!(
            tree.pool[idx as usize].virtual_loss_count, 0,
            "leaf {idx} was dropped from the batch and kept its virtual loss — its PUCT \
             score stays depressed for the rest of the search"
        );
    }
}

#[test]
fn a_FULL_batch_is_unaffected_by_the_unwind() {
    // The control: the ordinary path must not lose a leaf or an update.
    let (mut tree, _a, _b) = setup_two_child_tree(1.5);
    tree.pool[0].n_visits = 1;
    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let leaves = tree.select_leaves(2).expect("no desync");
    let n = leaves.len();
    let policies: Vec<Vec<f32>> = (0..n)
        .map(|_| vec![1.0 / n_actions as f32; n_actions])
        .collect();
    tree.expand_and_backup(&policies, &vec![0.0; n]);
    assert!(tree.pending.is_empty(), "every pending leaf was consumed");
}
