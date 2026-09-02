//! Structural regression guard for the Dirichlet noise application used on
//! both sides of the self-play `if gumbel_mcts` branch.
//!
//! An earlier audit flagged the duplicated Dirichlet call sites as a minor drift
//! risk. Code inspection confirmed the two branches are structurally identical —
//! same `sample_dirichlet` args, same `apply_dirichlet_to_root` args, same
//! intermediate-ply gate. This file pins the **building blocks** both branches
//! depend on: if any of these regress, both branches fail together.
//!
//! What this file asserts:
//!   1. `sample_dirichlet` output is non-negative and sums to 1.0.
//!   2. `apply_dirichlet_to_root` applies the linear blend
//!      `new = (1-eps)*old + eps*noise` per child.
//!   3. `apply_dirichlet_to_root` with epsilon=0 is a byte-exact no-op.
//!   4. The `is_intermediate_ply = moves_remaining == 1 && ply > 0` gate
//!      produces the correct truth value across a played-out move sequence.

use mantis_core::board::{Board, BOARD_SIZE};
use mantis_search::mcts::dirichlet::sample_dirichlet;
use mantis_search::MCTSTree;

/// Build an `MCTSTree` from a fresh board, expand the root by feeding a
/// uniform policy and zero value, and return the expanded tree.
fn expanded_root_tree() -> MCTSTree {
    let mut tree = MCTSTree::new_full(1.5, 1.0, 0.0);
    tree.new_game(Board::new());

    // One select/expand cycle expands the root (the only leaf on a fresh tree).
    let leaves = tree.select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    assert_eq!(leaves.len(), 1, "fresh tree must yield exactly one root leaf");

    let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];
    tree.expand_and_backup(&[uniform], &[0.0]);

    assert!(
        tree.root_n_children() > 0,
        "root must have children after expand_and_backup"
    );
    tree
}

#[test]
fn sample_dirichlet_sums_to_one_and_is_nonneg() {
    // Mirrors the shape used by the self-play loop in both branches: alpha from
    // config (0.3 is the default) and n = root children count, which for a fresh
    // board sits at ~24 (hex neighbourhood). We sample a few sizes to make sure
    // the normalisation is not length-specific.
    let mut rng = rand::rng();
    for &n in &[1_usize, 2, 5, 24, 50] {
        for _ in 0..10 {
            let v = sample_dirichlet(0.3, n, &mut rng);
            assert_eq!(v.len(), n, "sample_dirichlet returned wrong length");
            for &x in &v {
                assert!(
                    x >= 0.0,
                    "sample_dirichlet produced negative entry {x} for n={n}"
                );
            }
            let sum: f32 = v.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "sample_dirichlet sum {sum} not within 1e-5 of 1.0 for n={n}"
            );
        }
    }
}

#[test]
fn apply_dirichlet_to_root_blends_linearly() {
    // Both self-play call sites ultimately feed through
    // MCTSTree::apply_dirichlet_to_root, which does
    //     child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[j];
    // This test pins that formula: if anyone edits it, both branches break.
    let mut tree = expanded_root_tree();
    let n_ch = tree.root_n_children();
    assert!(n_ch >= 2, "expected >=2 root children on a fresh board, got {n_ch}");

    // Snapshot (pool_idx, pre_prior) for every root child.
    let pre: Vec<(u32, f32)> = tree.get_root_children_info();
    assert_eq!(pre.len(), n_ch);

    // Use a non-uniform noise so the test would fail if the blend silently
    // collapsed to `new = old` or `new = noise`. Pick a deterministic shape:
    // ramp 0, 1, 2, ..., n-1 then normalise to sum-to-1.
    let mut noise: Vec<f32> = (0..n_ch).map(|j| (j as f32) + 1.0).collect();
    let s: f32 = noise.iter().sum();
    for x in &mut noise {
        *x /= s;
    }

    let eps = 0.25_f32; // default
    tree.apply_dirichlet_to_root(&noise, eps);

    let post: Vec<(u32, f32)> = tree.get_root_children_info();
    assert_eq!(post.len(), n_ch);

    // get_root_children_info iterates children in pool order, so the j-th
    // entry in `pre`, `post`, and `noise` correspond.
    for j in 0..n_ch {
        let (idx_pre, old_prior) = pre[j];
        let (idx_post, new_prior) = post[j];
        assert_eq!(
            idx_pre, idx_post,
            "child pool_idx reordered across apply_dirichlet_to_root"
        );
        let expected = (1.0 - eps) * old_prior + eps * noise[j];
        assert!(
            (new_prior - expected).abs() < 1e-6,
            "child {j}: expected {expected}, got {new_prior} \
             (old={old_prior}, noise={noise_j}, eps={eps})",
            noise_j = noise[j],
        );
    }
}

#[test]
fn apply_dirichlet_with_zero_epsilon_is_noop() {
    // epsilon=0 is a degenerate case that must leave priors bit-exact:
    //     child.prior = 1.0 * child.prior + 0.0 * noise = child.prior.
    // If anyone ever "simplifies" the blend to `eps * noise + child.prior`
    // or similar, this test catches it.
    let mut tree = expanded_root_tree();
    let pre: Vec<(u32, f32)> = tree.get_root_children_info();
    let n_ch = pre.len();

    // Use a wildly non-uniform noise so any fractional bleed shows up.
    let noise: Vec<f32> = (0..n_ch)
        .map(|j| if j == 0 { 1.0 } else { 0.0 })
        .collect();

    tree.apply_dirichlet_to_root(&noise, 0.0);

    let post: Vec<(u32, f32)> = tree.get_root_children_info();
    assert_eq!(post.len(), n_ch);

    for j in 0..n_ch {
        let (idx_pre, old_prior) = pre[j];
        let (idx_post, new_prior) = post[j];
        assert_eq!(idx_pre, idx_post);
        // Bit-exact: `(1.0 - 0.0) * x + 0.0 * y == x` under IEEE-754 for finite x,
        // since 1.0*x = x and 0.0*y = 0.0 (for finite y) and x + 0.0 = x.
        assert_eq!(
            new_prior.to_bits(),
            old_prior.to_bits(),
            "zero-epsilon changed child {j}: {old_prior} -> {new_prior}"
        );
    }
}

#[test]
fn intermediate_ply_gate_matches_self_play_spec() {
    // Exact formula duplicated at both self-play call sites:
    //     let is_intermediate_ply = board.moves_remaining == 1 && board.ply > 0;
    //
    // Turn structure for Hex Tac Toe:
    //     ply=0, mr=1  → P1 opens with one stone          (NOT intermediate)
    //     ply=1, mr=2  → P2 starts compound turn          (NOT intermediate)
    //     ply=2, mr=1  → P2 mid-turn, one stone left       (INTERMEDIATE)
    //     ply=3, mr=2  → P1 starts compound turn          (NOT intermediate)
    //     ply=4, mr=1  → P1 mid-turn, one stone left       (INTERMEDIATE)
    //     ply=5, mr=2  → P2 starts compound turn          (NOT intermediate)
    //
    // If `Board` ever changes how `ply` or `moves_remaining` advance (e.g.
    // starting P1 with 2 moves instead of 1), this test fails loudly and
    // forces both call sites to be reviewed together.

    fn is_intermediate(b: &Board) -> bool {
        b.moves_remaining == 1 && b.ply.index() > 0
    }

    let mut board = Board::new();

    // ply=0 — opening position, a turn boundary even though mr=1.
    assert_eq!(board.ply.index(), 0);
    assert_eq!(board.moves_remaining, 1);
    assert!(
        !is_intermediate(&board),
        "ply=0 must not be an intermediate ply (opening stone is a turn boundary)"
    );

    // Walk a short sequence of legal non-overlapping moves. Coordinates
    // stay within the 5×5 initial legality window from Board::new().
    let seq: &[(i32, i32)] = &[(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)];
    let expected_after_move: &[(u32, u8, bool)] = &[
        // (ply, moves_remaining, is_intermediate_expected)
        (1, 2, false), // P1 done; P2 starts compound turn
        (2, 1, true),  // P2 mid-turn — INTERMEDIATE
        (3, 2, false), // P2 done; P1 starts compound turn
        (4, 1, true),  // P1 mid-turn — INTERMEDIATE
        (5, 2, false), // P1 done; P2 starts compound turn
    ];
    assert_eq!(seq.len(), expected_after_move.len());

    for (i, &(q, r)) in seq.iter().enumerate() {
        board.apply_move(q, r).expect("move must be legal in test sequence");
        let (want_ply, want_mr, want_inter) = expected_after_move[i];
        assert_eq!(
            board.ply.index(), want_ply,
            "move {i}: ply mismatch (got {got}, want {want_ply})",
            got = board.ply.index(),
        );
        assert_eq!(
            board.moves_remaining, want_mr,
            "move {i}: moves_remaining mismatch (got {got}, want {want_mr})",
            got = board.moves_remaining,
        );
        assert_eq!(
            is_intermediate(&board),
            want_inter,
            "move {i}: is_intermediate_ply mismatch at ply={p} mr={mr}",
            p = board.ply.index(),
            mr = board.moves_remaining,
        );
    }
}
