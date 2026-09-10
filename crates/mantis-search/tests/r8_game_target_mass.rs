//! A driven r8 game's policy target sums to 1 over the FULL legal set, at every ply.
//!
//! Under `SearchKind::Gumbel` the exported target is the completed-Q improved policy and its
//! support is the whole legal set, including moves no simulation visited and moves outside the
//! dense window. A target supported on the visited or top-K set would teach the net that every
//! other legal move is unplayable — hundreds per ply at radius 8.
//!
//! The tree is driven directly rather than through `SelfPlayRunner` because what is measured is
//! the SEARCH's own export: the RECORD stores the row sparse and is a different object, measured
//! by `crates/mantis-search/tests/gumbel_sparse_row_kl.rs`.
//!
//! The root still MATERIALIZES every legal child; a root that samples and completes over the full
//! legal prior while creating children only for the sampled candidates is not landed, and
//! `the_root_materializes_every_legal_child_today` pins today's behaviour so the change is loud.

use mantis_core::Board;
use mantis_search::{MCTSTree, MctxRootState, SearchKind};

/// 19-window stride with a pass slot.
const N_ACTIONS: usize = 19 * 19 + 1;
/// Small on purpose: the support property does not depend on the budget.
const SIMS: usize = 24;
const GUMBEL_M: usize = 8;
const PLIES: usize = 12;

fn r8_board() -> Board {
    let mut board = Board::new();
    board.set_legal_move_radius(8);
    board
        .apply_move(0, 0)
        .expect("(0,0) is legal on a fresh board");
    board
}

/// A skewed but everywhere-positive prior: uniform would let a bug that collapsed the
/// support to one cell still sum to 1.
fn stub_policy() -> Vec<f32> {
    let raw: Vec<f32> = (0..N_ACTIONS)
        .map(|i| 1.0 + (i % 13) as f32 * 0.25)
        .collect();
    let total: f32 = raw.iter().sum();
    raw.into_iter().map(|x| x / total).collect()
}

/// One Gumbel search over `board`, returning the tree it leaves behind.
fn search(board: &Board, policy: &[f32], seed: u64) -> MCTSTree {
    let mut tree = MCTSTree::new(1.5);
    tree.configure_quiescence(false, 0.0);
    tree.configure_search(SearchKind::Gumbel, 50.0, 0.1);
    tree.new_game(board.clone());

    // ONE root leaf, charged against the budget as the self-play drive charges it.
    let root = tree.select_leaves(1).expect("a fresh root selects itself");
    assert_eq!(root.len(), 1);
    tree.expand_and_backup(&[policy.to_vec()], &[0.1]);

    let budget = SIMS - 1;
    let state = MctxRootState::new_seeded(&tree, GUMBEL_M, budget, seed);
    let mut spent = 0usize;
    while spent < budget {
        let mut round = state.round_batch(&tree, 50.0, 0.1);
        if round.is_empty() {
            break;
        }
        round.truncate(budget - spent);
        let Ok(leaves) = tree.select_leaves_forced(&round) else {
            break;
        };
        if leaves.is_empty() {
            break;
        }
        let policies: Vec<Vec<f32>> = (0..leaves.len()).map(|_| policy.to_vec()).collect();
        // Values that vary per round, so the completion has something to complete with.
        let values: Vec<f32> = (0..leaves.len())
            .map(|i| 0.3 - 0.05 * ((spent + i) % 7) as f32)
            .collect();
        tree.expand_and_backup(&policies, &values);
        spent += leaves.len();
    }
    tree
}

#[test]
fn every_ply_of_a_driven_r8_game_exports_a_target_over_the_full_legal_set() {
    let policy = stub_policy();
    let mut board = r8_board();
    let mut plies_measured = 0usize;
    let mut widest_legal = 0usize;
    let mut plies_with_overflow = 0usize;

    for ply in 0..PLIES {
        let legal = board.legal_moves();
        if legal.is_empty() {
            break;
        }
        widest_legal = widest_legal.max(legal.len());

        let tree = search(&board, &policy, 20260909 + ply as u64);
        let target = tree.get_improved_policy_ls(N_ACTIONS, 50.0, 0.1);

        // SUPPORT: every legal move carries mass, and nothing else does.
        let support = target.dense.iter().filter(|&&m| m > 0.0).count() + target.overflow.len();
        assert_eq!(
            support,
            legal.len(),
            "ply {ply}: the exported target puts mass on {support} cells against {} legal \\
             moves. A target whose support is the visited or top-K set teaches the net that \\
             every other legal move is unplayable.",
            legal.len()
        );

        // MASS: summed over the dense window AND the off-window overflow — at radius 8 most of
        // the legal set is outside the window, so a dropped overflow still looks normalized to a
        // dense-only reader.
        let dense_mass: f32 = target.dense.iter().sum();
        let overflow_mass: f32 = target.overflow.values().sum();
        let total = dense_mass + overflow_mass;
        assert!(
            (total - 1.0).abs() < 1e-4,
            "ply {ply}: the exported target sums to {total} (dense {dense_mass}, overflow \\
             {overflow_mass}), not 1"
        );
        if overflow_mass > 0.0 {
            plies_with_overflow += 1;
        }
        // ADDRESSABILITY: a support COUNT could be met by masses sitting on the wrong cells.
        let (bcq, bcr) = board.window_center();
        let trunk = board.cluster_window_size() as i32;
        let half = (trunk - 1) / 2;
        for &(q, r) in &legal {
            let mass = target.get(q, r, bcq, bcr, trunk, half, 0.0);
            assert!(
                mass > 0.0,
                "ply {ply}: legal move ({q}, {r}) carries no target mass"
            );
        }

        println!(
            "ply {ply}: legal {} support {support} dense {dense_mass:.6} overflow \
             {overflow_mass:.6} total {total:.6}",
            legal.len()
        );

        // Play the search's own answer, so the game walks the line the search chose.
        let mv = *legal
            .first()
            .expect("the legal set was checked non-empty above");
        board.apply_move(mv.0, mv.1).expect("a legal move applies");
        plies_measured += 1;
    }

    println!(
        "r8 target mass: {plies_measured} plies, widest legal {widest_legal}, plies with \
         off-window overflow {plies_with_overflow}, per-node cap {}",
        mantis_search::MAX_CHILDREN_PER_NODE
    );
    assert_eq!(
        plies_measured, PLIES,
        "the game ended before the measured span — {plies_measured} of {PLIES} plies"
    );
    // THE RAGGED HALF HAS TO BE EXERCISED, or the sums above are a dense-only statement. An early
    // ply fits the window, so this is asserted over the GAME rather than per ply.
    assert!(
        plies_with_overflow > 0,
        "no ply of this game put mass outside the dense window, so the export's ragged half \
         is untested here and the per-ply sums prove less than they look"
    );
    // The regime has to be one where the caps could bind, or none of the above tests anything.
    assert!(
        widest_legal > mantis_search::MAX_CHILDREN_PER_NODE,
        "the widest legal set reached was {widest_legal}, under the per-node cap \\
         ({}), so no ply in this game could have truncated and the support assertions are \\
         vacuous",
        mantis_search::MAX_CHILDREN_PER_NODE
    );
}

/// Today's root materializes the whole legal set, which is what makes the support property above
/// hold and what costs `MAX_ROOT_CHILDREN` pool slots. When a non-materializing root lands this
/// test reds and its sibling above must still pass — that pair is the whole content of "samples
/// over the full vector, materializes only candidates".
#[test]
fn the_root_materializes_every_legal_child_today() {
    let policy = stub_policy();
    let board = r8_board();
    let legal = board.legal_moves().len();
    let tree = search(&board, &policy, 7);
    println!(
        "root children {} against legal {legal}",
        tree.root_n_children()
    );
    assert_eq!(
        tree.root_n_children(),
        legal,
        "the root holds {} children against {legal} legal moves — if this has become `<= m` \\
         the non-materializing root has landed, and this pin is what should be deleted (the \\
         support test above is the one that must keep passing)",
        tree.root_n_children()
    );
}
