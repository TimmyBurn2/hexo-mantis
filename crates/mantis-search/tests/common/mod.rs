//! The Gumbel drive, the skewed and compact priors and the radius-8 boards the search tests share.

use mantis_core::Board;
use mantis_search::{analyze, MCTSTree, MctxRootState, QSigma, SearchKind, Terminal};

/// 19-window stride with a pass slot.
pub const N_ACTIONS: usize = 19 * 19 + 1;
pub const C_VISIT: f32 = 50.0;

pub fn r8_board() -> Board {
    let mut board = Board::new();
    board.set_legal_move_radius(8);
    board
        .apply_move(0, 0)
        .expect("(0,0) is legal on a fresh board");
    board
}

/// A skewed but everywhere-positive prior: a uniform one hides a flat fallback, and would let a
/// bug that collapsed the support to one cell still sum to 1.
pub fn stub_policy() -> Vec<f32> {
    let raw: Vec<f32> = (0..N_ACTIONS)
        .map(|i| 1.0 + (i % 13) as f32 * 0.25)
        .collect();
    let total: f32 = raw.iter().sum();
    raw.into_iter().map(|x| x / total).collect()
}

/// `moves` on a radius-8 board.
pub fn replay(moves: &[(i32, i32)]) -> Board {
    let mut b = Board::new();
    b.set_legal_move_radius(8);
    for &(q, r) in moves {
        b.apply_move(q, r).expect("a legal fixture move");
    }
    b
}

/// P2 holds five on r = 3 and places two: its finish is a terminal child a compact prior piles onto.
pub const FIVE: [(i32, i32); 13] = [
    (0, 0),
    (0, 3),
    (1, 3),
    (0, -3),
    (5, -5),
    (2, 3),
    (3, 3),
    (-5, 0),
    (-4, -2),
    (4, 3),
    (-3, 6),
    (-6, 2),
    (-2, -6),
];

/// A prior decaying from the window centre (a compact net), with 0.9 of the mass on the mover's finish where it has one.
pub fn compact_prior(board: &Board) -> Vec<f32> {
    let mut p: Vec<f32> = (0..N_ACTIONS - 1)
        .map(|i| {
            let (dq, dr) = ((i / 19) as i32 - 9, (i % 19) as i32 - 9);
            (-1.5 * dq.abs().max(dr.abs()).max((dq + dr).abs()) as f32).exp()
        })
        .collect();
    p.push(0.0);
    let total: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= total);
    let facts = analyze(board);
    if facts.terminal == Some(Terminal::Win) {
        let idx = board.window_flat_idx(facts.finish[0].0, facts.finish[0].1);
        if idx < p.len() {
            p.iter_mut().for_each(|x| *x *= 0.1);
            p[idx] += 0.9;
        }
    }
    p
}

/// One Gumbel search of `sims` sims over `board` at `m` candidates, driven as self-play drives it.
pub fn gumbel_search(
    board: &Board,
    policy: &[f32],
    seed: u64,
    sims: usize,
    m: usize,
    c_scale: f32,
) -> MCTSTree {
    gumbel_search_with_state(board, policy, seed, sims, m, c_scale).0
}

/// `gumbel_search` with the search's root state, for a reader of its halving.
pub fn gumbel_search_with_state(
    board: &Board,
    policy: &[f32],
    seed: u64,
    sims: usize,
    m: usize,
    c_scale: f32,
) -> (MCTSTree, MctxRootState) {
    let sigma = QSigma {
        c_visit: C_VISIT,
        c_scale,
        rescale: true,
    };
    let mut tree = MCTSTree::new(1.5);
    tree.configure_quiescence(false, 0.0);
    tree.configure_search(SearchKind::Gumbel, sigma);
    tree.new_game(board.clone());

    // ONE root leaf, charged against the budget as the self-play drive charges it.
    let root = tree.select_leaves(1).expect("a fresh root selects itself");
    assert_eq!(root.len(), 1);
    tree.expand_and_backup(&[policy.to_vec()], &[0.1]);

    let budget = sims - 1;
    let state = MctxRootState::new_seeded(&tree, m, budget, seed);
    let mut spent = 0usize;
    while spent < budget {
        let mut round = state.round_batch(&tree, sigma);
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
    (tree, state)
}
