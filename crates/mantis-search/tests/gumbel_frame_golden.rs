//! Gumbel never scores a pending child, so its seeded trees are pinned to the digest recorded before PUCT's frame moved.

#[allow(dead_code)]
mod common;

use common::{r8_board, stub_policy, C_VISIT};
use mantis_core::Board;
use mantis_search::{MCTSTree, MctxRootState, QSigma, SearchKind};

/// A leaf's value from its stones alone (order-free), so the tree's values follow the positions its picks reach.
fn positional_value(board: &Board) -> f32 {
    let acc = board.cells_iter().fold(0u64, |acc, (&(q, r), &cell)| {
        let key = (q as i64 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (r as i64 as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
            ^ (cell as i64 as u64).wrapping_mul(0x1656_67b1_9e37_79f9);
        acc.wrapping_add(key.wrapping_mul(0xff51_afd7_ed55_8ccd) >> 7)
    });
    (acc % 1000) as f32 / 1000.0 * 0.8 - 0.4
}

/// The shared Gumbel drive, its leaf values positional instead of by batch slot.
fn search(seed: u64, sims: usize, m: usize) -> MCTSTree {
    let sigma = QSigma {
        c_visit: C_VISIT,
        c_scale: 1.0,
        rescale: true,
    };
    let policy = stub_policy();
    let mut tree = MCTSTree::new(1.5);
    tree.configure_quiescence(false, 0.0);
    tree.configure_search(SearchKind::Gumbel, sigma);
    tree.new_game(r8_board());
    tree.select_leaves(1).expect("a fresh root selects itself");
    tree.expand_and_backup(std::slice::from_ref(&policy), &[0.1]);
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
        let values: Vec<f32> = leaves.iter().map(positional_value).collect();
        tree.expand_and_backup(&vec![policy.clone(); leaves.len()], &values);
        spent += leaves.len();
    }
    tree
}

/// FNV-1a over every allocated node's links, visits and value bits, seed by seed.
fn digest() -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for seed in 0..6u64 {
        let tree = search(seed, 96, 16);
        for node in &tree.pool[..tree.next_free_slot() as usize] {
            for x in [
                node.parent,
                node.action_idx,
                node.n_visits,
                node.first_child,
            ] {
                mix(u64::from(x));
            }
            mix(u64::from(node.n_children));
            mix(u64::from(node.w_value.to_bits()));
        }
    }
    h
}

#[test]
fn a_seeded_gumbel_search_plays_the_trees_recorded_before_the_frame_change() {
    assert_eq!(
        digest(),
        GOLDEN,
        "a Gumbel search moved: the frame change reached Gumbel"
    );
}

const GOLDEN: u64 = 13_468_732_391_583_029_684;
