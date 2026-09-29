//! `last_tt_hits` is exactly a PUCT select call's table-valued descents: a terminal revisit backs up its own value.

use mantis_core::Board;
use mantis_search::{analyze, MCTSTree, Terminal};

use common::compact_prior;

mod common;

fn terminal_visits(tree: &MCTSTree) -> u64 {
    tree.pool[..tree.next_free_slot() as usize]
        .iter()
        .filter(|n| n.is_terminal)
        .map(|n| u64::from(n.n_visits))
        .sum()
}

/// Every select call raises the root by its table hits plus its terminal revisits; (hits, revisits) over `sims`.
fn drive(board: &Board, sims: usize) -> (u64, u64) {
    let mut tree = MCTSTree::new(1.5);
    tree.configure_quiescence(false, 0.0);
    tree.new_game(board.clone());
    let (mut hits, mut revisits, mut spent) = (0u64, 0u64, 0usize);
    for call in 0..10 * sims {
        if spent >= sims {
            break;
        }
        let (root0, term0) = (tree.root_visits(), terminal_visits(&tree));
        let leaves = tree
            .select_leaves((sims - spent).min(8))
            .expect("the board replays every selected cell");
        let rose = u64::from(tree.root_visits() - root0);
        let (h, t) = (tree.last_tt_hits() as u64, terminal_visits(&tree) - term0);
        assert_eq!(
            rose,
            h + t,
            "call {call}: the root rose by {rose} with {h} table hits and {t} terminal revisits"
        );
        hits += h;
        revisits += t;
        if leaves.is_empty() {
            continue;
        }
        let policies: Vec<Vec<f32>> = leaves.iter().map(compact_prior).collect();
        let values: Vec<f32> = (0..leaves.len()).map(|i| 0.05 * (i % 3) as f32).collect();
        tree.expand_and_backup(&policies, &values);
        spent += leaves.len();
    }
    (hits, revisits)
}

fn replay(moves: &[(i32, i32)]) -> Board {
    let mut b = Board::new();
    b.set_legal_move_radius(8);
    for &(q, r) in moves {
        b.apply_move(q, r).expect("a legal fixture move");
    }
    b
}

#[test]
fn a_two_stone_turn_transposes_and_every_table_hit_is_counted() {
    // P2 to place two: A then B and B then A reach one position.
    let (hits, revisits) = drive(&replay(&[(0, 0)]), 400);
    assert!(
        hits > 0,
        "a compact PUCT search over a two-stone turn never transposed"
    );
    assert_eq!(revisits, 0, "no terminal is reachable this early");
}

/// PLANTED BREAK: count the `cached` branch's terminal revisits too and the oracle in `drive` reds.
#[test]
fn a_terminal_revisit_is_not_a_table_hit() {
    // P2 holds five on r = 3 and places two: its finish is a terminal child the prior piles onto.
    let board = replay(&[
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
    ]);
    assert_eq!(analyze(&board).terminal, Some(Terminal::Win));
    let (_hits, revisits) = drive(&board, 200);
    assert!(
        revisits > 0,
        "the search never revisited its winning child, so the case is not exercised"
    );
}
