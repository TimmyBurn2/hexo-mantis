//! A PUCT select call counts every descent it backs up against its `n`, whatever backed it: the net, the table or a
//! terminal. A budget loop that sums the three spends exactly its budget, and the root's visits are that sum.

use mantis_core::Board;
use mantis_search::mcts::TacticsConfig;
use mantis_search::MCTSTree;

mod common;

use common::compact_prior;

const LEAF_BATCH: usize = 8;

/// The design's leaf budgets without a root offence or audit: solver terminals, and no decided root.
const LEAF_TACTICS: TacticsConfig = TacticsConfig {
    leaf_turns: 2,
    leaf_nodes: 64,
    root_turns: 8,
    root_nodes: 0,
    audit: None,
};

/// One search's counted descents by head.
#[derive(Default)]
struct Spent {
    served: usize,
    inline: usize,
    table: usize,
}

impl Spent {
    fn total(&self) -> usize {
        self.served + self.inline + self.table
    }
}

/// Drive `sims` descents as the budget loops do; every call must count at least one and at most its `n`.
fn search(board: &Board, sims: usize, tactics: Option<TacticsConfig>) -> (MCTSTree, Spent) {
    let mut tree = MCTSTree::new(1.5);
    tree.configure_quiescence(false, 0.0);
    tree.configure_tactics(tactics);
    tree.new_game(board.clone());
    let mut spent = Spent::default();
    while spent.total() < sims {
        let n = LEAF_BATCH.min(sims - spent.total());
        let leaves = tree
            .select_leaves(n)
            .expect("the board replays every selected cell");
        let (inline, table) = (tree.last_inline_descents(), tree.last_tt_hits());
        let counted = leaves.len() + inline + table;
        assert!(
            (1..=n).contains(&counted),
            "a call of n = {n} counted {} returned + {inline} inline + {table} table: a call that counts \
             nothing ends the search early, and one that counts past n overspends it",
            leaves.len()
        );
        if !leaves.is_empty() {
            let policies: Vec<Vec<f32>> = leaves.iter().map(compact_prior).collect();
            let values: Vec<f32> = (0..leaves.len()).map(|i| 0.05 * (i % 3) as f32).collect();
            tree.expand_and_backup(&policies, &values);
        }
        spent.served += leaves.len();
        spent.inline += inline;
        spent.table += table;
    }
    (tree, spent)
}

fn replay(moves: &[(i32, i32)]) -> Board {
    let mut b = Board::new();
    b.set_legal_move_radius(8);
    for &(q, r) in moves {
        b.apply_move(q, r).expect("a legal fixture move");
    }
    b
}

/// The count is exact: each counted descent backed one value up through the root, and no uncounted one did.
fn assert_root_saw_every_descent(tree: &MCTSTree, spent: &Spent, sims: usize) {
    assert_eq!(spent.total(), sims, "the search spent its budget exactly");
    assert_eq!(
        tree.root_visits() as usize,
        sims,
        "the root saw {} backups against {} served + {} inline + {} table counted",
        tree.root_visits(),
        spent.served,
        spent.inline,
        spent.table
    );
}

/// PLANTED BREAK: drop the table branch's `i += 1` in `MCTSTree::select_leaves` and every case here reds.
#[test]
fn a_transposing_search_counts_its_table_hits_against_the_budget() {
    // P2 to place two: A then B and B then A reach one position.
    let (tree, spent) = search(&replay(&[(0, 0)]), 400, None);
    assert!(
        spent.table > 0,
        "the search never transposed, so the table case is not exercised"
    );
    assert_root_saw_every_descent(&tree, &spent, 400);
}

/// The plain head's early end: its winning child's revisits back up on the table path, and a call of them counted nothing.
#[test]
fn a_terminal_revisit_on_the_table_path_is_a_counted_descent() {
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
    let (tree, spent) = search(&board, 256, None);
    assert!(
        spent.inline > 0,
        "no descent ended at the terminal, so the case is not exercised"
    );
    assert_root_saw_every_descent(&tree, &spent, 256);
}

#[test]
fn solver_terminals_and_table_hits_count_alike_with_tactics_on() {
    // P1 to place two against P2's open four: forced children, then decided leaves below them.
    let board = replay(&[(0, 0), (10, 5), (11, 5), (1, 0), (2, 0), (12, 5), (13, 5)]);
    let (tree, spent) = search(&board, 256, Some(LEAF_TACTICS));
    assert!(
        spent.inline > 0,
        "no leaf was decided, so the solver case is not exercised"
    );
    assert_root_saw_every_descent(&tree, &spent, 256);
    let rows = tree.tactics_counters();
    assert_eq!(
        rows.descents as usize,
        spent.total(),
        "the tactics rows count the same descents"
    );
    assert_eq!(
        rows.served_leaves + rows.solver_terminals() + rows.terminal_revisits + rows.table_hits,
        rows.descents,
        "the rows add up: {rows:?}"
    );
}
