//! The audit walks the root in the kind's order, one helper for deploy and self-play: Halving's ranking under Gumbel.

use mantis_search::MCTSTree;

#[allow(dead_code)]
mod common;

/// PLANTED BREAK: return `None` for a Gumbel tree in `MCTSTree::audit_order` and this reds.
#[test]
fn gumbel_audits_in_the_halving_ranking_and_puct_by_visits() {
    let board = common::r8_board();
    let (tree, state) =
        common::gumbel_search_with_state(&board, &common::stub_policy(), 7, 64, 16, 1.0);
    let ranking = state.ranking(&tree, tree.q_sigma());
    assert_eq!(tree.audit_order(Some(&state)), Some(ranking));
    assert_eq!(
        tree.audit_order(None),
        None,
        "no root state, no halving order"
    );
    let mut puct = MCTSTree::new(1.5);
    puct.new_game(board);
    assert_eq!(puct.audit_order(Some(&state)), None, "PUCT ranks by visits");
}
