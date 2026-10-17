//! The tail module's finished-game counts and its exit, pinned off the game loop.

use std::collections::HashMap;

use mantis_core::{Board, Player};
use mantis_search::mcts::TacticsConfig;
use mantis_search::MCTSTree;

use super::{count_finished_tail, winner_offence, GameTails, TacticsTotals};
use crate::replay::hexg::GraphRecord;
use crate::runner::{MoveArm, ARM_FAST, ARM_TAIL};

/// PLANTED BREAK: a finished tail whose loser row was never written reads one ply more than its rows.
#[test]
fn a_finished_tail_counts_its_plies_off_the_arms_and_its_rows_off_the_rows() {
    let arms: Vec<MoveArm> = vec![(64, ARM_FAST), (0, ARM_TAIL), (0, ARM_TAIL), (0, ARM_TAIL)];
    let row = |proven_tail| GraphRecord {
        proven_tail,
        ..GraphRecord::default()
    };
    let read = |rows: &[GraphRecord]| {
        let totals = TacticsTotals::new();
        count_finished_tail(&totals, &arms, rows, &GameTails::default());
        let by: HashMap<&str, u64> = totals.snapshot().into_iter().collect();
        (by["proven_tail_plies"], by["proven_tail_rows"])
    };
    let whole = [row(false), row(true), row(true), row(true)];
    assert_eq!(read(&whole), (3, 3));
    assert_eq!(read(&whole[..3]), (3, 2), "a dropped row is a gap");
}

/// PLANTED BREAK: an offence with no stone closes the tail, and its finished game reports the exit and the skips.
#[test]
fn an_offence_with_no_stone_exits_the_tail_and_its_game_reports_it() {
    let mut tree = MCTSTree::new(1.5);
    tree.configure_tactics(Some(TacticsConfig {
        leaf_turns: 2,
        leaf_nodes: 64,
        root_turns: 2,
        root_nodes: 256,
        audit: None,
    }));
    tree.new_game(Board::new());
    let mut tails = GameTails::default();
    tails.start(Player::One);
    tails.skipped = 3;
    assert_eq!(winner_offence(&mut tree, &mut tails), Ok(None));
    assert!(!tails.in_play(), "an exit closes the tail");
    let totals = TacticsTotals::new();
    count_finished_tail(&totals, &[], &[], &tails);
    let by: HashMap<&str, u64> = totals.snapshot().into_iter().collect();
    assert_eq!((by["tail_exits"], by["tail_search_skipped"]), (1, 3));
}
