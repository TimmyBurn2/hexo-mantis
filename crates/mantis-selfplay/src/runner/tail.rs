//! A decided tail: from the root offence's first decided stone the game runs out with no search, a row per ply.

use std::sync::atomic::{AtomicU64, Ordering};

use rand::rngs::ThreadRng;

use mantis_core::{Board, Player};
use mantis_search::{mcts::TacticsError, MCTSTree};

use crate::replay::hexg::GraphRecord;

use super::record::record_tail_dispatch;
use super::search_drive::{
    count_live_failure, draw_arm, infer_and_expand_graph, FatalDefectLatch, InferContext,
    LeafSelection, MoveAccumulators, MoveOutcome, MovePlayContext,
};
use super::stats::{MoveRow, TacticsTotals};
use super::{MoveArm, ARM_TAIL};

/// A decided tail in play: the side its proof wins for, and the loser's stored second stone with the ply it is owed at.
struct Tail {
    winner: Player,
    owed: Option<(u32, (i32, i32))>,
}

/// A game's decided tails: the one in play, and the searches it skipped and the exits its finished game reports.
#[derive(Default)]
pub(crate) struct GameTails {
    current: Option<Tail>,
    skipped: u64,
    exits: u64,
}

impl GameTails {
    pub(crate) fn in_play(&self) -> bool {
        self.current.is_some()
    }

    /// Start a tail at its winner's first decided stone.
    pub(crate) fn start(&mut self, winner: Player) {
        self.current = Some(Tail { winner, owed: None });
    }
}

/// A decided tail's stone that did not play: run-fatal, its game dropped before it is finalized.
#[derive(Debug)]
pub(crate) struct TailStoneIllegal {
    stone: (i32, i32),
    ply_index: u32,
}

impl std::fmt::Display for TailStoneIllegal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { stone, ply_index } = self;
        write!(
            f,
            "TailStoneIllegal: tail stone {stone:?} at ply_index={ply_index} is not legal"
        )
    }
}

/// The per-game record a tail ply writes into, borrowed from the game loop.
pub(crate) struct TailRecord<'a> {
    pub(crate) graph_records: &'a mut Vec<GraphRecord>,
    pub(crate) move_history: &'a mut Vec<(i32, i32)>,
    pub(crate) move_arms: &'a mut Vec<MoveArm>,
    pub(crate) version_seen: &'a mut Vec<u64>,
}

/// One tail ply, the winner's offence stone or the loser's; `None`: the offence gave no stone, the tree left on this board.
#[allow(clippy::too_many_arguments)]
pub(crate) fn play_tail_ply(
    tree: &mut MCTSTree,
    board: &mut Board,
    record: TailRecord<'_>,
    tails: &mut GameTails,
    ctx: MovePlayContext,
    rng: &mut ThreadRng,
    agg_trunk_sz: i32,
    infer: InferContext,
    accumulators: MoveAccumulators,
    fatal_latch: FatalDefectLatch,
) -> Option<MoveOutcome> {
    tree.new_game(board.clone());
    let winner = tails.current.as_ref()?.winner;
    if board.current_player == winner {
        let stone = match winner_offence(tree, tails) {
            Ok(Some(stone)) => stone,
            Ok(None) => return None,
            Err(err) => {
                fatal_latch.store(err.to_string());
                return Some(MoveOutcome::Break);
            }
        };
        return Some(play_winner_stone(
            tree,
            board,
            record,
            stone,
            ctx,
            rng,
            tails,
            accumulators,
            fatal_latch,
        ));
    }
    let tail = tails.current.as_mut()?;
    let stone = match loser_stone(
        tree,
        board,
        tail,
        &mut *record.version_seen,
        agg_trunk_sz,
        infer,
        accumulators.root_expansion_failed,
        fatal_latch,
    ) {
        Ok(stone) => stone,
        Err(outcome) => return Some(outcome),
    };
    Some(play_tail_stone(
        tree,
        board,
        record,
        stone,
        (false, true),
        &mut tails.skipped,
        accumulators,
        fatal_latch,
    ))
}

/// The winner's offence stone, else the tail's exit: counted and closed, `Ok(None)`; a tactics refusal passes through.
fn winner_offence(
    tree: &mut MCTSTree,
    tails: &mut GameTails,
) -> Result<Option<(i32, i32)>, TacticsError> {
    match tree.root_offence() {
        Ok(Some(stone)) => Ok(Some(stone)),
        Ok(None) | Err(TacticsError::ProofStoneIllegal { .. }) => {
            tails.exits += 1;
            tails.current = None;
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

/// The winner's tail stone, its row the one-hot at +1; its searches-skipped row reads the arm `play_one_move` draws.
#[allow(clippy::too_many_arguments)]
pub(crate) fn play_winner_stone(
    tree: &MCTSTree,
    board: &mut Board,
    record: TailRecord<'_>,
    stone: (i32, i32),
    ctx: MovePlayContext,
    rng: &mut ThreadRng,
    tails: &mut GameTails,
    accumulators: MoveAccumulators,
    fatal_latch: FatalDefectLatch,
) -> MoveOutcome {
    // A quick draw's decided root was already unsearched, so only a full draw's search is skipped here.
    let (full, _) = draw_arm(ctx, rng);
    play_tail_stone(
        tree,
        board,
        record,
        stone,
        (true, full),
        &mut tails.skipped,
        accumulators,
        fatal_latch,
    )
}

/// The loser's owed second stone, else the solver's longest resistance at a turn start, else the net's argmax.
#[allow(clippy::too_many_arguments)]
fn loser_stone(
    tree: &mut MCTSTree,
    board: &Board,
    tail: &mut Tail,
    version_seen: &mut Vec<u64>,
    agg_trunk_sz: i32,
    infer: InferContext,
    root_expansion_failed: &AtomicU64,
    fatal_latch: FatalDefectLatch,
) -> Result<(i32, i32), MoveOutcome> {
    let ply = board.ply.index();
    if let Some((owed_ply, stone)) = tail.owed.take() {
        if owed_ply == ply && board.legal_moves_set().contains(&stone) {
            return Ok(stone);
        }
    }
    if board.moves_remaining == 2 {
        if let Some([first, second]) = tree.tail_resistance() {
            tail.owed = Some((ply + 1, second));
            return Ok(first);
        }
    }
    net_argmax(
        tree,
        version_seen,
        agg_trunk_sz,
        infer,
        root_expansion_failed,
        fatal_latch,
    )
}

/// The net's highest-prior root child with no search: the root's own evaluation, the first of equals in pool order.
fn net_argmax(
    tree: &mut MCTSTree,
    version_seen: &mut Vec<u64>,
    agg_trunk_sz: i32,
    infer: InferContext,
    root_expansion_failed: &AtomicU64,
    fatal_latch: FatalDefectLatch,
) -> Result<(i32, i32), MoveOutcome> {
    let served = match infer_and_expand_graph(tree, LeafSelection::Batch(1), agg_trunk_sz, infer) {
        Ok((served, unserved)) => served + unserved,
        Err(err) => {
            fatal_latch.store_inference_failure(err.to_string());
            return Err(MoveOutcome::Break);
        }
    };
    let root = tree.pool[0];
    if served == 0 || !root.is_expanded() || root.n_children == 0 {
        count_live_failure(infer.running, root_expansion_failed);
        return Err(MoveOutcome::Continue);
    }
    let version = infer.model_version.load(Ordering::Relaxed);
    if !version_seen.contains(&version) {
        version_seen.push(version);
    }
    let first = root.first_child as usize;
    let children = &tree.pool[first..first + usize::from(root.n_children)];
    let mut best = children[0];
    for node in &children[1..] {
        if node.prior > best.prior {
            best = *node;
        }
    }
    Ok(best.cell())
}

/// Write the tail row before the stone (the winner's one-hot, the loser's value-only), then play it.
#[allow(clippy::too_many_arguments)]
fn play_tail_stone(
    tree: &MCTSTree,
    board: &mut Board,
    record: TailRecord<'_>,
    stone: (i32, i32),
    (winner_row, search_skipped): (bool, bool),
    skipped: &mut u64,
    accumulators: MoveAccumulators,
    fatal_latch: FatalDefectLatch,
) -> MoveOutcome {
    if let Err(err) = record_tail_dispatch(board, record.graph_records, winner_row.then_some(stone))
    {
        fatal_latch.store(err.to_string());
        return MoveOutcome::Break;
    }
    accumulators
        .tactics_totals
        .add_search(&tree.tactics_counters());
    if board.apply_move(stone.0, stone.1).is_err() {
        let ply_index = board.ply.index();
        fatal_latch.store(TailStoneIllegal { stone, ply_index }.to_string());
        return MoveOutcome::Break;
    }
    record.move_history.push(stone);
    record.move_arms.push((0, ARM_TAIL));
    *skipped += u64::from(search_skipped);
    accumulators
        .positions_generated
        .fetch_add(1, Ordering::Relaxed);
    MoveOutcome::Played
}

/// A finished game's four tail rows: plies off its arms, rows off its rows (a dropped row is a gap), skips and exits off its tally.
pub(crate) fn count_finished_tail(
    totals: &TacticsTotals,
    move_arms: &[MoveArm],
    rows: &[GraphRecord],
    tails: &GameTails,
) {
    let plies = move_arms
        .iter()
        .filter(|&&(_, arm)| arm == ARM_TAIL)
        .count();
    let flagged = rows.iter().filter(|r| r.proven_tail).count();
    totals.add_moves(MoveRow::ProvenTailPly, plies as u64);
    totals.add_moves(MoveRow::ProvenTailRow, flagged as u64);
    totals.add_moves(MoveRow::TailSearchSkipped, tails.skipped);
    totals.add_moves(MoveRow::TailExit, tails.exits);
}

#[cfg(test)]
#[path = "tail_tests.rs"]
mod tests;
