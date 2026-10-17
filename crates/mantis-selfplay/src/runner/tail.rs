//! A decided tail: from the root offence's first decided stone the game runs out with no search, a row per ply.

use std::sync::atomic::{AtomicU64, Ordering};

use rand::rngs::ThreadRng;
use rand::RngExt;

use mantis_core::{Board, Player};
use mantis_search::mcts::TacticsError;
use mantis_search::MCTSTree;

use crate::replay::hexg::GraphRecord;

use super::record::record_tail_dispatch;
use super::search_drive::{
    count_live_failure, infer_and_expand_graph, FatalDefectLatch, InferContext, LeafSelection,
    MoveAccumulators, MoveOutcome,
};
use super::stats::MoveRow;
use super::{MoveArm, ARM_TAIL};

/// A game's decided tail: the side its proof wins for, and the loser's stored second stone with the ply it is owed at.
pub(crate) struct Tail {
    winner: Player,
    owed: Option<(u32, (i32, i32))>,
}

impl Tail {
    pub(crate) fn new(winner: Player) -> Self {
        Self { winner, owed: None }
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
    tail: &mut Tail,
    full_search_prob: f32,
    rng: &mut ThreadRng,
    agg_trunk_sz: i32,
    infer: InferContext,
    accumulators: MoveAccumulators,
    fatal_latch: FatalDefectLatch,
) -> Option<MoveOutcome> {
    tree.new_game(board.clone());
    if board.current_player == tail.winner {
        let stone = match tree.root_offence() {
            Ok(Some(stone)) => stone,
            Ok(None) | Err(TacticsError::ProofStoneIllegal { .. }) => {
                accumulators.tactics_totals.add_move(MoveRow::TailExit);
                return None;
            }
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
            full_search_prob,
            rng,
            accumulators,
            fatal_latch,
        ));
    }
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
        false,
        true,
        accumulators,
        fatal_latch,
    ))
}

/// The winner's tail stone, its row the one-hot at +1; its searches-skipped row reads the arm `play_one_move` would draw.
#[allow(clippy::too_many_arguments)]
pub(crate) fn play_winner_stone(
    tree: &MCTSTree,
    board: &mut Board,
    record: TailRecord<'_>,
    stone: (i32, i32),
    full_search_prob: f32,
    rng: &mut ThreadRng,
    accumulators: MoveAccumulators,
    fatal_latch: FatalDefectLatch,
) -> MoveOutcome {
    // A quick draw's decided root was already unsearched, so only a full draw's search is skipped here.
    let skipped = full_search_prob <= 0.0 || rng.random::<f32>() < full_search_prob;
    play_tail_stone(
        tree,
        board,
        record,
        stone,
        true,
        skipped,
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
    if served == 0 || !root.is_expanded() {
        count_live_failure(infer.running, root_expansion_failed);
        return Err(MoveOutcome::Continue);
    }
    let version = infer.model_version.load(Ordering::Relaxed);
    if !version_seen.contains(&version) {
        version_seen.push(version);
    }
    let first = root.first_child as usize;
    let mut best: Option<(f32, (i32, i32))> = None;
    for node in &tree.pool[first..first + usize::from(root.n_children)] {
        if best.is_none_or(|(prior, _)| node.prior > prior) {
            best = Some((node.prior, node.cell()));
        }
    }
    best.map(|(_, cell)| cell).ok_or(MoveOutcome::Continue)
}

/// Write the tail row before the stone (the winner's one-hot, the loser's value-only), then play it and count the ply.
#[allow(clippy::too_many_arguments)]
fn play_tail_stone(
    tree: &MCTSTree,
    board: &mut Board,
    record: TailRecord<'_>,
    stone: (i32, i32),
    winner_row: bool,
    search_skipped: bool,
    accumulators: MoveAccumulators,
    fatal_latch: FatalDefectLatch,
) -> MoveOutcome {
    if let Err(err) = record_tail_dispatch(board, record.graph_records, winner_row.then_some(stone))
    {
        fatal_latch.store(err.to_string());
        return MoveOutcome::Break;
    }
    let totals = accumulators.tactics_totals;
    totals.add_search(&tree.tactics_counters());
    if record.graph_records.last().is_some_and(|r| r.proven_tail) {
        totals.add_move(MoveRow::ProvenTailRow);
    }
    if board.apply_move(stone.0, stone.1).is_err() {
        return MoveOutcome::Break;
    }
    record.move_history.push(stone);
    record.move_arms.push((0, ARM_TAIL));
    totals.add_move(MoveRow::ProvenTailPly);
    if search_skipped {
        totals.add_move(MoveRow::TailSearchSkipped);
    }
    accumulators
        .positions_generated
        .fetch_add(1, Ordering::Relaxed);
    MoveOutcome::Played
}
