//! A self-play row's future-occupancy field is its game's next placements, re-derived by replaying the game's moves.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_core::{Board, BoardGeometry};
use mantis_encoding::lookup_or_panic;
use mantis_selfplay::replay::hexg::{GraphRecord, HexgBuffer, FUTURE_HORIZON};
use mantis_selfplay::runner::{GameResultRow, SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const ENCODING: &str = "gnn_axis_r8";
const LEAF_BATCH: usize = 8;

/// One worker under `producer` until `games` games finish: their result rows and every row they drained, in order.
fn drive(
    ply_cap: usize,
    games: usize,
    producer: fn(
        mantis_selfplay::queues::GraphQueue,
        usize,
        Arc<AtomicUsize>,
        usize,
    ) -> thread::JoinHandle<()>,
) -> (Vec<GameResultRow>, Vec<GraphRecord>) {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: ply_cap,
        n_simulations: 16,
        n_sims_quick: 4,
        full_search_prob: 0.5,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 2,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");
    let served = Arc::new(AtomicUsize::new(0));
    let handle = producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        served,
        LEAF_BATCH,
    );
    runner.start();
    let deadline = Instant::now() + Duration::from_secs(300);
    let (mut results, mut records) = (Vec::new(), Vec::new());
    while Instant::now() < deadline && results.len() < games && runner.fatal_defect().is_none() {
        // Rows first, then results: a game's result is never drained before its rows.
        records.extend(runner.drain_graph_records().expect("unpoisoned"));
        results.extend(runner.drain_game_results().expect("unpoisoned"));
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    handle.join().expect("producer exits");
    records.extend(runner.drain_graph_records().expect("unpoisoned"));
    results.extend(runner.drain_game_results().expect("unpoisoned"));
    assert!(
        defect.is_none(),
        "the drive latched a fatal defect: {defect:?}"
    );
    assert!(
        results.len() >= games,
        "only {} games finished",
        results.len()
    );
    (results, records)
}

/// Each placement's player, read off a board the moves are replayed on (the side to move before each, never parity).
fn placers(moves: &[(i32, i32)]) -> Vec<i8> {
    let spec = lookup_or_panic(ENCODING);
    let mut board = Board::with_geometry(BoardGeometry {
        legal_move_radius: spec.legal_move_radius as i32,
        cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
    });
    moves
        .iter()
        .map(|&(q, r)| {
            let who = board.current_player as i8;
            board.apply_move(q, r).expect("a played move replays");
            who
        })
        .collect()
}

/// The oracle: every row's field is the next placements of the game, `None` refused.
fn check(game: &GameResultRow, rows: &[&GraphRecord]) -> Result<(), String> {
    let moves = &game.2;
    let who = placers(moves);
    for r in rows {
        let ply = usize::from(r.ply_index);
        let end = (ply + FUTURE_HORIZON).min(moves.len());
        let want: Vec<(i16, i16, i8)> = (ply..end)
            .map(|j| (moves[j].0 as i16, moves[j].1 as i16, who[j]))
            .collect();
        if r.future_stones.as_ref() != Some(&want) {
            return Err(format!(
                "ply {ply}: field {:?} is not the game's next placements {want:?}",
                r.future_stones
            ));
        }
    }
    Ok(())
}

/// Each finished game beside its rows: one worker finalizes in order, each game's rows contiguous under one id.
fn paired<'a>(
    results: &'a [GameResultRow],
    records: &'a [GraphRecord],
) -> Vec<(&'a GameResultRow, Vec<&'a GraphRecord>)> {
    let mut by_game: Vec<(i64, Vec<&GraphRecord>)> = Vec::new();
    for r in records {
        match by_game.last_mut() {
            Some((id, rows)) if *id == r.game_id => rows.push(r),
            _ => by_game.push((r.game_id, vec![r])),
        }
    }
    assert_eq!(
        by_game.len(),
        results.len(),
        "every finished game drained its rows"
    );
    results
        .iter()
        .zip(by_game.into_iter().map(|(_, rows)| rows))
        .collect()
}

/// At a win and at the cap every row is its game's next placements, the ring takes it, and both lengths occur.
#[test]
fn every_rows_field_is_its_games_next_placements_at_a_win_and_at_the_cap() {
    let (decided, decided_rows) = drive(120, 3, common::spawn_compact_producer);
    let (capped, capped_rows) = drive(26, 2, common::spawn_uniform_producer);
    assert!(decided.iter().any(|g| g.4 == 0), "no game ended in a win");
    assert!(capped.iter().any(|g| g.4 == 2), "no game ended at the cap");
    let mut lengths = std::collections::HashSet::new();
    let mut ring = HexgBuffer::new(4096, ENCODING, 64).expect("ring");
    for (results, records) in [(&decided, &decided_rows), (&capped, &capped_rows)] {
        for (game, rows) in paired(results, records) {
            assert!(
                rows.iter().all(|r| usize::from(r.ply_index) >= 2),
                "opening plies make no row"
            );
            check(game, &rows).unwrap_or_else(|e| panic!("game of {} plies: {e}", game.0));
            for r in rows {
                lengths.insert(r.future_stones.as_ref().map_or(0, Vec::len) == FUTURE_HORIZON);
                ring.push_record_impl(r, r.game_id)
                    .expect("the ring takes the producer's row");
            }
        }
    }
    assert_eq!(
        lengths.len(),
        2,
        "both a full horizon and a game's end must occur"
    );
}

/// PLANTED BREAK — a row whose field is one placement off its game (the field of the row before) reds the oracle.
#[test]
fn a_row_whose_field_disagrees_with_its_game_reds() {
    let (results, records) = drive(26, 1, common::spawn_uniform_producer);
    let (game, rows) = paired(&results, &records).swap_remove(0);
    check(game, &rows).expect("the producer's rows agree with their game");
    let mut shifted = rows[1].clone();
    shifted.future_stones = rows[0].future_stones.clone();
    let err = check(game, &[&shifted]).expect_err("a field from the wrong ply must red");
    assert!(err.contains("next placements"), "{err}");
}
