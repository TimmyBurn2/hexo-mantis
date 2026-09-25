//! A 1-in-N game's result row carries, per searched ply, the root as the search left
//! it — the record the forced-move census could not read from the ring.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::SearchKind;
use mantis_selfplay::runner::{GameResultRow, SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const PLY_CAP: usize = 6;
const N_SIMS_QUICK: usize = 8;
const N_SIMS_FULL: usize = 24;
const ENCODING: &str = "gnn_axis_r8";

fn drive(search_stats_every: usize, want_games: usize) -> Vec<GameResultRow> {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: PLY_CAP,
        n_simulations: N_SIMS_QUICK,
        leaf_batch_size: 4,
        random_opening_plies: 0,
        search_stats_every,
        dirichlet_enabled: true,
        search_kind: SearchKind::Gumbel,
        quiescence_enabled: false,
        full_search_prob: 0.5,
        n_sims_quick: N_SIMS_QUICK,
        n_sims_full: N_SIMS_FULL,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");
    let served = Arc::new(AtomicUsize::new(0));
    let producer =
        common::spawn_uniform_producer(runner.graph_producer(), spec.policy_logit_count, served, 4);
    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut games = Vec::new();
    while Instant::now() < deadline {
        games.extend(runner.drain_game_results());
        if games.len() >= want_games || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    games.extend(runner.drain_game_results());
    assert!(defect.is_none(), "latched a fatal defect: {defect:?}");
    assert!(
        games.len() >= want_games,
        "only {} games inside the budget",
        games.len()
    );
    games
}

#[test]
fn one_in_n_games_carry_a_root_per_searched_ply_and_the_rest_carry_none() {
    let games = drive(2, 8);
    let mut with = 0usize;
    let mut without = 0usize;
    for (plies, _w, moves, _wk, _t, _mn, _mx, _d, arms, stats) in &games {
        match stats {
            None => without += 1,
            Some(rows) => {
                with += 1;
                let searched = arms.iter().filter(|&&(sims, _)| sims > 0).count();
                assert_eq!(
                    rows.len(),
                    searched,
                    "one entry per SEARCHED ply of a {plies}-ply game"
                );
                for (i, (ply, root_value, root_raw, children)) in rows.iter().enumerate() {
                    assert_eq!(*ply as usize, i, "plies are the searched moves in order");
                    assert!((-1.0..=1.0).contains(root_value));
                    assert!(
                        root_raw.is_some(),
                        "the Gumbel kind stores the raw root value"
                    );
                    assert!(!children.is_empty(), "a searched root has visited children");
                    for &(cell, visits, q, prior) in children {
                        assert!(visits >= 1, "only the SUPPORT is stored");
                        assert!((-1.0..=1.0).contains(&q));
                        assert!((0.0..=1.0).contains(&prior));
                        assert!(!moves.is_empty() && cell != (i32::MIN, i32::MIN));
                    }
                }
            }
        }
    }
    assert!(
        with >= 3 && without >= 3,
        "at every=2 both classes appear: with={with} without={without}"
    );
}

#[test]
fn zero_turns_the_producer_off() {
    let games = drive(0, 4);
    assert!(games.iter().all(|g| g.9.is_none()));
}

#[test]
fn every_game_is_sampled_at_one_and_the_played_move_is_in_the_support() {
    let games = drive(1, 3);
    for (_p, _w, moves, _wk, _t, _mn, _mx, _d, _arms, stats) in &games {
        let rows = stats.as_ref().expect("every=1 samples every game");
        // The row records what the search saw: the played move is among the visited candidates.
        for (ply, _v, _raw, children) in rows {
            let played = moves[*ply as usize];
            assert!(
                children.iter().any(|c| c.0 == played),
                "ply {ply}: the played move {played:?} is in the support"
            );
        }
    }
}
