//! The playout cap's quick arm runs Sequential Halving over its own `gumbel_m_quick`, the full arm over `gumbel_m`.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::SearchKind;
use mantis_selfplay::runner::{GameResultRow, SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const ENCODING: &str = "gnn_axis_r8";
const LEAF_BATCH: usize = 8;
const M_FULL: usize = 16;
const M_QUICK: usize = 4;

/// One worker, every game sampled, both arms drawn, until `games` games finished.
fn drive(m_quick: usize, games: usize) -> Vec<GameResultRow> {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 16,
        n_simulations: 64,
        n_sims_quick: 16,
        full_search_prob: 0.5,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        search_stats_every: 1,
        search_kind: SearchKind::Gumbel,
        gumbel_m: M_FULL,
        gumbel_m_quick: m_quick,
        quiescence_enabled: false,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");
    let producer = common::spawn_compact_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        Arc::new(AtomicUsize::new(0)),
        LEAF_BATCH,
    );
    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut out = Vec::new();
    while Instant::now() < deadline && out.len() < games && runner.fatal_defect().is_none() {
        runner.drain_graph_records().expect("unpoisoned");
        out.extend(runner.drain_game_results().expect("unpoisoned"));
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    assert!(defect.is_none(), "latched a fatal defect: {defect:?}");
    assert!(
        out.len() >= games,
        "only {} games inside the budget",
        out.len()
    );
    out
}

/// The widest visited root on each arm: (full arm, quick arm).
fn widest_roots(games: &[GameResultRow]) -> (usize, usize) {
    let (mut full, mut quick) = (0usize, 0usize);
    for game in games {
        let stats = game
            .9
            .as_ref()
            .expect("search_stats_every = 1 samples every game");
        let searched = game.8.iter().filter(|&&(sims, _)| sims > 0);
        for (&(_, is_full), s) in searched.zip(stats) {
            let widest = if is_full { &mut full } else { &mut quick };
            *widest = (*widest).max(s.3.len());
        }
    }
    (full, quick)
}

#[test]
fn a_quick_root_visits_at_most_its_own_m_and_a_full_root_more() {
    let (full, quick) = widest_roots(&drive(M_QUICK, 6));
    assert!(
        (2..=M_QUICK).contains(&quick),
        "the quick arm visited {quick} root children at m {M_QUICK}"
    );
    assert!(
        full > M_QUICK,
        "the full arm visited at most {full}: it ran at the quick arm's m"
    );
}

#[test]
fn at_the_full_arms_m_the_quick_arm_is_not_capped_below_it() {
    let (_, quick) = widest_roots(&drive(M_FULL, 6));
    assert!(
        quick > M_QUICK,
        "at m_quick = m the quick arm still visited only {quick}"
    );
}
