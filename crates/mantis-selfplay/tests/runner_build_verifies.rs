//! A self-play game builds every leaf with the producer verify: the one test in this binary, so the skip count is exact.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_graph::unverified_builds;
use mantis_search::SearchKind;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const LEAF_BATCH: usize = 8;

#[test]
fn a_mock_served_self_play_game_never_skips_the_producer_verify() {
    const ENCODING: &str = "gnn_axis_r8";
    let spec = lookup_or_panic(ENCODING);
    let before = unverified_builds();
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 12,
        n_simulations: 32,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        search_kind: SearchKind::Gumbel,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("runner constructs");
    let served = Arc::new(AtomicUsize::new(0));
    let producer = common::spawn_uniform_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        served.clone(),
        LEAF_BATCH,
    );
    runner.start();
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut records = 0usize;
    while Instant::now() < deadline && records < 6 && runner.fatal_defect().is_none() {
        records += runner.drain_graph_records().expect("unpoisoned").len();
        thread::sleep(Duration::from_millis(5));
    }
    runner.stop();
    producer.join().expect("producer exits");
    assert!(
        runner.fatal_defect().is_none(),
        "{:?}",
        runner.fatal_defect()
    );
    assert!(
        records >= 6,
        "only {records} searched plies: the game never built its leaves"
    );
    assert_eq!(
        unverified_builds(),
        before,
        "a self-play leaf build skipped its verify"
    );
}
