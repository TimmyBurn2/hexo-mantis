//! A PUCT descent that reaches a position its search's TT already holds is counted as a table hit, no leaf served.
//! Two-stone turns transpose (A then B is B then A), so a PUCT drive hits the table; Gumbel's forced select has no
//! table path and counts none.
//!
//! PLANTED BREAK: drop the `tt_hits += 1` in `MCTSTree::select_leaves` and the PUCT arm reds.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::SearchKind;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const ENCODING: &str = "gnn_axis_r8";
const LEAF_BATCH: usize = 8;
const WANT_RECORDS: usize = 24;

/// Drive one worker under `kind` until `WANT_RECORDS` searched plies are recorded; the runner's TT-hit total.
fn tt_hits_under(kind: SearchKind) -> u64 {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 40,
        n_simulations: 64,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        search_kind: kind,
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
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut records = 0;
    while Instant::now() < deadline && records < WANT_RECORDS && runner.fatal_defect().is_none() {
        records += runner.drain_graph_records().expect("unpoisoned").len();
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    // Read after the join: the producer serves until the queue closes.
    producer.join().expect("producer exits");
    assert!(
        defect.is_none(),
        "{kind:?} latched a fatal defect: {defect:?}"
    );
    assert!(
        records >= WANT_RECORDS,
        "{kind:?}: only {records} searched plies"
    );
    runner.stats_snapshot().tt_hits_total
}

#[test]
fn a_transposing_puct_search_counts_its_table_hits() {
    let hits = tt_hits_under(SearchKind::Puct);
    assert!(
        hits > 0,
        "a PUCT drive over two-stone turns never hit its TT: the count is dead"
    );
}

#[test]
fn a_gumbel_search_has_no_table_path_and_counts_none() {
    assert_eq!(tt_hits_under(SearchKind::Gumbel), 0);
}
