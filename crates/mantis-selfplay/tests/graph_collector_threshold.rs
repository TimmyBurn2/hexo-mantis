//! The collector's saturation threshold is DERIVED from the run's achievable supply, so it can
//! never sit above what the workers can put in flight and send every pop to its deadline.
//!
//! The planted break each test names is the FROZEN rule, `batch_size / 2` alone.

use std::thread;
use std::time::{Duration, Instant};

use mantis_graph::{build_axis_graph, BuildParams, StoneList};
use mantis_selfplay::queues::{saturation_threshold, GraphQueue};

/// The frozen threshold — the planted break. Kept here so every assertion below can state
/// what it would have been, rather than asserting a bare number.
fn frozen_threshold(batch_size: usize) -> usize {
    batch_size / 2
}

#[test]
fn threshold_is_clamped_to_the_declared_supply() {
    // The ledger's own configuration: inference_batch_size 64, dev's minted supply
    // n_workers 1 x leaf_batch_size 8 = 8.
    let derived = saturation_threshold(64, 8);
    assert_eq!(derived, 8, "the threshold clamps to the supply, not the half-batch");
    assert_eq!(frozen_threshold(64), 32);
    assert!(
        derived < frozen_threshold(64),
        "F-1's regime: the frozen threshold sits above a supply that can never reach it"
    );
}

#[test]
fn threshold_is_unchanged_where_supply_already_clears_it() {
    // `recal-mint-20260828`'s regime: 12 workers x 8 = 96, comfortably over the half-batch.
    assert_eq!(saturation_threshold(64, 96), frozen_threshold(64));
    // And exactly at the boundary, 4 x 8 = 32.
    assert_eq!(saturation_threshold(64, 32), frozen_threshold(64));
}

#[test]
fn undeclared_supply_keeps_the_frozen_threshold() {
    // `0` is the "no supply declared" posture every non-runner construction takes; it must
    // not be read as "supply of zero", which would make every pop return on the first graph.
    assert_eq!(saturation_threshold(64, 0), frozen_threshold(64));
    assert_eq!(saturation_threshold(8, 0), frozen_threshold(8));
}

fn one_graph() -> mantis_graph::AxisGraph {
    build_axis_graph(
        &StoneList { stones: vec![(0, 0, 1), (1, 0, -1), (2, 0, 1)] },
        &BuildParams {
            win_length: 6,
            radius: 6,
            current_player: 1,
            moves_remaining: 2,
            trunk_size: 19,
        },
    )
}

/// The mechanism, end to end: with the supply declared, a pop whose queue holds exactly the
/// supply returns WELL INSIDE its deadline. Under the frozen rule the same pop would block for
/// the whole `max_wait_ms`.
#[test]
fn a_pop_at_the_declared_supply_returns_before_its_deadline() {
    const SUPPLY: usize = 8;
    const BATCH: usize = 64;
    const WAIT_MS: u64 = 200;

    let q = GraphQueue::with_contract_version_and_supply(1, SUPPLY);
    assert_eq!(q.max_in_flight(), SUPPLY);
    assert!(
        saturation_threshold(BATCH, SUPPLY) <= SUPPLY,
        "the derived threshold must be reachable from the supply, or this test proves nothing"
    );

    // One "worker": submit the whole leaf batch and block, exactly as the production
    // consumer does.
    let qw = q.clone();
    let worker = thread::spawn(move || {
        let graphs: Vec<_> = (0..SUPPLY).map(|_| one_graph()).collect();
        let _ = qw.submit_graphs_and_wait(graphs);
    });

    // Let the worker enqueue before the collector looks.
    thread::sleep(Duration::from_millis(50));
    let t0 = Instant::now();
    let popped = q.pop_graph_batch(BATCH, WAIT_MS);
    let elapsed = t0.elapsed();

    assert_eq!(popped.len(), SUPPLY, "the pop takes the whole supply");
    assert!(
        elapsed < Duration::from_millis(WAIT_MS / 2),
        "the pop returned in {elapsed:?} against a {WAIT_MS} ms deadline; under the frozen \
         threshold of {} against a supply of {SUPPLY} it would have run to the deadline",
        frozen_threshold(BATCH),
    );

    // Release the blocked worker so the test does not hang.
    q.close();
    let _ = worker.join();
}
