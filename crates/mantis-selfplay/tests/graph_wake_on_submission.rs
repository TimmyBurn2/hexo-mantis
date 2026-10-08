//! A pop wakes once every declared submitter has a batch queued; the planted break is the leaf-count rule alone.

use std::thread;
use std::time::{Duration, Instant};

use mantis_graph::{build_axis_graph, AxisGraph, BuildParams, StoneList};
use mantis_selfplay::queues::{GraphQueue, WakeCounts};

const BATCH: usize = 64;
const SUPPLY: usize = 8;
const WAIT_MS: u64 = 900;

fn one_graph() -> AxisGraph {
    build_axis_graph(
        &StoneList {
            stones: vec![(0, 0, 1), (1, 0, -1), (2, 0, 1)],
        },
        &BuildParams {
            win_length: 6,
            radius: 6,
            current_player: 1,
            moves_remaining: 2,
            trunk_size: 19,
            empty_edges: mantis_graph::EmptyEdges::Kept,
        },
    )
}

/// One blocked submitter of `n` graphs, released when the test closes the queue.
fn submit(q: &GraphQueue, n: usize) -> thread::JoinHandle<()> {
    let qw = q.clone();
    thread::spawn(move || {
        let _ = qw.submit_graphs_and_wait((0..n).map(|_| one_graph()).collect());
    })
}

fn timed_pop(q: &GraphQueue, max: usize) -> (usize, Duration) {
    let t0 = Instant::now();
    let popped = q.pop_graph_batch(max, WAIT_MS);
    (popped.len(), t0.elapsed())
}

#[test]
fn a_lone_submitters_short_batch_pops_before_its_deadline() {
    let q = GraphQueue::for_submitters(1, SUPPLY, 1);
    let worker = submit(&q, 3);
    thread::sleep(Duration::from_millis(50));
    let (n, elapsed) = timed_pop(&q, BATCH);
    assert_eq!(n, 3, "the pop takes the submitter's whole batch");
    assert!(
        elapsed < Duration::from_millis(WAIT_MS / 3),
        "the pop took {elapsed:?}; under the leaf-count rule alone 3 leaves against a threshold \
         of {SUPPLY} run to the {WAIT_MS} ms deadline"
    );
    assert_eq!(
        q.wake_counts(),
        WakeCounts {
            all_submitted: 1,
            ..WakeCounts::default()
        },
        "the wake's own fire count records the pop"
    );
    q.close();
    let _ = worker.join();
}

#[test]
fn an_undeclared_queue_keeps_the_deadline() {
    let q = GraphQueue::with_contract_version_and_supply(1, SUPPLY);
    let worker = submit(&q, 3);
    thread::sleep(Duration::from_millis(50));
    let (n, elapsed) = timed_pop(&q, BATCH);
    assert_eq!(n, 3);
    assert!(
        elapsed >= Duration::from_millis(WAIT_MS - 30),
        "an undeclared queue must not wake early ({elapsed:?})"
    );
    assert_eq!(
        q.wake_counts(),
        WakeCounts {
            deadline: 1,
            ..WakeCounts::default()
        }
    );
    q.close();
    let _ = worker.join();
}

#[test]
fn one_of_two_declared_submitters_waits_and_both_pop_at_once() {
    let q = GraphQueue::for_submitters(1, SUPPLY, 2);
    let first = submit(&q, 2);
    thread::sleep(Duration::from_millis(50));
    let (n, elapsed) = timed_pop(&q, BATCH);
    assert_eq!(n, 2);
    assert!(
        elapsed >= Duration::from_millis(WAIT_MS - 30),
        "one of two must wait ({elapsed:?})"
    );

    let a = submit(&q, 2);
    let b = submit(&q, 2);
    thread::sleep(Duration::from_millis(50));
    let (n, elapsed) = timed_pop(&q, BATCH);
    assert_eq!(n, 4, "both batches ride one pop");
    assert!(
        elapsed < Duration::from_millis(WAIT_MS / 3),
        "both queued must not wait ({elapsed:?})"
    );
    assert_eq!(
        q.wake_counts(),
        WakeCounts {
            all_submitted: 1,
            deadline: 1,
            ..WakeCounts::default()
        }
    );
    q.close();
    for w in [first, a, b] {
        let _ = w.join();
    }
}

#[test]
fn a_batch_split_across_pops_counts_as_queued_until_its_last_leaf_pops() {
    let q = GraphQueue::for_submitters(1, SUPPLY, 2);
    let first = submit(&q, 5);
    thread::sleep(Duration::from_millis(50));
    // A pop smaller than the batch leaves three of its leaves queued, and the batch with them.
    let (n, _) = timed_pop(&q, 2);
    assert_eq!(n, 2);
    let second = submit(&q, 1);
    thread::sleep(Duration::from_millis(50));
    let (n, elapsed) = timed_pop(&q, BATCH);
    assert_eq!(n, 4, "the rest of the first batch and the whole second");
    assert!(
        elapsed < Duration::from_millis(WAIT_MS / 3),
        "two batches queued against two submitters must not wait ({elapsed:?})"
    );
    q.close();
    for w in [first, second] {
        let _ = w.join();
    }
}

#[test]
fn the_leaf_threshold_still_fires_first_when_it_is_met() {
    let q = GraphQueue::for_submitters(1, SUPPLY, 4);
    let worker = submit(&q, SUPPLY);
    thread::sleep(Duration::from_millis(50));
    let (n, elapsed) = timed_pop(&q, BATCH);
    assert_eq!(n, SUPPLY);
    assert!(elapsed < Duration::from_millis(WAIT_MS / 3));
    assert_eq!(
        q.wake_counts(),
        WakeCounts {
            threshold: 1,
            ..WakeCounts::default()
        }
    );
    q.close();
    let _ = worker.join();
}
