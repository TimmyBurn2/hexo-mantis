// R8 justify: the legs are ONE claim with ONE construction — a seam failure and a drain shutdown
// are the SAME `Err` arriving at the SAME line, separated only by the discriminator under test,
// over one shared producer and switch.
//! Pin the leaf-inference seam: a FAILED inference must never become a search reporting
//! `Completed`, and a drain shutdown must not be mistaken for one.
//!
//! A failure arm that returns 0 lets a search that backed up zero visits reach the target
//! exporter, which manufactures a policy target out of noise-mixed priors — surfacing 100+ plies
//! later as an unrelated refusal. But `stop()` also wakes every in-flight waiter with `Err`, so a
//! fix that made every `Err` run-fatal would report a defect on every clean stop.
//!
//! Killers: restore `Err(_) => return 0` in the seam arm (the failure leg times out RED); drop the
//! `is_closed()` discriminator (the shutdown leg goes RED); tick the wrong counter in the latch.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Requests served OK first, so the failure lands MID-SEARCH rather than at the root expansion.
const SERVE_OK_BEFORE_FAILURE: usize = 6;
const INJECTED_REASON: &str = "Graph inference failed: injected forward failure";

/// What a mock producer does once it has served its healthy prefix.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ThenDo {
    /// Fail every subsequent batch, as the real inference server does on a forward exception.
    Fail,
    /// POP the batch and never answer it: a waiter is then provably blocked, which is what makes
    /// the shutdown legs deterministic rather than timing-hopeful.
    ParkHoldingTheBatch,
}

fn graph_runner() -> SelfPlayRunner {
    SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 12,
        n_simulations: 24,
        leaf_batch_size: 4,
        standard_sims: 0,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        random_opening_plies: 0,
        encoding_name: Some("gnn_axis_v1".to_string()),
        ..Default::default()
    })
    .expect("gnn runner constructs")
}

/// Uniform probs through the PRODUCTION assembly, switching to `after` after the healthy prefix.
fn spawn_graph_producer(
    queue: GraphQueue,
    n_actions: usize,
    served: Arc<AtomicUsize>,
    after: ThenDo,
    parked: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(4, 5);
        if batch.is_empty() {
            if queue.is_closed() {
                break;
            }
            continue;
        }
        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        if served.load(Ordering::Relaxed) >= SERVE_OK_BEFORE_FAILURE {
            match after {
                ThenDo::Fail => {
                    queue.fail_remaining(&ids, INJECTED_REASON);
                    continue;
                }
                ThenDo::ParkHoldingTheBatch => {
                    parked.store(true, Ordering::SeqCst);
                    continue;
                }
            }
        }
        let mut results = Vec::with_capacity(batch.len());
        for (_, g) in &batch {
            let coords: Vec<(i32, i32)> = g
                .legal_node_gather
                .iter()
                .map(|&row| (g.node_coords[row as usize * 2], g.node_coords[row as usize * 2 + 1]))
                .collect();
            let n = coords.len();
            let probs = vec![1.0f32 / n.max(1) as f32; n];
            results.push(
                assemble_ls_from_gnn_probs(n_actions, &probs, &g.policy_scatter_index.0, &coords)
                    .map(|ls| (ls, 0.0f32)),
            );
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        queue.submit_graph_results(&ids, results);
    })
}


fn wait_for_defect(runner: &SelfPlayRunner, secs: u64) -> Option<String> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if let Some(msg) = runner.fatal_defect() {
            return Some(msg);
        }
        thread::sleep(Duration::from_millis(10));
    }
    None
}

fn wait_for(secs: u64, mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if done() {
            return true;
        }
        thread::sleep(Duration::from_millis(10));
    }
    false
}

#[test]
fn injected_graph_inference_failure_dies_loud_and_named_at_the_seam() {
    let spec = lookup_or_panic("gnn_axis_v1");
    let runner = graph_runner();

    // The counter is VISIBLE at 0 before anything runs, which distinguishes "no failures"
    // from "no producer".
    assert_eq!(runner.stats_snapshot().inference_failures_total, 0);
    assert!(runner.fatal_defect().is_none());

    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_graph_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        served.clone(),
        ThenDo::Fail,
        Arc::new(AtomicBool::new(false)),
    );

    runner.start();
    let msg = wait_for_defect(&runner, 120);
    let halted = !runner.is_running();
    let snap = runner.stats_snapshot();
    let drained = runner.drain_graph_records();
    runner.stop();
    producer.join().expect("producer exits");

    assert!(
        served.load(Ordering::Relaxed) >= SERVE_OK_BEFORE_FAILURE,
        "vacuous drive: the producer never served the healthy prefix, so the failure did \
         not land mid-search"
    );
    let msg = msg.expect(
        "an injected graph-inference failure never reached the fatal-defect latch — the \
         seam still degrades a failed batch into a silent skip (M-SEAM-1)",
    );
    assert!(
        msg.contains("InferenceSeamFailure"),
        "the variant name must survive seam → latch → drain face verbatim: {msg}"
    );
    assert!(msg.contains("graph"), "the failing ARM must ride the message: {msg}");
    assert!(
        msg.contains(INJECTED_REASON),
        "the waiter's reason must ride VERBATIM — dropping it is §7.3, the whole point of \
         carrying it: {msg}"
    );
    assert!(halted, "store-then-halt: running must be false once the latch stores (LAW-14)");
    assert_eq!(
        snap.inference_failures_total, 1,
        "the SEAM counter must count this fire (LAW-18)"
    );
    assert_eq!(
        snap.target_integrity_defects, 0,
        "the seam bit BEFORE the exporter — a target-integrity fire here would mean the \
         failed search reached the record dispatch after all, and would also mean the two \
         conjuncts share a counter (M-SEAM-3)"
    );
    assert!(
        drained.is_empty(),
        "{} record(s) reached the buffer from a search whose inference failed",
        drained.len()
    );
}

#[test]
fn graph_drain_shutdown_is_not_an_inference_failure() {
    let spec = lookup_or_panic("gnn_axis_v1");
    let runner = graph_runner();
    let served = Arc::new(AtomicUsize::new(0));
    let parked = Arc::new(AtomicBool::new(false));
    let producer = spawn_graph_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        served.clone(),
        ThenDo::ParkHoldingTheBatch,
        parked.clone(),
    );

    runner.start();
    // Wait until a batch is POPPED and unanswered, so a waiter is provably blocked at close;
    // waiting on `served` instead was measured leaving nothing in flight and passing the killer.
    let blocked = wait_for(120, || parked.load(Ordering::SeqCst));
    runner.stop();
    producer.join().expect("producer exits");

    assert!(
        blocked,
        "vacuous drive: the producer never parked holding a batch, so no waiter was in \
         flight at close and this oracle proves nothing"
    );
    let snap = runner.stats_snapshot();
    assert_eq!(
        snap.inference_failures_total, 0,
        "a clean stop was reported as an inference failure — the seam is classifying on the \
         Err alone instead of on `queue.is_closed()` (M-SEAM-2). Every run would end by \
         reporting a defect it did not have"
    );
    assert!(
        runner.fatal_defect().is_none(),
        "a clean stop latched a fatal defect: {:?}",
        runner.fatal_defect()
    );
}

#[test]
fn an_inference_server_death_that_closes_the_queue_is_a_failure_not_a_shutdown() {
    // `close()` carries no reason and the server closes the batcher from a `finally` on ANY loop
    // exit, so a DYING server would re-enter through the shutdown door as a silent batch-skip.
    // Closing the queue WITHOUT stopping the runner is what that looks like from the worker's
    // side, and is something `stop()` can never produce: it stores `running=false` first.
    let runner = graph_runner();
    let queue = runner.graph_producer();
    let served = Arc::new(AtomicUsize::new(0));
    let parked = Arc::new(AtomicBool::new(false));
    let spec = lookup_or_panic("gnn_axis_v1");
    let producer = spawn_graph_producer(
        queue.clone(),
        spec.policy_logit_count,
        served.clone(),
        ThenDo::ParkHoldingTheBatch,
        parked.clone(),
    );

    runner.start();
    let blocked = wait_for(120, || parked.load(Ordering::SeqCst));
    // The server "dies": the queue closes while the runner is still RUNNING.
    queue.close();
    let msg = wait_for_defect(&runner, 120);
    let snap = runner.stats_snapshot();
    runner.stop();
    producer.join().expect("producer exits");

    assert!(blocked, "vacuous drive: the producer never parked holding a batch");
    let msg = msg.expect(
        "a queue closed by something other than `stop()` was treated as a clean shutdown —          a dying inference server can therefore park every worker in a silent batch-skip          forever, with `running` still true and nothing raised. This is the F-816-9 degrade          re-entering through the shutdown door (R276(a))",
    );
    assert!(msg.contains("InferenceSeamFailure"), "variant name must ride: {msg}");
    assert_eq!(
        snap.inference_failures_total, 1,
        "the seam counter must count a server-death failure like any other"
    );
}
