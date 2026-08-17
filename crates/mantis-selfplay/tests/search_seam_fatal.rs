// R8 justify: the four legs are ONE claim with ONE construction — a seam failure and a
// drain shutdown are the SAME `Err` arriving at the SAME line, and the only thing that
// separates them is the discriminator under test. They share the two mock producers, the
// `ThenDo` switch that is the whole experiment, and the healthy-prefix constant; splitting
// failure from shutdown across files would put the two halves of one flip-set where a
// reader can green one and never run the other.
//! ⊕ F-816-9 Phase C — the SEAM conjunct pin (R275(b) conjunct 1, LAW-14/LAW-18).
//!
//! Subject: `search_drive::infer_and_expand{,_graph}` must never turn a FAILED leaf
//! inference into a search that reports `Completed`. Pre-fix every failure arm was
//! `return 0`: the waiter's reason string travelled back verbatim (D6) and was dropped,
//! the sim loop broke on `n == 0`, `run_mcts_search` returned `Completed`, and a search
//! that backed up zero visits reached the target exporter — which manufactured a policy
//! target out of ε-noise-mixed priors. On the box that presented, 100+ plies later, as a
//! `VisitSlotsExceeded` refusal naming neither the failure nor the leaf (Phase A §4).
//!
//! FLIP-SET (a): an injected inference failure mid-game dies LOUD at the seam and nothing
//! reaches the exporter or the buffer. Driven on BOTH arms, because both carry a failure
//! leg and the counter is published on both (R256: the instrument attaches to the
//! mechanism's measured live path, and that path is arm-independent here).
//!
//! THE DISCRIMINATOR IS THE OTHER HALF OF THE PIN. `stop()` flips `running=false` and then
//! closes both queues, waking every in-flight waiter with `Err` — the §P22/D12
//! drain-shutdown path. A seam fix that made every `Err` run-fatal would turn every clean
//! stop into a reported defect, which is a worse failure than the one being fixed. The
//! shutdown legs below are RED under that naive fix and GREEN under the shipped one.
//!
//! Killers: M-SEAM-1 (restore `Err(_) => return 0` in either arm — the failure legs time
//! out RED); M-SEAM-2 (drop the `is_closed()` discriminator and fail unconditionally — the
//! shutdown legs go RED); M-SEAM-3 (tick `fires` instead of `inference_failures` in the
//! latch — the conjunct-separation asserts go RED).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_selfplay::queues::{DenseQueue, GraphQueue};
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Requests served OK before the injected failure begins. Non-zero so the failure lands
/// MID-SEARCH rather than at the very first root expansion (which would exercise the
/// RootExpansionFailed arm instead of the sim-loop arm).
const SERVE_OK_BEFORE_FAILURE: usize = 6;
const INJECTED_REASON: &str = "Graph inference failed: injected forward failure";
const INJECTED_REASON_DENSE: &str = "injected dense forward failure";

/// What a mock producer does once it has served its healthy prefix.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ThenDo {
    /// Fail every subsequent batch, the way the real inference server does on a forward
    /// exception (`submit_graph_inference_failure` / `submit_failure`).
    Fail,
    /// POP the batch and never answer it, then idle. This is what makes the shutdown legs
    /// DETERMINISTIC rather than timing-hopeful: once a batch is popped and unanswered its
    /// waiter is provably blocked, so `stop()`'s `close()` is guaranteed to wake a real
    /// in-flight waiter with `Err`. A test that merely sleeps and stops may find no waiter
    /// in flight at all and would then pass under the very mutation it exists to kill.
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

fn dense_runner() -> SelfPlayRunner {
    SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 12,
        n_simulations: 16,
        leaf_batch_size: 2,
        standard_sims: 0,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        random_opening_plies: 0,
        encoding_name: Some("v6".to_string()),
        ..Default::default()
    })
    .expect("v6 runner constructs")
}

/// Mock graph producer (the target_wire_carry / target_latch_propagation pattern):
/// uniform probs through the PRODUCTION `assemble_ls_from_gnn_probs`. Once
/// `SERVE_OK_BEFORE_FAILURE` requests have been served it switches to `after` — for
/// `ThenDo::Fail` that is the same `fail_remaining` path the real inference server uses on
/// a forward exception (`inference_server.py` → `submit_graph_inference_failure`).
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

/// Dense sibling. Uniform policy rows; the failure leg is `submit_failure`, the dense
/// queue's own producer-side failure surface.
fn spawn_dense_producer(
    queue: DenseQueue,
    stride: usize,
    served: Arc<AtomicUsize>,
    after: ThenDo,
    parked: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_batch(4, 5);
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
                    queue.submit_failure(&ids, INJECTED_REASON_DENSE);
                    continue;
                }
                ThenDo::ParkHoldingTheBatch => {
                    parked.store(true, Ordering::SeqCst);
                    continue;
                }
            }
        }
        let mut flat: Vec<f32> = Vec::with_capacity(ids.len() * stride);
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(ids.len());
        let mut values: Vec<f32> = Vec::with_capacity(ids.len());
        for _ in &ids {
            let start = flat.len();
            flat.extend(std::iter::repeat_n(1.0f32 / stride as f32, stride));
            ranges.push(start..flat.len());
            values.push(0.0);
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        queue.submit_results(&ids, &Arc::new(flat), &ranges, &values);
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

// ── FLIP-SET (a): the failure legs ───────────────────────────────────────────────────

#[test]
fn injected_graph_inference_failure_dies_loud_and_named_at_the_seam() {
    let spec = lookup_or_panic("gnn_axis_v1");
    let runner = graph_runner();

    // LAW-18 idle posture: the counter is VISIBLE at 0 before anything runs, which is
    // what distinguishes "no failures" from "no producer".
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
fn injected_dense_inference_failure_dies_loud_and_named_at_the_seam() {
    let runner = dense_runner();
    assert_eq!(runner.stats_snapshot().inference_failures_total, 0);

    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_dense_producer(
        runner.dense_producer(),
        runner.policy_len(),
        served.clone(),
        ThenDo::Fail,
        Arc::new(AtomicBool::new(false)),
    );

    runner.start();
    let msg = wait_for_defect(&runner, 120);
    let halted = !runner.is_running();
    let snap = runner.stats_snapshot();
    runner.stop();
    producer.join().expect("producer exits");

    assert!(served.load(Ordering::Relaxed) >= SERVE_OK_BEFORE_FAILURE, "vacuous drive");
    let msg = msg.expect(
        "an injected DENSE inference failure never latched — the dense arm still skips the \
         batch silently. The dense leg matters on its own: it is the arm the R250 absence \
         rule would have wrongly excused this counter from (R256)",
    );
    assert!(msg.contains("InferenceSeamFailure"), "variant name must ride: {msg}");
    assert!(msg.contains("dense"), "the failing ARM must ride the message: {msg}");
    assert!(halted, "store-then-halt (LAW-14)");
    assert_eq!(snap.inference_failures_total, 1, "the seam counter must count the dense fire");
    assert_eq!(snap.target_integrity_defects, 0, "wrong counter ticked (M-SEAM-3)");
}

// ── The discriminator: a drain shutdown is NOT a defect ──────────────────────────────

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
    // The §P22 drive, made DETERMINISTIC: wait until the producer has POPPED a batch it
    // will never answer, so a waiter is provably blocked. `stop()` then closes both queues
    // and that waiter wakes with `Err` — the exact arm the discriminator must not call a
    // defect. Waiting on `served` instead would be timing-hopeful: the worker might have no
    // batch in flight at close, and this oracle would then pass under M-SEAM-2. It did,
    // when first written — measured, not supposed.
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
fn dense_drain_shutdown_is_not_an_inference_failure() {
    let runner = dense_runner();
    let served = Arc::new(AtomicUsize::new(0));
    let parked = Arc::new(AtomicBool::new(false));
    let producer = spawn_dense_producer(
        runner.dense_producer(),
        runner.policy_len(),
        served.clone(),
        ThenDo::ParkHoldingTheBatch,
        parked.clone(),
    );

    runner.start();
    let blocked = wait_for(120, || parked.load(Ordering::SeqCst));
    runner.stop();
    producer.join().expect("producer exits");

    assert!(blocked, "vacuous drive: the producer never parked holding a batch");
    assert_eq!(
        runner.stats_snapshot().inference_failures_total,
        0,
        "a clean dense stop was reported as an inference failure (M-SEAM-2)"
    );
    assert!(runner.fatal_defect().is_none(), "a clean stop latched a fatal defect");
}

// ── The discriminator reads the RIGHT queue (RED-TEAM M-RT-1) ────────────────────────

#[test]
fn the_dense_arm_classifies_on_its_own_queue_not_the_graph_queue() {
    // WHY THIS EXISTS. Cross-model RED-TEAM found that pointing the DENSE arm's
    // `is_closed()` check at `infer.graph_queue` instead of `infer.dense_queue` — the
    // obvious copy-paste error in a file built entirely out of dense/graph twins — left the
    // WHOLE crate suite green. Every other drive here closes both queues together (that is
    // all `stop()` can do), so nothing pinned WHICH queue the discriminator reads.
    //
    // The drive closes ONE queue and leaves its sibling open, which `stop()` cannot do but a
    // producer handle can. On a DENSE runner the graph queue is never used and stays open,
    // so a wrongly-pointed check sees `is_closed() == false` and reports the resulting
    // Err-on-close as a run-fatal inference failure. Correct code sees its own queue closed
    // and treats it as the shutdown it is.
    let runner = dense_runner();
    let queue = runner.dense_producer();
    let served = Arc::new(AtomicUsize::new(0));
    let parked = Arc::new(AtomicBool::new(false));
    let producer = spawn_dense_producer(
        queue.clone(),
        runner.policy_len(),
        served.clone(),
        ThenDo::ParkHoldingTheBatch,
        parked.clone(),
    );

    runner.start();
    let blocked = wait_for(120, || parked.load(Ordering::SeqCst));
    // Close ONLY the dense queue. The graph queue stays open for the whole drive.
    queue.close();
    // Give the woken waiter time to classify before the counter is read.
    let _ = wait_for(5, || false);
    let snap = runner.stats_snapshot();
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");

    assert!(blocked, "vacuous drive: the producer never parked holding a batch");
    assert_eq!(
        snap.inference_failures_total, 0,
        "closing the DENSE queue was classified as an inference failure — the dense arm is \
         reading the wrong queue's `is_closed()` (M-RT-1). Under it, the graph queue's state \
         decides whether a dense shutdown is a defect"
    );
    assert!(
        defect.is_none() || !defect.as_deref().unwrap_or("").contains("InferenceSeamFailure"),
        "a dense-queue close latched a seam failure: {defect:?}"
    );
}
