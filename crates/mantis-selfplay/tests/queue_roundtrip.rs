//! R8-justify: the P-07/P-08 queue round-trip pair (dense + graph) and the disjoint-pool invariant that binds them share one mock-producer scaffold; the graph leg's D6 reason-travels arms are the bulk of the overage.
//! P-07 / P-08 — mock-game queue round-trips (dense + graph), pyo3-free.
//!
//! A MOCK producer (D16 surrogate stand-in; the NN + numpy face is WP7) pops the
//! queue and submits deterministic results; the blocking consumer receives them.
//! Covers: submit → mock pop → submit results → consumer receives; single-read
//! (a second submit for the same id is a no-op); over-batch (an extra unknown id
//! is tolerantly dropped) and underflow (closed queue ⇒ `Err`); the DENSE
//! skip-on-Err path (reason NOT required, D6); and — for GRAPH — the D6
//! reason-travels guarantee (inference failure, `fail_remaining`, AND the
//! build-side reason now travels) with no orphaned waiter, plus the disjoint-pool
//! invariant (the graph batcher never touches the dense queue).
//!
//! The Q-FIND-1 batch-submit arms ride the SAME mock producer: one
//! `submit_graphs_and_wait` puts a whole leaf batch in flight, so a single pop
//! serves it, the collector's saturation threshold becomes reachable, submission
//! order survives an out-of-order producer, and neither a rejected graph nor a
//! mid-batch producer failure can orphan a waiter.

use std::ops::Range;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use fxhash::FxHashMap;
use mantis_graph::BUILDER_IMPL_NATIVE;
use mantis_search::LegalSetPolicy;
use mantis_selfplay::queues::{build_leaf_graph, DenseQueue, GraphQueue};

const FEAT: usize = 8;
const POLICY_LEN: usize = 4;

// ── mock producer helpers ────────────────────────────────────────────────────

// A pop batch-size of 2 gives a saturation threshold of 1, so `pop` BLOCKS (up to
// the timeout) until at least one request is enqueued — the drain waits for the
// consumer to enqueue rather than racing ahead of it. (batch-size 1 ⇒ threshold 0
// ⇒ non-blocking spin, which would race the consumer's enqueue.)

/// Collect exactly `expected` dense requests, blocking for each item.
fn drain_dense(q: &DenseQueue, expected: usize) -> Vec<(u64, Vec<f32>)> {
    let mut got = Vec::new();
    for _ in 0..400 {
        got.append(&mut q.pop_batch(2, 50));
        if got.len() >= expected {
            break;
        }
    }
    got
}

fn drain_graph(q: &GraphQueue, expected: usize) -> Vec<(u64, mantis_graph::AxisGraph)> {
    let mut got = Vec::new();
    for _ in 0..400 {
        got.append(&mut q.pop_graph_batch(2, 50));
        if got.len() >= expected {
            break;
        }
    }
    got
}

/// Deterministic mock dense policy for a request id: `[id, id+1, id+2, id+3]`.
fn mock_policy(id: u64) -> Vec<f32> {
    (0..POLICY_LEN).map(|k| id as f32 + k as f32).collect()
}
fn mock_value(id: u64) -> f32 {
    id as f32 * 0.5
}

// ── P-07 dense round-trip ─────────────────────────────────────────────────────

#[test]
fn dense_round_trip_two_requests() {
    let q = DenseQueue::new(FEAT);
    let qc = q.clone();
    let feats = vec![vec![1.0f32; FEAT], vec![2.0f32; FEAT]];
    let handle = thread::spawn(move || qc.submit_batch_and_wait(feats));

    let popped = drain_dense(&q, 2);
    assert_eq!(popped.len(), 2, "producer popped both requests");

    // mock inference: one shared Arc buffer + per-id ranges (§P74 share).
    let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
    let mut flat = Vec::new();
    for &id in &ids {
        flat.extend(mock_policy(id));
    }
    let arc = Arc::new(flat);
    let ranges: Vec<Range<usize>> = (0..ids.len())
        .map(|i| i * POLICY_LEN..(i + 1) * POLICY_LEN)
        .collect();
    let values: Vec<f32> = ids.iter().map(|&id| mock_value(id)).collect();
    q.submit_results(&ids, &arc, &ranges, &values);

    let out = handle.join().unwrap().expect("dense round-trip ok");
    assert_eq!(out.len(), 2);
    // key each output back to its id via policy[0] (== id) — robust to ordering.
    for (policy, value) in out {
        let id = policy[0] as u64;
        assert_eq!(policy, mock_policy(id), "policy for id {id}");
        assert!((value - mock_value(id)).abs() < 1e-6, "value for id {id}");
    }

    // Single-read: a second submit for the same ids is a no-op (waiters removed) —
    // no panic, nothing to deliver.
    q.submit_results(&ids, &arc, &ranges, &values);
}

#[test]
fn dense_over_batch_drops_unknown_id() {
    let q = DenseQueue::new(FEAT);
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_batch_and_wait(vec![vec![3.0f32; FEAT]]));
    let popped = drain_dense(&q, 1);
    assert_eq!(popped.len(), 1);
    let real = popped[0].0;

    // Over-batch: submit results for the real id PLUS an unknown id 9999.
    let ids = vec![real, 9999u64];
    let mut flat = Vec::new();
    flat.extend(mock_policy(real));
    flat.extend(mock_policy(9999));
    let arc = Arc::new(flat);
    let ranges = vec![0..POLICY_LEN, POLICY_LEN..2 * POLICY_LEN];
    let values = vec![mock_value(real), mock_value(9999)];
    q.submit_results(&ids, &arc, &ranges, &values); // 9999 has no waiter ⇒ dropped

    let out = handle.join().unwrap().expect("real request resolved");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, mock_policy(real));
}

#[test]
fn dense_underflow_and_length_mismatch_are_loud() {
    let q = DenseQueue::new(FEAT);
    // length-mismatch ⇒ Err(()).
    assert!(q.submit_batch_and_wait(vec![vec![0.0f32; FEAT + 1]]).is_err());
    // close ⇒ underflow: submit returns Err, and pop of the closed empty queue is empty.
    q.close();
    assert!(q.is_closed());
    assert!(q.submit_batch_and_wait(vec![vec![0.0f32; FEAT]]).is_err());
    assert!(q.pop_batch(64, 10).is_empty(), "pop of closed empty queue is empty");
}

#[test]
fn dense_close_wakes_blocked_waiter() {
    let q = DenseQueue::new(FEAT);
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_batch_and_wait(vec![vec![0.0f32; FEAT]]));
    // ensure the request is enqueued (waiter blocked) before closing.
    let _ = drain_dense(&q, 1);
    q.close();
    assert!(handle.join().unwrap().is_err(), "closed-while-waiting ⇒ Err(())");
}

#[test]
fn dense_submit_failure_makes_consumer_skip() {
    // D6: the dense reason is NOT consumed — the consumer just gets Err(()).
    let q = DenseQueue::new(FEAT);
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_batch_and_wait(vec![vec![5.0f32; FEAT]]));
    let popped = drain_dense(&q, 1);
    let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
    q.submit_failure(&ids, "mock inference boom");
    assert!(handle.join().unwrap().is_err(), "submit-Err ⇒ worker skips batch");
}

// ── P-08 graph round-trip + reason-travels ────────────────────────────────────

/// A uniform mock `LegalSetPolicy` over `n_legal` cells (dense-only; the exact
/// distribution is irrelevant to the queue transport under test).
fn mock_ls(n_legal: usize) -> LegalSetPolicy {
    LegalSetPolicy {
        dense: vec![1.0f32 / n_legal as f32; n_legal],
        overflow: FxHashMap::default(),
    }
}

fn valid_leaf() -> mantis_graph::AxisGraph {
    let stones = vec![(0i64, 0i64, 1i64), (30, 0, -1), (31, 0, -1)];
    build_leaf_graph(&stones, 1, 2, 6, 6, 19).expect("valid leaf builds")
}

#[test]
fn graph_round_trip_delivers_ls_and_value() {
    let q = GraphQueue::new();
    let graph = valid_leaf();
    let n_legal = graph.legal_node_gather.len();
    assert!(n_legal > 0);

    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graph_and_wait(graph));
    let popped = drain_graph(&q, 1);
    assert_eq!(popped.len(), 1);
    let id = popped[0].0;
    q.submit_graph_results(&[id], vec![Ok((mock_ls(n_legal), 0.5))]);

    let (ls, value) = handle.join().unwrap().expect("graph round-trip ok");
    assert!((value - 0.5).abs() < 1e-6);
    assert_eq!(ls.dense.len(), n_legal);
}

#[test]
fn graph_inference_failure_reason_travels() {
    // D6: an inference-failure reason travels VERBATIM to the waiting caller.
    let q = GraphQueue::new();
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graph_and_wait(valid_leaf()));
    let popped = drain_graph(&q, 1);
    let id = popped[0].0;
    q.submit_graph_results(&[id], vec![Err("segment desync boom".to_string())]);
    let res = handle.join().unwrap();
    assert_eq!(res.unwrap_err(), "segment desync boom", "reason travels verbatim");
}

#[test]
fn graph_fail_remaining_orphans_none() {
    // fail_remaining wakes EVERY still-pending waiter with the reason (no orphan).
    let q = GraphQueue::new();
    let mut handles = Vec::new();
    for _ in 0..3 {
        let qc = q.clone();
        handles.push(thread::spawn(move || qc.submit_graph_and_wait(valid_leaf())));
    }
    let popped = drain_graph(&q, 3);
    assert_eq!(popped.len(), 3);
    let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
    q.fail_remaining(&ids, "whole batch died");
    for h in handles {
        assert_eq!(h.join().unwrap().unwrap_err(), "whole batch died");
    }
}

#[test]
fn graph_build_failure_reason_preserved_and_travels() {
    // D6 build fix: build_leaf_graph returns the reason (NOT .ok()-swallowed None).
    let bad = build_leaf_graph(&[(0, 0, 1)], 2, 2, 6, 6, 19);
    let reason = bad.expect_err("bad current_player ⇒ build error");
    assert_eq!(
        reason,
        "graph request: current_player 2 out of range (expected +1 / -1)"
    );

    // And it TRAVELS: route the build reason to a waiting caller via fail_remaining.
    let q = GraphQueue::new();
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graph_and_wait(valid_leaf()));
    let popped = drain_graph(&q, 1);
    let id = popped[0].0;
    q.fail_remaining(&[id], &reason);
    assert_eq!(handle.join().unwrap().unwrap_err(), reason, "build reason travels to waiter");
}

#[test]
fn graph_non_native_handshake_rejected() {
    let q = GraphQueue::new();
    let mut graph = valid_leaf();
    graph.builder_impl = 0; // non-native tag
    let res = q.submit_graph_and_wait(graph);
    assert_eq!(
        res.unwrap_err(),
        "submit_graph_and_wait: non-native builder_impl (NonNativeSampleBuilder handshake)"
    );
}

#[test]
fn graph_single_read_second_submit_is_noop() {
    let q = GraphQueue::new();
    let graph = valid_leaf();
    let n_legal = graph.legal_node_gather.len();
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graph_and_wait(graph));
    let popped = drain_graph(&q, 1);
    let id = popped[0].0;
    q.submit_graph_results(&[id], vec![Ok((mock_ls(n_legal), 0.9))]);
    let first = handle.join().unwrap().expect("delivered once");
    assert!((first.1 - 0.9).abs() < 1e-6);
    // Second submit for the same (now-removed) id is a no-op — no panic.
    q.submit_graph_results(&[id], vec![Ok((mock_ls(n_legal), -1.0))]);
}

#[test]
fn graph_close_wakes_blocked_waiter() {
    let q = GraphQueue::new();
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graph_and_wait(valid_leaf()));
    let _ = drain_graph(&q, 1);
    q.close();
    assert_eq!(
        handle.join().unwrap().unwrap_err(),
        "graph batcher closed while request was waiting"
    );
}

#[test]
fn graph_closed_before_submit_is_loud() {
    let q = GraphQueue::new();
    q.close();
    assert_eq!(q.submit_graph_and_wait(valid_leaf()).unwrap_err(), "graph batcher is closed");
}

// ── Q-FIND-1 batch submit: the whole leaf batch in flight at once ─────────────

/// ONE pop, bounded: retry an empty pop (the submitter thread may not have pushed
/// yet) but never merge two pops — the width of the FIRST non-empty pop is the
/// property under test.
fn one_pop(q: &GraphQueue, cap: usize) -> Vec<(u64, mantis_graph::AxisGraph)> {
    for _ in 0..20 {
        let popped = q.pop_graph_batch(cap, 500);
        if !popped.is_empty() {
            return popped;
        }
    }
    Vec::new()
}

fn ok_results(popped: &[(u64, mantis_graph::AxisGraph)]) -> Vec<Result<(LegalSetPolicy, f32), String>> {
    popped
        .iter()
        .map(|(_, g)| Ok((mock_ls(g.legal_node_gather.len()), 0.5f32)))
        .collect()
}

#[test]
fn batch_submit_puts_the_whole_leaf_batch_in_flight_before_any_wait() {
    // The property that makes the collector threshold reachable at all: ONE
    // submit_graphs_and_wait leaves N graphs queued simultaneously, so a single pop
    // of capacity >= N serves the whole leaf batch in ONE forward. The serial
    // per-graph submit this replaces could never put more than 1 in the queue.
    let q = GraphQueue::new();
    let n = 8usize;
    let graphs: Vec<mantis_graph::AxisGraph> = (0..n).map(|_| valid_leaf()).collect();

    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graphs_and_wait(graphs));

    let popped = one_pop(&q, n);
    assert_eq!(
        popped.len(),
        n,
        "ONE pop must serve the whole leaf batch, not {} of {n}",
        popped.len()
    );

    let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
    q.submit_graph_results(&ids, ok_results(&popped));

    let out = handle.join().unwrap();
    assert_eq!(out.len(), n, "one result per submitted graph, in submission order");
    assert!(out.iter().all(Result::is_ok));
}

#[test]
fn a_pop_capacity_below_the_leaf_batch_clears_in_exactly_ceil_n_over_cap_pops() {
    // ceil(N/cap) is the CONTRACT, not an accident. N=8 at cap=3 clears in 3 pops and
    // never more; a regression to per-graph dispatch would need 8.
    let q = GraphQueue::new();
    let (n, cap) = (8usize, 3usize);
    let graphs: Vec<mantis_graph::AxisGraph> = (0..n).map(|_| valid_leaf()).collect();
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graphs_and_wait(graphs));

    let mut pops = 0usize;
    let mut served = 0usize;
    for _ in 0..(4 * n) {
        if served >= n {
            break;
        }
        let popped = q.pop_graph_batch(cap, 500);
        if popped.is_empty() {
            continue; // deadline expiry before the submitter's push landed
        }
        pops += 1;
        assert!(popped.len() <= cap);
        let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
        q.submit_graph_results(&ids, ok_results(&popped));
        served += popped.len();
    }
    assert_eq!(served, n);
    assert_eq!(pops, n.div_ceil(cap), "must clear in ceil(N/cap) pops, got {pops}");
    assert!(handle.join().unwrap().iter().all(Result::is_ok));
}

#[test]
fn a_rejected_graph_never_enters_the_queue_and_leaves_no_orphan_waiter() {
    // The pre-pass handshakes must still run BEFORE any enqueue. A batch carrying one
    // non-native tag must not half-enqueue: the surviving waiters would block forever
    // behind a caller that has already been handed a reason (DESIGN §3 A.5).
    let q = GraphQueue::new();
    let mut bad = valid_leaf();
    bad.builder_impl = 0;
    let graphs = vec![valid_leaf(), bad, valid_leaf()];

    let out = q.submit_graphs_and_wait(graphs);
    assert_eq!(out.len(), 3);
    assert_eq!(
        out[1].as_ref().unwrap_err(),
        "submit_graph_and_wait: non-native builder_impl (NonNativeSampleBuilder handshake)",
        "the handshake reason travels VERBATIM (D6), per-graph"
    );
    // The clean graphs are refused too, naming the offender — never half-enqueued.
    for i in [0usize, 2] {
        assert!(
            out[i].as_ref().unwrap_err().contains("graph 1"),
            "slot {i} must name the offending graph, got {:?}",
            out[i]
        );
    }
    assert!(
        q.pop_graph_batch(8, 5).is_empty(),
        "a rejected batch never touches the queue"
    );
}

#[test]
fn batch_submit_preserves_submission_order_through_the_fuse() {
    // `expand_and_backup_ls_at` consumes aggregated_ls / aggregated_values / centers
    // INDEX-ALIGNED with `leaves`. A map-keyed or reordered return would silently
    // misalign every leaf's expand frame (the class the always-on trunk assert in
    // search_drive guards). Position-encoded values make a permutation visible.
    let q = GraphQueue::new();
    let n = 5usize;
    let graphs: Vec<mantis_graph::AxisGraph> = (0..n).map(|_| valid_leaf()).collect();
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graphs_and_wait(graphs));

    let popped = one_pop(&q, n);
    assert_eq!(popped.len(), n, "the whole batch must be in flight before any reply");
    let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
    // Reply in REVERSE order with position-encoded values.
    for pos in (0..n).rev() {
        let g = &popped[pos].1;
        q.submit_graph_results(
            &[ids[pos]],
            vec![Ok((mock_ls(g.legal_node_gather.len()), pos as f32))],
        );
    }
    let out = handle.join().unwrap();
    for (i, r) in out.iter().enumerate() {
        let (_ls, v) = r.as_ref().expect("all ok");
        assert_eq!(*v as usize, i, "result {i} carries another graph's value — order broke");
    }
}

#[test]
fn a_reachable_threshold_returns_before_the_deadline() {
    // With supply >= threshold the pop returns ON the threshold, not on max_wait_ms.
    // Deliberately generous slack (500 ms deadline, asserted < 250 ms) so this pins
    // the mechanism, not the box's scheduler.
    let q = GraphQueue::new();
    let graphs: Vec<mantis_graph::AxisGraph> = (0..8).map(|_| valid_leaf()).collect();
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graphs_and_wait(graphs));

    let t0 = Instant::now();
    let popped = q.pop_graph_batch(8, 500); // batch_size 8 -> threshold 4, supply 8
    let elapsed = t0.elapsed();
    assert_eq!(popped.len(), 8);
    assert!(
        elapsed < Duration::from_millis(250),
        "pop took {elapsed:?}; a reachable threshold must not run to the deadline"
    );
    let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
    q.submit_graph_results(&ids, ok_results(&popped));
    handle.join().unwrap();
}

#[test]
fn an_unreachable_threshold_still_serves_on_the_deadline() {
    // The pre-fix behaviour must NOT be deleted: a starved queue still serves what it
    // has when the deadline expires. Losing this would turn a slow run into a hung one.
    let q = GraphQueue::new();
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graphs_and_wait(vec![valid_leaf()]));
    let popped = one_pop(&q, 64); // threshold 32, supply 1
    assert_eq!(popped.len(), 1, "deadline expiry serves the single queued graph");
    let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
    q.submit_graph_results(&ids, ok_results(&popped));
    assert!(handle.join().unwrap()[0].is_ok());
}

#[test]
fn concurrent_batch_submitters_each_receive_only_their_own_results() {
    // Models n_workers > 1: several threads each batch-submit their own leaf batch
    // into ONE shared queue and every caller must get back exactly its own graphs'
    // payloads. `graph_fail_remaining_orphans_none` proves nobody hangs; this proves
    // nobody gets crossed.
    let q = GraphQueue::new();
    let (workers, per_worker) = (4usize, 3usize);
    let mut handles = Vec::new();
    for w in 0..workers {
        let qc = q.clone();
        handles.push(thread::spawn(move || {
            let graphs: Vec<mantis_graph::AxisGraph> =
                (0..per_worker).map(|_| valid_leaf()).collect();
            (w, qc.submit_graphs_and_wait(graphs))
        }));
    }

    let total = workers * per_worker;
    let popped = drain_graph(&q, total);
    assert_eq!(
        popped.len(),
        total,
        "every worker's whole leaf batch must be in flight simultaneously"
    );

    // Encode the id itself in the value so a crossed wire is arithmetically visible.
    let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
    let results: Vec<Result<(LegalSetPolicy, f32), String>> = popped
        .iter()
        .map(|(id, g)| Ok((mock_ls(g.legal_node_gather.len()), *id as f32)))
        .collect();
    q.submit_graph_results(&ids, results);

    let mut seen: Vec<f32> = Vec::new();
    for h in handles {
        let (_w, out) = h.join().unwrap();
        assert_eq!(out.len(), per_worker);
        for r in out {
            seen.push(r.expect("every waiter resolves Ok").1);
        }
    }
    seen.sort_by(f32::total_cmp);
    let mut expected: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
    expected.sort_by(f32::total_cmp);
    assert_eq!(seen, expected, "every id delivered exactly once, to exactly one waiter");
}

#[test]
fn a_mid_batch_producer_failure_wakes_every_waiter_with_the_reason() {
    // Under batch submit ALL N are already enqueued, so a failure partway through the
    // producer's walk must not orphan the tail. `fail_remaining` is the vehicle and
    // its reason travels verbatim (D6).
    let q = GraphQueue::new();
    let n = 6usize;
    let graphs: Vec<mantis_graph::AxisGraph> = (0..n).map(|_| valid_leaf()).collect();
    let qc = q.clone();
    let handle = thread::spawn(move || qc.submit_graphs_and_wait(graphs));

    let popped = one_pop(&q, n);
    assert_eq!(popped.len(), n, "the whole batch must be in flight before the failure");
    let ids: Vec<u64> = popped.iter().map(|(id, _)| *id).collect();
    let n_legal = popped[0].1.legal_node_gather.len();
    q.submit_graph_results(&ids[..2], vec![Ok((mock_ls(n_legal), 0.5)); 2]);
    q.fail_remaining(&ids[2..], "forward blew up mid-batch");

    let out = handle.join().unwrap();
    assert_eq!(out.len(), n, "the caller returns only after EVERY waiter resolved");
    assert!(out[0].is_ok() && out[1].is_ok());
    for r in &out[2..] {
        assert_eq!(r.as_ref().unwrap_err(), "forward blew up mid-batch");
    }
}

// ── cross-queue disjointness + F-19 build-once structural note ────────────────

#[test]
fn graph_and_dense_queues_are_disjoint() {
    // Closing the graph queue must NOT close the dense queue (old :1415 — the
    // graph batcher never touches the dense pool).
    let dense = DenseQueue::new(FEAT);
    let graph = GraphQueue::new();
    graph.close();
    assert!(graph.is_closed());
    assert!(!dense.is_closed(), "dense pool untouched by the graph batcher");
}

#[test]
fn build_leaf_graph_is_one_native_build_per_leaf() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    // F-19 (structural, at the surface this stage owns): each leaf is built EXACTLY
    // once and stamps the native builder_impl. REAL observer: `builds` increments
    // INSIDE a counting shim wrapped around the actual `build_leaf_graph` call, so it
    // counts genuine build invocations over the corpus — a redundant/double build
    // bumps the counter and trips the assert. (The old `builds += 1`-per-iteration
    // counted loop turns, not builds, so `assert_eq!(n, n)` was tautological and a
    // double-build slipped through.)
    let builds = AtomicUsize::new(0);
    let build_once = |stones: &[(i64, i64, i64)]| {
        let g = build_leaf_graph(stones, 1, 2, 6, 6, 19).expect("leaf builds");
        builds.fetch_add(1, Ordering::Relaxed);
        g
    };
    let leaves = [
        vec![(0i64, 0i64, 1i64), (1, 0, -1), (0, 1, 1)],
        vec![(0i64, 0i64, 1i64), (30, 0, -1), (31, 0, -1)],
        vec![(0i64, 0i64, 1i64), (0, 1, -1), (1, 0, 1), (1, 1, -1)],
    ];
    for stones in &leaves {
        let g = build_once(stones);
        assert_eq!(g.builder_impl, BUILDER_IMPL_NATIVE);
    }
    assert_eq!(
        builds.load(Ordering::Relaxed),
        leaves.len(),
        "exactly one native build per leaf (the counter observes real build_leaf_graph calls)",
    );
}
