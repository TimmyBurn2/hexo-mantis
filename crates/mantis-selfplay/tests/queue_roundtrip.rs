//! R8-justify (339 lines): the P-07/P-08 queue round-trip pair (dense + graph) and the disjoint-pool invariant that binds them share one mock-producer scaffold; the graph leg's D6 reason-travels arms are the bulk of the overage.
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

use std::ops::Range;
use std::sync::Arc;
use std::thread;

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
