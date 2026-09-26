//! R8-justify: the graph queue and its `build_leaf_graph` (with the verbatim seam-guard
//! messages) are one cohesive port unit — splitting the builder from the queue it feeds would
//! scatter the reason-travels story across files.
//!
//! Graph inference queue — the pure-Rust half of the graph seam: its own queue, `Condvar` and
//! waiter map, whose payload is the ragged `(LegalSetPolicy, f32)`.
//!
//! A reason travels: `build_leaf_graph` and `submit_graph_and_wait` return `Err(reason)` verbatim.
//! Locks recover uncounted: guarded sections are container ops with no reachable unwind.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use fxhash::FxBuildHasher;
use mantis_graph::{build_axis_graph, AxisGraph, BuildParams, StoneList, BUILDER_IMPL_NATIVE};
use mantis_search::LegalSetPolicy;

use super::eval_cache::{EvalCache, EVAL_CACHE_BYTES};
use crate::poison::lock_or_recover;

/// One queued graph inference request (the once-per-leaf `AxisGraph` payload).
struct PendingGraphRequest {
    id: u64,
    graph: AxisGraph,
}

/// Graph waiter payload — the ragged `(LegalSetPolicy, value)`.
type GraphWaiterPayload = Result<(LegalSetPolicy, f32), String>;

#[derive(Default)]
struct GraphWaiter {
    result: Mutex<Option<GraphWaiterPayload>>,
    cv: Condvar,
}

struct GraphInner {
    queue: Mutex<VecDeque<PendingGraphRequest>>,
    queue_cv: Condvar,
    waiters: Mutex<HashMap<u64, Arc<GraphWaiter>, FxBuildHasher>>,
    next_id: AtomicU64,
    closed: AtomicBool,
    /// Graph-wire contract version this batcher speaks (spec-sourced; 1 is the only supported
    /// value). A non-1 value rejects the whole batch.
    contract_version: u32,
    /// The most graphs that can EVER be queued at once: `n_workers x leaf_batch_size`,
    /// because a worker blocks on its whole submitted batch. `0` means "not declared" and
    /// leaves the threshold at the raw half-batch — see [`saturation_threshold`].
    max_in_flight: usize,
    /// The exact per-net evaluation cache every worker of the run shares.
    eval_cache: EvalCache,
}

/// The queue depth at which [`GraphInner::pop_graph_batch_blocking`] returns BEFORE its
/// deadline — the collector's saturation threshold, DERIVED from what the run can supply.
///
/// Half the batch, clamped to `max_in_flight` so an unsuppliable half-batch cannot send every pop
/// to its deadline; the clamp only LOWERS it, never below what is achievable.
#[must_use]
pub fn saturation_threshold(batch_size: usize, max_in_flight: usize) -> usize {
    let half = batch_size / 2;
    if max_in_flight == 0 {
        half
    } else {
        half.min(max_in_flight)
    }
}

impl GraphInner {
    fn new(contract_version: u32, max_in_flight: usize, eval_cache_capacity: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            queue_cv: Condvar::new(),
            waiters: Mutex::new(HashMap::with_hasher(FxBuildHasher::default())),
            next_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
            contract_version,
            max_in_flight,
            eval_cache: EvalCache::new(eval_cache_capacity, EVAL_CACHE_BYTES),
        }
    }

    /// Pop up to `batch_size` requests, waiting at most `max_wait_ms` for the saturation threshold.
    fn pop_graph_batch_blocking(
        &self,
        batch_size: usize,
        max_wait_ms: u64,
    ) -> Vec<PendingGraphRequest> {
        let deadline = Instant::now() + Duration::from_millis(max_wait_ms);
        let mut queue = lock_or_recover(&self.queue, None);
        let threshold = saturation_threshold(batch_size, self.max_in_flight);
        while queue.len() < threshold && !self.closed.load(Ordering::SeqCst) {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline.saturating_duration_since(now);
            let (q, _) = self
                .queue_cv
                .wait_timeout(queue, remaining)
                .unwrap_or_else(PoisonError::into_inner);
            queue = q;
        }
        if queue.is_empty() {
            return Vec::new();
        }
        let take = batch_size.min(queue.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(req) = queue.pop_front() {
                out.push(req);
            }
        }
        out
    }
}

/// Rust-owned blocking graph inference queue. Clone shares one `Arc<GraphInner>`.
#[derive(Clone)]
pub struct GraphQueue {
    inner: Arc<GraphInner>,
}

impl Default for GraphQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphQueue {
    /// A queue speaking the sole supported graph-wire contract version (1). Used by tests and
    /// any caller with no spec in hand; the runner sources the version from the spec.
    #[must_use]
    pub fn new() -> Self {
        Self::with_contract_version_and_supply(1, 0)
    }

    /// A queue that also knows the run's achievable supply (`n_workers x leaf_batch_size`),
    /// from which the collector's saturation threshold is DERIVED. `max_in_flight = 0` declares
    /// no supply and keeps the raw half-batch threshold; the production runner always declares one.
    #[must_use]
    pub fn with_contract_version_and_supply(contract_version: u32, max_in_flight: usize) -> Self {
        Self::with_eval_cache(contract_version, max_in_flight, 0)
    }

    /// As [`Self::with_contract_version_and_supply`] with an explicit cache size; `0` turns it off.
    #[must_use]
    pub fn with_eval_cache(contract_version: u32, max_in_flight: usize, capacity: usize) -> Self {
        Self {
            inner: Arc::new(GraphInner::new(contract_version, max_in_flight, capacity)),
        }
    }

    /// The run's shared exact evaluation cache.
    #[must_use]
    pub fn eval_cache(&self) -> &EvalCache {
        &self.inner.eval_cache
    }

    /// The supply this queue was told about (`0` = undeclared). Read by the seam tests and
    /// by the bridge's readout, so the relation is observable rather than inferred.
    #[must_use]
    pub fn max_in_flight(&self) -> usize {
        self.inner.max_in_flight
    }

    /// CONSUMER (worker): enqueue one pre-built leaf graph, block on its waiter, and return
    /// the assembled `(LegalSetPolicy, value)`. The waiter's `Err(reason)` travels back VERBATIM.
    ///
    /// # Errors
    /// Returns `Err(reason)` if the queue is closed, this batcher speaks a non-1
    /// `graph_contract_version`, the graph's `builder_impl` is non-native, or inference for
    /// this leaf failed.
    pub fn submit_graph_and_wait(&self, graph: AxisGraph) -> Result<(LegalSetPolicy, f32), String> {
        if let Some(reason) = self.handshake_reject_reason(&graph) {
            return Err(reason);
        }

        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let waiter = Arc::new(GraphWaiter::default());
        // Register the waiter BEFORE enqueuing so a producer that pops this id can
        // never miss its waiter.
        {
            let mut wmap = lock_or_recover(&self.inner.waiters, None);
            wmap.insert(id, waiter.clone());
        }
        {
            let mut queue = lock_or_recover(&self.inner.queue, None);
            queue.push_back(PendingGraphRequest { id, graph });
            self.inner.queue_cv.notify_all();
        }

        self.wait_for(&waiter)
    }

    /// CONSUMER (worker): enqueue a WHOLE leaf batch in one shot, then block on every waiter in
    /// submission order. Returns one result per submitted graph, `Vec`-indexed by SUBMISSION
    /// ORDER — the alignment `expand_and_backup_ls_at` requires and a map-keyed return would not.
    ///
    /// The per-graph path puts exactly ONE graph in flight per worker, so the saturation
    /// threshold can never be reached and every forward runs to its deadline carrying a single
    /// leaf. Here all N are enqueued under ONE queue-lock hold and announced by ONE `notify_all`.
    ///
    /// The three handshakes run as a PRE-PASS over the whole batch, and the batch is rejected
    /// WHOLE: half-enqueuing would strand the surviving waiters behind a caller already handed a
    /// reason. Every graph is enqueued before the first wait, so this returns only after EVERY
    /// waiter has resolved — bailing on the first `Err` would drop a waiter `Arc` while the
    /// producer still holds its id, voiding `fail_remaining`'s no-orphan invariant.
    #[must_use]
    pub fn submit_graphs_and_wait(&self, graphs: Vec<AxisGraph>) -> Vec<GraphWaiterPayload> {
        let n = graphs.len();

        // Phase 1a — handshake pre-pass over the WHOLE batch, before any enqueue.
        let rejections: Vec<Option<String>> = graphs
            .iter()
            .map(|g| self.handshake_reject_reason(g))
            .collect();
        let culprit = rejections
            .iter()
            .enumerate()
            .find_map(|(i, reason)| reason.clone().map(|reason| (i, reason)));
        if let Some((first, culprit)) = culprit {
            return rejections
                .into_iter()
                .map(|reason| {
                    Err(reason.unwrap_or_else(|| {
                        format!(
                            "submit_graphs_and_wait: batch rejected before enqueue \
                             — graph {first}: {culprit}"
                        )
                    }))
                })
                .collect();
        }

        // Phase 1b — allocate every id + waiter and register the waiters BEFORE any enqueue
        // (the ordering that keeps a producer from popping an id whose waiter is not yet
        // there), then push all N and notify ONCE.
        let mut waiters: Vec<Arc<GraphWaiter>> = Vec::with_capacity(n);
        let mut requests: Vec<PendingGraphRequest> = Vec::with_capacity(n);
        for graph in graphs {
            let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
            waiters.push(Arc::new(GraphWaiter::default()));
            requests.push(PendingGraphRequest { id, graph });
        }
        if requests.is_empty() {
            return Vec::new();
        }
        {
            let mut wmap = lock_or_recover(&self.inner.waiters, None);
            for (waiter, req) in waiters.iter().zip(requests.iter()) {
                wmap.insert(req.id, waiter.clone());
            }
        }
        {
            let mut queue = lock_or_recover(&self.inner.queue, None);
            for req in requests {
                queue.push_back(req);
            }
            // ONE notify for N pushes — see the doc comment.
            self.inner.queue_cv.notify_all();
        }

        // Phase 2 — collect ALL, in submission order.
        waiters.iter().map(|w| self.wait_for(w)).collect()
    }

    /// The pre-enqueue handshake pre-pass, ONE authority for both submit paths: closed queue,
    /// non-1 `graph_contract_version` (checked BEFORE the per-graph `builder_impl` tag), and a
    /// non-native `builder_impl`. `None` means the graph may be enqueued, and the reason
    /// strings are the frozen ones, verbatim.
    fn handshake_reject_reason(&self, graph: &AxisGraph) -> Option<String> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Some("graph batcher is closed".to_string());
        }
        if self.inner.contract_version != 1 {
            return Some(format!(
                "submit_graph_and_wait: unsupported graph_contract_version {} (expected 1)",
                self.inner.contract_version
            ));
        }
        if graph.builder_impl != BUILDER_IMPL_NATIVE {
            return Some(
                "submit_graph_and_wait: non-native builder_impl (NonNativeSampleBuilder handshake)"
                    .to_string(),
            );
        }
        None
    }

    /// Block on one registered waiter until its payload lands. Spurious wakeups and close are
    /// re-checked on EVERY wake; the payload is a single `guard.take()` read and the reason
    /// travels with it.
    fn wait_for(&self, waiter: &Arc<GraphWaiter>) -> GraphWaiterPayload {
        let mut guard = lock_or_recover(&waiter.result, None);
        loop {
            if let Some(res) = guard.take() {
                return res;
            }
            if self.inner.closed.load(Ordering::SeqCst) {
                return Err("graph batcher closed while request was waiting".to_string());
            }
            guard = waiter
                .cv
                .wait(guard)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// PRODUCER: pop up to `max` queued graph requests as `(id, AxisGraph)`. Empty on timeout
    /// or a closed, empty queue.
    #[must_use]
    pub fn pop_graph_batch(&self, max: usize, timeout_ms: u64) -> Vec<(u64, AxisGraph)> {
        self.inner
            .pop_graph_batch_blocking(max, timeout_ms)
            .into_iter()
            .map(|req| (req.id, req.graph))
            .collect()
    }

    /// PRODUCER: wake each `ids[i]` waiter with its assembled result — the ragged
    /// `(LegalSetPolicy, value)` on `Ok`, a reason `String` on `Err`. An `id` with no waiter is
    /// tolerantly dropped.
    pub fn submit_graph_results(&self, ids: &[u64], results: Vec<GraphWaiterPayload>) {
        for (&id, res) in ids.iter().zip(results) {
            let removed = {
                let mut wmap = lock_or_recover(&self.inner.waiters, None);
                wmap.remove(&id)
            };
            if let Some(waiter) = removed {
                let mut guard = lock_or_recover(&waiter.result, None);
                *guard = Some(res);
                waiter.cv.notify_all();
            }
        }
    }

    /// PRODUCER: wake and drop every still-pending waiter in `ids` with `reason`, so a
    /// mid-loop error return never orphans a blocked worker. A waiter whose result is already
    /// set is left untouched, so this is idempotent; it is the vehicle the build-error reason
    /// rides to the failed waiter.
    pub fn fail_remaining(&self, ids: &[u64], reason: &str) {
        for &id in ids {
            let removed = {
                let mut wmap = lock_or_recover(&self.inner.waiters, None);
                wmap.remove(&id)
            };
            if let Some(waiter) = removed {
                let mut guard = lock_or_recover(&waiter.result, None);
                if guard.is_none() {
                    *guard = Some(Err(reason.to_string()));
                }
                waiter.cv.notify_all();
            }
        }
    }

    /// Close the graph queue and wake all blocked waiters.
    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.inner.queue_cv.notify_all();
        let wmap = lock_or_recover(&self.inner.waiters, None);
        for waiter in wmap.values() {
            waiter.cv.notify_all();
        }
    }

    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::SeqCst)
    }
}

/// Build one leaf's axis graph from its stones, running the seam guards.
///
/// Returns `Result<AxisGraph, String>` so the build error REASON can travel to the failed waiter.
/// The guard messages are VERBATIM:
/// `current_player` and each stone player in {-1, +1}; `moves_remaining` in [0, 255] before the
/// `u8` cast; each stone's `|q|,|r|` below `i32::MAX - radius`.
///
/// # Errors
/// Returns `Err(reason)` on any seam-guard violation or a non-native builder tag.
pub fn build_leaf_graph(
    stones: &[(i64, i64, i64)],
    current_player: i64,
    moves_remaining: i64,
    win_length: u8,
    radius: u16,
    trunk_size: i32,
) -> Result<AxisGraph, String> {
    if current_player != 1 && current_player != -1 {
        return Err(format!(
            "graph request: current_player {current_player} out of range (expected +1 / -1)"
        ));
    }
    if !(0..=i64::from(u8::MAX)).contains(&moves_remaining) {
        return Err(format!(
            "graph request: moves_remaining {moves_remaining} out of range 0..=255 \
             (narrowing-cast guard, WP1 Attack-4)"
        ));
    }
    let bound = i64::from(i32::MAX) - i64::from(radius) - 1;
    let mut typed: Vec<(i32, i32, i8)> = Vec::with_capacity(stones.len());
    for &(q, r, p) in stones {
        if q.abs() > bound || r.abs() > bound {
            return Err(format!(
                "graph request: stone coord ({q},{r}) exceeds |coord| < i32::MAX-radius \
                 (WP1 Attack-2 silent-wrap guard)"
            ));
        }
        if p != 1 && p != -1 {
            return Err(format!(
                "graph request: stone player {p} out of range (expected +1 / -1)"
            ));
        }
        typed.push((q as i32, r as i32, p as i8));
    }
    let params = BuildParams {
        win_length,
        radius,
        current_player: current_player as i8,
        moves_remaining: moves_remaining as u8,
        trunk_size,
    };
    let graph = build_axis_graph(&StoneList { stones: typed }, &params);
    if graph.builder_impl != BUILDER_IMPL_NATIVE {
        return Err(
            "graph request: non-native builder_impl (NonNativeSampleBuilder handshake)".to_string(),
        );
    }
    Ok(graph)
}

/// One leaf-inference request as it crosses the FFI: `(stones, current_player,
/// moves_remaining)`. Named rather than `#[allow]`ed past `clippy::type_complexity`, because
/// the shape appears in three signatures and a reader meeting the bare tuple has to infer
/// which `i64` is which.
pub type LeafRequest = (Vec<(i64, i64, i64)>, i64, i64);

/// Build one leaf graph per position across at most `n_threads` OS threads, IN INDEX ORDER.
///
/// Each leaf touches only its own stone list; `n_threads <= 1` runs the serial path IN THIS THREAD.
///
/// # Errors
/// Returns the FIRST error in index order, so a build failure names the same position it named on
/// the serial path. A panicking worker becomes a named error rather than a panic crossing the FFI.
pub fn build_leaf_graphs_batch(
    positions: &[LeafRequest],
    win_length: u8,
    radius: u16,
    trunk_size: i32,
    n_threads: usize,
) -> Result<Vec<AxisGraph>, String> {
    crate::par::map_in_order(
        positions,
        n_threads,
        "graph request: a leaf-build worker thread panicked",
        |p: &LeafRequest| build_leaf_graph(&p.0, p.1, p.2, win_length, radius, trunk_size),
    )
}

#[cfg(test)]
mod poison_tests {
    use std::sync::Arc;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use super::{build_leaf_graph, GraphQueue, GraphWaiter};
    use crate::poison::poison;

    /// Closes the queue when its thread ends or unwinds, so a dead producer fails the consumer, never hangs it.
    struct CloseOnExit(GraphQueue);

    impl Drop for CloseOnExit {
        fn drop(&mut self) {
            self.0.close();
        }
    }

    fn empty_graph() -> mantis_graph::AxisGraph {
        build_leaf_graph(&[], 1, 2, 6, 3, 19).expect("the empty board builds")
    }

    /// Serve `want` requests with an `Err("served <id>")` reply each; gives up after two seconds.
    fn serve(q: &GraphQueue, want: usize) -> JoinHandle<usize> {
        let producer = q.clone();
        std::thread::spawn(move || {
            let _close = CloseOnExit(producer.clone());
            let (deadline, mut served) = (Instant::now() + Duration::from_secs(2), 0);
            while served < want && Instant::now() < deadline {
                let ids: Vec<u64> = producer
                    .pop_graph_batch(8, 20)
                    .iter()
                    .map(|r| r.0)
                    .collect();
                let replies = ids.iter().map(|id| Err(format!("served {id}"))).collect();
                producer.submit_graph_results(&ids, replies);
                served += ids.len();
            }
            served
        })
    }

    /// With the queue and waiter-map locks poisoned, both submit paths round-trip and close wakes.
    #[test]
    fn a_poisoned_queue_still_serves_both_submit_paths_and_closes() {
        let q = GraphQueue::new();
        poison(&q.inner.queue);
        poison(&q.inner.waiters);
        let producer = serve(&q, 3);
        let batch = q.submit_graphs_and_wait(vec![empty_graph(), empty_graph()]);
        let single = q.submit_graph_and_wait(empty_graph());
        assert_eq!(producer.join().expect("the producer thread"), 3);
        for res in batch.iter().chain(std::iter::once(&single)) {
            let reason = res.as_ref().expect_err("the test producer replies Err");
            assert!(reason.starts_with("served "), "{reason}");
        }
        q.close();
        assert!(q.is_closed());
        let after = q.submit_graph_and_wait(empty_graph());
        assert_eq!(
            after.expect_err("a closed queue refuses"),
            "graph batcher is closed"
        );
    }

    /// A waiter blocked on a poisoned result lock still wakes to the payload it is handed late.
    #[test]
    fn a_poisoned_waiter_still_blocks_and_delivers_its_payload() {
        let q = GraphQueue::new();
        let waiter = Arc::new(GraphWaiter::default());
        poison(&waiter.result);
        q.inner
            .waiters
            .lock()
            .expect("unpoisoned in this test")
            .insert(7, waiter.clone());
        let producer = q.clone();
        let late = std::thread::spawn(move || {
            let _close = CloseOnExit(producer.clone());
            std::thread::sleep(Duration::from_millis(50));
            producer.submit_graph_results(&[7], vec![Err("payload".to_string())]);
        });
        let got = q.wait_for(&waiter);
        late.join().expect("the late producer");
        assert_eq!(got.expect_err("the payload is an Err"), "payload");
    }

    /// `fail_remaining` still fails a live waiter whose result lock is poisoned.
    #[test]
    fn fail_remaining_reaches_a_poisoned_waiter() {
        let q = GraphQueue::new();
        let waiter = Arc::new(GraphWaiter::default());
        poison(&waiter.result);
        q.inner
            .waiters
            .lock()
            .expect("unpoisoned in this test")
            .insert(9, waiter.clone());
        q.fail_remaining(&[9], "failed by the producer");
        assert_eq!(
            q.wait_for(&waiter).expect_err("fail_remaining set an Err"),
            "failed by the producer"
        );
    }
}
