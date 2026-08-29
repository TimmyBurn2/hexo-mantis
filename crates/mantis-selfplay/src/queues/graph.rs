//! R8-justify: the graph queue + its D6 `build_leaf_graph` (with the verbatim
//! WP-1 seam-guard messages) are one cohesive port unit — splitting the builder
//! from the queue it feeds would scatter the D6 reason-travels story across files.
//!
//! Graph inference queue — the pure-Rust half of the frozen `inference_bridge.rs`
//! parallel graph seam (WP6 D4/D6), pyo3/numpy STRIPPED.
//!
//! A DISJOINT structure from the dense queue (`queues::dense`): its own
//! `Mutex<VecDeque<PendingGraphRequest>>` + `Condvar` + waiter map
//! (`inference_bridge.rs:152-155`). The graph batcher NEVER touches the dense
//! pool. The waiter payload is the ragged `(LegalSetPolicy, f32)`
//! (`inference_bridge.rs:48`).
//!
//! D6 (reason-travels) is honoured in TWO places the frozen code dropped a
//! reason: (1) `build_leaf_graph` returns `Result<AxisGraph, String>` — the
//! build error REASON is preserved, NOT `.ok()`-swallowed to `None`
//! (`inference_bridge.rs:530`); (2) `submit_graph_and_wait` returns the waiter's
//! `Err(reason)` verbatim instead of the frozen `Err(())` collapse
//! (`inference_bridge.rs:453`). The dense path stays PORT-EXACT (reason not
//! required — the old worker never consumed it). This asymmetry is intentional.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use fxhash::FxBuildHasher;
use mantis_graph::{build_axis_graph, AxisGraph, BuildParams, StoneList, BUILDER_IMPL_NATIVE};
use mantis_search::LegalSetPolicy;

/// One queued graph inference request (the once-per-leaf `AxisGraph` payload,
/// `inference_bridge.rs:32`).
struct PendingGraphRequest {
    id: u64,
    graph: AxisGraph,
}

/// Graph waiter payload — the ragged `(LegalSetPolicy, value)` (`:48`).
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
    /// Graph-wire contract version this batcher speaks (spec-sourced; the amended
    /// ragged contract is version 1, the only supported value). The frozen
    /// `submit_batch_and_wait_graph_rust` rejects the whole batch on a non-1 value
    /// (`inference_bridge.rs:425`).
    contract_version: u32,
}

impl GraphInner {
    fn new(contract_version: u32) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            queue_cv: Condvar::new(),
            waiters: Mutex::new(HashMap::with_hasher(FxBuildHasher::default())),
            next_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
            contract_version,
        }
    }

    /// Graph counterpart of the dense pop (`pop_graph_batch_blocking:179`) — same
    /// saturation threshold / timeout, on the parallel graph queue.
    fn pop_graph_batch_blocking(
        &self,
        batch_size: usize,
        max_wait_ms: u64,
    ) -> Vec<PendingGraphRequest> {
        let deadline = Instant::now() + Duration::from_millis(max_wait_ms);
        let mut queue = self.queue.lock().expect("graph queue lock poisoned");
        let threshold = batch_size / 2;
        while queue.len() < threshold && !self.closed.load(Ordering::SeqCst) {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline.saturating_duration_since(now);
            let (q, _) = self
                .queue_cv
                .wait_timeout(queue, remaining)
                .expect("graph queue condvar poisoned");
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

/// Rust-owned blocking graph inference queue (the WP-3 parallel seam, pyo3
/// stripped). Clone shares one `Arc<GraphInner>`.
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
    /// A queue speaking the sole supported graph-wire contract version (1). Used by
    /// tests and any caller with no spec in hand; the runner sources the version from
    /// the spec via [`GraphQueue::with_contract_version`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_contract_version(1)
    }

    /// A queue speaking `contract_version` (spec-sourced). A non-1 value makes every
    /// `submit_graph_and_wait` reject its graph loud — the batch-level die-loud
    /// handshake the frozen `submit_batch_and_wait_graph_rust` runs
    /// (`inference_bridge.rs:425`).
    #[must_use]
    pub fn with_contract_version(contract_version: u32) -> Self {
        Self {
            inner: Arc::new(GraphInner::new(contract_version)),
        }
    }

    /// CONSUMER (worker): enqueue one pre-built leaf graph, block on its waiter,
    /// and return the assembled `(LegalSetPolicy, value)`. The waiter's
    /// `Err(reason)` travels back VERBATIM (D6) — no `Err(())` collapse.
    ///
    /// # Errors
    /// Returns `Err(reason)` if the queue is closed, this batcher speaks a non-1
    /// `graph_contract_version`, the graph's `builder_impl` is non-native, or
    /// inference for this leaf failed (the reason set by the producer via
    /// `submit_graph_results` / `fail_remaining`).
    pub fn submit_graph_and_wait(&self, graph: AxisGraph) -> Result<(LegalSetPolicy, f32), String> {
        if let Some(reason) = self.handshake_reject_reason(&graph) {
            return Err(reason);
        }

        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let waiter = Arc::new(GraphWaiter::default());
        // Register the waiter BEFORE enqueuing so a producer that pops this id can
        // never miss its waiter.
        {
            let mut wmap = self
                .inner
                .waiters
                .lock()
                .expect("graph waiter map lock poisoned");
            wmap.insert(id, waiter.clone());
        }
        {
            let mut queue = self.inner.queue.lock().expect("graph queue lock poisoned");
            queue.push_back(PendingGraphRequest { id, graph });
            self.inner.queue_cv.notify_all();
        }

        self.wait_for(&waiter)
    }

    /// CONSUMER (worker): enqueue a WHOLE leaf batch in one shot, then block on
    /// every waiter in submission order. Returns one result per submitted graph,
    /// `Vec`-indexed by SUBMISSION ORDER — the index alignment
    /// `expand_and_backup_ls_at` requires against `centers` / `leaves`. A map-keyed
    /// return would not carry it.
    ///
    /// Why this exists (Q-FIND-1 / R263): the per-graph `submit_graph_and_wait` puts
    /// exactly ONE graph in flight per worker, so the collector's saturation
    /// threshold (`pop_graph_batch_blocking`'s `batch_size / 2`) can never be
    /// reached and every forward runs to its `max_wait_ms` deadline carrying a
    /// single leaf. Here all N are enqueued under ONE queue-lock hold and announced
    /// by ONE `notify_all`, so the collector wakes once and re-evaluates
    /// `queue.len()` seeing all N rather than being woken N times and seeing 1 each
    /// time. One notify per push would restore the starved read.
    ///
    /// The three handshakes (`closed`, non-1 `contract_version`, non-native
    /// `builder_impl`) run as a PRE-PASS over the whole batch, before ANY enqueue.
    /// A rejected graph never touches the queue, and the batch is rejected WHOLE:
    /// half-enqueuing would strand the surviving waiters behind a caller that has
    /// already been handed a reason. The offending graph's slot carries its own
    /// frozen reason VERBATIM (D6); the rest name the offender.
    ///
    /// COLLECT-ALL-THEN-DECIDE: unlike the serial caller this replaces, every graph
    /// is already enqueued when the first wait begins, so this returns only after
    /// EVERY waiter has resolved. Bailing on the first `Err` would drop a waiter
    /// `Arc` while the producer still holds its id — survivable (the producer
    /// tolerantly drops an unknown id) but it voids the no-orphan invariant
    /// `fail_remaining` is built on. Callers scan the returned `Vec` AFTER it lands.
    #[must_use]
    pub fn submit_graphs_and_wait(&self, graphs: Vec<AxisGraph>) -> Vec<GraphWaiterPayload> {
        let n = graphs.len();

        // Phase 1a — handshake pre-pass over the WHOLE batch, before any enqueue.
        let rejections: Vec<Option<String>> = graphs
            .iter()
            .map(|g| self.handshake_reject_reason(g))
            .collect();
        if let Some(first) = rejections.iter().position(Option::is_some) {
            let culprit = rejections[first]
                .clone()
                .expect("position() found the reason");
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

        // Phase 1b — allocate every id + waiter, register the waiters BEFORE any
        // enqueue (the ordering the per-graph path relies on so a producer that pops
        // an id can never miss its waiter), then push all N and notify ONCE.
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
            let mut wmap = self
                .inner
                .waiters
                .lock()
                .expect("graph waiter map lock poisoned");
            for (waiter, req) in waiters.iter().zip(requests.iter()) {
                wmap.insert(req.id, waiter.clone());
            }
        }
        {
            let mut queue = self.inner.queue.lock().expect("graph queue lock poisoned");
            for req in requests {
                queue.push_back(req);
            }
            // ONE notify for N pushes — see the doc comment.
            self.inner.queue_cv.notify_all();
        }

        // Phase 2 — collect ALL, in submission order.
        waiters.iter().map(|w| self.wait_for(w)).collect()
    }

    /// The pre-enqueue handshake pre-pass, ONE authority for both submit paths:
    /// closed queue, non-1 `graph_contract_version` (frozen `inference_bridge.rs:425`
    /// — a batcher speaking a non-1 version rejects loud, BEFORE the per-graph
    /// `builder_impl` check), and the N5 non-native `builder_impl` tag. `None` ⇒ the
    /// graph may be enqueued. The reason strings are the frozen ones, verbatim.
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

    /// Block on one registered waiter until its payload lands. Spurious wakeups and
    /// close are re-checked on EVERY wake; the payload is a single read via
    /// `guard.take()` (`inference_bridge.rs:447`) and the reason travels (D6).
    fn wait_for(&self, waiter: &Arc<GraphWaiter>) -> GraphWaiterPayload {
        let mut guard = waiter.result.lock().expect("graph waiter lock poisoned");
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
                .expect("graph waiter condvar poisoned");
        }
    }

    /// PRODUCER: pop up to `max` queued graph requests as `(id, AxisGraph)`
    /// (pure-Rust replacement for `next_graph_batch`). Empty on timeout /
    /// closed-empty queue.
    #[must_use]
    pub fn pop_graph_batch(&self, max: usize, timeout_ms: u64) -> Vec<(u64, AxisGraph)> {
        self.inner
            .pop_graph_batch_blocking(max, timeout_ms)
            .into_iter()
            .map(|req| (req.id, req.graph))
            .collect()
    }

    /// PRODUCER: wake each `ids[i]` waiter with its assembled result (ragged
    /// `(LegalSetPolicy, value)` on `Ok`, a reason `String` on `Err`). The
    /// pure-Rust producer assembles the `LegalSetPolicy` itself (the numpy segment
    /// scatter + `assemble_ls_from_gnn_probs` marshaling is WP7). An `id` with no
    /// waiter is tolerantly dropped.
    pub fn submit_graph_results(&self, ids: &[u64], results: Vec<GraphWaiterPayload>) {
        for (&id, res) in ids.iter().zip(results) {
            let removed = {
                let mut wmap = self
                    .inner
                    .waiters
                    .lock()
                    .expect("graph waiter map lock poisoned");
                wmap.remove(&id)
            };
            if let Some(waiter) = removed {
                let mut guard = waiter.result.lock().expect("graph waiter lock poisoned");
                *guard = Some(res);
                waiter.cv.notify_all();
            }
        }
    }

    /// PRODUCER: wake + drop every still-pending waiter in `ids` with `reason`
    /// (`fail_remaining_graph_ids:537`) so a mid-loop error return never orphans a
    /// blocked worker. A waiter whose result is already set is left untouched
    /// (tolerant, idempotent). This is the vehicle the D6 build-error reason rides
    /// to the failed waiter.
    pub fn fail_remaining(&self, ids: &[u64], reason: &str) {
        for &id in ids {
            let removed = {
                let mut wmap = self
                    .inner
                    .waiters
                    .lock()
                    .expect("graph waiter map lock poisoned");
                wmap.remove(&id)
            };
            if let Some(waiter) = removed {
                let mut guard = waiter.result.lock().expect("graph waiter lock poisoned");
                if guard.is_none() {
                    *guard = Some(Err(reason.to_string()));
                }
                waiter.cv.notify_all();
            }
        }
    }

    /// Close the graph queue and wake all blocked waiters (`close_rust:394`, graph
    /// half). Disjoint from the dense queue — closing one leaves the other open.
    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.inner.queue_cv.notify_all();
        let wmap = self
            .inner
            .waiters
            .lock()
            .expect("graph waiter map lock poisoned");
        for waiter in wmap.values() {
            waiter.cv.notify_all();
        }
    }

    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::SeqCst)
    }
}

/// Build one leaf's axis graph from its stones, running the WP-1 red-team seam
/// guards (`build_graph_from_request:70` + `build_leaf_graph:516`).
///
/// D6 FIX: returns `Result<AxisGraph, String>` — the build error REASON is
/// preserved and can travel to the failed waiter, replacing the frozen
/// `.ok()`-swallow to `None` (`inference_bridge.rs:530`) that dropped the reason.
/// The guard messages are ported VERBATIM (only the `PyValueError::new_err`
/// wrapper is stripped):
///   - `current_player ∈ {-1, +1}` (range-validate before the `i8` cast);
///   - `moves_remaining ∈ [0, 255]` (before the `u8` cast — Attack-4);
///   - each stone `|q|,|r|` bounded below `i32::MAX - radius` (Attack-2);
///   - each stone player ∈ {-1, +1}; and the native-`builder_impl` handshake.
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
