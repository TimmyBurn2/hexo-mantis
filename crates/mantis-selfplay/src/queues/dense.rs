//! Dense inference queue — the pure-Rust half of the frozen `inference_bridge.rs`
//! dense batcher (WP6 D4), pyo3/numpy STRIPPED.
//!
//! Behaviour-identical structural port of the dense `Inner`
//! (`inference_bridge.rs:137-140`): a `Mutex<VecDeque<PendingRequest>>` request
//! queue + `Condvar` + a waiter map. The frozen `DashMap<u64, Arc<Waiter>>`
//! becomes a `Mutex<HashMap<u64, Arc<Waiter>, FxBuildHasher>>` — WP6 adds NO new
//! dependency (DESIGN §0.6), and the observable single-owner semantics are
//! identical (insert/remove/iterate are the only operations, none nested under a
//! held waiter lock).
//!
//! CONSUMER (worker) face: `submit_batch_and_wait` (from
//! `submit_batch_and_wait_rust:334`) — `Err(())` on closed / length-mismatch, so
//! the worker just skips the batch (dense reason NOT required, D6). PRODUCER face
//! is pure-Rust `pop_batch` + `submit_results` / `submit_failure` (replacing the
//! WP7-owed numpy `next_inference_batch` / `submit_inference_results`); tests +
//! benches drive a mock producer over these. The graph batcher is a DISJOINT
//! module (`queues::graph`) — the graph path never touches this pool.

use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use fxhash::FxBuildHasher;

/// One queued dense inference request (`inference_bridge.rs:17`).
#[derive(Clone)]
struct PendingRequest {
    id: u64,
    features: Vec<f32>,
}

/// Waiter result payload (§P74, `inference_bridge.rs:129`): the policy buffer is
/// delivered as `(Arc<Vec<f32>>, Range<usize>, f32)` so a single bulk `to_vec()`
/// at the submitter side replaces N per-row `to_vec()` allocations. Consumers
/// materialise the owned `Vec<f32>` only at pull-time via `arc[range].to_vec()`.
type WaiterPayload = Result<(Arc<Vec<f32>>, Range<usize>, f32), String>;

#[derive(Default)]
struct Waiter {
    result: Mutex<Option<WaiterPayload>>,
    cv: Condvar,
}

struct DenseInner {
    queue: Mutex<VecDeque<PendingRequest>>,
    queue_cv: Condvar,
    waiters: Mutex<HashMap<u64, Arc<Waiter>, FxBuildHasher>>,
    next_id: AtomicU64,
    closed: AtomicBool,
}

impl DenseInner {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            queue_cv: Condvar::new(),
            waiters: Mutex::new(HashMap::with_hasher(FxBuildHasher::default())),
            next_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
        }
    }

    /// Block until at least `batch_size/2` requests are queued OR the timeout
    /// expires, then pop up to `batch_size` (`pop_batch_blocking:271`).
    fn pop_batch_blocking(&self, batch_size: usize, max_wait_ms: u64) -> Vec<PendingRequest> {
        let deadline = Instant::now() + Duration::from_millis(max_wait_ms);
        let mut queue = self.queue.lock().expect("queue lock poisoned");
        // Target at least 50% saturation before waking (frozen: 32/64).
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
                .expect("queue condvar poisoned");
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

/// Rust-owned blocking dense inference queue (`InferenceBatcher` dense half,
/// pyo3 stripped). Clone shares one `Arc<DenseInner>`.
#[derive(Clone)]
pub struct DenseQueue {
    inner: Arc<DenseInner>,
    feature_len: usize,
}

impl DenseQueue {
    /// Construct a dense queue expecting `feature_len`-wide feature vectors.
    #[must_use]
    pub fn new(feature_len: usize) -> Self {
        Self {
            inner: Arc::new(DenseInner::new()),
            feature_len,
        }
    }

    #[must_use]
    pub fn feature_len(&self) -> usize {
        self.feature_len
    }

    /// CONSUMER (worker): enqueue every feature vector, block on each waiter, and
    /// return the `(policy, value)` per request in order. `Err(())` on a closed
    /// queue or a feature-length mismatch — the worker skips the batch (dense
    /// reason NOT required, D6). Verbatim port of `submit_batch_and_wait_rust:334`.
    ///
    /// # Errors
    /// Returns `Err(())` if the queue is closed, a feature vector's length does
    /// not equal `feature_len`, or inference for any request failed.
    #[allow(clippy::result_unit_err)]
    pub fn submit_batch_and_wait(
        &self,
        batch_features: Vec<Vec<f32>>,
    ) -> Result<Vec<(Vec<f32>, f32)>, ()> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(());
        }

        let n = batch_features.len();
        let mut waiters = Vec::with_capacity(n);
        {
            let mut queue = self.inner.queue.lock().expect("queue lock poisoned");
            let mut wmap = self.inner.waiters.lock().expect("waiter map lock poisoned");
            for features in batch_features {
                if features.len() != self.feature_len {
                    return Err(());
                }
                let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
                let waiter = Arc::new(Waiter::default());
                wmap.insert(id, waiter.clone());
                queue.push_back(PendingRequest { id, features });
                waiters.push(waiter);
            }
            drop(wmap);
            self.inner.queue_cv.notify_all();
        }

        let mut results = Vec::with_capacity(n);
        for waiter in waiters {
            let mut guard = waiter.result.lock().expect("waiter lock poisoned");
            loop {
                if let Some(res) = guard.take() {
                    // §P74: materialise the owned Vec at pull-time. Single-read via
                    // `guard.take()` (`inference_bridge.rs:367`).
                    match res {
                        Ok((policy_arc, range, value)) => {
                            let policy = policy_arc[range].to_vec();
                            results.push((policy, value));
                            break;
                        }
                        Err(_) => return Err(()),
                    }
                }
                if self.inner.closed.load(Ordering::SeqCst) {
                    return Err(());
                }
                guard = waiter.cv.wait(guard).expect("waiter condvar poisoned");
            }
        }
        Ok(results)
    }

    /// PRODUCER: pop up to `max` queued requests, returning each `(id, features)`
    /// (pure-Rust replacement for `next_inference_batch`). Empty on timeout /
    /// closed-empty queue (the underflow path).
    #[must_use]
    pub fn pop_batch(&self, max: usize, timeout_ms: u64) -> Vec<(u64, Vec<f32>)> {
        self.inner
            .pop_batch_blocking(max, timeout_ms)
            .into_iter()
            .map(|req| (req.id, req.features))
            .collect()
    }

    /// PRODUCER: wake each `ids[i]` waiter with `policies[ranges[i]]` + `values[i]`
    /// (§P74 single-Arc share — `submit_inference_results:722`). An `id` with no
    /// registered waiter is tolerantly dropped (unknown / already-consumed), the
    /// frozen `waiters.remove` semantics. A second submit for the same id is a
    /// no-op — single-delivery.
    pub fn submit_results(
        &self,
        ids: &[u64],
        policies: &Arc<Vec<f32>>,
        ranges: &[Range<usize>],
        values: &[f32],
    ) {
        for (i, &id) in ids.iter().enumerate() {
            let removed = {
                let mut wmap = self.inner.waiters.lock().expect("waiter map lock poisoned");
                wmap.remove(&id)
            };
            if let Some(waiter) = removed {
                let policy_arc = Arc::clone(policies);
                let mut guard = waiter.result.lock().expect("waiter lock poisoned");
                *guard = Some(Ok((policy_arc, ranges[i].clone(), values[i])));
                waiter.cv.notify_all();
            }
        }
    }

    /// PRODUCER: signal failure for a batch of requests
    /// (`submit_inference_failure:785`). The consumer collapses `Err(_) → Err(())`
    /// and skips the batch (dense reason not consumed, D6).
    pub fn submit_failure(&self, ids: &[u64], error_msg: &str) {
        for &id in ids {
            let removed = {
                let mut wmap = self.inner.waiters.lock().expect("waiter map lock poisoned");
                wmap.remove(&id)
            };
            if let Some(waiter) = removed {
                let mut guard = waiter.result.lock().expect("waiter lock poisoned");
                *guard = Some(Err(error_msg.to_string()));
                waiter.cv.notify_all();
            }
        }
    }

    /// Close the queue and wake all blocked waiters (`close_rust:394`, dense half).
    /// Blocked consumers then observe `closed` and return `Err(())`.
    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.inner.queue_cv.notify_all();
        let wmap = self.inner.waiters.lock().expect("waiter map lock poisoned");
        for waiter in wmap.values() {
            waiter.cv.notify_all();
        }
    }

    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::SeqCst)
    }
}
