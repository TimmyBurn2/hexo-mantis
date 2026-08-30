// Exceeds the 300-line soft cap (R8): the full 22-method `InferenceBatcher`
// Python surface (dense + graph paths + mock harness + model-version) REMAPPED
// over the WP6 `DenseQueue`/`GraphQueue`, plus the `GraphWire` pyclass (3 scalar
// + 13 numpy COPY getters + single-read `take()`) port as one line-auditable
// unit with their tests. Splitting the batcher from the wire it fuses would
// scatter the graph-seam story across files (out of the R2 4-file write scope).
//! `InferenceBatcher` (REMAP over WP6 `DenseQueue` + `GraphQueue`) + `GraphWire`
//! pyclass. Behaviour-exact structural port of the frozen `inference_bridge.rs`
//! PyO3 surface: NAME + Python-facing method names PRESERVED (WP8 consumer
//! compat), internals reimplemented over the already-`pub` WP6 queue API.
//!
//! Two new-side reconciliations, both bridge-internal (no cross-crate seam):
//! - The WP6 queues expose NO pending-request accessor, so `has_pending_*` reads
//!   a bridge-side mock-submit counter (accurate for the mock-game harness — the
//!   method's only consumer; a production batcher's real-worker submits flow
//!   straight to the shared queue and are not counted, and `has_pending_*` is a
//!   non-production introspection helper). See the module notes in IMPL_NOTES.
//! - `GraphWire`'s single-read `take()` latch is the NEW WP6 wire capability; the
//!   bridge pyclass owns the moved-out `GraphWireArrays` (repeatable COPY getters,
//!   old behaviour PRESERVED) PLUS a Python-facing `take()` latch that raises
//!   `WireAlreadyConsumed` on a second call.
//!
//! F-42: every pyclass sets `module = "mantis._engine"`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mantis_encoding::RegistrySpec;
use mantis_graph::{AxisGraph, BUILDER_IMPL_NATIVE};
use mantis_search::LegalSetPolicy;
use mantis_selfplay::queues::{
    build_leaf_graph, DenseQueue, GraphQueue, GraphWire, GraphWireArrays,
    WireAlreadyConsumed as WireConsumedGuard,
};
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::SelfPlayRunner;

use crate::encoding::PyRegistrySpec;

pyo3::create_exception!(
    mantis_engine,
    WireAlreadyConsumed,
    pyo3::exceptions::PyException,
    "Raised when GraphWire.take() is called a second time (the single-read latch)."
);

/// What a GIL-free (`py.detach`) section on the graph seam can fail with, carried across
/// the detach boundary as plain data and mapped to its Python face once the GIL is back.
///
/// `WireConsumed` is why this type exists. Both production fuse sites — the trainer's
/// `HexgBuffer::sample_graph_batch` and the server's `next_graph_batch` — used to
/// `expect()` the single-read guard, so a wire that had already yielded its arrays would
/// cross the FFI as a PanicException. Under `panic = "unwind"` that is caught rather than
/// aborting (R2/LAW-13), but the house rule is upstream of that: a production path fails
/// through a NAMED error that propagates, never through a panic.
pub(crate) enum SeamFailure {
    /// An impl-layer failure that already carries its own message.
    Message(String),
    /// The wire's single-read guard fired: the arrays were already taken.
    WireConsumed(WireConsumedGuard),
}

impl From<String> for SeamFailure {
    fn from(message: String) -> Self {
        SeamFailure::Message(message)
    }
}

impl From<WireConsumedGuard> for SeamFailure {
    fn from(guard: WireConsumedGuard) -> Self {
        SeamFailure::WireConsumed(guard)
    }
}

impl SeamFailure {
    /// Map to the Python face: an impl message keeps the `ValueError` the seam has always
    /// raised; the single-read guard routes as the named `WireAlreadyConsumed`.
    pub(crate) fn into_pyerr(self) -> PyErr {
        match self {
            SeamFailure::Message(message) => PyValueError::new_err(message),
            SeamFailure::WireConsumed(guard) => WireAlreadyConsumed::new_err(guard.to_string()),
        }
    }
}

/// Model-version source for a batcher. A standalone batcher (the `new(...)` ctor)
/// owns its counter; a runner-produced batcher (`SelfPlayRunner.batcher`) writes
/// through to the runner's `model_version` atomic so worker threads observe the
/// bump (frozen: the runner read `self.batcher.current_model_version()`; new: the
/// runner OWNS the atomic and the batcher bumps it via the SEAM).
#[derive(Clone)]
enum ModelVersionSrc {
    Own(Arc<AtomicU64>),
    Runner(Arc<SelfPlayRunner>),
}

impl ModelVersionSrc {
    /// Increment and return the new value. `Own`: atomic `fetch_add(1)+1` (frozen
    /// `bump_model_version`). `Runner`: read-inc-write via the SEAM (bumps are
    /// single-threaded — the InferenceServer swaps weights on one thread).
    fn bump(&self) -> u64 {
        match self {
            Self::Own(a) => a.fetch_add(1, Ordering::Relaxed) + 1,
            Self::Runner(r) => {
                let v = r.model_version() + 1;
                r.set_model_version(v);
                v
            }
        }
    }

    fn get(&self) -> u64 {
        match self {
            Self::Own(a) => a.load(Ordering::Relaxed),
            Self::Runner(r) => r.model_version(),
        }
    }
}

/// Per-in-flight-graph assemble metadata retained between `next_graph_batch`
/// (which moves the graph's wire arrays into numpy) and
/// `submit_graph_inference_results` (which reads the slot map + legal coords to
/// build the `LegalSetPolicy` Rust-side). The WP6 `GraphQueue` does NOT retain
/// this (the frozen batcher's `in_flight_graphs` DashMap has no WP6 counterpart),
/// so the bridge holds it.
struct InFlightGraph {
    policy_dst_slot: Vec<i32>,
    legal_coords: Vec<(i32, i32)>,
}

/// Take a lock, RECOVERING from poisoning instead of propagating it.
///
/// Poisoning is a one-way latch: the first panic while the guard is held marks the mutex
/// forever, so every later `.lock().expect(...)` panics too. On this seam that turns one bad
/// graph into a permanently bricked batcher — and under R2/LAW-13 (`panic = "unwind"`) the
/// panic crosses the FFI as a catchable `PanicException` rather than aborting, so the process
/// SURVIVES to keep hitting the dead lock for the rest of the run. Loud once, then silent
/// forever, is the worst of both.
///
/// Recovery is sound HERE specifically because the guarded value is plain owned data
/// (`HashMap<u64, InFlightGraph>` — no raw pointers, no cross-field invariant). The worst a
/// mid-mutation panic can leave behind is a missing or half-updated entry, and the seam
/// already tolerates a missing id by construction: `submit_graph_inference_results` skips
/// unknown ids under the frozen tolerant-remove semantics. A poisoned map is degraded, not
/// unsound, so continuing beats dying.
///
/// Every recovery bumps `counter`, which is what makes this observable rather than a silent
/// swallow (LAW-18: a lever under test logs its own fire-rate in-run).
pub(crate) fn lock_or_recover<'a, T>(
    mutex: &'a Mutex<T>,
    counter: &AtomicUsize,
) -> MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            counter.fetch_add(1, Ordering::SeqCst);
            poisoned.into_inner()
        }
    }
}

/// Saturating decrement of a mock-pending counter (never underflows on a
/// production batcher whose real submits the bridge never incremented).
fn decrement_pending(counter: &AtomicUsize, by: usize) {
    if by == 0 {
        return;
    }
    let mut cur = counter.load(Ordering::SeqCst);
    loop {
        let new = cur.saturating_sub(by);
        match counter.compare_exchange_weak(cur, new, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

/// Rust-owned blocking inference batcher exposed to Python — the fused-model NN
/// face over the WP6 dense + graph queues.
#[pyclass(name = "InferenceBatcher", module = "mantis._engine")]
#[derive(Clone)]
pub struct PyInferenceBatcher {
    dense: DenseQueue,
    graph: GraphQueue,
    feature_len: usize,
    policy_len: usize,
    is_graph: bool,
    representation: &'static str,
    graph_win_length: u8,
    graph_radius: u16,
    graph_trunk_size: i32,
    graph_contract_version: u32,
    model_version: ModelVersionSrc,
    in_flight_graphs: Arc<Mutex<HashMap<u64, InFlightGraph>>>,
    /// Times `in_flight_graphs` was found poisoned and recovered. Non-zero means a panic
    /// happened under the guard on some earlier call; the seam kept serving. Read from Python
    /// via the `lock_recoveries` getter so a run can alert on it instead of discovering it in
    /// a post-mortem (LAW-18).
    lock_recoveries: Arc<AtomicUsize>,
    completed_mock_games: Arc<AtomicUsize>,
    completed_graph_games: Arc<AtomicUsize>,
    dense_pending: Arc<AtomicUsize>,
    graph_pending: Arc<AtomicUsize>,
}

impl PyInferenceBatcher {
    /// Shared field-init body — the `new(...)` ctor and `from_runner` both funnel
    /// through here so the bridge-side state (in-flight map, counters) is created
    /// in one place.
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        dense: DenseQueue,
        graph: GraphQueue,
        feature_len: usize,
        policy_len: usize,
        is_graph: bool,
        representation: &'static str,
        graph_win_length: u8,
        graph_radius: u16,
        graph_trunk_size: i32,
        graph_contract_version: u32,
        model_version: ModelVersionSrc,
    ) -> Self {
        PyInferenceBatcher {
            dense,
            graph,
            feature_len,
            policy_len,
            is_graph,
            representation,
            graph_win_length,
            graph_radius,
            graph_trunk_size,
            graph_contract_version,
            model_version,
            in_flight_graphs: Arc::new(Mutex::new(HashMap::new())),
            lock_recoveries: Arc::new(AtomicUsize::new(0)),
            completed_mock_games: Arc::new(AtomicUsize::new(0)),
            completed_graph_games: Arc::new(AtomicUsize::new(0)),
            dense_pending: Arc::new(AtomicUsize::new(0)),
            graph_pending: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Build a batcher over a runner's live queues (the `SelfPlayRunner.batcher`
    /// getter). Shares the runner's `model_version` atomic (via the SEAM) so a
    /// `bump_model_version` reaches worker threads. `spec` is always present (the
    /// runner resolved it at construction).
    pub(crate) fn from_runner(
        spec: &'static RegistrySpec,
        dense: DenseQueue,
        graph: GraphQueue,
        runner: Arc<SelfPlayRunner>,
    ) -> Self {
        let is_graph = spec.is_graph();
        let (win_length, radius, trunk_size, contract_version) = graph_params(spec);
        Self::from_parts(
            dense,
            graph,
            spec.state_stride(),
            spec.policy_stride(),
            is_graph,
            spec.representation.as_str(),
            win_length,
            radius,
            trunk_size,
            contract_version,
            ModelVersionSrc::Runner(runner),
        )
    }

    /// Graph seam guard: a grid batcher (every dense construction) raises
    /// `RepresentationMismatch` (frozen `require_graph`, error text verbatim).
    /// The ONE place `in_flight_graphs` is locked. Poison-recovering (see `lock_or_recover`)
    /// and recovery-counting; no caller may re-introduce a bare `.lock().expect(...)`.
    fn lock_in_flight(&self) -> MutexGuard<'_, HashMap<u64, InFlightGraph>> {
        lock_or_recover(&self.in_flight_graphs, &self.lock_recoveries)
    }

    fn require_graph(&self) -> PyResult<()> {
        if !self.is_graph {
            return Err(PyValueError::new_err(
                "RepresentationMismatch: graph seam method called on a grid InferenceBatcher \
                 (construct with a representation=\"graph\" encoding spec)",
            ));
        }
        Ok(())
    }

    /// Drop the in-flight metadata for `ids` and wake+fail their still-pending
    /// graph waiters (frozen `fail_remaining_graph_ids`; the WP6 `fail_remaining`
    /// only sets `Err` on a not-yet-set waiter — idempotent).
    fn fail_remaining_graph_ids(&self, ids: &[u64], msg: &str) {
        {
            let mut in_flight = self.lock_in_flight();
            for &id in ids {
                in_flight.remove(&id);
            }
        }
        self.graph.fail_remaining(ids, msg);
    }
}

/// Resolve the `(win_length, radius, trunk_size, contract_version)` graph build
/// params from a spec. A graph spec `.expect`s its `Some` fields (frozen: a
/// missing field on a graph spec is a registry desync — die loud, LOCKED #4
/// PanicException). A grid spec returns the inert `(0, 0, 0, 1)`.
fn graph_params(spec: &'static RegistrySpec) -> (u8, u16, i32, u32) {
    if spec.is_graph() {
        (
            spec.win_length
                .expect("validate guarantees win_length for a graph spec") as u8,
            spec.graph_radius
                .expect("validate guarantees graph_radius for a graph spec") as u16,
            spec.trunk_size as i32,
            spec.contract_version
                .expect("validate guarantees contract_version for a graph spec"),
        )
    } else {
        (0, 0, 0, 1)
    }
}

#[pymethods]
impl PyInferenceBatcher {
    /// Construct a batcher. Feature/policy width precedence: explicit kwargs >
    /// `encoding_spec` derivation > error (the frozen legacy-v6 fallback arms are
    /// retired). `pool_size` is accepted for signature compat but inert — the WP6
    /// queues own no feature-buffer pool (the frozen flume pool is dropped).
    ///
    /// `max_in_flight` declares the most graphs this batcher's callers can ever have queued
    /// at once — `n_workers x leaf_batch_size` for a pool, the leaf-batch width for a
    /// single-stream deploy head. The collector's saturation threshold is DERIVED from it
    /// (ledger F-1/F-2). `0` is the UNDECLARED posture, not a supply of zero: it keeps the
    /// frozen half-batch threshold, which is what every construction that has no pool
    /// behind it wants.
    #[new]
    #[pyo3(signature = (encoding_spec = None, feature_len = None, policy_len = None, pool_size = None, max_in_flight = 0))]
    pub fn new(
        encoding_spec: Option<PyRegistrySpec>,
        feature_len: Option<usize>,
        policy_len: Option<usize>,
        pool_size: Option<usize>,
        max_in_flight: usize,
    ) -> PyResult<Self> {
        let _ = pool_size; // no feature-buffer pool over the WP6 queues (dropped).
        let spec_static: Option<&'static RegistrySpec> =
            encoding_spec.as_ref().map(PyRegistrySpec::inner);
        let (feature_len, policy_len) = match (feature_len, policy_len, spec_static) {
            (Some(f), Some(p), _) => (f, p),
            (None, None, Some(spec)) => (spec.state_stride(), spec.policy_stride()),
            (Some(f), None, Some(spec)) => (f, spec.policy_stride()),
            (None, Some(p), Some(spec)) => (spec.state_stride(), p),
            (None, _, None) | (_, None, None) => {
                return Err(PyValueError::new_err(
                    "InferenceBatcher: encoding_spec required when feature_len/policy_len omitted \
                     (the legacy v6 fallback arms are retired)",
                ));
            }
        };
        let representation = spec_static.map_or("grid", |s| s.representation.as_str());
        let is_graph = spec_static.is_some_and(|s| s.is_graph());
        let (win_length, radius, trunk_size, contract_version) =
            spec_static.map_or((0, 0, 0, 1), graph_params);
        Ok(Self::from_parts(
            DenseQueue::new(feature_len),
            GraphQueue::with_contract_version_and_supply(contract_version, max_in_flight),
            feature_len,
            policy_len,
            is_graph,
            representation,
            win_length,
            radius,
            trunk_size,
            contract_version,
            ModelVersionSrc::Own(Arc::new(AtomicU64::new(0))),
        ))
    }

    // ── dense path ──────────────────────────────────────────────────────────

    /// Spawn N mock inference requests on native threads (test utility). Each
    /// submits a zero feature vector and blocks; increments `completed_mock_games`
    /// on a successful reply.
    pub fn spawn_mock_games(&self, n_games: usize) {
        let feature_len = self.feature_len;
        for _ in 0..n_games {
            let dense = self.dense.clone();
            let completed = self.completed_mock_games.clone();
            let pending = self.dense_pending.clone();
            std::thread::spawn(move || {
                pending.fetch_add(1, Ordering::SeqCst);
                if dense
                    .submit_batch_and_wait(vec![vec![0.0f32; feature_len]])
                    .is_ok()
                {
                    completed.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
    }

    /// How many times the in-flight-graph lock was found poisoned and recovered.
    ///
    /// STAYS ZERO in a healthy run. Non-zero is a real defect report: a panic occurred under
    /// the guard, the seam recovered and kept serving, and the in-flight map may be missing an
    /// entry. Surfaced so a run can alert on it (LAW-18) rather than have it show up as
    /// unexplained missing-id skips much later.
    #[getter]
    pub fn lock_recoveries(&self) -> usize {
        self.lock_recoveries.load(Ordering::SeqCst)
    }

    /// Number of completed mock games (test assertions).
    pub fn completed_mock_games(&self) -> usize {
        self.completed_mock_games.load(Ordering::SeqCst)
    }

    /// Whether at least one mock inference request is currently pending.
    pub fn has_pending_requests(&self) -> bool {
        self.dense_pending.load(Ordering::SeqCst) > 0
    }

    /// Block until at least one request is available or the timeout expires.
    /// Returns `(request_ids, fused (N, feature_len) float32)`; empty on timeout.
    /// Releases the GIL around the blocking pop (frozen behaviour).
    #[pyo3(signature = (batch_size, max_wait_ms = 10))]
    pub fn next_inference_batch<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        max_wait_ms: u64,
    ) -> PyResult<(Vec<u64>, Bound<'py, PyArray2<f32>>)> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }
        let pulled = py.detach(|| self.dense.pop_batch(batch_size, max_wait_ms));
        decrement_pending(&self.dense_pending, pulled.len());
        if pulled.is_empty() {
            // Explicit 0×feature_len tensor for timeout/no-work polls (frozen: an
            // empty from_vec2 can raise and deadlock blocked submitters).
            let arr = PyArray2::<f32>::zeros(py, [0, self.feature_len], false);
            return Ok((Vec::new(), arr));
        }
        let n = pulled.len();
        let mut ids = Vec::with_capacity(n);
        let mut flat = Vec::with_capacity(n * self.feature_len);
        for (id, features) in pulled {
            ids.push(id);
            flat.extend_from_slice(&features);
        }
        let arr = flat.into_pyarray(py).reshape([n, self.feature_len])?;
        Ok((ids, arr))
    }

    /// Submit inference outputs and wake the corresponding waiting requests
    /// (§P74 single-Arc share of the whole policy buffer + per-id ranges).
    pub fn submit_inference_results(
        &self,
        request_ids: Vec<u64>,
        policies: PyReadonlyArray2<f32>,
        values: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let n = request_ids.len();
        if policies.shape()[0] != n || values.len() != n {
            return Err(PyValueError::new_err(format!(
                "length mismatch ids/policies/values: {}/{}/{}",
                n,
                policies.shape()[0],
                values.len()
            )));
        }
        if policies.shape()[1] != self.policy_len {
            return Err(PyValueError::new_err(format!(
                "policy length mismatch: expected {}, got {}",
                self.policy_len,
                policies.shape()[1]
            )));
        }
        let policies_slice = policies.as_slice()?;
        let values_slice = values.as_slice()?;
        let shared: Arc<Vec<f32>> = Arc::new(policies_slice.to_vec());
        let ranges: Vec<std::ops::Range<usize>> = (0..n)
            .map(|i| i * self.policy_len..(i + 1) * self.policy_len)
            .collect();
        self.dense
            .submit_results(&request_ids, &shared, &ranges, values_slice);
        Ok(())
    }

    /// Signal failure for a batch of dense requests.
    pub fn submit_inference_failure(
        &self,
        request_ids: Vec<u64>,
        error_msg: String,
    ) -> PyResult<()> {
        self.dense.submit_failure(&request_ids, &error_msg);
        Ok(())
    }

    /// Close both queues and wake all blocked waiters.
    pub fn close(&self) {
        self.dense.close();
        self.graph.close();
    }

    // ── model version ─────────────────────────────────────────────────────────

    /// Increment the monotonic model version; returns the new value.
    pub fn bump_model_version(&self) -> u64 {
        self.model_version.bump()
    }

    /// Read the current model version (snapshot).
    #[getter]
    pub fn model_version(&self) -> u64 {
        self.model_version.get()
    }

    #[getter]
    pub fn feature_len_py(&self) -> usize {
        self.dense.feature_len()
    }

    #[getter]
    pub fn policy_len_py(&self) -> usize {
        self.policy_len
    }

    /// Wire `representation` ("grid" | "graph").
    #[getter]
    pub fn representation_py(&self) -> &'static str {
        self.representation
    }

    // ── graph path (all but the two counters guard `require_graph`) ────────────

    /// Whether at least one mock graph request is currently pending.
    pub fn has_pending_graph_requests(&self) -> bool {
        self.graph_pending.load(Ordering::SeqCst) > 0
    }

    /// Number of completed mock graph games (test assertions).
    pub fn completed_graph_games(&self) -> usize {
        self.completed_graph_games.load(Ordering::SeqCst)
    }

    /// Seam obligations only: build a graph from the request params (running the
    /// coord / current_player / moves_remaining range guards) and discard it.
    /// Raises `ValueError` on any violation.
    pub fn check_graph_request(
        &self,
        stones: Vec<(i64, i64, i64)>,
        current_player: i64,
        moves_remaining: i64,
    ) -> PyResult<()> {
        self.require_graph()?;
        build_leaf_graph(
            &stones,
            current_player,
            moves_remaining,
            self.graph_win_length,
            self.graph_radius,
            self.graph_trunk_size,
        )
        .map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Spawn N mock graph games on native threads. Each builds a FIXED mixed
    /// spread board (two far clusters → in- + off-window legal nodes) and blocks
    /// on `submit_graph_and_wait`.
    /// The graph queue's DECLARED supply — the most graphs its callers can ever have in
    /// flight (`0` = undeclared). The collector's saturation threshold is derived from it,
    /// so exposing it is what makes the relation observable from a test rather than
    /// inferred from a timing.
    #[getter]
    fn graph_max_in_flight(&self) -> usize {
        self.graph.max_in_flight()
    }

    pub fn spawn_mock_graph_games(&self, n_games: usize) -> PyResult<()> {
        self.require_graph()?;
        let (win_length, radius, trunk_size) = (
            self.graph_win_length,
            self.graph_radius,
            self.graph_trunk_size,
        );
        for _ in 0..n_games {
            let graph_q = self.graph.clone();
            let completed = self.completed_graph_games.clone();
            let pending = self.graph_pending.clone();
            std::thread::spawn(move || {
                let mut stones: Vec<(i64, i64, i64)> = Vec::new();
                for q in 0..5i64 {
                    stones.push((q, 0, 1));
                }
                for q in 30..35i64 {
                    stones.push((q, 0, -1));
                }
                if let Ok(graph) = build_leaf_graph(&stones, 1, 2, win_length, radius, trunk_size) {
                    pending.fetch_add(1, Ordering::SeqCst);
                    if graph_q.submit_graph_and_wait(graph).is_ok() {
                        completed.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        }
        Ok(())
    }

    /// Pop up to `batch_size` graph requests, fuse them PyG block-diagonal, and
    /// return `(request_ids, GraphWire)`. Retains per-id assemble metadata for
    /// `submit_graph_inference_results`. Releases the GIL around the blocking pop.
    #[pyo3(signature = (batch_size, max_wait_ms = 10))]
    pub fn next_graph_batch(
        &self,
        py: Python<'_>,
        batch_size: usize,
        max_wait_ms: u64,
    ) -> PyResult<(Vec<u64>, PyGraphWire)> {
        self.require_graph()?;
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be > 0"));
        }
        let pulled = py.detach(|| self.graph.pop_graph_batch(batch_size, max_wait_ms));
        decrement_pending(&self.graph_pending, pulled.len());

        let mut ids: Vec<u64> = Vec::with_capacity(pulled.len());
        let mut graphs: Vec<AxisGraph> = Vec::with_capacity(pulled.len());
        {
            let mut in_flight = self.lock_in_flight();
            for (id, graph) in pulled {
                // builder_impl handshake (defense-in-depth; the build path asserted
                // it) — a non-native tag must never reach the wire.
                if graph.builder_impl != BUILDER_IMPL_NATIVE {
                    return Err(PyValueError::new_err(
                        "next_graph_batch: non-native builder_impl on a queued graph",
                    ));
                }
                // CHECKED, not indexed. This runs WITH `in_flight` held, so a panic here
                // poisons the lock for the rest of the process — and `legal_node_gather` is
                // queue-supplied data indexing a SEPARATE array (`node_coords`), which is
                // precisely the pairing where a builder bug or a truncated wire yields an
                // out-of-range row. `lock_or_recover` above is the second line of defence;
                // this is the first: turn the malformed graph into a `PyValueError` the
                // caller can see, and never enter the panic path at all.
                let legal_coords: Option<Vec<(i32, i32)>> = graph
                    .legal_node_gather
                    .iter()
                    .map(|&row| {
                        let base = usize::try_from(row).ok()?.checked_mul(2)?;
                        Some((
                            *graph.node_coords.get(base)?,
                            *graph.node_coords.get(base.checked_add(1)?)?,
                        ))
                    })
                    .collect();
                let Some(legal_coords) = legal_coords else {
                    return Err(PyValueError::new_err(format!(
                        "next_graph_batch: legal_node_gather holds a row out of range for \
                         node_coords (len {}) on graph id {id} — malformed graph wire",
                        graph.node_coords.len(),
                    )));
                };
                in_flight.insert(
                    id,
                    InFlightGraph {
                        policy_dst_slot: graph.policy_scatter_index.0.clone(),
                        legal_coords,
                    },
                );
                ids.push(id);
                graphs.push(graph);
            }
        }
        // GIL-FREE, like the pop above it (A5). The fuse is 25.3 % of the card and pure
        // Rust CPU (ledger §10.1 #2); holding the GIL through it blocks every other Python
        // thread in the process for its whole duration. `&[AxisGraph]` carries no Python
        // object, so the closure is `Ungil`.
        let contract_version = self.graph_contract_version;
        let fused = py.detach(move || {
            let mut wire = GraphWire::from_axis_graphs(&graphs, contract_version);
            wire.take().map_err(SeamFailure::from)
        });
        let arrays = fused.map_err(SeamFailure::into_pyerr)?;
        Ok((ids, PyGraphWire::from_arrays(arrays)))
    }

    /// Ragged OUTPUT: wake each graph waiter with its assembled
    /// `(LegalSetPolicy, value)`. `legal_offsets` segments `legal_probs_flat` per
    /// id; the per-leaf `LegalSetPolicy` is built Rust-side from the retained
    /// `policy_dst_slot` + coords via `assemble_ls_from_gnn_probs`.
    pub fn submit_graph_inference_results(
        &self,
        request_ids: Vec<u64>,
        legal_probs_flat: PyReadonlyArray1<f32>,
        legal_offsets: PyReadonlyArray1<i64>,
        values: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        self.require_graph()?;
        let n = request_ids.len();
        if values.len() != n {
            return Err(PyValueError::new_err(format!(
                "length mismatch ids/values: {}/{}",
                n,
                values.len()
            )));
        }
        let lo = legal_offsets.as_slice()?;
        if lo.len() != n + 1 {
            return Err(PyValueError::new_err(format!(
                "legal_offsets length {} != n+1 ({})",
                lo.len(),
                n + 1
            )));
        }
        let probs = legal_probs_flat.as_slice()?;
        let vals = values.as_slice()?;
        if lo[0] != 0 || (lo[n] as usize) != probs.len() {
            return Err(PyValueError::new_err(format!(
                "legal_offsets endpoints [{},{}] inconsistent with legal_probs_flat len {}",
                lo[0],
                lo[n],
                probs.len()
            )));
        }

        for i in 0..n {
            let id = request_ids[i];
            let start = lo[i];
            let end = lo[i + 1];
            if start < 0 || end < start || (end as usize) > probs.len() {
                let msg = format!("legal_offsets segment [{start},{end}] out of range for id {id}");
                self.fail_remaining_graph_ids(&request_ids[i..], &msg);
                return Err(PyValueError::new_err(msg));
            }
            let leaf_probs = &probs[start as usize..end as usize];

            let meta = { self.lock_in_flight().remove(&id) };
            // Unknown id (already consumed / never emitted) — skip (frozen tolerant
            // remove semantics).
            let Some(meta) = meta else { continue };
            if meta.policy_dst_slot.len() != leaf_probs.len() {
                let msg = format!(
                    "submit_graph_inference_results: segment len {} != n_legal {} for id {id}",
                    leaf_probs.len(),
                    meta.policy_dst_slot.len()
                );
                self.graph
                    .submit_graph_results(&[id], vec![Err(msg.clone())]);
                self.fail_remaining_graph_ids(&request_ids[i + 1..], &msg);
                return Err(PyValueError::new_err(msg));
            }
            match assemble_ls_from_gnn_probs(
                self.policy_len,
                leaf_probs,
                &meta.policy_dst_slot,
                &meta.legal_coords,
            ) {
                Ok(ls) => {
                    self.graph
                        .submit_graph_results(&[id], vec![Ok((ls, vals[i]))]);
                }
                Err(e) => {
                    self.graph.submit_graph_results(&[id], vec![Err(e.clone())]);
                    self.fail_remaining_graph_ids(&request_ids[i + 1..], &e);
                    return Err(PyValueError::new_err(e));
                }
            }
        }
        Ok(())
    }

    /// Signal failure for a batch of graph requests: wake each waiter with `Err`
    /// and drop its in-flight state (frozen `submit_graph_inference_failure`;
    /// DESIGN §a.1 row 20 maps this onto the idempotent WP6 `fail_remaining`).
    pub fn submit_graph_inference_failure(
        &self,
        request_ids: Vec<u64>,
        error_msg: String,
    ) -> PyResult<()> {
        self.require_graph()?;
        self.fail_remaining_graph_ids(&request_ids, &error_msg);
        Ok(())
    }

    /// Blocking graph-inference driver for the step-0 smoke / eval round-trip:
    /// build one axis graph per position, submit them, release the GIL, and block
    /// until each leaf's `LegalSetPolicy` is assembled. Returns per-position
    /// `(dense, overflow[(q,r)->prob], value)`.
    ///
    /// This is `submit_graphs_and_wait_ls` with the builder's window centre
    /// PROJECTED AWAY — ONE authority for the graph build + submit sequence, so
    /// the frame-carrying driver and this one cannot drift (WP12-R D-22). The
    /// 3-tuple output is byte-identical to the pre-WP12-R implementation.
    #[allow(clippy::type_complexity)]
    pub fn submit_graphs_and_wait(
        &self,
        py: Python<'_>,
        positions: Vec<(Vec<(i64, i64, i64)>, i64, i64)>,
    ) -> PyResult<Vec<(Vec<f32>, Vec<((i32, i32), f32)>, f32)>> {
        Ok(self
            .submit_graphs_and_wait_ls(py, positions)?
            .into_iter()
            .map(|(dense, overflow, value, _center)| (dense, overflow, value))
            .collect())
    }

    /// Graph submit-and-wait carrying the BUILDER's frame (WP12-R Phase
    /// EVALDECODE, operator ruling R138). Returns per-position
    /// `(dense, overflow[(q,r)->prob], value, window_center)`.
    ///
    /// Why the centre must come from HERE (DESIGN §c.2): self-play frames its
    /// expand on the builder's `g.window_center` (`search_drive.rs:373 -> :421`),
    /// `Board` does NOT expose `window_center` to Python, and a bridge method that
    /// silently recomputed one from the pending board would erase the only
    /// leaf/policy alignment cross-check there is — the hazard the CNN sibling
    /// guards against ("never trust a Python-supplied order", `mcts.rs:152-153`).
    /// `MCTSTree.expand_and_backup_ls_graph` cross-checks what this returns
    /// against the pending board with an always-on `PyValueError`.
    #[allow(clippy::type_complexity)]
    pub fn submit_graphs_and_wait_ls(
        &self,
        py: Python<'_>,
        positions: Vec<(Vec<(i64, i64, i64)>, i64, i64)>,
    ) -> PyResult<Vec<(Vec<f32>, Vec<((i32, i32), f32)>, f32, (i32, i32))>> {
        self.require_graph()?;
        let mut graphs = Vec::with_capacity(positions.len());
        for (stones, current_player, moves_remaining) in &positions {
            let g = build_leaf_graph(
                stones,
                *current_player,
                *moves_remaining,
                self.graph_win_length,
                self.graph_radius,
                self.graph_trunk_size,
            )
            .map_err(PyValueError::new_err)?;
            graphs.push(g);
        }
        // The builder's own centre, captured BEFORE the graphs are moved into the
        // detached submit loop; index-aligned with `positions` and therefore with
        // the results below.
        let centers: Vec<(i32, i32)> = graphs.iter().map(|g| g.window_center).collect();
        let graph_q = self.graph.clone();
        // ONE batch submit, not a serial loop of blocking submits (Q-FIND-1/R263).
        // The eval/arena decode shares the self-play dispatch behaviour by
        // construction — a serial arm here would make the two disagree about
        // batching, the drift the WP12-R D-22 "ONE authority" note above exists to
        // prevent. `collect` into a `Result` scans a Vec whose waiters have ALL
        // already resolved, so the first-`Err` return cannot orphan a tail waiter.
        let results: Result<Vec<(LegalSetPolicy, f32)>, String> =
            py.detach(|| graph_q.submit_graphs_and_wait(graphs).into_iter().collect());
        let results = results.map_err(PyValueError::new_err)?;
        let out = results
            .into_iter()
            .zip(centers)
            .map(|((ls, v), center)| {
                let overflow: Vec<((i32, i32), f32)> =
                    ls.overflow.iter().map(|(&k, &p)| (k, p)).collect();
                (ls.dense, overflow, v, center)
            })
            .collect();
        Ok(out)
    }
}

/// Block-diagonal ragged graph wire — the fuse-out of `next_graph_batch` /
/// `HexgBuffer.sample_graph_batch`. Owns the moved-out `GraphWireArrays`.
///
/// `take()` MOVES every array into numpy (`IntoPyArray`, which "consumes `self` and moves
/// its data into a NumPy array") rather than copying it — PERF-TRANCHE-1 A2, against ledger
/// §10.1 #4, `wire_copyout` 12.43 ms/pop of pure memcpy-plus-first-touch. The production
/// serve loop reads this face exactly once (`graph_wire_from_rust`), so the copy it used to
/// pay bought nothing.
///
/// CONTRACT CHANGE, deliberate: the 13 per-array getters still COPY and are still freely
/// repeatable, but only UNTIL `take()`. After `take()` the buffers are gone — they belong to
/// numpy — and every getter raises `WireAlreadyConsumed`. The single-read latch is now the
/// `Option` itself rather than a flag beside the data, so there is no state in which a getter
/// can hand back an empty array and have it read as a measurement.
#[pyclass(name = "GraphWire", module = "mantis._engine")]
pub struct PyGraphWire {
    arrays: Option<GraphWireArrays>,
}

impl PyGraphWire {
    /// Wrap the arrays a caller already moved out of the WP6 `GraphWire` (via one
    /// internal Rust `take()` at fuse time).
    pub(crate) fn from_arrays(arrays: GraphWireArrays) -> Self {
        PyGraphWire {
            arrays: Some(arrays),
        }
    }

    /// The arrays, or the named refusal if `take()` already moved them into numpy.
    fn arrays(&self) -> PyResult<&GraphWireArrays> {
        self.arrays.as_ref().ok_or_else(|| {
            WireAlreadyConsumed::new_err(
                "GraphWire arrays already consumed (take() MOVED them into numpy; the \
                 per-array getters read the wire's own buffers and there are none left)",
            )
        })
    }
}

#[pymethods]
impl PyGraphWire {
    #[getter]
    fn contract_version(&self) -> PyResult<u32> {
        Ok(self.arrays()?.contract_version)
    }
    #[getter]
    fn builder_impl(&self) -> PyResult<u8> {
        Ok(self.arrays()?.builder_impl)
    }
    #[getter]
    fn n_graphs(&self) -> PyResult<usize> {
        Ok(self.arrays()?.n_graphs)
    }
    #[getter]
    fn node_feat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.node_feat))
    }
    #[getter]
    fn node_coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.node_coords))
    }
    #[getter]
    fn edge_index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.edge_index))
    }
    #[getter]
    fn edge_attr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.edge_attr))
    }
    #[getter]
    fn node_offsets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.node_offsets))
    }
    #[getter]
    fn edge_offsets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.edge_offsets))
    }
    #[getter]
    fn legal_offsets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.legal_offsets))
    }
    #[getter]
    fn legal_node_gather<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.legal_node_gather))
    }
    #[getter]
    fn policy_dst_slot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.policy_dst_slot))
    }
    #[getter]
    fn n_nodes_checksum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.n_nodes_checksum))
    }
    #[getter]
    fn n_stones<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.n_stones))
    }
    #[getter]
    fn window_center<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.window_center))
    }
    #[getter]
    fn current_player<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i8>>> {
        Ok(PyArray1::from_slice(py, &self.arrays()?.current_player))
    }

    /// Single-read latch: MOVES all wire fields out once as a dict; a second call — and
    /// every per-array getter afterwards — raises `WireAlreadyConsumed`.
    ///
    /// `into_pyarray` hands numpy the `Vec`'s own allocation instead of memcpying it into a
    /// fresh one, which is the whole of A2. numpy's `resize` cannot be used on an array
    /// built this way; nothing in this repo resizes a wire array.
    fn take<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let a = self.arrays.take().ok_or_else(|| {
            WireAlreadyConsumed::new_err(
                "GraphWire arrays already consumed (single-read take() called twice)",
            )
        })?;
        let d = PyDict::new(py);
        d.set_item("contract_version", a.contract_version)?;
        d.set_item("builder_impl", a.builder_impl)?;
        d.set_item("n_graphs", a.n_graphs)?;
        d.set_item("node_feat", a.node_feat.into_pyarray(py))?;
        d.set_item("node_coords", a.node_coords.into_pyarray(py))?;
        d.set_item("edge_index", a.edge_index.into_pyarray(py))?;
        d.set_item("edge_attr", a.edge_attr.into_pyarray(py))?;
        d.set_item("node_offsets", a.node_offsets.into_pyarray(py))?;
        d.set_item("edge_offsets", a.edge_offsets.into_pyarray(py))?;
        d.set_item("legal_offsets", a.legal_offsets.into_pyarray(py))?;
        d.set_item("legal_node_gather", a.legal_node_gather.into_pyarray(py))?;
        d.set_item("policy_dst_slot", a.policy_dst_slot.into_pyarray(py))?;
        d.set_item("n_nodes_checksum", a.n_nodes_checksum.into_pyarray(py))?;
        d.set_item("n_stones", a.n_stones.into_pyarray(py))?;
        d.set_item("window_center", a.window_center.into_pyarray(py))?;
        d.set_item("current_player", a.current_player.into_pyarray(py))?;
        Ok(d)
    }
}

/// Register the `InferenceBatcher` + `GraphWire` pyclasses and the
/// `WireAlreadyConsumed` exception into `_engine`. Called by Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInferenceBatcher>()?;
    m.add_class::<PyGraphWire>()?;
    m.add(
        "WireAlreadyConsumed",
        m.py().get_type::<WireAlreadyConsumed>(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v6_spec() -> &'static RegistrySpec {
        mantis_encoding::lookup("v6").expect("v6 registered")
    }

    fn gnn_spec() -> &'static RegistrySpec {
        mantis_encoding::lookup("gnn_axis_v1").expect("gnn_axis_v1 registered")
    }

    // ── poisoned-lock recovery (item 2) ───────────────────────────────────────────────
    //
    // Poison a real `Mutex` the only way it can be poisoned — panic while the guard is held,
    // on another thread — then prove the seam keeps working. Without recovery every one of
    // these would panic instead, which is the bricked-batcher defect.

    /// Panic under the guard on a scratch thread; returns once the mutex is genuinely poisoned.
    fn poison<T: Send + 'static>(mutex: &Arc<Mutex<T>>) {
        let handle = Arc::clone(mutex);
        let joined = std::thread::spawn(move || {
            let _guard = handle.lock().expect("scratch thread takes a clean lock");
            panic!("deliberate poison for the recovery test");
        })
        .join();
        assert!(joined.is_err(), "the scratch thread was supposed to panic");
        assert!(mutex.is_poisoned(), "mutex did not actually get poisoned");
    }

    #[test]
    fn lock_or_recover_recovers_and_counts() {
        let mutex = Arc::new(Mutex::new(HashMap::<u64, InFlightGraph>::new()));
        let counter = AtomicUsize::new(0);
        lock_or_recover(&mutex, &counter).insert(
            7,
            InFlightGraph {
                policy_dst_slot: vec![1, 2],
                legal_coords: vec![(3, 4)],
            },
        );
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "a clean lock must not count"
        );

        poison(&mutex);

        let guard = lock_or_recover(&mutex, &counter);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "recovery must be counted (LAW-18)"
        );
        // The data is intact: recovery hands back the map, it does not reset it.
        assert_eq!(
            guard
                .get(&7)
                .expect("entry survived poisoning")
                .policy_dst_slot,
            vec![1, 2]
        );
        drop(guard);

        // Poisoning is a one-way latch, so every later lock recovers and counts again.
        drop(lock_or_recover(&mutex, &counter));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn seam_survives_a_poisoned_in_flight_lock_and_reports() {
        let b = PyInferenceBatcher::new(
            Some(PyRegistrySpec::from_static(gnn_spec())),
            None,
            None,
            None,
            0,
        )
        .expect("graph batcher constructs");
        assert_eq!(
            b.lock_recoveries(),
            0,
            "a fresh batcher has recovered nothing"
        );

        poison(&b.in_flight_graphs);

        // The real seam method, on the poisoned lock. Before the fix this panicked.
        b.fail_remaining_graph_ids(&[1, 2, 3], "post-poison call");

        assert!(
            b.lock_recoveries() >= 1,
            "the seam recovered but did not REPORT — a silent swallow is what LAW-18 forbids"
        );
        // And it is still usable afterwards, not wedged.
        b.fail_remaining_graph_ids(&[4], "second post-poison call");
        assert!(b.lock_recoveries() >= 2);
    }

    /// The mutation self-test for the recovery arm (LAW-07): if `lock_or_recover` ever stops
    /// counting, `lock_recoveries` becomes a phantom input that reads 0 through a real
    /// incident. Mechanism: the counter is the ONLY evidence a poisoning happened — the
    /// recovered map looks identical to a healthy one, so an uncounted recovery is invisible
    /// by construction.
    #[test]
    fn recovery_counter_is_the_only_evidence_and_it_moves() {
        let mutex = Arc::new(Mutex::new(HashMap::<u64, InFlightGraph>::new()));
        let counter = AtomicUsize::new(0);
        poison(&mutex);
        let before = counter.load(Ordering::SeqCst);
        let guard = lock_or_recover(&mutex, &counter);
        let after = counter.load(Ordering::SeqCst);
        assert!(
            guard.is_empty(),
            "recovered map is indistinguishable from a healthy empty one"
        );
        assert_eq!(
            after - before,
            1,
            "counter did not move on a recovery that definitely happened"
        );
    }

    #[test]
    fn grid_batcher_derives_shapes_and_is_grid() {
        let b = PyInferenceBatcher::new(
            Some(PyRegistrySpec::from_static(v6_spec())),
            None,
            None,
            None,
            0,
        )
        .expect("v6 batcher constructs");
        assert!(!b.is_graph);
        assert_eq!(b.representation, "grid");
        assert_eq!(b.feature_len, v6_spec().state_stride());
        assert_eq!(b.policy_len, v6_spec().policy_stride());
    }

    #[test]
    fn explicit_lens_without_spec_construct() {
        let b = PyInferenceBatcher::new(None, Some(2888), Some(362), None, 0)
            .expect("explicit lens construct");
        assert_eq!(b.feature_len, 2888);
        assert_eq!(b.policy_len, 362);
    }

    #[test]
    fn no_spec_no_lens_errors() {
        assert!(PyInferenceBatcher::new(None, None, None, None, 0).is_err());
        assert!(PyInferenceBatcher::new(None, Some(2888), None, None, 0).is_err());
    }

    #[test]
    fn graph_batcher_reads_graph_params() {
        let b = PyInferenceBatcher::new(
            Some(PyRegistrySpec::from_static(gnn_spec())),
            None,
            None,
            None,
            0,
        )
        .expect("graph batcher constructs");
        assert!(b.is_graph);
        assert_eq!(b.representation, "graph");
        assert_eq!(b.graph_win_length, 6);
        assert_eq!(b.graph_radius, 6);
        assert_eq!(b.graph_trunk_size, 19);
        assert_eq!(b.graph_contract_version, 1);
    }

    #[test]
    fn model_version_own_bump_and_get() {
        let b = PyInferenceBatcher::new(None, Some(8), Some(4), None, 0).unwrap();
        assert_eq!(b.model_version(), 0);
        assert_eq!(b.bump_model_version(), 1);
        assert_eq!(b.bump_model_version(), 2);
        assert_eq!(b.model_version(), 2);
    }

    #[test]
    fn grid_batcher_rejects_graph_seam_methods() {
        let b = PyInferenceBatcher::new(None, Some(8), Some(4), None, 0).unwrap();
        assert!(b.require_graph().is_err());
        assert!(b.check_graph_request(vec![(0, 0, 1)], 1, 2).is_err());
        assert!(b.spawn_mock_graph_games(1).is_err());
    }

    /// GraphWire single-read `take()` latch (O20 / ADV): a fresh wire is
    /// takeable once; a second acquisition of the latch fails, and `take()` on an
    /// already-consumed wire raises `WireAlreadyConsumed`. Numpy-free: the guard
    /// returns the mapped exception BEFORE materializing the array dict (the
    /// numpy-materializing legs of take()/getters are pinned by the Python O20
    /// surface post-ASM — the embedded cargo-test interpreter can't load numpy).
    #[test]
    fn graph_wire_take_is_single_read() {
        let stones: Vec<(i64, i64, i64)> = (0..5i64)
            .map(|q| (q, 0, 1))
            .chain((30..35i64).map(|q| (q, 0, -1)))
            .collect();
        let graph = build_leaf_graph(&stones, 1, 2, 6, 6, 19).expect("valid graph");
        let mut wire = GraphWire::from_axis_graphs(&[graph], 1);
        let arrays = wire.take().expect("first fuse take");
        let gw = PyGraphWire::from_arrays(arrays);

        // The NUMPY-FREE half of the latch, which is all this interpreter can witness: a
        // fresh wire serves its scalars, and a wire whose arrays are gone refuses them.
        // `take()` itself materialises numpy arrays and PANICS here on the absent module,
        // so the consumed-wire path is driven from Python, where numpy exists:
        // `tests/bridge/test_graph_wire_adv.py::test_wire_getters_refuse_after_take` and
        // `::test_take_moves_rather_than_copies`.
        assert_eq!(gw.n_graphs().expect("a fresh wire serves its scalars"), 1);

        // A2 made the latch the `Option` itself rather than a flag beside the data, so
        // "consumed" is constructible without going through numpy at all.
        let consumed = PyGraphWire { arrays: None };
        assert!(
            consumed.n_graphs().is_err(),
            "after take() the buffers belong to numpy, so every getter must refuse rather \
             than serve an empty array"
        );
        Python::initialize();
        Python::attach(|py| {
            let mut consumed = PyGraphWire { arrays: None };
            assert!(
                consumed.take(py).is_err(),
                "a consumed wire's take() raises WireAlreadyConsumed"
            );
        });
    }

    /// The routing both production fuse sites now use (SEAM-B1 §1(a)). A wire whose arrays
    /// are gone yields the guard as a VALUE that `SeamFailure` maps to the named
    /// `WireAlreadyConsumed`; the impl channel keeps its `ValueError`. Before this, the
    /// same state met `expect()` and crossed the FFI as a PanicException.
    #[test]
    fn consumed_wire_routes_named_rather_than_panicking() {
        let stones: Vec<(i64, i64, i64)> = (0..5i64)
            .map(|q| (q, 0, 1))
            .chain((30..35i64).map(|q| (q, 0, -1)))
            .collect();
        let graph = build_leaf_graph(&stones, 1, 2, 6, 6, 19).expect("valid graph");
        let mut wire = GraphWire::from_axis_graphs(&[graph], 1);
        wire.take().expect("the first take yields the fused arrays");

        // The exact expression both production sites run, on an already-consumed wire.
        let routed = wire.take().map_err(SeamFailure::from);
        let Err(failure) = routed else {
            panic!("a consumed wire must not yield arrays a second time");
        };

        Python::initialize();
        Python::attach(|py| {
            let named = failure.into_pyerr();
            assert!(
                named.is_instance_of::<WireAlreadyConsumed>(py),
                "the single-read guard routes as the named exception, not as a panic or a \
                 bare ValueError"
            );
            let message = SeamFailure::Message("ring said no".to_string()).into_pyerr();
            assert!(
                message.is_instance_of::<PyValueError>(py),
                "the impl channel keeps the ValueError face the seam has always raised"
            );
        });
    }
}
