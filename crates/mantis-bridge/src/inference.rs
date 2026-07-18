// Exceeds the 300-line soft cap (R8): the full 21-method `InferenceBatcher`
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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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
use mantis_selfplay::queues::{build_leaf_graph, DenseQueue, GraphQueue, GraphWire, GraphWireArrays};
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::SelfPlayRunner;

use crate::encoding::PyRegistrySpec;

pyo3::create_exception!(
    mantis_engine,
    WireAlreadyConsumed,
    pyo3::exceptions::PyException,
    "Raised when GraphWire.take() is called a second time (the single-read latch)."
);

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
            let mut in_flight = self
                .in_flight_graphs
                .lock()
                .expect("in_flight_graphs lock poisoned");
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
    #[new]
    #[pyo3(signature = (encoding_spec = None, feature_len = None, policy_len = None, pool_size = None))]
    pub fn new(
        encoding_spec: Option<PyRegistrySpec>,
        feature_len: Option<usize>,
        policy_len: Option<usize>,
        pool_size: Option<usize>,
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
            GraphQueue::with_contract_version(contract_version),
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
                if dense.submit_batch_and_wait(vec![vec![0.0f32; feature_len]]).is_ok() {
                    completed.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
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
    pub fn spawn_mock_graph_games(&self, n_games: usize) -> PyResult<()> {
        self.require_graph()?;
        let (win_length, radius, trunk_size) =
            (self.graph_win_length, self.graph_radius, self.graph_trunk_size);
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
                if let Ok(graph) =
                    build_leaf_graph(&stones, 1, 2, win_length, radius, trunk_size)
                {
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
            let mut in_flight = self
                .in_flight_graphs
                .lock()
                .expect("in_flight_graphs lock poisoned");
            for (id, graph) in pulled {
                // builder_impl handshake (defense-in-depth; the build path asserted
                // it) — a non-native tag must never reach the wire.
                if graph.builder_impl != BUILDER_IMPL_NATIVE {
                    return Err(PyValueError::new_err(
                        "next_graph_batch: non-native builder_impl on a queued graph",
                    ));
                }
                let legal_coords: Vec<(i32, i32)> = graph
                    .legal_node_gather
                    .iter()
                    .map(|&row| {
                        (
                            graph.node_coords[row as usize * 2],
                            graph.node_coords[row as usize * 2 + 1],
                        )
                    })
                    .collect();
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
        let mut wire = GraphWire::from_axis_graphs(&graphs, self.graph_contract_version);
        let arrays = wire.take().expect("a freshly fused wire always has arrays");
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

            let meta = {
                self.in_flight_graphs
                    .lock()
                    .expect("in_flight_graphs lock poisoned")
                    .remove(&id)
            };
            // Unknown id (already consumed / never emitted) — skip (frozen tolerant
            // remove semantics).
            let Some(meta) = meta else { continue };
            if meta.policy_dst_slot.len() != leaf_probs.len() {
                let msg = format!(
                    "submit_graph_inference_results: segment len {} != n_legal {} for id {id}",
                    leaf_probs.len(),
                    meta.policy_dst_slot.len()
                );
                self.graph.submit_graph_results(&[id], vec![Err(msg.clone())]);
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
                    self.graph.submit_graph_results(&[id], vec![Ok((ls, vals[i]))]);
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
    #[allow(clippy::type_complexity)]
    pub fn submit_graphs_and_wait(
        &self,
        py: Python<'_>,
        positions: Vec<(Vec<(i64, i64, i64)>, i64, i64)>,
    ) -> PyResult<Vec<(Vec<f32>, Vec<((i32, i32), f32)>, f32)>> {
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
        let graph_q = self.graph.clone();
        let results: Result<Vec<(LegalSetPolicy, f32)>, String> = py.detach(|| {
            let mut out = Vec::with_capacity(graphs.len());
            for g in graphs {
                out.push(graph_q.submit_graph_and_wait(g)?);
            }
            Ok(out)
        });
        let results = results.map_err(PyValueError::new_err)?;
        let out = results
            .into_iter()
            .map(|(ls, v)| {
                let overflow: Vec<((i32, i32), f32)> =
                    ls.overflow.iter().map(|(&k, &p)| (k, p)).collect();
                (ls.dense, overflow, v)
            })
            .collect();
        Ok(out)
    }
}

/// Block-diagonal ragged graph wire — the fuse-out of `next_graph_batch` /
/// `HexgBuffer.sample_graph_batch`. Owns the moved-out `GraphWireArrays`; the 13
/// per-array getters COPY into a fresh numpy array (repeatable, old behaviour
/// PRESERVED). The Python `take()` is a single-read latch over these arrays.
#[pyclass(name = "GraphWire", module = "mantis._engine")]
pub struct PyGraphWire {
    arrays: GraphWireArrays,
    taken: AtomicBool,
}

impl PyGraphWire {
    /// Wrap the arrays a caller already moved out of the WP6 `GraphWire` (via one
    /// internal Rust `take()` at fuse time).
    pub(crate) fn from_arrays(arrays: GraphWireArrays) -> Self {
        PyGraphWire {
            arrays,
            taken: AtomicBool::new(false),
        }
    }
}

#[pymethods]
impl PyGraphWire {
    #[getter]
    fn contract_version(&self) -> u32 {
        self.arrays.contract_version
    }
    #[getter]
    fn builder_impl(&self) -> u8 {
        self.arrays.builder_impl
    }
    #[getter]
    fn n_graphs(&self) -> usize {
        self.arrays.n_graphs
    }
    #[getter]
    fn node_feat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.arrays.node_feat)
    }
    #[getter]
    fn node_coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        PyArray1::from_slice(py, &self.arrays.node_coords)
    }
    #[getter]
    fn edge_index<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.arrays.edge_index)
    }
    #[getter]
    fn edge_attr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.arrays.edge_attr)
    }
    #[getter]
    fn node_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.arrays.node_offsets)
    }
    #[getter]
    fn edge_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.arrays.edge_offsets)
    }
    #[getter]
    fn legal_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.arrays.legal_offsets)
    }
    #[getter]
    fn legal_node_gather<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.arrays.legal_node_gather)
    }
    #[getter]
    fn policy_dst_slot<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        PyArray1::from_slice(py, &self.arrays.policy_dst_slot)
    }
    #[getter]
    fn n_nodes_checksum<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.arrays.n_nodes_checksum)
    }
    #[getter]
    fn n_stones<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        PyArray1::from_slice(py, &self.arrays.n_stones)
    }
    #[getter]
    fn window_center<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        PyArray1::from_slice(py, &self.arrays.window_center)
    }
    #[getter]
    fn current_player<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        PyArray1::from_slice(py, &self.arrays.current_player)
    }

    /// Single-read latch: yields all wire fields once as a dict; a second call
    /// raises `WireAlreadyConsumed` (the WP6 wire single-read guarantee, enforced
    /// at the Python boundary). The per-array getters above remain repeatable.
    fn take<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self
            .taken
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(WireAlreadyConsumed::new_err(
                "GraphWire arrays already consumed (single-read take() called twice)",
            ));
        }
        let a = &self.arrays;
        let d = PyDict::new(py);
        d.set_item("contract_version", a.contract_version)?;
        d.set_item("builder_impl", a.builder_impl)?;
        d.set_item("n_graphs", a.n_graphs)?;
        d.set_item("node_feat", PyArray1::from_slice(py, &a.node_feat))?;
        d.set_item("node_coords", PyArray1::from_slice(py, &a.node_coords))?;
        d.set_item("edge_index", PyArray1::from_slice(py, &a.edge_index))?;
        d.set_item("edge_attr", PyArray1::from_slice(py, &a.edge_attr))?;
        d.set_item("node_offsets", PyArray1::from_slice(py, &a.node_offsets))?;
        d.set_item("edge_offsets", PyArray1::from_slice(py, &a.edge_offsets))?;
        d.set_item("legal_offsets", PyArray1::from_slice(py, &a.legal_offsets))?;
        d.set_item("legal_node_gather", PyArray1::from_slice(py, &a.legal_node_gather))?;
        d.set_item("policy_dst_slot", PyArray1::from_slice(py, &a.policy_dst_slot))?;
        d.set_item("n_nodes_checksum", PyArray1::from_slice(py, &a.n_nodes_checksum))?;
        d.set_item("n_stones", PyArray1::from_slice(py, &a.n_stones))?;
        d.set_item("window_center", PyArray1::from_slice(py, &a.window_center))?;
        d.set_item("current_player", PyArray1::from_slice(py, &a.current_player))?;
        Ok(d)
    }
}

/// Register the `InferenceBatcher` + `GraphWire` pyclasses and the
/// `WireAlreadyConsumed` exception into `_engine`. Called by Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInferenceBatcher>()?;
    m.add_class::<PyGraphWire>()?;
    m.add("WireAlreadyConsumed", m.py().get_type::<WireAlreadyConsumed>())?;
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

    #[test]
    fn grid_batcher_derives_shapes_and_is_grid() {
        let b = PyInferenceBatcher::new(Some(PyRegistrySpec::from_static(v6_spec())), None, None, None)
            .expect("v6 batcher constructs");
        assert!(!b.is_graph);
        assert_eq!(b.representation, "grid");
        assert_eq!(b.feature_len, v6_spec().state_stride());
        assert_eq!(b.policy_len, v6_spec().policy_stride());
    }

    #[test]
    fn explicit_lens_without_spec_construct() {
        let b = PyInferenceBatcher::new(None, Some(2888), Some(362), None)
            .expect("explicit lens construct");
        assert_eq!(b.feature_len, 2888);
        assert_eq!(b.policy_len, 362);
    }

    #[test]
    fn no_spec_no_lens_errors() {
        assert!(PyInferenceBatcher::new(None, None, None, None).is_err());
        assert!(PyInferenceBatcher::new(None, Some(2888), None, None).is_err());
    }

    #[test]
    fn graph_batcher_reads_graph_params() {
        let b = PyInferenceBatcher::new(Some(PyRegistrySpec::from_static(gnn_spec())), None, None, None)
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
        let b = PyInferenceBatcher::new(None, Some(8), Some(4), None).unwrap();
        assert_eq!(b.model_version(), 0);
        assert_eq!(b.bump_model_version(), 1);
        assert_eq!(b.bump_model_version(), 2);
        assert_eq!(b.model_version(), 2);
    }

    #[test]
    fn grid_batcher_rejects_graph_seam_methods() {
        let b = PyInferenceBatcher::new(None, Some(8), Some(4), None).unwrap();
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
        let stones: Vec<(i64, i64, i64)> =
            (0..5i64).map(|q| (q, 0, 1)).chain((30..35i64).map(|q| (q, 0, -1))).collect();
        let graph = build_leaf_graph(&stones, 1, 2, 6, 6, 19).expect("valid graph");
        let mut wire = GraphWire::from_axis_graphs(&[graph], 1);
        let arrays = wire.take().expect("first fuse take");
        let gw = PyGraphWire::from_arrays(arrays);

        // Repeatable scalar getter serves regardless of the latch (numpy-free).
        assert_eq!(gw.n_graphs(), 1);
        // The single-read latch: acquire once, second acquisition fails.
        assert!(
            gw.taken.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok(),
            "a fresh wire's latch is acquirable"
        );
        assert!(
            gw.taken.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err(),
            "the latch is single-read"
        );
        // take() on an already-consumed wire maps to WireAlreadyConsumed.
        Python::initialize();
        Python::attach(|py| {
            assert!(gw.take(py).is_err(), "a consumed wire's take() raises WireAlreadyConsumed");
        });
    }
}
