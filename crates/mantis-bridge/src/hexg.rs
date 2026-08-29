// Exceeds the 300-line soft cap (R8): the HexgBuffer pyclass + its F-RT-2 typed
// push-refusal face + the in-src refusal oracle bank are one line-auditable unit.
//! `HexgBuffer` + `GraphTargets` pyclasses over `mantis_selfplay::replay::hexg`
//! (HEXG, WP5). The WP5 buffer is pyo3-STRIPPED: `push_record_impl` takes a plain
//! `GraphRecord`, and `sample_graph_batch_impl` returns the buffer-owned
//! `(Vec<AxisGraph>, GraphTargets)`. This bridge builds the record from Vec args
//! (no numpy) and fuses the sampled graphs into the `GraphWire` pyclass (via the
//! WP6 `from_axis_graphs`), pairing it with a `GraphTargets` pyclass whose getters
//! COPY out. `Send + Sync` (default derive). F-42: `module = "mantis._engine"`.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_selfplay::queues::GraphWire;
use mantis_selfplay::records::{TargetIntegrityError, TARGET_MASS_TOL};
use mantis_selfplay::replay::hexg::{
    derived_visit_capacity as derived_visit_capacity_impl, GraphRecord, GraphTargets, HexgBuffer,
};

use crate::inference::PyGraphWire;

/// WP12-R Phase T loop 2 (F-RT-2): `push_graph_position` is the SECOND public
/// graph-record constructor (the Python production route,
/// `pool_drain.py → pool_push.py`), and R161 unconstructibility is
/// constructor-quantified — so the FFI face refuses non-distribution rows with
/// the SAME typed semantics as `record_position_graph` (the
/// `TargetIntegrityError` Display, variant name first, mapped to `ValueError`;
/// `panic = "unwind"` untouched). Census grounds for refusing ALL
/// non-distribution rows here: the graph push face has NO legitimate
/// zero/value-only form — the fast-game zero-policy sentinel is the DENSE
/// recorder's (`runner/record.rs:67-78`; the graph dispatch takes no
/// `is_fast_game`), and graph quick-arm rows carry FULL mass (the frozen QA
/// oracle pins it). Duplicate-coord rows stay admitted (caught LOUD at
/// sample-align by `mass_drop_check`); per-entry NaN/negative/over-cap
/// refusals stay in `push_record_impl` (the independent second line).
/// `EmptyTarget.n_legal` carries the ROW length on this face (no board here).
fn refuse_non_distribution_row(
    visits: &[(i16, i16, f32)],
    ply_index: u16,
) -> Result<(), TargetIntegrityError> {
    let sum: f64 = visits.iter().map(|&(_, _, p)| f64::from(p)).sum();
    if !sum.is_finite() {
        return Err(TargetIntegrityError::MassNotUnity { sum, ply_index, n_cells: visits.len() });
    }
    if visits.is_empty() || sum.abs() <= TARGET_MASS_TOL {
        return Err(TargetIntegrityError::EmptyTarget { ply_index, n_legal: visits.len() });
    }
    if (sum - 1.0).abs() > TARGET_MASS_TOL {
        return Err(TargetIntegrityError::MassNotUnity { sum, ply_index, n_cells: visits.len() });
    }
    Ok(())
}

/// Graph-position replay ring (parallel to `ReplayBuffer`), exposed to Python.
#[pyclass(name = "HexgBuffer", module = "mantis._engine")]
pub struct PyHexgBuffer {
    inner: HexgBuffer,
}

#[pymethods]
impl PyHexgBuffer {
    /// Create a graph-position ring with `capacity` records and `visit_capacity`
    /// visit slots per record. `encoding` MUST be a graph spec and is REQUIRED — the
    /// `"gnn_axis_v1"` default was a silent-fallback arm (R45, LAW-11), and becomes
    /// actively wrong the moment a second graph schema exists (WP-AXIS2 adds
    /// `gnn_axis_v2`). `visit_capacity` is likewise REQUIRED with no default
    /// (R255/ADJ-D34): the production value is DERIVED from the sims regime via
    /// [`derived_hexg_visit_capacity`] at the composition site (`mantis.run`), never
    /// a literal. A grid encoding or an unstorable capacity is a LOUD `ValueError`;
    /// an unknown name panics through `lookup_or_panic` → `PanicException`.
    #[new]
    #[pyo3(signature = (capacity, encoding, visit_capacity))]
    pub fn new(capacity: usize, encoding: &str, visit_capacity: usize) -> PyResult<Self> {
        Ok(PyHexgBuffer {
            inner: HexgBuffer::new(capacity, encoding, visit_capacity)
                .map_err(PyValueError::new_err)?,
        })
    }

    /// Store one compact graph-position record. LOUD error if it exceeds
    /// `MAX_STONES` / the composed `visit_capacity`, and (F-RT-2) a typed refusal — the
    /// `TargetIntegrityError` Display as `ValueError` — when the visit row is
    /// not a distribution (empty / ~zero / off-unity mass): this ctor is the
    /// second public graph-record constructor and carries the same
    /// unconstructibility contract as `record_position_graph`.
    ///
    /// # Errors
    /// `ValueError` per the above; per-entry NaN/negative/over-cap refusals
    /// surface from `push_record_impl` unchanged.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (stones, visits, current_player, moves_remaining, ply_index, is_full_search, outcome, value_valid, game_length, game_id = -1))]
    pub fn push_graph_position(
        &mut self,
        stones: Vec<(i16, i16, i8)>,
        visits: Vec<(i16, i16, f32)>,
        current_player: i8,
        moves_remaining: u8,
        ply_index: u16,
        is_full_search: bool,
        outcome: f32,
        value_valid: bool,
        game_length: u16,
        game_id: i64,
    ) -> PyResult<()> {
        refuse_non_distribution_row(&visits, ply_index)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rec = GraphRecord {
            stones,
            visits,
            current_player,
            moves_remaining,
            ply_index,
            is_full_search,
            outcome,
            value_valid,
            game_length,
        };
        self.inner
            .push_record_impl(&rec, game_id)
            .map_err(PyValueError::new_err)
    }

    /// Sample `batch_size` records, rebuild + align each graph, and block-diagonal
    /// fuse them into `(GraphWire, GraphTargets)`. `recent_frac` (default 0.0)
    /// draws that fraction from the newest ring slots.
    ///
    /// GIL-FREE (B2). This call is the trainer's single longest — 1 386 ms of a 2 769 ms
    /// step, measured — and it is pure Rust CPU. Held under the GIL it stopped the
    /// in-process inference-server thread DEAD: the PERF-TRANCHE-1 M-1 joint drive measured
    /// the GIL unavailable for 99.9 % of every sample window and the server serving ZERO
    /// graphs across 16.85 s of them, against 79.9 requests/s outside. The twelve self-play
    /// workers keep building leaves through that; nothing answers them.
    ///
    /// Releasing it is sound because `&mut self` is exclusive: pyo3 hands out a `PyRefMut`,
    /// so a second Python thread touching this same buffer is refused by the borrow check
    /// rather than racing the released section. `HexgBuffer` holds no Python object.
    ///
    /// `n_threads` is the rebuild's width (B1). `1` is the serial path and the exact-parity
    /// control; the caller derives the budget from the run's own keys
    /// (`mantis.config.resolve.sample_threads`) rather than this layer inventing one, because
    /// the threads it may take are the ones the self-play workers are not already using.
    #[pyo3(signature = (batch_size, augment = false, recent_frac = 0.0, n_threads = 1))]
    pub fn sample_graph_batch(
        &mut self,
        py: Python<'_>,
        batch_size: usize,
        augment: bool,
        recent_frac: f32,
        n_threads: usize,
    ) -> PyResult<(PyGraphWire, PyGraphTargets)> {
        let inner = &mut self.inner;
        let sampled = py.detach(move || {
            let (graphs, targets) =
                inner.sample_graph_batch_impl(batch_size, augment, recent_frac, n_threads)?;
            // Single-source block-diagonal fuse (shared with the inference seam so the
            // C3/C8 union arithmetic never drifts). For a single graph local == global.
            let mut wire = GraphWire::from_axis_graphs(&graphs, inner.contract_version);
            let arrays = wire.take().expect("a freshly fused wire always has arrays");
            Ok::<_, String>((arrays, targets))
        });
        let (arrays, targets) = sampled.map_err(PyValueError::new_err)?;
        Ok((
            PyGraphWire::from_arrays(arrays),
            PyGraphTargets { inner: targets },
        ))
    }

    /// Grow to `new_capacity`, preserving all records.
    pub fn resize(&mut self, new_capacity: usize) -> PyResult<()> {
        self.inner.resize_impl(new_capacity).map_err(PyValueError::new_err)
    }

    /// Set the game-length weight schedule (identical semantics to `ReplayBuffer`).
    pub fn set_weight_schedule(
        &mut self,
        thresholds: Vec<u16>,
        weights: Vec<f32>,
        default_weight: f32,
    ) -> PyResult<()> {
        self.inner
            .set_weight_schedule_impl(thresholds, weights, default_weight)
            .map_err(PyValueError::new_err)
    }

    /// `(size, capacity, weight_histogram)` for dashboard display.
    pub fn get_buffer_stats(&self) -> (usize, usize, Vec<u64>) {
        self.inner.get_buffer_stats_impl()
    }

    /// Fresh monotonic game id.
    pub fn next_game_id(&mut self) -> i64 {
        self.inner.next_game_id()
    }

    /// Save records to a binary file (HEXG on-disk format).
    pub fn save_to_path(&self, path: &str) -> PyResult<()> {
        self.inner.save_to_path_impl(path).map_err(PyValueError::new_err)
    }

    /// Load records written by `save_to_path`; returns the number loaded.
    pub fn load_from_path(&mut self, path: &str) -> PyResult<usize> {
        self.inner.load_from_path_impl(path).map_err(PyValueError::new_err)
    }

    #[getter]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[getter]
    pub fn encoding_name(&self) -> &'static str {
        self.inner.encoding_name()
    }

    /// The DERIVED per-record visit-slot capacity this ring was composed with
    /// (R255/ADJ-D34) — read back by the composition pin so a literal on the
    /// compose path cannot survive unobserved.
    #[getter]
    pub fn visit_capacity(&self) -> usize {
        self.inner.visit_capacity
    }
}

/// R255/ADJ-D34 — the mint-side twin of the boot guard's capacity derivation.
///
/// Delegates VERBATIM to `mantis_selfplay::replay::hexg::derived_visit_capacity`
/// (one formula, two surfaces): returns the derived HEXG visit-slot capacity
/// `max(armed effective sim budgets) + leaf_batch_size − 1`, and raises
/// `ValueError` for a regime the record format cannot honor (the u16 count
/// ceiling; completed-Q below `MAX_CHILDREN_PER_NODE`). Live consumers: the
/// `RunConfig` schema validator (mint-time refusal) and `mantis.run`'s buffer
/// composition.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (n_simulations, standard_sims, fast_prob, fast_sims, full_search_prob, n_sims_quick, n_sims_full, leaf_batch_size, completed_q_values))]
pub fn derived_hexg_visit_capacity(
    n_simulations: usize,
    standard_sims: usize,
    fast_prob: f32,
    fast_sims: usize,
    full_search_prob: f32,
    n_sims_quick: usize,
    n_sims_full: usize,
    leaf_batch_size: usize,
    completed_q_values: bool,
) -> PyResult<usize> {
    derived_visit_capacity_impl(
        n_simulations,
        standard_sims,
        fast_prob,
        fast_sims,
        full_search_prob,
        n_sims_quick,
        n_sims_full,
        leaf_batch_size,
        completed_q_values,
    )
    .map_err(PyValueError::new_err)
}

/// Aligned training targets emitted alongside the `GraphWire` by
/// `sample_graph_batch`. The getters COPY out; `target_argmax_cells` decodes the
/// per-graph max-mass legal node (the AugRoundTrip runtime canary).
#[pyclass(name = "GraphTargets", module = "mantis._engine")]
pub struct PyGraphTargets {
    inner: GraphTargets,
}

#[pymethods]
impl PyGraphTargets {
    #[getter]
    fn policy_target<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.policy_target)
    }
    #[getter]
    fn outcomes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.outcomes)
    }
    #[getter]
    fn value_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        PyArray1::from_slice(py, &self.inner.value_valid)
    }
    #[getter]
    fn is_full_search<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        PyArray1::from_slice(py, &self.inner.is_full_search)
    }
    /// `[B]` list of `Optional[(q, r)]` — the collate `target_argmax_cells` arg.
    #[getter]
    fn target_argmax_cells(&self) -> Vec<Option<(i32, i32)>> {
        self.inner.target_argmax_cells()
    }
}

/// Register the `HexgBuffer` + `GraphTargets` pyclasses into `_engine`. Called by
/// Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHexgBuffer>()?;
    m.add_class::<PyGraphTargets>()?;
    m.add_function(wrap_pyfunction!(derived_hexg_visit_capacity, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_encoding_is_loud_error() {
        assert!(PyHexgBuffer::new(8, "v6", 128).is_err(), "HexgBuffer rejects a grid encoding");
    }

    #[test]
    fn construct_and_getters() {
        let b = PyHexgBuffer::new(16, "gnn_axis_v1", 128).expect("graph buffer constructs");
        assert_eq!(b.size(), 0);
        assert_eq!(b.capacity(), 16);
        assert_eq!(b.encoding_name(), "gnn_axis_v1");
    }

    /// Push a graph position, then sample_graph_batch rebuilds + fuses the graph
    /// into the `(GraphWire, GraphTargets)` pair. Numpy-free: the sample + fuse +
    /// pyclass construction are pure Rust; `target_argmax_cells` (a plain method)
    /// decodes one per-graph cell. The COPY numpy getters are pinned by the Python
    /// O-side tests post-ASM (the embedded interpreter cannot load numpy).
    #[test]
    fn push_then_sample_fuses_wire_and_targets() {
        let mut b = PyHexgBuffer::new(16, "gnn_axis_v1", 128).unwrap();
        // A small in-window board (3 stones) with a 1-cell visit target.
        let stones = vec![(0i16, 0i16, 1i8), (1, 0, -1), (0, 1, 1)];
        let visits = vec![(2i16, 0i16, 1.0f32)];
        b.push_graph_position(stones, visits, 1, 2, 0, true, 1.0, true, 4, -1)
            .expect("push ok");
        assert_eq!(b.size(), 1);

        // `sample_graph_batch` releases the GIL around the rebuild (B2), so it needs the
        // token; `n_threads = 1` is the serial path, which is what a one-record ring wants.
        Python::initialize();
        let targets = Python::attach(|py| {
            let (_wire, targets) = b.sample_graph_batch(py, 1, false, 0.0, 1).expect("sample ok");
            targets
        });
        // One sampled record → one per-graph argmax cell decoded.
        assert_eq!(targets.target_argmax_cells().len(), 1);
    }

    // ── WP12-R Phase T loop 2 (F-RT-2): the FFI ctor's non-distribution refusal,
    // Rust leg (the Python leg is tests/bridge/test_target_push_refusal.py).
    // Killer: M-Q (refusal removed → these red). ──────────────────────────────

    fn push_row(b: &mut PyHexgBuffer, visits: Vec<(i16, i16, f32)>) -> PyResult<()> {
        let stones = vec![(0i16, 0i16, 1i8), (1, 0, -1), (0, 1, 1)];
        b.push_graph_position(stones, visits, 1, 2, 3, true, 0.0, true, 4, -1)
    }

    #[test]
    fn push_refuses_non_distribution_rows_with_the_typed_message() {
        // Rendering a PyErr message needs the embedded interpreter (the
        // buffer.rs/mcts.rs in-src precedent).
        Python::initialize();
        Python::attach(|py| {
            let text = |e: PyErr| e.value(py).to_string();
            let mut b = PyHexgBuffer::new(8, "gnn_axis_v1", 128).unwrap();
            // Σ = 0.5 → MassNotUnity (variant name must lead the message).
            let e = text(push_row(&mut b, vec![(2, 0, 0.5)]).unwrap_err());
            assert!(e.contains("MassNotUnity") && e.contains("0.5"), "{e}");
            // Σ = 2.0 (over-unity) → MassNotUnity.
            let e = text(push_row(&mut b, vec![(2, 0, 1.5), (3, 0, 0.5)]).unwrap_err());
            assert!(e.contains("MassNotUnity"), "{e}");
            // Σ = 1.5 single positive entry (the F-RT-1 shipped quantity, at the
            // second constructor) → MassNotUnity.
            let e = text(push_row(&mut b, vec![(2, 0, 1.5)]).unwrap_err());
            assert!(e.contains("MassNotUnity") && e.contains("1.5"), "{e}");
            // all-zero row → EmptyTarget (no value-only form on the graph face).
            let e = text(push_row(&mut b, vec![(2, 0, 0.0), (3, 0, 0.0)]).unwrap_err());
            assert!(e.contains("EmptyTarget"), "{e}");
            // EMPTY visit list → EmptyTarget.
            let e = text(push_row(&mut b, vec![]).unwrap_err());
            assert!(e.contains("EmptyTarget"), "{e}");
            assert_eq!(b.size(), 0, "no refused row may reach the ring");
        });
    }

    #[test]
    fn push_admits_within_tol_and_exact_unity() {
        // F-RT-3 admit-side pin: the TOL window is the intended width — a
        // within-TOL row is ADMITTED (contract-conformant), unity likewise.
        let mut b = PyHexgBuffer::new(8, "gnn_axis_v1", 128).unwrap();
        push_row(&mut b, vec![(2, 0, 1.0)]).expect("exact unity admitted");
        push_row(&mut b, vec![(2, 0, 0.6), (3, 0, 0.4 + 5.0e-5)])
            .expect("a within-TOL (1 + 5e-5) row is ADMITTED — the documented window");
        assert_eq!(b.size(), 2);
    }
}
