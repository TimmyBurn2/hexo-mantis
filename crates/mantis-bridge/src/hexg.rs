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
use mantis_selfplay::replay::hexg::{GraphRecord, GraphTargets, HexgBuffer};

use crate::inference::PyGraphWire;

/// Graph-position replay ring (parallel to `ReplayBuffer`), exposed to Python.
#[pyclass(name = "HexgBuffer", module = "mantis._engine")]
pub struct PyHexgBuffer {
    inner: HexgBuffer,
}

#[pymethods]
impl PyHexgBuffer {
    /// Create a graph-position ring with `capacity` records. `encoding` MUST be a
    /// graph spec and is REQUIRED — the `"gnn_axis_v1"` default was a silent-fallback arm
    /// (R45, LAW-11), and becomes actively wrong the moment a second graph schema exists
    /// (WP-AXIS2 adds `gnn_axis_v2`). A grid encoding is a LOUD `ValueError`; an unknown
    /// name panics through `lookup_or_panic` → `PanicException`.
    #[new]
    #[pyo3(signature = (capacity, encoding))]
    pub fn new(capacity: usize, encoding: &str) -> PyResult<Self> {
        Ok(PyHexgBuffer {
            inner: HexgBuffer::new(capacity, encoding).map_err(PyValueError::new_err)?,
        })
    }

    /// Store one compact graph-position record. LOUD error if it exceeds
    /// `MAX_STONES` / `MAX_VISITS`.
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
    #[pyo3(signature = (batch_size, augment = false, recent_frac = 0.0))]
    pub fn sample_graph_batch(
        &mut self,
        batch_size: usize,
        augment: bool,
        recent_frac: f32,
    ) -> PyResult<(PyGraphWire, PyGraphTargets)> {
        let (graphs, targets) = self
            .inner
            .sample_graph_batch_impl(batch_size, augment, recent_frac)
            .map_err(PyValueError::new_err)?;
        // Single-source block-diagonal fuse (shared with the inference seam so the
        // C3/C8 union arithmetic never drifts). For a single graph local == global.
        let mut wire = GraphWire::from_axis_graphs(&graphs, self.inner.contract_version);
        let arrays = wire.take().expect("a freshly fused wire always has arrays");
        Ok((PyGraphWire::from_arrays(arrays), PyGraphTargets { inner: targets }))
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_encoding_is_loud_error() {
        assert!(PyHexgBuffer::new(8, "v6").is_err(), "HexgBuffer rejects a grid encoding");
    }

    #[test]
    fn construct_and_getters() {
        let b = PyHexgBuffer::new(16, "gnn_axis_v1").expect("graph buffer constructs");
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
        let mut b = PyHexgBuffer::new(16, "gnn_axis_v1").unwrap();
        // A small in-window board (3 stones) with a 1-cell visit target.
        let stones = vec![(0i16, 0i16, 1i8), (1, 0, -1), (0, 1, 1)];
        let visits = vec![(2i16, 0i16, 1.0f32)];
        b.push_graph_position(stones, visits, 1, 2, 0, true, 1.0, true, 4, -1)
            .expect("push ok");
        assert_eq!(b.size(), 1);

        let (_wire, targets) = b.sample_graph_batch(1, false, 0.0).expect("sample ok");
        // One sampled record → one per-graph argmax cell decoded.
        assert_eq!(targets.target_argmax_cells().len(), 1);
    }
}
