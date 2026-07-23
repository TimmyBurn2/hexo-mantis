// Exceeds the 300-line soft cap (R8): the full `ReplayBuffer` PyO3 facade — the
// three strided push marshalers (numpy views → the WP5 push-config slices) and
// the two sample marshalers (the WP5 `SampleBatch`/`SampleBatchWithPos` structs →
// 8-/9-tuple numpy, incl. the frozen f16-bits reinterpret) — ports as one
// line-auditable unit against the frozen `replay_buffer/mod.rs` facade.
//! `ReplayBuffer` pyclass over `mantis_selfplay::replay::ReplayBuffer` (HEXB,
//! WP5). The WP5 buffer is pyo3-STRIPPED: its push path takes plain borrowed
//! slices (the `push_config` view structs) and its sample path returns owned-Rust
//! `SampleBatch` / `SampleBatchWithPos` structs. This bridge marshals numpy views
//! IN and `into_pyarray` OUT, field-by-field. `Send + Sync` (default derive).
//! F-42: `module = "mantis._engine"`.

use half::f16;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_selfplay::replay::push_config::{PushGameConfig, PushManyConfig, PushSingleConfig};
use mantis_selfplay::replay::{ReplayBuffer, SampleBatch, SampleBatchWithPos};

use crate::encoding::PyRegistrySpec;

/// 8-tuple sample return: `(states f16, chain f16, policies f32, outcomes f32,
/// ownership u8, winning_line u8, is_full_search u8, value_target_valid u8)`.
type SampleBatchOut<'py> = (
    Bound<'py, PyArray4<f16>>,
    Bound<'py, PyArray4<f16>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
);

/// 9-tuple sample return — the 8-tuple with `position_indices` (u16) spliced at
/// index 7 before `value_target_valid`.
type SampleBatchWithPosOut<'py> = (
    Bound<'py, PyArray4<f16>>,
    Bound<'py, PyArray4<f16>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<u8>>,
);

/// Reinterpret a `Vec<u16>` (f16 bits) as `Vec<f16>` for numpy `float16` return
/// (frozen zero-copy reinterpret; the marshaling-bench copy avoided).
///
/// Safety: `f16` and `u16` have identical size/alignment; every `u16` bit pattern
/// is a valid `f16`, and only bits from valid f16 values (stored via `push`) are
/// held. The source `Vec` is leaked into the new one via `ManuallyDrop`.
fn u16_bits_to_f16(v: Vec<u16>) -> Vec<f16> {
    let mut v = std::mem::ManuallyDrop::new(v);
    unsafe { Vec::from_raw_parts(v.as_mut_ptr().cast::<f16>(), v.len(), v.capacity()) }
}

/// Ring-buffer replay buffer with 12-fold hexagonal augmentation, exposed to
/// Python.
#[pyclass(name = "ReplayBuffer", module = "mantis._engine")]
pub struct PyReplayBuffer {
    inner: ReplayBuffer,
}

#[pymethods]
impl PyReplayBuffer {
    /// Create a buffer with `capacity` positions. `encoding` is a registry name
    /// (default `"v6"`, frozen back-compat); an unknown name panics through
    /// `lookup_or_panic` (→ catchable `PanicException`, LOCKED #4).
    #[new]
    #[pyo3(signature = (capacity, encoding = "v6"))]
    pub fn new(capacity: usize, encoding: &str) -> Self {
        PyReplayBuffer {
            inner: ReplayBuffer::new(capacity, encoding),
        }
    }

    /// `(size, capacity, weight_histogram)` for dashboard display.
    pub fn get_buffer_stats(&self) -> (usize, usize, Vec<u64>) {
        self.inner.get_buffer_stats()
    }

    /// Fresh monotonic position id.
    pub fn next_game_id(&mut self) -> i64 {
        self.inner.next_game_id()
    }

    /// Store a single position.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (state, chain_planes, policy, outcome, ownership, winning_line, game_id = -1, game_length = 0, is_full_search = true, position_index = 0, value_target_valid = true))]
    pub fn push(
        &mut self,
        state: PyReadonlyArray3<f16>,
        chain_planes: PyReadonlyArray3<f16>,
        policy: PyReadonlyArray1<f32>,
        outcome: f32,
        ownership: PyReadonlyArray1<u8>,
        winning_line: PyReadonlyArray1<u8>,
        game_id: i64,
        game_length: u16,
        is_full_search: bool,
        position_index: u16,
        value_target_valid: bool,
    ) -> PyResult<()> {
        self.inner
            .push_impl(PushSingleConfig {
                state: state.as_slice()?,
                chain_planes: chain_planes.as_slice()?,
                policy: policy.as_slice()?,
                outcome,
                ownership: ownership.as_slice()?,
                winning_line: winning_line.as_slice()?,
                game_id,
                game_length,
                is_full_search,
                position_index,
                value_target_valid,
            })
            .map_err(PyValueError::new_err)
    }

    /// Store all positions from a completed game (shared scalar metadata + optional
    /// per-row `is_full_search` / `position_indices` / `value_target_valid`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (states, chain_planes, policies, outcomes, ownership, winning_line, game_id = -1, game_length = 0, is_full_search = None, position_indices = None, value_target_valid = None))]
    pub fn push_game(
        &mut self,
        states: PyReadonlyArray4<f16>,
        chain_planes: PyReadonlyArray4<f16>,
        policies: PyReadonlyArray2<f32>,
        outcomes: PyReadonlyArray1<f32>,
        ownership: PyReadonlyArray2<u8>,
        winning_line: PyReadonlyArray2<u8>,
        game_id: i64,
        game_length: u16,
        is_full_search: Option<PyReadonlyArray1<u8>>,
        position_indices: Option<PyReadonlyArray1<u16>>,
        value_target_valid: Option<PyReadonlyArray1<u8>>,
    ) -> PyResult<()> {
        let ifs = is_full_search.as_ref().map(|a| a.as_slice()).transpose()?;
        let pidx = position_indices.as_ref().map(|a| a.as_slice()).transpose()?;
        let vtv = value_target_valid.as_ref().map(|a| a.as_slice()).transpose()?;
        self.inner
            .push_game_impl(PushGameConfig {
                states: states.as_slice()?,
                chain_planes: chain_planes.as_slice()?,
                policies: policies.as_slice()?,
                outcomes: outcomes.as_slice()?,
                ownership: ownership.as_slice()?,
                winning_line: winning_line.as_slice()?,
                game_id,
                game_length,
                is_full_search: ifs,
                position_indices: pidx,
                value_target_valid: vtv,
            })
            .map_err(PyValueError::new_err)
    }

    /// Store N per-row positions in one call (all untagged, `game_id = -1`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (states, chain_planes, policies, outcomes, ownership, winning_line, game_lengths, is_full_search, position_indices = None, value_target_valid = None))]
    pub fn push_many(
        &mut self,
        states: PyReadonlyArray4<f16>,
        chain_planes: PyReadonlyArray4<f16>,
        policies: PyReadonlyArray2<f32>,
        outcomes: PyReadonlyArray1<f32>,
        ownership: PyReadonlyArray2<u8>,
        winning_line: PyReadonlyArray2<u8>,
        game_lengths: PyReadonlyArray1<u16>,
        is_full_search: PyReadonlyArray1<u8>,
        position_indices: Option<PyReadonlyArray1<u16>>,
        value_target_valid: Option<PyReadonlyArray1<u8>>,
    ) -> PyResult<()> {
        let pidx = position_indices.as_ref().map(|a| a.as_slice()).transpose()?;
        let vtv = value_target_valid.as_ref().map(|a| a.as_slice()).transpose()?;
        self.inner
            .push_many_impl(PushManyConfig {
                states: states.as_slice()?,
                chain_planes: chain_planes.as_slice()?,
                policies: policies.as_slice()?,
                outcomes: outcomes.as_slice()?,
                ownership: ownership.as_slice()?,
                winning_line: winning_line.as_slice()?,
                game_lengths: game_lengths.as_slice()?,
                is_full_search: is_full_search.as_slice()?,
                position_indices: pidx,
                value_target_valid: vtv,
            })
            .map_err(PyValueError::new_err)
    }

    /// Sample `batch_size` entries (optional 12-fold hex augmentation) → 8-tuple.
    pub fn sample_batch<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize,
        augment: bool,
    ) -> PyResult<SampleBatchOut<'py>> {
        let SampleBatch {
            states,
            chain,
            policies,
            outcomes,
            ownership,
            winning_line,
            is_full_search,
            value_target_valid,
            batch_size: b,
        } = self
            .inner
            .sample_batch_core(batch_size, augment)
            .map_err(PyValueError::new_err)?;

        let (n_planes, n_chain, trunk, n_logits) = self.shape_params();
        let states_np = u16_bits_to_f16(states)
            .into_pyarray(py)
            .reshape([b, n_planes, trunk, trunk])?;
        let chain_np = u16_bits_to_f16(chain)
            .into_pyarray(py)
            .reshape([b, n_chain, trunk, trunk])?;
        let policies_np = policies.into_pyarray(py).reshape([b, n_logits])?;
        let outcomes_np = outcomes.into_pyarray(py);
        let ownership_np = ownership.into_pyarray(py).reshape([b, trunk, trunk])?;
        let winning_line_np = winning_line.into_pyarray(py).reshape([b, trunk, trunk])?;
        let is_full_search_np = is_full_search.into_pyarray(py);
        let value_valid_np = value_target_valid.into_pyarray(py);

        Ok((
            states_np,
            chain_np,
            policies_np,
            outcomes_np,
            ownership_np,
            winning_line_np,
            is_full_search_np,
            value_valid_np,
        ))
    }

    /// `sample_batch` extended with per-row `position_indices` (u16) → 9-tuple.
    pub fn sample_batch_with_pos<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize,
        augment: bool,
    ) -> PyResult<SampleBatchWithPosOut<'py>> {
        let SampleBatchWithPos {
            states,
            chain,
            policies,
            outcomes,
            ownership,
            winning_line,
            is_full_search,
            position_indices,
            value_target_valid,
            batch_size: b,
        } = self
            .inner
            .sample_batch_with_pos_core(batch_size, augment)
            .map_err(PyValueError::new_err)?;

        let (n_planes, n_chain, trunk, n_logits) = self.shape_params();
        let states_np = u16_bits_to_f16(states)
            .into_pyarray(py)
            .reshape([b, n_planes, trunk, trunk])?;
        let chain_np = u16_bits_to_f16(chain)
            .into_pyarray(py)
            .reshape([b, n_chain, trunk, trunk])?;
        let policies_np = policies.into_pyarray(py).reshape([b, n_logits])?;
        let outcomes_np = outcomes.into_pyarray(py);
        let ownership_np = ownership.into_pyarray(py).reshape([b, trunk, trunk])?;
        let winning_line_np = winning_line.into_pyarray(py).reshape([b, trunk, trunk])?;
        let is_full_search_np = is_full_search.into_pyarray(py);
        let position_indices_np = position_indices.into_pyarray(py);
        let value_valid_np = value_target_valid.into_pyarray(py);

        Ok((
            states_np,
            chain_np,
            policies_np,
            outcomes_np,
            ownership_np,
            winning_line_np,
            is_full_search_np,
            position_indices_np,
            value_valid_np,
        ))
    }

    /// Grow the buffer to `new_capacity`, preserving all data.
    pub fn resize(&mut self, new_capacity: usize) -> PyResult<()> {
        self.inner.resize(new_capacity).map_err(PyValueError::new_err)
    }

    /// Count valid outcomes in the half-open interval `[lo, hi)` (live prefix
    /// only). Feeds `buffer_composition()`'s `draw_target_fraction`.
    pub fn outcome_in_range_count(&self, lo: f32, hi: f32) -> usize {
        self.inner.outcome_in_range_count(lo, hi)
    }

    /// Set the game-length weight schedule.
    pub fn set_weight_schedule(
        &mut self,
        thresholds: Vec<u16>,
        weights: Vec<f32>,
        default_weight: f32,
    ) -> PyResult<()> {
        self.inner
            .set_weight_schedule(thresholds, weights, default_weight)
            .map_err(PyValueError::new_err)
    }

    /// Save buffer contents to a binary file (HEXB on-disk format).
    pub fn save_to_path(&self, path: &str) -> PyResult<()> {
        self.inner.save_to_path(path).map_err(PyValueError::new_err)
    }

    /// Load buffer contents written by `save_to_path`; returns the number loaded.
    pub fn load_from_path(&mut self, path: &str) -> PyResult<usize> {
        self.inner.load_from_path(path).map_err(PyValueError::new_err)
    }

    #[getter]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// The encoding spec driving this buffer's geometry.
    #[getter]
    pub fn encoding(&self) -> PyRegistrySpec {
        PyRegistrySpec::from_static(self.inner.encoding)
    }
}

impl PyReplayBuffer {
    /// Spec-derived `(n_planes, n_chain_planes, trunk_size, policy_logit_count)`
    /// for the sample-return reshapes.
    fn shape_params(&self) -> (usize, usize, usize, usize) {
        let spec = self.inner.encoding;
        (
            spec.n_planes,
            spec.n_chain_planes,
            spec.trunk_size,
            spec.policy_logit_count,
        )
    }
}

/// Register the `ReplayBuffer` pyclass into `_engine`. Called by Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReplayBuffer>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_scalar_getters() {
        let b = PyReplayBuffer::new(16, "v6");
        assert_eq!(b.size(), 0);
        assert_eq!(b.capacity(), 16);
        assert_eq!(b.encoding().name(), "v6");
    }

    #[test]
    fn next_game_id_advances() {
        let mut b = PyReplayBuffer::new(8, "v6");
        assert_eq!(b.next_game_id(), 0);
        assert_eq!(b.next_game_id(), 1);
    }

    /// Populate via the pure-Rust test helper — pins the buffer wiring (size grows
    /// through the shared `inner`). The sample-return numpy shape parity (states
    /// (B,8,19,19), policies (B,362), etc.) is pinned by the Python O-side tests
    /// post-ASM — the embedded cargo-test interpreter cannot load numpy's C-ext.
    #[test]
    fn push_for_test_grows_size() {
        let mut b = PyReplayBuffer::new(32, "v6");
        for _ in 0..8 {
            b.inner.push_for_test(0.5, 10, true);
        }
        assert_eq!(b.size(), 8);
    }

    /// Empty-buffer sample errors at the Rust core BEFORE any numpy is built (so
    /// this leg is numpy-free and exercises the error marshaling).
    #[test]
    fn empty_sample_errors() {
        Python::initialize();
        Python::attach(|py| {
            let mut b = PyReplayBuffer::new(8, "v6");
            assert!(b.sample_batch(py, 4, false).is_err(), "empty buffer sample errors");
        });
    }
}
