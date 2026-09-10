// Exceeds the 300-line soft cap (R8): the HexgBuffer pyclass, its typed push-refusal face and
// the in-src refusal oracle bank are one line-auditable unit.
//! `HexgBuffer` + `GraphTargets` pyclasses over `mantis_selfplay::replay::hexg`. This bridge
//! builds the record from Vec args (no numpy) and fuses the sampled graphs into `GraphWire`,
//! paired with a `GraphTargets` whose getters COPY out.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_selfplay::queues::GraphWire;
use mantis_selfplay::records::{TargetIntegrityError, TARGET_MASS_TOL};
use mantis_selfplay::replay::hexg::{
    derived_visit_capacity as derived_visit_capacity_impl, GraphRecord, GraphTargets, HexgBuffer,
};

use crate::inference::{lock_or_recover, PyGraphWire, SeamFailure};

/// `push_graph_position` is the SECOND public graph-record constructor, so it refuses
/// non-distribution rows with the SAME typed semantics as `record_position_graph`: this face has
/// no legitimate zero/value-only form, the fast-game sentinel being the DENSE recorder's.
fn refuse_non_distribution_row(
    visits: &[(i16, i16, f32)],
    ply_index: u16,
    tail_mass: f32,
) -> Result<(), TargetIntegrityError> {
    // The row's distribution is its explicit entries PLUS the tail mass alpha; judging the
    // explicit half alone would refuse every Gumbel row this constructor exists to admit.
    let sum: f64 = visits.iter().map(|&(_, _, p)| f64::from(p)).sum::<f64>() + f64::from(tail_mass);
    if !sum.is_finite() {
        return Err(TargetIntegrityError::MassNotUnity {
            sum,
            ply_index,
            n_cells: visits.len(),
        });
    }
    if visits.is_empty() || sum.abs() <= TARGET_MASS_TOL {
        return Err(TargetIntegrityError::EmptyTarget {
            ply_index,
            n_legal: visits.len(),
        });
    }
    if (sum - 1.0).abs() > TARGET_MASS_TOL {
        return Err(TargetIntegrityError::MassNotUnity {
            sum,
            ply_index,
            n_cells: visits.len(),
        });
    }
    Ok(())
}

/// Graph-position replay ring (parallel to `ReplayBuffer`), exposed to Python.
#[pyclass(name = "HexgBuffer", module = "mantis._engine")]
pub struct PyHexgBuffer {
    /// INTERIOR MUTABILITY, not `&mut self`: a `&mut self` pymethod holds pyo3's `PyRefMut` for
    /// its whole body, including the ~1.4 s GIL-free sample, so a concurrent `.size` read was
    /// refused with `Already mutably borrowed` and killed the sole producer.
    inner: Mutex<HexgBuffer>,
    /// Times `inner` was found poisoned and recovered; non-zero means a panic happened under
    /// the guard on an earlier call. Read from Python via `lock_recoveries`.
    lock_recoveries: Arc<AtomicUsize>,
}

impl PyHexgBuffer {
    /// The ring, or a recovered guard over it. Never panics on a poisoned lock.
    fn ring(&self) -> MutexGuard<'_, HexgBuffer> {
        lock_or_recover(&self.inner, &self.lock_recoveries)
    }
}

#[pymethods]
impl PyHexgBuffer {
    /// Create a graph-position ring. `encoding` MUST be a graph spec and `visit_capacity` is
    /// likewise REQUIRED, DERIVED from the sims regime at the composition site: a default here
    /// was a silent-fallback arm.
    #[new]
    #[pyo3(signature = (capacity, encoding, visit_capacity))]
    pub fn new(capacity: usize, encoding: &str, visit_capacity: usize) -> PyResult<Self> {
        Ok(PyHexgBuffer {
            inner: Mutex::new(
                HexgBuffer::new(capacity, encoding, visit_capacity)
                    .map_err(PyValueError::new_err)?,
            ),
            lock_recoveries: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Store one compact graph-position record, refusing over-capacity rows loudly and
    /// non-distribution visit rows with the same typed contract as `record_position_graph`.
    ///
    /// # Errors
    /// `ValueError` per the above; per-entry refusals surface from `push_record_impl`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (stones, visits, current_player, moves_remaining, ply_index, is_full_search, outcome, value_valid, game_length, game_id = -1, tail_mass = 0.0))]
    pub fn push_graph_position(
        &self,
        py: Python<'_>,
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
        tail_mass: f32,
    ) -> PyResult<()> {
        refuse_non_distribution_row(&visits, ply_index, tail_mass)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rec = GraphRecord {
            stones,
            visits,
            tail_mass,
            current_player,
            moves_remaining,
            ply_index,
            is_full_search,
            outcome,
            value_valid,
            game_length,
            // The id travels as a separate argument to `push_record_impl`, the ring's own
            // authority over which slot it lands in.
            game_id: -1,
        };
        // GIL-FREE WAIT: this is the sole producer's write path and the trainer holds the ring
        // for a whole sample, so waiting under the GIL would re-stall the inference server.
        py.detach(|| self.ring().push_record_impl(&rec, game_id))
            .map_err(PyValueError::new_err)
    }

    /// Sample `batch_size` records, rebuild + align each graph, and block-diagonal fuse them into
    /// `(GraphWire, GraphTargets)`, drawing `recent_frac` from the newest slots.
    ///
    /// GIL-FREE: measured at 1386 ms of a 2769 ms step, and held under the GIL it left the
    /// in-process inference server serving ZERO graphs across 16.85 s of sample windows against
    /// 79.9 requests/s outside. The release is NOT sound by `&mut self` exclusivity — the
    /// `PyRefMut` spanning the window is what raised `Already mutably borrowed` — so exclusion is
    /// held by the `inner` mutex with every pymethod on `&self`. `n_threads` is the rebuild's
    /// width, `1` being the serial exact-parity path.
    #[pyo3(signature = (batch_size, augment = false, recent_frac = 0.0, n_threads = 1))]
    pub fn sample_graph_batch(
        &self,
        py: Python<'_>,
        batch_size: usize,
        augment: bool,
        recent_frac: f32,
        n_threads: usize,
    ) -> PyResult<(PyGraphWire, PyGraphTargets)> {
        let sampled = py.detach(|| {
            let inner = &mut *self.ring();
            let (graphs, targets) =
                inner.sample_graph_batch_impl(batch_size, augment, recent_frac, n_threads)?;
            // Single-source block-diagonal fuse, shared with the inference seam.
            let mut wire = GraphWire::from_axis_graphs(&graphs, inner.contract_version);
            let arrays = wire.take()?;
            Ok::<_, SeamFailure>((arrays, targets))
        });
        let (arrays, targets) = sampled.map_err(SeamFailure::into_pyerr)?;
        Ok((
            PyGraphWire::from_arrays(arrays),
            PyGraphTargets { inner: targets },
        ))
    }

    /// Grow to `new_capacity`, preserving all records.
    pub fn resize(&self, py: Python<'_>, new_capacity: usize) -> PyResult<()> {
        py.detach(|| self.ring().resize_impl(new_capacity))
            .map_err(PyValueError::new_err)
    }

    /// Set the game-length weight schedule (identical semantics to `ReplayBuffer`).
    pub fn set_weight_schedule(
        &self,
        py: Python<'_>,
        thresholds: Vec<u16>,
        weights: Vec<f32>,
        default_weight: f32,
    ) -> PyResult<()> {
        py.detach(|| {
            self.ring()
                .set_weight_schedule_impl(thresholds, weights, default_weight)
        })
        .map_err(PyValueError::new_err)
    }

    /// `(size, capacity, weight_histogram)` for dashboard display.
    pub fn get_buffer_stats(&self, py: Python<'_>) -> (usize, usize, Vec<u64>) {
        py.detach(|| self.ring().get_buffer_stats_impl())
    }

    /// Re-seed the sampler so this ring's batch stream is reproducible from a declared seed.
    /// Construction seeds from OS entropy; production calls this immediately after, from
    /// `config.seed`. It buys RUN-TO-RUN reproducibility, not stop/resume continuity.
    pub fn seed_sampler(&self, py: Python<'_>, seed: u64) {
        py.detach(|| self.ring().seed_sampler(seed));
    }

    /// Fresh monotonic game id.
    pub fn next_game_id(&self, py: Python<'_>) -> i64 {
        py.detach(|| self.ring().next_game_id())
    }

    /// Save records to a binary file (HEXG on-disk format).
    pub fn save_to_path(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.ring().save_to_path_impl(path))
            .map_err(PyValueError::new_err)
    }

    /// Load records written by `save_to_path`; returns the number loaded.
    pub fn load_from_path(&self, py: Python<'_>, path: &str) -> PyResult<usize> {
        py.detach(|| self.ring().load_from_path_impl(path))
            .map_err(PyValueError::new_err)
    }

    /// The getters detach too: they are cheap ONCE the lock is held, but acquiring it can wait
    /// out a whole sample, and waiting under the GIL is the stall this class exists to avoid.
    #[getter]
    pub fn size(&self, py: Python<'_>) -> usize {
        py.detach(|| self.ring().size())
    }

    #[getter]
    pub fn capacity(&self, py: Python<'_>) -> usize {
        py.detach(|| self.ring().capacity())
    }

    /// The LAST sampled batch's composition, as a plain dict: whether same-game dedupe is doing
    /// anything, how much of the batch it could not see, and how far back from the newest the
    /// rows came — a ring that has stopped being fed samples happily from older data and the loss
    /// curve never says so. All zeros before the first sample.
    pub fn last_batch_composition(&self, py: Python<'_>) -> std::collections::HashMap<String, u32> {
        let (distinct, max_rows, untagged, ages) = py.detach(|| {
            let r = self.ring();
            (
                r.last_batch_distinct_games,
                r.last_batch_max_rows_per_game,
                r.last_batch_untagged_rows,
                r.last_batch_age_quantiles,
            )
        });
        std::collections::HashMap::from([
            ("distinct_games".to_string(), distinct),
            ("max_rows_per_game".to_string(), max_rows),
            ("untagged_rows".to_string(), untagged),
            ("age_p50".to_string(), ages[0]),
            ("age_p90".to_string(), ages[1]),
            ("age_p99".to_string(), ages[2]),
        ])
    }

    /// The `game_id` stored in ring slot `index`, oldest-first (`-1` = untagged), since a
    /// raw-slot index would expose the ring's head rotation as if it were data.
    ///
    /// # Errors
    /// `IndexError` when `index >= size`.
    pub fn game_id_at(&self, py: Python<'_>, index: usize) -> PyResult<i64> {
        py.detach(|| self.ring().game_id_at(index)).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "game_id_at: index {index} is past the ring's size"
            ))
        })
    }

    #[getter]
    pub fn encoding_name(&self, py: Python<'_>) -> &'static str {
        py.detach(|| self.ring().encoding_name())
    }

    /// Poisoned-lock recoveries: non-zero means a panic happened under the ring guard and the
    /// seam kept going, so a run alerts on it instead of finding it post-mortem.
    #[getter]
    pub fn lock_recoveries(&self) -> usize {
        self.lock_recoveries.load(Ordering::SeqCst)
    }

    /// The DERIVED per-record visit-slot capacity this ring was composed with, read back by the
    /// composition pin so a literal on the compose path cannot survive unobserved.
    #[getter]
    pub fn visit_capacity(&self, py: Python<'_>) -> usize {
        py.detach(|| self.ring().visit_capacity)
    }
}

/// The mint-side twin of the boot guard's capacity derivation, delegating VERBATIM to one
/// formula across two surfaces and raising `ValueError` for a regime the record format cannot
/// honor. Live consumers: the schema validator and the buffer composition.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (n_simulations, standard_sims, fast_prob, fast_sims, full_search_prob, n_sims_quick, n_sims_full, leaf_batch_size, gumbel_m, search_kind))]
pub fn derived_hexg_visit_capacity(
    n_simulations: usize,
    standard_sims: usize,
    fast_prob: f32,
    fast_sims: usize,
    full_search_prob: f32,
    n_sims_quick: usize,
    n_sims_full: usize,
    leaf_batch_size: usize,
    gumbel_m: usize,
    search_kind: &str,
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
        gumbel_m,
        search_kind,
    )
    .map_err(PyValueError::new_err)
}

/// Aligned training targets emitted alongside the `GraphWire`. The getters COPY out;
/// `target_argmax_cells` decodes the per-graph max-mass legal node.
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
    /// Flat `[Lg]`, 1 where the row stored an EXPLICIT entry for that legal node; its per-graph
    /// complement is the remaining legal set the tail mass spreads over.
    #[getter]
    fn explicit_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        PyArray1::from_slice(py, &self.inner.explicit_mask)
    }
    /// R347(a) — `[B]` per-row tail mass alpha.
    #[getter]
    fn tail_mass<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.tail_mass)
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

/// Register the `HexgBuffer` + `GraphTargets` pyclasses into `_engine`.
/// The ring's per-record stone ceiling, read from the engine rather than transcribed: without it
/// a consumer had to type a `256`, a second authority over a fixed-width allocation.
#[pyfunction]
pub fn max_stones() -> usize {
    mantis_selfplay::replay::hexg::MAX_STONES
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHexgBuffer>()?;
    m.add_class::<PyGraphTargets>()?;
    m.add_function(wrap_pyfunction!(derived_hexg_visit_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(max_stones, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_encoding_is_loud_error() {
        // `"v6"` was a REGISTERED grid row before its deletion; the refusal now comes from the
        // registry, and the message assertion stops this reading as a bare typo check.
        let err = PyHexgBuffer::new(8, "v6", 128)
            .err()
            .expect("HexgBuffer rejects a grid encoding");
        Python::initialize();
        Python::attach(|py| {
            let msg = err.value(py).to_string();
            assert!(
                msg.contains("v6") && msg.contains("gnn_axis_v1"),
                "the refusal must name the offered encoding and the registered set: {msg}"
            );
        });
    }

    #[test]
    fn construct_and_getters() {
        Python::initialize();
        Python::attach(|py| {
            let b = PyHexgBuffer::new(16, "gnn_axis_v1", 128).expect("graph buffer constructs");
            assert_eq!(b.size(py), 0);
            assert_eq!(b.capacity(py), 16);
            assert_eq!(b.encoding_name(py), "gnn_axis_v1");
            assert_eq!(
                b.lock_recoveries(),
                0,
                "a fresh ring has never been poisoned"
            );
        });
    }

    /// Push a graph position, then sample_graph_batch rebuilds + fuses it into the
    /// `(GraphWire, GraphTargets)` pair. Numpy-free; the COPY numpy getters are pinned Python-side.
    #[test]
    fn push_then_sample_fuses_wire_and_targets() {
        let b = PyHexgBuffer::new(16, "gnn_axis_v1", 128).unwrap();
        // A small in-window board (3 stones) with a 1-cell visit target.
        let stones = vec![(0i16, 0i16, 1i8), (1, 0, -1), (0, 1, 1)];
        let visits = vec![(2i16, 0i16, 1.0f32)];
        // Each method detaches around its own ring acquisition, so a contender waits GIL-free.
        Python::initialize();
        let targets = Python::attach(|py| {
            b.push_graph_position(py, stones, visits, 1, 2, 0, true, 1.0, true, 4, -1, 0.0)
                .expect("push ok");
            assert_eq!(b.size(py), 1);
            // `n_threads = 1` is the serial path, which is what a one-record ring wants.
            let (_wire, targets) = b
                .sample_graph_batch(py, 1, false, 0.0, 1)
                .expect("sample ok");
            targets
        });
        // One sampled record → one per-graph argmax cell decoded.
        assert_eq!(targets.target_argmax_cells().len(), 1);
    }

    // The FFI ctor's non-distribution refusal, Rust leg.
    // Killer: M-Q (refusal removed -> these red).

    fn push_row(py: Python<'_>, b: &PyHexgBuffer, visits: Vec<(i16, i16, f32)>) -> PyResult<()> {
        let stones = vec![(0i16, 0i16, 1i8), (1, 0, -1), (0, 1, 1)];
        b.push_graph_position(py, stones, visits, 1, 2, 3, true, 0.0, true, 4, -1, 0.0)
    }

    #[test]
    fn push_refuses_non_distribution_rows_with_the_typed_message() {
        // Rendering a PyErr message needs the embedded interpreter.
        Python::initialize();
        Python::attach(|py| {
            let text = |e: PyErr| e.value(py).to_string();
            let b = PyHexgBuffer::new(8, "gnn_axis_v1", 128).unwrap();
            // Σ = 0.5 → MassNotUnity (variant name must lead the message).
            let e = text(push_row(py, &b, vec![(2, 0, 0.5)]).unwrap_err());
            assert!(e.contains("MassNotUnity") && e.contains("0.5"), "{e}");
            // Σ = 2.0 (over-unity) → MassNotUnity.
            let e = text(push_row(py, &b, vec![(2, 0, 1.5), (3, 0, 0.5)]).unwrap_err());
            assert!(e.contains("MassNotUnity"), "{e}");
            // Sigma = 1.5 single positive entry, at the second constructor -> MassNotUnity.
            let e = text(push_row(py, &b, vec![(2, 0, 1.5)]).unwrap_err());
            assert!(e.contains("MassNotUnity") && e.contains("1.5"), "{e}");
            // all-zero row → EmptyTarget (no value-only form on the graph face).
            let e = text(push_row(py, &b, vec![(2, 0, 0.0), (3, 0, 0.0)]).unwrap_err());
            assert!(e.contains("EmptyTarget"), "{e}");
            // EMPTY visit list → EmptyTarget.
            let e = text(push_row(py, &b, vec![]).unwrap_err());
            assert!(e.contains("EmptyTarget"), "{e}");
            assert_eq!(b.size(py), 0, "no refused row may reach the ring");
        });
    }

    #[test]
    fn push_admits_within_tol_and_exact_unity() {
        // Admit-side pin: the TOL window is the intended width — a within-TOL row is ADMITTED.
        Python::initialize();
        Python::attach(|py| {
            let b = PyHexgBuffer::new(8, "gnn_axis_v1", 128).unwrap();
            push_row(py, &b, vec![(2, 0, 1.0)]).expect("exact unity admitted");
            push_row(py, &b, vec![(2, 0, 0.6), (3, 0, 0.4 + 5.0e-5)])
                .expect("a within-TOL (1 + 5e-5) row is ADMITTED — the documented window");
            assert_eq!(b.size(py), 2);
        });
    }
}
