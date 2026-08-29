//! DIAGNOSTIC readout for `mantis_selfplay::perf` (PERF-BASELINE, 2026-08-29).
//!
//! Two free fns so a Python driver can bracket a measurement window over the Rust-side
//! search-drive stage timers. Inert unless `MANTIS_PERF_STAGES=1` is set.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use mantis_selfplay::perf;

/// Snapshot the search-drive stage accumulators.
///
/// Returns `{"enabled": bool, "calls": int, "leaves": int,
/// "stages": {name: {"count", "total_ns", "max_ns"}}, "leaf_histogram": {bucket: count}}`.
/// Every stage row is present even at zero samples; `enabled` says whether a zero is a
/// measurement or an un-armed timer.
#[pyfunction]
pub(crate) fn selfplay_perf_snapshot(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let (stages, calls, leaves, hist) = perf::snapshot();
    let out = PyDict::new(py);
    out.set_item("enabled", perf::enabled())?;
    out.set_item("calls", calls)?;
    out.set_item("leaves", leaves)?;
    let sd = PyDict::new(py);
    for (name, count, total_ns, max_ns) in stages {
        let row = PyDict::new(py);
        row.set_item("count", count)?;
        row.set_item("total_ns", total_ns)?;
        row.set_item("max_ns", max_ns)?;
        sd.set_item(name, row)?;
    }
    out.set_item("stages", sd)?;
    let hd = PyDict::new(py);
    for (bucket, count) in hist {
        hd.set_item(bucket, count)?;
    }
    out.set_item("leaf_histogram", hd)?;
    Ok(out.into())
}

/// Zero every search-drive stage accumulator, discarding a warm-up window.
#[pyfunction]
pub(crate) fn selfplay_perf_reset() {
    perf::reset();
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(selfplay_perf_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay_perf_reset, m)?)?;
    Ok(())
}
