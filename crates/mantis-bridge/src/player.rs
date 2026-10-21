//! The web player's Python surface: `web_batch`, the batch both net backends read, for the export's witness.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mantis_player::batch::{build_batch, Position};

/// Contract #13's batch of `positions` under `encoding`: `x` f32 flat, every other array int32, and `n_graphs`; `ValueError` on an unknown or refused row, a seam guard or a pack refusal.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn web_batch<'py>(
    py: Python<'py>,
    positions: Vec<Position>,
    encoding: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = mantis_encoding::lookup(encoding).ok_or_else(|| {
        PyValueError::new_err(format!("web_batch: unknown encoding {encoding:?}"))
    })?;
    let batch = py
        .detach(|| build_batch(&positions, spec))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("n_graphs", batch.n_graphs)?;
    out.set_item("x", batch.x.into_pyarray(py))?;
    for (name, array) in [
        ("src", batch.src),
        ("dst", batch.dst),
        ("code", batch.code),
        ("legal", batch.legal),
        ("node_graph", batch.node_graph),
        ("legal_graph", batch.legal_graph),
        ("divisor", batch.divisor),
        ("real", batch.real),
        ("mean_mask", batch.mean_mask),
        ("mean_den", batch.mean_den),
    ] {
        out.set_item(name, array.into_pyarray(py))?;
    }
    Ok(out)
}

/// Register `web_batch` into `_engine`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(web_batch, m)?)?;
    Ok(())
}
