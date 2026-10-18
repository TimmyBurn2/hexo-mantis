//! `build_canvas`: one position's hex canvas (contract #12) for the conformance suite and tools; serving builds in Rust.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use mantis_graph::{CanvasParams, StoneList};
use mantis_selfplay::runner::params::resolve_canvas_geometry;

use crate::encoding::PyRegistrySpec;

/// The canvas of `stones` under a `hexcanvas` row as a dict of contract #12's fields (`planes` `[6, h, w]` u8 bytes); `ValueError`: not a canvas row.
#[pyfunction]
#[pyo3(name = "build_canvas")]
pub fn build_canvas_py<'py>(
    py: Python<'py>,
    encoding_spec: PyRegistrySpec,
    stones: Vec<(i32, i32, i8)>,
    current_player: i8,
    moves_remaining: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let params = resolve_canvas_geometry(encoding_spec.inner())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let c = mantis_graph::build_canvas(
        &StoneList { stones },
        &CanvasParams {
            current_player,
            moves_remaining,
            ..params
        },
    );
    let d = PyDict::new(py);
    d.set_item("axis_pair", c.axis_pair)?;
    d.set_item("height", c.height)?;
    d.set_item("width", c.width)?;
    d.set_item("origin", c.origin)?;
    d.set_item("far", c.far)?;
    d.set_item("fit_side", c.fit_side)?;
    d.set_item("planes", PyBytes::new(py, &c.planes))?;
    d.set_item("legal", c.legal)?;
    d.set_item("legal_gather", c.legal_gather)?;
    d.set_item("policy_dst_slot", c.policy_dst_slot)?;
    d.set_item("window_center", c.window_center)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_canvas_py, m)?)?;
    Ok(())
}
