//! mantis-bridge: ALL PyO3 lives here. Builds the `mantis._engine` extension module.
use pyo3::prelude::*;

/// Returns a static greeting proving the Rust->Python bridge is alive.
#[pyfunction]
fn hello() -> &'static str {
    "mantis._engine alive"
}

/// Names of the in-workspace crates linked into this module (DAG edge liveness).
#[pyfunction]
fn workspace_crates() -> Vec<&'static str> {
    vec![
        mantis_core::CRATE_NAME,
        mantis_graph::CRATE_NAME,
        mantis_encoding::CRATE_NAME,
        mantis_search::CRATE_NAME,
        mantis_selfplay::CRATE_NAME,
    ]
}

/// Compiled mantis engine bridge (PyO3). All Python-facing Rust lives here.
#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(workspace_crates, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn hello_static() {
        assert_eq!(super::hello(), "mantis._engine alive");
    }
    #[test]
    fn five_workspace_crates() {
        assert_eq!(super::workspace_crates().len(), 5);
    }
}
