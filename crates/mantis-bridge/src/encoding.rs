//! Python-visible `RegistrySpec` — wraps `&'static mantis_encoding::RegistrySpec`
//! (`Copy`, read-only) — plus the NEW-BUILD module fns `all_specs()`,
//! `registry_sha()`, `registry_sha_hex()` over the existing Rust primitives.
//!
//! `all_specs()` returns the full spec set in ONE call (kills the Python shim's
//! per-name `_load` loop + `_REGISTERED_NAMES`); `registry_sha()`/`registry_sha_hex()`
//! feed the import-time on-disk-vs-compiled registry handshake. F-42: every
//! pyclass sets `module = "mantis._engine"` explicitly (PyO3's default is the
//! `'builtins'` wart).

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyType};

use mantis_encoding::{PolicyPool, RegistrySpec as RustRegistrySpec, ValuePool};

/// Python-visible RegistrySpec — wraps `&'static mantis_encoding::RegistrySpec`.
/// Returned by `RegistrySpec.from_registry(name)` and `all_specs()`. Carries
/// derived shape accessors (`state_stride()`, `policy_stride()`) so PyO3 callers
/// constructing `SelfPlayRunner` / `InferenceBatcher` can derive
/// `feature_len` / `policy_len` from the canonical registry instead of
/// duplicating the per-encoding shape table.
///
/// Read-only — clone is `Copy` (just the &'static pointer). `from_py_object`
/// opts into pyo3 0.28's `FromPyObject` derivation for a `Clone` pyclass.
#[pyclass(name = "RegistrySpec", module = "mantis._engine", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyRegistrySpec {
    inner: &'static RustRegistrySpec,
}

#[pymethods]
impl PyRegistrySpec {
    #[getter] pub fn name(&self) -> &'static str { self.inner.name }
    #[getter] pub fn board_size(&self) -> usize { self.inner.board_size }
    #[getter] pub fn trunk_size(&self) -> usize { self.inner.trunk_size }
    #[getter] pub fn cluster_window_size(&self) -> Option<usize> { self.inner.cluster_window_size }
    #[getter] pub fn cluster_threshold(&self) -> Option<usize> { self.inner.cluster_threshold }
    #[getter] pub fn legal_move_radius(&self) -> usize { self.inner.legal_move_radius }
    #[getter] pub fn n_planes(&self) -> usize { self.inner.n_planes }
    #[getter] pub fn plane_layout(&self) -> Vec<&'static str> { self.inner.plane_layout.to_vec() }
    #[getter] pub fn policy_logit_count(&self) -> usize { self.inner.policy_logit_count }
    #[getter] pub fn has_pass_slot(&self) -> bool { self.inner.has_pass_slot }
    #[getter] pub fn is_multi_window(&self) -> bool { self.inner.is_multi_window }
    /// Wire-format pool enums exposed as strings (matches the Python `Literal`
    /// shape returned by the retired @dataclass `value_pool` / `policy_pool` fields).
    #[getter] pub fn value_pool(&self) -> &'static str {
        match self.inner.value_pool {
            ValuePool::None => "none",
            ValuePool::Min => "min",
            ValuePool::Max => "max",
            ValuePool::Mean => "mean",
        }
    }
    #[getter] pub fn policy_pool(&self) -> &'static str {
        match self.inner.policy_pool {
            PolicyPool::None => "none",
            PolicyPool::ScatterMax => "scatter_max",
            PolicyPool::ScatterMean => "scatter_mean",
            PolicyPool::LegalSetScatterMax => "legal_set_scatter_max",
        }
    }
    #[getter] pub fn sym_table_id(&self) -> &'static str { self.inner.sym_table_id }
    #[getter] pub fn schema_version(&self) -> u32 { self.inner.schema_version }
    #[getter] pub fn notes(&self) -> &'static str { self.inner.notes }
    /// Physical source-plane indices retained by wire format.
    #[getter] pub fn kept_plane_indices(&self) -> Vec<usize> {
        self.inner.kept_plane_indices.to_vec()
    }
    /// Source tensor plane count before the `kept_plane_indices` slice.
    #[getter] pub fn n_source_planes(&self) -> usize { self.inner.n_source_planes }
    /// Multi-window cluster-count upper bound per position emitted by
    /// `Board::get_cluster_views()`. = 1 for single-window encodings.
    #[getter] pub fn k_max(&self) -> u32 { self.inner.k_max }

    // ── GNN-integration schema — representation discriminant + graph geom.
    /// "grid" (dense CNN planes) | "graph" (axis-graph GNN). Grid for every
    /// pre-graph encoding.
    #[getter] pub fn representation(&self) -> &'static str { self.inner.representation.as_str() }
    /// Convenience mirror of `representation == "graph"`.
    #[getter] pub fn is_graph(&self) -> bool { self.inner.is_graph() }
    /// Per-node feature width (graph only; `None` for grid). = 11.
    #[getter] pub fn node_feat_dim(&self) -> Option<usize> { self.inner.node_feat_dim }
    /// Per-edge feature width (graph only). = 5.
    #[getter] pub fn edge_feat_dim(&self) -> Option<usize> { self.inner.edge_feat_dim }
    /// GNN win-length (graph only). = 6.
    #[getter] pub fn win_length(&self) -> Option<usize> { self.inner.win_length }
    /// GNN legal-move / axis-walk radius (graph only). = 6.
    #[getter] pub fn graph_radius(&self) -> Option<usize> { self.inner.graph_radius }
    /// Number of win axes (graph only). = 3.
    #[getter] pub fn win_axes(&self) -> Option<usize> { self.inner.win_axes }
    /// Ragged-payload contract version this encoding speaks (graph only). = 1.
    #[getter] pub fn contract_version(&self) -> Option<u32> { self.inner.contract_version }
    /// Required native builder tag the resolver asserts (graph only). = 1.
    #[getter] pub fn builder_impl_required(&self) -> Option<u8> { self.inner.builder_impl_required }

    /// Alias for `policy_logit_count` — matches the retired Python @dataclass
    /// `n_actions` @property.
    #[getter] pub fn n_actions(&self) -> usize { self.inner.policy_logit_count }
    /// Cells per trunk input tensor = trunk_size². Semantic: trunk_size, not board_size.
    #[getter] pub fn n_cells(&self) -> usize { self.inner.n_cells() }
    /// State plane stride = n_planes × n_cells.
    #[getter] pub fn state_stride(&self) -> usize { self.inner.state_stride() }
    /// Chain plane stride = n_chain_planes × n_cells.
    #[getter] pub fn chain_stride(&self) -> usize { self.inner.chain_stride() }
    /// Aux plane stride = n_cells (single aux plane).
    #[getter] pub fn aux_stride(&self) -> usize { self.inner.aux_stride() }
    /// Policy logit count = `policy_logit_count` (mirror of the field).
    #[getter] pub fn policy_stride(&self) -> usize { self.inner.policy_stride() }

    pub fn __repr__(&self) -> String {
        format!(
            "RegistrySpec(name={:?}, board_size={}, n_planes={}, policy_logit_count={}, is_multi_window={})",
            self.inner.name, self.inner.board_size, self.inner.n_planes,
            self.inner.policy_logit_count, self.inner.is_multi_window,
        )
    }

    /// Registry-backed lookup. Returns a `PyRegistrySpec` (full-schema record
    /// incl. policy_logit_count + n_planes). Raises `ValueError` on an unknown
    /// name (listing the registered set).
    #[classmethod]
    pub fn from_registry(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<Self> {
        if let Some(spec) = mantis_encoding::lookup(name) {
            Ok(PyRegistrySpec { inner: spec })
        } else {
            let mut known: Vec<&str> =
                mantis_encoding::all_specs().map(|s| s.name).collect();
            known.sort_unstable();
            Err(PyValueError::new_err(format!(
                "RegistrySpec.from_registry: unknown encoding {name:?}; registered: {known:?}"
            )))
        }
    }
}

impl PyRegistrySpec {
    /// Crate-internal accessor — used by `SelfPlayRunner::new` /
    /// `InferenceBatcher::new` (Slice R2) to read the static pointer.
    pub(crate) fn inner(&self) -> &'static RustRegistrySpec {
        self.inner
    }

    /// Construct from a `&'static RegistrySpec` — lets Rust integration tests
    /// (and R2 bindings) pass a `PyRegistrySpec` without going through the
    /// Python boundary.
    pub(crate) fn from_static(spec: &'static RustRegistrySpec) -> Self {
        PyRegistrySpec { inner: spec }
    }
}

/// Every registered encoding spec, as a list of `RegistrySpec` pyclasses, from
/// ONE call over `mantis_encoding::all_specs()`. The Python shim builds its cache
/// from this (replacing the per-name `_load` loop + the killed `_REGISTERED_NAMES`).
#[pyfunction]
pub(crate) fn all_specs() -> Vec<PyRegistrySpec> {
    mantis_encoding::all_specs()
        .map(|s| PyRegistrySpec { inner: s })
        .collect()
}

/// The 32 raw bytes of `sha256(embedded registry.toml)`. The Python import-time
/// handshake hashes the on-disk `registry.toml` and compares to this, hard-
/// erroring on drift (a stale `.so` / stale registry cannot silently serve).
#[pyfunction]
pub(crate) fn registry_sha<'py>(py: Python<'py>) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &mantis_encoding::registry_sha())
}

/// Lowercase-hex convenience over `registry_sha()`.
#[pyfunction]
pub(crate) fn registry_sha_hex() -> &'static str {
    mantis_encoding::registry_sha_hex()
}

/// Register the `RegistrySpec` pyclass + the 3 NEW-BUILD module fns into
/// `_engine`. Called by Slice ASM's `#[pymodule]` assembly.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRegistrySpec>()?;
    m.add_function(wrap_pyfunction!(all_specs, m)?)?;
    m.add_function(wrap_pyfunction!(registry_sha, m)?)?;
    m.add_function(wrap_pyfunction!(registry_sha_hex, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered_names() -> Vec<&'static str> {
        let mut names: Vec<&'static str> = mantis_encoding::all_specs().map(|s| s.name).collect();
        names.sort_unstable();
        names
    }

    #[test]
    fn all_specs_covers_the_registered_set() {
        let via_fn: Vec<&'static str> = {
            let mut v: Vec<&'static str> = all_specs().into_iter().map(|s| s.inner.name).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(via_fn, registered_names());
        // Pin the pruned 4-entry set (registry.toml authority).
        assert_eq!(
            via_fn,
            vec!["gnn_axis_v1", "v6", "v6_live2_ls", "v6w25"],
            "all_specs must expose exactly the 4 registered encodings"
        );
    }

    #[test]
    fn registry_sha_hex_matches_raw_bytes() {
        let raw = mantis_encoding::registry_sha();
        let hex = registry_sha_hex();
        let rebuilt: String = raw.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, rebuilt);
        assert_eq!(hex.len(), 64, "sha256 hex is 64 chars");
    }

    #[test]
    fn from_static_round_trips_derived_accessors() {
        let spec = mantis_encoding::lookup("v6").expect("v6 registered");
        let py = PyRegistrySpec::from_static(spec);
        assert_eq!(py.name(), "v6");
        assert_eq!(py.board_size(), 19);
        assert_eq!(py.policy_stride(), spec.policy_stride());
        assert_eq!(py.n_cells(), spec.n_cells());
        assert_eq!(py.state_stride(), spec.state_stride());
        assert_eq!(py.chain_stride(), spec.chain_stride());
        assert_eq!(py.aux_stride(), spec.aux_stride());
        assert_eq!(py.n_actions(), spec.policy_logit_count);
        assert_eq!(py.representation(), spec.representation.as_str());
        // Crate-internal accessor returns the same static pointer.
        assert!(std::ptr::eq(py.inner(), spec));
    }
}
