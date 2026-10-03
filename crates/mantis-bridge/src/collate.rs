//! `collate_pack`: `pack_wire` over numpy views with the GIL released, a refusal returned as `(class_name, message)`.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use mantis_selfplay::queues::collate::{pack_wire, PackError, PackOut, WireRef};

/// Checks 4–13 and the pack into the `out_*` views, which must not overlap; `RuntimeError` is a wiring break.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
#[pyo3(signature = (
    n_graphs, node_feat, edge_index, edge_attr, node_offsets, edge_offsets, legal_offsets,
    legal_node_gather, policy_dst_slot, n_nodes_checksum, n_stones, window_center, current_player,
    out_x, out_edge_index, out_edge_attr, out_legal_offsets, out_legal_node_gather, out_node_offsets,
    out_n_stones, node_feat_dim, edge_feat_dim, threads,
))]
pub(crate) fn collate_pack(
    py: Python<'_>,
    n_graphs: usize,
    node_feat: PyReadonlyArray1<'_, f32>,
    edge_index: PyReadonlyArray1<'_, i64>,
    edge_attr: PyReadonlyArray1<'_, f32>,
    node_offsets: PyReadonlyArray1<'_, i64>,
    edge_offsets: PyReadonlyArray1<'_, i64>,
    legal_offsets: PyReadonlyArray1<'_, i64>,
    legal_node_gather: PyReadonlyArray1<'_, i64>,
    policy_dst_slot: PyReadonlyArray1<'_, i32>,
    n_nodes_checksum: PyReadonlyArray1<'_, u32>,
    n_stones: PyReadonlyArray1<'_, u16>,
    window_center: PyReadonlyArray1<'_, i32>,
    current_player: PyReadonlyArray1<'_, i8>,
    mut out_x: PyReadwriteArray1<'_, f32>,
    mut out_edge_index: PyReadwriteArray1<'_, i64>,
    mut out_edge_attr: PyReadwriteArray1<'_, f32>,
    mut out_legal_offsets: PyReadwriteArray1<'_, i64>,
    mut out_legal_node_gather: PyReadwriteArray1<'_, i64>,
    mut out_node_offsets: PyReadwriteArray1<'_, i64>,
    mut out_n_stones: PyReadwriteArray1<'_, i64>,
    node_feat_dim: usize,
    edge_feat_dim: usize,
    threads: usize,
) -> PyResult<Option<(&'static str, String)>> {
    let wire = WireRef {
        n_graphs,
        node_feat: node_feat.as_slice()?,
        edge_index: edge_index.as_slice()?,
        edge_attr: edge_attr.as_slice()?,
        node_offsets: node_offsets.as_slice()?,
        edge_offsets: edge_offsets.as_slice()?,
        legal_offsets: legal_offsets.as_slice()?,
        legal_node_gather: legal_node_gather.as_slice()?,
        policy_dst_slot: policy_dst_slot.as_slice()?,
        n_nodes_checksum: n_nodes_checksum.as_slice()?,
        n_stones: n_stones.as_slice()?,
        window_center: window_center.as_slice()?,
        current_player: current_player.as_slice()?,
    };
    let mut out = PackOut {
        x: out_x.as_slice_mut()?,
        edge_index: out_edge_index.as_slice_mut()?,
        edge_attr: out_edge_attr.as_slice_mut()?,
        legal_offsets: out_legal_offsets.as_slice_mut()?,
        legal_node_gather: out_legal_node_gather.as_slice_mut()?,
        node_offsets: out_node_offsets.as_slice_mut()?,
        n_stones: out_n_stones.as_slice_mut()?,
    };
    // Plain slices are `Send`; the numpy borrows above outlive the detached section.
    match py.detach(|| pack_wire(&wire, &mut out, node_feat_dim, edge_feat_dim, threads)) {
        Ok(()) => Ok(None),
        Err(PackError::Contract { kind, message }) => Ok(Some((kind.name(), message))),
        Err(caller @ PackError::Caller(_)) => Err(PyRuntimeError::new_err(caller.to_string())),
    }
}

/// Register `collate_pack` into `_engine`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(collate_pack, m)?)?;
    Ok(())
}
