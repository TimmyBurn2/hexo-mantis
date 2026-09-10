// Exceeds the 300-line soft cap (R8): the plain-slice edge-geometry validator and its full
// mutation-oracle test suite port as one line-auditable unit.
//! `verify_edge_geometry` — the ragged graph-wire contract's `EdgeAttrGeometryMismatch` semantic
//! check, ported onto the fused multi-graph wire as a single Rust pass over the SAME post-marshal
//! arrays, read as zero-copy readonly views. It mirrors `mantis_graph::verify_contract`'s
//! per-edge re-derivation, adapted to global edge ids and a `node_offsets`-derived ownership map.
//!
//! Every sub-assertion moves verbatim — dummy-edge all-zero attrs, clean axis one-hot, an
//! integral and bounded `signed_dist`, `src_player` identity — none sampled, skipped or
//! debug-gated, and a mismatch raises the generic error the Python call site re-raises as the
//! NAMED exception.
//!
//! Deliberately NOT in `mantis-graph`, which stays wasm32-clean: this is bridge-crate PyO3 glue.
//! It never panics on malformed input, and the logic lives in the plain-slice impl so
//! `cargo test` can drive it without an interpreter.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_graph::WIN_AXES;
/// Recompute + verify every real (non-dummy) edge's `edge_attr` row against the geometry implied
/// by its endpoints' `node_coords` and the owning graph's `current_player`. Pure Rust over plain
/// slices, on the post-marshal flat layout: `node_feat` row-major `(N, node_feat_dim)` (col 0
/// own, col 1 opp), `node_coords` flat `(N, 2)`, `edge_index` flat `(2, E)`, `edge_attr`
/// row-major `[onehot0, onehot1, onehot2, signed_dist, src_player]`, `node_offsets` CSR whose
/// last row per graph is its dummy node, `current_player` per-graph in `{+1, -1}`.
///
/// # Errors
/// Returns `Err(message)` on the first geometry violation or malformed-shape input. Every index
/// is range-checked before use rather than trusting the structural layer.
/// Returns `Err(message)` on the first geometry violation or malformed-shape input. Every index
/// is range-checked before use rather than trusting the structural layer.
// `float_cmp` allowed: the compared floats are EXACT constants the builder itself wrote, so
// approximation would WEAKEN the check.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::float_cmp
)]
fn verify_edge_geometry_impl(
    node_feat: &[f32],
    node_coords: &[i32],
    edge_index: &[i64],
    edge_attr: &[f32],
    node_offsets: &[i64],
    current_player: &[i8],
    node_feat_dim: usize,
    edge_feat_dim: usize,
    win_length: i64,
) -> Result<(), String> {
    // Shape guards: this is the externally injectable producer, so it must never index out of
    // range on a corrupt input. The dim bound is >= 2 because the body reads channel 1.
    if node_feat_dim < 2 || edge_feat_dim < 5 {
        return Err(format!(
            "verify_edge_geometry: degenerate dims node_feat_dim={node_feat_dim} \
             edge_feat_dim={edge_feat_dim} (node_feat_dim must be >= 2: this check reads \
             channel 1, the opponent-stone plane, of every node's row)"
        ));
    }
    if !node_feat.len().is_multiple_of(node_feat_dim) {
        return Err(format!(
            "verify_edge_geometry: len(node_feat)={} not divisible by node_feat_dim={node_feat_dim}",
            node_feat.len()
        ));
    }
    let n = node_feat.len() / node_feat_dim;
    if node_coords.len() != 2 * n {
        return Err(format!(
            "verify_edge_geometry: len(node_coords)={} != 2N={}",
            node_coords.len(),
            2 * n
        ));
    }
    if !edge_attr.len().is_multiple_of(edge_feat_dim) {
        return Err(format!(
            "verify_edge_geometry: len(edge_attr)={} not divisible by edge_feat_dim={edge_feat_dim}",
            edge_attr.len()
        ));
    }
    let e = edge_attr.len() / edge_feat_dim;
    if edge_index.len() != 2 * e {
        return Err(format!(
            "verify_edge_geometry: len(edge_index)={} != 2E={}",
            edge_index.len(),
            2 * e
        ));
    }
    if e == 0 {
        return Ok(()); // mirrors the Python `if E > 0:` guard — nothing to check
    }
    if node_offsets.is_empty() {
        return Err("verify_edge_geometry: node_offsets is empty".to_string());
    }
    let b = node_offsets.len() - 1;
    if current_player.len() != b {
        return Err(format!(
            "verify_edge_geometry: len(current_player)={} != B={b}",
            current_player.len()
        ));
    }

    // node_is_dummy/node_graph: one O(N) pass over node_offsets, shared prep per edge.
    let mut node_is_dummy = vec![false; n];
    let mut node_graph = vec![0u32; n];
    for g in 0..b {
        let start = node_offsets[g];
        let end = node_offsets[g + 1];
        if start < 0 || end < start || (end as usize) > n {
            return Err(format!(
                "verify_edge_geometry: node_offsets[{g}..{}] = [{start},{end}] invalid for N={n}",
                g + 1
            ));
        }
        node_graph[(start as usize)..(end as usize)].fill(g as u32);
        if end > start {
            let dummy_idx = (end - 1) as usize;
            node_is_dummy[dummy_idx] = true;
        }
    }

    let win_max = win_length - 1;

    for edge in 0..e {
        let s = edge_index[edge];
        let d = edge_index[e + edge];
        if s < 0 || d < 0 || (s as usize) >= n || (d as usize) >= n {
            return Err(format!(
                "verify_edge_geometry: edge {edge} endpoints ({s},{d}) out of [0,{n})"
            ));
        }
        let (s, d) = (s as usize, d as usize);
        let a = &edge_attr[edge * edge_feat_dim..edge * edge_feat_dim + edge_feat_dim];

        if node_is_dummy[s] || node_is_dummy[d] {
            // dummy edges must be all-zero.
            if a.iter().any(|&x| x.abs() > 1e-6) {
                return Err(format!(
                    "a dummy edge has non-zero attrs (edge {edge}): {a:?}"
                ));
            }
            continue;
        }

        // exactly one of the 3 axis one-hots is 1.0 (first-max tie-break, matching np.argmax).
        let onehot = [a[0], a[1], a[2]];
        let mut axis = 0usize;
        for k in 1..3 {
            if onehot[k] > onehot[axis] {
                axis = k;
            }
        }
        let onehot_ok = (0..3).all(|k| onehot[k] == if k == axis { 1.0 } else { 0.0 });
        if !onehot_ok {
            return Err(format!(
                "edge axis one-hot is not a clean one-hot (edge {edge}): {onehot:?}"
            ));
        }

        // signed_dist must be integral: a fractional value never equals its own rounded form.
        let dist = a[3]; // edge_attr col 3 = signed_dist (contract §2.1, not a state plane)
        let di = dist.round() as i64;
        if dist != di as f32 {
            return Err(format!("signed_dist is non-integral (edge {edge}): {dist}"));
        }

        let (aq, ar) = WIN_AXES[axis];
        let sq = i64::from(node_coords[s * 2]);
        let sr = i64::from(node_coords[s * 2 + 1]);
        let dq = i64::from(node_coords[d * 2]);
        let dr = i64::from(node_coords[d * 2 + 1]);
        let delta_q = dq - sq;
        let delta_r = dr - sr;
        let expect_q = di * i64::from(aq);
        let expect_r = di * i64::from(ar);
        if delta_q != expect_q || delta_r != expect_r || di == 0 || di.abs() > win_max {
            return Err(format!(
                "edge delta != signed_dist * axis_vec (rows misaligned/scrambled) (edge {edge}): \
                 delta=({delta_q},{delta_r}) expected=({expect_q},{expect_r}) di={di} win_max={win_max}"
            ));
        }

        // src_player: relative own/opp cols × current_player[graph], 0 for empty.
        let g_of = node_graph[s] as usize;
        let own = node_feat[s * node_feat_dim];
        let opp = node_feat[s * node_feat_dim + 1];
        let cp = f32::from(current_player[g_of]);
        let expect_sp = (own - opp) * cp;
        if (a[4] - expect_sp).abs() > 1e-6 {
            // edge_attr col 4 = src_player (contract §2.1)
            return Err(format!(
                "edge src_player != node stone identity (edge {edge}): got {} expected {expect_sp}",
                a[4]
            ));
        }
    }

    Ok(())
}

/// PyO3 shim: extract zero-copy readonly slices, delegate, map `Err(String)` to `PyValueError`,
/// which the Python call site re-raises as the named `EdgeAttrGeometryMismatch`.
///
/// # Errors
/// `PyValueError` on any geometry violation or malformed-shape input; also propagates
/// `AsSliceError` if a caller passes a non-contiguous numpy view.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_edge_geometry(
    node_feat: PyReadonlyArray1<'_, f32>,
    node_coords: PyReadonlyArray1<'_, i32>,
    edge_index: PyReadonlyArray1<'_, i64>,
    edge_attr: PyReadonlyArray1<'_, f32>,
    node_offsets: PyReadonlyArray1<'_, i64>,
    current_player: PyReadonlyArray1<'_, i8>,
    node_feat_dim: usize,
    edge_feat_dim: usize,
    win_length: i64,
) -> PyResult<()> {
    verify_edge_geometry_impl(
        node_feat.as_slice()?,
        node_coords.as_slice()?,
        edge_index.as_slice()?,
        edge_attr.as_slice()?,
        node_offsets.as_slice()?,
        current_player.as_slice()?,
        node_feat_dim,
        edge_feat_dim,
        win_length,
    )
    .map_err(PyValueError::new_err)
}

/// Register the `verify_edge_geometry` free fn into `_engine`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_edge_geometry, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::verify_edge_geometry_impl;

    const NODE_FEAT_DIM: usize = 11;
    const EDGE_FEAT_DIM: usize = 5;
    const WIN_LENGTH: i64 = 6;

    /// One graph: 2 stones (own at (0,0), opp at (1,0)), 1 legal node (2,0), 1 dummy (row 3),
    /// current_player = +1, one clean axis-0 edge with signed_dist=2, src_player=1.
    #[allow(clippy::type_complexity)] // 6-array flat-wire fixture tuple
    fn clean_fixture() -> (Vec<f32>, Vec<i32>, Vec<i64>, Vec<f32>, Vec<i64>, Vec<i8>) {
        let mut node_feat = vec![0.0f32; 4 * NODE_FEAT_DIM];
        node_feat[0] = 1.0; // stone 0: own=1, opp=0
        node_feat[NODE_FEAT_DIM + 1] = 1.0; // stone 1: own=0, opp=1
        let node_coords: Vec<i32> = vec![0, 0, 1, 0, 2, 0, 0, 0]; // stone0,stone1,legal,dummy
        let edge_index: Vec<i64> = vec![0, 2]; // src=[0], dst=[2] (flat (2,E))
        let mut edge_attr = vec![0.0f32; EDGE_FEAT_DIM];
        edge_attr[0] = 1.0; // axis 0 one-hot
        edge_attr[3] = 2.0; // signed_dist
        edge_attr[4] = 1.0; // src_player = (1-0)*cp(+1) = 1
        let node_offsets: Vec<i64> = vec![0, 4]; // B=1, N=4
        let current_player: Vec<i8> = vec![1];
        (
            node_feat,
            node_coords,
            edge_index,
            edge_attr,
            node_offsets,
            current_player,
        )
    }

    #[test]
    fn clean_edge_passes() {
        let (nf, nc, ei, ea, no, cp) = clean_fixture();
        assert!(verify_edge_geometry_impl(
            &nf,
            &nc,
            &ei,
            &ea,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH
        )
        .is_ok());
    }

    #[test]
    fn empty_edge_set_is_ok() {
        let (nf, nc, _ei, _ea, no, cp) = clean_fixture();
        assert!(verify_edge_geometry_impl(
            &nf,
            &nc,
            &[],
            &[],
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH
        )
        .is_ok());
    }

    /// The ADV-8 corruption: a flipped `signed_dist` sign no longer matches the coord delta.
    #[test]
    fn flipped_signed_dist_raises() {
        let (nf, nc, ei, mut ea, no, cp) = clean_fixture();
        ea[3] = -ea[3];
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &ei,
            &ea,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(
            err.contains("edge delta"),
            "expected a geometry-delta mismatch message, got: {err}"
        );
    }

    #[test]
    fn dirty_onehot_raises() {
        let (nf, nc, ei, mut ea, no, cp) = clean_fixture();
        ea[1] = 1.0; // now two axes set — not a clean one-hot
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &ei,
            &ea,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("one-hot"), "got: {err}");
    }

    #[test]
    fn non_integral_signed_dist_raises() {
        let (nf, nc, ei, mut ea, no, cp) = clean_fixture();
        ea[3] = 2.5;
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &ei,
            &ea,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("non-integral"), "got: {err}");
    }

    #[test]
    fn wrong_src_player_raises() {
        let (nf, nc, ei, mut ea, no, cp) = clean_fixture();
        ea[4] = -1.0; // should be +1
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &ei,
            &ea,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("src_player"), "got: {err}");
    }

    #[test]
    fn nonzero_dummy_edge_raises() {
        let (nf, nc, _ei, _ea, no, cp) = clean_fixture();
        let edge_index: Vec<i64> = vec![0, 3]; // src=stone0, dst=dummy(row 3)
        let mut edge_attr = vec![0.0f32; EDGE_FEAT_DIM];
        edge_attr[3] = 1.0; // non-zero on a dummy edge
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &edge_index,
            &edge_attr,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("dummy edge"), "got: {err}");
    }

    #[test]
    fn clean_dummy_edge_is_ok() {
        let (nf, nc, _ei, _ea, no, cp) = clean_fixture();
        let edge_index: Vec<i64> = vec![0, 3]; // src=stone0, dst=dummy(row 3)
        let edge_attr = vec![0.0f32; EDGE_FEAT_DIM]; // all-zero
        assert!(verify_edge_geometry_impl(
            &nf,
            &nc,
            &edge_index,
            &edge_attr,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH
        )
        .is_ok());
    }

    #[test]
    fn out_of_range_edge_endpoint_raises_not_panics() {
        let (nf, nc, _ei, ea, no, cp) = clean_fixture();
        let edge_index: Vec<i64> = vec![0, 99]; // dst way out of [0, N)
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &edge_index,
            &ea,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("out of"), "got: {err}");
    }

    #[test]
    fn negative_edge_endpoint_raises_not_panics() {
        let (nf, nc, _ei, ea, no, cp) = clean_fixture();
        let edge_index: Vec<i64> = vec![-1, 2];
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &edge_index,
            &ea,
            &no,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("out of"), "got: {err}");
    }

    #[test]
    fn malformed_node_offsets_raises_not_panics() {
        let (nf, nc, ei, ea, _no, cp) = clean_fixture();
        let node_offsets: Vec<i64> = vec![0, 999]; // end > N
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &ei,
            &ea,
            &node_offsets,
            &cp,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("node_offsets"), "got: {err}");
    }

    #[test]
    fn current_player_length_mismatch_raises() {
        let (nf, nc, ei, ea, no, _cp) = clean_fixture();
        let current_player: Vec<i8> = vec![1, -1]; // B=1 expected, got 2
        let err = verify_edge_geometry_impl(
            &nf,
            &nc,
            &ei,
            &ea,
            &no,
            &current_player,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("current_player"), "got: {err}");
    }

    /// A distance outside [1, win_length-1] raises even when one-hot and delta agree.
    #[test]
    fn out_of_window_distance_raises() {
        let node_feat = vec![0.0f32; 4 * NODE_FEAT_DIM];
        let node_coords: Vec<i32> = vec![0, 0, 6, 0, 2, 0, 0, 0];
        let edge_index: Vec<i64> = vec![0, 1];
        let mut edge_attr = vec![0.0f32; EDGE_FEAT_DIM];
        edge_attr[0] = 1.0;
        edge_attr[3] = 6.0; // di=6 > win_max=5, but delta = (6,0) = di*axis0 — consistent
        let node_offsets: Vec<i64> = vec![0, 4];
        let current_player: Vec<i8> = vec![1];
        let err = verify_edge_geometry_impl(
            &node_feat,
            &node_coords,
            &edge_index,
            &edge_attr,
            &node_offsets,
            &current_player,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            WIN_LENGTH,
        )
        .unwrap_err();
        assert!(err.contains("edge delta"), "got: {err}");
    }
}
