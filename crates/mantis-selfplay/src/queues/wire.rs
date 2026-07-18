//! `GraphWire` — the block-diagonal ragged graph batch tensor (WP6 D5), the
//! fuse-out of the graph queue's pop, ported from `inference_bridge.rs:1149` with
//! pyo3 STRIPPED.
//!
//! A plain-Rust flat-`Vec` type (the `#[pyclass]` + per-field `#[getter]` numpy
//! copies are WP7). `from_axis_graphs` is the VERBATIM block-diagonal fuse
//! (`inference_bridge.rs:1207`, offset accumulation `:1230-1262`): index arrays
//! come out ALREADY globally offset (`i64`); `edge_index` is
//! `[src_global (E) ‖ dst_global (E)]`.
//!
//! WP6 makes single-read a TYPE guarantee the frozen getters lacked: `take()`
//! moves every array out exactly once (inner `Option::take()`); a second call is
//! the NAMED error `WireAlreadyConsumed`. The `-1` off-window sentinel in
//! `policy_dst_slot` travels VERBATIM (copied from each graph's
//! `policy_scatter_index.0`, never re-densified — NO fixed-width fallback).

use std::fmt;

use mantis_graph::{AxisGraph, BUILDER_IMPL_NATIVE};

/// The owned flat arrays of one fused batch, yielded exactly once by
/// [`GraphWire::take`]. All index arrays are already globally offset.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphWireArrays {
    pub contract_version: u32,
    pub builder_impl: u8,
    pub n_graphs: usize,
    pub node_feat: Vec<f32>,
    pub node_coords: Vec<i32>,
    /// `[src_global (E) ‖ dst_global (E)]` — reshape to `(2, E)`.
    pub edge_index: Vec<i64>,
    pub edge_attr: Vec<f32>,
    pub node_offsets: Vec<i64>,
    pub edge_offsets: Vec<i64>,
    pub legal_offsets: Vec<i64>,
    pub legal_node_gather: Vec<i64>,
    /// Verbatim concat of per-graph `policy_scatter_index.0`, `-1` sentinel kept.
    pub policy_dst_slot: Vec<i32>,
    pub n_nodes_checksum: Vec<u32>,
    pub n_stones: Vec<u16>,
    pub window_center: Vec<i32>,
    pub current_player: Vec<i8>,
}

/// A fused wire whose arrays can be moved out exactly once. Wraps
/// `Option<GraphWireArrays>` so [`GraphWire::take`] is single-read.
pub struct GraphWire {
    arrays: Option<GraphWireArrays>,
}

/// Named error for a second [`GraphWire::take`] — the single-read guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireAlreadyConsumed;

impl fmt::Display for WireAlreadyConsumed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GraphWire arrays already consumed (single-read take() called twice)")
    }
}

impl std::error::Error for WireAlreadyConsumed {}

impl GraphWire {
    /// Block-diagonal fuse a batch of per-leaf `AxisGraph`s into ONE ragged wire.
    /// Single source of the fusion arithmetic (the self-play inference seam and
    /// the HEXG training sample path both call this). `edge_index` /
    /// `legal_node_gather` / all offsets come out ALREADY globally offset (`i64`);
    /// `edge_index` is `[src_global ‖ dst_global]`. Assumes each graph's
    /// `builder_impl == BUILDER_IMPL_NATIVE` (the caller runs the handshake).
    ///
    /// Verbatim port of `from_axis_graphs` (`inference_bridge.rs:1207`).
    #[must_use]
    pub fn from_axis_graphs(graphs: &[AxisGraph], contract_version: u32) -> GraphWire {
        let b = graphs.len();
        let mut node_feat: Vec<f32> = Vec::new();
        let mut node_coords: Vec<i32> = Vec::new();
        let mut edge_attr: Vec<f32> = Vec::new();
        let mut edge_src: Vec<i64> = Vec::new();
        let mut edge_dst: Vec<i64> = Vec::new();
        let mut legal_node_gather: Vec<i64> = Vec::new();
        let mut policy_dst_slot: Vec<i32> = Vec::new();
        let mut node_offsets: Vec<i64> = Vec::with_capacity(b + 1);
        let mut edge_offsets: Vec<i64> = Vec::with_capacity(b + 1);
        let mut legal_offsets: Vec<i64> = Vec::with_capacity(b + 1);
        let mut n_nodes_checksum: Vec<u32> = Vec::with_capacity(b);
        let mut n_stones: Vec<u16> = Vec::with_capacity(b);
        let mut window_center: Vec<i32> = Vec::with_capacity(b * 2);
        let mut current_player: Vec<i8> = Vec::with_capacity(b);
        node_offsets.push(0);
        edge_offsets.push(0);
        legal_offsets.push(0);

        let mut node_off: i64 = 0;
        let mut edge_off: i64 = 0;
        let mut legal_off: i64 = 0;
        for g in graphs {
            let n_g = g.num_nodes() as i64;
            let e_g = g.num_edges() as i64;
            let lg_g = g.legal_node_gather.len() as i64;

            node_feat.extend_from_slice(&g.node_feat.0);
            node_coords.extend_from_slice(&g.node_coords);
            edge_attr.extend_from_slice(&g.edge_attr.0);
            for &s in &g.edge_index.src {
                edge_src.push(node_off + i64::from(s));
            }
            for &d in &g.edge_index.dst {
                edge_dst.push(node_off + i64::from(d));
            }
            for &row in &g.legal_node_gather {
                legal_node_gather.push(node_off + i64::from(row));
            }
            policy_dst_slot.extend_from_slice(&g.policy_scatter_index.0);
            n_nodes_checksum.push(g.n_nodes_checksum);
            n_stones.push(g.n_stones);
            window_center.push(g.window_center.0);
            window_center.push(g.window_center.1);
            current_player.push(g.current_player);

            node_off += n_g;
            edge_off += e_g;
            legal_off += lg_g;
            node_offsets.push(node_off);
            edge_offsets.push(edge_off);
            legal_offsets.push(legal_off);
        }
        // edge_index = [src_global (E) | dst_global (E)] → reshape (2, E).
        let mut edge_index = edge_src;
        edge_index.extend(edge_dst);

        GraphWire {
            arrays: Some(GraphWireArrays {
                contract_version,
                builder_impl: BUILDER_IMPL_NATIVE,
                n_graphs: b,
                node_feat,
                node_coords,
                edge_index,
                edge_attr,
                node_offsets,
                edge_offsets,
                legal_offsets,
                legal_node_gather,
                policy_dst_slot,
                n_nodes_checksum,
                n_stones,
                window_center,
                current_player,
            }),
        }
    }

    /// Move every array out exactly once. A second call returns
    /// [`WireAlreadyConsumed`] (single-read guard).
    ///
    /// # Errors
    /// Returns `Err(WireAlreadyConsumed)` if the arrays were already taken.
    pub fn take(&mut self) -> Result<GraphWireArrays, WireAlreadyConsumed> {
        self.arrays.take().ok_or(WireAlreadyConsumed)
    }

    /// Whether the arrays are still present (not yet taken).
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.arrays.is_some()
    }
}
