//! PERF-TRANCHE-1 A1 — the reserve-and-single-pass fuse is BYTE-IDENTICAL to the
//! growth-and-concat fuse it replaces.
//!
//! `queue_fuse_pin.rs` pins the fuse's block-diagonal ARITHMETIC against frozen inputs.
//! This pins something narrower and, for a perf rewrite, sharper: the pre-A1 fuse is
//! transcribed here verbatim as `fuse_reference`, and every array of the shipped fuse must
//! equal it element-for-element on a corpus wide enough to exercise the seams the rewrite
//! actually moved — the `2E` single buffer, the second dst walk, and the reserved
//! capacities. A rewrite that is merely *self*-consistent passes the pin test; it cannot
//! pass this one.
//!
//! The reference is a LOAD-BEARING transcription, not commentary: it is the only remaining
//! copy of the arithmetic the shipped code no longer performs, and it must not be
//! "simplified" toward the shipped version.

use mantis_graph::{
    AxisGraph, EdgeAttr, EdgeIndex, NodeFeat, PolicyScatterIndex, BUILDER_IMPL_NATIVE,
    EDGE_FEAT_DIM, NODE_FEAT_DIM,
};
use mantis_selfplay::queues::{GraphWire, GraphWireArrays};

/// The pre-A1 fuse, transcribed: unreserved `Vec::new()` growth, per-element widening
/// `push` into separate `edge_src`/`edge_dst`, terminal `edge_index = edge_src; extend(dst)`.
fn fuse_reference(graphs: &[AxisGraph], contract_version: u32) -> GraphWireArrays {
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
    let mut edge_index = edge_src;
    edge_index.extend(edge_dst);

    GraphWireArrays {
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
    }
}

/// Pinned splitmix64 — the same generator family the committed benches use, so a corpus
/// is reproducible with no entropy and no model.
fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn draw(s: &mut u64, lo: u64, hi: u64) -> u64 {
    lo + splitmix64(s) % (hi - lo + 1)
}

/// A synthetic `AxisGraph` with the declared shapes and index ranges the fuse relies on.
/// Values are drawn, not meaningful: the fuse is index arithmetic and a verbatim concat,
/// so the ONLY thing that matters for parity is that every array is populated and every
/// local index is in range.
fn synth_graph(seed: &mut u64, n_nodes: usize, n_edges: usize, n_legal: usize) -> AxisGraph {
    let node_feat: Vec<f32> = (0..n_nodes * NODE_FEAT_DIM)
        .map(|_| (draw(seed, 0, 2000) as f32) / 1000.0 - 1.0)
        .collect();
    let node_coords: Vec<i32> = (0..n_nodes * 2)
        .map(|_| draw(seed, 0, 40) as i32 - 20)
        .collect();
    let edge_attr: Vec<f32> = (0..n_edges * EDGE_FEAT_DIM)
        .map(|_| (draw(seed, 0, 2000) as f32) / 1000.0 - 1.0)
        .collect();
    let src: Vec<u32> = (0..n_edges)
        .map(|_| draw(seed, 0, (n_nodes - 1) as u64) as u32)
        .collect();
    let dst: Vec<u32> = (0..n_edges)
        .map(|_| draw(seed, 0, (n_nodes - 1) as u64) as u32)
        .collect();
    let legal_node_gather: Vec<u32> = (0..n_legal)
        .map(|_| draw(seed, 0, (n_nodes - 1) as u64) as u32)
        .collect();
    // `-1` is the off-window sentinel and travels VERBATIM; seed it deliberately.
    let policy_scatter: Vec<i32> = (0..n_legal)
        .map(|i| {
            if i % 7 == 0 {
                -1
            } else {
                draw(seed, 0, 360) as i32
            }
        })
        .collect();
    AxisGraph {
        node_feat: NodeFeat(node_feat),
        edge_index: EdgeIndex { src, dst },
        edge_attr: EdgeAttr(edge_attr),
        legal_mask: vec![false; n_nodes],
        stone_mask: vec![false; n_nodes],
        policy_scatter_index: PolicyScatterIndex(policy_scatter),
        node_coords,
        legal_node_gather,
        n_stones: draw(seed, 0, 40) as u16,
        n_nodes_checksum: n_nodes as u32,
        window_center: (draw(seed, 0, 20) as i32 - 10, draw(seed, 0, 20) as i32 - 10),
        current_player: if draw(seed, 0, 1) == 0 { 1 } else { -1 },
        builder_impl: BUILDER_IMPL_NATIVE,
    }
}

fn assert_wire_eq(got: &GraphWireArrays, want: &GraphWireArrays, case: &str) {
    assert_eq!(
        got.contract_version, want.contract_version,
        "{case}: contract_version"
    );
    assert_eq!(got.builder_impl, want.builder_impl, "{case}: builder_impl");
    assert_eq!(got.n_graphs, want.n_graphs, "{case}: n_graphs");
    assert_eq!(got.node_feat, want.node_feat, "{case}: node_feat");
    assert_eq!(got.node_coords, want.node_coords, "{case}: node_coords");
    assert_eq!(got.edge_index, want.edge_index, "{case}: edge_index");
    assert_eq!(got.edge_attr, want.edge_attr, "{case}: edge_attr");
    assert_eq!(got.node_offsets, want.node_offsets, "{case}: node_offsets");
    assert_eq!(got.edge_offsets, want.edge_offsets, "{case}: edge_offsets");
    assert_eq!(
        got.legal_offsets, want.legal_offsets,
        "{case}: legal_offsets"
    );
    assert_eq!(
        got.legal_node_gather, want.legal_node_gather,
        "{case}: legal_node_gather"
    );
    assert_eq!(
        got.policy_dst_slot, want.policy_dst_slot,
        "{case}: policy_dst_slot"
    );
    assert_eq!(
        got.n_nodes_checksum, want.n_nodes_checksum,
        "{case}: n_nodes_checksum"
    );
    assert_eq!(got.n_stones, want.n_stones, "{case}: n_stones");
    assert_eq!(
        got.window_center, want.window_center,
        "{case}: window_center"
    );
    assert_eq!(
        got.current_player, want.current_player,
        "{case}: current_player"
    );
}

fn fuse_shipped(graphs: &[AxisGraph], cv: u32) -> GraphWireArrays {
    let mut w = GraphWire::from_axis_graphs(graphs, cv);
    w.take().expect("a freshly fused wire always has arrays")
}

#[test]
fn reserve_fuse_is_byte_identical_across_batch_shapes() {
    // Batch shapes that bracket production: the empty batch, the single graph (local ==
    // global), the two-graph offset seam, and a 40-graph batch — the ledger's measured
    // achieved batch at twelve workers is 39.29 of 64.
    for &b in &[0usize, 1, 2, 5, 40, 64] {
        let mut seed = 0x0A1F_0000_0000_0001 ^ (b as u64);
        let graphs: Vec<AxisGraph> = (0..b)
            .map(|_| {
                let n = draw(&mut seed, 40, 400) as usize;
                let e = draw(&mut seed, 1, 3000) as usize;
                let lg = draw(&mut seed, 1, n as u64) as usize;
                synth_graph(&mut seed, n, e, lg)
            })
            .collect();
        let want = fuse_reference(&graphs, 1);
        let got = fuse_shipped(&graphs, 1);
        assert_wire_eq(&got, &want, &format!("batch of {b}"));
    }
}

#[test]
fn reserve_fuse_is_byte_identical_on_degenerate_arrays() {
    // Zero-edge and zero-legal graphs are the seams where a reserved capacity of 0 and an
    // empty second walk could diverge from the growth path without any offset changing.
    let mut seed = 0x0A1F_0000_0000_0002;
    let graphs = vec![
        synth_graph(&mut seed, 12, 0, 4),
        synth_graph(&mut seed, 30, 90, 1),
        synth_graph(&mut seed, 7, 5, 7),
    ];
    let want = fuse_reference(&graphs, 3);
    let got = fuse_shipped(&graphs, 3);
    assert_wire_eq(&got, &want, "degenerate arrays");
}

#[test]
fn dst_half_carries_its_own_offset_walk() {
    // LAW-07 mutation self-test for THIS rewrite's one genuinely new mechanism: the dst
    // half is written by a SECOND walk over the graphs, with its own offset accumulator.
    // A reader that dropped that accumulator (dst written un-offset) must be caught here,
    // and nowhere else in the suite would notice — the offsets and the src half are both
    // still correct.
    let mut seed = 0x0A1F_0000_0000_0003;
    let graphs = vec![
        synth_graph(&mut seed, 50, 200, 10),
        synth_graph(&mut seed, 60, 300, 12),
    ];
    let arrays = fuse_shipped(&graphs, 1);
    let e_total: usize = graphs.iter().map(AxisGraph::num_edges).sum();
    let n0 = graphs[0].num_nodes() as i64;

    let dst_half = &arrays.edge_index[e_total..];
    let g1_dst = &dst_half[graphs[0].num_edges()..];
    let want_g1: Vec<i64> = graphs[1]
        .edge_index
        .dst
        .iter()
        .map(|&d| n0 + i64::from(d))
        .collect();
    assert_eq!(
        g1_dst,
        want_g1.as_slice(),
        "graph 1's dst half is offset by graph 0's node count"
    );
    // The mutation the self-test names: un-offset dst. It must NOT equal the shipped output.
    let unoffset: Vec<i64> = graphs[1]
        .edge_index
        .dst
        .iter()
        .map(|&d| i64::from(d))
        .collect();
    assert_ne!(
        g1_dst,
        unoffset.as_slice(),
        "an un-offset dst half must differ from the shipped one, or this test proves nothing"
    );
}
