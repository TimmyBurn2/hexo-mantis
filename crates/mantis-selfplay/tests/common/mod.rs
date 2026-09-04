//! Shared `S-PREFUSE` harness — ONE implementation, used by both the parity proof
//! (`tests/prefuse_concat_parity.rs`) and the bench arm (`benches/queue_fuse_bench.rs`).
//!
//! WHY SHARED AND NOT COPIED. The bench times `concat_by_offset` and the test proves it
//! byte-identical to the whole-batch fuse. Two copies would let the proof drift off the thing
//! being timed, and the number would then describe code nobody verified. `tests/common/` is a
//! subdirectory, so cargo does not compile it as its own test binary; the bench pulls it in
//! with `#[path]`.

#![allow(dead_code)]

use mantis_graph::AxisGraph;
use mantis_selfplay::queues::{build_leaf_graph, GraphWire, GraphWireArrays};

const WIN_LENGTH: u8 = 6;
const TRUNK_SIZE: i32 = 19;
const COORD_HALF: i64 = 12;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn draw(s: &mut u64, lo: i64, hi: i64) -> i64 {
    lo + (splitmix64(s) % ((hi - lo + 1) as u64)) as i64
}

/// A pop-shaped corpus: `n` graphs at run5-ish stone counts, deterministic in `seed`.
pub fn corpus(n: usize, radius: u16, seed: u64) -> Vec<AxisGraph> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let want = draw(&mut s, 70, 130) as usize;
        let mut stones: Vec<(i64, i64, i64)> = Vec::with_capacity(want);
        let mut seen = std::collections::HashSet::new();
        while stones.len() < want {
            let q = draw(&mut s, -COORD_HALF, COORD_HALF);
            let r = draw(&mut s, -COORD_HALF, COORD_HALF);
            if seen.insert((q, r)) {
                let player = if stones.len().is_multiple_of(2) { 1 } else { -1 };
                stones.push((q, r, player));
            }
        }
        let cp = if splitmix64(&mut s) & 1 == 0 { 1 } else { -1 };
        out.push(
            build_leaf_graph(&stones, cp, 1, WIN_LENGTH, radius, TRUNK_SIZE)
                .expect("a valid stone list builds"),
        );
    }
    out
}

/// Concatenate already-fused worker-local wires. **Offset arithmetic only** — no `AxisGraph`
/// is read, nothing is re-fused. This is the exact operation the card's server thread would
/// perform, written here so its output can be compared against the whole-batch fuse.
pub fn concat_by_offset(parts: Vec<GraphWireArrays>) -> GraphWireArrays {
    let cv = parts[0].contract_version;
    let impl_id = parts[0].builder_impl;
    let mut out = GraphWireArrays {
        contract_version: cv,
        builder_impl: impl_id,
        n_graphs: 0,
        node_feat: Vec::new(),
        node_coords: Vec::new(),
        edge_index: Vec::new(),
        edge_attr: Vec::new(),
        node_offsets: vec![0],
        edge_offsets: vec![0],
        legal_offsets: vec![0],
        legal_node_gather: Vec::new(),
        policy_dst_slot: Vec::new(),
        n_nodes_checksum: Vec::new(),
        n_stones: Vec::new(),
        window_center: Vec::new(),
        current_player: Vec::new(),
    };
    let mut node_base: i64 = 0;
    let mut edge_base: i64 = 0;
    let mut legal_base: i64 = 0;
    // Pass 1: everything except `edge_index`'s dst half.
    for p in &parts {
        let n_p = *p.node_offsets.last().expect("a wire carries B+1 node offsets");
        let e_p = *p.edge_offsets.last().expect("a wire carries B+1 edge offsets");
        let l_p = *p.legal_offsets.last().expect("a wire carries B+1 legal offsets");
        let e_p_us = e_p as usize;

        out.n_graphs += p.n_graphs;
        out.node_feat.extend_from_slice(&p.node_feat);
        out.node_coords.extend_from_slice(&p.node_coords);
        out.edge_attr.extend_from_slice(&p.edge_attr);
        out.edge_index.extend(p.edge_index[..e_p_us].iter().map(|&s| s + node_base));
        out.legal_node_gather.extend(p.legal_node_gather.iter().map(|&g| g + node_base));
        out.policy_dst_slot.extend_from_slice(&p.policy_dst_slot);
        out.n_nodes_checksum.extend_from_slice(&p.n_nodes_checksum);
        out.n_stones.extend_from_slice(&p.n_stones);
        out.window_center.extend_from_slice(&p.window_center);
        out.current_player.extend_from_slice(&p.current_player);
        out.node_offsets.extend(p.node_offsets[1..].iter().map(|&o| o + node_base));
        out.edge_offsets.extend(p.edge_offsets[1..].iter().map(|&o| o + edge_base));
        out.legal_offsets.extend(p.legal_offsets[1..].iter().map(|&o| o + legal_base));

        node_base += n_p;
        edge_base += e_p;
        legal_base += l_p;
    }
    // Pass 2: the dst half, appended after every src — the splice `edge_index`'s
    // `[src ‖ dst]` layout forces. An append instead of a splice yields a wire that passes
    // every length check and is wrong about every edge.
    let mut node_base_dst: i64 = 0;
    for p in &parts {
        let e_p_us = *p.edge_offsets.last().expect("edge offsets") as usize;
        out.edge_index.extend(p.edge_index[e_p_us..].iter().map(|&d| d + node_base_dst));
        node_base_dst += *p.node_offsets.last().expect("node offsets");
    }
    out
}

pub fn fuse(graphs: &[AxisGraph]) -> GraphWireArrays {
    let mut w = GraphWire::from_axis_graphs(graphs, 1);
    w.take().expect("a freshly fused wire always has arrays")
}

