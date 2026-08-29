//! P-09 ⊕ — the multi-graph GraphWire block-diagonal fuse pin (written FIRST).
//!
//! Recovers the O-24 multi-graph offset-accumulation coverage WP5 shed (WP5
//! re-anchored O-24 to a single-graph `Vec`, so local == global there). Loads the
//! 3 dispatcher-frozen fuse-INPUT `AxisGraph`s (`graphwire_multigraph_input.bin`,
//! CAPTURE_LOG §A.2/§C g8 — the fuse is `pub(crate)` old-side, so the INPUTS are
//! pinned and P-09 RECONSTRUCTS), runs `GraphWire::from_axis_graphs`, and asserts
//! the block-diagonal reconstruction EXACTLY: offsets are running prefix sums;
//! `edge_index` src/dst = local + `node_off`; `legal_node_gather` = local +
//! `node_off`; `policy_dst_slot` = verbatim concat incl. the `-1` off-window
//! sentinel. Plus the single-graph degenerate case (local == global), the
//! single-read `take()` guard, and the mandatory LAW-07 mutation self-test.

use mantis_graph::{
    AxisGraph, EdgeAttr, EdgeIndex, NodeFeat, PolicyScatterIndex, BUILDER_IMPL_NATIVE,
};
use mantis_selfplay::queues::{GraphWire, GraphWireArrays, WireAlreadyConsumed};

// Frozen expected offsets (CAPTURE_LOG §A.2): running prefix sums of the
// per-graph (nodes, edges, legal) counts (148,3316,144)/(359,8236,348)/(167,3758,162).
const EXPECT_NODE_OFFSETS: [i64; 4] = [0, 148, 507, 674];
const EXPECT_EDGE_OFFSETS: [i64; 4] = [0, 3316, 11552, 15310];
const EXPECT_LEGAL_OFFSETS: [i64; 4] = [0, 144, 492, 654];

// ── fixture reader (CAPTURE_LOG §C g8 byte layout, little-endian) ────────────

/// A per-graph slice's LOCAL arrays, retained alongside the built `AxisGraph`
/// so the fused output can be reconstructed against them.
struct Local {
    edge_src: Vec<u32>,
    edge_dst: Vec<u32>,
    legal_gather: Vec<u32>,
    policy_scatter: Vec<i32>,
}

struct Cur<'a> {
    data: &'a [u8],
    off: usize,
}

impl Cur<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.data[self.off..self.off + 4].try_into().unwrap());
        self.off += 4;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes(self.data[self.off..self.off + 2].try_into().unwrap());
        self.off += 2;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.data[self.off..self.off + 4].try_into().unwrap());
        self.off += 4;
        v
    }
    fn i8(&mut self) -> i8 {
        let v = self.data[self.off] as i8;
        self.off += 1;
        v
    }
    fn vec_u32(&mut self) -> Vec<u32> {
        let n = self.u32() as usize;
        (0..n).map(|_| self.u32()).collect()
    }
    fn vec_i32(&mut self) -> Vec<i32> {
        let n = self.u32() as usize;
        (0..n).map(|_| self.i32()).collect()
    }
    fn vec_f32(&mut self) -> Vec<f32> {
        let n = self.u32() as usize;
        (0..n)
            .map(|_| {
                let v = f32::from_le_bytes(self.data[self.off..self.off + 4].try_into().unwrap());
                self.off += 4;
                v
            })
            .collect()
    }
}

fn read_input() -> (Vec<AxisGraph>, Vec<Local>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker")
        .join("graphwire_multigraph_input.bin");
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut cur = Cur {
        data: &data,
        off: 0,
    };

    let n_graphs = cur.u32() as usize;
    assert_eq!(
        n_graphs, 3,
        "fixture carries 3 fuse-input graphs (CAPTURE_LOG §C g8)"
    );
    let mut graphs = Vec::with_capacity(n_graphs);
    let mut locals = Vec::with_capacity(n_graphs);
    for _ in 0..n_graphs {
        let n_nodes = cur.u32();
        let n_edges = cur.u32();
        let node_feat = cur.vec_f32();
        let node_coords = cur.vec_i32();
        let edge_src = cur.vec_u32();
        let edge_dst = cur.vec_u32();
        let edge_attr = cur.vec_f32();
        let legal_gather = cur.vec_u32();
        let policy_scatter = cur.vec_i32();
        let n_nodes_checksum = cur.u32();
        let n_stones = cur.u16();
        let wc_q = cur.i32();
        let wc_r = cur.i32();
        let current_player = cur.i8();

        let g = AxisGraph {
            node_feat: NodeFeat(node_feat),
            edge_index: EdgeIndex {
                src: edge_src.clone(),
                dst: edge_dst.clone(),
            },
            edge_attr: EdgeAttr(edge_attr),
            legal_mask: Vec::new(),
            stone_mask: Vec::new(),
            policy_scatter_index: PolicyScatterIndex(policy_scatter.clone()),
            node_coords,
            legal_node_gather: legal_gather.clone(),
            n_stones,
            n_nodes_checksum,
            window_center: (wc_q, wc_r),
            current_player,
            builder_impl: BUILDER_IMPL_NATIVE,
        };
        // Fixture-consistency: the fuse reads num_nodes()/num_edges() (feat/edge-
        // derived), which must equal the stored per-graph counts.
        assert_eq!(
            g.num_nodes() as u32,
            n_nodes,
            "node_feat length ⇒ num_nodes"
        );
        assert_eq!(
            g.num_edges() as u32,
            n_edges,
            "edge_index.src length ⇒ num_edges"
        );
        graphs.push(g);
        locals.push(Local {
            edge_src,
            edge_dst,
            legal_gather,
            policy_scatter,
        });
    }
    (graphs, locals)
}

// ── reconstruction check (P-09 mechanism; also the mutation-bite target) ─────

/// Verify the fused arrays reconstruct the per-graph locals block-diagonally.
/// Uses the fused offsets as the source of truth for each graph's `node_off`, so
/// a corrupted `node_offsets` entry makes the reconstruction disagree.
fn reconstruct_ok(
    a: &GraphWireArrays,
    graphs: &[AxisGraph],
    locals: &[Local],
) -> Result<(), String> {
    let b = graphs.len();
    // (i) offsets are running prefix sums of the per-graph counts.
    let mut node_acc = 0i64;
    let mut edge_acc = 0i64;
    let mut legal_acc = 0i64;
    if a.node_offsets[0] != 0 || a.edge_offsets[0] != 0 || a.legal_offsets[0] != 0 {
        return Err("offset[0] != 0".to_string());
    }
    for (i, g) in graphs.iter().enumerate() {
        node_acc += g.num_nodes() as i64;
        edge_acc += g.num_edges() as i64;
        legal_acc += g.legal_node_gather.len() as i64;
        if a.node_offsets[i + 1] != node_acc {
            return Err(format!("node_offsets[{}] != prefix-sum {node_acc}", i + 1));
        }
        if a.edge_offsets[i + 1] != edge_acc {
            return Err(format!("edge_offsets[{}] != prefix-sum {edge_acc}", i + 1));
        }
        if a.legal_offsets[i + 1] != legal_acc {
            return Err(format!(
                "legal_offsets[{}] != prefix-sum {legal_acc}",
                i + 1
            ));
        }
    }
    // total directed edge count E; edge_index = [src (E) | dst (E)].
    let e_total = a.edge_offsets[b] as usize;
    if a.edge_index.len() != 2 * e_total {
        return Err("edge_index len != 2*E".to_string());
    }
    // (ii)/(iii)/(iv) per-graph slice reconstruction.
    for (i, loc) in locals.iter().enumerate() {
        let node_off = a.node_offsets[i];
        let es = a.edge_offsets[i] as usize;
        for (j, &s) in loc.edge_src.iter().enumerate() {
            if a.edge_index[es + j] != node_off + i64::from(s) {
                return Err(format!("edge_index src g{i}[{j}] != local+node_off"));
            }
        }
        for (j, &d) in loc.edge_dst.iter().enumerate() {
            if a.edge_index[e_total + es + j] != node_off + i64::from(d) {
                return Err(format!("edge_index dst g{i}[{j}] != local+node_off"));
            }
        }
        let ls = a.legal_offsets[i] as usize;
        let le = a.legal_offsets[i + 1] as usize;
        for (j, &row) in loc.legal_gather.iter().enumerate() {
            if a.legal_node_gather[ls + j] != node_off + i64::from(row) {
                return Err(format!("legal_node_gather g{i}[{j}] != local+node_off"));
            }
        }
        // (iv) policy_dst_slot is the VERBATIM concat (incl. -1 sentinel).
        if a.policy_dst_slot[ls..le] != loc.policy_scatter[..] {
            return Err(format!("policy_dst_slot g{i} != verbatim scatter"));
        }
    }
    Ok(())
}

// ── P-09 positive pins ───────────────────────────────────────────────────────

#[test]
fn fuse_offsets_match_frozen_prefix_sums() {
    let (graphs, _locals) = read_input();
    let mut wire = GraphWire::from_axis_graphs(&graphs, 1);
    let a = wire.take().expect("first take yields the arrays");

    assert_eq!(a.n_graphs, 3);
    assert_eq!(a.contract_version, 1);
    assert_eq!(a.builder_impl, BUILDER_IMPL_NATIVE);
    assert_eq!(a.node_offsets, EXPECT_NODE_OFFSETS, "node_offsets");
    assert_eq!(a.edge_offsets, EXPECT_EDGE_OFFSETS, "edge_offsets");
    assert_eq!(a.legal_offsets, EXPECT_LEGAL_OFFSETS, "legal_offsets");

    // per-graph metadata travels in order.
    assert_eq!(a.n_stones, vec![3u16, 10, 4]);
    assert_eq!(a.window_center, vec![0, 0, 17, 0, 1, 0]);
    assert_eq!(a.current_player, vec![1i8, 1, 1]);
    assert_eq!(a.n_nodes_checksum, vec![148u32, 359, 167]);
}

#[test]
fn fuse_reconstructs_each_graph_block_diagonal() {
    let (graphs, locals) = read_input();
    let mut wire = GraphWire::from_axis_graphs(&graphs, 1);
    let a = wire.take().expect("arrays");
    reconstruct_ok(&a, &graphs, &locals).expect("block-diagonal reconstruction holds");

    // -1 off-window sentinel travels verbatim (g1 carries it), never densified.
    assert!(
        a.policy_dst_slot.contains(&-1),
        "the -1 off-window sentinel must survive the fuse verbatim"
    );
}

#[test]
fn single_graph_local_equals_global() {
    // O-24 degenerate case: one graph → node_off stays 0, local == global.
    let (graphs, locals) = read_input();
    let one = vec![graphs[1].clone()];
    let one_local = &locals[1];
    let mut wire = GraphWire::from_axis_graphs(&one, 1);
    let a = wire.take().expect("arrays");
    assert_eq!(a.node_offsets, vec![0, 359]);
    assert_eq!(a.edge_offsets, vec![0, 8236]);
    assert_eq!(a.legal_offsets, vec![0, 348]);
    // node_off == 0 ⇒ globalised arrays equal the local ones.
    let e = 8236usize;
    for (j, &s) in one_local.edge_src.iter().enumerate() {
        assert_eq!(a.edge_index[j], i64::from(s));
    }
    for (j, &d) in one_local.edge_dst.iter().enumerate() {
        assert_eq!(a.edge_index[e + j], i64::from(d));
    }
    for (j, &row) in one_local.legal_gather.iter().enumerate() {
        assert_eq!(a.legal_node_gather[j], i64::from(row));
    }
    assert_eq!(a.policy_dst_slot, one_local.policy_scatter);
}

#[test]
fn take_is_single_read() {
    let (graphs, _locals) = read_input();
    let mut wire = GraphWire::from_axis_graphs(&graphs, 1);
    assert!(wire.is_available());
    let _first = wire.take().expect("first take ok");
    assert!(!wire.is_available());
    let second = wire.take();
    assert_eq!(
        second,
        Err(WireAlreadyConsumed),
        "second take is the named error"
    );
}

// ── P-09 LAW-07 mutation self-test (mandatory) ───────────────────────────────

#[test]
fn corrupt_node_offset_fires_reconstruction() {
    let (graphs, locals) = read_input();
    let mut wire = GraphWire::from_axis_graphs(&graphs, 1);
    let good = wire.take().expect("arrays");
    // sanity: the clean arrays reconstruct.
    reconstruct_ok(&good, &graphs, &locals).expect("clean arrays reconstruct");

    // Corrupt ONE node_offsets entry by +1 in a shadow copy.
    let mut bad = good.clone();
    bad.node_offsets[1] += 1;
    let fired = reconstruct_ok(&bad, &graphs, &locals);
    assert!(
        fired.is_err(),
        "a corrupted node_offsets entry MUST fire the reconstruction assertion (bite proof)"
    );
}
