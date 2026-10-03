// >300 justify (R8): every structural kind, the coded pack and the thread split are pinned against one fixture
// set, so a kind added to the pack lands beside the cases that prove it.

//! The pack's byte parity against the fused wire and every structural kind, in the contract's order.

use super::*;
use crate::queues::{build_leaf_graph, GraphWire, GraphWireArrays};

const NODE_DIM: usize = 11;
const EDGE_DIM: usize = 5;

fn wire(positions: &[&[(i64, i64, i64)]]) -> GraphWireArrays {
    let graphs: Vec<_> = positions
        .iter()
        .map(|stones| build_leaf_graph(stones, 1, 2, 6, 6, 19).expect("a legal test position"))
        .collect();
    GraphWire::from_axis_graphs(&graphs, 1)
        .take()
        .expect("a fresh wire")
}

fn three() -> GraphWireArrays {
    wire(&[
        &[(0, 0, 1), (1, 0, -1), (0, 1, -1)],
        &[(0, 0, 1)],
        &[(0, 0, 1), (2, -1, -1), (3, -1, 1), (5, 0, -1)],
    ])
}

/// Past the parallel threshold, so the thread-count tests split the copy for real.
fn many() -> GraphWireArrays {
    let a: &[(i64, i64, i64)] = &[(0, 0, 1), (2, -1, -1), (3, -1, 1), (5, 0, -1)];
    let b: &[(i64, i64, i64)] = &[(0, 0, 1), (1, 0, -1), (0, 1, -1)];
    let positions: Vec<&[(i64, i64, i64)]> =
        (0..64).map(|i| if i % 3 == 0 { b } else { a }).collect();
    let w = wire(&positions);
    assert!(
        w.edge_attr.len() / EDGE_DIM >= PARALLEL_MIN_EDGES,
        "the wire must cross the threshold"
    );
    w
}

struct Out {
    x: Vec<f32>,
    edge_index: Vec<i64>,
    edge_attr: Vec<f32>,
    legal_offsets: Vec<i64>,
    legal_node_gather: Vec<i64>,
    node_offsets: Vec<i64>,
    n_stones: Vec<i64>,
}

fn out_for(a: &GraphWireArrays) -> Out {
    Out {
        x: vec![f32::NAN; a.node_feat.len()],
        edge_index: vec![-7; a.edge_index.len()],
        edge_attr: vec![f32::NAN; a.edge_attr.len()],
        legal_offsets: vec![-7; a.legal_offsets.len()],
        legal_node_gather: vec![-7; a.legal_node_gather.len()],
        node_offsets: vec![-7; a.node_offsets.len()],
        n_stones: vec![-7; a.n_stones.len()],
    }
}

fn pack(a: &GraphWireArrays, out: &mut Out, threads: usize) -> Result<(), PackError> {
    let w = WireRef {
        n_graphs: a.n_graphs,
        node_feat: &a.node_feat,
        edge_index: &a.edge_index,
        edge_attr: &a.edge_attr,
        node_offsets: &a.node_offsets,
        edge_offsets: &a.edge_offsets,
        legal_offsets: &a.legal_offsets,
        legal_node_gather: &a.legal_node_gather,
        policy_dst_slot: &a.policy_dst_slot,
        n_nodes_checksum: &a.n_nodes_checksum,
        n_stones: &a.n_stones,
        window_center: &a.window_center,
        current_player: &a.current_player,
    };
    let mut o = PackOut {
        x: &mut out.x,
        edge_index: &mut out.edge_index,
        edges: EdgeOut::Attr(&mut out.edge_attr),
        legal_offsets: &mut out.legal_offsets,
        legal_node_gather: &mut out.legal_node_gather,
        node_offsets: &mut out.node_offsets,
        n_stones: &mut out.n_stones,
    };
    pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, threads)
}

fn kind_of(a: &GraphWireArrays) -> StructuralKind {
    let mut out = out_for(a);
    match pack(a, &mut out, 1) {
        Err(PackError::Contract { kind, .. }) => kind,
        other => panic!("expected a contract error, got {other:?}"),
    }
}

#[test]
fn a_clean_wire_packs_every_device_array_byte_equal() {
    let a = three();
    let mut out = out_for(&a);
    pack(&a, &mut out, 1).expect("a clean wire packs");
    assert_eq!(
        out.x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        a.node_feat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(out.edge_index, a.edge_index);
    assert_eq!(
        out.edge_attr
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        a.edge_attr.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(out.legal_offsets, a.legal_offsets);
    assert_eq!(out.legal_node_gather, a.legal_node_gather);
    assert_eq!(out.node_offsets, a.node_offsets);
    assert_eq!(
        out.n_stones,
        a.n_stones.iter().map(|&s| i64::from(s)).collect::<Vec<_>>()
    );
}

#[test]
fn the_pack_is_the_same_at_every_thread_count() {
    let a = many();
    let mut serial = out_for(&a);
    pack(&a, &mut serial, 1).expect("serial");
    for threads in [2, 3, 8, 64] {
        let mut par = out_for(&a);
        pack(&a, &mut par, threads).expect("parallel");
        assert_eq!(par.edge_index, serial.edge_index, "threads={threads}");
        assert_eq!(
            par.edge_attr
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            serial
                .edge_attr
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "threads={threads}"
        );
    }
}

#[test]
fn a_parallel_pack_still_reports_a_boundary_crossing() {
    let mut a = many();
    let e = a.edge_index.len() / 2;
    // The last graph's first src row moved into graph 0's node range: in bounds, wrong graph.
    let last_edge = usize::try_from(a.edge_offsets[63]).expect("offset");
    a.edge_index[last_edge] = 0;
    assert!(last_edge < e);
    let mut out = out_for(&a);
    match pack(&a, &mut out, 3) {
        Err(PackError::Contract { kind, .. }) => {
            assert_eq!(kind, StructuralKind::EdgeCrossesGraphBoundary);
        }
        other => panic!("expected EdgeCrossesGraphBoundary, got {other:?}"),
    }
}

#[test]
fn a_short_per_graph_array_is_a_batch_count_mismatch() {
    let mut a = three();
    a.n_stones.pop();
    assert_eq!(kind_of(&a), StructuralKind::BatchCountMismatch);
    let mut b = three();
    b.policy_dst_slot.pop();
    assert_eq!(kind_of(&b), StructuralKind::BatchCountMismatch);
}

#[test]
fn a_bad_offset_array_is_non_monotonic() {
    let mut a = three();
    a.node_offsets[0] = 1;
    assert_eq!(kind_of(&a), StructuralKind::OffsetsNonMonotonic);
    let mut b = three();
    *b.legal_offsets.last_mut().expect("B+1") += 1;
    assert_eq!(kind_of(&b), StructuralKind::OffsetsNonMonotonic);
    let mut c = three();
    c.edge_offsets[2] = c.edge_offsets[1] - 1;
    assert_eq!(kind_of(&c), StructuralKind::OffsetsNonMonotonic);
}

#[test]
fn a_node_count_that_disagrees_with_its_checksum_is_refused() {
    let mut a = three();
    a.n_nodes_checksum[1] += 1;
    assert_eq!(kind_of(&a), StructuralKind::NodeCountChecksum);
    let mut b = three();
    b.n_stones[0] = u16::try_from(b.n_nodes_checksum[0]).expect("small");
    assert_eq!(kind_of(&b), StructuralKind::NodeCountChecksum);
}

#[test]
fn an_edge_row_outside_the_batch_is_out_of_bounds_before_any_boundary_check() {
    let mut a = three();
    let n = *a.node_offsets.last().expect("B+1");
    a.edge_index[0] = n;
    assert_eq!(kind_of(&a), StructuralKind::EdgeIndexOutOfBounds);
    let mut b = three();
    let e = b.edge_index.len() / 2;
    b.edge_index[e] = -1;
    assert_eq!(kind_of(&b), StructuralKind::EdgeIndexOutOfBounds);
}

#[test]
fn a_gather_row_in_another_graph_crosses() {
    let mut a = three();
    let first_of_graph_1 = usize::try_from(a.legal_offsets[1]).expect("offset");
    a.legal_node_gather[first_of_graph_1] = 0;
    assert_eq!(kind_of(&a), StructuralKind::ScatterGatherCrossesGraph);
    let mut b = three();
    b.legal_node_gather[0] = -1;
    assert_eq!(kind_of(&b), StructuralKind::ScatterGatherCrossesGraph);
}

#[test]
fn a_slot_past_the_window_or_a_stray_negative_is_out_of_bounds() {
    let mut a = three();
    a.policy_dst_slot[0] = 362;
    assert_eq!(kind_of(&a), StructuralKind::ScatterSlotOutOfBounds);
    let mut b = three();
    b.policy_dst_slot[0] = -2;
    assert_eq!(kind_of(&b), StructuralKind::ScatterSlotOutOfBounds);
}

#[test]
fn two_legal_nodes_of_one_graph_on_one_slot_alias() {
    let mut a = three();
    a.policy_dst_slot[1] = a.policy_dst_slot[0];
    assert_ne!(a.policy_dst_slot[0], -1);
    assert_eq!(kind_of(&a), StructuralKind::ScatterSlotAliasing);
}

#[test]
fn the_off_window_sentinel_never_aliases() {
    let mut a = three();
    a.policy_dst_slot[0] = -1;
    a.policy_dst_slot[1] = -1;
    let mut out = out_for(&a);
    pack(&a, &mut out, 1).expect("two sentinels are not an alias");
}

#[test]
fn a_gather_that_does_not_ascend_is_refused_by_name() {
    let mut a = three();
    a.legal_node_gather.swap(0, 1);
    a.policy_dst_slot.swap(0, 1);
    assert_eq!(kind_of(&a), StructuralKind::GatherNotStrictlyIncreasing);
}

#[test]
fn an_empty_legal_set_is_refused() {
    // One stone and the dummy, no legal node and no edge: every earlier check holds.
    let a = GraphWireArrays {
        contract_version: 1,
        builder_impl: 1,
        n_graphs: 1,
        node_feat: vec![0.0; 2 * 11],
        node_coords: vec![0; 4],
        edge_index: vec![],
        edge_attr: vec![],
        node_offsets: vec![0, 2],
        edge_offsets: vec![0, 0],
        legal_offsets: vec![0, 0],
        legal_node_gather: vec![],
        policy_dst_slot: vec![],
        n_nodes_checksum: vec![2],
        n_stones: vec![1],
        window_center: vec![0, 0],
        current_player: vec![1],
    };
    assert_eq!(kind_of(&a), StructuralKind::EmptyLegalSet);
}

#[test]
fn the_earlier_check_wins_when_two_fail() {
    let mut a = three();
    a.n_nodes_checksum[0] += 1;
    a.policy_dst_slot[0] = 400;
    assert_eq!(kind_of(&a), StructuralKind::NodeCountChecksum);
}

#[test]
fn an_output_the_caller_sized_wrong_is_a_caller_error_not_a_panic() {
    let a = three();
    let mut out = out_for(&a);
    out.edge_attr.pop();
    assert!(matches!(pack(&a, &mut out, 2), Err(PackError::Caller(_))));
}

fn pack_coded(a: &GraphWireArrays, codes: &mut [u8], threads: usize) -> Result<(), PackError> {
    let mut out = out_for(a);
    let w = WireRef {
        n_graphs: a.n_graphs,
        node_feat: &a.node_feat,
        edge_index: &a.edge_index,
        edge_attr: &a.edge_attr,
        node_offsets: &a.node_offsets,
        edge_offsets: &a.edge_offsets,
        legal_offsets: &a.legal_offsets,
        legal_node_gather: &a.legal_node_gather,
        policy_dst_slot: &a.policy_dst_slot,
        n_nodes_checksum: &a.n_nodes_checksum,
        n_stones: &a.n_stones,
        window_center: &a.window_center,
        current_player: &a.current_player,
    };
    let mut o = PackOut {
        x: &mut out.x,
        edge_index: &mut out.edge_index,
        edges: EdgeOut::Code {
            codes,
            win_length: 6,
        },
        legal_offsets: &mut out.legal_offsets,
        legal_node_gather: &mut out.legal_node_gather,
        node_offsets: &mut out.node_offsets,
        n_stones: &mut out.n_stones,
    };
    pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, threads)
}

#[test]
fn a_coded_pack_writes_each_edges_vocabulary_code_at_every_thread_count() {
    let a = many();
    let want: Vec<u8> = a
        .edge_attr
        .chunks(EDGE_DIM)
        .map(|row| mantis_graph::edge_code(row, 6).expect("a built row is in the vocabulary"))
        .collect();
    for threads in [1, 2, 3, 8] {
        let mut codes = vec![255u8; want.len()];
        pack_coded(&a, &mut codes, threads).expect("a clean wire packs coded");
        assert_eq!(codes, want, "threads={threads}");
    }
}

#[test]
fn a_coded_pack_refuses_a_row_outside_the_vocabulary_by_check_14s_class() {
    let mut a = three();
    let e = a.edge_attr.len() / EDGE_DIM;
    a.edge_attr[(e - 1) * EDGE_DIM + 3] = 2.5;
    let mut codes = vec![0u8; e];
    match pack_coded(&a, &mut codes, 3) {
        Err(PackError::Contract { kind, message }) => {
            assert_eq!(kind, StructuralKind::EdgeAttrGeometryMismatch);
            assert!(message.contains(&format!("edge {}", e - 1)), "{message}");
        }
        other => panic!("expected EdgeAttrGeometryMismatch, got {other:?}"),
    }
}

#[test]
fn a_coded_pack_sized_for_attrs_is_a_caller_error() {
    let a = three();
    let mut codes = vec![0u8; a.edge_attr.len()];
    assert!(matches!(
        pack_coded(&a, &mut codes, 1),
        Err(PackError::Caller(_))
    ));
}

#[test]
fn a_structural_error_outranks_a_row_outside_the_vocabulary() {
    let mut a = three();
    let e = a.edge_attr.len() / EDGE_DIM;
    a.edge_attr[3] = 2.5;
    a.legal_node_gather[0] = -1;
    let mut codes = vec![0u8; e];
    match pack_coded(&a, &mut codes, 2) {
        Err(PackError::Contract { kind, .. }) => {
            assert_eq!(kind, StructuralKind::ScatterGatherCrossesGraph);
        }
        other => panic!("expected ScatterGatherCrossesGraph, got {other:?}"),
    }
}

#[test]
fn the_lowest_edge_outside_the_vocabulary_is_named_whatever_the_thread_split() {
    let mut a = many();
    let e = a.edge_attr.len() / EDGE_DIM;
    let late = usize::try_from(a.edge_offsets[60]).expect("offset");
    let early = usize::try_from(a.edge_offsets[3]).expect("offset") + 1;
    for edge in [late, early] {
        a.edge_attr[edge * EDGE_DIM + 3] = 0.5;
    }
    for threads in [1, 4, 8] {
        let mut codes = vec![0u8; e];
        match pack_coded(&a, &mut codes, threads) {
            Err(PackError::Contract { kind, message }) => {
                assert_eq!(kind, StructuralKind::EdgeAttrGeometryMismatch);
                assert!(
                    message.starts_with(&format!("edge {early} ")),
                    "threads={threads}: {message}"
                );
            }
            other => panic!("expected EdgeAttrGeometryMismatch, got {other:?}"),
        }
    }
}

#[test]
fn a_coded_pack_without_a_vocabulary_is_a_wiring_break_not_a_wire_defect() {
    let a = three();
    let mut out = out_for(&a);
    let mut codes = vec![0u8; a.edge_attr.len() / EDGE_DIM];
    let w = WireRef {
        n_graphs: a.n_graphs,
        node_feat: &a.node_feat,
        edge_index: &a.edge_index,
        edge_attr: &a.edge_attr,
        node_offsets: &a.node_offsets,
        edge_offsets: &a.edge_offsets,
        legal_offsets: &a.legal_offsets,
        legal_node_gather: &a.legal_node_gather,
        policy_dst_slot: &a.policy_dst_slot,
        n_nodes_checksum: &a.n_nodes_checksum,
        n_stones: &a.n_stones,
        window_center: &a.window_center,
        current_player: &a.current_player,
    };
    let mut o = PackOut {
        x: &mut out.x,
        edge_index: &mut out.edge_index,
        edges: EdgeOut::Code {
            codes: &mut codes,
            win_length: 16,
        },
        legal_offsets: &mut out.legal_offsets,
        legal_node_gather: &mut out.legal_node_gather,
        node_offsets: &mut out.node_offsets,
        n_stones: &mut out.n_stones,
    };
    assert!(matches!(
        pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, 1),
        Err(PackError::Caller(_))
    ));
}
