// >300 justify (R8): every structural kind, the coded and padded packs and the thread split are pinned against one
// fixture set, so a kind added to the pack lands beside the cases that prove it.

//! The pack's byte parity against the fused wire and every structural kind, in the contract's order.

use super::*;
use crate::queues::{build_leaf_graph, GraphWire, GraphWireArrays};

const NODE_DIM: usize = 11;
const EDGE_DIM: usize = 5;
const TRUNK: i32 = 19;

fn wire(positions: &[&[(i64, i64, i64)]]) -> GraphWireArrays {
    let graphs: Vec<_> = positions
        .iter()
        .map(|stones| {
            build_leaf_graph(stones, 1, 2, 6, 6, 19, mantis_graph::EmptyEdges::Kept)
                .expect("a legal test position")
        })
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
        node_coords: &a.node_coords,
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
    pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, TRUNK, threads, None)
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

/// `a` with graph `g` centred far off its cells, so each of its slots is honestly the sentinel.
fn off_window(mut a: GraphWireArrays, g: usize) -> GraphWireArrays {
    let rows = usize::try_from(a.legal_offsets[g]).expect("offset")
        ..usize::try_from(a.legal_offsets[g + 1]).expect("offset");
    a.policy_dst_slot[rows].fill(-1);
    a.window_center[2 * g] = i32::MIN;
    a
}

#[test]
fn the_off_window_sentinel_never_aliases() {
    let a = off_window(three(), 0);
    assert!(a.legal_offsets[1] >= 2, "two sentinels in one graph");
    let mut out = out_for(&a);
    pack(&a, &mut out, 1).expect("two sentinels are not an alias");
}

fn message_of(a: &GraphWireArrays) -> String {
    let mut out = out_for(a);
    match pack(a, &mut out, 1) {
        Err(PackError::Contract { message, .. }) => message,
        other => panic!("expected a contract error, got {other:?}"),
    }
}

#[test]
fn a_gather_row_on_a_stone_or_the_dummy_is_not_a_legal_node() {
    let mut a = three();
    a.legal_node_gather[0] = a.node_offsets[0];
    assert_eq!(kind_of(&a), StructuralKind::GatherNotLegalNode);
    assert!(message_of(&a).contains("stone or dummy node"));
    let mut b = three();
    let last = b.legal_node_gather.len() - 1;
    b.legal_node_gather[last] = b.node_offsets[3] - 1;
    assert_eq!(kind_of(&b), StructuralKind::GatherNotLegalNode);
}

#[test]
fn a_legal_count_off_its_checksum_is_not_a_legal_node() {
    let mut a = three();
    a.n_stones[0] -= 1;
    assert_eq!(kind_of(&a), StructuralKind::GatherNotLegalNode);
    assert!(
        message_of(&a).contains("graph 0 holds"),
        "{}",
        message_of(&a)
    );
}

#[test]
fn a_slot_off_its_cells_canonical_window_slot_is_refused() {
    let mut a = three();
    a.window_center[2] += 1;
    assert_eq!(kind_of(&a), StructuralKind::ScatterSlotCanonicalMismatch);
    assert!(message_of(&a).contains("about graph 1's centre"));
    let mut b = three();
    let inside = b
        .policy_dst_slot
        .iter()
        .position(|&s| s >= 0)
        .expect("an in-window slot");
    b.policy_dst_slot[inside] = -1;
    assert_eq!(kind_of(&b), StructuralKind::ScatterSlotCanonicalMismatch);
}

#[test]
fn checks_15_and_16_run_after_13_and_after_the_vocabulary_in_that_order() {
    let mut order = three();
    order.window_center[0] += 1;
    order.legal_node_gather.swap(0, 1);
    order.policy_dst_slot.swap(0, 1);
    assert_eq!(kind_of(&order), StructuralKind::GatherNotStrictlyIncreasing);
    let mut count = three();
    count.window_center[0] += 1;
    count.n_stones[2] -= 1;
    assert_eq!(kind_of(&count), StructuralKind::GatherNotLegalNode);
    let mut vocab = three();
    vocab.window_center[0] += 1;
    vocab.edge_attr[3] = 2.5;
    let mut codes = vec![0u8; vocab.edge_attr.len() / EDGE_DIM];
    match pack_coded(&vocab, &mut codes, 1) {
        Err(PackError::Contract { kind, .. }) => {
            assert_eq!(kind, StructuralKind::EdgeAttrGeometryMismatch);
        }
        other => panic!("expected EdgeAttrGeometryMismatch, got {other:?}"),
    }
}

#[test]
fn a_padded_pack_checks_the_real_graphs_before_its_padding() {
    let mut a = three();
    a.window_center[4] -= 1;
    let (b, n, e, lg) = (
        a.n_graphs,
        a.node_feat.len() / NODE_DIM,
        a.edge_attr.len() / EDGE_DIM,
        a.legal_node_gather.len(),
    );
    let pad = PadTo {
        n_graphs: b + 1,
        n_nodes: n + 2,
        n_edges: e,
        n_legal: lg + 1,
    };
    let mut out = Out {
        x: vec![0.0; pad.n_nodes * NODE_DIM],
        edge_index: vec![0; 2 * pad.n_edges],
        edge_attr: vec![0.0; pad.n_edges * EDGE_DIM],
        legal_offsets: vec![0; pad.n_graphs + 1],
        legal_node_gather: vec![0; pad.n_legal],
        node_offsets: vec![0; pad.n_graphs + 1],
        n_stones: vec![0; pad.n_graphs],
    };
    let w = WireRef {
        n_graphs: a.n_graphs,
        node_feat: &a.node_feat,
        node_coords: &a.node_coords,
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
    match pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, TRUNK, 1, Some(pad)) {
        Err(PackError::Contract { kind, .. }) => {
            assert_eq!(kind, StructuralKind::ScatterSlotCanonicalMismatch);
        }
        other => panic!("expected ScatterSlotCanonicalMismatch, got {other:?}"),
    }
}

#[test]
fn the_wrong_trunk_moves_every_canonical_slot() {
    let a = three();
    let mut out = out_for(&a);
    let w = WireRef {
        n_graphs: a.n_graphs,
        node_feat: &a.node_feat,
        node_coords: &a.node_coords,
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
    assert!(matches!(
        pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, 25, 1, None),
        Err(PackError::Contract {
            kind: StructuralKind::ScatterSlotCanonicalMismatch,
            ..
        })
    ));
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
        node_coords: &a.node_coords,
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
    pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, TRUNK, threads, None)
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
        node_coords: &a.node_coords,
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
        pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, TRUNK, 1, None),
        Err(PackError::Caller(_))
    ));
}

#[test]
fn a_padded_pack_writes_the_real_arrays_then_one_padding_graph() {
    let a = three();
    let (b, n, e, lg) = (
        a.n_graphs,
        a.node_feat.len() / NODE_DIM,
        a.edge_attr.len() / EDGE_DIM,
        a.legal_node_gather.len(),
    );
    let pad = PadTo {
        n_graphs: b + 3,
        n_nodes: n + 10,
        n_edges: e + 25,
        n_legal: lg + 7,
    };
    let mut real_codes = vec![0u8; e];
    pack_coded(&a, &mut real_codes, 1).expect("unpadded");
    let mut x = vec![f32::NAN; pad.n_nodes * NODE_DIM];
    let mut ei = vec![-7i64; 2 * pad.n_edges];
    let mut codes = vec![255u8; pad.n_edges];
    let mut lo = vec![-7i64; pad.n_graphs + 1];
    let mut lgat = vec![-7i64; pad.n_legal];
    let mut no = vec![-7i64; pad.n_graphs + 1];
    let mut ns = vec![-7i64; pad.n_graphs];
    let w = WireRef {
        n_graphs: a.n_graphs,
        node_feat: &a.node_feat,
        node_coords: &a.node_coords,
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
        x: &mut x,
        edge_index: &mut ei,
        edges: EdgeOut::Code {
            codes: &mut codes,
            win_length: 6,
        },
        legal_offsets: &mut lo,
        legal_node_gather: &mut lgat,
        node_offsets: &mut no,
        n_stones: &mut ns,
    };
    pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, TRUNK, 2, Some(pad)).expect("a padded pack");
    let as_i64 = |v: usize| i64::try_from(v).expect("small");
    assert_eq!(
        x[..n * NODE_DIM]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        a.node_feat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert!(x[n * NODE_DIM..].iter().all(|&v| v.to_bits() == 0));
    assert_eq!(&ei[..e], &a.edge_index[..e]);
    assert_eq!(&ei[pad.n_edges..pad.n_edges + e], &a.edge_index[e..]);
    for j in 0..25 {
        let sink = as_i64(n + j % 10);
        assert_eq!(
            (ei[e + j], ei[pad.n_edges + e + j]),
            (sink, sink),
            "padding edge {j}"
        );
    }
    assert_eq!(&codes[..e], &real_codes[..]);
    assert!(codes[e..].iter().all(|&c| c == 0));
    assert_eq!(&lo[..=b], &a.legal_offsets[..]);
    assert_eq!(&lo[b + 1..], &[as_i64(lg), as_i64(lg), as_i64(lg + 7)]);
    assert_eq!(&no[..=b], &a.node_offsets[..]);
    assert_eq!(&no[b + 1..], &[as_i64(n), as_i64(n), as_i64(n + 10)]);
    assert_eq!(&lgat[..lg], &a.legal_node_gather[..]);
    assert_eq!(
        &lgat[lg..],
        &(0..7).map(|k| as_i64(n + k % 10)).collect::<Vec<_>>()[..]
    );
    assert!(ns[b..].iter().all(|&s| s == 0));
}

#[test]
fn a_pad_that_leaves_no_padding_node_or_graph_is_a_caller_error() {
    let a = three();
    let (b, n, e, lg) = (
        a.n_graphs,
        a.node_feat.len() / NODE_DIM,
        a.edge_attr.len() / EDGE_DIM,
        a.legal_node_gather.len(),
    );
    for pad in [
        PadTo {
            n_graphs: b,
            n_nodes: n + 1,
            n_edges: e,
            n_legal: lg,
        },
        PadTo {
            n_graphs: b + 1,
            n_nodes: n,
            n_edges: e,
            n_legal: lg,
        },
        PadTo {
            n_graphs: b + 1,
            n_nodes: n + 1,
            n_edges: e - 1,
            n_legal: lg,
        },
    ] {
        let mut out = out_for(&a);
        let w = WireRef {
            n_graphs: a.n_graphs,
            node_feat: &a.node_feat,
            node_coords: &a.node_coords,
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
        assert!(
            matches!(
                pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, TRUNK, 1, Some(pad)),
                Err(PackError::Caller(_))
            ),
            "{pad:?}"
        );
    }
}

#[test]
fn padding_legal_entries_wrap_over_the_padding_nodes_when_they_outnumber_them() {
    let a = three();
    let (b, n, e, lg) = (
        a.n_graphs,
        a.node_feat.len() / NODE_DIM,
        a.edge_attr.len() / EDGE_DIM,
        a.legal_node_gather.len(),
    );
    let pad = PadTo {
        n_graphs: b + 1,
        n_nodes: n + 3,
        n_edges: e + 5,
        n_legal: lg + 7,
    };
    let mut out = Out {
        x: vec![f32::NAN; pad.n_nodes * NODE_DIM],
        edge_index: vec![-7; 2 * pad.n_edges],
        edge_attr: vec![f32::NAN; pad.n_edges * EDGE_DIM],
        legal_offsets: vec![-7; pad.n_graphs + 1],
        legal_node_gather: vec![-7; pad.n_legal],
        node_offsets: vec![-7; pad.n_graphs + 1],
        n_stones: vec![-7; pad.n_graphs],
    };
    let w = WireRef {
        n_graphs: a.n_graphs,
        node_feat: &a.node_feat,
        node_coords: &a.node_coords,
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
    pack_wire(&w, &mut o, NODE_DIM, EDGE_DIM, TRUNK, 1, Some(pad)).expect("a padded pack");
    let as_i64 = |v: usize| i64::try_from(v).expect("small");
    let tail: Vec<i64> = (0..7).map(|k| as_i64(n + k % 3)).collect();
    assert_eq!(&out.legal_node_gather[lg..], &tail[..]);
    assert!(out.edge_attr[e * EDGE_DIM..]
        .iter()
        .all(|&v| v.to_bits() == 0));
}
