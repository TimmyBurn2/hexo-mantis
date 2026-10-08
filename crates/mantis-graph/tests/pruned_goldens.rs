//! The pruned builder's goldens: the kept build less exactly its empty pairs, in walk order, on every fixture case and r8 position.

#[allow(dead_code)]
mod common;

#[path = "common/positions.rs"]
mod positions;

use mantis_graph::{
    build_axis_graph, AxisGraph, BuildParams, EmptyEdges, StoneList, EDGE_FEAT_DIM,
};
use sha2::{Digest, Sha256};

/// sha256 over every case's pruned `edge_src`, `edge_dst` and `edge_attr` bytes, in case order.
const PRUNED_EDGES_SHA256: &str =
    "ed429535190b45fff9bfba28fbf66959369766c094b4d97b38a85f5957dd4e94";
/// The same digest over the recorded radius-8 positions, in file order.
const PRUNED_R8_EDGES_SHA256: &str =
    "b7beda3fc74e2535d12dd8e2b7ed696cc9764f2aca56bf80914a5416fe4d9d45";

fn cases() -> Vec<common::CaseInput> {
    let root = common::fixture_root();
    common::verify_fixture_root(&root).unwrap_or_else(|e| panic!("{e}"));
    let cases = common::read_inputs_bin(&root.join("inputs.bin")).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(cases.len(), 1696, "inputs.bin case count");
    cases
}

fn joins_two_empties(g: &AxisGraph, e: usize) -> bool {
    let (s, d) = (g.edge_index.src[e] as usize, g.edge_index.dst[e] as usize);
    g.legal_mask[s] && g.legal_mask[d]
}

/// The recorded radius-8 positions as `(stones, params)` under `empty_edges`.
fn r8_positions(empty_edges: EmptyEdges) -> Vec<(StoneList, BuildParams)> {
    let recorded = positions::read_positions_r8().unwrap_or_else(|e| panic!("{e}"));
    recorded
        .into_iter()
        .map(|p| {
            let params = BuildParams {
                radius: 8,
                current_player: p.to_move,
                moves_remaining: p.moves_remaining,
                empty_edges,
                ..BuildParams::V1_GEOMETRY
            };
            (StoneList { stones: p.stones }, params)
        })
        .collect()
}

/// Asserts `pruned` is `kept` with exactly its empty pairs removed; returns the two edge counts.
fn assert_kept_less_empty_pairs(
    label: &str,
    kept: &AxisGraph,
    pruned: &AxisGraph,
) -> (usize, usize) {
    let (kf, pf) = (
        common::canonical_fields(kept),
        common::canonical_fields(pruned),
    );
    for (k, p) in kf.iter().zip(&pf) {
        if !k.name.starts_with("edge_") {
            assert_eq!(
                k.payload, p.payload,
                "{label}: node field {} moved under pruning",
                k.name
            );
        }
    }
    let rows = |g: &AxisGraph, e: usize| {
        let attr = &g.edge_attr.0[e * EDGE_FEAT_DIM..(e + 1) * EDGE_FEAT_DIM];
        (g.edge_index.src[e], g.edge_index.dst[e], attr.to_vec())
    };
    let kept_rows: Vec<_> = (0..kept.num_edges())
        .filter(|&e| !joins_two_empties(kept, e))
        .map(|e| rows(kept, e))
        .collect();
    let pruned_rows: Vec<_> = (0..pruned.num_edges()).map(|e| rows(pruned, e)).collect();
    assert_eq!(
        pruned_rows, kept_rows,
        "{label}: pruned edges are not the kept edges less the empty pairs"
    );
    (kept.num_edges(), pruned.num_edges())
}

#[test]
fn the_pruned_build_is_the_kept_build_less_its_empty_pairs_on_every_case() {
    let (mut kept_total, mut pruned_total) = (0usize, 0usize);
    for case in &cases() {
        let kept = common::build_case(case);
        let pruned = common::build_case_with(case, EmptyEdges::Pruned);
        let (k, p) =
            assert_kept_less_empty_pairs(&format!("case {}", case.case_id), &kept, &pruned);
        kept_total += k;
        pruned_total += p;
    }
    assert!(
        pruned_total < kept_total,
        "no case had an empty pair to prune; the test measured nothing"
    );
}

#[test]
fn the_pruned_build_is_the_kept_build_less_its_empty_pairs_at_radius_eight() {
    let (kept_set, pruned_set) = (
        r8_positions(EmptyEdges::Kept),
        r8_positions(EmptyEdges::Pruned),
    );
    assert_eq!(pruned_set.len(), positions::POSITIONS_R8_COUNT);
    let mut h = Sha256::new();
    for (i, ((ks, kp), (ps, pp))) in kept_set.iter().zip(&pruned_set).enumerate() {
        let (kept, pruned) = (build_axis_graph(ks, kp), build_axis_graph(ps, pp));
        assert_kept_less_empty_pairs(&format!("r8 position {i}"), &kept, &pruned);
        for f in common::canonical_fields(&pruned)
            .iter()
            .filter(|f| f.name.starts_with("edge_"))
        {
            h.update(&f.payload);
        }
    }
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        got, PRUNED_R8_EDGES_SHA256,
        "the pruned builder's radius-8 edge bytes moved"
    );
}

#[test]
fn the_pruned_edge_bytes_hash_to_the_pinned_digest() {
    let mut h = Sha256::new();
    for case in &cases() {
        let g = common::build_case_with(case, EmptyEdges::Pruned);
        for f in common::canonical_fields(&g)
            .iter()
            .filter(|f| f.name.starts_with("edge_"))
        {
            h.update(&f.payload);
        }
    }
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        got, PRUNED_EDGES_SHA256,
        "the pruned builder's edge bytes moved"
    );
}
