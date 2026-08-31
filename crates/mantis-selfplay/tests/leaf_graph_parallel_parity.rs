//! NIGHTRUN-1 E1 — the parallel leaf-graph build is BIT-IDENTICAL to the serial one.
//!
//! `submit_graphs_and_wait_ls` built its leaves in a serial loop on the calling thread while
//! holding the GIL. The NIGHTRUN-1 Leg 1 profile puts **95.3 % of the whole eval game loop**
//! inside that call, and an N-sweep at a real 64-move board separates it into a 5.2 ms/leaf
//! slope against a 2.4 ms round-trip intercept — so the eval path IS this loop.
//!
//! WHAT MUST NOT MOVE, and why each half is here:
//!
//! 1. **The graphs themselves.** Every array the wire carries — node features, coords,
//!    edge index, edge attributes, the window centre — is what the net sees. A build that
//!    is correct but different is a silently re-encoded position.
//! 2. **THE ORDER.** The caller zips the returned graphs against `positions` to recover each
//!    leaf's window centre (`inference.rs`), and `MCTSTree.expand_and_backup_ls_graph`
//!    cross-checks priors in that frame. Results returned in COMPLETION order rather than
//!    index order would mis-pair every leaf against its policy while every length still
//!    checked out — the same failure mode B1's parity test exists for, one layer up.
//! 3. **The error, and WHICH error.** The serial path returns the first failure in index
//!    order. A chunked build that surfaced whichever worker failed first would name a
//!    different position on a bad input, and a build error is a diagnostic.

use mantis_graph::AxisGraph;
use mantis_selfplay::queues::{build_leaf_graph, build_leaf_graphs_batch, LeafRequest};

const WIN_LENGTH: u8 = 6;
const RADIUS: u16 = 6;
const TRUNK: i32 = 19;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` distinct positions of VARYING stone count, so a chunked split is non-trivial and the
/// chunks are not interchangeable. A corpus of identical positions cannot see a reordering.
fn corpus(n: usize) -> Vec<LeafRequest> {
    let mut s = 0x51AF_0E01_u64;
    (0..n)
        .map(|i| {
            let n_stones = 4 + (splitmix64(&mut s) % 60) as i64;
            let stones: Vec<(i64, i64, i64)> = (0..n_stones)
                .map(|q| {
                    let r = (splitmix64(&mut s) % 9) as i64 - 4;
                    (q % 13, r, if (q + i as i64) % 2 == 0 { 1 } else { -1 })
                })
                .collect();
            let current_player = if i % 2 == 0 { 1 } else { -1 };
            (stones, current_player, (128 - (i as i64 % 100)).max(0))
        })
        .collect()
}

fn assert_same(a: &AxisGraph, b: &AxisGraph, idx: usize) {
    // EVERY field, enumerated rather than a derived `PartialEq` the struct does not carry:
    // a field added later and not compared here is a field this file stops guarding.
    assert_eq!(a.node_feat, b.node_feat, "node_feat differs at {idx}");
    assert_eq!(a.edge_index, b.edge_index, "edge_index differs at {idx}");
    assert_eq!(a.edge_attr, b.edge_attr, "edge_attr differs at {idx}");
    assert_eq!(a.legal_mask, b.legal_mask, "legal_mask differs at {idx}");
    assert_eq!(a.stone_mask, b.stone_mask, "stone_mask differs at {idx}");
    assert_eq!(
        a.policy_scatter_index, b.policy_scatter_index,
        "policy_scatter_index differs at {idx}"
    );
    assert_eq!(a.node_coords, b.node_coords, "node_coords differs at {idx}");
    assert_eq!(
        a.legal_node_gather, b.legal_node_gather,
        "legal_node_gather differs at {idx}"
    );
    assert_eq!(a.n_stones, b.n_stones, "n_stones differs at {idx}");
    assert_eq!(
        a.n_nodes_checksum, b.n_nodes_checksum,
        "n_nodes_checksum differs at {idx}"
    );
    assert_eq!(a.window_center, b.window_center, "window_center differs at {idx}");
    assert_eq!(a.current_player, b.current_player, "current_player differs at {idx}");
    assert_eq!(a.builder_impl, b.builder_impl, "builder_impl differs at {idx}");
}

#[test]
fn parallel_leaf_build_is_bit_identical_to_serial_at_every_width() {
    let positions = corpus(37);
    let serial = build_leaf_graphs_batch(&positions, WIN_LENGTH, RADIUS, TRUNK, 1)
        .expect("serial build");
    assert_eq!(serial.len(), positions.len());
    // Widths that do and do NOT divide the corpus evenly: an off-by-one in `div_ceil`
    // chunking shows up only on a ragged split.
    for threads in [2usize, 3, 4, 5, 8, 12, 37, 64] {
        let par = build_leaf_graphs_batch(&positions, WIN_LENGTH, RADIUS, TRUNK, threads)
            .unwrap_or_else(|e| panic!("threaded build at {threads} failed: {e}"));
        assert_eq!(par.len(), serial.len(), "length differs at {threads} threads");
        for (i, (a, b)) in serial.iter().zip(par.iter()).enumerate() {
            assert_same(a, b, i);
        }
    }
}

#[test]
fn the_batch_builder_agrees_with_the_one_shot_builder_position_by_position() {
    // The batch entry point must not become a second builder. Driven against
    // `build_leaf_graph` itself so a divergence in either path is visible here.
    let positions = corpus(11);
    let batch = build_leaf_graphs_batch(&positions, WIN_LENGTH, RADIUS, TRUNK, 4)
        .expect("threaded build");
    for (i, (stones, cp, mr)) in positions.iter().enumerate() {
        let one = build_leaf_graph(stones, *cp, *mr, WIN_LENGTH, RADIUS, TRUNK)
            .expect("one-shot build");
        assert_same(&one, &batch[i], i);
    }
}

#[test]
fn the_order_is_index_order_and_a_reordering_would_be_visible() {
    // The positive control for the assertion above: these graphs are pairwise DISTINCT, so
    // "identical in order" is a real constraint rather than one satisfied by every
    // permutation. Without this row the parity test would pass over a shuffled result.
    let positions = corpus(9);
    let graphs = build_leaf_graphs_batch(&positions, WIN_LENGTH, RADIUS, TRUNK, 3)
        .expect("threaded build");
    for i in 0..graphs.len() {
        for j in (i + 1)..graphs.len() {
            assert!(
                graphs[i].node_feat != graphs[j].node_feat
                    || graphs[i].node_coords != graphs[j].node_coords,
                "corpus positions {i} and {j} build to the same graph, so this file's \
                 parity assertions are satisfiable by a permutation"
            );
        }
    }
}

#[test]
fn a_bad_position_returns_the_same_error_serial_and_threaded() {
    // `current_player = 0` is out of range; the guard is in `build_leaf_graph`. The error
    // must survive the chunked path AND name the same position.
    let mut positions = corpus(20);
    positions[13].1 = 0;
    let serial = build_leaf_graphs_batch(&positions, WIN_LENGTH, RADIUS, TRUNK, 1);
    let threaded = build_leaf_graphs_batch(&positions, WIN_LENGTH, RADIUS, TRUNK, 6);
    let serial_err = serial.expect_err("serial build must refuse current_player 0");
    let threaded_err = threaded.expect_err("threaded build must refuse current_player 0");
    assert!(serial_err.contains("current_player"), "{serial_err}");
    assert_eq!(serial_err, threaded_err, "the threaded path renamed the failure");
}

#[test]
fn an_empty_batch_is_empty_at_every_width() {
    for threads in [0usize, 1, 8] {
        let out = build_leaf_graphs_batch(&[], WIN_LENGTH, RADIUS, TRUNK, threads)
            .expect("empty build");
        assert!(out.is_empty());
    }
}
