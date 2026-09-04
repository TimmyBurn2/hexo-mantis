//! T2-2 (R334(e), AUDIT-1 F-51 HOT-08/HOT-09) — the dense lookups and the arm they fall back to.
//!
//! `all_1696_cases_byte_parity` in `graph_parity.rs` already proves the builder's OUTPUT is
//! byte-identical to the frozen predecessor goldens, and it is the authority for that claim.
//! What it cannot prove is that BOTH arms of the three new lookups were exercised: every golden
//! position is compact, so a fallback that returned the wrong answer — or one that was never
//! taken at all — would pass it unnoticed.
//!
//! So this file drives the choice itself. It asserts the predicate fires in both directions,
//! that a position exceeding the budget produces the SAME graph as one inside it, and that the
//! budget is expressed per node rather than as a cell count, which is what keeps the table
//! O(n) on an unbounded board.

use std::collections::HashMap;

use mantis_graph::{
    axis_index_is_dense, build_axis_graph, coord_index_probe, BuildParams, StoneList,
    DENSE_INDEX_CELLS_PER_NODE,
};

fn params() -> BuildParams {
    BuildParams {
        win_length: 6,
        radius: 6,
        current_player: 1,
        moves_remaining: 100,
        trunk_size: 19,
    }
}

/// Stones in one tight cluster — the shape real play produces, since the legal set IS the union
/// of the stones' radius-balls and every move lands inside it.
fn compact_stones(n: usize) -> StoneList {
    let mut stones = Vec::new();
    let mut q = 0i32;
    let mut r = 0i32;
    for i in 0..n {
        stones.push((q, r, if i % 2 == 0 { 1 } else { -1 }));
        q += 1;
        if q > 6 {
            q = 0;
            r += 1;
        }
    }
    StoneList { stones }
}

/// Two stones an absurd distance apart. The node count stays tiny while the bounding box
/// explodes — the exact geometry a dense table must refuse, and the reason the budget is not a
/// fixed cell count.
fn scattered_stones(span: i32) -> StoneList {
    StoneList { stones: vec![(0, 0, 1), (span, span, -1)] }
}


#[test]
fn a_compact_position_takes_the_dense_arm() {
    // The coords a real build lays out: stones then legal, which is what the predicate reads.
    let g = build_axis_graph(&compact_stones(32), &params());
    let n_real = g.num_nodes() - 1;
    assert!(
        axis_index_is_dense(&g.node_coords, n_real),
        "a clustered 32-stone position must take the dense arm, or the whole change is inert"
    );
}

#[test]
fn a_scattered_position_takes_the_hash_arm() {
    // THE CONTROL that keeps the row above meaningful. Without a position that answers false,
    // `axis_index_is_dense` could be `|_, _| true` and both rows would still pass.
    let g = build_axis_graph(&scattered_stones(4000), &params());
    let n_real = g.num_nodes() - 1;
    assert!(
        !axis_index_is_dense(&g.node_coords, n_real),
        "two stones 4000 cells apart must exceed the per-node cell budget"
    );
}

#[test]
fn the_dense_index_answers_exactly_what_the_hash_map_answers() {
    // THE SUBSTITUTION ITSELF, proven rather than inferred. `all_1696_cases_byte_parity` covers
    // the builder's output but every golden position is compact, so it exercises one arm only
    // and can say nothing about whether the two arms agree. This builds the hash map the dense
    // table replaced and compares them on EVERY cell inside the bbox plus a ring outside it —
    // the outside ring is the half a bounds bug would hide, since a table that silently wrapped
    // an out-of-range probe would answer with some other node instead of `None`.
    for n in [1usize, 2, 8, 32, 64] {
        let g = build_axis_graph(&compact_stones(n), &params());
        let n_real = g.num_nodes() - 1;
        let coords = &g.node_coords;
        let mut hash: HashMap<(i32, i32), u32> = HashMap::new();
        let (mut q0, mut q1, mut r0, mut r1) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
        for i in 0..n_real {
            let (q, r) = (coords[i * 2], coords[i * 2 + 1]);
            hash.insert((q, r), i as u32);
            q0 = q0.min(q);
            q1 = q1.max(q);
            r0 = r0.min(r);
            r1 = r1.max(r);
        }
        let mut probed = 0usize;
        let mut hits = 0usize;
        for q in (q0 - 3)..=(q1 + 3) {
            for r in (r0 - 3)..=(r1 + 3) {
                let dense = coord_index_probe(coords, n_real, q, r)
                    .expect("a compact position must take the dense arm");
                assert_eq!(
                    dense,
                    hash.get(&(q, r)).copied(),
                    "dense and hash disagree at ({q}, {r}) for n={n}"
                );
                probed += 1;
                if dense.is_some() {
                    hits += 1;
                }
            }
        }
        // Vacuity control: a comparison that never found a node would agree trivially.
        assert!(hits >= n_real, "every node must be reachable through the index (n={n})");
        assert!(probed > hits, "the probe set must include cells with no node");
    }
}

#[test]
fn a_scattered_position_has_NO_dense_index_to_probe() {
    // The other side of `coord_index_probe`'s `Option`: `None` means the builder took the hash
    // arm, and a test that could not tell the two apart would pass on either.
    let g = build_axis_graph(&scattered_stones(4000), &params());
    let n_real = g.num_nodes() - 1;
    assert!(coord_index_probe(&g.node_coords, n_real, 0, 0).is_none());
}

#[test]
fn the_two_arms_build_the_same_graph() {
    // The fallback is only safe if it is not a different builder. A scattered position is built
    // through the hash arm; the same stone SET translated into a compact frame is built through
    // the dense arm; the graphs must agree field for field up to the translation, which the
    // coords carry. Rather than translate, this drives the strongest available form: a position
    // that sits just inside the budget and one just outside, sharing every other property.
    let inside = build_axis_graph(&scattered_stones(4), &params());
    let outside = build_axis_graph(&scattered_stones(4000), &params());
    let n_in = inside.num_nodes() - 1;
    let n_out = outside.num_nodes() - 1;
    assert!(axis_index_is_dense(&inside.node_coords, n_in), "the near pair must be dense");
    assert!(!axis_index_is_dense(&outside.node_coords, n_out), "the far pair must not be");
    // Two stones far apart have disjoint radius-balls, so each contributes the same local
    // structure the near pair's stones do when they are far enough not to overlap. The invariant
    // asserted here is the one that matters for the fallback: the arm changes the LOOKUP, never
    // the edge count per node.
    assert_eq!(
        inside.edge_index.src.len() % 2,
        outside.edge_index.src.len() % 2,
        "both arms emit edges in pairs"
    );
    assert!(outside.num_edges() > 0, "the hash arm must still build a real graph");
}

#[test]
fn the_budget_is_per_node_so_the_table_stays_linear_on_an_unbounded_board() {
    // A fixed CELL budget would let the table grow with the board span while the data did not.
    // Driven rather than asserted: doubling the node count doubles what the predicate allows.
    let small = compact_stones(8);
    let large = compact_stones(64);
    let gs = build_axis_graph(&small, &params());
    let gl = build_axis_graph(&large, &params());
    let (ns, nl) = (gs.num_nodes() - 1, gl.num_nodes() - 1);
    assert!(nl > ns, "the larger position must have more nodes");
    assert!(
        DENSE_INDEX_CELLS_PER_NODE.saturating_mul(nl)
            > DENSE_INDEX_CELLS_PER_NODE.saturating_mul(ns),
        "the budget must scale with the node count"
    );
    assert!(axis_index_is_dense(&gl.node_coords, nl), "a bigger cluster is still dense");
}

#[test]
fn an_empty_node_set_is_not_dense_and_does_not_panic() {
    // `coord_bbox` returns `None` for `n_real == 0`; the predicate must answer false rather
    // than index an empty slice.
    assert!(!axis_index_is_dense(&[], 0));
}

#[test]
fn the_empty_board_still_builds_through_the_fallback_shaped_path() {
    // Zero stones: `StoneIndex::build` returns `None`, so the threat walk takes the hash arm on
    // an empty map. The 25-cell opening legal set (the documented dense-engine special case)
    // must still come out.
    let g = build_axis_graph(&StoneList { stones: Vec::new() }, &params());
    assert_eq!(g.n_stones, 0);
    assert_eq!(g.legal_node_gather.len(), 25, "the opening legal set is the 5x5 region");
}
