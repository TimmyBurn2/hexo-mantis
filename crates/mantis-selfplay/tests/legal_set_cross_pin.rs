//! AUDIT-1 F-18 — the graph's legal nodes and the board's legal set are ONE set, at BOTH radii.
//!
//! THE DEFECT. `mantis-graph` is dep-free by contract, so it re-implements the geometry
//! `mantis-core` owns: `legal_moves_from_stones` twins `Board::legal_moves_set`, `window_center`
//! twins `Board::window_center`, and the empty-board `-2..=2` region is hand-copied on both
//! sides. Nothing compared them. The MCTS legal set is built at `spec.legal_move_radius`
//! (`runner::game`) and the graph's legal nodes at `spec.graph_radius`
//! (`runner::search_drive`, `replay::hexg`); the registry now REFUSES the two typed apart
//! (`mantis_encoding::spec::validate`, pinned by
//! `registry_census.rs::the_two_radii_typed_APART_are_refused_at_parse`), but equal NUMBERS on
//! two independent implementations is not the same claim as one cell set. This file makes it one.
//!
//! WHY BOTH RADII. Leg 3 moves the shipped identity from `gnn_axis_v1` (r6) to `gnn_axis_r8`
//! (r8). A pin at one radius proves the copies agree at one number, which is exactly the state
//! `graph_child_parity.rs` was already in: it pins both sides against the SAME frozen r6 fixture,
//! never against each other.

use mantis_core::{Board, Cell, Player};
use mantis_graph::{build_axis_graph, BuildParams, StoneList};

/// The registry's two shipped graph radii. Typed here rather than read from the registry
/// because this crate must not depend on `mantis-encoding` to state a geometry claim; the
/// registry's own agreement with these numbers is pinned by `registry_census.rs`.
const SHIPPED_GRAPH_RADII: [i32; 2] = [6, 8];

/// Stone layouts chosen for what they stress, not for coverage: a lone stone (the ball is
/// exactly one hex ball), two stones far enough apart that the balls are disjoint, two close
/// enough that they overlap (the union-vs-sum branch of core's capacity bound), and a
/// negative-coordinate cluster (window_center truncates toward zero, so sign matters).
fn layouts() -> Vec<Vec<(i32, i32, i8)>> {
    vec![
        vec![(0, 0, 1)],
        vec![(0, 0, 1), (12, -3, -1)],
        vec![(0, 0, 1), (1, 1, -1), (2, -1, 1)],
        vec![(-5, -4, 1), (-3, -7, -1), (-6, 2, 1)],
    ]
}

fn board_at(radius: i32, stones: &[(i32, i32, i8)]) -> Board {
    let cells: Vec<((i32, i32), Cell)> = stones
        .iter()
        .map(|&(q, r, p)| ((q, r), if p > 0 { Cell::P1 } else { Cell::P2 }))
        .collect();
    let mut b = Board::from_stones(&cells, Player::One, 2, stones.len() as u32, None);
    b.set_legal_move_radius(radius);
    b
}

fn graph_legal_coords(radius: i32, stones: &[(i32, i32, i8)]) -> Vec<(i32, i32)> {
    let g = build_axis_graph(
        &StoneList { stones: stones.to_vec() },
        &BuildParams { radius: radius as u16, ..BuildParams::V1_GEOMETRY },
    );
    let mut out: Vec<(i32, i32)> = g
        .legal_node_gather
        .iter()
        .map(|&row| {
            let i = row as usize * 2;
            (g.node_coords[i], g.node_coords[i + 1])
        })
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn graph_legal_nodes_equal_board_legal_set_at_r6_and_r8() {
    for radius in SHIPPED_GRAPH_RADII {
        for stones in layouts() {
            let from_graph = graph_legal_coords(radius, &stones);
            let from_board = board_at(radius, &stones).legal_moves();
            assert_eq!(
                from_graph, from_board,
                "radius {radius}, stones {stones:?}: the graph's legal nodes and the board's \
                 legal set are the same cells or the ragged policy covers cells the tree \
                 never expands"
            );
            assert!(!from_graph.is_empty(), "radius {radius}: an empty comparison proves nothing");
        }
    }
}

#[test]
fn graph_window_center_equals_board_window_center() {
    for radius in SHIPPED_GRAPH_RADII {
        for stones in layouts() {
            let g = build_axis_graph(
                &StoneList { stones: stones.clone() },
                &BuildParams { radius: radius as u16, ..BuildParams::V1_GEOMETRY },
            );
            assert_eq!(
                g.window_center,
                board_at(radius, &stones).window_center(),
                "radius {radius}, stones {stones:?}: the policy-slot origin is one point"
            );
        }
    }
}

#[test]
fn the_empty_board_region_is_the_same_hand_copied_literal_on_both_sides() {
    // The planted break the audit names: change the graph's `-2..=2` to `-3..=3` and this reds.
    // Both sides ignore the radius on a stoneless board — deliberate and documented on each —
    // so the assertion is over the LITERAL and must hold at every radius.
    for radius in SHIPPED_GRAPH_RADII {
        let from_graph = graph_legal_coords(radius, &[]);
        let from_board = board_at(radius, &[]).legal_moves();
        assert_eq!(from_graph.len(), 25, "radius {radius}: the graph's stoneless region");
        assert_eq!(
            from_graph, from_board,
            "radius {radius}: the stoneless 5x5 region is hand-copied on both sides"
        );
    }
}
