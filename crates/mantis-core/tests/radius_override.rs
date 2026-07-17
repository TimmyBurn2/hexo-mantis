//! Radius curriculum: Board::override_legal_move_radius tests.
//!
//! Re-anchored from the predecessor's registry-spec construction to plain
//! geometry values: (8, 8, 25) is exactly what the "v6w25" record resolved
//! to (pinned by the geometry-ctor tests in state/core.rs).

use mantis_core::board::{Board, BoardGeometry};

fn wide_board() -> Board {
    Board::with_geometry(BoardGeometry {
        legal_move_radius: 8,
        cluster_threshold: 8,
        cluster_window_size: 25,
    })
}

/// T1: a wide-geometry board overridden to radius 5 produces a 90-cell
/// legal-move set (not the radius-8 set) once a stone is placed.
#[test]
fn test_board_override_radius() {
    let mut board = wide_board();
    assert_eq!(board.legal_move_radius(), 8);

    // Place a stone so legal_moves_set() uses the radius-based expansion.
    board.apply_move(0, 0).unwrap();

    let default_legal = board.legal_moves();
    assert!(
        default_legal.len() > 150,
        "R=8 with one stone should give >150 legal moves; got {}",
        default_legal.len()
    );

    // Override to R=5
    board.override_legal_move_radius(5);
    assert_eq!(board.legal_move_radius(), 5);
    let overridden_legal = board.legal_moves();
    // Hex-ball radius 5 has 91 cells total; minus the occupied stone = 90 empty.
    assert_eq!(
        overridden_legal.len(),
        90,
        "R=5 hex-ball around one stone = 90 empty cells; got {}",
        overridden_legal.len()
    );
}

/// T2: after a radius override, the OTHER geometry values are unchanged
/// (re-anchor of the predecessor's "spec unchanged" assertion — the spec
/// binding died with the registry decoupling; the value-level contract is
/// what survives).
#[test]
fn test_board_override_preserves_geometry() {
    let mut board = wide_board();

    board.override_legal_move_radius(5);

    // Non-radius geometry values must NOT be mutated.
    assert_eq!(board.cluster_window_size(), 25, "cluster_window_size unchanged");
    assert_eq!(board.cluster_threshold(), 8, "cluster_threshold unchanged");
    // The introspection surface reflects the override on radius only.
    assert_eq!(
        board.geometry(),
        BoardGeometry { legal_move_radius: 5, cluster_threshold: 8, cluster_window_size: 25 }
    );
}

/// T3: `Board::set_legal_move_radius` (the plain setter) works silently on a
/// geometry-constructed board — it has no Rust-level guard; any boundary
/// guard lives at the boundary layer.
#[test]
fn test_set_legal_move_radius_no_rust_guard() {
    let mut board = wide_board();
    // `set_legal_move_radius` does NOT error; it silently overrides.
    board.set_legal_move_radius(5);
    assert_eq!(board.legal_move_radius(), 5);
}
