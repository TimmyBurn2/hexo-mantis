//! Window flat-index pins. `window_flat_idx` / `window_coords` /
//! `window_flat_idx_at` / `window_flat_idx_at_geom` are public and their
//! in-repo consumers land in later work packages — without these pins a
//! transcription error would survive every gate in this crate. Round-trip
//! consistency alone cannot catch a transposed layout, so literal layout
//! values are pinned too (q-major row-major order, pinned by value).

use mantis_core::board::{Board, BoardGeometry, HALF};

fn wide_board_25() -> Board {
    Board::with_geometry(BoardGeometry {
        legal_move_radius: 8,
        cluster_threshold: 8,
        cluster_window_size: 25,
    })
}

/// Round-trip `window_coords(window_flat_idx(q,r)) == (q,r)` over ALL
/// in-window cells at window sizes 19 (default board) and 25 (wide geometry);
/// out-of-window cells return `usize::MAX`.
#[test]
fn window_flat_idx_round_trips_at_19_and_25() {
    for (board, size) in [(Board::new(), 19i32), (wide_board_25(), 25i32)] {
        let half = (size - 1) / 2;
        // Empty board → window centered at (0,0).
        assert_eq!(board.window_center(), (0, 0));
        for q in -half..=half {
            for r in -half..=half {
                let flat = board.window_flat_idx(q, r);
                assert_ne!(flat, usize::MAX, "in-window ({q},{r}) must index at size {size}");
                assert!(flat < (size * size) as usize);
                assert_eq!(
                    board.window_coords(flat),
                    (q, r),
                    "window_coords(window_flat_idx({q},{r}))) must round-trip at size {size}"
                );
            }
        }
        // One-beyond-edge cells are out-of-window → usize::MAX.
        for &(q, r) in &[
            (half + 1, 0), (-(half + 1), 0), (0, half + 1), (0, -(half + 1)),
            (half + 1, half + 1), (-(half + 1), -(half + 1)),
        ] {
            assert_eq!(
                board.window_flat_idx(q, r),
                usize::MAX,
                "out-of-window ({q},{r}) must return usize::MAX at size {size}"
            );
        }
    }
}

/// `window_flat_idx_at` is exactly the geometry kernel at (19, 9).
#[test]
fn window_flat_idx_at_matches_geom_kernel_19_9() {
    for q in -12i32..=12 {
        for r in -12i32..=12 {
            for &(cq, cr) in &[(0i32, 0i32), (3, -2), (-5, 7)] {
                assert_eq!(
                    Board::window_flat_idx_at(q, r, cq, cr),
                    Board::window_flat_idx_at_geom(q, r, cq, cr, 19, HALF),
                    "window_flat_idx_at must equal window_flat_idx_at_geom(..,19,9) at ({q},{r}) c=({cq},{cr})"
                );
            }
        }
    }
}

/// Literal layout pins: center flat index HALF*19+HALF = 180 at size 19 and
/// 12*25+12 = 312 at size 25, plus one off-center literal per size so a
/// transposed (r-major) layout cannot hide behind self-consistent round-trips.
#[test]
fn window_flat_idx_literal_layout_pins() {
    // Size 19, empty default board (center (0,0), half 9).
    let b19 = Board::new();
    assert_eq!(b19.window_flat_idx(0, 0), 180, "center cell at size 19 is 9*19+9 = 180");
    // Off-center: (1,0) → wq=10, wr=9 → 10*19+9 = 199 (q-major: q strides by 19).
    assert_eq!(b19.window_flat_idx(1, 0), 199, "(1,0) at size 19 is 10*19+9 = 199");
    // The transposed layout would give (0,1) → 199; the correct q-major gives 181.
    assert_eq!(b19.window_flat_idx(0, 1), 181, "(0,1) at size 19 is 9*19+10 = 181");

    // Size 25, empty wide board (center (0,0), half 12).
    let b25 = wide_board_25();
    assert_eq!(b25.window_flat_idx(0, 0), 312, "center cell at size 25 is 12*25+12 = 312");
    // Off-center: (1,0) → wq=13, wr=12 → 13*25+12 = 337 (q-major: q strides by 25).
    assert_eq!(b25.window_flat_idx(1, 0), 337, "(1,0) at size 25 is 13*25+12 = 337");
    assert_eq!(b25.window_flat_idx(0, 1), 313, "(0,1) at size 25 is 12*25+13 = 313");

    // The associated-fn kernel pins the same literals independently of Board.
    assert_eq!(Board::window_flat_idx_at(0, 0, 0, 0), 180);
    assert_eq!(Board::window_flat_idx_at_geom(0, 0, 0, 0, 25, 12), 312);
}
