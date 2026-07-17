//! Named 0-stone semantics pin: the empty-board legal-move set is EXACTLY the
//! 5×5 axial rectangle {(dq,dr) : dq,dr ∈ [-2,2]} (25 cells) — pinned by
//! VALUE, not by count. The predecessor pinned this only implicitly (a
//! 25-count assert and an undo-restores-the-set assert); a shifted or
//! reshaped region with the same cardinality would have passed those. This
//! was once a launch blocker — hence the explicit set-equality pin.

use mantis_core::board::Board;
use std::collections::BTreeSet;

fn rect_5x5() -> BTreeSet<(i32, i32)> {
    let mut s = BTreeSet::new();
    for dq in -2i32..=2 {
        for dr in -2i32..=2 {
            s.insert((dq, dr));
        }
    }
    s
}

#[test]
fn empty_board_legal_moves_are_5x5_axial_rect() {
    let expected = rect_5x5();
    assert_eq!(expected.len(), 25);

    // Fresh board: exact SET equality.
    let b = Board::new();
    let got: BTreeSet<(i32, i32)> = b.legal_moves_set().iter().copied().collect();
    assert_eq!(got, expected, "fresh-board legal set must be the exact 5×5 axial rectangle");

    // After a full undo of a random game the same exact set must be restored
    // (exercises the rebuild path, not the ctor-seeded cache: undo marks the
    // cache dirty, so this hits the cells-empty rebuild branch).
    let mut b = Board::new();
    let mut diffs = Vec::new();
    let mut s = 0x5eed_0000_0000_0001u64;
    for _ in 0..17 {
        let legal = b.legal_moves();
        assert!(!legal.is_empty());
        // splitmix64
        s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        let (q, r) = legal[(z as usize) % legal.len()];
        diffs.push(b.apply_move_tracked(q, r).expect("legal move"));
    }
    while let Some(d) = diffs.pop() {
        b.undo_move(d);
    }
    let restored: BTreeSet<(i32, i32)> = b.legal_moves_set().iter().copied().collect();
    assert_eq!(
        restored, expected,
        "full undo must restore the exact 5×5 axial-rectangle legal set (0-stone semantics)"
    );
}
