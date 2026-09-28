//! Positions the leaf and root tactics tests share: stone sequences from `Board::new`, played at radius 8.

use mantis_core::board::Board;

/// T2's `fix219@17`: P2 to move with two stones holds a one-turn strict win at (-4, -11), (-4, -10).
pub(crate) const FIX219: [(i32, i32); 17] = [
    (-1, -8),
    (-4, -8),
    (-4, -9),
    (-5, -8),
    (0, -10),
    (1, -13),
    (-6, -9),
    (-5, -7),
    (3, -13),
    (-6, -12),
    (-5, -10),
    (-6, -15),
    (4, -16),
    (-1, -14),
    (2, -15),
    (-13, 0),
    (-10, 0),
];

/// P1's four (0..3, 0) and P2's scatter; P1 is to move with two stones after the last.
pub(crate) const P1_FOUR: [(i32, i32); 11] = [
    (0, 0),
    (20, 20),
    (22, 20),
    (1, 0),
    (2, 0),
    (20, 24),
    (22, 24),
    (3, 0),
    (-9, 9),
    (30, 30),
    (32, 30),
];

/// P2's open four at (10..13, 5); after the last stone P1 is to move with ONE stone.
pub(crate) const P2_FOUR_ONE_LEFT: [(i32, i32); 8] = [
    (0, 0),
    (10, 5),
    (11, 5),
    (1, 0),
    (2, 0),
    (12, 5),
    (13, 5),
    (-5, 0),
];

/// `seq` played from an empty board at radius 8.
pub(crate) fn played(seq: &[(i32, i32)]) -> Board {
    let mut b = Board::new();
    b.set_legal_move_radius(8);
    for &(q, r) in seq {
        b.apply_move(q, r)
            .expect("a test sequence places on empty cells");
    }
    b
}
