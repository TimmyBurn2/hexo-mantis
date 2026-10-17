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

/// The solver goldens' row ending `rung_00127@11`: P2's seven-turn win; P1's replies to its first turn re-prove in 5, 5, 6.
pub(crate) const RUNG127_AT11: [(i32, i32); 11] = [
    (2, -1),
    (1, -5),
    (-3, 0),
    (0, -9),
    (3, -2),
    (-3, -1),
    (-4, 0),
    (-2, -2),
    (-3, -2),
    (-5, 1),
    (-4, 1),
];

/// The goldens' `fix77@15`: P2 to move holds a five-turn strict win; P1's replies to its first turn re-prove in 4, 4, 3.
pub(crate) const FIX77_AT15: [(i32, i32); 15] = [
    (-6, 5),
    (-4, 4),
    (-6, 6),
    (-3, 3),
    (-5, 3),
    (-6, 4),
    (-2, 1),
    (-8, 7),
    (-5, 4),
    (-6, 7),
    (-7, 5),
    (-8, 6),
    (-3, 5),
    (-6, 9),
    (-7, 6),
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
