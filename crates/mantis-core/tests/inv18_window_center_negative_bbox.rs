//! `Board::window_center` truncates toward zero on negative-bbox sums.
//!
//! An automated clippy `manual_midpoint` fix once substituted `(a + b) / 2` with
//! `i32::midpoint(a, b)`, which floors toward -∞. The two diverge by one cell
//! whenever `(min + max)` is negative-odd: `(-5 + -2) / 2 == -3` but
//! `i32::midpoint(-5, -2) == -4`. Anchor checkpoints were trained against the
//! truncating semantics, so the substitution was reverted and these tests pin it.
//!
//! `apply_move` checks only cell occupancy, so negative and out-of-radius
//! coordinates are accepted as direct writes — enough to exercise the bbox
//! centroid without overriding the legal-move radius.

use mantis_core::board::Board;

/// A negative-odd q-sum truncates to cq=-3; `i32::midpoint` would floor it to -4.
#[test]
fn test_window_center_negative_q_axis_truncate() {
    let mut b = Board::new();
    // A test-side construction: `apply_move` accepts these as direct writes.
    b.apply_move(-5, 0).expect("apply -5,0");
    b.apply_move(-2, 0).expect("apply -2,0");

    let (cq, cr) = b.window_center();
    // bbox q = [-5, -2], sum = -7, truncating to -3; flooring would give -4.
    assert_eq!(
        cq, -3,
        "negative-odd q-sum: (a+b)/2 truncates toward 0 — i32::midpoint floor would give -4"
    );
    // r bbox = [0, 0]: the two semantics agree.
    assert_eq!(cr, 0, "r-axis collapsed bbox; centroid 0 invariant");
}

/// The symmetric pin on the r axis.
#[test]
fn test_window_center_negative_r_axis_truncate() {
    let mut b = Board::new();
    b.apply_move(0, -5).expect("apply 0,-5");
    b.apply_move(0, -2).expect("apply 0,-2");

    let (cq, cr) = b.window_center();
    assert_eq!(cq, 0, "q-axis collapsed bbox; centroid 0 invariant");
    assert_eq!(
        cr, -3,
        "negative-odd r-sum: (a+b)/2 truncates toward 0 — i32::midpoint floor would give -4"
    );
}

/// Positive-bbox guard: truncate and floor agree on non-negative sums, and any
/// revisit of the midpoint semantics must keep that behaviour stable.
#[test]
fn test_window_center_positive_bbox_unchanged() {
    let mut b = Board::new();
    b.apply_move(1, 3).expect("apply 1,3");
    b.apply_move(5, 7).expect("apply 5,7");

    let (cq, cr) = b.window_center();
    // bbox q = [1, 5] -> 3 and r = [3, 7] -> 5 under both semantics.
    assert_eq!(
        (cq, cr),
        (3, 5),
        "positive-bbox centroid must remain stable across midpoint-semantic revisits"
    );
}
