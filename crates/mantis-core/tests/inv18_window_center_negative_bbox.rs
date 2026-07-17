//! INV18 — `Board::window_center` truncate-toward-zero pin on negative-bbox
//! sums.
//!
//! An automated clippy `manual_midpoint` fix in the predecessor codebase once
//! substituted `(a + b) / 2` (truncate) with `i32::midpoint(a, b)` (floor
//! toward -∞) at production sites including `window_center`. The two
//! semantics diverge by one cell whenever `(min + max)` is negative-odd: e.g.
//! `(-5 + -2) / 2 == -3` (truncate); `i32::midpoint(-5, -2) == -4` (floor).
//!
//! Legacy anchor checkpoints were trained against truncate semantics; the
//! substitution was reverted and this file pins the revert with three
//! regression tests (falsified-register midpoint row; predecessor forensic
//! record):
//!
//!   1. `test_window_center_negative_q_axis_truncate` — bbox q=[-5,-2],
//!      sum=-7 (negative-odd). Truncate yields cq=-3. Under `i32::midpoint`
//!      this would yield cq=-4 — the test would have FAILED at the bad
//!      commit and passes post-revert.
//!   2. `test_window_center_negative_r_axis_truncate` — bbox r=[-5,-2]
//!      on the r-axis; same divergence rationale.
//!   3. `test_window_center_positive_bbox_unchanged` — bbox q=[1,5]
//!      r=[3,7], even-q-sum 6→3 and even-r-sum 10→5; truncate and floor
//!      identical. Regression guard for any future revisit of midpoint
//!      semantics: positive-bbox behaviour must remain stable.
//!
//! Construction: `Board::new()` + `Board::apply_move(q, r)`. `apply_move`
//! checks only cell occupancy; negative coordinates and out-of-radius
//! placements are accepted as direct cell writes — sufficient to exercise
//! the bbox-centroid path without needing legal-move radius overrides.

use mantis_core::board::Board;

/// Negative-q-axis bbox sum is negative-odd — truncate semantic yields
/// cq=-3. `i32::midpoint(-5, -2)` would yield cq=-4 (floor toward -∞).
#[test]
fn test_window_center_negative_q_axis_truncate() {
    let mut b = Board::new();
    // apply_move only checks occupancy; negative coords + cells outside
    // the default legal radius are accepted as direct writes (this is a
    // test-side construction pattern, not production behaviour).
    b.apply_move(-5, 0).expect("apply -5,0");
    b.apply_move(-2, 0).expect("apply -2,0");

    let (cq, cr) = b.window_center();
    // bbox q = [-5, -2], sum = -7, (-7) / 2 = -3 (truncate toward zero).
    // Under the floor-semantic substitution this would have been -4.
    assert_eq!(
        cq, -3,
        "negative-odd q-sum: (a+b)/2 truncates toward 0 — i32::midpoint floor would give -4"
    );
    // r bbox = [0, 0], sum = 0, midpoint identical under both semantics.
    assert_eq!(cr, 0, "r-axis collapsed bbox; centroid 0 invariant");
}

/// Symmetric pin on r-axis; same divergence rationale as the q-axis case.
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

/// Positive-bbox regression guard. Under both `(a+b)/2` truncate and
/// `i32::midpoint` floor the result is identical when `(a+b)` is non-negative
/// (truncate and floor agree on non-negative integers). Any future revisit of
/// midpoint semantics must preserve positive-bbox behaviour.
#[test]
fn test_window_center_positive_bbox_unchanged() {
    let mut b = Board::new();
    b.apply_move(1, 3).expect("apply 1,3");
    b.apply_move(5, 7).expect("apply 5,7");

    let (cq, cr) = b.window_center();
    // bbox q = [1, 5], sum = 6, 6/2 = 3 (both semantics agree).
    // bbox r = [3, 7], sum = 10, 10/2 = 5 (both semantics agree).
    assert_eq!(
        (cq, cr),
        (3, 5),
        "positive-bbox centroid must remain stable across midpoint-semantic revisits"
    );
}
