//! INV18b — `Board::get_cluster_views` truncate-toward-zero pin on
//! negative-bbox cluster centres.
//!
//! Companion to INV18 (`tests/inv18_window_center_negative_bbox.rs`)
//! covering the cluster-centre code path (small-cluster centroid and
//! massive-cluster fallback in `state/cluster.rs`) instead of the
//! bbox-centroid path in `state/core.rs`.
//!
//! An automated clippy `manual_midpoint` fix in the predecessor codebase once
//! substituted `(a + b) / 2` (truncate) with `i32::midpoint(a, b)` (floor
//! toward -∞) at both cluster-centre sites. The two semantics diverge by one
//! cell whenever `(min + max)` is negative-odd (falsified-register midpoint
//! row; predecessor forensic record identifies the small-cluster path as the
//! higher-exposure call site under normal play).
//!
//! Tests:
//!   1. `test_cluster_center_small_path_negative_q` — default Board with
//!      stones placed at `q ∈ {-5, -4, -3, -2}, r=0`. All within
//!      hex-distance 1 of each other → single cluster. Span q=3 ≤
//!      threshold (window_size - 4 = 15 for the default window) →
//!      small-cluster path. Bbox q=[-5, -2], sum=-7 → centroid q under
//!      truncate=-3. Under `i32::midpoint` floor this would yield -4.
//!   2. `test_cluster_center_v6w25_kept_planes` — wide geometry
//!      (radius 8, threshold 8, window 25 — the values the predecessor's
//!      "v6w25" record resolves to), single cluster on the r-axis with
//!      negative-odd r-sum (r ∈ {-5, -4, -3, -2}, q=0). Asserts the
//!      cluster's centre is `(0, -3)` (truncate) not `(0, -4)` (floor).
//!
//! Construction notes:
//!   - `apply_move(q, r)` validates only cell occupancy; negative coords and
//!     cells outside the default legal-move radius are accepted as direct
//!     writes (test-side construction pattern).
//!   - Turn rhythm: P1 opens with 1 move, then P1+P2 alternate 2-move
//!     turns. Cluster ownership is irrelevant — `get_clusters()` groups
//!     stones by hex-distance regardless of player.
//!   - Multiple clusters can be returned; tests use a `contains` check
//!     against `final_centers` rather than indexing the first slot,
//!     because cluster iteration order depends on the underlying
//!     HashMap iteration order.

use mantis_core::board::{Board, BoardGeometry};

/// Default Board, single small cluster on q-axis with negative-odd q-sum.
/// Asserts the small-cluster path produces centroid `(-3, 0)` (truncate) —
/// not `(-4, 0)` (floor).
#[test]
fn test_cluster_center_small_path_negative_q() {
    let mut b = Board::new();
    // Turn 1 (P1): 1 move.
    b.apply_move(-5, 0).expect("apply -5,0");
    // Turn 2 (P2): 2 moves — placed within the same cluster (within
    // hex-distance 5 of the P1 stone) so all 4 stones form one cluster.
    b.apply_move(-4, 0).expect("apply -4,0");
    b.apply_move(-3, 0).expect("apply -3,0");
    // Turn 3 (P1): 2 moves — last in-cluster stone + a far-away filler
    // to avoid the empty-clusters fallback edge.
    b.apply_move(-2, 0).expect("apply -2,0");
    b.apply_move(100, 100).expect("apply 100,100 (far filler)");

    let (_views, centers) = b.get_cluster_views();

    // The negative-q cluster bbox is q=[-5,-2], r=[0,0].
    // Truncate: ((-5) + (-2)) / 2 == -3 (NOT i32::midpoint's -4).
    let want = (-3, 0);
    assert!(
        centers.contains(&want),
        "small-cluster centroid must include {:?} (truncate semantic) — \
         i32::midpoint floor would give (-4, 0). Got centers: {:?}",
        want, centers,
    );
    // Negative regression: -4 must NOT appear as a centre on this bbox
    // (would indicate floor-semantic reintroduction).
    assert!(
        !centers.contains(&(-4, 0)),
        "centroid (-4, 0) would indicate i32::midpoint floor — got {:?}",
        centers,
    );
}

/// Wide geometry (8, 8, 25), single small cluster on r-axis with
/// negative-odd r-sum. Asserts truncate centroid `(0, -3)` (not floor
/// `(0, -4)`). Pins the wide-window cluster path so future multi-window
/// work doesn't silently reintroduce the floor semantic.
#[test]
fn test_cluster_center_v6w25_kept_planes() {
    let mut b = Board::with_geometry(BoardGeometry {
        legal_move_radius: 8,
        cluster_threshold: 8,
        cluster_window_size: 25,
    });
    // Sanity: wide-window cluster geometry must be present.
    assert_eq!(b.cluster_window_size(), 25, "window_size precondition");

    // Same construction as test 1 but rotated to the r-axis to also
    // exercise the cr cluster-centre code path symmetrically.
    b.apply_move(0, -5).expect("apply 0,-5");
    b.apply_move(0, -4).expect("apply 0,-4");
    b.apply_move(0, -3).expect("apply 0,-3");
    b.apply_move(0, -2).expect("apply 0,-2");
    b.apply_move(100, 100).expect("apply 100,100 (far filler)");

    let (_views, centers) = b.get_cluster_views();

    // Cluster bbox: q=[0,0], r=[-5,-2]. Truncate: ((-5) + (-2)) / 2 == -3.
    // i32::midpoint floor would yield (0, -4).
    let want = (0, -3);
    assert!(
        centers.contains(&want),
        "wide-geometry cluster centroid must include {:?} (truncate semantic) — \
         i32::midpoint floor would give (0, -4). Got centers: {:?}",
        want, centers,
    );
    assert!(
        !centers.contains(&(0, -4)),
        "centroid (0, -4) would indicate i32::midpoint floor reintroduction — got {:?}",
        centers,
    );
}
