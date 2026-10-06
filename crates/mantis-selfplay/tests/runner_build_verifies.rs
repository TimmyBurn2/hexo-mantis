//! Self-play's leaf build always runs the producer verify: the one test in this binary, so the skip count is exact.

use mantis_graph::unverified_builds;
use mantis_selfplay::queues::build_leaf_graph;

#[test]
fn self_plays_leaf_build_never_skips_the_producer_verify() {
    let before = unverified_builds();
    let boards: [&[(i64, i64, i64)]; 3] = [
        &[],
        &[(0, 0, 1), (1, 0, -1)],
        &[(0, 0, 1), (1, 0, -1), (0, 1, -1), (2, -1, 1)],
    ];
    for (i, stones) in boards.iter().enumerate() {
        let (player, left) = [(1, 2), (-1, 1), (1, 1)][i];
        build_leaf_graph(stones, player, left, 6, 6, 19).expect("a legal test position");
    }
    assert_eq!(
        unverified_builds(),
        before,
        "a self-play build skipped its verify"
    );
}
