//! Graph-build microbench over the `build_leaf_graph` wrapper, not the raw
//! `build_axis_graph` kernel: the wrapper is what adds the `StoneList` extraction,
//! the range guards, the native `builder_impl` handshake and the error path.
//!
//! The regime is spec `gnn_axis_v1` (`win_length = 6 / radius = 6 / trunk = 19`)
//! over a fixed 64-leaf corpus drawn deterministically from `LEAF_CORPUS_SEED`, so
//! it reproduces on every box with no model and no entropy.
//!
//! There is no capturable old/new pair — the old wrapper was crate-private — so this
//! is a new-side regression anchor. Kernel parity is inherited from the committed
//! byte-parity goldens, and build-once-per-leaf is gated structurally by
//! `assert_one_build_per_leaf`, never by wall-clock.

use std::collections::HashSet;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use mantis_graph::BUILDER_IMPL_NATIVE;
use mantis_selfplay::queues::build_leaf_graph;

// Regime constants, fixed.
const LEAF_CORPUS_SIZE: usize = 64;
const LEAF_CORPUS_SEED: u64 = 0x6C65_6166_0006_0001;
// gnn_axis_v1 build params.
const WIN_LENGTH: u8 = 6;
const RADIUS: u16 = 6;
const TRUNK_SIZE: i32 = 19;
// Coordinate half-extent: a spread that yields a non-trivial in-window plus off-window
// legal set under the 19-trunk window.
const COORD_HALF: i64 = 9;

// Pinned splitmix64, canonical constants.
fn splitmix64_step(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One deterministic leaf request in `build_leaf_graph`'s argument shape.
struct LeafRequest {
    stones: Vec<(i64, i64, i64)>,
    current_player: i64,
    moves_remaining: i64,
}

/// Draw an integer in `[lo, hi]` inclusive from the running splitmix64 stream.
fn draw_range(s: &mut u64, lo: i64, hi: i64) -> i64 {
    let span = (hi - lo + 1) as u64;
    lo + (splitmix64_step(s) % span) as i64
}

/// Build the fixed 64-leaf corpus from one continuous splitmix64 stream, so the
/// same boards reproduce on every box. Coordinates are deduped and every input stays
/// in range, so all 64 build `Ok` and the timed path is the success wrapper.
fn build_leaf_corpus() -> Vec<LeafRequest> {
    let mut s = LEAF_CORPUS_SEED;
    let mut corpus = Vec::with_capacity(LEAF_CORPUS_SIZE);
    for _ in 0..LEAF_CORPUS_SIZE {
        // 8..=48 stones: early to deep mid-game leaf sizes.
        let target = draw_range(&mut s, 8, 48) as usize;
        let mut seen: HashSet<(i64, i64)> = HashSet::with_capacity(target);
        let mut stones: Vec<(i64, i64, i64)> = Vec::with_capacity(target);
        // Draw up to 3x the target to refill after dedupe collisions; a skipped
        // duplicate still consumes its two stream draws, so this stays deterministic.
        for _ in 0..(target * 3) {
            if stones.len() >= target {
                break;
            }
            let q = draw_range(&mut s, -COORD_HALF, COORD_HALF);
            let r = draw_range(&mut s, -COORD_HALF, COORD_HALF);
            if seen.insert((q, r)) {
                // Alternate players by placement order, P1 first.
                let player = if stones.len().is_multiple_of(2) {
                    1
                } else {
                    -1
                };
                stones.push((q, r, player));
            }
        }
        let current_player = if splitmix64_step(&mut s) & 1 == 0 {
            1
        } else {
            -1
        };
        let moves_remaining = draw_range(&mut s, 1, 200);
        corpus.push(LeafRequest {
            stones,
            current_player,
            moves_remaining,
        });
    }
    corpus
}

/// Assert one build per leaf and a native `builder_impl` stamp, once at setup and
/// outside timing: a duplicated build is invisible to both the median and the byte pins.
fn assert_one_build_per_leaf(corpus: &[LeafRequest]) {
    let mut builds = 0usize;
    for leaf in corpus {
        let g = build_leaf_graph(
            &leaf.stones,
            leaf.current_player,
            leaf.moves_remaining,
            WIN_LENGTH,
            RADIUS,
            TRUNK_SIZE,
        )
        .expect("every corpus leaf builds (inputs in range)");
        assert_eq!(
            g.builder_impl, BUILDER_IMPL_NATIVE,
            "F-19: each leaf stamps the native builder_impl"
        );
        builds += 1;
    }
    assert_eq!(
        builds, LEAF_CORPUS_SIZE,
        "F-19: exactly one build per leaf (corpus size)"
    );
}

/// Time `build_leaf_graph` over the whole 64-leaf corpus; the per-build number is the
/// reported median / 64.
fn graph_build_gnn_axis_v1_leafcorpus(c: &mut Criterion) {
    let corpus = build_leaf_corpus();
    assert_eq!(corpus.len(), LEAF_CORPUS_SIZE);
    // Structural gate, once, before timing.
    assert_one_build_per_leaf(&corpus);

    c.bench_function("graph_build_gnn_axis_v1_leafcorpus", |b| {
        b.iter(|| {
            for leaf in &corpus {
                let g = build_leaf_graph(
                    black_box(&leaf.stones),
                    black_box(leaf.current_player),
                    black_box(leaf.moves_remaining),
                    WIN_LENGTH,
                    RADIUS,
                    TRUNK_SIZE,
                )
                .expect("corpus leaf builds");
                black_box(&g);
            }
        });
    });
}

criterion_group! {
    name = benches;
    // Pinned criterion regime: warm_up_time = 3s, sample_size = 100.
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100);
    targets = graph_build_gnn_axis_v1_leafcorpus
}
criterion_main!(benches);
