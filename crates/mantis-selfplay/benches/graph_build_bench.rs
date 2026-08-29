//! WP6 Metric (1) — the GATED graph-build microbench, RE-TARGETED to the
//! WP6-OWNED `build_leaf_graph` wrapper (`queues/graph.rs`), NOT the raw
//! `build_axis_graph` kernel (WP1 code, ported byte-identical → a bench on it
//! would fire on no WP6 code mechanism). `build_leaf_graph` is the exact site of
//! the WP6 changes: the leaf `(stones, current_player, moves_remaining)` →
//! `StoneList` extraction, the range/narrowing seam-guards, the
//! `build_axis_graph` call, the `builder_impl` native handshake, and the D6
//! error path (`Result<AxisGraph, String>` instead of the frozen `.ok()`-swallow).
//!
//! Regime `graph_build_gnn_axis_v1_leafcorpus` (spec `gnn_axis_v1`:
//! `win_length = 6 / radius = 6 / trunk = 19`) over a FIXED leaf corpus of
//! `LEAF_CORPUS_SIZE = 64` requests generated deterministically from
//! `LEAF_CORPUS_SEED` via the pinned splitmix64 stream (CAPTURE_LOG §B / PREREG
//! C-10) — same generator family as the mock NN, so the corpus is reproducible
//! new-side with no model and no entropy.
//!
//! HONESTY (CAPTURE_LOG §A.2): the OLD-side `build_leaf_graph` is
//! `pub(crate)`-unreachable externally (like `GraphWire::from_axis_graphs`), so
//! there is NO capturable old/new pair. This is a NEW-SIDE regression ANCHOR: run
//! it, record the median. Kernel parity (`build_axis_graph`) is INHERITED from
//! WP1's committed 1,696-case byte-parity goldens; the wrapper adds only the
//! `StoneList` extraction + the native handshake, which this bench times. F-19
//! (build-once-per-leaf) is gated STRUCTURALLY here (`assert_one_build_per_leaf`)
//! and in the queue oracles (`tests/queue_roundtrip.rs`), NOT by wall-clock.
//!
//! Metric (2) — pool gen/hr under the mock NN — is DEFERRED (see
//! `wp/WP6/BENCH_ADJUDICATION.md`): it is CONFOUNDED (never a gate), the mock-NN
//! producer face is not driven end-to-end anywhere yet, and a criterion bench
//! over spawned worker threads + a mock producer thread (condvar timeouts,
//! wall-clock) is inherently flaky. Correctness is gated on the byte pins, not on
//! any gen/hr number.

use std::collections::HashSet;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use mantis_graph::BUILDER_IMPL_NATIVE;
use mantis_selfplay::queues::build_leaf_graph;

// ── Regime constants (PREREG "Metric 1 leaf corpus"; FIXED) ──────────────────
const LEAF_CORPUS_SIZE: usize = 64;
const LEAF_CORPUS_SEED: u64 = 0x6C65_6166_0006_0001;
// gnn_axis_v1 build params.
const WIN_LENGTH: u8 = 6;
const RADIUS: u16 = 6;
const TRUNK_SIZE: i32 = 19;
// Coordinate half-extent of the deterministic corpus (a spread that yields a
// non-trivial in-window + off-window legal set under the 19-trunk window).
const COORD_HALF: i64 = 9;

// ── Pinned splitmix64 (CAPTURE_LOG §B / PREREG C-10 — canonical constants) ────
fn splitmix64_step(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One deterministic leaf request: the exact `build_leaf_graph` argument shape
/// (stones as `(q, r, player)`, `current_player`, `moves_remaining`).
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

/// Build the FIXED 64-leaf corpus deterministically. The whole corpus is one
/// continuous splitmix64 stream seeded by `LEAF_CORPUS_SEED`, so the exact same
/// 64 leaf boards are reproduced on every run and every box. Coordinates are
/// deduped (a real board never stacks two stones on one cell); players alternate
/// P1/P2 by placement order (the realistic mid-game leaf); every input is kept in
/// the ranges `build_leaf_graph` accepts so all 64 build `Ok` (the timed path is
/// the success wrapper, not the reject arm).
fn build_leaf_corpus() -> Vec<LeafRequest> {
    let mut s = LEAF_CORPUS_SEED;
    let mut corpus = Vec::with_capacity(LEAF_CORPUS_SIZE);
    for _ in 0..LEAF_CORPUS_SIZE {
        // 8..=48 stones — a spread of leaf sizes (early to deep mid-game).
        let target = draw_range(&mut s, 8, 48) as usize;
        let mut seen: HashSet<(i64, i64)> = HashSet::with_capacity(target);
        let mut stones: Vec<(i64, i64, i64)> = Vec::with_capacity(target);
        // Draw up to 3× the target to fill after dedupe collisions (deterministic:
        // a skipped duplicate still consumes its two stream draws).
        for _ in 0..(target * 3) {
            if stones.len() >= target {
                break;
            }
            let q = draw_range(&mut s, -COORD_HALF, COORD_HALF);
            let r = draw_range(&mut s, -COORD_HALF, COORD_HALF);
            if seen.insert((q, r)) {
                // Alternate players by placement order (P1 first).
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

/// F-19 (structural, NOT wall-clock): build every leaf EXACTLY once and assert
/// the native `builder_impl` stamp + a build count == the corpus size. A
/// redundant/duplicated build (invisible to a per-call median AND to the byte
/// pins) would double the count. Runs ONCE at bench setup, outside timing.
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

/// Metric (1) — the GATED criterion median: `build_leaf_graph` over the fixed
/// 64-leaf corpus. Each iteration builds all 64 leaves (`black_box`-ing each
/// result so the allocation is not elided); the per-build number is the reported
/// corpus median / 64.
fn graph_build_gnn_axis_v1_leafcorpus(c: &mut Criterion) {
    let corpus = build_leaf_corpus();
    assert_eq!(corpus.len(), LEAF_CORPUS_SIZE);
    // F-19 structural gate (once, before timing).
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
    // PREREG-pinned criterion regime: warm_up_time = 3s, sample_size = 100.
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100);
    targets = graph_build_gnn_axis_v1_leafcorpus
}
criterion_main!(benches);
