//! PERF-TRANCHE-1 A1 — the block-diagonal fuse, benched at run5's MEASURED pop shape.
//!
//! Prereg hotspot (LAW-09): `GraphWire::from_axis_graphs`, ledger §10.1 line #2 —
//! `wire_fuse` 33.78 ms/pop, 0.859 ms/sim, 25.3 % of the card, at a derived 1.35 GB/s of
//! output. The A/B is old-side vs new-side on ONE host in one session; this bench carries
//! no `tools/bench_floors.toml` row because the attested floor host is not the box the
//! tranche measures on, and a floor comparison across hosts is not a measurement.
//!
//! The corpus is shaped to the ledger's own contended reading — 39.29 graphs per served
//! pop, 1 203 310 fused edges — and `assert_corpus_shape` refuses to run if it has drifted
//! off that shape, because a fuse benched at the wrong edge count measures a different
//! function than the one the ledger ranked.

use std::collections::HashSet;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use mantis_graph::AxisGraph;
use mantis_selfplay::queues::{build_leaf_graph, GraphWire};

// ── Regime constants, pinned to the ledger's §2.2 contended reading ──────────
/// Achieved batch at twelve workers was 39.29 of 64; 40 is that, rounded to the batch the
/// pop actually hands the fuse.
const POP_GRAPHS: usize = 40;
const CORPUS_SEED: u64 = 0x0A1F_6675_7365_0001;
/// Coordinate half-extent that reproduces the measured ~30 600 edges per graph under the
/// gnn_axis_v1 build params. Measured, not guessed: at half = 9 the builder yields ~21 000.
const COORD_HALF: i64 = 12;
const STONES_LO: i64 = 70;
const STONES_HI: i64 = 130;
// gnn_axis_v1 build params.
const WIN_LENGTH: u8 = 6;
const RADIUS: u16 = 6;
const TRUNK_SIZE: i32 = 19;
/// The ledger's measured mean fused edges per pop, and the band the corpus must land in.
const LEDGER_MEAN_FUSED_EDGES: usize = 1_203_310;
const SHAPE_BAND: f64 = 0.20;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn draw_range(s: &mut u64, lo: i64, hi: i64) -> i64 {
    lo + (splitmix64(s) % ((hi - lo + 1) as u64)) as i64
}

/// One served pop's worth of built graphs, deterministic from `CORPUS_SEED`.
fn build_pop_corpus() -> Vec<AxisGraph> {
    let mut s = CORPUS_SEED;
    let mut graphs = Vec::with_capacity(POP_GRAPHS);
    for _ in 0..POP_GRAPHS {
        let target = draw_range(&mut s, STONES_LO, STONES_HI) as usize;
        let mut seen: HashSet<(i64, i64)> = HashSet::with_capacity(target);
        let mut stones: Vec<(i64, i64, i64)> = Vec::with_capacity(target);
        for _ in 0..(target * 4) {
            if stones.len() >= target {
                break;
            }
            let q = draw_range(&mut s, -COORD_HALF, COORD_HALF);
            let r = draw_range(&mut s, -COORD_HALF, COORD_HALF);
            if seen.insert((q, r)) {
                let player = if stones.len().is_multiple_of(2) {
                    1
                } else {
                    -1
                };
                stones.push((q, r, player));
            }
        }
        let current_player = if splitmix64(&mut s) & 1 == 0 { 1 } else { -1 };
        let moves_remaining = draw_range(&mut s, 1, 200);
        graphs.push(
            build_leaf_graph(
                &stones,
                current_player,
                moves_remaining,
                WIN_LENGTH,
                RADIUS,
                TRUNK_SIZE,
            )
            .expect("every corpus leaf builds (inputs in range)"),
        );
    }
    graphs
}

/// Refuse to bench a corpus that has drifted off the ledger's measured pop shape.
fn assert_corpus_shape(graphs: &[AxisGraph]) {
    let edges: usize = graphs.iter().map(AxisGraph::num_edges).sum();
    let lo = (LEDGER_MEAN_FUSED_EDGES as f64 * (1.0 - SHAPE_BAND)) as usize;
    let hi = (LEDGER_MEAN_FUSED_EDGES as f64 * (1.0 + SHAPE_BAND)) as usize;
    assert!(
        (lo..=hi).contains(&edges),
        "fuse-bench corpus is off the ledger's measured pop shape: {edges} fused edges over \
         {} graphs, band [{lo}, {hi}]. A fuse benched at the wrong edge count is a different \
         measurement.",
        graphs.len(),
    );
}

fn queue_fuse_from_axis_graphs_pop40(c: &mut Criterion) {
    let graphs = build_pop_corpus();
    assert_corpus_shape(&graphs);
    let edges: usize = graphs.iter().map(AxisGraph::num_edges).sum();
    let nodes: usize = graphs.iter().map(AxisGraph::num_nodes).sum();
    println!(
        "fuse-bench corpus: {} graphs, {nodes} nodes, {edges} fused edges",
        graphs.len()
    );

    c.bench_function("queue_fuse_from_axis_graphs_pop40", |b| {
        b.iter(|| {
            let mut wire = GraphWire::from_axis_graphs(black_box(&graphs), 1);
            let arrays = wire.take().expect("a freshly fused wire always has arrays");
            black_box(&arrays);
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100);
    targets = queue_fuse_from_axis_graphs_pop40
}
criterion_main!(benches);
