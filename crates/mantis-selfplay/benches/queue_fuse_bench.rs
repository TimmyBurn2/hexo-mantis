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
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use mantis_graph::AxisGraph;
use mantis_selfplay::queues::{build_leaf_graph, GraphWire};

// The SAME `concat_by_offset` the parity proof pins byte-identical to the whole-batch fuse
// (`tests/prefuse_concat_parity.rs`). Timing a second copy would be timing unverified code.
#[path = "../tests/common/mod.rs"]
mod common;

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
    build_pop_corpus_seeded(CORPUS_SEED)
}

/// The same corpus at a caller-chosen seed. The cross-thread arm needs a FRESH pop per
/// sample — re-handing one corpus would leave it warm and measure nothing.
fn build_pop_corpus_seeded(seed: u64) -> Vec<AxisGraph> {
    let mut s = seed;
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

/// R335(e) Leg 2 — SCOUT §1.2's FALSIFIER, run without a GPU.
///
/// §1.2 argues the in-run fuse's 12.8 ms/pop remainder over cache-hot is not bandwidth but
/// COLD READS: HOT-14's mechanism is that the graphs were written by OTHER worker cores and
/// the server pays a cross-core transfer for every dirty line. The falsifier §1.2 states is
/// *"re-run the criterion arm with the source arrays deliberately evicted"*.
///
/// EVICTION IS THE WRONG INSTRUMENT ON THIS SHAPE and this arm does better. The pop's source
/// arrays are ~37 MB against a 16 MB L3, so they do not fit in either arm — a residency
/// experiment cannot separate them. What DOES separate them is WHO WROTE THE LINES, which is
/// the mechanism HOT-14 actually names. So: a builder thread builds each pop and hands it
/// over, and the fuse is timed on a thread that has never touched those bytes. Against
/// `..._pop40` — same corpus shape, same fuse, same process — the ratio is the mechanism's own
/// number.
///
/// A ratio near 1.0 FALSIFIES the cold-read model on this host and HOT-14 needs re-opening.
fn queue_fuse_cross_thread_pop40(c: &mut Criterion) {
    let (want_tx, want_rx) = mpsc::channel::<u64>();
    let (corpus_tx, corpus_rx) = mpsc::channel::<Vec<AxisGraph>>();
    // The builder thread exists to make the bytes DIRTY IN ANOTHER CORE'S CACHE, which is the
    // only property under test; it exits when the request channel closes.
    let builder = thread::spawn(move || {
        while let Ok(seed) = want_rx.recv() {
            if corpus_tx.send(build_pop_corpus_seeded(seed)).is_err() {
                break;
            }
        }
    });

    let mut seed = CORPUS_SEED;
    c.bench_function("queue_fuse_cross_thread_pop40", |b| {
        b.iter_batched(
            || {
                seed = seed.wrapping_add(1);
                want_tx.send(seed).expect("builder thread alive");
                corpus_rx.recv().expect("builder thread produced a pop")
            },
            |graphs| {
                let mut wire = GraphWire::from_axis_graphs(black_box(&graphs), 1);
                let arrays = wire.take().expect("a freshly fused wire always has arrays");
                black_box(&arrays);
            },
            BatchSize::LargeInput,
        );
    });
    drop(want_tx);
    let _ = builder.join();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .sample_size(100);
    targets = queue_fuse_from_axis_graphs_pop40
}

/// R335(e) Leg 2 — THE NUMBER THAT DECIDES `S-PREFUSE`, and it is not the fuse's cost.
///
/// Under the card the workers fuse and the server CONCATENATES. The server therefore still
/// touches every byte: `concat_by_offset` memcpys the bulk arrays and adds a running base to
/// the index and offset arrays. The card's saving is `fuse − concat` on the server thread,
/// not the whole fuse — so this arm is benched cross-thread, exactly like the fuse arm above,
/// and the two are read as a pair.
fn queue_concat_cross_thread_pop40(c: &mut Criterion) {
    let (want_tx, want_rx) = mpsc::channel::<u64>();
    let (parts_tx, parts_rx) = mpsc::channel::<Vec<mantis_selfplay::queues::GraphWireArrays>>();
    // The builder thread does what a WORKER would under the card: build its slice AND fuse it,
    // so the server-side arm receives already-fused wire written by another core.
    let builder = thread::spawn(move || {
        while let Ok(seed) = want_rx.recv() {
            let graphs = build_pop_corpus_seeded(seed);
            // 5 workers x 8 graphs — the pop composition `n_workers 12` most often produces.
            let parts: Vec<_> = graphs.chunks(8).map(common::fuse).collect();
            if parts_tx.send(parts).is_err() {
                break;
            }
        }
    });

    let mut seed = CORPUS_SEED;
    c.bench_function("queue_concat_cross_thread_pop40", |b| {
        b.iter_batched(
            || {
                seed = seed.wrapping_add(1);
                want_tx.send(seed).expect("builder thread alive");
                parts_rx.recv().expect("builder thread produced a pop")
            },
            |parts| {
                let joined = common::concat_by_offset(black_box(parts));
                black_box(&joined);
            },
            BatchSize::LargeInput,
        );
    });
    drop(want_tx);
    let _ = builder.join();
}

criterion_group! {
    name = cross_thread;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        // The setup builds a whole pop per sample, so this arm is deliberately smaller: the
        // number wanted is a RATIO against the arm above, not a tight CI of its own.
        .sample_size(30);
    targets = queue_fuse_cross_thread_pop40, queue_concat_cross_thread_pop40
}
criterion_main!(benches, cross_thread);
