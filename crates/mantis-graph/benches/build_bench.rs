//! BUILD-HOT criterion bench for the once-per-leaf axis-graph builder.
//!
//! Predecessor verdict BUILD-HOT (predecessor measurement, private records
//! archive): the strix Rust-builder proxy is 0.539 ms/pos = 77-161% of the GNN
//! forward, so the build is a hot path and the builder carries a perf
//! sub-package. This bench deserializes the REAL predecessor self-play
//! position set ONCE (the 320 as-recorded base cases from the committed
//! binary fixture `tests/fixtures/graph_parity/inputs.bin`), then times build
//! only (no I/O, no parallelism — the caller parallelizes over leaves).
//!
//! Run (the committed workspace release profile already uses
//! `panic = "unwind"`, so no RUSTFLAGS override is needed):
//!
//!   cargo bench -p mantis-graph --bench build_bench --locked
//!
//! Targets: beat the 0.539 ms/pos strix proxy; contract budget ≤1.5 ms/pos.
//! Median ns/pos is the headline number.

// Fixture-bounded binary-fixture-parse casts; silence the pedantic cast lints.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::doc_markdown
)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use mantis_graph::{
    build_axis_graph, build_axis_graph_verified_by, BuildParams, ProducerVerify, StoneList,
};

// Shared dep-free fixture reader (same module the parity tests use).
#[path = "../tests/common/mod.rs"]
#[allow(dead_code)]
mod common;

#[path = "../tests/common/positions.rs"]
mod positions;

fn load_positions() -> Vec<(StoneList, BuildParams)> {
    // The frozen predecessor self-play set = the fixture's `class == base` cases (320 corpus
    // positions IN ORDER, same stones/params as the predecessor bench: wl=6, r=6, trunk=19, cp/mr).
    let root = common::fixture_root();
    common::verify_fixture_root(&root).unwrap_or_else(|e| panic!("{e}"));
    let cases = common::read_inputs_bin(&root.join("inputs.bin")).unwrap_or_else(|e| panic!("{e}"));
    let set: Vec<(StoneList, BuildParams)> = cases
        .into_iter()
        .filter(|c| c.class == common::CLASS_BASE)
        .map(|c| {
            let params = BuildParams {
                win_length: c.win_length,
                radius: c.radius,
                current_player: c.current_player,
                moves_remaining: c.moves_remaining,
                trunk_size: c.trunk_size,
            };
            (StoneList { stones: c.stones }, params)
        })
        .collect();
    assert!(
        set.len() == 320,
        "class==base must select exactly 320 cases, got {}",
        set.len()
    );
    set
}

/// The recorded radius-8 leaf positions, each at its own side to move and stones to place.
fn load_r8_positions() -> Vec<(StoneList, BuildParams)> {
    let recorded = positions::read_positions_r8().unwrap_or_else(|e| panic!("{e}"));
    recorded
        .into_iter()
        .map(|p| {
            let params = BuildParams {
                radius: 8,
                current_player: p.to_move,
                moves_remaining: p.moves_remaining,
                ..BuildParams::V1_GEOMETRY
            };
            (StoneList { stones: p.stones }, params)
        })
        .collect()
}

/// One build per iteration, cycling `set`, so the time is ns per position over its distribution.
fn bench_per_position(
    c: &mut Criterion,
    group_name: &str,
    set: &[(StoneList, BuildParams)],
    verify: ProducerVerify,
) {
    let n = set.len();
    let mut idx = 0usize;
    let mut group = c.benchmark_group(group_name);
    group.throughput(criterion::Throughput::Elements(1));
    group.bench_function("per_position", |b| {
        b.iter_batched(
            || {
                let cur = idx % n;
                idx += 1;
                &set[cur]
            },
            |(stones, params)| {
                build_axis_graph_verified_by(
                    std::hint::black_box(stones),
                    std::hint::black_box(params),
                    verify,
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_build(c: &mut Criterion) {
    let set = load_positions();
    let n = set.len();

    // The predecessor set at radius 6 (mean 490 nodes), then the recorded radius-8 positions.
    let r8 = load_r8_positions();
    bench_per_position(c, "axis_graph_build", &set, ProducerVerify::Builder);
    bench_per_position(c, "axis_graph_build_r8", &r8, ProducerVerify::Builder);
    // Where the consumer re-runs the edge-geometry check on every batch, the builder's own verify is skipped.
    bench_per_position(
        c,
        "axis_graph_build_r8_unverified",
        &r8,
        ProducerVerify::ConsumerEveryBatch,
    );

    // Whole-set sweep: build all N once, for a stable aggregate median.
    let mut g2 = c.benchmark_group("axis_graph_build_full_set");
    g2.throughput(criterion::Throughput::Elements(n as u64));
    g2.bench_function("all_positions", |b| {
        b.iter(|| {
            for (stones, params) in &set {
                std::hint::black_box(build_axis_graph(stones, params));
            }
        });
    });
    g2.finish();
}

criterion_group!(benches, bench_build);
criterion_main!(benches);
