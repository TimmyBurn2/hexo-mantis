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
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap, clippy::doc_markdown)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use mantis_graph::{build_axis_graph, BuildParams, StoneList};

// Shared dep-free fixture reader (same module the parity tests use).
#[path = "../tests/common/mod.rs"]
#[allow(dead_code)]
mod common;

fn load_positions() -> Vec<(StoneList, BuildParams)> {
    // The frozen predecessor self-play set = exactly the `class == base` cases
    // of the committed fixture (320 corpus positions IN ORDER, identical
    // stones and identical per-position params to what the predecessor bench
    // loaded: wl=6, r=6, trunk=19, corpus cp/mr).
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
    assert!(set.len() == 320, "class==base must select exactly 320 cases, got {}", set.len());
    set
}

fn bench_build(c: &mut Criterion) {
    let set = load_positions();
    let n = set.len();

    // Per-position throughput: one build per iteration, cycling the set so the
    // reported time is ns/pos over the real distribution (mean 490 nodes).
    let mut idx = 0usize;
    let mut group = c.benchmark_group("axis_graph_build");
    group.throughput(criterion::Throughput::Elements(1));
    group.bench_function("per_position", |b| {
        b.iter_batched(
            || {
                let cur = idx % n;
                idx += 1;
                &set[cur]
            },
            |(stones, params)| build_axis_graph(std::hint::black_box(stones), std::hint::black_box(params)),
            BatchSize::SmallInput,
        );
    });
    group.finish();

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
