//! WP5 dense sample-path bench. TWO benches, matched to the old bodies:
//!
//!  1. `scatter_state_v6_b512` — the OPERATIVE gate (CAPTURE_LOG §B.1): 512×
//!     the ported `sym::apply_symmetry_state` over a v6 8-plane state buffer
//!     (2888 u16), rotating D6 sym 0..11. GIL/numpy-free — the new-side scatter
//!     sub-cost decomposition compared within ±5% to the old scatter estimate
//!     (419.96 µs).
//!  2. `dense_sample_core_v6_b512_aug` — INFORMATIONAL: the full new
//!     `sample_batch_core(512, augment=true)` returning the owned `SampleBatch`,
//!     buffer built with a fixed-seed `StdRng` (0x5A_11_9E), 100k v6 capacity
//!     filled with `push_for_test`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;

use mantis_selfplay::replay::sample::apply_symmetry_state;
use mantis_selfplay::replay::sym::SymTables;
use mantis_selfplay::replay::ReplayBuffer;

/// Bench 1 — the operative scatter sub-cost gate. Body BYTE-IDENTICAL to the
/// dispatcher's old-side `scatter_state_v6_b512` (CAPTURE_LOG §B.1): src content,
/// `black_box` on both call args, and `black_box(&dst)` OUTSIDE the loop.
fn scatter_state_v6_b512(c: &mut Criterion) {
    // v6 D6 tables (19×19, 8 planes). state stride = n_planes * n_cells = 8*361 = 2888.
    let tables = SymTables::new();
    let n_cells = tables.n_cells;
    let n_planes = tables.n_planes;
    let stride = n_cells * n_planes;
    let src: Vec<u16> = (0..stride).map(|i| (i % 65536) as u16).collect();
    let mut dst = vec![0u16; stride];
    c.bench_function("scatter_state_v6_b512", |b| {
        b.iter(|| {
            for i in 0..512usize {
                let sym = i % 12; // rotate through the 12 D6 elements
                apply_symmetry_state(black_box(&src), black_box(&mut dst), sym, &tables);
            }
            black_box(&dst);
        });
    });
}

/// Bench 2 — informational full-sample core (scatter/augment only; marshaling is
/// WP7). Buffer filled so `weighted_sample_one` accepts on first draw
/// (weight 1.0), game_id=-1 so dedup does batch_size skips → index-independent
/// work; augment=true drives the apply_sym scatter under the R245(c) per-record gate.
fn dense_sample_core_v6_b512_aug(c: &mut Criterion) {
    let capacity = 100_000usize;
    let mut buf = ReplayBuffer::new(capacity, "v6");
    buf.rng = StdRng::seed_from_u64(0x5A_11_9E);
    for _ in 0..capacity {
        buf.push_for_test(0.0, 0, true); // game_id=-1, weight 1.0
    }
    assert_eq!(buf.size(), capacity);

    // Warm sample. R245(c): the draw is gated per record, and `push_for_test` writes
    // all-NEUTRAL rows, so every record in this buffer is COMPACT and the draw is the
    // full 12-element group — at batch 512 a non-identity sym is drawn with
    // probability 1 − 12^−512 and the scatter is always exercised. NOTE for LAW-09:
    // the work mix here is the PRE-R245 one (the same 12-element draw over the same
    // full/cell-dropping element mix) plus one `compact[idx]` byte read per draw. It
    // is NOT the raised, all-full-window mix the un-gated R245 restriction produced —
    // a floor captured against that intermediate would be wrong for this code.
    let _ = buf.sample_batch_core(512, true).expect("warm sample");

    c.bench_function("dense_sample_core_v6_b512_aug", |b| {
        b.iter(|| {
            black_box(buf.sample_batch_core(black_box(512), true).expect("sample"));
        });
    });
}

criterion_group!(benches, scatter_state_v6_b512, dense_sample_core_v6_b512_aug);
criterion_main!(benches);
