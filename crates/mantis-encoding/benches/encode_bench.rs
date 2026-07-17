//! WP3 NEW-SIDE encode-kernel bench — mirrors the canonical old-side TIMED-BODY
//! TEMPLATE (wp/WP3/oldside/bench_old_encode/benches/encode_bench.rs) VERBATIM
//! modulo one change: the old `impl Board` methods become mantis-encoding FREE
//! functions. `encode_chain_planes` is already a free fn on both sides.
//! Everything else — inputs, buffer-reuse shape, black_box placement, benchmark
//! names — stays identical so the ±5% comparison times byte-identical work.
//!
//! Inputs are reproduced from the deterministic recipe (the single-source
//! generators in bench_old_encode/src/lib.rs — byte-identical, portable
//! 0.0/1.0 patterns), so the bytes this bench times equal the bytes emitted to
//! wp/WP3/bench_cases/ that the old-side baseline timed (CAPTURE_LOG step 5).
//!
//! The 5 pinned cases (DESIGN §a.10 / P-14), named 1:1 onto the P-14 slots:
//!   encode_state_to_buffer/v6
//!   encode_state_to_buffer_channels/v6_live2_ls
//!   encode_state_to_buffer_channels/v6w25
//!   encode_chain_planes/v6
//!   encode_chain_planes/v6w25
//!
//! Buffer-reuse: `out` is pre-allocated ZERO-INITIALISED once per case and
//! reused across iters (NOT re-zeroed per iter) — replicates the self-play
//! pooled-buffer path; the state kernel relies on the caller's zero-init for
//! history planes 1-7/9-15. IMPL does NOT run the comparison bench (WP2 D3).

use criterion::{criterion_group, criterion_main, Criterion};
use mantis_core::{Board, Ply};
use mantis_encoding::{encode_chain_planes, encode_state_to_buffer, encode_state_to_buffer_channels};
use std::hint::black_box;

const N_CELLS_V6: usize = 361;
const N_CELLS_V6W25: usize = 625;
const TRUNK_V6: i32 = 19;
const TRUNK_V6W25: i32 = 25;
const BENCH_MOVES_REMAINING: u8 = 2;
const BENCH_PLY: u32 = 41;
const CHANNELS_V6_LIVE2_LS: [usize; 4] = [0, 8, 16, 17];
const CHANNELS_V6W25: [usize; 8] = [0, 1, 2, 3, 8, 9, 10, 11];

/// Board carrying only the two fields the state kernels read (`moves_remaining`,
/// `ply`). Stone content is supplied via `planes_2`.
fn bench_board() -> Board {
    let mut b = Board::new();
    b.moves_remaining = BENCH_MOVES_REMAINING;
    b.ply = Ply::new(BENCH_PLY);
    b
}

/// Deterministic 2-plane [my | opp] view (recipe: Knuth multiplicative hash).
fn gen_planes2(n_cells: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 2 * n_cells];
    for i in 0..n_cells {
        let h = (i as u32).wrapping_mul(2_654_435_761) >> 13;
        match h % 5 {
            0 => v[i] = 1.0,
            1 => v[n_cells + i] = 1.0,
            _ => {}
        }
    }
    v
}

/// Deterministic (cur_mask, opp_mask) for encode_chain_planes (spatial hash).
fn gen_chain_masks(n_cells: usize, trunk_sz: i32) -> (Vec<f32>, Vec<f32>) {
    let half = (trunk_sz - 1) / 2;
    let mut cur = vec![0.0f32; n_cells];
    let mut opp = vec![0.0f32; n_cells];
    for q in -half..=half {
        for r in -half..=half {
            let idx = ((q + half) as usize) * (trunk_sz as usize) + (r + half) as usize;
            let hq = (q + half) as u32;
            let hr = (r + half) as u32;
            let h = hq.wrapping_mul(73_856_093) ^ hr.wrapping_mul(19_349_663);
            match h % 5 {
                0 | 1 => cur[idx] = 1.0,
                2 => opp[idx] = 1.0,
                _ => {}
            }
        }
    }
    (cur, opp)
}

// ── Case 1: encode_state_to_buffer — v6 full-board (out 18 x 361) ─────────────
fn bench_encode_state_to_buffer(c: &mut Criterion) {
    let board = bench_board();
    let planes_2 = gen_planes2(N_CELLS_V6);
    let mut group = c.benchmark_group("encode_state_to_buffer");
    group.bench_function("v6", |b| {
        let mut out = vec![0.0f32; 18 * N_CELLS_V6];
        b.iter(|| encode_state_to_buffer(&board, black_box(&planes_2), black_box(&mut out)));
    });
    group.finish();
}

// ── Case 2 & 3: encode_state_to_buffer_channels ──────────────────────────────
fn bench_encode_state_to_buffer_channels(c: &mut Criterion) {
    let board = bench_board();
    let mut group = c.benchmark_group("encode_state_to_buffer_channels");

    // Case 2: v6_live2_ls 4-plane, channels [0,8,16,17], n_cells 361, out 4 x 361.
    {
        let planes_2 = gen_planes2(N_CELLS_V6);
        let channels = CHANNELS_V6_LIVE2_LS;
        let n_cells = N_CELLS_V6;
        group.bench_function("v6_live2_ls", |b| {
            let mut out = vec![0.0f32; channels.len() * n_cells];
            b.iter(|| {
                encode_state_to_buffer_channels(
                    &board,
                    black_box(&planes_2),
                    black_box(&mut out),
                    black_box(&channels),
                    black_box(n_cells),
                );
            });
        });
    }

    // Case 3: v6w25 8-plane, channels [0,1,2,3,8,9,10,11], n_cells 625, out 8 x 625.
    {
        let planes_2 = gen_planes2(N_CELLS_V6W25);
        let channels = CHANNELS_V6W25;
        let n_cells = N_CELLS_V6W25;
        group.bench_function("v6w25", |b| {
            let mut out = vec![0.0f32; channels.len() * n_cells];
            b.iter(|| {
                encode_state_to_buffer_channels(
                    &board,
                    black_box(&planes_2),
                    black_box(&mut out),
                    black_box(&channels),
                    black_box(n_cells),
                );
            });
        });
    }

    group.finish();
}

// ── Case 4 & 5: encode_chain_planes (free fn) ────────────────────────────────
fn bench_encode_chain_planes(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_chain_planes");

    // Case 4: v6 mid-game, n_cells 361, trunk 19, out 6 x 361.
    {
        let (cur, opp) = gen_chain_masks(N_CELLS_V6, TRUNK_V6);
        let n_cells = N_CELLS_V6;
        let trunk_sz = TRUNK_V6;
        group.bench_function("v6", |b| {
            let mut out = vec![0.0f32; 6 * n_cells];
            b.iter(|| {
                encode_chain_planes(
                    black_box(&cur),
                    black_box(&opp),
                    black_box(&mut out),
                    black_box(n_cells),
                    black_box(trunk_sz),
                );
            });
        });
    }

    // Case 5: v6w25 mid-game, n_cells 625, trunk 25, out 6 x 625 (heaviest).
    {
        let (cur, opp) = gen_chain_masks(N_CELLS_V6W25, TRUNK_V6W25);
        let n_cells = N_CELLS_V6W25;
        let trunk_sz = TRUNK_V6W25;
        group.bench_function("v6w25", |b| {
            let mut out = vec![0.0f32; 6 * n_cells];
            b.iter(|| {
                encode_chain_planes(
                    black_box(&cur),
                    black_box(&opp),
                    black_box(&mut out),
                    black_box(n_cells),
                    black_box(trunk_sz),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode_state_to_buffer,
    bench_encode_state_to_buffer_channels,
    bench_encode_chain_planes,
);
criterion_main!(benches);
