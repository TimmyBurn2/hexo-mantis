//! CI gate 5's smoke bench — a REAL hot-path floor. The body is the self-play inner
//! loop's per-move core: clone + legal-move pick + apply_move (inline Zobrist XOR) +
//! last-move-anchored check_win on a mid-game board — the same subjects board_bench
//! floors in depth; this is the 1-second smoke cut of them, so gate 5 exercises
//! production code, not a fold stub.
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

mod common;
use common::board_with_n_stones;

fn hot_path_smoke(c: &mut Criterion) {
    let base = board_with_n_stones(20);
    c.bench_function("apply_move_plus_check_win_20_stones", |b| {
        b.iter(|| {
            let mut board = base.clone();
            let mv = *board
                .legal_moves_set()
                .iter()
                .min()
                .expect("no legal moves");
            black_box(board.apply_move(mv.0, mv.1)).expect("apply failed");
            black_box(board.check_win())
        })
    });
}
criterion_group!(benches, hot_path_smoke);
criterion_main!(benches);
