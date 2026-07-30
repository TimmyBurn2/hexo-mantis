//! CI gate 5's smoke bench — a REAL hot-path floor since WPBOX CB-3 (R29: "replace
//! smoke_bench CI stub with a real hot-path floor"). The body is the self-play inner
//! loop's per-move core: clone + legal-move pick + apply_move (inline Zobrist XOR) +
//! last-move-anchored check_win on a mid-game board — the same subjects board_bench
//! floors in depth; this is the 1-second smoke cut of them, so gate 5 exercises
//! production code, not a fold stub.
use criterion::{criterion_group, criterion_main, Criterion};
use mantis_core::board::Board;
use std::hint::black_box;

/// Deterministic mid-game board: lexicographic-minimum legal move each ply
/// (board_bench's own helper, restated).
fn board_with_n_stones(n_stones: usize) -> Board {
    let mut b = Board::new();
    for _ in 0..n_stones {
        let mv = *b.legal_moves_set().iter().min().expect("no legal moves");
        b.apply_move(mv.0, mv.1).expect("apply failed");
    }
    b
}

fn hot_path_smoke(c: &mut Criterion) {
    let base = board_with_n_stones(20);
    c.bench_function("apply_move_plus_check_win_20_stones", |b| {
        b.iter(|| {
            let mut board = base.clone();
            let mv = *board.legal_moves_set().iter().min().expect("no legal moves");
            black_box(board.apply_move(mv.0, mv.1)).expect("apply failed");
            black_box(board.check_win())
        })
    });
}
criterion_group!(benches, hot_path_smoke);
criterion_main!(benches);
