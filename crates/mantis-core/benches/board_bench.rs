//! Criterion benchmarks for board operations.
//!
//! Groups (WP2 parity protocol — identical bodies compiled against the old and
//! the new crate):
//!   1. win_detection — Board::check_win() (last-move-anchored HashMap scan)
//!   2. board_clone — Board::clone() cost at various game depths
//!   3. reconstruct_board_path_replay — clone root + replay D moves
//!   4. zobrist_incremental — apply_move as an incremental-hash proxy
//!   5. legal_move_gen — full legal-set rebuild (invalidate + rebuild); the
//!      cache-construct gate (envelope ≤ +2% new vs old on every case)
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mantis_core::board::Board;
use std::hint::black_box;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a board with `n_stones` placed by always picking the lexicographic-minimum
/// legal move.  Guaranteed collision-free and deterministic.
fn board_with_n_stones(n_stones: usize) -> Board {
    let mut b = Board::new();
    for _ in 0..n_stones {
        let mv = *b.legal_moves_set()
            .iter()
            .min()
            .expect("no legal moves");
        b.apply_move(mv.0, mv.1).expect("apply failed");
    }
    b
}

// ── 1. Win detection: last-move-anchored HashMap scan ────────────────────────

fn bench_win_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("win_detection");

    for &n_stones in &[10usize, 20, 40] {
        let board = board_with_n_stones(n_stones);

        group.bench_with_input(
            BenchmarkId::new("hashmap_check_win", n_stones),
            &n_stones,
            |b, _| b.iter(|| black_box(board.check_win())),
        );
    }

    // Win case: 6-in-a-row on E axis.
    let mut win_board = Board::new();
    win_board.apply_move(0, 0).unwrap();
    win_board.apply_move(-9, 5).unwrap();
    win_board.apply_move(-9, 6).unwrap();
    win_board.apply_move(1, 0).unwrap();
    win_board.apply_move(2, 0).unwrap();
    win_board.apply_move(-9, 7).unwrap();
    win_board.apply_move(-9, 8).unwrap();
    win_board.apply_move(3, 0).unwrap();
    win_board.apply_move(4, 0).unwrap();
    win_board.apply_move(-9, -5).unwrap();
    win_board.apply_move(-9, -6).unwrap();
    win_board.apply_move(5, 0).unwrap();

    group.bench_function("hashmap_check_win_TRUE", |b| {
        b.iter(|| black_box(win_board.check_win()))
    });

    group.finish();
}

// ── 2. Board::clone() cost — the MCTS reconstruct_board bottleneck ────────────

fn bench_board_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("board_clone");

    for &n_stones in &[5usize, 15, 30, 60] {
        let board = board_with_n_stones(n_stones);

        group.bench_with_input(
            BenchmarkId::new("clone_n_stones", n_stones),
            &n_stones,
            |b, _| b.iter(|| black_box(board.clone())),
        );
    }

    group.finish();
}

// ── 3. Reconstruct-board path replay cost (simulates MCTS select_one_leaf) ───

/// Simulate what a tree reconstruct does: clone root + replay D moves.
fn simulate_reconstruct(root: &Board, path: &[(i32, i32)]) -> Board {
    let mut b = root.clone();
    for &(q, r) in path {
        b.apply_move(q, r).unwrap();
    }
    b
}

fn bench_reconstruct_path_replay(c: &mut Criterion) {
    let mut group = c.benchmark_group("reconstruct_board_path_replay");

    // Build a root board and a sequence of legal moves to replay.
    let root = Board::new();

    // Pre-compute a sequence of moves that can be applied from an empty board.
    // Interleave P1 and P2 moves to respect turn structure.
    let all_moves: Vec<(i32, i32)> = vec![
        (0, 0),             // P1 single first move
        (-1, -1), (-2, -2), // P2 turn
        (1, 1), (2, 2),     // P1 turn
        (-3, -3), (-4, -4), // P2 turn
        (3, 3), (4, 4),     // P1 turn
        (-5, -5), (-6, -6), // P2 turn
        (5, 0), (6, 0),     // P1 turn
        (-7, 1), (-8, 1),   // P2 turn
        (0, 5), (0, 6),     // P1 turn
        (1, -5), (1, -6),   // P2 turn
    ];

    for &depth in &[5usize, 10, 15, 20] {
        let path = &all_moves[..depth.min(all_moves.len())];

        group.bench_with_input(
            BenchmarkId::new("clone_and_replay", depth),
            &depth,
            |b, _| b.iter(|| black_box(simulate_reconstruct(&root, path))),
        );
    }

    group.finish();
}

// ── 4. Zobrist: incremental XOR (confirms near-zero marginal cost) ────────────

fn bench_zobrist_incremental(c: &mut Criterion) {
    // apply_move already XORs the Zobrist key inline.
    // This bench measures the total apply_move cost as a proxy.
    let mut group = c.benchmark_group("zobrist_incremental");

    group.bench_function("apply_move_with_zobrist_update", |b| {
        b.iter(|| {
            let mut board = Board::new();
            black_box(board.apply_move(0, 0)).unwrap();
            black_box(board.zobrist_hash)
        })
    });

    group.finish();
}

// ── 5. Legal-move generation: full rebuild (the cache-construct gate) ─────────

/// Invalidate + rebuild each iteration. `set_legal_move_radius` marks the cache
/// dirty through the public API (same semantics both sides); the following
/// `legal_moves_set()` pays the full rebuild.
fn bench_legal_move_gen(c: &mut Criterion) {
    let mut group = c.benchmark_group("legal_move_gen");

    for &n_stones in &[10usize, 30, 60] {
        let mut board = board_with_n_stones(n_stones);
        group.bench_with_input(
            BenchmarkId::new("rebuild_stones_r5", n_stones),
            &n_stones,
            |b, _| {
                b.iter(|| {
                    board.set_legal_move_radius(5); // invalidate (default radius)
                    black_box(board.legal_moves_set().len())
                })
            },
        );
    }

    for &radius in &[4i32, 6, 8] {
        let mut board = board_with_n_stones(30);
        group.bench_with_input(
            BenchmarkId::new("rebuild_radius_s30", radius),
            &radius,
            |b, _| {
                b.iter(|| {
                    board.set_legal_move_radius(radius); // invalidate at this radius
                    black_box(board.legal_moves_set().len())
                })
            },
        );
    }

    group.finish();
}

// ── Registration ──────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_win_detection,
    bench_board_clone,
    bench_reconstruct_path_replay,
    bench_zobrist_incremental,
    bench_legal_move_gen,
);
criterion_main!(benches);
