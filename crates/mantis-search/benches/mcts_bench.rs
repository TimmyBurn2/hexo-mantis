//! Criterion micro-benchmark for MCTS simulations.
//!
//! Run with:
//!   cargo bench -p mantis-search --bench mcts_bench
//!
//! (The `bench_win_detection` micro-bench that lived beside this in the old tree
//! times `Board::check_win` — a core primitive — and belongs with mantis-core;
//! it is excluded here.)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use mantis_core::board::Board;
use mantis_search::MCTSTree;

fn bench_mcts_simulations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_sims_cpu_only");
    for &n in &[100u64, 400, 800] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let board = Board::new();
            let mut tree = MCTSTree::new(1.5);
            tree.new_game(board);
            b.iter(|| {
                tree.run_simulations_cpu_only(n as usize);
                tree.reset();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mcts_simulations);
criterion_main!(benches);
