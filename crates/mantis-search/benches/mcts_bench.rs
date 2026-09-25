//! Criterion micro-benchmark for MCTS simulations.
//!
//! Run with:
//!   cargo bench -p mantis-search --bench mcts_bench
//!
//! `expand_leaf` is a recorded characterisation with **no abort attached**: it times the dense
//! expand rule against the legal-set rule (a hash lookup per off-window cell), a structural
//! differential that bounds the explainable part of the eval-round wall-clock move.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use mantis_core::board::Board;
use mantis_core::BoardGeometry;
use mantis_search::{LegalSetPolicy, MCTSTree};

/// `gnn_axis_v1`'s geometry (crates/mantis-encoding/src/registry.toml): radius 6, trunk 19,
/// hence `policy_logit_count` 362 and a flat index ≥ 361 is exactly "off-window".
const TRUNK_SZ: i32 = 19;
const POLICY_STRIDE: usize = 362;
const OFF_WINDOW_FLAT: usize = 361;

/// A dispersed radius-6 position, built by a DETERMINISTIC uniform-legal walk —
/// the same shape as the random-legal playouts that measured 364 legal moves at
/// 8 stones and 1294 at 32. Inline LCG rather than `rand`, so the bench carries
/// its own reproducibility and takes no seeded-stream dependency.
fn dispersed_board(n_stones: usize) -> Board {
    let mut board = Board::with_geometry(BoardGeometry {
        legal_move_radius: 6,
        cluster_window_size: TRUNK_SZ as usize,
    });
    let mut state: u64 = 0x2026_0731;
    for _ in 0..n_stones {
        let legal = board.legal_moves();
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let (q, r) = legal[(state >> 33) as usize % legal.len()];
        board
            .apply_move(q, r)
            .expect("the walk picked a legal move");
    }
    board
}

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

fn bench_expand_leaf(c: &mut Criterion) {
    let mut group = c.benchmark_group("expand_leaf");
    for &n_stones in &[8usize, 32] {
        let board = dispersed_board(n_stones);
        let legal = board.legal_moves();
        let n_legal = legal.len();
        let center = board.window_center();

        // Uniform priors: this group times the EXPAND RULES, not a net. The ls half carries every
        // off-window legal cell, as the graph producer's overflow does here (0 absent coords at 4/4).
        let dense: Vec<f32> = vec![1.0 / POLICY_STRIDE as f32; POLICY_STRIDE];
        let mut ls = LegalSetPolicy {
            dense: dense.clone(),
            ..LegalSetPolicy::default()
        };
        for &(q, r) in &legal {
            if board.window_flat_idx(q, r) >= OFF_WINDOW_FLAT {
                ls.overflow.insert((q, r), 1.0 / n_legal as f32);
            }
        }

        // One pending leaf per iteration; the expand consumes `pending`, so the
        // tree is rebuilt in the (untimed) setup half of `iter_batched_ref`.
        let armed = || {
            let mut tree = MCTSTree::new(1.5);
            tree.new_game(board.clone());
            tree.select_leaves(1)
                .expect("select_leaves: no desync in this fixture");
            tree
        };
        group.bench_with_input(BenchmarkId::new("dense", n_legal), &n_legal, |b, _| {
            b.iter_batched_ref(
                armed,
                |tree| tree.expand_and_backup(std::slice::from_ref(&dense), &[0.0]),
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("ls_graph", n_legal), &n_legal, |b, _| {
            b.iter_batched_ref(
                armed,
                |tree| {
                    tree.expand_and_backup_ls_at(
                        std::slice::from_ref(&ls),
                        &[0.0],
                        std::slice::from_ref(&center),
                        TRUNK_SZ,
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mcts_simulations, bench_expand_leaf);
criterion_main!(benches);
