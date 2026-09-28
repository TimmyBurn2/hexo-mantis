//! The tactics module's costs on the goldens positions; per-solve medians and p99s print as JSON lines.

use std::time::Instant;

use criterion::{criterion_group, criterion_main, Criterion};
use mantis_core::board::Board;
use mantis_search::tactics::{analyze, TurnSolver};

const TABLE: usize = 1 << 18;

fn corpus() -> Vec<Board> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tactics/turn_solver_goldens.jsonl"
    );
    let text = std::fs::read_to_string(path).expect("the goldens fixture is tracked");
    text.lines()
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).expect("a fixture row is JSON");
            let mut b = Board::new();
            b.set_legal_move_radius(v["radius"].as_i64().expect("radius") as i32);
            for s in v["stones"].as_array().expect("stones") {
                let q = s[0].as_i64().expect("q") as i32;
                let r = s[1].as_i64().expect("r") as i32;
                b.apply_move(q, r).expect("a recorded stone is placeable");
            }
            b
        })
        .collect()
}

fn quiet_two_stone(boards: &[Board]) -> Vec<&Board> {
    boards
        .iter()
        .filter(|b| {
            let t = analyze(b);
            b.moves_remaining == 2 && t.terminal.is_none() && t.forced.is_empty()
        })
        .collect()
}

/// Median, p99, max per-solve µs (each the middle of three runs): `search_quiet` alone, or `solve` with `analyze`.
fn distribution(boards: &[&Board], turns: u8, nodes: u64, search_only: bool) -> (f64, f64, f64) {
    let mut solver = TurnSolver::new(TABLE);
    let mut us: Vec<f64> = boards
        .iter()
        .map(|b| {
            let mut runs = [0f64; 3];
            for run in &mut runs {
                solver.clear();
                let t0 = Instant::now();
                if search_only {
                    std::hint::black_box(solver.search_quiet(b, turns, nodes));
                } else {
                    std::hint::black_box(solver.solve(b, turns, nodes));
                }
                *run = t0.elapsed().as_secs_f64() * 1e6;
            }
            runs.sort_by(f64::total_cmp);
            runs[1]
        })
        .collect();
    us.sort_by(f64::total_cmp);
    let at = |q: f64| us[((us.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.99), us[us.len() - 1])
}

fn bench_tactics(c: &mut Criterion) {
    let boards = corpus();
    let heavy: Vec<Board> = boards
        .iter()
        .filter(|b| b.ply.index() >= 80)
        .cloned()
        .collect();
    let quiet = quiet_two_stone(&boards);
    let mut group = c.benchmark_group("tactics");
    group.bench_function("analyze_per_leaf_all", |b| {
        b.iter(|| {
            for board in &boards {
                std::hint::black_box(analyze(board));
            }
        });
    });
    group.bench_function("analyze_per_leaf_80_stones_up", |b| {
        b.iter(|| {
            for board in &heavy {
                std::hint::black_box(analyze(board));
            }
        });
    });
    group.bench_function("search_2_64_quiet_corpus", |b| {
        let mut solver = TurnSolver::new(TABLE);
        b.iter(|| {
            for board in &quiet {
                solver.clear();
                std::hint::black_box(solver.search_quiet(board, 2, 64));
            }
        });
    });
    group.finish();
    for (turns, nodes, search_only) in [
        (2u8, 64u64, true),
        (2, 64, false),
        (8, 20_000, true),
        (8, 20_000, false),
    ] {
        let (p50, p99, max) = distribution(&quiet, turns, nodes, search_only);
        println!(
            "{{\"bench\": \"tactics_solve_distribution\", \"turns\": {turns}, \"nodes\": {nodes}, \
             \"search_only\": {search_only}, \"positions\": {}, \"median_us\": {p50:.1}, \"p99_us\": {p99:.1}, \
             \"max_us\": {max:.1}}}",
            quiet.len()
        );
    }
    println!(
        "{{\"bench\": \"tactics_corpus\", \"positions\": {}, \"heavy\": {}, \"quiet_two_stone\": {}}}",
        boards.len(),
        heavy.len(),
        quiet.len()
    );
}

criterion_group!(benches, bench_tactics);
criterion_main!(benches);
