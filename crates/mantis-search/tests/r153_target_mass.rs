//! Regression oracle: the exported training target must drop no visit mass.
//!
//! No reference implementation is needed — `get_policy_ls` normalises by the total visit
//! count over ALL children, so a no-drop export sums to exactly 1.0 and any deficit IS the
//! dropped mass. Prereg and measurement records live at
//! `docs/design/measurements/{PREREG,MEASUREMENT}_R153.md`.

use mantis_core::board::{Board, BoardGeometry};
use mantis_encoding::lookup_or_panic;
use mantis_search::MCTSTree;

const N_SIMS: usize = 150; // run5's deploy_sims
const LEAF_BATCH: usize = 8;
const TEMPERATURE: f32 = 1.0; // the training export's branch
const TOL: f64 = 1e-6; // verdict threshold

/// One position's measurement.
struct Row {
    ply: u32,
    n_children: usize,
    n_legal: usize,
    dropped_mass: f64,
}

fn geometry_for(enc: &str) -> (BoardGeometry, usize) {
    let spec = lookup_or_panic(enc);
    (
        BoardGeometry {
            legal_move_radius: spec.legal_move_radius as i32,
            cluster_threshold: spec.cluster_threshold.unwrap_or(5) as i32,
            cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
        },
        spec.policy_logit_count,
    )
}

fn run_uniform_search(tree: &mut MCTSTree, n_actions: usize) {
    let uniform = vec![1.0_f32 / n_actions as f32; n_actions];
    let mut done = 0;
    while done < N_SIMS {
        let take = LEAF_BATCH.min(N_SIMS - done);
        let boards = tree
            .select_leaves(take)
            .expect("select_leaves: no desync in this fixture");
        if boards.is_empty() {
            break;
        }
        let policies: Vec<Vec<f32>> = (0..boards.len()).map(|_| uniform.clone()).collect();
        let values = vec![0.0_f32; boards.len()];
        tree.expand_and_backup(&policies, &values);
        done += boards.len();
    }
}

/// Measure ONE root position. Returns None if the root never expanded (terminal / no legal).
fn measure(board: &Board, n_actions: usize, ply: u32) -> Option<Row> {
    let mut tree = MCTSTree::new(1.5); // pyo3 ctor default; deploy head's value
    tree.new_game(board.clone());
    run_uniform_search(&mut tree, n_actions);

    let root = &tree.pool[0];
    if !root.is_expanded() {
        return None;
    }
    let n_ch = root.n_children as usize;
    if n_ch == 0 {
        return None;
    }

    let policy = tree.get_policy_ls(TEMPERATURE, n_actions);
    let exported: f64 = policy.dense.iter().map(|&p| p as f64).sum::<f64>()
        + policy.overflow.values().map(|&p| p as f64).sum::<f64>();
    let dropped_mass = 1.0 - exported;

    // A surplus is a double-count, not a drop.
    assert!(
        exported <= 1.0 + TOL,
        "ply {ply}: exported mass {exported} EXCEEDS 1.0 — double-count, not a drop; \
         the PREREG §2 invariant does not hold and the instrument is wrong"
    );

    Some(Row {
        ply,
        n_children: n_ch,
        n_legal: board.legal_moves().len(),
        dropped_mass,
    })
}

/// Dispersed tail probe: drive stones apart so the legal set grows past the 361-cell
/// in-window ceiling, a regime a game-only sample never reaches.
fn dispersed_and_measure(enc: &str, max_plies: u32) -> Vec<Row> {
    let (geom, n_actions) = geometry_for(enc);
    let mut board = Board::with_geometry(geom);
    let mut rows = Vec::new();
    for ply in 0..max_plies {
        let legal = board.legal_moves();
        if legal.is_empty() {
            break;
        }
        if let Some(row) = measure(&board, n_actions, ply) {
            rows.push(row);
        }
        // Farthest from the window centre: deterministic, and disperses monotonically.
        let (cq, cr) = board.window_center();
        let &(q, r) = legal
            .iter()
            .max_by_key(|&&(q, r): &&(i32, i32)| {
                let (dq, dr) = (q - cq, r - cr);
                dq.abs().max(dr.abs()).max((dq + dr).abs())
            })
            .unwrap();
        if board.apply_move(q, r).is_err() {
            break;
        }
    }
    rows
}

/// Play one complete game at `enc`'s geometry, measuring EVERY root position.
fn play_and_measure(enc: &str, seed: u64, max_plies: u32) -> Vec<Row> {
    let (geom, n_actions) = geometry_for(enc);
    let mut board = Board::with_geometry(geom);
    let mut rows = Vec::new();
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);

    for ply in 0..max_plies {
        let legal = board.legal_moves();
        if legal.is_empty() {
            break;
        }
        if let Some(row) = measure(&board, n_actions, ply) {
            rows.push(row);
        }
        // Deterministic LCG playout — reproducible at a fixed seed.
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let (q, r) = legal[(state >> 33) as usize % legal.len()];
        if board.apply_move(q, r).is_err() {
            break;
        }
    }
    rows
}

fn report(enc: &str, rows: &[Row]) -> (usize, f64, f64, usize) {
    let affected = rows.iter().filter(|r| r.dropped_mass > TOL).count();
    let mut masses: Vec<f64> = rows.iter().map(|r| r.dropped_mass).collect();
    masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if masses.is_empty() {
        0.0
    } else {
        masses[masses.len() / 2]
    };
    let max = masses.last().copied().unwrap_or(0.0);

    let max_legal = rows.iter().map(|r| r.n_legal).max().unwrap_or(0);
    println!(
        "\n=== {enc} — {} positions, {affected} affected ===",
        rows.len()
    );
    println!("  dropped_mass: median {median:.6}  max {max:.6}");
    println!("  max n_legal reached: {max_legal}  (PREREG abort 1 needs >361)");
    for r in rows.iter().filter(|r| r.dropped_mass > TOL).take(12) {
        println!(
            "    ply {:>3}  n_legal {:>5}  n_children {:>4}  dropped {:.6}",
            r.ply, r.n_legal, r.n_children, r.dropped_mass
        );
    }
    (affected, median, max, max_legal)
}

#[test]
fn r153_characterize_exported_target_dropped_mass() {
    // run5's own encoding, then run6's identity row at the wider radius.
    let encodings = ["gnn_axis_v1", "gnn_axis_r8"];
    let seeds = [20_260_731_u64, 8_675_309, 42]; // 3 distinct games (LAW-04)

    let mut any_drop = false;
    let mut total_positions = 0usize;

    for enc in encodings {
        let mut rows = Vec::new();
        for seed in seeds {
            rows.extend(play_and_measure(enc, seed, 128));
        }
        // Tail probe — the regime a game-only sample misses.
        let tail = dispersed_and_measure(enc, 96);
        println!("  --- dispersed tail ---");
        let (tail_affected, _tm, tail_max, tail_max_legal) =
            report(&format!("{enc}/dispersed"), &tail);
        rows.extend(tail);

        total_positions += rows.len();
        let (affected, _median, _max, max_legal) = report(enc, &rows);
        if affected > 0 || tail_affected > 0 {
            any_drop = true;
        }
        // The permanent regression assertion.
        for r in &rows {
            assert!(
                r.dropped_mass <= TOL,
                "{enc}: ply {} (n_legal {}, n_children {}) drops {:.6} target mass \
                 (> {TOL}) — the no-drop export law (records.rs:468-479, R34/R153) is \
                 violated",
                r.ply,
                r.n_legal,
                r.n_children,
                r.dropped_mass
            );
        }
        // Abort 1: the sample MUST reach the >361-legal regime.
        assert!(
            max_legal > 361 || tail_max_legal > 361,
            "{enc}: sample never reached the >361-legal regime (max {max_legal}, tail \
             {tail_max_legal}) — PREREG abort 1: HOLD, this sample is not representative"
        );
        // Abort 3: the same seed must give the same numbers.
        let a: Vec<f64> = play_and_measure(enc, seeds[0], 64)
            .iter()
            .map(|r| r.dropped_mass)
            .collect();
        let b: Vec<f64> = play_and_measure(enc, seeds[0], 64)
            .iter()
            .map(|r| r.dropped_mass)
            .collect();
        assert_eq!(
            a, b,
            "{enc}: instrument is NOT deterministic at a fixed seed"
        );
        println!("  tail max dropped_mass {tail_max:.6}");
    }

    assert!(
        total_positions > 0,
        "the instrument measured nothing — sample is empty"
    );
    println!(
        "\n=== R153 VERDICT INPUTS ===\n  any position with dropped_mass > {TOL}: {any_drop}\
         \n  total positions measured: {total_positions}"
    );
}
