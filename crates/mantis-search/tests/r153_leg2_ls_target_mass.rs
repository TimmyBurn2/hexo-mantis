// R8 >300 justify: the measurement harness (frozen semantics) and the flipped assertion arm
// live in ONE file so the regression oracle and the prereg'd measurement it derives from can
// never drift apart.
//! Dropped target mass through each encoding's PRODUCTION expand.
//!
//! A permanent regression oracle: every sampled row must satisfy `dropped_mass <= 1e-6` and
//! the degenerate count must be 0, on both production-expand encodings. The prereg and
//! measurement artifacts are committed verbatim under `docs/design/measurements/`.
//!
//! A measurement clears an encoding ONLY when driven through that encoding's production
//! expand: under the DENSE expand off-window cells get `sort_prior = 0.0` and are truncated
//! by the per-node child cap, so a zero there is structural rather than a clearance. Instrument
//! abort 4, the off-window-child guard, is what keeps that distinction honest forever.
//!
//! | encoding | expand |
//! |---|---|
//! | `gnn_axis_v1` | `expand_and_backup_ls_at` (production) |
//! | `gnn_axis_r8` | `expand_and_backup_ls` (unframed) |

use mantis_core::board::{Board, BoardGeometry};
use mantis_encoding::lookup_or_panic;
use mantis_search::{LegalSetPolicy, MCTSTree};

const N_SIMS: usize = 50; // run5 selfplay.mcts.n_simulations (target generation), NOT deploy_sims
const LEAF_BATCH: usize = 8;
const TEMPERATURE: f32 = 1.0;
const TOL: f64 = 1e-6;

#[derive(Clone, Copy)]
enum Expand {
    /// Frame-explicit: legal-set expand against the BUILDER's window centre.
    LsAt,
    /// Unframed: legal-set expand that reads the prior in the Board's own window frame.
    Ls,
}

struct Row {
    ply: u32,
    n_legal: usize,
    n_children: usize,
    dropped_mass: f64,
    /// Off-window children that actually exist in the tree — what abort 4 checks.
    offwindow_children: usize,
}

fn geometry_for(enc: &str) -> (BoardGeometry, usize, i32) {
    let spec = lookup_or_panic(enc);
    (
        BoardGeometry {
            legal_move_radius: spec.legal_move_radius as i32,
            cluster_threshold: spec.cluster_threshold.unwrap_or(5) as i32,
            cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
        },
        spec.policy_logit_count,
        spec.trunk_size as i32,
    )
}

/// The NO-DROP uniform legal-set policy: `1/n_legal` on EVERY legal move, split into the
/// in-window `dense` half and the coord-keyed off-window `overflow` half, matching the
/// producer's own contract. This is what makes off-window cells become real children with
/// real priors — the condition a dense expand cannot create.
fn no_drop_uniform(board: &Board, n_actions: usize) -> LegalSetPolicy {
    let legal = board.legal_moves();
    let p = 1.0_f32 / legal.len().max(1) as f32;
    let mut ls = LegalSetPolicy { dense: vec![0.0; n_actions], overflow: Default::default() };
    for (q, r) in legal {
        let idx = board.window_flat_idx(q, r);
        if idx < n_actions {
            ls.dense[idx] = p;
        } else {
            ls.overflow.insert((q, r), p);
        }
    }
    ls
}

fn run_search(tree: &mut MCTSTree, n_actions: usize, trunk_sz: i32, mode: Expand) {
    let mut done = 0;
    while done < N_SIMS {
        let take = LEAF_BATCH.min(N_SIMS - done);
        let boards = tree.select_leaves(take)
        .expect("select_leaves: no desync in this fixture");
        if boards.is_empty() {
            break;
        }
        let policies: Vec<LegalSetPolicy> =
            boards.iter().map(|b| no_drop_uniform(b, n_actions)).collect();
        let values = vec![0.0_f32; boards.len()];
        match mode {
            Expand::LsAt => {
                let centers: Vec<(i32, i32)> = boards.iter().map(|b| b.window_center()).collect();
                tree.expand_and_backup_ls_at(&policies, &values, &centers, trunk_sz);
            }
            Expand::Ls => tree.expand_and_backup_ls(&policies, &values),
        }
        done += boards.len();
    }
}

fn measure(board: &Board, n_actions: usize, trunk_sz: i32, mode: Expand, ply: u32) -> Option<Row> {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board.clone());
    run_search(&mut tree, n_actions, trunk_sz, mode);

    let root = &tree.pool[0];
    if !root.is_expanded() {
        return None;
    }
    let first = root.first_child as usize;
    let n_ch = root.n_children as usize;
    if n_ch == 0 {
        return None;
    }

    let policy = tree.get_policy_ls(TEMPERATURE, n_actions);
    let exported: f64 = policy.dense.iter().map(|&p| p as f64).sum::<f64>()
        + policy.overflow.values().map(|&p| p as f64).sum::<f64>();
    let dropped_mass = 1.0 - exported;

    assert!(
        exported <= 1.0 + TOL,
        "ply {ply}: exported mass {exported} EXCEEDS 1.0 — double-count (PREREG abort 2)"
    );

    let mut offwindow_children = 0usize;
    for i in first..first + n_ch {
        let val = tree.pool[i].action_idx;
        let q = (val >> 16) as i32 - 32768;
        let r = (val & 0xFFFF) as i32 - 32768;
        if board.window_flat_idx(q, r) >= n_actions {
            offwindow_children += 1;
        }
    }

    Some(Row {
        ply,
        n_legal: board.legal_moves().len(),
        n_children: n_ch,
        dropped_mass,
        offwindow_children,
    })
}

/// Same generators as leg 1, so the position sets are comparable row for row.
fn game_rows(enc: &str, seed: u64, max_plies: u32, mode: Expand) -> Vec<Row> {
    let (geom, n_actions, trunk) = geometry_for(enc);
    let mut board = Board::with_geometry(geom);
    let mut rows = Vec::new();
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    for ply in 0..max_plies {
        let legal = board.legal_moves();
        if legal.is_empty() {
            break;
        }
        if let Some(row) = measure(&board, n_actions, trunk, mode, ply) {
            rows.push(row);
        }
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

fn dispersed_rows(enc: &str, max_plies: u32, mode: Expand) -> Vec<Row> {
    let (geom, n_actions, trunk) = geometry_for(enc);
    let mut board = Board::with_geometry(geom);
    let mut rows = Vec::new();
    for ply in 0..max_plies {
        let legal = board.legal_moves();
        if legal.is_empty() {
            break;
        }
        if let Some(row) = measure(&board, n_actions, trunk, mode, ply) {
            rows.push(row);
        }
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

fn report(label: &str, rows: &[Row]) {
    let affected = rows.iter().filter(|r| r.dropped_mass > TOL).count();
    let max = rows.iter().map(|r| r.dropped_mass).fold(0.0_f64, f64::max);
    let max_legal = rows.iter().map(|r| r.n_legal).max().unwrap_or(0);
    let with_offwindow = rows.iter().filter(|r| r.offwindow_children > 0).count();
    let mut aff: Vec<f64> = rows.iter().map(|r| r.dropped_mass).filter(|&m| m > TOL).collect();
    aff.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f64| -> f64 {
        if aff.is_empty() { 0.0 } else { aff[((aff.len() - 1) as f64 * q) as usize] }
    };
    let degenerate = rows.iter().filter(|r| r.dropped_mass >= 0.99).count();
    let over_half = rows.iter().filter(|r| r.dropped_mass >= 0.5).count();
    println!("\n=== {label} — {} positions ===", rows.len());
    println!("  affected {affected} ({:.1}%)   max dropped_mass {max:.6}   max n_legal {max_legal}",
             100.0 * affected as f64 / rows.len().max(1) as f64);
    println!("  among affected: median {:.4}  p90 {:.4}", pct(0.5), pct(0.9));
    println!("  positions losing >=50% of the target: {over_half}");
    println!("  DEGENERATE (>=99% dropped, target ~all-zero): {degenerate}");
    println!("  positions WITH off-window children: {with_offwindow} (abort 4 needs > 0)");
    for r in rows.iter().filter(|r| r.dropped_mass > TOL).take(10) {
        println!(
            "    ply {:>3}  n_legal {:>5}  n_children {:>4}  offwin {:>4}  dropped {:.6}",
            r.ply, r.n_legal, r.n_children, r.offwindow_children, r.dropped_mass
        );
    }
}

fn collect(enc: &str, mode: Expand) -> Vec<Row> {
    let mut rows = Vec::new();
    for seed in [20_260_731_u64, 8_675_309, 42] {
        rows.extend(game_rows(enc, seed, 128, mode));
    }
    rows.extend(dispersed_rows(enc, 96, mode));
    rows
}

/// Every sampled row conserves target mass and the degenerate class is EXTINCT. Fails naming
/// the first offending row.
fn assert_no_dropped_mass(label: &str, rows: &[Row]) {
    for r in rows {
        assert!(
            r.dropped_mass <= TOL,
            "{label}: ply {} (n_legal {}, n_children {}) drops {:.6} target mass \
             (> {TOL}) — the no-drop export law (records.rs:468-479, R34/R153) is \
             violated on the production path",
            r.ply, r.n_legal, r.n_children, r.dropped_mass
        );
    }
    let degenerate = rows.iter().filter(|r| r.dropped_mass >= 0.99).count();
    assert_eq!(
        degenerate, 0,
        "{label}: {degenerate} DEGENERATE rows (>=99% dropped) — the R157 class must be \
         unconstructible after the Phase T fix"
    );
}

#[test]
fn r153_leg2_run5_exposure_through_production_expand() {
    // PRIMARY: run5's encoding, its production expand. This decides run5 exposure.
    let rows = collect("gnn_axis_v1", Expand::LsAt);
    report("gnn_axis_v1 / expand_and_backup_ls_at [PRODUCTION]", &rows);

    let affected = rows.iter().filter(|r| r.dropped_mass > TOL).count();
    let max_legal = rows.iter().map(|r| r.n_legal).max().unwrap_or(0);
    let with_offwindow = rows.iter().filter(|r| r.offwindow_children > 0).count();

    // PREREG abort 1 — the tail must be reached.
    assert!(max_legal > 361, "gnn_axis_v1: sample never reached >361 legal (max {max_legal})");

    // PREREG abort 4 — THE ONE THAT MATTERS. A zero reached because the tree still holds no
    // off-window child would be a false clear.
    assert!(
        with_offwindow > 0,
        "ABORT 4: no position produced an off-window CHILD, so a zero here would be \
         structural, exactly like leg 1. The no-drop overflow is not reaching the expand — \
         fix the instrument; do NOT report this as REFUTED."
    );

    // PREREG abort 3 — determinism.
    let repeat = game_rows("gnn_axis_v1", 20_260_731, 64, Expand::LsAt);
    let again = game_rows("gnn_axis_v1", 20_260_731, 64, Expand::LsAt);
    let a: Vec<f64> = repeat.iter().map(|r| r.dropped_mass).collect();
    let b: Vec<f64> = again.iter().map(|r| r.dropped_mass).collect();
    assert_eq!(a, b, "gnn_axis_v1: instrument not deterministic at a fixed seed");

    // SECONDARY: the wider-radius graph row through the UNFRAMED ls expand. Reported; it
    // does not decide run5.
    let ls_rows = collect("gnn_axis_r8", Expand::Ls);
    report("gnn_axis_r8 / expand_and_backup_ls", &ls_rows);

    // Flipped report arms — the permanent regression assertions.
    assert_no_dropped_mass("gnn_axis_v1 / LsAt", &rows);
    assert_no_dropped_mass("gnn_axis_r8 / Ls", &ls_rows);

    println!(
        "\n=== R153 LEG 2 VERDICT INPUTS (gnn_axis_v1, PRODUCTION path) ===\n\
         \x20 positions: {}\n\x20 affected (dropped_mass > {TOL}): {affected}\n\
         \x20 positions with off-window children: {with_offwindow}\n\
         \x20 max n_legal: {max_legal}\n\
         \x20 -> RUN5 EXPOSURE {}",
        rows.len(),
        if affected > 0 { "CONFIRMED" } else { "REFUTED" }
    );
}
