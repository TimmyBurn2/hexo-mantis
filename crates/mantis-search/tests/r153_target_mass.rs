//! R153 characterization: does the EXPORTED training target drop visit mass?
//!
//! Pre-registered instrument. The verdict rule was frozen in
//! `wp/WP12R/PREREG_R153.md` BEFORE this file was first executed; nothing here was
//! authored with knowledge of the outcome.
//! COMMITTED with the WP12-R Phase T fix as a PERMANENT regression oracle (R92,
//! DESIGN_T §5 O-5): the prereg + measurement artifacts are committed VERBATIM at
//! `docs/design/measurements/PREREG_R153.md` and
//! `docs/design/measurements/MEASUREMENT_R153.md` (the in-repo citation chain).
//!
//! [T-2 ORACLE-WRITE] Report arms FLIPPED TO ASSERTIONS per O-5 — measurement
//! semantics (generators, seeds, dense expand, TOL) UNCHANGED: every sampled row must
//! now satisfy `dropped_mass <= 1e-6` on every encoding (post-§3.1 the export sums to
//! 1 by construction on EVERY expand path). Instrument abort conditions retained
//! as-is. PRE-FIX this test is RED (leg-1 measured drops at the 150-sim deploy
//! reference; `MEASUREMENT_R153.md` §2).
//!
//! ## The subject
//!
//! Two in-tree documents disagree about the same export:
//!   - `runner/records.rs:481` — training targets deliberately do NOT inherit the
//!     off-window skip; the policy target is the raw visit distribution (R34).
//!   - `mcts/policy.rs:166-168` — "Off-window children with NO cluster coverage are
//!     dropped (today's `get_policy` behaviour)."
//!
//! R153 ruled the AUTHORITY is the documented target semantics. This measures only whether
//! the export diverges from it, and characterises by how much.
//!
//! ## Why the invariant needs no reference implementation
//!
//! `get_policy_ls` normalises by the total visit count over ALL children (`v / total`), so a
//! no-drop export sums to exactly 1.0. Any deficit IS the dropped mass. Re-deriving a
//! "correct" target here would just add a second thing that can be wrong.

use mantis_core::board::{Board, BoardGeometry};
use mantis_encoding::lookup_or_panic;
use mantis_search::{is_covered, MCTSTree};

const N_SIMS: usize = 150; // run5's deploy_sims
const LEAF_BATCH: usize = 8;
const TEMPERATURE: f32 = 1.0; // the training export's branch
const TOL: f64 = 1e-6; // §4 verdict threshold

/// One position's measurement.
struct Row {
    ply: u32,
    n_children: usize,
    n_legal: usize,
    dropped_mass: f64,
    dropped_children: usize,
    /// Every dropped child was an off-window cell failing `is_covered` — §4 criterion 2.
    all_drops_attributable: bool,
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
        let boards = tree.select_leaves(take);
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
    let first = root.first_child as usize;
    let n_ch = root.n_children as usize;
    if n_ch == 0 {
        return None;
    }

    let policy = tree.get_policy_ls(TEMPERATURE, n_actions);
    let exported: f64 =
        policy.dense.iter().map(|&p| p as f64).sum::<f64>()
            + policy.overflow.values().map(|&p| p as f64).sum::<f64>();
    let dropped_mass = 1.0 - exported;

    // Abort condition 2 (PREREG §6): a surplus is a double-count, not a drop.
    assert!(
        exported <= 1.0 + TOL,
        "ply {ply}: exported mass {exported} EXCEEDS 1.0 — double-count, not a drop; \
         the PREREG §2 invariant does not hold and the instrument is wrong"
    );

    // Attribution (§4 criterion 2): recompute, per child, whether it is off-window AND
    // uncovered — i.e. exactly the cells `get_policy_ls`'s gate discards.
    let (_views, centers) = board.get_cluster_views();
    let trunk_sz = board.cluster_window_size() as i32;
    let half = (trunk_sz - 1) / 2;
    let total_visits: u32 = (first..first + n_ch).map(|i| tree.pool[i].n_visits).sum();

    let mut dropped_children = 0usize;
    let mut unattributed = 0usize;
    for i in first..first + n_ch {
        if tree.pool[i].n_visits == 0 || total_visits == 0 {
            continue;
        }
        let val = tree.pool[i].action_idx;
        let q = (val >> 16) as i32 - 32768;
        let r = (val & 0xFFFF) as i32 - 32768;
        let in_window = board.window_flat_idx(q, r) < n_actions;
        if in_window {
            continue;
        }
        if is_covered(q, r, &centers, trunk_sz, half) {
            continue; // lands in overflow — retained
        }
        dropped_children += 1;
        let _ = &mut unattributed;
    }

    // A drop with no uncovered off-window child would be unattributable (§4: reported as a
    // separate finding, never folded into this verdict).
    let all_drops_attributable = dropped_mass <= TOL || dropped_children > 0;

    Some(Row {
        ply,
        n_children: n_ch,
        n_legal: board.legal_moves().len(),
        dropped_mass,
        dropped_children,
        all_drops_attributable,
    })
}

/// DISPERSED tail probe (PREREG §3): drive stones apart so the legal set grows past the
/// 361-cell in-window ceiling. A game-only sample cannot reach the regime where the
/// coverage gate's exposure actually lives, and PREREG §6 abort 1 refuses a sample that
/// never gets there.
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
        // Farthest from the board's WINDOW CENTRE — deterministic, and it disperses
        // monotonically, which is what grows the legal set past the in-window ceiling.
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
        // Deterministic LCG playout — reproducible at a fixed seed (PREREG §6 abort 3).
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
    let median = if masses.is_empty() { 0.0 } else { masses[masses.len() / 2] };
    let max = masses.last().copied().unwrap_or(0.0);

    let max_legal = rows.iter().map(|r| r.n_legal).max().unwrap_or(0);
    println!("\n=== {enc} — {} positions, {affected} affected ===", rows.len());
    println!("  dropped_mass: median {median:.6}  max {max:.6}");
    println!("  max n_legal reached: {max_legal}  (PREREG abort 1 needs >361)");
    for r in rows.iter().filter(|r| r.dropped_mass > TOL).take(12) {
        println!(
            "    ply {:>3}  n_legal {:>5}  n_children {:>4}  dropped {:.6}  dropped_children {}",
            r.ply, r.n_legal, r.n_children, r.dropped_mass, r.dropped_children
        );
    }
    (affected, median, max, max_legal)
}

#[test]
fn r153_characterize_exported_target_dropped_mass() {
    // PRIMARY: run5's own encoding. SECONDARY: the class boundary (R71) — the ruled dense
    // control arm and the multi-window grid row.
    let encodings = ["gnn_axis_v1", "v6_live2_ls", "v6w25"];
    let seeds = [20_260_731_u64, 8_675_309, 42]; // 3 distinct games (LAW-04)

    let mut any_drop = false;
    let mut any_unattributed = false;
    let mut total_positions = 0usize;

    for enc in encodings {
        let mut rows = Vec::new();
        for seed in seeds {
            rows.extend(play_and_measure(enc, seed, 128));
        }
        // PREREG §3 tail probe — the regime a game-only sample misses.
        let tail = dispersed_and_measure(enc, 96);
        println!("  --- dispersed tail ---");
        let (tail_affected, _tm, tail_max, tail_max_legal) = report(&format!("{enc}/dispersed"), &tail);
        rows.extend(tail);

        total_positions += rows.len();
        let (affected, _median, _max, max_legal) = report(enc, &rows);
        if affected > 0 || tail_affected > 0 {
            any_drop = true;
        }
        // [T-2, R92/O-5] Flipped report arm — the permanent regression assertion.
        for r in &rows {
            assert!(
                r.dropped_mass <= TOL,
                "{enc}: ply {} (n_legal {}, n_children {}) drops {:.6} target mass \
                 (> {TOL}) — the no-drop export law (records.rs:468-479, R34/R153) is \
                 violated",
                r.ply, r.n_legal, r.n_children, r.dropped_mass
            );
        }
        if rows.iter().any(|r| !r.all_drops_attributable) {
            any_unattributed = true;
        }
        // PREREG §6 abort 1: the sample MUST reach the >361-legal regime.
        assert!(
            max_legal > 361 || tail_max_legal > 361,
            "{enc}: sample never reached the >361-legal regime (max {max_legal}, tail \
             {tail_max_legal}) — PREREG abort 1: HOLD, this sample is not representative"
        );
        // Reproducibility (PREREG §6 abort 3): same seed must give the same numbers.
        let a: Vec<f64> = play_and_measure(enc, seeds[0], 64).iter().map(|r| r.dropped_mass).collect();
        let b: Vec<f64> = play_and_measure(enc, seeds[0], 64).iter().map(|r| r.dropped_mass).collect();
        assert_eq!(a, b, "{enc}: instrument is NOT deterministic at a fixed seed");
        println!("  tail max dropped_mass {tail_max:.6}");
    }

    assert!(total_positions > 0, "the instrument measured nothing — sample is empty");
    println!(
        "\n=== R153 VERDICT INPUTS ===\n  any position with dropped_mass > {TOL}: {any_drop}\
         \n  any UNATTRIBUTED drop (not is_covered): {any_unattributed}\
         \n  total positions measured: {total_positions}"
    );

    // [T-2, R92/O-5] HISTORY: this test originally CHARACTERISED (the verdict was read
    // off the printed distribution into the measurement document). It now ALSO asserts
    // the post-fix law per row (the flipped report arm above); the instrument's own
    // abort conditions (surplus mass, determinism, non-empty sample) are retained
    // verbatim and must never be satisfied by a broken probe.
}
