// R8 >300 justify: the four stage-1 exporter oracles (S1a x3 arms + S1b) share one
// search-driving harness (generators + no-drop prior + run loop) that must stay byte-equal
// to the r153 instruments'; splitting would fork the harness the oracles' provenance
// depends on.
//! ⊕ WP12-R Phase T (TARGET INTEGRITY) — S1a / S1b: the stage-1 exporter sum + full-set
//! oracles, written BEFORE the fix (T-2 ORACLE-WRITE; frozen bytes through IMPL).
//!
//! POST-FIX contract (DESIGN_T §3.1/§3.2, authority records.rs:468-479 + R34/R153/R156):
//! `get_policy_ls` and `get_improved_policy_ls` export the visit/improved distribution
//! over the FULL root-child set — in- AND off-window, covered or not — summing to 1;
//! zero-visit roots return the prior-fallback distribution (the improved exporters'
//! semantics); no coverage gate anywhere in the export.
//!
//! PRE-FIX status at HEAD (recorded in ORACLE_NOTES_T.md): S1a-T1 RED (sum < 1 on the
//! band-2 position, HEAD drop 0.306122), S1a-T0 RED (all-zero export on the degenerate
//! board), S1a-ZV RED (all-zero instead of prior fallback), S1b RED (uncovered child
//! absent from the improved export — the subset-renorm arm).
//!
//! Killers (PREREG_T §3): S1a — M-A (coverage gate reinstated in `get_policy_ls`);
//! S1b — M-B (pre-softmax drop reinstated in `get_improved_policy_ls`).
//!
//! Positions come from the r153 instruments' generators at fixed seeds (LCG verbatim),
//! so every construction is reproducible and shared with the O1r fixture family.

use mantis_core::board::{Board, BoardGeometry};
use mantis_encoding::lookup_or_panic;
use mantis_search::{is_covered, LegalSetPolicy, MCTSTree};

const N_SIMS: usize = 50; // run5 target-generation regime (PROVENANCE_T0 §1)
const LEAF_BATCH: usize = 8;
const TOL: f64 = 1e-6;

fn geometry() -> (BoardGeometry, usize, i32) {
    let spec = lookup_or_panic("gnn_axis_v1");
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

/// The no-drop uniform legal-set prior — byte-equal to the leg-2 instrument's.
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

fn run_search(tree: &mut MCTSTree, n_actions: usize, trunk_sz: i32, sims: usize) {
    let mut done = 0;
    while done < sims {
        let take = LEAF_BATCH.min(sims - done);
        let boards = tree.select_leaves(take)
        .expect("select_leaves: no desync in this fixture");
        if boards.is_empty() {
            break;
        }
        let policies: Vec<LegalSetPolicy> =
            boards.iter().map(|b| no_drop_uniform(b, n_actions)).collect();
        let values = vec![0.0_f32; boards.len()];
        let centers: Vec<(i32, i32)> = boards.iter().map(|b| b.window_center()).collect();
        tree.expand_and_backup_ls_at(&policies, &values, &centers, trunk_sz);
        done += boards.len();
    }
}

/// LCG game generator — byte-equal to the r153 instruments'.
fn game_board(seed: u64, n_plies: usize) -> Board {
    let (geom, _, _) = geometry();
    let mut board = Board::with_geometry(geom);
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    for _ in 0..n_plies {
        let legal = board.legal_moves();
        if legal.is_empty() {
            break;
        }
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let (q, r) = legal[(state >> 33) as usize % legal.len()];
        if board.apply_move(q, r).is_err() {
            break;
        }
    }
    board
}

fn coord_of(tree: &MCTSTree, i: usize) -> (i32, i32) {
    let val = tree.pool[i].action_idx;
    ((val >> 16) as i32 - 32768, (val & 0xFFFF) as i32 - 32768)
}

fn export_mass(ls: &LegalSetPolicy) -> f64 {
    ls.dense.iter().map(|&p| f64::from(p)).sum::<f64>()
        + ls.overflow.values().map(|&p| f64::from(p)).sum::<f64>()
}

/// Read the exported mass at a board coord (dense by flat, else overflow, else 0).
fn mass_at(ls: &LegalSetPolicy, board: &Board, q: i32, r: i32, n_actions: usize) -> f64 {
    let flat = board.window_flat_idx(q, r);
    if flat < n_actions {
        f64::from(ls.dense[flat])
    } else {
        f64::from(ls.overflow.get(&(q, r)).copied().unwrap_or(0.0))
    }
}

/// Precondition harness: run the production search on a generator board and return the
/// tree plus the (visited, uncovered-off-window) child census — the constructions must
/// PROVE they exercise the defect regime rather than pass vacuously.
struct Setup {
    board: Board,
    tree: MCTSTree,
    n_actions: usize,
    visited: Vec<(i32, i32, u32)>,
    visited_uncovered: Vec<(i32, i32)>,
}

fn setup(seed: u64, plies: usize, sims: usize) -> Setup {
    let (_, n_actions, trunk) = geometry();
    let board = game_board(seed, plies);
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board.clone());
    run_search(&mut tree, n_actions, trunk, sims);
    let root = &tree.pool[0];
    assert!(root.is_expanded(), "setup: root must expand");
    let first = root.first_child as usize;
    let n_ch = root.n_children as usize;

    let (_views, centers) = board.get_cluster_views();
    let ct = board.cluster_window_size() as i32;
    let half = (ct - 1) / 2;

    let mut visited = Vec::new();
    let mut visited_uncovered = Vec::new();
    for i in first..first + n_ch {
        let v = tree.pool[i].n_visits;
        if v == 0 {
            continue;
        }
        let (q, r) = coord_of(&tree, i);
        visited.push((q, r, v));
        if board.window_flat_idx(q, r) >= n_actions && !is_covered(q, r, &centers, ct, half) {
            visited_uncovered.push((q, r));
        }
    }
    Setup { board, tree, n_actions, visited, visited_uncovered }
}

// ── S1a arm 1: T > 0 — full-mass export, off-window mass carried ─────────────────────
#[test]
fn s1a_t1_export_sums_to_unity_with_offwindow_mass_carried() {
    // Band-2 position (survey: seed 8675309 ply 3, n_legal 232, HEAD drop 0.306122).
    let s = setup(8_675_309, 3, N_SIMS);
    assert!(
        !s.visited_uncovered.is_empty(),
        "construction failed: no visited uncovered off-window child — the position no \
         longer exercises the defect regime (re-derive from the survey)"
    );
    let ls = s.tree.get_policy_ls(1.0, s.n_actions);
    let total: f64 = export_mass(&ls);
    assert!(
        (total - 1.0).abs() <= TOL,
        "get_policy_ls T=1 must export FULL visit mass (authority records.rs:468-479, \
         R34/R153): got {total} (dropped {})",
        1.0 - total
    );
    // Full-set carry: EVERY visited child's coord exports nonzero mass.
    for &(q, r, v) in &s.visited {
        let m = mass_at(&ls, &s.board, q, r, s.n_actions);
        assert!(
            m > 0.0,
            "visited child ({q},{r}) with {v} visits exported ZERO mass — the coverage \
             gate is back (M-A shape)"
        );
    }
}

// ── S1a arm 2: T = 0 — uncovered one-hot lands in overflow (flip-set row 7) ───────────
#[test]
fn s1a_t0_uncovered_best_child_exports_a_one_hot() {
    // Degenerate board (survey: seed 20260731 ply 53, HEAD drop 1.000000 — every visited
    // child off-window + uncovered, so the argmax child is deterministically uncovered).
    // Construction deviation from flip-set row 7's "two-cluster golden board" recorded in
    // ORACLE_NOTES_T.md: that board has no uncovered LEGAL cell at gnn geometry; this one
    // provably does, and the predicate (best child uncovered -> one-hot in overflow) is
    // row 7's, unchanged.
    let s = setup(20_260_731, 53, N_SIMS);
    let (best_q, best_r, best_v) = *s
        .visited
        .iter()
        .max_by_key(|&&(_, _, v)| v)
        .expect("search must visit at least one child");
    assert!(best_v > 0);
    assert!(
        s.visited_uncovered.contains(&(best_q, best_r)),
        "construction failed: the max-visit child ({best_q},{best_r}) is not uncovered \
         off-window — the T=0 arm's defect regime is not exercised"
    );

    let ls = s.tree.get_policy_ls(0.0, s.n_actions);
    let total = export_mass(&ls);
    assert!(
        (total - 1.0).abs() <= TOL,
        "T=0 export must be a one-hot (sum 1), got {total} — the all-zero T=0 arm \
         (DESIGN_T §1.1 arm 2) is live"
    );
    let m = f64::from(ls.overflow.get(&(best_q, best_r)).copied().unwrap_or(0.0));
    assert!(
        (m - 1.0).abs() <= TOL,
        "the uncovered best child ({best_q},{best_r}) must carry the one-hot in overflow, \
         got {m}"
    );
}

// ── S1a arm 3: zero-visit root — prior fallback == the improved siblings' semantics
//    (flip-set row 8) ───────────────────────────────────────────────────────────────────
#[test]
fn s1a_zero_visit_root_falls_back_to_the_prior_distribution() {
    // ONE expand pass: the root expands (sim 1 is consumed by the root expansion), its
    // children all carry 0 visits — the sims=1 regime DESIGN_T §3.1 names.
    let s = setup(8_675_309, 3, 1);
    assert!(
        s.visited.is_empty(),
        "construction failed: a child was visited — total > 0 and the zero-visit arm is \
         not exercised"
    );
    let ls = s.tree.get_policy_ls(1.0, s.n_actions);
    let improved = s.tree.get_improved_policy_ls(s.n_actions, 50.0, 1.0);

    // Both are the prior fallback over the FULL child set → identical by coord, sum 1.
    let total = export_mass(&ls);
    assert!(
        (total - 1.0).abs() <= TOL,
        "zero-visit get_policy_ls must ship the prior fallback (sum 1), got {total} — \
         the silent all-zero arm (DESIGN_T §1.1 arm 4) is live"
    );
    let imp_total = export_mass(&improved);
    assert!((imp_total - 1.0).abs() <= TOL, "improved-ls fallback must sum to 1, got {imp_total}");

    let root = &s.tree.pool[0];
    let first = root.first_child as usize;
    for i in first..first + root.n_children as usize {
        let (q, r) = coord_of(&s.tree, i);
        let a = mass_at(&ls, &s.board, q, r, s.n_actions);
        let b = mass_at(&improved, &s.board, q, r, s.n_actions);
        assert!(
            (a - b).abs() <= TOL,
            "zero-visit fallback diverges from the improved sibling at ({q},{r}): \
             {a} vs {b} — §3.1 pins the exporters to ONE fallback semantics"
        );
    }
}

// ── S1b: improved-ls FULL-SET check — the anti-subset-renorm oracle ──────────────────
#[test]
fn s1b_improved_ls_keeps_every_child_including_uncovered() {
    let s = setup(8_675_309, 3, N_SIMS);
    assert!(
        !s.visited_uncovered.is_empty(),
        "construction failed: no visited uncovered off-window child"
    );
    let improved = s.tree.get_improved_policy_ls(s.n_actions, 50.0, 1.0);
    let total = export_mass(&improved);
    // A sum check ALONE cannot see arm 3 (subset renorm sums to 1); the full-set
    // presence check below is the load-bearing assert.
    assert!((total - 1.0).abs() <= 1e-4, "improved-ls must sum to 1, got {total}");

    let root = &s.tree.pool[0];
    let first = root.first_child as usize;
    for i in first..first + root.n_children as usize {
        let (q, r) = coord_of(&s.tree, i);
        let m = mass_at(&improved, &s.board, q, r, s.n_actions);
        assert!(
            m > 0.0,
            "root child ({q},{r}) carries ZERO improved mass — the pre-softmax drop \
             (policy.rs:261-264 form, §1.1 arm 3) redistributed it over the subset (M-B)"
        );
    }
    // And the uncovered children specifically (the arm-3 victims) are present.
    for &(q, r) in &s.visited_uncovered {
        assert!(
            improved.overflow.contains_key(&(q, r)),
            "visited uncovered child ({q},{r}) absent from the improved overflow — \
             subset renormalization (the authority's second forbidden mode)"
        );
    }
}
