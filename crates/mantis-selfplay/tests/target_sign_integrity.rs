//! ⊕ WP12-R Phase T (TARGET INTEGRITY) — T-3 loop-2 addition (RED_TEAM_T F-RT-1/F-RT-3;
//! dispatcher freeze addition FA-3): the SIGN-COVERAGE oracles. NOT part of the T-2
//! frozen bytes — additive.
//!
//! Subject: the guarded-quantity == shipped-quantity conjunct in
//! `record_position_graph` (the loop-2 stored-sum check). Pre-loop-2, a
//! sign-cancelling ls ({+1.5, −0.5}: PRE-filter scan sum == 1, stored mass == 1.5)
//! CONSTRUCTED a non-distribution record and shipped it to the loss at 1.5× weight —
//! R161 unconstructibility defeated by cancellation (RED_TEAM_T §1 F-RT-1, reproducer
//! verbatim as case 1). Negative mass reaches the ls in production only through the
//! sign-unchecked GNN seam (`assemble_ls_from_gnn_probs` checks Σ and slots, not sign)
//! — defence-in-depth, not a live-softmax defect.
//!
//! Cases: (1) the verbatim {+1.5, −0.5} reproducer at the direct constructor —
//! `MassNotUnity` carrying the SHIPPED sum 1.5 (≠ the scanned 1.0 — the
//! pre-vs-post-filter equality pin); (2)+(3) the full production chain through BOTH
//! production expands (`expand_and_backup_ls_at` / `expand_and_backup_ls`):
//! negative-mixture prior → sims-1 zero-visit §3.1 fallback passes the negative
//! through → export Σ==1 with a negative entry (asserted precondition) → record
//! REFUSES; (4) over-unity Σ=2.0 → `MassNotUnity` (the frozen bank had no over-unity
//! case); (5) F-RT-3 admit-side: Σ = 1 + 5e-5 (all-positive, within the ABSOLUTE
//! 1e-4 TOL) is ADMITTED — pins the intended window width.
//!
//! Killer: M-P (the stored-sum equality reverted → cases 1-3 red; 4/5 stay their
//! colors — recorded in PREREG_T AMENDMENT A-9).

use mantis_core::board::Board;
use mantis_core::{Cell, Player};
use mantis_search::{LegalSetPolicy, MCTSTree};
use mantis_selfplay::records::{record_position_graph, TargetIntegrityError};
/// Test slot geometry passed explicitly to `record_position_graph` (post-R255
/// the production value is DERIVED from the sims regime at composition).
const VISIT_CAP: usize = 128;

const NA: usize = 362;
const TRUNK: i32 = 19;

/// Three well-separated stones → a wide legal set (the postfix bank's wide_board).
fn wide_board() -> Board {
    let stones: Vec<((i32, i32), Cell)> =
        vec![((0, 0), Cell::P1), ((8, 0), Cell::P2), ((0, 8), Cell::P1)];
    Board::from_stones(&stones, Player::One, 2, 0, None)
}

/// ls carrying `masses[i]` at the i-th legal coord (dense/overflow routed by flat).
fn ls_on_first_legal(board: &Board, masses: &[f32]) -> LegalSetPolicy {
    let legal = board.legal_moves();
    assert!(legal.len() >= masses.len());
    let mut ls = LegalSetPolicy { dense: vec![0.0; NA], overflow: Default::default() };
    for (i, &m) in masses.iter().enumerate() {
        let (q, r) = legal[i];
        let flat = board.window_flat_idx(q, r);
        if flat < NA {
            ls.dense[flat] = m;
        } else {
            ls.overflow.insert((q, r), m);
        }
    }
    ls
}

fn record(board: &Board, ls: &LegalSetPolicy) -> Result<mantis_selfplay::replay::hexg::GraphRecord, TargetIntegrityError> {
    record_position_graph(board, ls, TRUNK, 1, 2, 3, true, VISIT_CAP)
}

// ── (1) the F-RT-1 reproducer, verbatim: {+1.5, −0.5} — scan Σ==1, stored 1.5 ────────
#[test]
fn sign_cancelling_ls_is_unconstructible_and_the_error_carries_the_shipped_sum() {
    let board = wide_board();
    let ls = ls_on_first_legal(&board, &[1.5, -0.5]); // Σ over the legal scan == 1.0
    let err = record(&board, &ls)
        .expect_err("a sign-cancelling non-distribution target must be unconstructible");
    match err {
        TargetIntegrityError::MassNotUnity { sum, n_cells, .. } => {
            // The error carries the SHIPPED (post-filter) mass — the quantity the
            // record would have stored — not the cancelled scan sum.
            assert!(
                (sum - 1.5).abs() < 1e-6,
                "MassNotUnity must carry the shipped sum 1.5 (scan sum was 1.0), got {sum}"
            );
            assert_eq!(n_cells, 1, "only the positive cell would have shipped");
        }
        other => panic!("expected MassNotUnity, got {other}"),
    }
}

// ── (2)+(3) full production chains through BOTH expands ─────────────────────────────
/// Compact board (3 close stones): n_legal < MAX_CHILDREN_PER_NODE, so EVERY
/// legal cell — including the negative-prior one — becomes a root child (on an
/// over-192-legal board the top-K sort cuts the most-negative prior and the
/// chain self-heals; the defect regime needs the full child set).
fn compact_board() -> Board {
    let stones: Vec<((i32, i32), Cell)> =
        vec![((0, 0), Cell::P1), ((2, 0), Cell::P2), ((0, 2), Cell::P1)];
    Board::from_stones(&stones, Player::One, 2, 0, None)
}

/// Drive a negative-mixture PRIOR through `expand` at sims=1 (zero-visit root →
/// the §3.1 prior fallback passes negatives through when total_prior > 0), assert
/// the export is Σ==1 WITH a negative entry (the defect regime, preconditioned),
/// then assert the record REFUSES with `MassNotUnity`.
fn drive_negative_prior_chain(use_at: bool) {
    let board = compact_board();
    let legal = board.legal_moves();
    assert!(
        legal.len() < 192,
        "construction: n_legal {} must be under the child cap so the negative-prior \
         cell becomes a child",
        legal.len()
    );
    // Negative mixture summing to 1 over the legal set: one cell at +1.5, one at
    // −0.5, rest 0 — mirrors the reproducer through the production seam.
    let mut prior = LegalSetPolicy { dense: vec![0.0; NA], overflow: Default::default() };
    let f0 = board.window_flat_idx(legal[0].0, legal[0].1);
    let f1 = board.window_flat_idx(legal[1].0, legal[1].1);
    assert!(f0 < NA && f1 < NA, "first two legal cells must be in-window");
    prior.dense[f0] = 1.5;
    prior.dense[f1] = -0.5;

    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board.clone());
    let leaves = tree.select_leaves(1);
    assert_eq!(leaves.len(), 1);
    if use_at {
        let centers = vec![board.window_center()];
        tree.expand_and_backup_ls_at(&[prior], &[0.0f32], &centers, TRUNK);
    } else {
        tree.expand_and_backup_ls(&[prior], &[0.0f32]);
    }
    assert!(tree.pool[0].is_expanded(), "root must expand");

    let ls = tree.get_policy_ls(1.0, NA); // zero-visit → prior fallback
    let scan: f64 = ls.dense.iter().map(|&p| f64::from(p)).sum::<f64>()
        + ls.overflow.values().map(|&p| f64::from(p)).sum::<f64>();
    let min_entry = ls
        .dense
        .iter()
        .chain(ls.overflow.values())
        .copied()
        .fold(f32::INFINITY, f32::min);
    assert!(
        (scan - 1.0).abs() < 1e-4 && min_entry < 0.0,
        "precondition: the export must be Σ==1 with a NEGATIVE entry \
         (Σ={scan}, min={min_entry}) — the sign-cancellation regime"
    );

    let err = record(&board, &ls)
        .expect_err("the production chain must not construct a non-distribution record");
    assert!(
        matches!(err, TargetIntegrityError::MassNotUnity { .. }),
        "expected MassNotUnity, got {err}"
    );
}

#[test]
fn negative_prior_chain_through_expand_and_backup_ls_at_refuses_at_record() {
    drive_negative_prior_chain(true);
}

#[test]
fn negative_prior_chain_through_expand_and_backup_ls_refuses_at_record() {
    drive_negative_prior_chain(false);
}

// ── (4) over-unity: Σ = 2.0 (all-positive) → MassNotUnity ───────────────────────────
#[test]
fn over_unity_all_positive_is_unconstructible() {
    let board = wide_board();
    let ls = ls_on_first_legal(&board, &[1.5, 0.5]);
    let err = record(&board, &ls).expect_err("Σ=2.0 must be unconstructible");
    match err {
        TargetIntegrityError::MassNotUnity { sum, .. } => {
            assert!((sum - 2.0).abs() < 1e-6, "carries the off-unity sum, got {sum}");
        }
        other => panic!("expected MassNotUnity, got {other}"),
    }
}

// ── (5) F-RT-3 admit-side: within the ABSOLUTE 1e-4 TOL is ADMITTED ─────────────────
#[test]
fn within_tol_sum_is_admitted_the_documented_window() {
    let board = wide_board();
    let ls = ls_on_first_legal(&board, &[0.6, 0.4 + 5.0e-5]); // Σ = 1 + 5e-5 < TOL
    let rec = record(&board, &ls)
        .expect("a within-TOL (1 + 5e-5) all-positive target is ADMITTED — F-RT-3 pin");
    assert_eq!(rec.visits.len(), 2);
}
