//! ⊕ WP12-R Phase T (TARGET INTEGRITY) — S2c: `aggregate_policy_to_local_ls` projection
//! oracles (DESIGN_T §3.5, flip-set rows 9-10), written at T-2, byte-frozen through IMPL.
//!
//! POST-FIX contract:
//!  * covered off-window mass projects through the `:598-602` overflow read (row 9 —
//!    the prior WP's fix, PINNED here, unchanged);
//!  * the window-renorm KEEPS its K-window conditional semantics (relative-mass parity
//!    with the global ls — pinned, not changed);
//!  * a cluster window with ZERO visible visit mass yields the ALL-ZERO local row
//!    (the pipeline's pre-existing value-only sentinel, `policy_valid`-masked at the
//!    loss) — the FABRICATED UNIFORM fill (`:616-619`) becomes unconstructible from
//!    any nonzero global ls (row 10; Q1 RATIFIED with conditions, REVIEW_DESIGN_T §3).
//!
//! PRE-FIX status at HEAD: rows 9 + parity GREEN (already correct); the zero-row case
//! RED (HEAD fabricates uniform). Killers (PREREG_T §3): M-E (uniform fallback
//! reinstated → zero-row case reds), M-A (band-10 global-ls leg upstream).
//!
//! Board construction mirrors records.rs `ls_tests::spread_board` (two far-apart stone
//! groups; global centre (17,0); (28,0) off the global window but covered by cluster-2).

use fxhash::FxHashMap;
use mantis_core::{Board, Cell, Player};
use mantis_search::LegalSetPolicy;
use mantis_selfplay::records::aggregate_policy_to_local_ls;

const TRUNK: i32 = 19;
const NA: usize = 19 * 19 + 1; // 362

fn spread_board() -> Board {
    let stones: Vec<((i32, i32), Cell)> = (0..5i32)
        .map(|q| ((q, 0), Cell::P1))
        .chain((30..35i32).map(|q| ((q, 0), Cell::P2)))
        .collect();
    Board::from_stones(&stones, Player::One, 2, 0, None)
}

/// Global ls carrying `mass` at the covered off-window cell (28,0) and the remainder at
/// an in-window cell, so the global target is a valid distribution.
fn ls_with_covered_offwindow(board: &Board, mass: f32) -> (LegalSetPolicy, usize, usize) {
    let legal = board.legal_moves();
    assert!(
        legal.contains(&(28, 0)),
        "(28,0) must be legal near cluster-2"
    );
    // an in-window legal cell for the remainder
    let &(iq, ir) = legal
        .iter()
        .find(|&&(q, r)| board.window_flat_idx(q, r) < NA)
        .expect("spread board has in-window legal cells");
    let in_flat = board.window_flat_idx(iq, ir);
    let mut dense = vec![0.0f32; NA];
    dense[in_flat] = 1.0 - mass;
    let mut overflow: FxHashMap<(i32, i32), f32> = FxHashMap::default();
    overflow.insert((28, 0), mass);
    // local index of (28,0) in cluster-2's (32,0) frame: wq=28-32+9=5, wr=9 → 104.
    let local_28 = 5usize * TRUNK as usize + 9;
    (LegalSetPolicy { dense, overflow }, in_flat, local_28)
}

// ── flip-set row 9: covered off-window mass projects (pin, GREEN pre- and post-fix) ──
#[test]
fn s2c_covered_offwindow_mass_projects_into_the_covering_cluster() {
    let board = spread_board();
    assert_eq!(
        board.window_center(),
        (17, 0),
        "spread-board geometry drifted"
    );
    let legal = board.legal_moves();
    let (ls, _, local_28) = ls_with_covered_offwindow(&board, 0.4);

    let local = aggregate_policy_to_local_ls(NA, true, TRUNK, &board, &(32, 0), &ls, &legal);
    // cluster-2's window sees ONLY (28,0) → window-renorm makes it the whole row.
    assert!(
        (local[local_28] - 1.0).abs() < 1e-6,
        "covered off-window mass must project through the overflow read and renorm to \
         its window-conditional (got {})",
        local[local_28]
    );
}

// ── window-renorm relative-mass parity with the global ls (pin) ───────────────────────
#[test]
fn s2c_window_renorm_preserves_relative_mass() {
    let board = spread_board();
    let legal = board.legal_moves();
    // Two cells visible in cluster-1's (2,0) window with global masses 0.3 / 0.1:
    // the local row must carry them at ratio 3:1 (the K-window conditional).
    let mut vis: Vec<((i32, i32), f32)> = Vec::new();
    for &(q, r) in &legal {
        let wq = q - 2 + 9;
        let wr = r + 9;
        if (0..TRUNK).contains(&wq) && (0..TRUNK).contains(&wr) && board.window_flat_idx(q, r) < NA
        {
            vis.push(((q, r), 0.0));
            if vis.len() == 2 {
                break;
            }
        }
    }
    assert_eq!(
        vis.len(),
        2,
        "cluster-1 window must see >=2 in-window legal cells"
    );
    let mut dense = vec![0.0f32; NA];
    dense[board.window_flat_idx(vis[0].0 .0, vis[0].0 .1)] = 0.3;
    dense[board.window_flat_idx(vis[1].0 .0, vis[1].0 .1)] = 0.1;
    // remainder elsewhere (off this window) so the global sums to 1
    let mut overflow: FxHashMap<(i32, i32), f32> = FxHashMap::default();
    overflow.insert((28, 0), 0.6);
    let ls = LegalSetPolicy { dense, overflow };

    let local = aggregate_policy_to_local_ls(NA, true, TRUNK, &board, &(2, 0), &ls, &legal);
    let l0 = local[((vis[0].0 .0 - 2 + 9) as usize) * TRUNK as usize + (vis[0].0 .1 + 9) as usize];
    let l1 = local[((vis[1].0 .0 - 2 + 9) as usize) * TRUNK as usize + (vis[1].0 .1 + 9) as usize];
    assert!(
        l0 > 0.0 && l1 > 0.0,
        "visible cells must survive projection"
    );
    assert!(
        (l0 / l1 - 3.0).abs() < 1e-4,
        "window renorm must preserve relative mass (3:1), got {l0}/{l1}"
    );
    let sum: f32 = local.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "a window WITH visible mass renorms to 1, got {sum}"
    );
}

// ── flip-set row 10: zero-visible-window → the ZERO row, never a fabricated uniform ──
#[test]
fn s2c_zero_visible_window_yields_the_zero_row_not_uniform() {
    let board = spread_board();
    let legal = board.legal_moves();
    // ALL visit mass at (28,0) — visible ONLY to cluster-2. Cluster-1's (2,0) window
    // sees zero mass: post-fix its row is ALL-ZERO (the value-only sentinel,
    // runner/record.rs:67-78 / trainer core.py:397 convention); pre-fix it is a
    // fabricated UNIFORM distribution trained at full weight (the R157-worse object).
    let (ls, _, _) = ls_with_covered_offwindow(&board, 1.0);
    let global_mass: f32 = ls.dense.iter().sum::<f32>() + ls.overflow.values().sum::<f32>();
    assert!(
        (global_mass - 1.0).abs() < 1e-6,
        "construction: global ls must be a distribution"
    );

    let local = aggregate_policy_to_local_ls(NA, true, TRUNK, &board, &(2, 0), &ls, &legal);
    let sum: f32 = local.iter().sum();
    let uniform = 1.0 / NA as f32;
    let is_uniform = local.iter().all(|&p| (p - uniform).abs() < 1e-9);
    assert!(
        !is_uniform,
        "a zero-visible-mass window fabricated the UNIFORM row — the §3.5 fabrication \
         (records.rs:616-619 form) is live (M-E shape)"
    );
    assert!(
        local.iter().all(|&p| p == 0.0) && sum == 0.0,
        "a zero-visible-mass window must yield the ALL-ZERO local row (the value-only \
         sentinel the loss masks via policy_valid), got sum {sum}"
    );
}
