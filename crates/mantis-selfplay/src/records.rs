//! R8-justify: verbatim behaviour-exact port of the frozen `game_runner/records.rs`, kept
//! single-file for 1:1 audit against the frozen original.
//!
//! Per-game record helpers for the self-play worker loop: policy aggregation between
//! cluster-local and global action frames, visit-count policy sampling, and the game-end
//! reprojection of final ownership + winning-line targets. All functions are pure and free.

use fxhash::{FxHashMap, FxHashSet};
use mantis_core::Board;
use mantis_search::{LegalSetPolicy, MCTSTree};
use rand::{rng, RngExt};

// The ragged legal-set policy functions below are the no-drop counterparts of the dense
// scatter-max path. That path is byte-identical for in-global-window cells; only off-window cells
// covered by some cluster are additionally retained, in `overflow`.

/// GNN counterpart of `aggregate_policy_ls`: the GNN policy head emits ONE probability per legal
/// node directly, so this is a pure re-key of the ragged probs into the existing `LegalSetPolicy`
/// no-drop consumer — in-window to `dense[slot]`, off-window (`policy_dst_slot ==
/// OFF_WINDOW_SLOT`) to `overflow[coord]`, where a dense `[B,362]` scatter would drop them.
///
/// `legal_probs` are pre-normalized by the per-graph segmented softmax, so there is NO renorm
/// here, and the `OFF_WINDOW_SLOT` sentinel is single-sourced from the builder crate.
///
/// Returns `Err(reason)` and never panics on a ragged length desync or an out-of-range slot: a
/// panic must cross the FFI as a PanicException rather than abort the run. Both legs are
/// unreachable in practice, but a public seam reading an untrusted slice fails gracefully.
#[inline]
pub fn assemble_ls_from_gnn_probs(
    n_actions: usize,
    legal_probs: &[f32],
    policy_dst_slot: &[i32],
    legal_coords: &[(i32, i32)],
) -> Result<LegalSetPolicy, String> {
    if legal_probs.len() != policy_dst_slot.len() || legal_probs.len() != legal_coords.len() {
        return Err(format!(
            "assemble_ls_from_gnn_probs: ragged length mismatch — probs {} slots {} coords {}",
            legal_probs.len(),
            policy_dst_slot.len(),
            legal_coords.len()
        ));
    }

    let mut dense = vec![0.0f32; n_actions];
    let mut overflow: FxHashMap<(i32, i32), f32> = FxHashMap::default();

    for i in 0..legal_probs.len() {
        let slot = policy_dst_slot[i];
        if slot == mantis_graph::OFF_WINDOW_SLOT {
            overflow.insert(legal_coords[i], legal_probs[i]);
        } else {
            // In-window slot: bounded by the builder's always-on `verify_contract`, but guarded
            // here too, since this fn is a public seam consumer of an untrusted slice.
            let idx = slot as usize;
            if slot < 0 || idx >= n_actions {
                return Err(format!(
                    "assemble_ls_from_gnn_probs: slot {slot} out of range 0..{n_actions}"
                ));
            }
            dense[idx] = legal_probs[i];
        }
    }

    // Segmented-softmax invariant: probs already sum to 1 over the legal set. ALWAYS-ON, because
    // a debug_assert is inert in the release .so self-play actually runs.
    let sum: f32 = dense.iter().sum::<f32>() + overflow.values().sum::<f32>();
    if !(legal_probs.is_empty() || (sum - 1.0).abs() < 1e-3) {
        return Err(format!(
            "assemble_ls_from_gnn_probs: probs sum to {sum} not 1 (segmented-softmax desync)"
        ));
    }

    Ok(LegalSetPolicy { dense, overflow })
}

/// Typed target-integrity refusal (LAW-14), in two scopes.
///
/// `MassNotUnity` / `EmptyTarget` / `VisitSlotsExceeded` bind the GRAPH record constructor only,
/// making the degenerate target class unconstructible there; the dense fast-game zero-policy arm
/// is a deliberate value-only sentinel and never reaches this constructor. `ZeroVisitSearch` binds
/// the TARGET-EXPORT boundary on BOTH arms, since the dense arm's zero-visit export is an all-zero
/// row indistinguishable from its legitimate sentinel.
///
/// `Display` leads with the variant name, so the name survives verbatim through the runner
/// fatal-defect latch to the supervisor.
#[derive(Clone, Debug, PartialEq)]
pub enum TargetIntegrityError {
    /// The pre-filter legal-scan mass is non-finite or off unity — a NaN/inf anywhere in the ls
    /// poisons the f64 sum and lands here with a non-finite `sum`.
    MassNotUnity {
        sum: f64,
        ply_index: u16,
        n_cells: usize,
    },
    /// ~Zero mass over a NON-empty legal set: a named degenerate class, checked BEFORE unity.
    EmptyTarget { ply_index: u16, n_legal: usize },
    /// More positive-mass cells than the DERIVED HEXG visit slot holds — the silent top-k
    /// truncation's typed replacement. `max` is the composed `visit_capacity`, never a literal.
    VisitSlotsExceeded {
        n: usize,
        max: usize,
        ply_index: u16,
    },
    /// The search backed up ZERO child visits, so there is no visit distribution to export and
    /// every exporter would ship the prior fallback instead. See [`refuse_zero_visit_export`].
    ZeroVisitSearch { ply_index: u16, n_children: usize },
}

impl std::fmt::Display for TargetIntegrityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetIntegrityError::MassNotUnity {
                sum,
                ply_index,
                n_cells,
            } => write!(
                f,
                "MassNotUnity: policy-target mass sum={sum} != 1 at ply_index={ply_index} \
                 ({n_cells} positive-mass cells) — the exporter/injector upstream shipped an \
                 invalid distribution (LAW-14: run-fatal, never recorded)"
            ),
            TargetIntegrityError::EmptyTarget { ply_index, n_legal } => write!(
                f,
                "EmptyTarget: policy target carries ~zero mass over a non-empty legal set \
                 (ply_index={ply_index}, n_legal={n_legal}) — the degenerate all-zero class \
                 is unconstructible (LAW-14: run-fatal, never recorded)"
            ),
            TargetIntegrityError::VisitSlotsExceeded { n, max, ply_index } => write!(
                f,
                "VisitSlotsExceeded: {n} positive-mass cells exceed the derived visit \
                 capacity {max} at ply_index={ply_index}; the capacity is derived from \
                 the configured sims regime (max armed arm + leaf_batch_size − 1, \
                 R255/ADJ-D34), so an exporter emitting more support than the regime \
                 admits is a defect upstream — silent truncation is deleted (LAW-14: \
                 run-fatal, never recorded)"
            ),
            TargetIntegrityError::ZeroVisitSearch {
                ply_index,
                n_children,
            } => write!(
                f,
                "ZeroVisitSearch: the search at ply_index={ply_index} backed up ZERO visits \
                 across its {n_children} root children, so there is no visit distribution to \
                 export — every exporter falls back to the (noise-mixed) priors and the \
                 result is indistinguishable in the buffer from a real target. A target \
                 cannot be built from a search that did not run (R275(b) exporter conjunct; \
                 LAW-14: run-fatal, never recorded)"
            ),
        }
    }
}

impl std::error::Error for TargetIntegrityError {}

/// Tolerance for the unity tripwire: an ABSOLUTE tolerance on comparisons anchored at 1.0, NOT the
/// relative form `mass_drop_check` uses — the two share only the numeric width 1e-4, which at the
/// unity anchor makes absolute and relative identical. `pub`, so the bridge push face refuses with
/// the SAME window.
pub const TARGET_MASS_TOL: f64 = 1e-4;

/// Refuse to build a training target out of a search that backed up nothing. Returns the backed-up
/// root-child visit total.
///
/// The class: *a search that did not run is exported as if it had.* The seam conjunct stops the
/// usual CAUSE and this stops the CONSEQUENCE; each sees a population the other cannot, since a
/// failure landing LATE leaves a truncated search with nonzero visits that only the seam catches,
/// while this one also catches a zero-visit search with every inference healthy.
///
/// Deleting the prior fallback instead would swap a prior dump for an all-zero target, which the
/// DENSE recorder does not catch at all. This pin, its seam sibling and the capacity formula are
/// all derived from the CURRENT visit-limited construction and re-derive with it.
///
/// # Errors
/// [`TargetIntegrityError::ZeroVisitSearch`] when the root is unexpanded, has no children, or
/// every child carries `n_visits == 0`. The caller latches it run-fatal (LAW-14).
pub fn refuse_zero_visit_export(
    tree: &MCTSTree,
    ply_index: u16,
) -> Result<u32, TargetIntegrityError> {
    let root = &tree.pool[0];
    let n_children = if root.is_expanded() {
        root.n_children as usize
    } else {
        0
    };
    let first = root.first_child as usize;
    // Root visits are NOT the subject: the root's own expansion backs up one visit to itself. The
    // exporters normalize over the CHILDREN, so the children are what must have been visited.
    let backed_up: u32 = (first..first + n_children)
        .map(|i| tree.pool[i].n_visits)
        .sum();
    if backed_up == 0 {
        return Err(TargetIntegrityError::ZeroVisitSearch {
            ply_index,
            n_children,
        });
    }
    Ok(backed_up)
}

/// Build the ONE compact graph-position record from the search-root board plus the assembled
/// ragged `LegalSetPolicy`. Whole-board: no K-cluster loop, no dense planes, no aux. The visit
/// target is the coord→prob map over the FULL legal set, read BY COORD, so the off-window skip is
/// not inherited; `outcome`/`value_valid` are placeholders stamped by `finalize_graph_outcome`.
///
/// The f64 mass accumulates on the RAW `ls.get` read INSIDE the legal scan, BEFORE the `p > 0.0`
/// keep-test, so a NaN/inf poisons the sum and is seen. Check order is PINNED: guarded-equals-
/// shipped; non-finite → `MassNotUnity`; ~0-with-legal → `EmptyTarget`; off-unity →
/// `MassNotUnity`; `len > max_visits` → `VisitSlotsExceeded`. `explicit_support`, when supplied,
/// is the set of cells the row stores CELL BY CELL; the rest is summed into `tail_mass` α.
///
/// # Errors
/// Returns [`TargetIntegrityError`] per the pinned order above — the caller latches it run-fatal,
/// never a silent skip.
#[allow(clippy::too_many_arguments)] // mirrors the dense record fns' spec-derived scalar surface
pub fn record_position_graph(
    board: &Board,
    ls: &LegalSetPolicy,
    trunk_sz: i32,
    current_player: i8,
    moves_remaining: u8,
    ply_index: u16,
    is_full_search: bool,
    max_visits: usize,
    explicit_support: Option<&FxHashSet<(i32, i32)>>,
) -> Result<crate::replay::hexg::GraphRecord, TargetIntegrityError> {
    let (bcq, bcr) = board.window_center();
    let half = (trunk_sz - 1) / 2;

    // Visit target: read the ragged mass at each legal coord (no floor — a cell absent from `ls`
    // is truly 0-visit), accumulating on the RAW read, PRE-filter. With an `explicit_support` the
    // row is SPARSE: a positive-mass cell outside that set is summed into α instead of a slot.
    let legal = board.legal_moves();
    let mut visits: Vec<(i16, i16, f32)> = Vec::with_capacity(legal.len());
    let mut sum: f64 = 0.0;
    let mut stored: f64 = 0.0;
    let mut tail: f64 = 0.0;
    for &(q, r) in &legal {
        let p = ls.get(q, r, bcq, bcr, trunk_sz, half, 0.0);
        sum += f64::from(p);
        if p > 0.0 {
            if explicit_support.is_some_and(|set| !set.contains(&(q, r))) {
                tail += f64::from(p);
            } else {
                visits.push((q as i16, r as i16, p));
                stored += f64::from(p);
            }
        }
    }

    // The guarded quantity must equal the SHIPPED quantity: `sum` is the PRE-filter scan
    // (NaN-visible) and `stored + tail` the post-`p > 0.0` mass from the SAME scan, so a
    // sign-cancelling ls ({+1.5, −0.5}: scan sum 1, stored 1.5) cannot ship. A NaN `sum` makes
    // this FALSE and falls through to the finiteness arm, preserving the pinned order below.
    if (sum - (stored + tail)).abs() > TARGET_MASS_TOL {
        return Err(TargetIntegrityError::MassNotUnity {
            // The SHIPPED mass, both halves: the explicit entries plus the tail scalar.
            sum: stored + tail,
            ply_index,
            n_cells: visits.len(),
        });
    }

    // §3.3 pinned order-arms 1..3 + the §3.4 slot guard.
    if !sum.is_finite() {
        return Err(TargetIntegrityError::MassNotUnity {
            sum,
            ply_index,
            n_cells: visits.len(),
        });
    }
    if sum.abs() <= TARGET_MASS_TOL && !legal.is_empty() {
        return Err(TargetIntegrityError::EmptyTarget {
            ply_index,
            n_legal: legal.len(),
        });
    }
    if (sum - 1.0).abs() > TARGET_MASS_TOL {
        return Err(TargetIntegrityError::MassNotUnity {
            sum,
            ply_index,
            n_cells: visits.len(),
        });
    }
    if visits.len() > max_visits {
        return Err(TargetIntegrityError::VisitSlotsExceeded {
            n: visits.len(),
            max: max_visits,
            ply_index,
        });
    }

    // Stones from the board's sparse occupied-cell map; order is irrelevant, since the rebuild
    // coordinate-sorts. `Cell` is `#[repr(i8)]` (P1=1, P2=-1).
    let mut stones: Vec<(i16, i16, i8)> = Vec::new();
    for (&(q, r), &cell) in board.cells_iter() {
        stones.push((q as i16, r as i16, cell as i8));
    }

    Ok(crate::replay::hexg::GraphRecord {
        stones,
        visits,
        // Clamped, not merely cast: the f64 sum of positive masses can land a few ULP past 1.0,
        // and the push guard refuses anything outside 0..=1 by design.
        tail_mass: (tail as f32).clamp(0.0, 1.0),
        current_player,
        moves_remaining,
        ply_index,
        is_full_search,
        outcome: 0.0,      // placeholder → finalize_graph_outcome
        value_valid: true, // placeholder → finalize_graph_outcome
        game_length: 0,    // placeholder → finalize_graph_outcome
        game_id: -1,       // placeholder → finalize_game_graph (R345(b)(6))
    })
}

/// Stamp the per-row outcome and draw-mask onto a graph record at game end: it reads winner /
/// terminal_reason / the row's move-time player only and no cell geometry, so the outcome split
/// transfers to graph rows unchanged. `terminal_reason == 2` is the ply-cap branch, whose
/// fabricated label is masked from the value loss.
#[inline]
#[must_use]
pub fn finalize_graph_outcome(
    rec_player: i8,
    winner: Option<mantis_core::Player>,
    terminal_reason: u8,
    ply_cap_value: f32,
    draw_reward: f32,
) -> (f32, u8) {
    let outcome = match winner {
        Some(p) => {
            if p as i8 == rec_player {
                1.0
            } else {
                -1.0
            }
        }
        None => {
            if terminal_reason == 2 {
                ply_cap_value
            } else {
                draw_reward
            }
        }
    };
    (outcome, u8::from(terminal_reason != 2))
}

/// Legal-set counterpart of `sample_policy`: samples from `legal_moves` proportional to the ragged
/// `ls` mass at each move's coord, off-window covered moves included. `floor` is the no-coverage prior.
pub(crate) fn sample_policy_ls(
    ls: &LegalSetPolicy,
    legal_moves: &[(i32, i32)],
    board: &Board,
    trunk_sz: i32,
    floor: f32,
) -> Option<(i32, i32)> {
    let half = (trunk_sz - 1) / 2;
    let (bcq, bcr) = board.window_center();

    let mut probs = Vec::with_capacity(legal_moves.len());
    let mut sum = 0.0;
    for &(q, r) in legal_moves {
        let p = ls.get(q, r, bcq, bcr, trunk_sz, half, floor);
        probs.push(p);
        sum += p;
    }

    if sum < 1e-9 {
        return None;
    }

    let mut rng = rng();
    let mut rv: f32 = rng.random();
    rv *= sum;

    let mut current = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        current += p;
        if rv <= current {
            return Some(legal_moves[i]);
        }
    }
    Some(legal_moves[legal_moves.len() - 1])
}

#[cfg(test)]
mod gnn_assemble_tests {
    //! `assemble_ls_from_gnn_probs` on REAL axis-graph fixtures from the native
    //! `mantis_graph::build_axis_graph` producer. An off-window argmax MUST survive into `overflow`.
    use super::*;
    use mantis_graph::{build_axis_graph, BuildParams, StoneList, OFF_WINDOW_SLOT};

    /// Two far-apart stone clusters → bbox-midpoint window centre (17,0), trunk 19 covering
    /// q∈[8,26], so legal cells near cluster-2 (q≈29-35) fall OFF-window.
    fn spread_graph() -> (mantis_graph::AxisGraph, Vec<(i32, i32)>) {
        let mut stones: Vec<(i32, i32, i8)> = Vec::new();
        for q in 0..5i32 {
            stones.push((q, 0, 1)); // P1
        }
        for q in 30..35i32 {
            stones.push((q, 0, -1)); // P2
        }
        let params = BuildParams {
            win_length: 6,
            radius: 6,
            current_player: 1,
            moves_remaining: 2,
            trunk_size: 19,
        };
        let g = build_axis_graph(&StoneList { stones }, &params);
        let legal_coords: Vec<(i32, i32)> = g
            .legal_node_gather
            .iter()
            .map(|&row| {
                (
                    g.node_coords[row as usize * 2],
                    g.node_coords[row as usize * 2 + 1],
                )
            })
            .collect();
        (g, legal_coords)
    }

    #[test]
    fn assemble_splits_in_window_and_off_window() {
        let (g, coords) = spread_graph();
        let slots = &g.policy_scatter_index.0;
        let n_legal = slots.len();
        let n_off = slots.iter().filter(|&&s| s == OFF_WINDOW_SLOT).count();
        let n_in = n_legal - n_off;
        assert!(
            n_off > 0 && n_in > 0,
            "fixture must be a MIXED board (in {n_in}, off {n_off})"
        );

        // uniform distribution over the legal set (pre-normalized, as §4.3 emits)
        let probs = vec![1.0f32 / n_legal as f32; n_legal];
        let ls = assemble_ls_from_gnn_probs(362, &probs, slots, &coords).expect("assemble ok");

        // every off-window node landed in overflow, every in-window node at its dense slot.
        assert_eq!(
            ls.overflow.len(),
            n_off,
            "all off-window nodes retained in overflow"
        );
        for (i, &slot) in slots.iter().enumerate() {
            if slot == OFF_WINDOW_SLOT {
                let v = ls.overflow.get(&coords[i]).copied().unwrap_or(0.0);
                assert!(
                    (v - probs[i]).abs() < 1e-6,
                    "off-window prob preserved by coord"
                );
            } else {
                assert!(
                    (ls.dense[slot as usize] - probs[i]).abs() < 1e-6,
                    "in-window prob at dense slot"
                );
            }
        }
        let total: f32 = ls.dense.iter().sum::<f32>() + ls.overflow.values().sum::<f32>();
        assert!((total - 1.0).abs() < 1e-4, "no mass dropped, sum={total}");
    }

    #[test]
    fn assemble_off_window_argmax_survives() {
        // The motivating case: the model's CHOSEN move (argmax) is OFF-window. A dense-[B,362]
        // drop would erase it; this path must carry it into overflow and keep it global argmax.
        let (g, coords) = spread_graph();
        let slots = &g.policy_scatter_index.0;
        let n_legal = slots.len();
        let off_idx = slots
            .iter()
            .position(|&s| s == OFF_WINDOW_SLOT)
            .expect("fixture has an off-window legal node");

        // 0.9 mass on the off-window node, 0.1 spread over the rest → argmax off.
        let mut probs = vec![0.1f32 / (n_legal - 1) as f32; n_legal];
        probs[off_idx] = 0.9;
        let ls = assemble_ls_from_gnn_probs(362, &probs, slots, &coords).expect("assemble ok");

        let off_coord = coords[off_idx];
        let carried = ls
            .overflow
            .get(&off_coord)
            .copied()
            .expect("off-window argmax in overflow");
        assert!(
            (carried - 0.9).abs() < 1e-6,
            "off-window argmax mass preserved, got {carried}"
        );
        let dense_max = ls.dense.iter().copied().fold(0.0f32, f32::max);
        let overflow_max = ls.overflow.values().copied().fold(0.0f32, f32::max);
        assert!(overflow_max >= dense_max, "the off-window cell is the GLOBAL argmax (overflow {overflow_max} >= dense {dense_max})");
        assert!(
            (overflow_max - 0.9).abs() < 1e-6,
            "global argmax is the off-window chosen move"
        );
    }

    #[test]
    fn assemble_all_in_window_leaves_overflow_empty() {
        // A compact single-cluster board: every legal cell is in-window, so overflow stays empty.
        let stones = vec![(0, 0, 1i8), (1, 0, -1i8), (0, 1, 1i8)];
        let params = BuildParams {
            win_length: 6,
            radius: 6,
            current_player: 1,
            moves_remaining: 2,
            trunk_size: 19,
        };
        let g = build_axis_graph(&StoneList { stones }, &params);
        let slots = &g.policy_scatter_index.0;
        assert!(
            slots.iter().all(|&s| s != OFF_WINDOW_SLOT),
            "compact board is fully in-window"
        );
        let coords: Vec<(i32, i32)> = g
            .legal_node_gather
            .iter()
            .map(|&row| {
                (
                    g.node_coords[row as usize * 2],
                    g.node_coords[row as usize * 2 + 1],
                )
            })
            .collect();
        let n = slots.len();
        let probs = vec![1.0f32 / n as f32; n];
        let ls = assemble_ls_from_gnn_probs(362, &probs, slots, &coords).expect("assemble ok");
        assert!(
            ls.overflow.is_empty(),
            "no off-window cells → empty overflow"
        );
        assert!((ls.dense.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn assemble_read_back_at_builder_center_matches_and_wrong_center_misreads() {
        // The assembled `dense` slots are baked against the BUILDER's `window_center`. This pins
        // the coord/slot class: reading back at the builder centre recovers every prob, and a
        // WRONG centre shifts the in-window slot map into misreads.
        let (g, coords) = spread_graph();
        let slots = &g.policy_scatter_index.0;
        let n = slots.len();
        // Distinct probs per node so a misread is detectable (not all-equal).
        let raw: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let s: f32 = raw.iter().sum();
        let probs: Vec<f32> = raw.iter().map(|p| p / s).collect();
        let ls = assemble_ls_from_gnn_probs(362, &probs, slots, &coords).expect("assemble ok");

        let (bcq, bcr) = g.window_center;
        let (trunk, half, floor) = (19i32, 9i32, -1.0f32);

        // (a) matched-centre round-trip: recover every legal node's prob.
        for i in 0..n {
            let (q, r) = coords[i];
            let got = ls.get(q, r, bcq, bcr, trunk, half, floor);
            assert!(
                (got - probs[i]).abs() < 1e-6,
                "read-back at builder centre must recover the assembled prob (coord {q},{r})"
            );
        }

        // (b) a wrong centre shifts in-window slots, so at least one in-window coord misreads.
        let mut any_in_window = false;
        let mut any_misread = false;
        for i in 0..n {
            if slots[i] == OFF_WINDOW_SLOT {
                continue;
            }
            any_in_window = true;
            let (q, r) = coords[i];
            let got = ls.get(q, r, bcq + 3, bcr + 3, trunk, half, floor);
            if (got - probs[i]).abs() > 1e-6 {
                any_misread = true;
                break;
            }
        }
        assert!(
            any_in_window,
            "spread fixture must have in-window legal nodes"
        );
        assert!(
            any_misread,
            "a wrong window centre MUST misread in-window slots — the F1 frame class S2 guards"
        );
    }

    /// A 3-stone mid-turn board (P1 to move, 2 moves remaining).
    fn small_board() -> Board {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 ply 0
        b.apply_move(2, 0).unwrap(); // P2 first
        b.apply_move(0, 2).unwrap(); // P2 second — turn passes to P1
        b
    }

    #[test]
    fn record_position_graph_captures_stones_and_visit_mass() {
        let b = small_board();
        let (bcq, bcr) = b.window_center();
        let (trunk, half) = (19i32, 9i32);
        let legal = b.legal_moves();
        // Plant known mass on two legal cells via their in-window dense slots.
        let mut dense = vec![0.0f32; 362];
        let c0 = legal[0];
        let c1 = legal[1];
        let i0 = Board::window_flat_idx_at_geom(c0.0, c0.1, bcq, bcr, trunk, half);
        let i1 = Board::window_flat_idx_at_geom(c1.0, c1.1, bcq, bcr, trunk, half);
        assert!(i0 < 362 && i1 < 362, "chosen legal cells must be in-window");
        dense[i0] = 0.7;
        dense[i1] = 0.3;
        let ls = LegalSetPolicy {
            dense,
            overflow: FxHashMap::default(),
        };

        let rec = super::record_position_graph(
            &b,
            &ls,
            trunk,
            b.current_player as i8,
            b.moves_remaining,
            b.ply.index() as u16,
            true,
            128,
            None,
        )
        .expect("a full-mass target must record");

        // 3 stones, matching the board's occupied cells.
        assert_eq!(rec.stones.len(), 3);
        let stone_set: std::collections::HashSet<(i16, i16)> =
            rec.stones.iter().map(|&(q, r, _)| (q, r)).collect();
        assert!(
            stone_set.contains(&(0, 0))
                && stone_set.contains(&(2, 0))
                && stone_set.contains(&(0, 2))
        );

        // Visit target carries exactly the planted mass by coord (no drop).
        let vmap: std::collections::HashMap<(i16, i16), f32> =
            rec.visits.iter().map(|&(q, r, p)| ((q, r), p)).collect();
        assert!((vmap[&(c0.0 as i16, c0.1 as i16)] - 0.7).abs() < 1e-6);
        assert!((vmap[&(c1.0 as i16, c1.1 as i16)] - 0.3).abs() < 1e-6);
        assert_eq!(
            rec.visits.len(),
            2,
            "only the two nonzero-mass cells stored (sparse)"
        );
        assert_eq!(rec.outcome, 0.0, "outcome is a placeholder at record time");
        assert!(rec.is_full_search);
    }

    #[test]
    fn record_position_graph_refuses_over_cap_with_the_typed_error() {
        // This test formerly pinned the SILENT top-k truncation; it is DELETED, and an over-cap
        // target is now a typed refusal.
        let b = small_board();
        let (bcq, bcr) = b.window_center();
        let (trunk, half) = (19i32, 9i32);
        let legal = b.legal_moves();
        assert!(legal.len() >= 5);
        // A VALID distribution over 5 legal cells (Σ == 1) against max_visits=2.
        let mut dense = vec![0.0f32; 362];
        for &(q, r) in legal.iter().take(5) {
            let idx = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk, half);
            assert!(idx < 362, "chosen legal cells must be in-window");
            dense[idx] = 0.2;
        }
        let ls = LegalSetPolicy {
            dense,
            overflow: FxHashMap::default(),
        };
        let err = super::record_position_graph(&b, &ls, trunk, 1, 2, 0, true, 2, None)
            .expect_err("5 cells against max_visits=2 must raise, never silently truncate");
        match err {
            super::TargetIntegrityError::VisitSlotsExceeded { n, max, .. } => {
                assert_eq!(
                    (n, max),
                    (5, 2),
                    "error carries the offending count + the cap"
                );
            }
            other => panic!("expected VisitSlotsExceeded, got {other}"),
        }
    }

    #[test]
    fn finalize_graph_outcome_matches_178_split() {
        use mantis_core::Player;
        // Win as this row's player → +1, supervised.
        assert_eq!(
            super::finalize_graph_outcome(1, Some(Player::One), 0, -0.5, -0.1),
            (1.0, 1)
        );
        // Win as the opponent → −1, supervised.
        assert_eq!(
            super::finalize_graph_outcome(-1, Some(Player::One), 0, -0.5, -0.1),
            (-1.0, 1)
        );
        // Ply-cap (terminal_reason 2) → ply_cap_value, MASKED (value_valid 0).
        assert_eq!(
            super::finalize_graph_outcome(1, None, 2, -0.5, -0.1),
            (-0.5, 0)
        );
        // Organic draw (terminal_reason 3) → draw_reward, supervised.
        assert_eq!(
            super::finalize_graph_outcome(1, None, 3, -0.5, -0.1),
            (-0.1, 1)
        );
    }
}
