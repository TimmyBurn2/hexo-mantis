// >300 justify (R8): a row's target edits, its tail seal and the move rows read off the row written are one unit.
//! A self-play target under tactics: vetoed moves carry zero mass, the tail reaches no veto or non-block, rows counted.

use fxhash::FxHashSet;
use mantis_core::Board;
use mantis_search::mcts::TacticsCounters;
use mantis_search::LegalSetPolicy;

use super::stats::{MoveRow, TacticsTotals};
use crate::records::TARGET_MASS_TOL;
use crate::replay::hexg::GraphRecord;

/// What zeroing the vetoes did to a target.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TargetEdit {
    /// No vetoed cell held mass.
    Unchanged,
    /// The vetoed mass is gone and the rest sums to one again.
    Zeroed,
    /// Every unit of mass sat on vetoed cells: the target is left as searched, and its row records no policy.
    Emptied,
}

/// Zero `vetoes`' mass in `ls` and scale the rest back to one, over the board's global window geometry.
pub(crate) fn zero_vetoes(
    ls: &mut LegalSetPolicy,
    vetoes: &[(i32, i32)],
    board: &Board,
    trunk_sz: i32,
) -> TargetEdit {
    let (bcq, bcr) = board.window_center();
    let half = (trunk_sz - 1) / 2;
    let mut cells: Vec<(i32, i32)> = vetoes.to_vec();
    cells.sort_unstable();
    cells.dedup();
    let removed: f64 = cells
        .iter()
        .map(|&(q, r)| f64::from(ls.get(q, r, bcq, bcr, trunk_sz, half, 0.0)))
        .sum();
    if removed <= 0.0 {
        return TargetEdit::Unchanged;
    }
    let total: f64 = ls
        .dense
        .iter()
        .chain(ls.overflow.values())
        .map(|&p| f64::from(p))
        .sum();
    if total - removed <= TARGET_MASS_TOL {
        return TargetEdit::Emptied;
    }
    for &(q, r) in &cells {
        let flat = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
        if flat < ls.dense.len() {
            ls.dense[flat] = 0.0;
        } else if let Some(p) = ls.overflow.get_mut(&(q, r)) {
            *p = 0.0;
        }
    }
    let scale = (1.0 / (total - removed)) as f32;
    ls.dense
        .iter_mut()
        .chain(ls.overflow.values_mut())
        .for_each(|p| *p *= scale);
    TargetEdit::Zeroed
}

/// What sealing a sparse row's tail did.
pub(crate) struct Seal {
    /// The row's explicit support, every veto it could hold among it at zero.
    pub(crate) support: FxHashSet<(i32, i32)>,
    /// The unsealed tail would have reached a vetoed or a non-blocking cell.
    pub(crate) leaked: bool,
    /// The tail had to fold and no stored cell held mass, so the row records no policy.
    pub(crate) no_policy: bool,
}

/// Seal a sparse row's training tail: vetoes stored at zero within `cap`, else (a forced root, an unslotted veto) folded.
pub(crate) fn seal_tail(
    ls: &mut LegalSetPolicy,
    mut support: FxHashSet<(i32, i32)>,
    vetoes: &[(i32, i32)],
    root_forced: bool,
    board: &Board,
    trunk_sz: i32,
    cap: usize,
) -> Seal {
    let legal = board.legal_moves_set();
    let (bcq, bcr) = board.window_center();
    let half = (trunk_sz - 1) / 2;
    let mass = |ls: &LegalSetPolicy, (q, r): (i32, i32)| {
        f64::from(ls.get(q, r, bcq, bcr, trunk_sz, half, 0.0))
    };
    let tail: f64 = legal
        .iter()
        .filter(|c| !support.contains(c))
        .map(|&c| mass(ls, c))
        .sum();
    let mut unstored: Vec<(i32, i32)> = vetoes
        .iter()
        .copied()
        .filter(|c| legal.contains(c) && !support.contains(c))
        .collect();
    unstored.sort_unstable();
    unstored.dedup();
    let leaked = tail > 0.0 && (root_forced || !unstored.is_empty());
    let fits = support.len() + unstored.len() <= cap;
    if fits {
        support.extend(unstored);
    }
    if !(tail > 0.0 && (root_forced || !fits)) {
        return Seal {
            support,
            leaked,
            no_policy: false,
        };
    }
    let kept: f64 = support.iter().map(|&c| mass(ls, c)).sum();
    if kept <= TARGET_MASS_TOL {
        return Seal {
            support,
            leaked,
            no_policy: true,
        };
    }
    for &(q, r) in legal.iter().filter(|c| !support.contains(c)) {
        let flat = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
        if flat < ls.dense.len() {
            ls.dense[flat] = 0.0;
        } else if let Some(p) = ls.overflow.get_mut(&(q, r)) {
            *p = 0.0;
        }
    }
    let scale = (1.0 / kept) as f32;
    ls.dense
        .iter_mut()
        .chain(ls.overflow.values_mut())
        .for_each(|p| *p *= scale);
    Seal {
        support,
        leaked,
        no_policy: false,
    }
}

/// A written row with a policy whose tail still reaches a vetoed cell, or any cell past a root's blocks: run-fatal.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TailLeak {
    pub(crate) ply_index: u16,
    pub(crate) unstored_vetoes: usize,
    pub(crate) root_forced: bool,
}

impl std::error::Error for TailLeak {}

impl std::fmt::Display for TailLeak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TailLeak: the row at ply_index={} keeps a training tail with {} vetoed cell(s) unstored (root restricted \
             to its blocks: {}), so training would hand mass to a move the tactics module proved losing",
            self.ply_index, self.unstored_vetoes, self.root_forced
        )
    }
}

/// Refuse a written row whose training tail could still reach a vetoed or a non-blocking cell.
fn refuse_tail_leak(
    rec: &GraphRecord,
    vetoes: &[(i32, i32)],
    root_forced: bool,
) -> Result<(), TailLeak> {
    if !rec.is_full_search || rec.tail_mass <= 0.0 {
        return Ok(());
    }
    let stored = |c: (i32, i32)| {
        rec.visits
            .iter()
            .any(|v| (i32::from(v.0), i32::from(v.1)) == c)
    };
    let unstored: FxHashSet<(i32, i32)> = vetoes.iter().copied().filter(|&c| !stored(c)).collect();
    let unstored_vetoes = unstored.len();
    if root_forced || unstored_vetoes > 0 {
        return Err(TailLeak {
            ply_index: rec.ply_index,
            unstored_vetoes,
            root_forced,
        });
    }
    Ok(())
}

/// The record's side of a move: the row written, the arm drawn, and what the root and the vetoes made of it.
pub(crate) struct Written<'a> {
    pub(crate) record: Option<&'a GraphRecord>,
    pub(crate) drawn_full: bool,
    pub(crate) decided: bool,
    pub(crate) decided_lost: bool,
    pub(crate) edit: TargetEdit,
    /// The row's unsealed tail would have reached a vetoed or a non-blocking cell.
    pub(crate) tail_leaked: bool,
    /// The seal's fold found no stored mass, so the row records no policy.
    pub(crate) tail_emptied: bool,
    /// The search restricted its root to the blocks.
    pub(crate) root_forced: bool,
    /// A quick draw's decided root, played with no search: its row is value-only.
    pub(crate) unsearched: bool,
}

/// Refuse a row whose tail leaks (run-fatal, counted nowhere), else sum the search's rows, its record's one kind and the two cross-counts.
pub(crate) fn account_row(
    totals: &TacticsTotals,
    rows: &TacticsCounters,
    written: &Written<'_>,
    vetoes: &[(i32, i32)],
) -> Result<(), TailLeak> {
    if let Some(rec) = written.record {
        refuse_tail_leak(rec, vetoes, written.root_forced)?;
    }
    totals.add_search(rows);
    // Counted ahead of the full-draw filter: every one of these rows is a quick draw's.
    if written.unsearched {
        totals.add_move(MoveRow::UnsearchedDecided);
    }
    let Some(rec) = written.record.filter(|_| written.drawn_full) else {
        return Ok(());
    };
    let row = if written.decided && rec.is_full_search {
        Some(MoveRow::ProvenRoot)
    } else if written.decided_lost && !rec.is_full_search {
        Some(MoveRow::DecidedLost)
    } else if written.edit == TargetEdit::Emptied && !rec.is_full_search {
        Some(MoveRow::EmptiedTarget)
    } else if written.tail_emptied && !rec.is_full_search {
        Some(MoveRow::TailEmptied)
    } else if written.edit == TargetEdit::Zeroed
        && rec.is_full_search
        && vetoes_hold_no_mass(rec, vetoes)
    {
        Some(MoveRow::VetoedTarget)
    } else {
        None
    };
    if let Some(row) = row {
        totals.add_move(row);
    }
    // Every all-vetoed root whatever its row holds: the emptied rows equal it only while each records no policy.
    if written.edit == TargetEdit::Emptied && !written.decided_lost {
        totals.add_move(MoveRow::VetoedAll);
    }
    // Every row whose unsealed tail would have reached a vetoed or non-blocking cell, whatever the seal made of it.
    if written.tail_leaked {
        totals.add_move(MoveRow::TailLeak);
    }
    Ok(())
}

/// Every vetoed cell is stored at zero, or is absent from a row with no tail (a tail would hand it mass in training).
fn vetoes_hold_no_mass(rec: &GraphRecord, vetoes: &[(i32, i32)]) -> bool {
    vetoes.iter().all(|&(q, r)| {
        match rec
            .visits
            .iter()
            .find(|v| (i32::from(v.0), i32::from(v.1)) == (q, r))
        {
            Some(v) => v.2 == 0.0,
            None => rec.tail_mass == 0.0,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRUNK: i32 = 19;

    /// A board whose window centre is the origin, and a target of `mass` at each `cells` entry.
    fn target(cells: &[((i32, i32), f32)]) -> (Board, LegalSetPolicy) {
        let mut board = Board::new();
        board.apply_move(0, 0).expect("the first stone");
        let (bcq, bcr) = board.window_center();
        let half = (TRUNK - 1) / 2;
        let mut ls = LegalSetPolicy {
            dense: vec![0.0; (TRUNK * TRUNK) as usize],
            ..LegalSetPolicy::default()
        };
        for &((q, r), p) in cells {
            let flat = Board::window_flat_idx_at_geom(q, r, bcq, bcr, TRUNK, half);
            if flat < ls.dense.len() {
                ls.dense[flat] = p;
            } else {
                ls.overflow.insert((q, r), p);
            }
        }
        (board, ls)
    }

    fn mass(ls: &LegalSetPolicy, board: &Board, cell: (i32, i32)) -> f32 {
        let (bcq, bcr) = board.window_center();
        ls.get(cell.0, cell.1, bcq, bcr, TRUNK, (TRUNK - 1) / 2, 0.0)
    }

    #[test]
    fn a_vetoed_move_loses_its_mass_and_the_rest_renormalises() {
        let (board, mut ls) = target(&[((1, 0), 0.5), ((2, 0), 0.3), ((40, 0), 0.2)]);
        assert_eq!(
            zero_vetoes(&mut ls, &[(1, 0)], &board, TRUNK),
            TargetEdit::Zeroed
        );
        assert_eq!(mass(&ls, &board, (1, 0)), 0.0);
        assert!((mass(&ls, &board, (2, 0)) - 0.6).abs() < 1e-6);
        assert!(
            (mass(&ls, &board, (40, 0)) - 0.4).abs() < 1e-6,
            "an off-window cell renormalises too"
        );
    }

    #[test]
    fn a_vetoed_move_with_a_trace_of_mass_is_still_zeroed() {
        let (board, mut ls) = target(&[((1, 0), 1e-8), ((2, 0), 1.0)]);
        assert_eq!(
            zero_vetoes(&mut ls, &[(1, 0)], &board, TRUNK),
            TargetEdit::Zeroed
        );
        assert_eq!(mass(&ls, &board, (1, 0)), 0.0);
    }

    #[test]
    fn a_veto_on_no_mass_leaves_the_target_and_all_mass_vetoed_empties_it() {
        let (board, mut ls) = target(&[((1, 0), 1.0)]);
        assert_eq!(
            zero_vetoes(&mut ls, &[(3, 0), (60, 0)], &board, TRUNK),
            TargetEdit::Unchanged
        );
        assert_eq!(mass(&ls, &board, (1, 0)), 1.0);
        assert_eq!(
            zero_vetoes(&mut ls, &[(1, 0)], &board, TRUNK),
            TargetEdit::Emptied
        );
        assert_eq!(
            mass(&ls, &board, (1, 0)),
            1.0,
            "an emptied target stays as searched: its row records no policy"
        );
    }

    /// The move rows `account_row` adds for one written row of an all-vetoed root, by name.
    fn counted(
        rec: &GraphRecord,
        drawn_full: bool,
        decided_lost: bool,
    ) -> Vec<(&'static str, u64)> {
        let totals = TacticsTotals::new();
        let written = Written {
            record: Some(rec),
            drawn_full,
            decided: false,
            decided_lost,
            edit: TargetEdit::Emptied,
            tail_leaked: false,
            tail_emptied: false,
            root_forced: false,
            unsearched: false,
        };
        account_row(&totals, &TacticsCounters::default(), &written, &[]).expect("no tail, no leak");
        totals
            .snapshot()
            .into_iter()
            .filter(|&(_, v)| v > 0)
            .collect()
    }

    #[test]
    fn an_all_vetoed_row_counts_whatever_it_wrote_and_as_emptied_only_without_a_policy() {
        let mut rec = GraphRecord::default();
        assert_eq!(
            counted(&rec, true, false),
            vec![("emptied_target_rows", 1), ("vetoed_all_rows", 1)]
        );
        assert_eq!(
            counted(&rec, true, true),
            vec![("decided_lost_rows", 1)],
            "lost first"
        );
        assert!(
            counted(&rec, false, false).is_empty(),
            "a quick draw adds no move row"
        );
        rec.is_full_search = true;
        assert_eq!(
            counted(&rec, true, false),
            vec![("vetoed_all_rows", 1)],
            "a row written with a policy is not an emptied one"
        );
    }

    fn cells(set: &FxHashSet<(i32, i32)>) -> Vec<(i32, i32)> {
        let mut v: Vec<(i32, i32)> = set.iter().copied().collect();
        v.sort_unstable();
        v
    }

    fn stored(cells: &[(i32, i32)]) -> FxHashSet<(i32, i32)> {
        cells.iter().copied().collect()
    }

    /// Visited (1,0) and (2,0) hold 0.8, unvisited (3,0) the tail's 0.2; the veto (0,1) holds nothing.
    fn searched() -> (Board, LegalSetPolicy) {
        target(&[((1, 0), 0.5), ((2, 0), 0.3), ((3, 0), 0.2)])
    }

    /// PLANTED BREAK: skip the pinning and the unvisited veto is left for the tail to feed.
    #[test]
    fn an_unvisited_veto_is_stored_at_zero_and_the_tail_stays_for_the_rest() {
        let (board, mut ls) = searched();
        let seal = seal_tail(
            &mut ls,
            stored(&[(1, 0), (2, 0)]),
            &[(0, 1)],
            false,
            &board,
            TRUNK,
            16,
        );
        assert_eq!(cells(&seal.support), vec![(0, 1), (1, 0), (2, 0)]);
        assert!(seal.leaked && !seal.no_policy);
        assert_eq!(mass(&ls, &board, (3, 0)), 0.2, "the tail keeps its share");
        assert_eq!(mass(&ls, &board, (0, 1)), 0.0);
    }

    /// PLANTED BREAK: skip the fold and the tail spreads over the non-blocking cells.
    #[test]
    fn a_root_restricted_to_its_blocks_folds_its_tail_into_the_stored_cells() {
        let (board, mut ls) = searched();
        let seal = seal_tail(
            &mut ls,
            stored(&[(1, 0), (2, 0)]),
            &[],
            true,
            &board,
            TRUNK,
            16,
        );
        assert!(seal.leaked && !seal.no_policy);
        assert_eq!(mass(&ls, &board, (3, 0)), 0.0);
        assert!((mass(&ls, &board, (1, 0)) - 0.625).abs() < 1e-6);
        assert!((mass(&ls, &board, (2, 0)) - 0.375).abs() < 1e-6);
    }

    #[test]
    fn a_veto_the_row_has_no_slot_for_folds_the_tail_instead() {
        let (board, mut ls) = searched();
        let seal = seal_tail(
            &mut ls,
            stored(&[(1, 0), (2, 0)]),
            &[(0, 1)],
            false,
            &board,
            TRUNK,
            2,
        );
        assert_eq!(
            cells(&seal.support),
            vec![(1, 0), (2, 0)],
            "within the slots"
        );
        assert!(seal.leaked);
        assert_eq!(
            mass(&ls, &board, (3, 0)),
            0.0,
            "no tail is left to reach the veto"
        );
    }

    #[test]
    fn a_row_with_no_veto_no_restriction_or_no_tail_is_left_as_searched() {
        let (board, mut ls) = searched();
        let seal = seal_tail(
            &mut ls,
            stored(&[(1, 0), (2, 0)]),
            &[],
            false,
            &board,
            TRUNK,
            16,
        );
        assert!(!seal.leaked && cells(&seal.support) == vec![(1, 0), (2, 0)]);
        assert_eq!(mass(&ls, &board, (3, 0)), 0.2);
        let seal = seal_tail(
            &mut ls,
            stored(&[(1, 0), (2, 0), (3, 0)]),
            &[(0, 1)],
            true,
            &board,
            TRUNK,
            16,
        );
        assert!(
            !seal.leaked,
            "every unit of mass is stored: there is no tail to leak"
        );
    }

    #[test]
    fn a_fold_over_cells_holding_nothing_records_no_policy() {
        let (board, mut ls) = target(&[((3, 0), 1.0)]);
        let seal = seal_tail(&mut ls, stored(&[(1, 0)]), &[], true, &board, TRUNK, 16);
        assert!(seal.leaked && seal.no_policy);
    }

    fn row(full: bool, tail: f32, visits: &[(i16, i16)]) -> GraphRecord {
        GraphRecord {
            is_full_search: full,
            tail_mass: tail,
            visits: visits.iter().map(|&(q, r)| (q, r, 0.0)).collect(),
            ..GraphRecord::default()
        }
    }

    #[test]
    fn a_written_row_whose_tail_reaches_an_unstored_veto_or_a_forced_roots_rest_is_refused() {
        let leaked = refuse_tail_leak(&row(true, 0.2, &[(1, 0)]), &[(0, 1)], false);
        assert_eq!(leaked.map_err(|e| e.unstored_vetoes), Err(1));
        assert!(refuse_tail_leak(&row(true, 0.2, &[(1, 0), (0, 1)]), &[(0, 1)], false).is_ok());
        assert!(refuse_tail_leak(&row(true, 0.2, &[(1, 0)]), &[], true).is_err());
        assert!(
            refuse_tail_leak(&row(true, 0.0, &[(1, 0)]), &[(0, 1)], true).is_ok(),
            "no tail"
        );
        assert!(
            refuse_tail_leak(&row(false, 0.2, &[(1, 0)]), &[(0, 1)], true).is_ok(),
            "no policy"
        );
    }

    /// PLANTED BREAK: drop the refusal from `account_row` and a leaking row is counted and written.
    #[test]
    fn a_leaking_row_is_refused_before_any_row_is_counted() {
        let totals = TacticsTotals::new();
        let rec = row(true, 0.2, &[(1, 0)]);
        let written = Written {
            record: Some(&rec),
            drawn_full: true,
            decided: false,
            decided_lost: false,
            edit: TargetEdit::Zeroed,
            tail_leaked: true,
            tail_emptied: false,
            root_forced: false,
            unsearched: false,
        };
        let refused = account_row(
            &totals,
            &TacticsCounters::default(),
            &written,
            &[(0, 1), (0, 1)],
        );
        assert_eq!(
            refused.map_err(|e| e.unstored_vetoes),
            Err(1),
            "a repeated veto is one cell"
        );
        assert!(totals.snapshot().iter().all(|&(_, v)| v == 0));
    }

    /// PLANTED BREAK: count it behind the full-draw filter and a quick draw's unsearched row reads nowhere.
    #[test]
    fn an_unsearched_decided_row_counts_on_its_own_row_beside_its_roots_rows() {
        let totals = TacticsTotals::new();
        let rec = row(false, 0.0, &[]);
        let written = Written {
            record: Some(&rec),
            drawn_full: false,
            decided: true,
            decided_lost: false,
            edit: TargetEdit::Unchanged,
            tail_leaked: false,
            tail_emptied: false,
            root_forced: false,
            unsearched: true,
        };
        let root = TacticsCounters {
            root_proofs_found: 1,
            ..TacticsCounters::default()
        };
        account_row(&totals, &root, &written, &[]).expect("no tail, no leak");
        let counted: Vec<(&str, u64)> = totals
            .snapshot()
            .into_iter()
            .filter(|&(_, v)| v > 0)
            .collect();
        assert_eq!(
            counted,
            vec![("root_proofs_found", 1), ("unsearched_decided_rows", 1)]
        );
    }

    /// PLANTED BREAK: drop or swap either seal row's increment and its name reads wrong.
    #[test]
    fn a_sealed_leak_and_an_emptied_fold_count_on_their_own_rows() {
        let by_name = |rec: &GraphRecord, leaked: bool, emptied: bool| {
            let totals = TacticsTotals::new();
            let written = Written {
                record: Some(rec),
                drawn_full: true,
                decided: false,
                decided_lost: false,
                edit: TargetEdit::Unchanged,
                tail_leaked: leaked,
                tail_emptied: emptied,
                root_forced: false,
                unsearched: false,
            };
            account_row(&totals, &TacticsCounters::default(), &written, &[]).expect("sealed");
            totals
                .snapshot()
                .into_iter()
                .filter(|&(_, v)| v > 0)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            by_name(&row(true, 0.0, &[(1, 0)]), true, false),
            vec![("tail_leak_rows", 1)]
        );
        assert_eq!(
            by_name(&row(false, 0.0, &[(1, 0)]), false, true),
            vec![("tail_emptied_rows", 1)]
        );
    }
}
