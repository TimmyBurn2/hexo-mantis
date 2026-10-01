//! A self-play target under tactics: vetoed moves carry zero mass, and the move rows are read off the row written.

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

/// The record's side of a move: the row written, the arm drawn, and what the root and the vetoes made of it.
pub(crate) struct Written<'a> {
    pub(crate) record: Option<&'a GraphRecord>,
    pub(crate) drawn_full: bool,
    pub(crate) decided: bool,
    pub(crate) decided_lost: bool,
    pub(crate) edit: TargetEdit,
}

/// Sum a search's tactics rows, then its record's one kind (disjoint, lost first) and the all-vetoed cross-count.
pub(crate) fn count_rows(
    totals: &TacticsTotals,
    rows: &TacticsCounters,
    written: &Written<'_>,
    vetoes: &[(i32, i32)],
) {
    totals.add_search(rows);
    let Some(rec) = written.record.filter(|_| written.drawn_full) else {
        return;
    };
    let row = if written.decided && rec.is_full_search {
        Some(MoveRow::ProvenRoot)
    } else if written.decided_lost && !rec.is_full_search {
        Some(MoveRow::DecidedLost)
    } else if written.edit == TargetEdit::Emptied && !rec.is_full_search {
        Some(MoveRow::EmptiedTarget)
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

    /// The move rows `count_rows` adds for one written row of an all-vetoed root, by name.
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
        };
        count_rows(&totals, &TacticsCounters::default(), &written, &[]);
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
}
