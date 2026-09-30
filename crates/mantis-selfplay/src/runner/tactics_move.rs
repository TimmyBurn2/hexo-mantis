//! A self-play target under tactics: vetoed moves carry zero mass, and the move rows are read off the row written.

use fxhash::FxHashSet;
use mantis_core::Board;
use mantis_search::mcts::{all_vetoed, TacticsCounters};
use mantis_search::LegalSetPolicy;

use super::stats::{MoveRow, TacticsTotals};
use crate::replay::hexg::GraphRecord;

/// What zeroing the vetoes did to a target.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TargetEdit {
    /// No vetoed cell held mass.
    Unchanged,
    /// The vetoed mass is gone and the rest sums to one again.
    Zeroed,
    /// Every unit of mass sat on vetoed cells: the target is left as searched, for the root's re-search to replace.
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
    if all_vetoed(total, removed) {
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

/// A decided root's target is mixed with its proof where it holds less than this on the proof.
pub(crate) const PROOF_MASS_FLOOR: f64 = 0.5;

/// The proof's weight in the mixture: the target becomes `(1 − α)·searched + α·proof`.
pub(crate) const PROOF_MIX_ALPHA: f32 = 0.5;

/// Mix `proof` (its stones in equal shares) into a decided root's target where it holds under the floor on them.
pub(crate) fn mix_proof(
    ls: &mut LegalSetPolicy,
    proof: &[(i32, i32)],
    board: &Board,
    trunk_sz: i32,
) -> bool {
    let mut cells = proof.to_vec();
    cells.sort_unstable();
    cells.dedup();
    if cells.is_empty() {
        return false;
    }
    let (bcq, bcr) = board.window_center();
    let half = (trunk_sz - 1) / 2;
    let on_proof: f64 = cells
        .iter()
        .map(|&(q, r)| f64::from(ls.get(q, r, bcq, bcr, trunk_sz, half, 0.0)))
        .sum();
    if on_proof >= PROOF_MASS_FLOOR {
        return false;
    }
    ls.dense
        .iter_mut()
        .chain(ls.overflow.values_mut())
        .for_each(|p| *p *= 1.0 - PROOF_MIX_ALPHA);
    let share = PROOF_MIX_ALPHA / cells.len() as f32;
    for (q, r) in cells {
        let flat = Board::window_flat_idx_at_geom(q, r, bcq, bcr, trunk_sz, half);
        if flat < ls.dense.len() {
            ls.dense[flat] += share;
        } else {
            *ls.overflow.entry((q, r)).or_insert(0.0) += share;
        }
    }
    true
}

/// The sparse row's support: its candidates, and `pinned` cells the prior-shaped tail would misstate, within `cap`.
pub(crate) fn fitted_support(
    candidates: Vec<(i32, i32)>,
    pinned: &[(i32, i32)],
    target: &LegalSetPolicy,
    board: &Board,
    trunk_sz: i32,
    cap: usize,
) -> FxHashSet<(i32, i32)> {
    let legal = board.legal_moves_set();
    let mut pins: Vec<(i32, i32)> = pinned
        .iter()
        .copied()
        .filter(|c| legal.contains(c))
        .collect();
    pins.sort_unstable();
    pins.dedup();
    pins.truncate(cap);
    let (bcq, bcr) = board.window_center();
    let half = (trunk_sz - 1) / 2;
    let mut rest: Vec<((i32, i32), f32)> = candidates
        .into_iter()
        .filter(|c| !pins.contains(c))
        .map(|(q, r)| ((q, r), target.get(q, r, bcq, bcr, trunk_sz, half, 0.0)))
        .collect();
    rest.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    rest.truncate(cap - pins.len());
    pins.into_iter()
        .chain(rest.into_iter().map(|(c, _)| c))
        .collect()
}

/// The record's side of a move: the row written, the arm drawn, and what the root and the vetoes made of it.
pub(crate) struct Written<'a> {
    pub(crate) record: Option<&'a GraphRecord>,
    pub(crate) drawn_full: bool,
    pub(crate) decided: bool,
    pub(crate) decided_lost: bool,
    pub(crate) edit: TargetEdit,
    /// The decided root's target took its proof in the mixture.
    pub(crate) mixed: bool,
}

/// Sum a search's tactics rows, then the one move row its written record carries; the kinds are disjoint, lost first.
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
    // A sub-count of the proven-root rows, not a kind of its own.
    if matches!(row, Some(MoveRow::ProvenRoot)) && written.mixed {
        totals.add_move(MoveRow::Mixed);
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

    fn support(cells: &[(i32, i32)]) -> Vec<(i32, i32)> {
        let mut v = cells.to_vec();
        v.sort_unstable();
        v
    }

    fn sorted(set: fxhash::FxHashSet<(i32, i32)>) -> Vec<(i32, i32)> {
        let mut v: Vec<(i32, i32)> = set.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// PLANTED BREAK: drop `fitted_support`'s pins and a vetoed cell leaves the row, for the tail to feed.
    #[test]
    fn pinned_cells_take_slots_from_the_lowest_mass_candidates_within_the_cap() {
        let (board, ls) = target(&[((1, 0), 0.5), ((2, 0), 0.3), ((3, 0), 0.15), ((4, 0), 0.05)]);
        let candidates = vec![(1, 0), (2, 0), (3, 0), (4, 0)];
        assert_eq!(
            sorted(fitted_support(
                candidates.clone(),
                &[],
                &ls,
                &board,
                TRUNK,
                4
            )),
            support(&candidates),
            "no pin: the searched candidates, as before"
        );
        assert_eq!(
            sorted(fitted_support(
                candidates.clone(),
                &[(5, 0)],
                &ls,
                &board,
                TRUNK,
                4
            )),
            support(&[(1, 0), (2, 0), (3, 0), (5, 0)]),
            "a pin displaces the lowest-mass candidate"
        );
        assert_eq!(
            sorted(fitted_support(
                candidates.clone(),
                &[(5, 0), (2, 0)],
                &ls,
                &board,
                TRUNK,
                4
            )),
            support(&[(1, 0), (2, 0), (3, 0), (5, 0)]),
            "a pinned candidate keeps its one slot"
        );
        assert_eq!(
            sorted(fitted_support(
                candidates,
                &[(5, 0), (400, 400)],
                &ls,
                &board,
                TRUNK,
                8
            )),
            support(&[(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]),
            "room for all, and an illegal pin takes none"
        );
    }

    #[test]
    fn a_re_searched_row_stores_its_vetoes_at_zero_so_the_tail_cannot_reach_them() {
        let (board, ls) = target(&[((1, 0), 0.6), ((2, 0), 0.3), ((3, 0), 0.1)]);
        let set = fitted_support(vec![(1, 0), (2, 0)], &[(-2, 0)], &ls, &board, TRUNK, 2);
        let rec = crate::records::record_position_graph(
            &board,
            &ls,
            TRUNK,
            board.current_player as i8,
            board.moves_remaining,
            board.ply.index() as u16,
            true,
            2,
            Some(&set),
        )
        .expect("a fitted row records");
        let stored: Vec<(i16, i16, f32)> = rec.visits.clone();
        assert!(
            stored.contains(&(-2, 0, 0.0)),
            "the veto is explicit, at zero: {stored:?}"
        );
        assert!(stored.contains(&(1, 0, 0.6)));
        assert!(
            (rec.tail_mass - 0.4).abs() < 1e-6,
            "the displaced candidates' mass joins the tail"
        );
    }

    fn masses(ls: &LegalSetPolicy, board: &Board, cells: &[(i32, i32)]) -> Vec<f32> {
        cells.iter().map(|&c| mass(ls, board, c)).collect()
    }

    /// PLANTED BREAK: return before the mixture and a weak row keeps 0.3 on its proof.
    #[test]
    fn a_target_weak_on_its_proof_mixes_it_in_at_one_half_and_a_strong_one_is_left() {
        let cells = [(1, 0), (2, 0), (3, 0)];
        let (board, mut ls) = target(&[((1, 0), 0.7), ((2, 0), 0.2), ((3, 0), 0.1)]);
        assert!(mix_proof(&mut ls, &[(2, 0), (3, 0)], &board, TRUNK));
        for (got, want) in masses(&ls, &board, &cells)
            .into_iter()
            .zip([0.35, 0.35, 0.3])
        {
            assert!((got - want).abs() < 1e-6, "{got} against {want}");
        }

        let (board, mut ls) = target(&[((1, 0), 0.7), ((2, 0), 0.2), ((3, 0), 0.1)]);
        assert!(
            !mix_proof(&mut ls, &[(1, 0)], &board, TRUNK),
            "0.7 on the proof: searched"
        );
        assert!(
            !mix_proof(&mut ls, &[], &board, TRUNK),
            "no proof: nothing to mix"
        );
        assert_eq!(masses(&ls, &board, &cells), vec![0.7, 0.2, 0.1]);

        let (board, mut ls) = target(&[((1, 0), 0.5), ((2, 0), 0.5)]);
        assert!(
            !mix_proof(&mut ls, &[(2, 0)], &board, TRUNK),
            "exactly one half is not below it"
        );
    }

    #[test]
    fn a_one_stone_proof_is_one_hot_and_an_off_window_proof_cell_takes_its_share() {
        let (board, mut ls) = target(&[((1, 0), 0.9), ((3, 0), 0.1)]);
        assert!(mix_proof(&mut ls, &[(3, 0)], &board, TRUNK));
        assert_eq!(masses(&ls, &board, &[(1, 0), (3, 0)]), vec![0.45, 0.55]);

        let (board, mut ls) = target(&[((1, 0), 1.0)]);
        assert!(mix_proof(&mut ls, &[(40, 0), (2, 0)], &board, TRUNK));
        assert_eq!(
            masses(&ls, &board, &[(1, 0), (2, 0), (40, 0)]),
            vec![0.5, 0.25, 0.25],
            "the off-window cell is the overflow's"
        );
    }
}
