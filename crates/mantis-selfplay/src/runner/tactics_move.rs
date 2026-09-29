//! A self-play target under the tactics block: the audit's vetoed moves carry zero mass, the rest renormalised.

use mantis_core::Board;
use mantis_search::LegalSetPolicy;

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

/// Mass at or below this is no mass: the record's own `TARGET_MASS_TOL` scale.
const EMPTY: f64 = 1e-6;

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
    if removed <= EMPTY {
        return TargetEdit::Unchanged;
    }
    let total: f64 = ls
        .dense
        .iter()
        .chain(ls.overflow.values())
        .map(|&p| f64::from(p))
        .sum();
    if total - removed <= EMPTY {
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
}
