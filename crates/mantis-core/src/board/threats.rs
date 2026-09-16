//! The two-stone-turn threat unit: pure length-6 windows a side completes within a stone budget.

use super::state::{Board, Cell, Player, HEX_AXES};

/// A pure length-6 window (only `player`'s stones and empties) with `n_empty` (1 or 2) legal
/// empty cells: `player` completes six there with `n_empty` stones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenWindow {
    empties: [(i32, i32); 2],
    n_empty: u8,
}

impl OpenWindow {
    /// The cells `player` still needs, 1 or 2 of them.
    #[must_use]
    pub fn empties(&self) -> &[(i32, i32)] {
        &self.empties[..self.n_empty as usize]
    }

    /// True when a stone on `cell` kills this window.
    #[must_use]
    pub fn is_hit_by(&self, cell: (i32, i32)) -> bool {
        self.empties().contains(&cell)
    }
}

/// The fewest stones hitting every window (0, 1 or 2; `None` past two): a hitting set of size
/// <= 2 holds a cell of the first window and, if that misses one, a cell of the first it misses.
#[must_use]
pub fn min_hitting_stones(windows: &[OpenWindow]) -> Option<u8> {
    let Some(first) = windows.first() else {
        return Some(0);
    };
    if first
        .empties()
        .iter()
        .any(|&c| windows.iter().all(|w| w.is_hit_by(c)))
    {
        return Some(1);
    }
    for &c1 in first.empties() {
        let Some(second) = windows.iter().find(|w| !w.is_hit_by(c1)) else {
            return Some(1);
        };
        if second
            .empties()
            .iter()
            .any(|&c2| windows.iter().all(|w| w.is_hit_by(c1) || w.is_hit_by(c2)))
        {
            return Some(2);
        }
    }
    None
}

impl Board {
    /// Every pure window of `player` completable with at most `max_empty` (1 or 2) stones, each
    /// window once; a window counts only if each of its empties is a legal move.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // line indices are <= 10
    pub fn open_windows(&self, player: Player, max_empty: u8) -> Vec<OpenWindow> {
        let pcell = match player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        let max_empty = max_empty.min(2);
        let legal = self.legal_moves_set();
        let mut out = Vec::new();
        for (&(sq, sr), &c) in &self.cells {
            if c != pcell {
                continue;
            }
            for &(dq, dr) in &HEX_AXES {
                // The 11-cell line through the stone, the stone at index 5.
                let mut line = [Cell::Empty; 11];
                for (i, slot) in line.iter_mut().enumerate() {
                    let k = i as i32 - 5;
                    *slot = if k == 0 {
                        pcell
                    } else {
                        self.get(sq + k * dq, sr + k * dr)
                    };
                }
                for start in 0..=5usize {
                    let window = &line[start..start + 6];
                    // Reported from the window's FIRST player stone only, so each window is once.
                    if window[..5 - start].contains(&pcell) {
                        continue;
                    }
                    let mut empties = [(0i32, 0i32); 2];
                    let mut n_empty = 0u8;
                    let mut dead = false;
                    for (j, &x) in window.iter().enumerate() {
                        if x == Cell::Empty {
                            if n_empty == 2 {
                                dead = true;
                                break;
                            }
                            let off = start as i32 + j as i32 - 5;
                            empties[n_empty as usize] = (sq + off * dq, sr + off * dr);
                            n_empty += 1;
                        } else if x != pcell {
                            dead = true;
                            break;
                        }
                    }
                    if dead || n_empty == 0 || n_empty > max_empty {
                        continue;
                    }
                    if !empties[..n_empty as usize]
                        .iter()
                        .all(|e| legal.contains(e))
                    {
                        continue;
                    }
                    out.push(OpenWindow { empties, n_empty });
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Play `seq` from an empty board under the real cadence (1 stone, then 2 per turn).
    fn played(seq: &[(i32, i32)]) -> Board {
        let mut b = Board::new();
        for &(q, r) in seq {
            b.apply_move(q, r)
                .expect("test sequence places on empty cells");
        }
        b
    }

    fn sorted_empties(ws: &[OpenWindow]) -> Vec<Vec<(i32, i32)>> {
        let mut v: Vec<Vec<(i32, i32)>> = ws
            .iter()
            .map(|w| {
                let mut e = w.empties().to_vec();
                e.sort_unstable();
                e
            })
            .collect();
        v.sort();
        v
    }

    /// P1 stones at (0..4, 0) with P2's six SCATTERED far away (pairs on rows 20, 22, 24 — no
    /// P2 window holds more than three): an open five along the E/W axis.
    /// P1 idx 0; P2 idx 1,2; P1 idx 3,4; P2 idx 5,6; P1 idx 7,8; P2 idx 9,10.
    const FIVE: [(i32, i32); 11] = [
        (0, 0),
        (20, 20),
        (21, 20),
        (1, 0),
        (2, 0),
        (20, 22),
        (21, 22),
        (3, 0),
        (4, 0),
        (20, 24),
        (21, 24),
    ];

    #[test]
    fn an_open_five_is_two_one_stone_windows_and_four_two_stone_windows() {
        let b = played(&FIVE);
        assert_eq!(b.current_player, Player::One);
        let ones = b.open_windows(Player::One, 1);
        assert_eq!(sorted_empties(&ones), vec![vec![(-1, 0)], vec![(5, 0)]]);
        let twos = b.open_windows(Player::One, 2);
        // The 1-empty windows [-1..4] and [0..5], plus [-2..3] {-2,-1}, [1..6] {5,6}.
        assert_eq!(
            sorted_empties(&twos),
            vec![
                vec![(-2, 0), (-1, 0)],
                vec![(-1, 0)],
                vec![(5, 0)],
                vec![(5, 0), (6, 0)]
            ]
        );
        assert!(
            b.open_windows(Player::Two, 2).is_empty(),
            "P2's scattered stones open nothing"
        );
    }

    #[test]
    fn a_window_with_an_opponent_stone_is_dead() {
        // Same five, then P1 plays elsewhere and P2 caps the west end at (-1, 0).
        let mut seq = FIVE.to_vec();
        seq.extend([(30, 30), (31, 30), (-1, 0)]);
        let b = played(&seq);
        let ones = b.open_windows(Player::One, 1);
        assert_eq!(sorted_empties(&ones), vec![vec![(5, 0)]]);
    }

    #[test]
    fn a_gapped_five_is_reported_and_every_reported_empty_is_legal() {
        // P1: (0,0),(1,0),(2,0),(4,0),(5,0) — the gap (3,0) completes six in one stone.
        let seq = [
            (0, 0),
            (20, 20),
            (21, 20),
            (1, 0),
            (2, 0),
            (20, 22),
            (21, 22),
            (4, 0),
            (5, 0),
            (20, 24),
            (21, 24),
        ];
        let b = played(&seq);
        let ones = sorted_empties(&b.open_windows(Player::One, 1));
        assert!(
            ones.contains(&vec![(3, 0)]),
            "the gap cell completes six: {ones:?}"
        );
        let legal = b.legal_moves_set();
        for w in b.open_windows(Player::One, 2) {
            for e in w.empties() {
                assert!(legal.contains(e));
            }
        }
    }

    #[test]
    fn each_window_is_reported_once() {
        let b = played(&FIVE);
        let twos = b.open_windows(Player::One, 2);
        let mut seen = sorted_empties(&twos);
        let n = seen.len();
        seen.dedup();
        assert_eq!(
            seen.len(),
            n,
            "a window reached from two of its stones was reported twice"
        );
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)] // e.len() <= 2
    fn hitting_stones_none_one_two_and_more() {
        let w = |e: &[(i32, i32)]| {
            let mut empties = [(0, 0); 2];
            empties[..e.len()].copy_from_slice(e);
            OpenWindow {
                empties,
                n_empty: e.len() as u8,
            }
        };
        assert_eq!(min_hitting_stones(&[]), Some(0));
        // One four: one stone in either empty.
        assert_eq!(min_hitting_stones(&[w(&[(0, 0), (1, 0)])]), Some(1));
        // The three windows of an open four __XXXX__: {-2,-1}, {-1,4}, {4,5} — no single cell.
        assert_eq!(
            min_hitting_stones(&[
                w(&[(-2, 0), (-1, 0)]),
                w(&[(-1, 0), (4, 0)]),
                w(&[(4, 0), (5, 0)])
            ]),
            Some(2)
        );
        // Two fives sharing their cell: one stone.
        assert_eq!(min_hitting_stones(&[w(&[(7, 7)]), w(&[(7, 7)])]), Some(1));
        // Three disjoint fives: more than two stones.
        assert_eq!(
            min_hitting_stones(&[w(&[(1, 1)]), w(&[(2, 2)]), w(&[(3, 3)])]),
            None
        );
    }

    #[test]
    fn the_open_four_costs_a_whole_turn_and_beats_one_stone() {
        // P2 builds __OOOO__ at (10..13, 5); after 7 half-moves P1 is to move with two stones.
        let seq = [(0, 0), (10, 5), (11, 5), (1, 0), (2, 0), (12, 5), (13, 5)];
        let b = played(&seq);
        assert_eq!(
            (b.current_player, b.moves_remaining, b.ply.index()),
            (Player::One, 2, 7)
        );
        let threats = b.open_windows(Player::Two, 2);
        assert_eq!(min_hitting_stones(&threats), Some(2));
    }
}
