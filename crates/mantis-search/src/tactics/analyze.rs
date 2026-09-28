//! Exact one-turn tactics for the side to move, from window counts: its finish, a lost cover, the forced blocks.

use mantis_core::board::{min_hitting_stones, Board, OpenWindow};

/// A position decided within one turn, for the side to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terminal {
    /// The mover completes six with the stones it has left.
    Win,
    /// The opponent's threat windows need more blockers than the mover has stones: lost on cover.
    Loss,
}

/// One turn's facts for the side to move. `forced` is empty unless the opponent holds blockable threats.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeafTactics {
    pub terminal: Option<Terminal>,
    /// Every empty cell of the opponent's threat windows, sorted: a stone of this turn must block there.
    pub forced: Vec<(i32, i32)>,
    /// The fewest stones hitting every opponent five (`None` past two), set when the threats are blockable.
    pub fives_cover: Option<u8>,
    /// The mover's finishing stones on a `Win`, sorted: its window with the fewest empties, then the smallest.
    pub finish: Vec<(i32, i32)>,
}

/// Plies before which no side can hold four stones in a window: P2 first holds four after seven stones.
const FIRST_FOUR_PLY: u32 = 7;

/// The one-turn facts of `board` for its side to move (Six's `analyze` less the solver call).
#[must_use]
pub fn analyze(board: &Board) -> LeafTactics {
    if board.ply.index() < FIRST_FOUR_PLY {
        return LeafTactics::default();
    }
    let mover = board.current_player;
    let k = board.moves_remaining;
    let own = board.open_windows(mover, k);
    if let Some(finish) = own
        .iter()
        .map(sorted_empties)
        .min_by(|a, b| (a.len(), a).cmp(&(b.len(), b)))
    {
        return LeafTactics {
            terminal: Some(Terminal::Win),
            finish,
            ..LeafTactics::default()
        };
    }
    let threats = board.open_windows(mover.other(), 2);
    match min_hitting_stones(&threats) {
        Some(0) => LeafTactics::default(),
        Some(n) if n <= k => {
            let mut forced: Vec<(i32, i32)> = threats
                .iter()
                .flat_map(|w| w.empties().iter().copied())
                .collect();
            forced.sort_unstable();
            forced.dedup();
            let fives: Vec<OpenWindow> = threats
                .iter()
                .copied()
                .filter(|w| w.empties().len() == 1)
                .collect();
            LeafTactics {
                forced,
                fives_cover: min_hitting_stones(&fives),
                ..LeafTactics::default()
            }
        }
        _ => LeafTactics {
            terminal: Some(Terminal::Loss),
            ..LeafTactics::default()
        },
    }
}

fn sorted_empties(w: &OpenWindow) -> Vec<(i32, i32)> {
    let mut e = w.empties().to_vec();
    e.sort_unstable();
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Play `seq` from an empty board under the real cadence (1 stone, then 2 per turn).
    fn played(seq: &[(i32, i32)]) -> Board {
        let mut b = Board::new();
        b.set_legal_move_radius(8);
        for &(q, r) in seq {
            b.apply_move(q, r)
                .expect("test sequence places on empty cells");
        }
        b
    }

    /// P1 builds (0..3, 0) while P2 scatters far away; P1 to move with two stones holds an open four.
    const P1_FOUR: [(i32, i32); 9] = [
        (0, 0),
        (20, 20),
        (22, 20),
        (1, 0),
        (2, 0),
        (20, 24),
        (22, 24),
        (3, 0),
        (-9, 9),
    ];

    #[test]
    fn a_mover_that_can_finish_is_a_win_on_its_fewest_empties() {
        // P2's scattered turn hands P1 the move with two stones and the four (0..3, 0).
        let mut seq = P1_FOUR.to_vec();
        seq.extend([(30, 30), (32, 30)]);
        let b = played(&seq);
        assert_eq!(
            (b.current_player, b.moves_remaining),
            (mantis_core::Player::One, 2)
        );
        let t = analyze(&b);
        assert_eq!(t.terminal, Some(Terminal::Win));
        assert_eq!(
            t.finish,
            vec![(-2, 0), (-1, 0)],
            "the smallest of three two-empty windows"
        );
        assert!(t.forced.is_empty());
    }

    #[test]
    fn an_open_four_against_one_stone_is_lost_on_cover() {
        // P2 builds __OOOO__ at (10..13, 5); P1 then places one stone elsewhere and holds one more.
        let b = played(&[
            (0, 0),
            (10, 5),
            (11, 5),
            (1, 0),
            (2, 0),
            (12, 5),
            (13, 5),
            (-5, 0),
        ]);
        assert_eq!(b.moves_remaining, 1);
        assert_eq!(analyze(&b).terminal, Some(Terminal::Loss));
    }

    #[test]
    fn a_blockable_four_forces_every_empty_of_its_windows_in_order() {
        let b = played(&[(0, 0), (10, 5), (11, 5), (1, 0), (2, 0), (12, 5), (13, 5)]);
        assert_eq!(b.moves_remaining, 2);
        let t = analyze(&b);
        assert_eq!(t.terminal, None);
        assert_eq!(t.forced, vec![(8, 5), (9, 5), (14, 5), (15, 5)]);
        assert_eq!(t.fives_cover, Some(0), "an open four has no five");
    }

    #[test]
    fn an_early_board_is_quiet() {
        let b = played(&[(0, 0), (1, 0), (2, 0)]);
        assert_eq!(analyze(&b), LeafTactics::default());
    }
}
