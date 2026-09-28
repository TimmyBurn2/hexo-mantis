//! CHECK 1, ported to `mantis_core::Board`: a claimed win replayed against EVERY non-losing defence.
//!
//! Independent of the solver's generator: the defences come from this file's own brute-force window scan, the
//! attacker re-solves at each of its turns with the horizon the claim leaves, and the six is checked on the board.

use std::collections::HashSet;

use mantis_core::board::{Board, Cell, MoveDiff, Player, HEX_AXES};
use mantis_search::tactics::solver::{Turn, TurnSolver, Verdict};

/// What CHECK 1 read on one claim.
#[derive(Debug, PartialEq, Eq)]
pub enum Check {
    Confirmed,
    Refuted(String),
    /// The re-solve ran out of nodes, or the replay passed its cap: neither proven nor refuted.
    Inconclusive(String),
}

fn own(p: Player) -> Cell {
    match p {
        Player::One => Cell::P1,
        Player::Two => Cell::P2,
    }
}

/// The empties of every window holding at least `min` of `p`'s stones and none of the other side's.
pub fn pure_windows(board: &Board, p: Player, min: usize) -> Vec<Vec<(i32, i32)>> {
    let mine = own(p);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (&(sq, sr), &c) in board.cells_iter() {
        if c != mine {
            continue;
        }
        for (axis, &(dq, dr)) in HEX_AXES.iter().enumerate() {
            for k in 0..6 {
                let start = (sq - k * dq, sr - k * dr);
                if !seen.insert((start, axis)) {
                    continue;
                }
                let cells: Vec<(i32, i32)> = (0..6)
                    .map(|i| (start.0 + i * dq, start.1 + i * dr))
                    .collect();
                let stones = cells
                    .iter()
                    .filter(|&&(q, r)| board.get(q, r) == mine)
                    .count();
                let empty: Vec<(i32, i32)> = cells
                    .iter()
                    .copied()
                    .filter(|&(q, r)| board.get(q, r) == Cell::Empty)
                    .collect();
                if stones >= min && stones + empty.len() == 6 {
                    out.push(empty);
                }
            }
        }
    }
    out
}

/// The fewest cells hitting every window (3 meaning more than two), by brute force over their sorted union.
pub fn cover(windows: &[Vec<(i32, i32)>]) -> (u8, Vec<(i32, i32)>) {
    let mut union: Vec<(i32, i32)> = windows.iter().flatten().copied().collect();
    union.sort_unstable();
    union.dedup();
    if windows.is_empty() {
        return (0, union);
    }
    if union.iter().any(|c| windows.iter().all(|w| w.contains(c))) {
        return (1, union);
    }
    for (i, x) in union.iter().enumerate() {
        for y in &union[i + 1..] {
            if windows.iter().all(|w| w.contains(x) || w.contains(y)) {
                return (2, union);
            }
        }
    }
    (3, union)
}

fn undo_all(board: &mut Board, diffs: &mut Vec<MoveDiff>) {
    while let Some(d) = diffs.pop() {
        board.undo_move(d);
    }
}

pub struct StrictCheck<'a> {
    pub solver: &'a mut TurnSolver,
    /// The re-solve's node budget at each attacker turn.
    pub resolve_nodes: u64,
    /// Attacker turns replayed before the check gives up as inconclusive.
    pub cap: usize,
    pub visited: usize,
}

impl StrictCheck<'_> {
    /// Verify that the side to move at `board` wins with `first` within `turns` attacking turns.
    pub fn check(&mut self, board: &mut Board, first: Turn, turns: u8) -> Check {
        self.visited = 0;
        self.attacker_turn(board, Some((first, turns)), turns)
    }

    fn attacker_turn(
        &mut self,
        board: &mut Board,
        given: Option<(Turn, u8)>,
        turns_left: u8,
    ) -> Check {
        self.visited += 1;
        if self.visited > self.cap {
            return Check::Inconclusive(format!("the replay passed {} attacker turns", self.cap));
        }
        let (first, turns) = match given {
            Some(claim) => claim,
            None => {
                self.solver.clear();
                match self
                    .solver
                    .solve(board, turns_left, self.resolve_nodes)
                    .verdict
                {
                    Verdict::Win { first, turns } => (first, turns),
                    Verdict::Unknown { exhausted: true } => {
                        return Check::Inconclusive(format!(
                            "re-solve exhausted at {turns_left} turns"
                        ));
                    }
                    other => {
                        return Check::Refuted(format!(
                            "the claimer finds {other:?} with {turns_left} turns left"
                        ));
                    }
                }
            }
        };
        if turns > turns_left {
            return Check::Refuted(format!(
                "a {turns}-turn line where {turns_left} were claimed"
            ));
        }
        let attacker = board.current_player;
        let mut diffs = Vec::new();
        for &(q, r) in first.stones() {
            if !board.legal_moves_set().contains(&(q, r)) {
                undo_all(board, &mut diffs);
                return Check::Refuted(format!("the attacker's stone ({q}, {r}) is not legal"));
            }
            diffs.push(
                board
                    .apply_move_tracked(q, r)
                    .expect("a legal cell is empty"),
            );
            if board.check_win() {
                let won = board.winner() == Some(attacker);
                undo_all(board, &mut diffs);
                return if won {
                    Check::Confirmed
                } else {
                    Check::Refuted("the six is the defender's".into())
                };
            }
        }
        let verdict = if turns == 0 {
            Check::Refuted("a finish that made no six".into())
        } else {
            self.defender_turn(board, attacker, turns_left)
        };
        undo_all(board, &mut diffs);
        verdict
    }

    /// Every non-losing defence to the attacker's turn just played, each followed by the attacker's next turn.
    fn defender_turn(&mut self, board: &mut Board, attacker: Player, turns_left: u8) -> Check {
        let defender = attacker.other();
        if board.current_player != defender || board.moves_remaining != 2 {
            return Check::Refuted("the attacker's turn did not end on two defender stones".into());
        }
        if pure_windows(board, defender, 4)
            .iter()
            .any(|w| !w.is_empty())
        {
            return Check::Refuted("the defender holds a four and finishes a six".into());
        }
        let threats = pure_windows(board, attacker, 4);
        let (need, union) = cover(&threats);
        match need {
            0 => return Check::Refuted("the attacking turn made no four".into()),
            1 => return Check::Refuted("not strict: one stone blocks every four".into()),
            _ => {}
        }
        let legal = board.legal_moves_set().clone();
        if union.iter().any(|c| !legal.contains(c)) {
            return Check::Refuted("a threat window's empty is not a legal cell".into());
        }
        for (i, &x) in union.iter().enumerate() {
            for &y in &union[i + 1..] {
                if need == 2 && !threats.iter().all(|w| w.contains(&x) || w.contains(&y)) {
                    continue;
                }
                let mut diffs = vec![
                    board
                        .apply_move_tracked(x.0, x.1)
                        .expect("an empty of a window"),
                    board
                        .apply_move_tracked(y.0, y.1)
                        .expect("an empty of a window"),
                ];
                let verdict = if need == 3 {
                    self.six_follows(board, attacker)
                } else if pure_windows(board, defender, 4)
                    .iter()
                    .any(|w| !w.is_empty())
                {
                    Check::Refuted(format!(
                        "the reply ({x:?}, {y:?}) gives the defender a four"
                    ))
                } else if turns_left <= 1 {
                    Check::Refuted("the claim's last turn left a cover-2 set".into())
                } else {
                    self.attacker_turn(board, None, turns_left - 1)
                };
                undo_all(board, &mut diffs);
                if verdict != Check::Confirmed {
                    return verdict;
                }
            }
        }
        Check::Confirmed
    }

    /// After a defence to a cover-3 set some window is untouched: fill it and see the six on the board.
    fn six_follows(&mut self, board: &mut Board, attacker: Player) -> Check {
        let Some(cells) = pure_windows(board, attacker, 4)
            .into_iter()
            .find(|w| !w.is_empty())
        else {
            return Check::Refuted("a defence covered a cover-3 set".into());
        };
        let mut diffs = Vec::new();
        for (q, r) in cells {
            diffs.push(
                board
                    .apply_move_tracked(q, r)
                    .expect("an empty of a window"),
            );
        }
        let won = board.winner() == Some(attacker);
        undo_all(board, &mut diffs);
        if won {
            Check::Confirmed
        } else {
            Check::Refuted("the unblocked window made no six".into())
        }
    }
}
