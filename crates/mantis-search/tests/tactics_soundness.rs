//! Soundness: every win `TurnSolver` claims survives every non-losing defence (CHECK 1, not the generator's).
//! PLANTED BREAK: `gen::covering_pairs` cut to its first pair reds both CHECK 1 tests (untried defender replies).

mod tactics_common;

use mantis_core::board::zobrist::splitmix64_next;
use mantis_core::{Board, Cell};
use mantis_search::tactics::analyze::analyze;
use mantis_search::tactics::solver::{TurnSolver, Verdict};
use tactics_common::golden_rows;
use tactics_common::strict_check::{cover, pure_windows, Check, StrictCheck};

const TABLE: usize = 1 << 18;

#[derive(Default, Debug)]
struct Tally {
    claims: usize,
    confirmed: usize,
    inconclusive: usize,
    refuted: Vec<String>,
}

fn verify_all(positions: &[(String, Board)], configs: &[(u8, u64)]) -> Tally {
    let mut solver = TurnSolver::new(TABLE);
    let mut checker_solver = TurnSolver::new(TABLE);
    let mut tally = Tally::default();
    for (label, board) in positions {
        // A claim another budget already made here replays identically: checked once.
        let mut checked: Vec<(Vec<(i32, i32)>, u8)> = Vec::new();
        for &(t, n) in configs {
            solver.clear();
            let Verdict::Win { first, turns } = solver.solve(board, t, n).verdict else {
                continue;
            };
            let claim = (first.stones().to_vec(), turns);
            if turns == 0 || checked.contains(&claim) {
                continue;
            }
            checked.push(claim);
            tally.claims += 1;
            let mut check = StrictCheck {
                solver: &mut checker_solver,
                resolve_nodes: 60_000,
                cap: 20_000,
                visited: 0,
            };
            match check.check(&mut board.clone(), first, turns) {
                Check::Confirmed => tally.confirmed += 1,
                Check::Inconclusive(_) => tally.inconclusive += 1,
                Check::Refuted(why) => tally
                    .refuted
                    .push(format!("{label} {t}/{n} {turns}-turn: {why}")),
            }
        }
    }
    tally
}

#[test]
fn every_win_the_port_claims_on_the_goldens_survives_every_defence() {
    let positions: Vec<(String, Board)> = golden_rows()
        .into_iter()
        .filter(|r| r.set != "*")
        .map(|r| (r.pid, r.board))
        .collect();
    let tally = verify_all(&positions, &[(2, 64), (8, 2000), (8, 20_000)]);
    println!("goldens CHECK 1: {tally:?}");
    assert!(
        tally.refuted.is_empty(),
        "REFUTED win claims:\n{}",
        tally.refuted.join("\n")
    );
    assert!(
        tally.claims > 200,
        "only {} distinct claims: a vacuous check",
        tally.claims
    );
    assert!(
        tally.inconclusive * 100 <= tally.claims,
        "{} of {} claims inconclusive (> 1 %)",
        tally.inconclusive,
        tally.claims
    );
}

/// Clustered random play at radius 8 to a random stone count in 8..=24, kept when quiet with two stones to move.
fn quiet_positions(seed: u64, want: usize) -> Vec<(String, Board)> {
    let mut state = seed;
    let mut out = Vec::new();
    let mut game = 0;
    while out.len() < want {
        game += 1;
        let target = 8 + (splitmix64_next(&mut state) % 17) as usize;
        let mut b = Board::new();
        b.set_legal_move_radius(8);
        let mut placed: Vec<(i32, i32)> = Vec::new();
        let mut tries = 0;
        while placed.len() < target && tries < 2000 {
            tries += 1;
            let (q, r) = if placed.is_empty() {
                (0, 0)
            } else {
                let tail = &placed[placed.len().saturating_sub(5)..];
                let (aq, ar) = tail[(splitmix64_next(&mut state) as usize) % tail.len()];
                let dq = (splitmix64_next(&mut state) % 5) as i32 - 2;
                let dr = (splitmix64_next(&mut state) % 5) as i32 - 2;
                (aq + dq, ar + dr)
            };
            if b.get(q, r) != Cell::Empty || b.apply_move(q, r).is_err() {
                continue;
            }
            if b.check_win() {
                break;
            }
            placed.push((q, r));
        }
        let facts = analyze(&b);
        if !b.check_win()
            && b.moves_remaining == 2
            && facts.terminal.is_none()
            && facts.forced.is_empty()
        {
            out.push((format!("seed{seed:x}-game{game}"), b));
        }
    }
    out
}

#[test]
fn random_quiet_positions_hold_no_refuted_win() {
    let positions = quiet_positions(0x07ac_71c5, 1500);
    let tally = verify_all(&positions, &[(4, 2000)]);
    println!("random CHECK 1: {tally:?}");
    assert!(
        tally.refuted.is_empty(),
        "REFUTED win claims:\n{}",
        tally.refuted.join("\n")
    );
    assert!(
        tally.claims >= 50,
        "only {} claims on random positions: a vacuous fuzz",
        tally.claims
    );
    assert_eq!(
        tally.inconclusive, 0,
        "a 4-turn / 2000-node claim should always replay"
    );
}

/// Whether some two-stone turn of the mover leaves threats that two blockers cannot cover, by brute force over
/// every pair of empties of the mover's pure windows holding two or three stones (no other cell adds a four).
fn cover_three_pair_exists(board: &Board) -> bool {
    let mover = board.current_player;
    let mut cells: Vec<(i32, i32)> = pure_windows(board, mover, 2)
        .into_iter()
        .flatten()
        .collect();
    cells.sort_unstable();
    cells.dedup();
    let mut b = board.clone();
    for (i, &x) in cells.iter().enumerate() {
        for &y in &cells[i + 1..] {
            let dx = b.apply_move_tracked(x.0, x.1).expect("empty");
            let dy = b.apply_move_tracked(y.0, y.1).expect("empty");
            let (need, _) = cover(&pure_windows(&b, mover, 4));
            b.undo_move(dy);
            b.undo_move(dx);
            if need == 3 {
                return true;
            }
        }
    }
    false
}

#[test]
fn at_one_turn_the_solver_misses_no_cover_three_pair() {
    let mut solver = TurnSolver::new(TABLE);
    let (mut with_pair, mut checked) = (0usize, 0usize);
    for (label, board) in quiet_positions(0xde71, 400) {
        let threes: usize = {
            let mut c: Vec<(i32, i32)> = pure_windows(&board, board.current_player, 3)
                .into_iter()
                .filter(|w| w.len() == 3)
                .flatten()
                .collect();
            c.sort_unstable();
            c.dedup();
            c.len()
        };
        if threes > 48 {
            continue;
        }
        checked += 1;
        if !cover_three_pair_exists(&board) {
            continue;
        }
        with_pair += 1;
        solver.clear();
        let got = solver.solve(&board, 1, 1_000_000).verdict;
        assert!(
            matches!(got, Verdict::Win { turns: 1, .. }),
            "{label}: a cover-3 pair exists but the one-turn solve reads {got:?}"
        );
    }
    println!("depth-1 completeness: {with_pair} of {checked} positions hold a cover-3 pair");
    assert!(
        with_pair >= 20,
        "only {with_pair} positions with a cover-3 pair: a vacuous check"
    );
}

/// P, a strict win; Q reflects its twelfth stone (9, -18) to (-9, 18), a pair core's out-of-table keys conflate.
const P_REFLECTED: [(i32, i32); 23] = [
    (6, -12),
    (7, -12),
    (9, -14),
    (4, -11),
    (6, -13),
    (8, -10),
    (8, -16),
    (10, -12),
    (8, -15),
    (10, -15),
    (9, -12),
    (9, -18),
    (7, -17),
    (7, -18),
    (8, -17),
    (6, -16),
    (5, -17),
    (5, -18),
    (6, -15),
    (9, -17),
    (4, -20),
    (4, -17),
    (6, -17),
];

#[test]
fn a_root_with_a_four_or_a_small_radius_is_refused_not_searched() {
    // P1 to move with two stones and its own four (0..3, 0); then P1 against P2's open four at (10..13, 5).
    let own_four = [
        (0, 0),
        (20, 20),
        (22, 20),
        (1, 0),
        (2, 0),
        (20, 24),
        (22, 24),
        (3, 0),
        (-9, 9),
        (30, 30),
        (32, 30),
    ];
    let their_four = [(0, 0), (10, 5), (11, 5), (1, 0), (2, 0), (12, 5), (13, 5)];
    let mut solver = TurnSolver::new(TABLE);
    for seq in [&own_four[..], &their_four[..]] {
        let board = tactics_common::replay(seq, 8);
        let got = solver.search_quiet(&board, 2, 2000);
        assert_eq!(
            got.verdict,
            Verdict::Unknown { exhausted: false },
            "{seq:?}: a four is no quiet root"
        );
    }
    let mut narrow = tactics_common::replay(&P_REFLECTED, 8);
    narrow.set_legal_move_radius(4);
    assert_eq!(
        solver.solve(&narrow, 4, 2000).verdict,
        Verdict::Unknown { exhausted: false }
    );
    assert_eq!(
        solver.search_quiet(&narrow, 4, 2000).verdict,
        Verdict::Unknown { exhausted: false }
    );
}

#[test]
fn a_warm_table_never_carries_a_win_across_a_reflected_stone() {
    let mut reflected = P_REFLECTED;
    reflected[11] = (-9, 18);
    let (p, q) = (
        tactics_common::replay(&P_REFLECTED, 8),
        tactics_common::replay(&reflected, 8),
    );
    let mut solver = TurnSolver::new(TABLE);
    assert!(
        matches!(solver.solve(&p, 4, 2000).verdict, Verdict::Win { .. }),
        "P is a strict win"
    );
    let warm = solver.solve(&q, 4, 2000);
    solver.clear();
    let cold = solver.solve(&q, 4, 2000);
    assert_eq!(
        warm, cold,
        "Q's verdict must not depend on P having been solved first"
    );
}
