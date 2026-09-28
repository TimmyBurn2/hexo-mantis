//! The port against Six: `TurnSolver` reproduces Six's `ThreatSolver` verdicts on TACTICS-DESIGN's T2 positions.
//!
//! The rule (G1–G4) is `mantis-records/tactics-deploy/drivers/RULES_GOLDENS.md`, hashed before the first
//! comparison: exact where Six finished well inside its budget, within a band where the budget binds.

mod tactics_common;

use mantis_search::tactics::solver::{Solved, TurnSolver, Verdict};
use tactics_common::golden_rows;

/// The design's table size: 2^18 slots, as Six's 8 MiB table.
const TABLE: usize = 1 << 18;

#[test]
fn the_port_reproduces_six_on_the_goldens() {
    let rows = golden_rows();
    assert!(rows.len() > 500, "the fixture holds {} rows", rows.len());
    let mut solver = TurnSolver::new(TABLE);
    let mut g1_misses = Vec::new();
    let (mut exact, mut exact_nodes_equal, mut band_misses) = (0usize, 0usize, Vec::new());
    let (mut bound, mut bound_agree) = (0usize, 0usize);
    let mut g4_misses = Vec::new();
    for row in &rows {
        for &((t, n), ref six) in &row.six {
            solver.clear();
            let got = solver.solve(&row.board, t, n);
            if row.set == "*" {
                let ok = matches!(
                    (row.cls.as_str(), got.verdict),
                    ("W1", Verdict::Win { turns: 0, .. })
                        | ("LOST", Verdict::Loss)
                        | ("BLOCK", Verdict::Unknown { exhausted: false })
                );
                if !ok {
                    g4_misses.push(format!(
                        "{} {} {t}/{n}: {:?}",
                        row.pid, row.cls, got.verdict
                    ));
                }
                continue;
            }
            let (found, turns, first) = match got.verdict {
                Verdict::Win { first, turns } => {
                    (true, turns, Some((first.stones()[0], first.stones()[1])))
                }
                _ => (false, 0, None),
            };
            if !six.exhausted && six.nodes <= n / 2 {
                exact += 1;
                if (found, turns, first) != (six.found, six.turns, six.first) {
                    g1_misses.push(format!(
                        "{} [{} {}] {t}/{n}: ours ({found}, {turns}, {first:?}, {} nodes) six ({}, {}, {:?}, {} nodes)",
                        row.pid, row.set, row.class, got.nodes, six.found, six.turns, six.first, six.nodes
                    ));
                }
                if got.nodes == six.nodes {
                    exact_nodes_equal += 1;
                } else if got.nodes.abs_diff(six.nodes) > 2.max(six.nodes / 10) {
                    band_misses.push(format!(
                        "{} {t}/{n}: nodes ours {} six {}",
                        row.pid, got.nodes, six.nodes
                    ));
                }
            } else {
                bound += 1;
                if found == six.found {
                    bound_agree += 1;
                }
            }
        }
    }
    println!(
        "goldens: {} rows; exact stratum {exact} solves, nodes equal {exact_nodes_equal}; budget-bound {bound}, \
         agreeing {bound_agree}",
        rows.len()
    );
    assert!(
        g4_misses.is_empty(),
        "G4 analyze-class misses:\n{}",
        g4_misses.join("\n")
    );
    assert!(
        g1_misses.is_empty(),
        "G1 misses ({}):\n{}",
        g1_misses.len(),
        g1_misses.join("\n")
    );
    assert!(
        band_misses.is_empty(),
        "G2 node-band misses:\n{}",
        band_misses.join("\n")
    );
    assert!(
        exact_nodes_equal * 100 >= exact * 95,
        "G2: nodes equal on {exact_nodes_equal} of {exact} exact-stratum solves (< 95 %)"
    );
    assert!(
        bound_agree * 100 >= bound * 80,
        "G3: {bound_agree} of {bound} budget-bound solves agree (< 80 %)"
    );
}

fn solve_all(clear_each: bool) -> Vec<Solved> {
    let rows = golden_rows();
    let mut solver = TurnSolver::new(TABLE);
    let mut out = Vec::new();
    for row in &rows {
        for &((t, n), _) in &row.six {
            if clear_each {
                solver.clear();
            }
            out.push(solver.solve(&row.board, t, n));
        }
    }
    out
}

#[test]
fn two_passes_with_the_table_cleared_are_identical_nodes_included() {
    assert!(solve_all(true) == solve_all(true));
}

#[test]
fn two_passes_through_a_warm_table_are_identical_too() {
    assert!(solve_all(false) == solve_all(false));
}

#[test]
fn a_wins_line_opens_with_its_first_turn() {
    let mut solver = TurnSolver::new(TABLE);
    let mut lines = 0;
    for row in golden_rows().iter().filter(|r| r.set != "*") {
        solver.clear();
        if let Verdict::Win { first, .. } = solver.solve(&row.board, 8, 2000).verdict {
            let line = solver.line(&row.board);
            assert!(line.len() >= 2, "{}: {line:?}", row.pid);
            assert_eq!(
                &line[..2],
                first.stones(),
                "{}: the line's opening turn",
                row.pid
            );
            lines += 1;
        }
    }
    assert!(lines > 100, "only {lines} lines read");
}
