// >300 justify (R8): the drive, the replay checker and the pins that read the checker's verdicts are one unit.
//! A decided tail runs its game out with no search, a flagged row a ply; `tail_check` replays each game and reds on a wrong stone.

use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_core::{Board, BoardGeometry, Player};
use mantis_encoding::lookup_or_panic;
use mantis_search::mcts::{AuditConfig, AuditMode, TacticsConfig};
use mantis_search::{MCTSTree, SearchKind};
use mantis_selfplay::replay::hexg::GraphRecord;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig, ARM_TAIL};

mod common;

const ENCODING: &str = "gnn_axis_r8";
const LEAF_BATCH: usize = 8;
const SIMS: usize = 64;
/// A ply cap short enough that some tails meet it.
const CAP: usize = 40;

/// The tactics suite's block at test budgets: leaf, root offence and audit all armed.
const BLOCK: TacticsConfig = TacticsConfig {
    leaf_turns: 2,
    leaf_nodes: 64,
    root_turns: 4,
    root_nodes: 2_000,
    audit: Some(AuditConfig {
        turns: 4,
        nodes: 256,
        k: 4,
        m: 4,
        total_nodes: 4_000,
        mode: AuditMode::Hold,
    }),
};

type Rows = HashMap<&'static str, u64>;

/// One finished game as the runner reported it, its rows in ply order.
#[derive(Clone, Debug)]
struct Game {
    moves: Vec<(i32, i32)>,
    arms: Vec<&'static str>,
    winner: u8,
    rows: Vec<GraphRecord>,
}

/// Where a game's decided tail ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TailEnd {
    NoTail,
    Won,
    Capped,
    Exited,
}

/// One worker under `kind` and playout-cap probability `cap` until `want` holds of the finished games or the deadline.
fn drive(
    decided_tails: bool,
    kind: SearchKind,
    max_moves: usize,
    cap: f32,
    want: fn(&[Game]) -> bool,
) -> (Vec<Game>, Rows) {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: max_moves,
        n_simulations: SIMS,
        n_sims_quick: SIMS / 4,
        full_search_prob: cap,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        search_kind: kind,
        quiescence_enabled: false,
        encoding_name: Some(ENCODING.to_string()),
        tactics: Some(BLOCK),
        decided_tails,
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");
    let producer = common::spawn_compact_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        Arc::new(AtomicUsize::new(0)),
        LEAF_BATCH,
    );
    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let (mut records, mut results) = (Vec::new(), Vec::new());
    let mut games: Vec<Game> = Vec::new();
    // Results before rows: a game's rows are queued before its result, so every drained result has its rows.
    let drain = |records: &mut Vec<GraphRecord>, results: &mut Vec<_>| {
        results.extend(runner.drain_game_results().expect("unpoisoned"));
        records.extend(runner.drain_graph_records().expect("unpoisoned"));
    };
    while Instant::now() < deadline && !want(&games) && runner.fatal_defect().is_none() {
        drain(&mut records, &mut results);
        games = assemble(&records, &results);
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    assert!(defect.is_none(), "a fatal defect latched: {defect:?}");
    drain(&mut records, &mut results);
    games = assemble(&records, &results);
    let rows = runner.tactics_totals().into_iter().collect();
    (games, rows)
}

/// The finished games, a worker's `k`-th game holding the rows stamped `game_id == k`.
fn assemble(
    records: &[GraphRecord],
    results: &[mantis_selfplay::runner::GameResultRow],
) -> Vec<Game> {
    let mut by_game: HashMap<i64, Vec<GraphRecord>> = HashMap::new();
    for r in records {
        by_game.entry(r.game_id).or_default().push(r.clone());
    }
    results
        .iter()
        .enumerate()
        .map(|(k, res)| {
            let mut rows = by_game.remove(&(k as i64)).unwrap_or_default();
            rows.sort_by_key(|r| r.ply_index);
            Game {
                moves: res.2.clone(),
                arms: res.8.iter().map(|&(_, label)| label).collect(),
                winner: res.1,
                rows,
            }
        })
        .collect()
}

fn fresh_board() -> Board {
    let spec = lookup_or_panic(ENCODING);
    Board::with_geometry(BoardGeometry {
        legal_move_radius: spec.legal_move_radius as i32,
        cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
    })
}

/// TailCheck: replay `game` through the tree's offence and resistance, checking each tail ply's stone, row, z and result.
fn tail_check(game: &Game, block: TacticsConfig, max_moves: usize) -> Result<TailEnd, String> {
    if game.rows.len() != game.moves.len() || game.arms.len() != game.moves.len() {
        return Err(format!(
            "{} rows and {} arms for {} plies",
            game.rows.len(),
            game.arms.len(),
            game.moves.len()
        ));
    }
    let (mut board, mut tree) = (fresh_board(), MCTSTree::new(1.5));
    tree.configure_tactics(Some(block));
    let (mut winner, mut owed, mut end): (Option<Player>, Option<(i32, i32)>, TailEnd) =
        (None, None, TailEnd::NoTail);
    let mut tail_rows: Vec<(usize, bool)> = Vec::new();
    for (ply, (&stone, row)) in game.moves.iter().zip(&game.rows).enumerate() {
        if usize::from(row.ply_index) != ply {
            return Err(format!("row {ply} carries ply_index {}", row.ply_index));
        }
        tree.new_game(board.clone());
        let in_tail = match winner {
            None => {
                let decided = tree.root_offence().ok().flatten();
                if decided.is_some() {
                    winner = Some(board.current_player);
                    end = TailEnd::Won;
                }
                decided.map(|d| (d, true))
            }
            Some(w) if board.current_player == w => match tree.root_offence().ok().flatten() {
                Some(d) => Some((d, true)),
                None => {
                    // An exited tail's rows meet no result: its z is checked by no segment.
                    (winner, end) = (None, TailEnd::Exited);
                    tail_rows.clear();
                    None
                }
            },
            Some(_) => {
                let want = if board.moves_remaining == 2 {
                    tree.tail_resistance().map(|[first, second]| {
                        owed = Some(second);
                        first
                    })
                } else {
                    owed.take()
                };
                Some((want.unwrap_or(stone), false))
            }
        };
        match in_tail {
            Some((want, by_winner)) => {
                if stone != want {
                    return Err(format!(
                        "ply {ply}: played {stone:?}, the tail plays {want:?}"
                    ));
                }
                if game.arms[ply] != ARM_TAIL || !row.proven_tail || row.is_full_search {
                    return Err(format!(
                        "ply {ply}: a tail ply recorded as arm {:?}, flag {}, full {}",
                        game.arms[ply], row.proven_tail, row.is_full_search
                    ));
                }
                let one_hot = vec![(stone.0 as i16, stone.1 as i16, 1.0f32)];
                let (visits, value) = if by_winner {
                    (one_hot, 1.0)
                } else {
                    (Vec::new(), -1.0)
                };
                if row.visits != visits
                    || row.tail_mass != 0.0
                    || !row.root_value_valid
                    || row.root_value != value
                {
                    return Err(format!(
                        "ply {ply}: tail row {row:?} is not the {} row",
                        if by_winner { "winner's" } else { "loser's" }
                    ));
                }
                tail_rows.push((ply, by_winner));
            }
            None => {
                if game.arms[ply] == ARM_TAIL || row.proven_tail {
                    return Err(format!("ply {ply}: a tail row outside a tail"));
                }
            }
        }
        board
            .apply_move(stone.0, stone.1)
            .map_err(|_| format!("ply {ply}: {stone:?} is not legal"))?;
    }
    let replayed = match board.winner() {
        Some(Player::One) => 1,
        Some(_) => 2,
        None => 0,
    };
    if replayed != game.winner {
        return Err(format!("replays to {replayed}, recorded {}", game.winner));
    }
    match (end, winner, board.winner()) {
        (TailEnd::Won, Some(w), Some(p)) if p == w => {}
        (TailEnd::Won, _, None) if game.moves.len() >= max_moves => end = TailEnd::Capped,
        (TailEnd::Won, _, _) => {
            return Err(format!("the tail's game ends {replayed}, not its proof"))
        }
        _ => {}
    }
    for (ply, by_winner) in tail_rows {
        let row = &game.rows[ply];
        let want = match end {
            TailEnd::Capped => (0.0, false),
            _ if by_winner => (1.0, true),
            _ => (-1.0, true),
        };
        if end != TailEnd::Exited && (row.outcome, row.value_valid) != want {
            return Err(format!(
                "ply {ply}: z {} valid {} against {want:?}",
                row.outcome, row.value_valid
            ));
        }
    }
    Ok(end)
}

fn ends(games: &[Game], max_moves: usize) -> Vec<TailEnd> {
    games
        .iter()
        .map(|g| tail_check(g, BLOCK, max_moves).unwrap_or_else(|e| panic!("tail_check: {e}")))
        .collect()
}

/// The drained games' tail plies off their arms, flagged rows, loser rows (flagged, no entry), and exits (a tail ply then a searched one).
fn finished(games: &[Game]) -> (u64, u64, u64, u64) {
    let count = |f: &dyn Fn(&Game) -> usize| games.iter().map(f).sum::<usize>() as u64;
    (
        count(&|g| g.arms.iter().filter(|&&a| a == ARM_TAIL).count()),
        count(&|g| g.rows.iter().filter(|r| r.proven_tail).count()),
        count(&|g| {
            g.rows
                .iter()
                .filter(|r| r.proven_tail && r.visits.is_empty())
                .count()
        }),
        count(&|g| {
            g.arms
                .windows(2)
                .filter(|w| w[0] == ARM_TAIL && w[1] != ARM_TAIL)
                .count()
        }),
    )
}

/// Enough winner plies that a drawn arm is both full and quick among them but for 2^-23.
fn many_winner_plies(games: &[Game]) -> bool {
    let (plies, _, loser, _) = finished(games);
    plies.saturating_sub(loser) >= 24
}

#[test]
fn every_tail_game_replays_to_its_proof_with_a_flagged_row_a_ply() {
    for kind in [SearchKind::Puct, SearchKind::Gumbel] {
        let (games, rows) = drive(true, kind, 160, 0.5, |g| {
            g.len() >= 12 && many_winner_plies(g) && ends(g, 160).contains(&TailEnd::Won)
        });
        let seen = ends(&games, 160);
        println!(
            "{kind:?}: {} games, ends {seen:?}, rows {rows:?}",
            games.len()
        );
        assert!(
            seen.contains(&TailEnd::Won),
            "{kind:?}: no tail ran to its six"
        );
        // All four rows count at each finished game, and every finished game is drained here.
        let (plies, flagged, loser, exits) = finished(&games);
        assert_eq!(
            (
                rows["proven_tail_plies"],
                rows["proven_tail_rows"],
                rows["tail_exits"]
            ),
            (plies, flagged, exits),
            "{kind:?}: the rows' counts are the drained games'"
        );
        assert_eq!(plies, flagged, "{kind:?}: a tail ply without its row");
        // PLANTED BREAK: every loser ply skips a search, a winner's only on a full draw; a skip that ignores the arm reds.
        let skipped = rows["tail_search_skipped"];
        assert!(
            loser > 0 && loser < skipped && skipped < plies,
            "{kind:?}: {skipped} skipped of {plies} tail plies, {loser} the loser's"
        );
        assert_eq!(
            (rows["proven_root_rows"], rows["unsearched_decided_rows"]),
            (0, 0),
            "{kind:?}: a decided root searched"
        );
    }
}

#[test]
fn a_tail_the_cap_cuts_replays_to_the_cap_with_every_row_masked() {
    let (games, _) = drive(true, SearchKind::Gumbel, CAP, 0.5, |g| {
        ends(g, CAP).contains(&TailEnd::Capped)
    });
    let seen = ends(&games, CAP);
    println!("cap {CAP}: ends {seen:?}");
    assert!(seen.contains(&TailEnd::Capped), "no tail met the cap");
    for (g, _) in games
        .iter()
        .zip(&seen)
        .filter(|&(_, &e)| e == TailEnd::Capped)
    {
        assert!(
            g.rows
                .iter()
                .all(|r| (r.outcome, r.value_valid) == (0.0, false)),
            "a capped game's row kept a z"
        );
    }
}

/// PLANTED BREAKS: the winner's first tail stone swapped for another legal cell, or its row naming another cell.
#[test]
fn a_planted_wrong_proof_stone_is_refused() {
    let (games, _) = drive(true, SearchKind::Gumbel, 160, 0.5, |g| {
        ends(g, 160).contains(&TailEnd::Won)
    });
    let game = games
        .iter()
        .find(|g| tail_check(g, BLOCK, 160) == Ok(TailEnd::Won))
        .expect("a tail game");
    let t = game
        .arms
        .iter()
        .position(|&a| a == ARM_TAIL)
        .expect("a tail ply");
    let mut board = fresh_board();
    for &(q, r) in &game.moves[..t] {
        board.apply_move(q, r).expect("replays");
    }
    let other = *board
        .legal_moves()
        .iter()
        .find(|&&c| c != game.moves[t])
        .expect("another legal cell");
    let mut swapped = game.clone();
    swapped.moves[t] = other;
    swapped.rows[t].visits = vec![(other.0 as i16, other.1 as i16, 1.0)];
    assert!(
        tail_check(&swapped, BLOCK, 160).is_err(),
        "a non-proof stone passed"
    );
    let mut misnamed = game.clone();
    misnamed.rows[t].visits = vec![(other.0 as i16, other.1 as i16, 1.0)];
    assert!(
        tail_check(&misnamed, BLOCK, 160).is_err(),
        "a row naming an unplayed stone passed"
    );
}

/// The known-bad: tails switched off, the search plays the decided plies and writes their rows without the flag.
#[test]
fn a_tail_the_search_played_reds_the_check() {
    let (games, rows) = drive(false, SearchKind::Gumbel, 160, 0.5, |g| {
        g.iter().any(|g| tail_check(g, BLOCK, 160).is_err())
    });
    let errs: Vec<String> = games
        .iter()
        .filter_map(|g| tail_check(g, BLOCK, 160).err())
        .collect();
    assert!(
        !errs.is_empty()
            && errs
                .iter()
                .all(|e| e.contains("a tail ply recorded as arm")),
        "the searched decided plies red the tail rule, and only it: {errs:?}"
    );
    assert_eq!(
        rows["proven_tail_plies"]
            + rows["proven_tail_rows"]
            + rows["tail_search_skipped"]
            + rows["tail_exits"],
        0
    );
    for g in &games {
        assert_eq!(g.rows.len(), g.moves.len(), "a searched game dropped a row");
    }
}

#[test]
fn every_game_writes_a_row_a_ply_with_tails_on_and_off() {
    for decided_tails in [false, true] {
        let (games, _) = drive(decided_tails, SearchKind::Gumbel, 160, 0.5, |g| {
            g.len() >= 8
        });
        for g in &games {
            assert_eq!(
                g.rows.len(),
                g.moves.len(),
                "tails {decided_tails}: rows against plies"
            );
        }
    }
}

/// With the playout cap unarmed every move would have searched, so every tail ply skips one.
#[test]
fn with_the_cap_unarmed_every_tail_ply_skips_a_search() {
    let (games, rows) = drive(true, SearchKind::Gumbel, 160, 0.0, |g| {
        g.len() >= 6 && ends(g, 160).contains(&TailEnd::Won)
    });
    let (plies, _, _, _) = finished(&games);
    assert!(plies > 0, "{rows:?}");
    assert_eq!(rows["tail_search_skipped"], plies);
}

#[test]
fn boot_refuses_tails_without_an_armed_root_offence() {
    let cfg = |tactics: Option<TacticsConfig>| SelfPlayRunnerConfig {
        n_workers: 1,
        encoding_name: Some(ENCODING.to_string()),
        tactics,
        decided_tails: true,
        ..Default::default()
    };
    for refused in [
        None,
        Some(TacticsConfig {
            root_nodes: 0,
            ..BLOCK
        }),
    ] {
        let err = SelfPlayRunner::new(cfg(refused)).err().expect("refused");
        assert!(err.contains("decided_tails"), "{err}");
    }
    assert!(SelfPlayRunner::new(cfg(Some(BLOCK))).is_ok());
}
