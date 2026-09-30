//! A decided root plays its stone after its search, the owed stone follows, the audit vets the rest; rows are summed.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_core::{Board, BoardGeometry};
use mantis_encoding::lookup_or_panic;
use mantis_search::mcts::{AuditConfig, AuditMode, TacticsConfig};
use mantis_search::{MCTSTree, SearchKind};
use mantis_selfplay::replay::hexg::GraphRecord;
use mantis_selfplay::runner::{RunnerStatsSnapshot, SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const ENCODING: &str = "gnn_axis_r8";
const LEAF_BATCH: usize = 8;
const SIMS: usize = 64;

/// The block of record's shape at test budgets: leaf, root offence and audit all armed.
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

/// The block with an audit as wide as Gumbel's candidates: its walk reaches cells the search never visited.
const WIDE: TacticsConfig = TacticsConfig {
    audit: Some(AuditConfig {
        k: 16,
        turns: 4,
        nodes: 256,
        m: 4,
        total_nodes: 8_000,
        mode: AuditMode::Hold,
    }),
    ..BLOCK
};

type Rows = HashMap<&'static str, u64>;

/// A drive's summed rows, its stats snapshot and the rows it drained.
struct Drove {
    rows: Rows,
    snap: RunnerStatsSnapshot,
    records: Vec<GraphRecord>,
}

impl Drove {
    /// Drained rows that record no policy target.
    fn no_policy(&self) -> u64 {
        self.records.iter().filter(|r| !r.is_full_search).count() as u64
    }
}

/// One worker under `kind` and `block` over compact play for `want` plies, then until `more` is false or the deadline.
fn drive(kind: SearchKind, block: TacticsConfig, want: usize, more: fn(&Rows) -> bool) -> Drove {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 80,
        n_simulations: SIMS,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        search_kind: kind,
        quiescence_enabled: false,
        encoding_name: Some(ENCODING.to_string()),
        tactics: Some(block),
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
    let mut drained: Vec<GraphRecord> = Vec::new();
    let totals = |r: &SelfPlayRunner| -> Rows { r.tactics_totals().into_iter().collect() };
    while Instant::now() < deadline
        && (drained.len() < want || more(&totals(&runner)))
        && runner.fatal_defect().is_none()
    {
        drained.extend(runner.drain_graph_records().expect("unpoisoned"));
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    drained.extend(runner.drain_graph_records().expect("unpoisoned"));
    assert!(
        defect.is_none(),
        "{kind:?} latched a fatal defect: {defect:?}"
    );
    assert!(
        drained.len() >= want,
        "{kind:?}: only {} searched plies",
        drained.len()
    );
    Drove {
        rows: totals(&runner),
        snap: runner.stats_snapshot(),
        records: drained,
    }
}

/// PLANTED BREAKS: a decided root playing the search's move; a lost row on its drawn arm; vetoes zeroed on a copy.
#[test]
fn a_decided_root_is_searched_then_plays_its_stone_and_the_owed_stone_follows() {
    for kind in [SearchKind::Puct, SearchKind::Gumbel] {
        // PUCT drives on until a zeroed-veto row lands: a short drive can meet none.
        let more: fn(&Rows) -> bool = match kind {
            SearchKind::Puct => |r| r["vetoed_target_rows"] == 0,
            _ => |_| false,
        };
        let drove = drive(kind, BLOCK, 300, more);
        let (rows, no_policy) = (&drove.rows, drove.no_policy());
        println!("{kind:?} over {} plies: {rows:?}", drove.records.len());
        assert_eq!(
            drove.snap.max_sims_per_search, SIMS as u64,
            "{kind:?}: a decided root still spends its budget"
        );
        assert!(
            rows["root_proofs_found"] + rows["finishes_played"] > 0,
            "{kind:?}: no root was decided"
        );
        assert!(
            rows["proof_stones_played"] > 0,
            "{kind:?}: no owed proof stone was played, so the decided first stones were not"
        );
        assert_eq!(
            rows["proven_root_rows"],
            rows["root_proofs_found"] + rows["finishes_played"] + rows["proof_stones_played"],
            "{kind:?}: every decided root records its searched target (every move here is a full search)"
        );
        assert!(
            rows["decided_lost"] > 0,
            "{kind:?}: no root was lost, so the case proves nothing"
        );
        assert_eq!(
            rows["decided_lost_rows"], rows["decided_lost"],
            "{kind:?}: every lost root records no policy target"
        );
        // A game in flight at `stop()` drops its records, so the drained rows can only fall short of the count.
        assert!(
            no_policy <= rows["decided_lost_rows"] + rows["emptied_target_rows"] && no_policy > 0,
            "{kind:?}: {no_policy} no-policy rows drained against the lost and emptied ones: {rows:?}"
        );
        // Every edit needs a veto; under Gumbel a vetoed move can carry no target mass, so not every veto edits.
        let edits =
            rows["vetoed_target_rows"] + rows["emptied_target_rows"] + rows["research_count"];
        assert!(
            edits <= rows["root_vetoes"] && edits > 0,
            "{kind:?}: {edits} edited targets against {} vetoes",
            rows["root_vetoes"]
        );
        if kind == SearchKind::Puct {
            assert!(
                rows["vetoed_target_rows"] > 0,
                "PUCT: no row kept its vetoes at zero: {rows:?}"
            );
        }
        assert!(
            rows["descents"] > 0 && rows["audit_calls"] > 0,
            "{kind:?}: the rows are summed: {rows:?}"
        );
    }
}

/// PLANTED BREAK: store vetoed cells in the sparse support; an unvisited veto overflows `gumbel_m` and latches.
#[test]
fn a_gumbel_audit_wider_than_its_candidates_records_within_the_rows_slots() {
    let drove = drive(SearchKind::Gumbel, WIDE, 300, |r| r["root_vetoes"] < 8);
    let rows = &drove.rows;
    println!(
        "Gumbel wide audit over {} plies: {rows:?}",
        drove.records.len()
    );
    assert!(
        rows["root_vetoes"] >= 8,
        "too few vetoes to reach an unvisited cell: {rows:?}"
    );
}

/// PLANTED BREAK: skip `play_one_move`'s re-search and an all-vetoed root records no policy.
#[test]
fn an_all_vetoed_root_is_re_searched_its_row_records_a_policy_and_both_searches_spend_their_budget()
{
    for (kind, want) in [(SearchKind::Gumbel, 3), (SearchKind::Puct, 1)] {
        let more: fn(&Rows) -> bool = match kind {
            SearchKind::Gumbel => |r| r["research_count"] + r["emptied_target_rows"] < 3,
            _ => |r| r["research_count"] + r["emptied_target_rows"] < 1,
        };
        let drove = drive(kind, BLOCK, 300, more);
        let (rows, snap, no_policy) = (&drove.rows, drove.snap, drove.no_policy());
        println!("{kind:?} over {} plies: {rows:?}", drove.records.len());
        assert!(
            rows["research_count"] >= want,
            "{kind:?}: too few re-searched roots to read: {rows:?}"
        );
        assert_eq!(
            rows["emptied_target_rows"], 0,
            "{kind:?}: an all-vetoed root recorded no policy: {rows:?}"
        );
        assert!(
            no_policy <= rows["decided_lost_rows"],
            "{kind:?}: {no_policy} no-policy rows drained, only lost roots record none: {rows:?}"
        );
        // Each search spends exactly its budget, and the re-search's descents count beside the first's.
        assert_eq!(
            (snap.max_sims_per_search, snap.starved_searches),
            (SIMS as u64, 0)
        );
        let moves = snap.pcr_full_moves;
        let searches = |done: u64| (done + rows["research_count"]) * SIMS as u64;
        assert!(
            (searches(moves - 1)..=searches(moves)).contains(&rows["descents"]),
            "{kind:?}: {} descents over {moves} moves and {} re-searches at {SIMS}",
            rows["descents"],
            rows["research_count"]
        );
    }
}

/// The stones of a row, as a set.
fn stones_of(rec: &GraphRecord) -> HashSet<(i32, i32)> {
    rec.stones
        .iter()
        .map(|&(q, r, _)| (i32::from(q), i32::from(r)))
        .collect()
}

/// Every full-search row the root decision proves, with its proof: each game replayed from its rows, every row asked.
fn proven_rows(records: &[GraphRecord]) -> Vec<(&GraphRecord, Vec<(i32, i32)>)> {
    let spec = lookup_or_panic(ENCODING);
    let geometry = BoardGeometry {
        legal_move_radius: spec.legal_move_radius as i32,
        cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
    };
    let mut games: HashMap<i64, Vec<&GraphRecord>> = HashMap::new();
    for rec in records {
        games.entry(rec.game_id).or_default().push(rec);
    }
    let mut tree = MCTSTree::new(1.5);
    tree.configure_tactics(Some(BLOCK));
    let mut proven = Vec::new();
    for rows in games.values_mut() {
        rows.sort_by_key(|r| r.ply_index);
        let mut board = Board::with_geometry(geometry);
        for (i, rec) in rows.iter().enumerate() {
            assert_eq!(
                usize::from(rec.ply_index),
                i,
                "a game's rows are every ply from its first"
            );
            tree.new_game(board.clone());
            if matches!(tree.root_offence(), Ok(Some(_))) && rec.is_full_search {
                proven.push((*rec, tree.last_root_proof().to_vec()));
            }
            let Some(next) = rows.get(i + 1) else {
                break;
            };
            let placed: Vec<(i32, i32)> = stones_of(next)
                .difference(&stones_of(rec))
                .copied()
                .collect();
            assert_eq!(placed.len(), 1, "one stone a ply");
            board
                .apply_move(placed[0].0, placed[0].1)
                .expect("the game's own move");
        }
    }
    proven
}

/// PLANTED BREAK: skip `play_one_move`'s mixture and a proven row keeps under half its mass on the proof.
#[test]
fn a_proven_root_weak_on_its_proof_records_the_mixture_and_every_proven_row_holds_half_on_it() {
    for kind in [SearchKind::Gumbel, SearchKind::Puct] {
        let drove = drive(kind, BLOCK, 300, |r| r["proven_root_rows"] < 20);
        let rows = &drove.rows;
        println!("{kind:?} over {} plies: {rows:?}", drove.records.len());
        let proven = proven_rows(&drove.records);
        assert!(!proven.is_empty(), "{kind:?}: no drained row was proven");
        for (rec, proof) in &proven {
            let on_proof: f32 = rec
                .visits
                .iter()
                .filter(|v| proof.contains(&(i32::from(v.0), i32::from(v.1))))
                .map(|v| v.2)
                .sum();
            assert!(
                on_proof >= 0.5 - 1e-5,
                "{kind:?}: ply {} stores {on_proof} on its proof {proof:?}",
                rec.ply_index
            );
        }
        assert!(
            rows["mixed_rows"] > 0 && rows["mixed_rows"] <= rows["proven_root_rows"],
            "{kind:?}: the mixture fires on some proven rows, never on another kind: {rows:?}"
        );
    }
}

#[test]
fn a_runner_without_a_block_sums_no_rows() {
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("runner constructs");
    assert!(runner.tactics_totals().iter().all(|&(_, v)| v == 0));
    let names: std::collections::HashSet<&str> =
        runner.tactics_totals().iter().map(|&(n, _)| n).collect();
    for want in [
        "descents",
        "root_proofs_found",
        "proven_root_rows",
        "decided_lost_rows",
        "vetoed_target_rows",
        "research_count",
        "mixed_rows",
    ] {
        assert!(names.contains(want), "the totals name {want}");
    }
}
