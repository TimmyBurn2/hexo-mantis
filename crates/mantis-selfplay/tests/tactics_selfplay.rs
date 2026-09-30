//! A decided root plays its stone after its search, the owed stone follows, the audit vets the rest; rows are summed.

use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::mcts::{AuditConfig, AuditMode, TacticsConfig};
use mantis_search::SearchKind;
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

/// One worker under `kind` and `block` over compact play for `want` plies, then until `more` is false or the deadline.
fn drive(
    kind: SearchKind,
    block: TacticsConfig,
    want: usize,
    more: fn(&Rows) -> bool,
) -> (Rows, RunnerStatsSnapshot, usize, u64) {
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
    let (records, no_policy) = (
        drained.len(),
        drained.iter().filter(|r| !r.is_full_search).count() as u64,
    );
    assert!(
        defect.is_none(),
        "{kind:?} latched a fatal defect: {defect:?}"
    );
    assert!(records >= want, "{kind:?}: only {records} searched plies");
    let rows = totals(&runner);
    (rows, runner.stats_snapshot(), records, no_policy)
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
        let (rows, snap, records, no_policy) = drive(kind, BLOCK, 300, more);
        println!("{kind:?} over {records} plies: {rows:?}");
        assert_eq!(
            snap.max_sims_per_search, SIMS as u64,
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
    let (rows, _, records, _) = drive(SearchKind::Gumbel, WIDE, 300, |r| r["root_vetoes"] < 8);
    println!("Gumbel wide audit over {records} plies: {rows:?}");
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
        let (rows, snap, records, no_policy) = drive(kind, BLOCK, 300, more);
        println!("{kind:?} over {records} plies: {rows:?}");
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
    ] {
        assert!(names.contains(want), "the totals name {want}");
    }
}
