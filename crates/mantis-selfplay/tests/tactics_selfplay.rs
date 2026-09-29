//! A decided root plays its stone after its search, the owed stone follows, the audit vets the rest; rows are summed.

use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::mcts::{AuditConfig, AuditMode, TacticsConfig};
use mantis_search::SearchKind;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

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

/// One worker under `kind` and the block over compact play, until `want` searched plies; the rows and the max search.
fn drive(kind: SearchKind, want: usize) -> (HashMap<&'static str, u64>, u64, usize) {
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
        tactics: Some(BLOCK),
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
    let mut records = 0;
    while Instant::now() < deadline && records < want && runner.fatal_defect().is_none() {
        records += runner.drain_graph_records().expect("unpoisoned").len();
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    assert!(
        defect.is_none(),
        "{kind:?} latched a fatal defect: {defect:?}"
    );
    assert!(records >= want, "{kind:?}: only {records} searched plies");
    let rows = runner.tactics_totals().into_iter().collect();
    (rows, runner.stats_snapshot().max_sims_per_search, records)
}

/// PLANTED BREAK: play the search's move at a decided root and `proof_stones_played` reads 0.
#[test]
fn a_decided_root_is_searched_then_plays_its_stone_and_the_owed_stone_follows() {
    for kind in [SearchKind::Puct, SearchKind::Gumbel] {
        let (rows, max_sims, records) = drive(kind, 200);
        println!("{kind:?} over {records} plies: {rows:?}");
        assert_eq!(
            max_sims, SIMS as u64,
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
        assert_eq!(
            rows["decided_lost_rows"], rows["decided_lost"],
            "{kind:?}: every lost root records no policy target"
        );
        // Every edit needs a veto; under Gumbel a vetoed move can carry no target mass, so not every veto edits.
        let edits = rows["vetoed_target_rows"] + rows["emptied_target_rows"];
        assert!(
            edits <= rows["root_vetoes"] && edits > 0,
            "{kind:?}: {edits} edited targets against {} vetoes",
            rows["root_vetoes"]
        );
        assert!(
            rows["descents"] > 0 && rows["audit_calls"] > 0,
            "{kind:?}: the rows are summed: {rows:?}"
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
    ] {
        assert!(names.contains(want), "the totals name {want}");
    }
}
