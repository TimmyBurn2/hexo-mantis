// >300 justify (R8): one invariant's witness across budgets, encodings, kinds and tactics, read as one.
//! A search spends EXACTLY `n_simulations` descents, never more and never fewer, and none ends short.
//! `N` means `N descents` on both arms, root evaluation included: a descent that backs up a value is one,
//! whatever backed it, so served leaves, inline descents (a terminal or the solver) and table hits are the descents.
//!
//! The budget arms hold the KIND fixed and vary the radius; the run6-regime arms hold the
//! ENCODING fixed and vary the kind, so that comparison is of searches and nothing else.
//!
//! The primary assertion is `max_sims_per_search`, not the served tally: the tally is an
//! AGGREGATE, and a worker that has begun the next game when `stop()` lands has already served
//! leaves no record accounts for. The max is exact because it advances with the search it measures.
//!
//! Killer / PLANTED BREAK: revert the PUCT clamp in `search_drive::run_mcts_search` and the
//! PUCT arms red — before the fix they read 56 served against 50 at `leaf_batch_size 8`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::mcts::TacticsConfig;
use mantis_search::SearchKind;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

mod common;

const LEAF_BATCH: usize = 8;

/// The design's leaf budgets, the wiring the self-play runner shares with the deploy head.
const LEAF_TACTICS: TacticsConfig = TacticsConfig {
    leaf_turns: 2,
    leaf_nodes: 64,
    root_turns: 8,
    root_nodes: 20_000,
    audit: None,
};

/// The mock server a drive plays against.
#[derive(Clone, Copy)]
enum Producer {
    /// A uniform prior: wide, short searches that rarely transpose.
    Uniform,
    /// A compact prior: play reaches decided positions within a short game, and two-stone turns transpose.
    Compact,
}

/// What one drive read: served leaves, searched plies, the widest search, inline, table and starved rows.
#[derive(Clone, Copy)]
struct Drive {
    served: usize,
    records: usize,
    max_sims: u64,
    inline: u64,
    table: u64,
    starved: u64,
    shortfall: u64,
}

/// Drive one worker on the GRAPH path until `want_records` searched plies are recorded.
fn drive_graph(
    encoding: &str,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
    tactics: Option<TacticsConfig>,
    producer: Producer,
) -> Drive {
    let spec = lookup_or_panic(encoding);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: ply_cap,
        n_simulations,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        dirichlet_enabled: true,
        search_kind: SearchKind::Puct,
        encoding_name: Some(encoding.to_string()),
        tactics,
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    let spawn = match producer {
        Producer::Uniform => common::spawn_uniform_producer,
        Producer::Compact => common::spawn_compact_producer,
    };
    let producer = spawn(
        runner.graph_producer(),
        spec.policy_logit_count,
        served.clone(),
        LEAF_BATCH,
    );

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut records = Vec::new();
    while Instant::now() < deadline {
        records.extend(runner.drain_graph_records().expect("unpoisoned"));
        if records.len() >= want_records || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    // The producer keeps serving until the queue closes, so every count is read AFTER the
    // join or the ratio is taken across a moving denominator.
    producer.join().expect("producer exits");
    records.extend(runner.drain_graph_records().expect("unpoisoned"));
    let snap = runner.stats_snapshot();

    assert!(
        defect.is_none(),
        "{encoding} @ {n_simulations} latched a fatal defect: {defect:?}"
    );
    assert!(
        records.len() >= want_records,
        "{encoding} @ {n_simulations}: only {} searched plies inside the budget — a drive \
         that records nothing cannot speak about served sims at all",
        records.len()
    );
    Drive {
        served: served.load(Ordering::Relaxed),
        records: records.len(),
        max_sims: snap.max_sims_per_search,
        inline: snap.inline_descents_total,
        table: snap.tt_hits_total,
        starved: snap.starved_searches,
        shortfall: snap.starved_descents,
    }
}

/// Drive one worker under `kind` at the run6 identity row, ENCODING held fixed.
fn drive_kind(
    kind: SearchKind,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
    tactics: Option<TacticsConfig>,
    producer: Producer,
) -> Drive {
    drive_kind_at(
        "gnn_axis_r8",
        kind,
        n_simulations,
        ply_cap,
        want_records,
        tactics,
        producer,
    )
}

/// [`drive_kind`] at the registry row `encoding`.
fn drive_kind_at(
    encoding: &str,
    kind: SearchKind,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
    tactics: Option<TacticsConfig>,
    producer: Producer,
) -> Drive {
    let spec = lookup_or_panic(encoding);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: ply_cap,
        n_simulations,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        // Dirichlet is a PUCT mechanism, armed to keep that arm honest and inert under
        // Gumbel by construction.
        dirichlet_enabled: true,
        search_kind: kind,
        quiescence_enabled: false,
        encoding_name: Some(encoding.to_string()),
        tactics,
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    let spawn = match producer {
        Producer::Uniform => common::spawn_uniform_producer,
        Producer::Compact => common::spawn_compact_producer,
    };
    let producer = spawn(
        runner.graph_producer(),
        spec.policy_logit_count,
        served.clone(),
        LEAF_BATCH,
    );

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut records = Vec::new();
    while Instant::now() < deadline {
        records.extend(runner.drain_graph_records().expect("unpoisoned"));
        if records.len() >= want_records || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    records.extend(runner.drain_graph_records().expect("unpoisoned"));
    let snap = runner.stats_snapshot();

    assert!(
        defect.is_none(),
        "{kind:?} @ {n_simulations} latched a fatal defect: {defect:?}"
    );
    assert!(
        records.len() >= want_records,
        "{kind:?} @ {n_simulations}: only {} searched plies inside the budget — a drive that \
         records nothing cannot speak about served sims at all",
        records.len()
    );
    Drive {
        served: served.load(Ordering::Relaxed),
        records: records.len(),
        max_sims: snap.max_sims_per_search,
        inline: snap.inline_descents_total,
        table: snap.tt_hits_total,
        starved: snap.starved_searches,
        shortfall: snap.starved_descents,
    }
}

/// `n` descents a search: the widest spent `n`, none ended short, served + inline + table is `records × n` (+1 in flight).
fn assert_exact(label: &str, drive: &Drive, n_simulations: usize) {
    let Drive {
        served,
        records,
        max_sims,
        inline,
        table,
        starved,
        shortfall,
    } = *drive;
    println!(
        "{label} @ {n_simulations}: served {served} + inline {inline} + table {table} over {records} searches, \
         widest {max_sims}"
    );
    assert_eq!(
        max_sims, n_simulations as u64,
        "{label} @ n_simulations={n_simulations}, leaf_batch_size={LEAF_BATCH}: the widest search spent \
         {max_sims} descents. A search must stop at EXACTLY N (R335(c)): an overshoot makes every `fixed nodes` \
         claim and every g/h derived from `n_simulations` wrong by the same factor, which is what the \
         53.46-vs-50 ledger line recorded; an undershoot means the budget is not being spent."
    );
    assert_eq!(
        (starved, shortfall),
        (0, 0),
        "{label} @ {n_simulations}: {starved} searches ended short by {shortfall} descents"
    );
    let descents = served + inline as usize + table as usize;
    let expected = records * n_simulations;
    assert!(
        descents >= expected && descents < expected + n_simulations,
        "{label} @ {n_simulations}: {served} served + {inline} inline + {table} table over {records} searches = \
         {:.2} descents a search; expected [{expected}, {}) with at most one search in flight",
        descents as f64 / records as f64,
        expected + n_simulations
    );
}

#[test]
fn r6_at_fifty_sims_serves_exactly_fifty_per_search() {
    let drive = drive_graph("gnn_axis_v1", 50, 4, 8, None, Producer::Uniform);
    assert_exact("gnn_axis_v1", &drive, 50);
}

#[test]
fn r8_at_fifty_sims_serves_exactly_fifty_per_search() {
    let drive = drive_graph("gnn_axis_r8", 50, 4, 8, None, Producer::Uniform);
    assert_exact("gnn_axis_r8", &drive, 50);
    let drive = drive_graph(
        "gnn_axis_r8",
        50,
        60,
        60,
        Some(LEAF_TACTICS),
        Producer::Compact,
    );
    assert_exact("gnn_axis_r8 tactics on", &drive, 50);
    assert!(
        drive.inline > 0,
        "no descent ended at a decided leaf, so the solver case proves nothing"
    );
}

#[test]
fn r6_at_six_hundred_sims_serves_exactly_six_hundred_per_search() {
    let drive = drive_graph("gnn_axis_v1", 600, 2, 2, None, Producer::Uniform);
    assert_exact("gnn_axis_v1", &drive, 600);
}

#[test]
fn r8_at_six_hundred_sims_serves_exactly_six_hundred_per_search() {
    let drive = drive_graph("gnn_axis_r8", 600, 2, 2, None, Producer::Uniform);
    assert_exact("gnn_axis_r8", &drive, 600);
}

// The run6 regime's own budgets on BOTH kinds: a claim taken at 50 and 600 says nothing about
// the numbers a run is actually minted at.

/// The net, the solver and the table each end a counted descent. PLANTED BREAK: drop `select_leaves`' table `i += 1`.
#[test]
fn both_kinds_serve_exactly_sixty_four() {
    for kind in [SearchKind::Puct, SearchKind::Gumbel] {
        let drive = drive_kind(kind, 64, 60, 60, Some(LEAF_TACTICS), Producer::Compact);
        assert_exact(&format!("{kind:?} tactics on"), &drive, 64);
        assert!(
            drive.inline > 0,
            "{kind:?}: no descent ended at a decided leaf, so the solver case proves nothing"
        );
        let drive = drive_kind(kind, 64, 3, 4, None, Producer::Uniform);
        assert_exact(&format!("{kind:?}"), &drive, 64);
        let drive = drive_kind(kind, 64, 40, 24, None, Producer::Compact);
        assert_exact(&format!("{kind:?} transposing"), &drive, 64);
        // Gumbel's forced descents have no table path; PUCT's two-stone turns transpose.
        assert_eq!(
            drive.table > 0,
            kind == SearchKind::Puct,
            "{kind:?}: {} table hits over a transposing drive",
            drive.table
        );
    }
}

#[test]
fn both_kinds_serve_exactly_three_hundred_and_twenty() {
    for kind in [SearchKind::Puct, SearchKind::Gumbel] {
        let drive = drive_kind(kind, 320, 2, 2, None, Producer::Uniform);
        assert_exact(&format!("{kind:?}"), &drive, 320);
    }
}

/// The pruned row serves exactly its budget on both kinds, with the net, the solver and the table each ending a descent.
#[test]
fn the_pruned_row_serves_exactly_its_budget_on_both_kinds() {
    const PRUNED: &str = "gnn_axis_r8_pruned";
    let drive = drive_graph(PRUNED, 50, 4, 8, None, Producer::Uniform);
    assert_exact(PRUNED, &drive, 50);
    for kind in [SearchKind::Puct, SearchKind::Gumbel] {
        let drive = drive_kind_at(
            PRUNED,
            kind,
            64,
            60,
            60,
            Some(LEAF_TACTICS),
            Producer::Compact,
        );
        assert_exact(&format!("{PRUNED} {kind:?} tactics on"), &drive, 64);
        assert!(
            drive.inline > 0,
            "{kind:?}: no descent ended at a decided leaf"
        );
        let drive = drive_kind_at(PRUNED, kind, 64, 40, 24, None, Producer::Compact);
        assert_exact(&format!("{PRUNED} {kind:?} transposing"), &drive, 64);
        assert_eq!(
            drive.table > 0,
            kind == SearchKind::Puct,
            "{kind:?}: {} table hits",
            drive.table
        );
    }
}
