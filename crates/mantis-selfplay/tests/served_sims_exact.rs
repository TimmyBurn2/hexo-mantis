// >300 justify (R8): one invariant's witness across budgets, encodings, kinds and tactics, read as one.
//! A search spends EXACTLY `n_simulations` descents, never more and never fewer.
//! `N` means `N descents` on both arms, root evaluation included: a solver terminal is a descent and
//! network leaves are their own row, so served leaves plus inline descents are the descents.
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

/// What one drive read: producer-served leaves, searched plies, the widest search, inline descents.
struct Drive {
    served: usize,
    records: usize,
    max_sims: u64,
    inline: u64,
}

/// Drive one worker on the GRAPH path until `want_records` searched plies are recorded.
fn drive_graph(
    encoding: &str,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
    tactics: Option<TacticsConfig>,
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
    // With tactics on, compact play reaches decided positions within a short game.
    let spawn = if tactics.is_some() {
        common::spawn_compact_producer
    } else {
        common::spawn_uniform_producer
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
    }
}

/// Drive one worker under `kind` at the run6 identity row, ENCODING held fixed.
fn drive_kind(
    kind: SearchKind,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
    tactics: Option<TacticsConfig>,
) -> Drive {
    const ENCODING: &str = "gnn_axis_r8";
    let spec = lookup_or_panic(ENCODING);
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
        encoding_name: Some(ENCODING.to_string()),
        tactics,
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    // With tactics on, compact play reaches decided positions within a short game.
    let spawn = if tactics.is_some() {
        common::spawn_compact_producer
    } else {
        common::spawn_uniform_producer
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
    let snap = runner.stats_snapshot();
    runner.stop();
    producer.join().expect("producer exits");
    records.extend(runner.drain_graph_records().expect("unpoisoned"));

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
    }
}

fn assert_exact_graph(encoding: &str, n_simulations: usize, ply_cap: usize, want_records: usize) {
    let Drive {
        served,
        records,
        max_sims,
        ..
    } = drive_graph(encoding, n_simulations, ply_cap, want_records, None);
    println!(
        "{encoding} @ {n_simulations}: served {served} leaves over {records} searches, widest \
         search {max_sims}"
    );

    // No search served more than its budget, and at least one spent the whole of it.
    assert_eq!(
        max_sims, n_simulations as u64,
        "{encoding} @ n_simulations={n_simulations}, leaf_batch_size={LEAF_BATCH}: the widest \
         search served {max_sims} leaves against a budget of {n_simulations}. A search must \
         stop at EXACTLY N (R335(c)) — an overshoot makes every `fixed nodes` claim and every \
         g/h derived from `n_simulations` wrong by the same factor, which is what the \
         53.46-vs-50 ledger line recorded; an undershoot means the budget is not being spent."
    );

    // The published `served / records` figure, bounded by one in-flight search — tighter than
    // the pre-clamp readings (r6@50 446, r8@50 443, both @600 1204).
    let expected = records * n_simulations;
    assert!(
        served >= expected && served < expected + n_simulations,
        "{encoding} @ n_simulations={n_simulations}: served {served} leaves over {records} \
         searches = {:.2} sims/move against a configured {n_simulations}; expected \
         [{expected}, {}) — the upper bound allows exactly ONE in-flight search and nothing more.",
        served as f64 / records as f64,
        expected + n_simulations
    );
}

#[test]
fn r6_at_fifty_sims_serves_exactly_fifty_per_search() {
    assert_exact_graph("gnn_axis_v1", 50, 4, 8);
}

/// The tactics-on case: exactly `n` descents a search, served plus inline adding up, and the tactics fired.
fn assert_exact_with_tactics(label: &str, drive: Drive, n_simulations: usize) {
    let Drive {
        served,
        records,
        max_sims,
        inline,
    } = drive;
    println!("{label} @ {n_simulations} tactics on: served {served} + inline {inline} over {records} searches");
    assert_eq!(
        max_sims, n_simulations as u64,
        "{label} @ {n_simulations} tactics on: the widest search spent {max_sims} descents"
    );
    assert!(
        inline > 0,
        "{label}: no descent ended at a decided leaf, so the case proves nothing"
    );
    let descents = served + inline as usize;
    let expected = records * n_simulations;
    assert!(
        descents >= expected && descents < expected + n_simulations,
        "{label} @ {n_simulations} tactics on: {served} served + {inline} inline over {records} \
         searches; expected [{expected}, {}) with one search in flight",
        expected + n_simulations
    );
}

#[test]
fn r8_at_fifty_sims_serves_exactly_fifty_per_search() {
    assert_exact_graph("gnn_axis_r8", 50, 4, 8);
    let drive = drive_graph("gnn_axis_r8", 50, 60, 60, Some(LEAF_TACTICS));
    assert_exact_with_tactics("gnn_axis_r8", drive, 50);
}

#[test]
fn r6_at_six_hundred_sims_serves_exactly_six_hundred_per_search() {
    assert_exact_graph("gnn_axis_v1", 600, 2, 2);
}

#[test]
fn r8_at_six_hundred_sims_serves_exactly_six_hundred_per_search() {
    assert_exact_graph("gnn_axis_r8", 600, 2, 2);
}

// The run6 regime's own budgets on BOTH kinds: a claim taken at 50 and 600 says nothing about
// the numbers a run is actually minted at.

#[test]
fn both_kinds_serve_exactly_sixty_four() {
    for kind in [SearchKind::Puct, SearchKind::Gumbel] {
        let drive = drive_kind(kind, 64, 60, 60, Some(LEAF_TACTICS));
        assert_exact_with_tactics(&format!("{kind:?}"), drive, 64);
        let Drive {
            served,
            records,
            max_sims,
            ..
        } = drive_kind(kind, 64, 3, 4, None);
        println!("{kind:?} @ 64: served {served} over {records} searches, widest {max_sims}");
        assert_eq!(
            max_sims, 64,
            "{kind:?} @ 64: the widest search served {max_sims} leaves. The root's own \
             evaluation is charged on BOTH arms, so N means N leaves of network work and \
             neither an N-1 (an unallocated halving remainder) nor an N+1 (an uncharged \
             root) is admissible."
        );
    }
}

#[test]
fn both_kinds_serve_exactly_three_hundred_and_twenty() {
    for kind in [SearchKind::Puct, SearchKind::Gumbel] {
        let Drive {
            served,
            records,
            max_sims,
            ..
        } = drive_kind(kind, 320, 2, 2, None);
        println!("{kind:?} @ 320: served {served} over {records} searches, widest {max_sims}");
        assert_eq!(
            max_sims, 320,
            "{kind:?} @ 320: the widest search served {max_sims} leaves against the full \
             arm's budget."
        );
    }
}
