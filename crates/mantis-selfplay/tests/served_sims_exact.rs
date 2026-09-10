// >300 justify (R8): one claim under BOTH search kinds, through drives sharing the counting
// producer that IS the measurement; split, the two arms' numbers stop being comparable.
//! A search serves EXACTLY `n_simulations` leaves, never more and never fewer.
//!
//! The finding this exists for: 53.46 served sims/move against `n_simulations: 50`, because the
//! PUCT loop requested a full `leaf_batch_size` while fewer than that remained in the budget.
//! Gumbel had the opposite defect, a phase allocator dropping its integer-division remainder,
//! at 49 of 50 and 599 of 600. `N` now means `N leaves of network work` on both arms, root
//! evaluation included: a fixed-node witness is unstatable while the served count disagrees
//! with the config, and every quantity derived from `n_simulations` is wrong by that factor.
//!
//! The budget arms hold the KIND fixed and vary the radius; the run6-regime arms hold the
//! ENCODING fixed and vary the kind, so that comparison is of searches and nothing else.
//!
//! The primary assertion is `max_sims_per_search`, not the served tally: the tally is an
//! AGGREGATE, and a worker that has begun the next game when `stop()` lands has already served
//! leaves no record accounts for — 401 against an expected 400 after the clamp. The max is
//! exact because it advances with the search it measures.
//!
//! Killer / PLANTED BREAK: revert the PUCT clamp in `search_drive::run_mcts_search` and the
//! PUCT arms red — before the fix they read 56 served against 50 at `leaf_batch_size 8`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::SearchKind;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

const LEAF_BATCH: usize = 8;

fn spawn_counting_producer(
    queue: GraphQueue,
    n_actions: usize,
    served: Arc<AtomicUsize>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(LEAF_BATCH, 5);
        if batch.is_empty() {
            if queue.is_closed() {
                break;
            }
            continue;
        }
        let mut ids = Vec::with_capacity(batch.len());
        let mut results = Vec::with_capacity(batch.len());
        for (id, g) in batch {
            let coords: Vec<(i32, i32)> = g
                .legal_node_gather
                .iter()
                .map(|&row| {
                    (
                        g.node_coords[row as usize * 2],
                        g.node_coords[row as usize * 2 + 1],
                    )
                })
                .collect();
            let n = coords.len();
            let probs = vec![1.0f32 / n.max(1) as f32; n];
            ids.push(id);
            results.push(
                assemble_ls_from_gnn_probs(n_actions, &probs, &g.policy_scatter_index.0, &coords)
                    .map(|ls| (ls, 0.0f32)),
            );
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        queue.submit_graph_results(&ids, results);
    })
}

/// Drive one worker on the GRAPH path until `want_records` searched plies are recorded.
fn drive_graph(
    encoding: &str,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
) -> (usize, usize, u64) {
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
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_counting_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        served.clone(),
    );

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut records = Vec::new();
    while Instant::now() < deadline {
        records.extend(runner.drain_graph_records());
        if records.len() >= want_records || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    let snap = runner.stats_snapshot();
    runner.stop();
    // The producer keeps serving until the queue closes, so both halves are read AFTER the
    // join or the ratio is taken across a moving denominator.
    producer.join().expect("producer exits");
    records.extend(runner.drain_graph_records());

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
    (
        served.load(Ordering::Relaxed),
        records.len(),
        snap.max_sims_per_search,
    )
}

/// Drive one worker under `kind` at the run6 identity row, ENCODING held fixed.
fn drive_kind(
    kind: SearchKind,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
) -> (usize, usize, u64) {
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
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_counting_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        served.clone(),
    );

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut records = Vec::new();
    while Instant::now() < deadline {
        records.extend(runner.drain_graph_records());
        if records.len() >= want_records || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    let snap = runner.stats_snapshot();
    runner.stop();
    producer.join().expect("producer exits");
    records.extend(runner.drain_graph_records());

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
    (
        served.load(Ordering::Relaxed),
        records.len(),
        snap.max_sims_per_search,
    )
}

fn assert_exact_graph(encoding: &str, n_simulations: usize, ply_cap: usize, want_records: usize) {
    let (served, records, max_sims) = drive_graph(encoding, n_simulations, ply_cap, want_records);
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

#[test]
fn r8_at_fifty_sims_serves_exactly_fifty_per_search() {
    assert_exact_graph("gnn_axis_r8", 50, 4, 8);
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
        let (served, records, max_sims) = drive_kind(kind, 64, 3, 4);
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
        let (served, records, max_sims) = drive_kind(kind, 320, 2, 2);
        println!("{kind:?} @ 320: served {served} over {records} searches, widest {max_sims}");
        assert_eq!(
            max_sims, 320,
            "{kind:?} @ 320: the widest search served {max_sims} leaves against the full \
             arm's budget."
        );
    }
}
