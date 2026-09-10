//! A Gumbel halving round is ONE inference round trip.
//!
//! Sequential Halving visits every candidate alive at the current considered visit level before
//! that level advances, so a ROUND is `m` descents, then `m/2`, and so on; issuing them together
//! needs no virtual loss because the candidates' subtrees are disjoint. The lever is invisible in
//! outputs — a search back at one leaf per round trip yields the same visit counts and the same
//! move, only N times slower — so the width is counted in-run and this file reads the counter.
//!
//! The criterion is one-sided: leaves per round trip >= the schedule's own mean at the budget the
//! SEARCH gets, which is `N_SIMS - 1` because the root's own evaluation is charged. The
//! schedule's round-width profile is DISCONTINUOUS at its budget, so both means are DERIVED by
//! `schedule_mean` and printed rather than transcribed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::mcts::seq_halving::considered_visits_sequence;
use mantis_search::SearchKind;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// The run6 full-arm regime the ruling names.
const N_SIMS: usize = 320;
const GUMBEL_M: usize = 16;
const ENCODING: &str = "gnn_axis_r8";

/// A uniform-prior mock inference server on the graph queue. The POLICY is irrelevant to a
/// round-width count; what matters is that every requested leaf is answered promptly.
fn spawn_producer(queue: GraphQueue, n_actions: usize, served: Arc<AtomicUsize>) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(8, 5);
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

/// `(round_leaves, rounds, max_sims_per_search)` from one driven worker.
fn drive(kind: SearchKind, want_records: usize) -> (u64, u64, u64) {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 2,
        n_simulations: N_SIMS,
        gumbel_m: GUMBEL_M,
        leaf_batch_size: 8,
        random_opening_plies: 0,
        dirichlet_enabled: true,
        search_kind: kind,
        quiescence_enabled: false,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        served.clone(),
    );

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut records = 0usize;
    while Instant::now() < deadline {
        records += runner.drain_graph_records().len();
        if records >= want_records || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    let snap = runner.stats_snapshot();

    assert!(
        defect.is_none(),
        "{kind:?}: latched a fatal defect: {defect:?}"
    );
    assert!(
        records >= want_records,
        "{kind:?}: only {records} records inside the budget — a drive that searches nothing \
         cannot speak about round width"
    );
    (
        snap.gumbel_round_leaves,
        snap.gumbel_rounds,
        snap.max_sims_per_search,
    )
}

/// The mean the SCHEDULE itself implies at a given budget — derived, never transcribed.
fn schedule_mean(m: usize, budget: usize) -> f64 {
    let seq = considered_visits_sequence(m, budget);
    let mut rounds = 0usize;
    let mut i = 0usize;
    while i < seq.len() {
        let mut j = i;
        while j < seq.len() && seq[j] == seq[i] {
            j += 1;
        }
        rounds += 1;
        i = j;
    }
    seq.len() as f64 / rounds as f64
}

#[test]
fn a_gumbel_round_is_one_round_trip_and_its_width_is_the_halving_phase() {
    let (leaves, rounds, max_sims) = drive(SearchKind::Gumbel, 2);

    assert!(
        rounds > 0,
        "no Gumbel round was issued — the drive measured nothing"
    );
    assert_eq!(
        max_sims, N_SIMS as u64,
        "the search served {max_sims} leaves against a budget of {N_SIMS}; the round-width \
         mean below is only meaningful if the budget was actually spent"
    );

    let mean = leaves as f64 / rounds as f64;
    // Printed, not merely asserted: a witness that only says "in band" cannot be quoted.
    println!(
        "gumbel round width at {N_SIMS}/{GUMBEL_M}: {leaves} leaves over {rounds} round \
         trips = {mean:.3} leaves per round trip"
    );

    // THE HARD CEILING is structural: no round can be wider than the candidate set.
    assert!(
        mean <= GUMBEL_M as f64,
        "measured {mean:.3} leaves per round trip against m={GUMBEL_M} — a round cannot be \
         wider than the candidate set it is drawn from"
    );

    // THE CRITERION, one-sided: a driver issuing NARROWER rounds than the schedule's phases is
    // not batching them, while wider ones are already bounded by the structural ceiling above.
    let expected = schedule_mean(GUMBEL_M, N_SIMS - 1);
    println!(
        "criterion: driven {mean:.3} >= schedule mean {expected:.3} at ({}, {GUMBEL_M}); \
         the schedule at ({N_SIMS}, {GUMBEL_M}) means {:.3}",
        N_SIMS - 1,
        schedule_mean(GUMBEL_M, N_SIMS)
    );
    assert!(
        mean >= expected,
        "measured {mean:.3} leaves per round trip against the schedule's own {expected:.3} \
         at a budget of {} — below the schedule's own mean, so the rounds being issued are \
         not the schedule's phases",
        N_SIMS - 1
    );

    // Stated on its own so a future loosening cannot swallow it: un-batched, every round trip
    // carries exactly one leaf.
    assert!(
        mean > 1.0,
        "measured {mean:.3} leaves per round trip: the Gumbel arm is issuing ONE leaf per \
         round trip, which is the pre-batching behaviour"
    );

    // THE DISCONTINUITY, as a relation rather than a transcribed pair: the schedule's mean at
    // `N_SIMS` is STRICTLY GREATER than at `N_SIMS - 1`, which is why the criterion must name
    // which budget it means.
    assert!(
        schedule_mean(GUMBEL_M, N_SIMS) > expected,
        "the {N_SIMS}-vs-{}-discontinuity this file documents no longer holds: schedule mean \
         is {:.3} at {N_SIMS} and {expected:.3} at {}",
        N_SIMS - 1,
        schedule_mean(GUMBEL_M, N_SIMS),
        N_SIMS - 1
    );
}

#[test]
fn the_puct_kind_issues_no_rounds_and_publishes_no_width() {
    let (leaves, rounds, _max) = drive(SearchKind::Puct, 2);
    assert_eq!(
        (leaves, rounds),
        (0, 0),
        "the PUCT arm issued Gumbel rounds — the counters must read ABSENT on a kind that \
         has no halving phases, so a reader publishes the absence rather than a 0/0"
    );
}
