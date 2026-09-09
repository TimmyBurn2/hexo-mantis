//! ⊕ CANDIDATE-PARALLEL BATCHING: a Gumbel halving round is ONE inference round trip.
//!
//! WHAT IS BEING MEASURED, and why a counter rather than a stopwatch. Sequential Halving
//! visits every candidate alive at the current considered visit level before that level
//! advances, so a ROUND is `m` descents, then `m/2`, and so on. Issuing them together is the
//! Gumbel analogue of `leaf_batch_size` — and unlike PUCT's batch it needs no virtual loss,
//! because the candidates' subtrees are disjoint. The lever is invisible in outputs: a search
//! that had silently gone back to one leaf per round trip would produce the SAME root visit
//! counts and the same move, only N times slower. So the width is counted in-run (LAW-18) and
//! this file reads the counter.
//!
//! THE DENSE PATH, for `served_sims_exact.rs`'s reason: `search.kind: gumbel` on the GRAPH
//! path is refused at boot by `replay::hexg::derived_visit_capacity`.
//!
//! THE MEASURED MEAN AT 320/16 IS 3.71, NOT 4, AND THE GAP IS ARITHMETIC RATHER THAN A
//! SHORTFALL. The schedule's round-width profile is discontinuous at its budget: a schedule
//! built for 320 simulations means 4.267 leaves per round trip, and one built for 319 means
//! 3.709. The Gumbel search gets 319 because the ROOT'S OWN EVALUATION IS CHARGED against
//! `n_simulations` — the same clause that makes "N means N leaves" true. The two numbers
//! cannot both be had at `n_simulations: 320`. The floor asserted below is therefore derived
//! from the schedule itself rather than transcribed from a target, and the thing it has to
//! separate is batching from no batching: un-batched is 1.0.

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_search::mcts::seq_halving::considered_visits_sequence;
use mantis_search::SearchKind;
use mantis_selfplay::queues::DenseQueue;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// The run6 full-arm regime the ruling names.
const N_SIMS: usize = 320;
const GUMBEL_M: usize = 16;

fn spawn_producer(
    queue: DenseQueue,
    policy_stride: usize,
    served: Arc<AtomicUsize>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_batch(1, 5);
        if batch.is_empty() {
            if queue.is_closed() {
                break;
            }
            continue;
        }
        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        let mut flat: Vec<f32> = Vec::new();
        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(batch.len());
        let mut values: Vec<f32> = Vec::with_capacity(batch.len());
        let uniform = 1.0f32 / policy_stride as f32;
        for _ in &batch {
            let start = flat.len();
            flat.extend(std::iter::repeat_n(uniform, policy_stride));
            ranges.push(start..flat.len());
            values.push(0.0);
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        let arc = Arc::new(flat);
        queue.submit_results(&ids, &arc, &ranges, &values);
    })
}

/// `(round_leaves, rounds, max_sims_per_search)` from one driven worker.
fn drive(kind: SearchKind, want_rows: usize) -> (u64, u64, u64) {
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
        solver_enabled: false,
        forced_win_policy_enabled: false,
        encoding_name: Some("v6".to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let policy_stride = runner.policy_len();
    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_producer(runner.dense_producer(), policy_stride, served);

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut rows = 0usize;
    while Instant::now() < deadline {
        rows += runner.drain_training_rows().len();
        if rows >= want_rows || runner.fatal_defect().is_some() {
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
        rows >= want_rows,
        "{kind:?}: only {rows} rows inside the budget — a drive that searches nothing \
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
    // Printed, not merely asserted: the QUANTITY is what a re-mint reads, and a witness that
    // only says "in band" cannot be quoted (LAW-01, measurement mandatory).
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

    // THE BAND is the schedule's own mean at the budget the SEARCH gets, which is
    // `N_SIMS - 1` because the root's own evaluation is charged. The driven mean tracks it
    // rather than equalling it, in BOTH directions and for stated reasons: a round is
    // truncated by the remaining budget and an overlapping descent is skipped (pulling it
    // down), while a search cut short at `stop()` loses the schedule's NARROW tail rounds
    // (pushing it up). A 20 % band is what those two residuals fit in at this drive's size.
    let expected = schedule_mean(GUMBEL_M, N_SIMS - 1);
    assert!(
        mean >= expected * 0.8 && mean <= expected * 1.2,
        "measured {mean:.3} leaves per round trip against the schedule's own {expected:.3} \
         at a budget of {} — outside the residual band, so the rounds being issued are not \
         the schedule's phases",
        N_SIMS - 1
    );

    // AND THE THING THE BAND EXISTS TO SEPARATE, stated on its own so a future widening of
    // the band cannot swallow it: un-batched, every round trip carries exactly one leaf.
    assert!(
        mean > 1.0,
        "measured {mean:.3} leaves per round trip: the Gumbel arm is issuing ONE leaf per \
         round trip, which is the pre-batching behaviour"
    );

    // The ruling's target for this regime is 4; the schedule delivers 4.267 at a budget of
    // 320 and 3.709 at 319, and the search gets 319 because the root is charged. Asserted so
    // the discrepancy is a recorded fact rather than a silent shortfall.
    assert!(
        schedule_mean(GUMBEL_M, N_SIMS) > 4.0 && expected < 4.0,
        "the 320-vs-319 discontinuity this file documents no longer holds: schedule mean is \
         {:.3} at {N_SIMS} and {expected:.3} at {}",
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
