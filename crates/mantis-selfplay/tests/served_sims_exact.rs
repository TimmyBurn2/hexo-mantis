// R8 justify: one claim — "a search serves the budget its config states, under BOTH search
// kinds" — measured through real `SelfPlayRunner` drives. The two kinds share the counting
// producer that IS the measurement; split it and the numbers stop being comparable, which is
// the whole point of reading the same property on both arms.
//! ⊕ a search serves EXACTLY `n_simulations` leaves, never more and never fewer.
//!
//! THE FINDING THIS EXISTS FOR. `PERF_TRANCHE2_RESULTS.md` §7/§20 measured **53.46 served
//! sims/move against `mcts.n_simulations: 50`** — a ~7 % overshoot — and stated it rather
//! than correcting it. The mechanism was the last batch of a search: the PUCT loop requested
//! a full `leaf_batch_size` while fewer than that remained in the budget, so the budget was
//! overrun by up to `leaf_batch_size − 1` on every move. The Gumbel side had the opposite
//! defect: a phase allocator that dropped its integer-division remainder, measured at 49 of
//! 50 and 599 of 600.
//!
//! BOTH ARE NOW CLOSED, AND THE ROOT IS CHARGED ON BOTH ARMS. `N` means `N leaves of
//! network work`: the root's own evaluation is one of the N under either kind, and no config
//! key can move that. The deleted `gumbel_root_counts` made the charge a mint decision, which
//! meant "equal NN work at a fixed budget" was a claim a config could quietly falsify.
//!
//! WHY IT IS A MINT PRECONDITION AND NOT A PERF ITEM. R334(f)(ii) pre-registers the run6
//! success witness as *"beats `sealbot_d5` at FIXED NODES"*. A fixed-node claim is unstatable
//! while the served node count disagrees with the number the config carries, and every g/h
//! derived from `n_simulations` alone is wrong by the same factor.
//!
//! WHICH ENCODING EACH ARM DRIVES. Every arm now drives a GRAPH encoding, because after
//! R346(f) there is exactly one representation and one recorder left. The four budget arms
//! (50 / 600) hold the KIND fixed at `puct` and vary the radius, `gnn_axis_v1` against
//! `gnn_axis_r8`, so a served-sims claim is not read off one geometry; the two run6-regime
//! arms (64 / 320) hold the ENCODING fixed at `gnn_axis_r8` and vary the kind, so the
//! comparison between `puct` and `gumbel` is a comparison of searches and of nothing else.
//! That second half is what the arrangement was always for — this file used to hold the
//! recorder fixed by driving the GRID encoding under both kinds, and holding one graph row
//! fixed is the same control with the only representation that still exists.
//!
//! WHAT IS MEASURED. The mock producers count every leaf they serve. With `n_workers: 1` and
//! `random_opening_plies: 0` exactly one search is in flight at a time.
//!
//! WHY THE PRIMARY ASSERTION IS A COUNTER AND NOT THE PRODUCER'S TALLY. The served tally is an
//! AGGREGATE over the drive, and a worker that has begun the next game when `stop()` lands has
//! already served leaves for a search no record will ever account for. Measured: the tally read
//! 401 against an expected 400 AFTER the clamp — a harness residual of one in-flight search,
//! not a defect. `max_sims_per_search` is exact because it advances with the search it
//! measures, and it is a MAX rather than a mean because a mean hides one overshooting search
//! among many.
//!
//! Killer / PLANTED BREAK: revert the PUCT clamp in `search_drive::run_mcts_search` and the
//! PUCT arms red — at HEAD before the fix they read 56 served against 50 at
//! `leaf_batch_size 8`. Stop charging the root and every arm reads N−1 or N+1.

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

/// Drive one worker on the GRAPH path until `want_records` searched plies have been
/// recorded, returning `(served_leaves, records, max_sims_per_search)`.
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
        solver_enabled: false,
        forced_win_policy_enabled: false,
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
    // Rows finalized between the break and `stop` are still this drive's searches; the
    // producer keeps serving until the queue closes, so both halves must be read AFTER the
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

/// Drive one worker under `kind` at the run6 identity row, returning
/// `(served_leaves, records, max_sims_per_search)`. The ENCODING is held fixed here so the
/// only thing that varies between the two calls is the search kind.
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
        // Dirichlet is a PUCT mechanism; leaving it armed keeps the PUCT arm honest and it
        // is inert under Gumbel by construction.
        dirichlet_enabled: true,
        search_kind: kind,
        quiescence_enabled: false,
        solver_enabled: false,
        forced_win_policy_enabled: false,
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

    // (1) THE PROPERTY, exactly: no search served more than its budget, and at least one
    // search spent the whole of it (so a runner that silently searched less would also red).
    assert_eq!(
        max_sims, n_simulations as u64,
        "{encoding} @ n_simulations={n_simulations}, leaf_batch_size={LEAF_BATCH}: the widest \
         search served {max_sims} leaves against a budget of {n_simulations}. A search must \
         stop at EXACTLY N (R335(c)) — an overshoot makes every `fixed nodes` claim and every \
         g/h derived from `n_simulations` wrong by the same factor, which is what the \
         53.46-vs-50 ledger line recorded; an undershoot means the budget is not being spent."
    );

    // (2) THE LEDGER'S OWN DENOMINATOR, bounded. `served / records` is the served-sims figure
    // §20 published. It cannot be asserted exactly — see the header — so it is bounded by one
    // in-flight search, which is strictly tighter than the pre-clamp reading at every shape
    // measured (r6@50 446, r8@50 443, both @600 1204).
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

// ── the run6 target regime's own two budgets, on BOTH kinds ─────────────────────
//
// 64 is the fast arm's budget and 320 the full arm's. They are asserted here because a
// served-sims claim taken at 50 and 600 says nothing about the numbers a run will actually
// be minted at, and "N means N leaves" is the property the fixed-node witness rests on.

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
