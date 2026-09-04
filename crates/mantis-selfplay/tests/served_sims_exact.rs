//! ⊕ R335(c) — a search serves EXACTLY `n_simulations` leaves, never more.
//!
//! THE FINDING THIS EXISTS FOR. `PERF_TRANCHE2_RESULTS.md` §7/§20 measured **53.46 served
//! sims/move against `mcts.n_simulations: 50`** — a ~7 % overshoot — and stated it rather
//! than correcting it. The mechanism is the last batch of a search: the PUCT and Gumbel
//! fallback loops requested a full `leaf_batch_size` while fewer than that remained in the
//! budget, so the budget was overrun by up to `leaf_batch_size − 1` on every move.
//!
//! WHY IT IS A MINT PRECONDITION AND NOT A PERF ITEM. R334(f)(ii) pre-registers the run6
//! success witness as *"beats `sealbot_d5` at FIXED NODES"*. A fixed-node claim is unstatable
//! while the served node count is 7 % above the number the config carries, and every g/h
//! derived from `n_simulations` alone is optimistic by the same factor.
//!
//! THE TWO PYTHON HEADS ALREADY DID THIS. `arena/deploy_head.py::select_move` and
//! `selfplay/worker.py` both compute `min(leaf_batch_size, n_sims − sims_done)`; only the
//! Rust search drive did not. The deploy head's own comment names a DIFFERENT self-play
//! divergence — crediting the request vs the return — which this file does not touch and
//! which R318(b)(iii) settled; the clamp is orthogonal to it.
//!
//! WHAT IS MEASURED. The mock producer counts every leaf it serves. With `n_workers: 1` and
//! `random_opening_plies: 0` exactly one search is in flight at a time and every searched ply
//! produces one record, so `served / records` IS the ledger's served-sims figure, re-measured
//! at unit scale against the same denominator.
//!
//! WHY THE PLY CAP IS TINY, and it is not an arbitrary speed knob. Rows reach
//! `drain_graph_records` only when a GAME FINALIZES — an in-progress game's rows live in the
//! worker's local vec. The first draft of this file used the production 128-ply cap and three
//! of its four drives recorded ZERO inside a 600 s deadline, because a debug-build game at 600
//! sims does not finish. A short cap makes each game a whole number of searches that lands in
//! the drain promptly; the property under test is per-SEARCH and does not care how deep the
//! board is.
//!
//! WHY THE PRIMARY ASSERTION IS A COUNTER AND NOT THE PRODUCER'S TALLY. The served tally is an
//! AGGREGATE over the drive, and a worker that has begun the next game when `stop()` lands has
//! already served leaves for a search no record will ever account for. Measured: the tally read
//! 401 against an expected 400 AFTER the clamp — a harness residual of one in-flight search,
//! not a defect. `max_sims_per_search` is exact because it advances with the search it
//! measures, and it is a MAX rather than a mean because a mean hides one overshooting search
//! among many. The tally is kept as the SECOND assertion, bounded rather than exact, because it
//! is the quantity the ledger's `53.46 sims/move` line is denominated in and a witness that
//! measured only the counter could not speak to that line at all.
//!
//! Killer / PLANTED BREAK: revert either clamp in `search_drive::run_mcts_search` and this
//! file reds — at HEAD before the fix it read 56 served against 50 at `leaf_batch_size 8`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
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
                .map(|&row| (g.node_coords[row as usize * 2], g.node_coords[row as usize * 2 + 1]))
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

/// Drive one worker until `want_records` searched plies have been recorded, returning
/// `(served_leaves, records, max_sims_per_search)`.
fn drive(
    encoding: &str,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
    gumbel_mcts: bool,
) -> (usize, usize, u64) {
    let spec = lookup_or_panic(encoding);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: ply_cap,
        n_simulations,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        dirichlet_enabled: true,
        gumbel_mcts,
        solver_enabled: false,
        forced_win_policy_enabled: false,
        encoding_name: Some(encoding.to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    let producer =
        spawn_counting_producer(runner.graph_producer(), spec.policy_logit_count, served.clone());

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

    assert!(defect.is_none(), "{encoding} @ {n_simulations} latched a fatal defect: {defect:?}");
    assert!(
        records.len() >= want_records,
        "{encoding} @ {n_simulations}: only {} searched plies inside the budget — a drive that \
         records nothing cannot speak about served sims at all",
        records.len()
    );
    (served.load(Ordering::Relaxed), records.len(), snap.max_sims_per_search)
}

fn assert_exact(encoding: &str, n_simulations: usize, ply_cap: usize, want_records: usize) {
    assert_exact_arm(encoding, n_simulations, ply_cap, want_records, false);
}

fn assert_exact_arm(
    encoding: &str,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
    gumbel_mcts: bool,
) {
    let (served, records, max_sims) =
        drive(encoding, n_simulations, ply_cap, want_records, gumbel_mcts);

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
    assert_exact("gnn_axis_v1", 50, 4, 8);
}

#[test]
fn r8_at_fifty_sims_serves_exactly_fifty_per_search() {
    assert_exact("gnn_axis_r8", 50, 4, 8);
}

#[test]
fn r6_at_six_hundred_sims_serves_exactly_six_hundred_per_search() {
    assert_exact("gnn_axis_v1", 600, 2, 2);
}

#[test]
fn r8_at_six_hundred_sims_serves_exactly_six_hundred_per_search() {
    assert_exact("gnn_axis_r8", 600, 2, 2);
}

/// GUMBEL NEVER OVERSHOOTS — AND IT UNDERSHOOTS, WHICH IS A SEPARATE, PRE-EXISTING FACT.
///
/// The PUCT tests above cannot reach the Gumbel dispatcher, so without these arms the whole
/// Gumbel side of the budget would be unmeasured. What they measure is NOT `== N`: sequential
/// halving allocates `sims_per = remaining_budget / (remaining_phases * candidates)`, and the
/// integer division's REMAINDER is never allocated to anyone. Measured at HEAD: **49 of 50**
/// and **599 of 600**.
///
/// THE UNDERSHOOT PREDATES THIS LEG AND IS NOT ITS DOING. It reads 49 and 599 identically with
/// and without the `move_sims` clamp R335(c) briefly added to the halving loop — which is the
/// measurement that proved that clamp DEAD and removed it. Chasing the remainder would be a
/// change to the Gumbel allocator, and Gumbel is explicitly out of this tranche (R334(e)) and
/// is one half of R335(d)'s operator-decided ablation pair. So it is PINNED here, not fixed:
/// the assertion is the property R335(c) actually rules on (never MORE than the budget), plus
/// the measured undershoot as a tripwire, so whoever arms Gumbel sees this line first.
fn assert_no_overshoot_and_pin_undershoot(
    encoding: &str,
    n_simulations: usize,
    ply_cap: usize,
    want_records: usize,
    pinned_max: u64,
) {
    let (served, records, max_sims) = drive(encoding, n_simulations, ply_cap, want_records, true);
    assert!(
        max_sims <= n_simulations as u64,
        "gumbel {encoding} @ {n_simulations}: the widest search served {max_sims} leaves \
         against a budget of {n_simulations}. R335(c) — a search never serves MORE than N."
    );
    assert_eq!(
        max_sims, pinned_max,
        "gumbel {encoding} @ {n_simulations}: the widest search served {max_sims}, pinned at \
         {pinned_max}. The gap to {n_simulations} is sequential halving's unallocated \
         integer-division remainder, measured at HEAD and DELIBERATELY not fixed (Gumbel is \
         out of this tranche, R334(e), and is R335(d)'s operator ablation). If this moved, \
         the allocator moved — re-derive before arming Gumbel."
    );
    assert!(
        served >= records * pinned_max as usize,
        "gumbel {encoding} @ {n_simulations}: served {served} over {records} searches is below \
         even the pinned per-search maximum — the drive is not measuring what it claims."
    );
}

#[test]
fn gumbel_r6_at_fifty_sims_never_exceeds_the_budget() {
    assert_no_overshoot_and_pin_undershoot("gnn_axis_v1", 50, 4, 8, 49);
}

#[test]
fn gumbel_r8_at_six_hundred_sims_never_exceeds_the_budget() {
    assert_no_overshoot_and_pin_undershoot("gnn_axis_r8", 600, 2, 2, 599);
}
