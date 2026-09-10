//! ⊕ WP12-R Phase T (TARGET INTEGRITY) — T-3 loop-1 addition (REVIEW_IMPL_T F-1;
//! registered by the dispatcher as freeze addition FA-2): the END-TO-END latch
//! propagation oracle. NOT part of the T-2 frozen bytes — additive.
//!
//! Subject: the LAW-14 Err edge from a PRODUCTION store site in `search_drive.rs`
//! play_one_move to the drain-face read (`SelfPlayRunner::fatal_defect()` — the exact read
//! the bridge's `collect_graph_data` raises from). T-2's O4b pinned the latch MECHANISM by
//! mocking at the runner API; the call-site glue that hands a `TargetIntegrityError` to the
//! latch had NO killing oracle (the reviewer's M-N — swallow the Err — left the whole frozen
//! bank green). This test is RED under a swallow at that glue and GREEN at the fix tree; see
//! the R43 edit note below for WHICH store site it now drives and what that costs.
//!
//! Drive: a 1-worker gnn runner at sims=1 / leaf_batch=1. The single sim is consumed by the
//! root expansion and every child carries 0 visits.
//!
//! R346(f) EDIT, DISCLOSED. The drive used to plant a DISPERSED 8-ply prefix through
//! `seed_fraction` / `seed_corpus` so the first searched root provably held >128 children,
//! and asserted that width as a construction precondition. Those two config keys went with
//! the seed-corpus lever, and the runner has no start-position seeding left, so the width
//! is now whatever `random_opening_plies` produces and is no longer asserted. The refusal
//! under test does NOT depend on it — `refuse_zero_visit_export` fires on any zero-visit
//! search — but the file no longer drives the byte-identical pre-fix construction, and the
//! R43 note below should be read with that in mind.
//!
//! ── R43 EDIT, F-816-9 Phase C (R275(b)). DISCLOSED, NOT QUIET ────────────────────────
//! This drive USED to end at `record_position_graph`'s `VisitSlotsExceeded`: the zero-visit
//! root shipped the §3.1 prior fallback over its full >128-child set and the capacity guard
//! refused it. R275(b)'s EXPORTER conjunct now refuses the same search EARLIER and by its
//! actual name — `records::refuse_zero_visit_export` raises `ZeroVisitSearch` before any
//! exporter runs — so this drive dies upstream of the record dispatch and the asserted
//! variant changes with it. Every other claim in this file is unchanged and still asserted:
//! store-then-halt, the variant name surviving verbatim to the drain face, the fire count
//! visible, and no refused record reaching the drain queue.
//!
//! WHAT THIS EDIT COSTS, stated rather than absorbed. M-N was "swallow the Err at the
//! `record_position_graph_dispatch` call site", and this file was its only killer. After the
//! exporter pin, that Err has NO reachable production driver at all: with visits > 0 the
//! exported support is bounded by the sims actually backed up, and R255 derives capacity as
//! `max_armed + leaf_batch − 1`, so a healthy search cannot exceed it (Phase A measured max
//! 11 against capacity 57 across a full 128-ply game). The capacity guard is now pure
//! defense-in-depth — correct, required to stay (R274(c)), and unfireable in the current
//! visit-limited construction. It becomes load-bearing again exactly under R275(a)'s
//! retirement clause (completed-Q-on-graph, whose support is child-count-wide). So M-N is
//! no longer killed by any end-to-end drive, and no test here pretends otherwise; the
//! surviving coverage of that seam is `record.rs`'s in-src forwarding oracle plus the
//! frozen `target_integrity_postfix.rs` bank, which calls the constructor directly.
//!
//! Killer (post-edit): M-STORE (swallow the Err at the exporter pin's call site in
//! `play_one_move`) — under it the latch never stores, the runner never halts, and this test
//! times out RED. Producer harness = the target_wire_carry mock-graph-producer pattern
//! (production `assemble_ls_from_gnn_probs`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Mock graph producer: uniform probs over each request's legal nodes through the
/// PRODUCTION `assemble_ls_from_gnn_probs` (the target_wire_carry harness pattern).
fn spawn_graph_producer(queue: GraphQueue, n_actions: usize, served: Arc<AtomicUsize>) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(4, 5);
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
            let res = assemble_ls_from_gnn_probs(n_actions, &probs, &g.policy_scatter_index.0, &coords)
                .map(|ls| (ls, 0.0f32));
            ids.push(id);
            results.push(res);
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        queue.submit_graph_results(&ids, results);
    })
}

#[test]
fn latch_carries_the_variant_name_from_the_production_store_site_to_the_drain_face() {
    let spec = lookup_or_panic("gnn_axis_v1");
    let n_actions = spec.policy_logit_count;

    // sims=1 + batch=1: the single sim is the root expansion → all children 0 visits →
    // `refuse_zero_visit_export` raises `ZeroVisitSearch` before any exporter runs. The
    // 8 random opening plies stand in for the deleted dispersed seed prefix: they widen the
    // first searched root without asserting a width the refusal does not depend on.
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 20,
        n_simulations: 1,
        leaf_batch_size: 1,
        standard_sims: 0,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        random_opening_plies: 8,
        encoding_name: Some("gnn_axis_v1".to_string()),
        ..Default::default()
    })
    .expect("gnn runner must construct");

    // LAW-18 idle posture: the latch surface is VISIBLE at rest.
    assert!(runner.fatal_defect().is_none(), "fresh runner carries no defect");
    assert_eq!(runner.stats_snapshot().target_integrity_defects, 0);

    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_graph_producer(runner.graph_producer(), n_actions, served.clone());

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut defect: Option<String> = None;
    while Instant::now() < deadline {
        defect = runner.fatal_defect();
        if defect.is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    // Read the halt state BEFORE our own stop() so the store-then-halt property is
    // what is observed, not the test teardown.
    let halted = !runner.is_running();
    let fires = runner.stats_snapshot().target_integrity_defects;
    let drained = runner.drain_graph_records();
    runner.stop();
    producer.join().expect("mock graph producer exits on close");

    assert!(served.load(Ordering::Relaxed) > 0, "no graph inference served — vacuous drive");
    let msg = defect.expect(
        "the TargetIntegrityError never reached the drain face — the production store site \
         swallowed the Err (M-STORE, the M-N shape one seam further up): the LAW-14 latch \
         edge is dead",
    );
    assert!(
        msg.contains("ZeroVisitSearch"),
        "the VARIANT NAME must survive store site → latch → drain face verbatim: {msg}"
    );
    assert!(
        msg.contains("backed up ZERO visits"),
        "the Display must name WHAT the defect is, not merely that one occurred — the \
         pre-fix death said `192 cells exceed capacity 57` and named neither the failed \
         search nor its cause, which is what cost this defect its diagnosis: {msg}"
    );
    assert!(halted, "store-then-halt: running must be false once the latch stores (LAW-14)");
    assert!(fires >= 1, "the latch fire-count must be visible on the stats surface");
    assert!(
        drained.is_empty(),
        "the refused record must never reach the drain queue ({} records leaked)",
        drained.len()
    );
}
