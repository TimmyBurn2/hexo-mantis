//! The END-TO-END latch propagation oracle: the LAW-14 Err edge from a PRODUCTION store site in
//! `search_drive.rs` `play_one_move` to the drain-face read (`SelfPlayRunner::fatal_defect()`,
//! the exact read the bridge's `collect_graph_data` raises from). The latch MECHANISM is pinned
//! elsewhere by mocking at the runner API; the call-site glue that hands a `TargetIntegrityError`
//! to the latch had no killing oracle, so swallowing that Err left the whole frozen bank green.
//!
//! Drive: a 1-worker gnn runner at sims=1 / leaf_batch=1, so the single sim is consumed by the
//! root expansion, every child carries 0 visits, and `records::refuse_zero_visit_export` raises
//! `ZeroVisitSearch` before any exporter runs. Asserted: store-then-halt, the variant name
//! surviving verbatim to the drain face, the fire count visible on the stats surface, and no
//! refused record reaching the drain queue.
//!
//! Killer: M-STORE — swallow the Err at the exporter pin's call site in `play_one_move`; the
//! latch then never stores, the runner never halts, and this test times out RED.

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

    // sims=1 + batch=1: the single sim is the root expansion, so all children carry 0 visits and
    // `refuse_zero_visit_export` raises `ZeroVisitSearch` before any exporter runs. The 8 random
    // opening plies widen the first searched root; the refusal does not depend on that width.
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
    // Read the halt state BEFORE our own stop(), so store-then-halt is what is observed.
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
