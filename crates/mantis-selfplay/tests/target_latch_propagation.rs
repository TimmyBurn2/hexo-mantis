//! ⊕ WP12-R Phase T (TARGET INTEGRITY) — T-3 loop-1 addition (REVIEW_IMPL_T F-1;
//! registered by the dispatcher as freeze addition FA-2): the END-TO-END latch
//! propagation oracle. NOT part of the T-2 frozen bytes — additive.
//!
//! Subject: the LAW-14 Err edge from `record_position_graph` through the PRODUCTION
//! dispatch site (`search_drive.rs` play_one_move → `record_position_graph_dispatch`)
//! to the drain-face read (`SelfPlayRunner::fatal_defect()` — the exact read the
//! bridge's `collect_graph_data` raises from). T-2's O4b pinned the latch MECHANISM
//! by mocking at the runner API; the 3-line dispatch-site glue had NO killing oracle
//! (the reviewer's M-N — swallow the Err — left the whole frozen bank green). This
//! test is RED under M-N and GREEN at the fix tree.
//!
//! Drive: the §3.4 stated-reachable raise, reached through a REAL worker — a 1-worker
//! gnn runner at sims=1 / leaf_batch=1 on a dispersed seed prefix whose first searched
//! root holds >128 children. The single sim is consumed by the root expansion, every
//! child carries 0 visits, `get_policy_ls` ships the §3.1 prior fallback over the FULL
//! child set (>128 cells), and `record_position_graph` refuses with
//! `VisitSlotsExceeded` — loud and correct (a >128-child prior dump under a real run
//! evidences a degenerate search; DESIGN_T §3.4 names this raise as CORRECT). The
//! dispatch site must latch it: store-then-halt, variant name readable at the drain
//! face, fire count visible (LAW-18 idle-at-0 asserted before start).
//!
//! Killer: M-N (dispatch-site swallow) — under it the latch never stores, the runner
//! never halts on the defect, and this test times out RED. Producer harness = the
//! target_wire_carry mock-graph-producer pattern (production `assemble_ls_from_gnn_probs`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_core::board::{Board, BoardGeometry};
use mantis_encoding::lookup_or_panic;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

fn gnn_geometry() -> BoardGeometry {
    let spec = lookup_or_panic("gnn_axis_v1");
    BoardGeometry {
        legal_move_radius: spec.legal_move_radius as i32,
        cluster_threshold: spec.cluster_threshold.unwrap_or(5) as i32,
        cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
    }
}

/// The r153 dispersal rule (farthest from the window centre), as a move prefix —
/// byte-equal construction to target_wire_carry's.
fn dispersed_prefix(n_plies: usize) -> Vec<(i32, i32)> {
    let mut board = Board::with_geometry(gnn_geometry());
    let mut moves = Vec::new();
    for _ in 0..n_plies {
        let legal = board.legal_moves();
        if legal.is_empty() {
            break;
        }
        let (cq, cr) = board.window_center();
        let &(q, r) = legal
            .iter()
            .max_by_key(|&&(q, r): &&(i32, i32)| {
                let (dq, dr) = (q - cq, r - cr);
                dq.abs().max(dr.abs()).max((dq + dr).abs())
            })
            .unwrap();
        if board.apply_move(q, r).is_err() {
            break;
        }
        moves.push((q, r));
    }
    moves
}

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
fn latch_carries_the_variant_name_from_the_dispatch_site_to_the_drain_face() {
    let spec = lookup_or_panic("gnn_axis_v1");
    let n_actions = spec.policy_logit_count;

    // Precondition (asserted, not assumed): the first searched root must hold >128
    // children so the zero-visit prior fallback overflows the HEXG visit slot.
    let prefix = dispersed_prefix(8);
    assert_eq!(prefix.len(), 8, "dispersal prefix must build 8 plies");
    let mut probe = Board::with_geometry(gnn_geometry());
    for &(q, r) in &prefix {
        probe.apply_move(q, r).expect("prefix replays");
    }
    let n_legal = probe.legal_moves().len();
    assert!(
        n_legal > 160,
        "construction: post-prefix legal set must exceed the child regime that busts \
         the derived visit slot (post-R255: sims 1 + batch 1 - 1 = capacity 1; \
         got n_legal {n_legal})"
    );

    // sims=1 + batch=1: the single sim is the root expansion → all children 0 visits
    // → §3.1 prior fallback over the FULL (>128) child set → VisitSlotsExceeded at
    // record. Boot guard 1 admits (1 + 1 - 1 = 1 <= 128) — the raise is the §3.4
    // stated-reachable arm, not a guard bypass.
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 20,
        n_simulations: 1,
        leaf_batch_size: 1,
        standard_sims: 0,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        random_opening_plies: 0,
        encoding_name: Some("gnn_axis_v1".to_string()),
        seed_fraction: 1.0,
        seed_corpus: Some(vec![prefix]),
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
        "the TargetIntegrityError never reached the drain face — the dispatch site \
         swallowed the Err (M-N shape): the LAW-14 latch edge is dead",
    );
    assert!(
        msg.contains("VisitSlotsExceeded"),
        "the VARIANT NAME must survive dispatch → latch → drain face verbatim: {msg}"
    );
    assert!(
        msg.contains("derived visit capacity 1 at"),
        "the Display fields (the DERIVED bound, 1 + 1 - 1 = 1 under this regime) must \
         ride the latch verbatim (R255): {msg}"
    );
    assert!(halted, "store-then-halt: running must be false once the latch stores (LAW-14)");
    assert!(fires >= 1, "the latch fire-count must be visible on the stats surface");
    assert!(
        drained.is_empty(),
        "the refused record must never reach the drain queue ({} records leaked)",
        drained.len()
    );
}
