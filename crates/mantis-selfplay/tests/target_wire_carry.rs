// R8 justify: ONE oracle needs a live graph runner — the mock graph producer, the off-window
// re-derivation and the drain loop are its single-purpose harness.
//! Wire/record carry through finalize → drain: visits reach the drain verbatim.
//!
//! Drive: a REAL 1-worker graph runner (gnn_axis_v1) with a mock graph producer (uniform
//! legal-node probs through the production `assemble_ls_from_gnn_probs`) and a 40-ply random
//! opening, so recorded positions carry off-window visit mass. The game ply-caps within a few
//! moves and `drain_graph_records` returns the finalized records. Measured on this generator:
//! 8 of 8 records carry off-window visit coords (20 coords) at 8 sims.
//!
//! Asserts, per drained record: the visit target is a full-mass distribution (Σ == 1 ± 1e-4 —
//! finalize stamps outcome/value_valid/game_length ONLY and must not touch `visits`), coords
//! are unique, and across the drain at least one record carries an OFF-WINDOW visit coord,
//! re-derived from the record's own stones — the de-vacuum precondition.
//!
//! Killer: M-K (a perturbed visit in the finalize push breaks Σ == 1); M-J also reds this suite.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_core::board::Board;
use mantis_core::Cell;
use mantis_encoding::lookup_or_panic;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::replay::hexg::GraphRecord;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Mock graph producer: uniform probs over each request's legal nodes, assembled by the
/// PRODUCTION `assemble_ls_from_gnn_probs` (the shared producer both consumers reach).
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

/// Re-derive off-window-ness of a visit coord from the record's OWN stones (the rebuild
/// board recomputes the identical bbox window centre — `Board::from_stones` contract).
fn record_offwindow_visits(rec: &GraphRecord, n_actions: usize) -> usize {
    let stones: Vec<((i32, i32), Cell)> = rec
        .stones
        .iter()
        .map(|&(q, r, p)| {
            ((i32::from(q), i32::from(r)), if p == 1 { Cell::P1 } else { Cell::P2 })
        })
        .collect();
    let player = if rec.current_player == 1 {
        mantis_core::Player::One
    } else {
        mantis_core::Player::Two
    };
    let board = Board::from_stones(&stones, player, rec.moves_remaining, 0, None);
    rec.visits
        .iter()
        .filter(|&&(q, r, _)| board.window_flat_idx(i32::from(q), i32::from(r)) >= n_actions - 1)
        .count()
}

#[test]
fn s2w_drained_graph_records_carry_full_mass_visits_verbatim() {
    let spec = lookup_or_panic("gnn_axis_v1");
    let n_actions = spec.policy_logit_count;
    let cfg = SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 44, // opening 40 + a few searched moves → fast ply-cap finalize
        n_simulations: 8,
        leaf_batch_size: 4,
        standard_sims: 0,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        random_opening_plies: 40,
        encoding_name: Some("gnn_axis_v1".to_string()),
        ..Default::default()
    };
    let runner = SelfPlayRunner::new(cfg).expect("gnn runner must construct");
    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_graph_producer(runner.graph_producer(), n_actions, served.clone());

    runner.start();
    // Bounded wait for >=1 COMPLETED game, so drain returns finalized records.
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut records: Vec<GraphRecord> = Vec::new();
    while Instant::now() < deadline {
        records.extend(runner.drain_graph_records());
        if records.len() >= 3 {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    runner.stop();
    producer.join().expect("mock graph producer exits on close");

    assert!(
        served.load(Ordering::Relaxed) > 0,
        "no graph inference served — the worker never searched (vacuous drive)"
    );
    assert!(
        records.len() >= 3,
        "expected >=3 finalized graph records from the seeded ply-cap game, got {} — \
         the finalize→drain path was not exercised",
        records.len()
    );

    let mut any_offwindow = 0usize;
    for rec in &records {
        let sum: f64 = rec.visits.iter().map(|&(_, _, p)| f64::from(p)).sum();
        assert!(
            (sum - 1.0).abs() <= 1e-4,
            "drained record ply {} carries visit mass {sum} != 1 — the target was \
             mutated between export and drain (finalize must stamp outcome ONLY)",
            rec.ply_index
        );
        let mut coords: Vec<(i16, i16)> = rec.visits.iter().map(|&(q, r, _)| (q, r)).collect();
        coords.sort_unstable();
        let n = coords.len();
        coords.dedup();
        assert_eq!(coords.len(), n, "duplicate visit coord in a drained record");
        any_offwindow += record_offwindow_visits(rec, n_actions);
    }
    assert!(
        any_offwindow > 0,
        "no drained record carries an off-window visit coord — the drive failed to \
         exercise the regime the carry oracle exists for (de-vacuum)"
    );
}

// Post-fix only: `export_offwindow_mass_moves` has a LIVE producer — the same drive as above,
// read through `stats_snapshot()`, with idle-at-0 asserted before start.
// Killer: M-H (export-counter sub-run).
#[cfg(feature = "phase_t_postfix")]
#[test]
fn ctr_export_offwindow_mass_moves_fires_on_a_dispersed_run() {
    let spec = lookup_or_panic("gnn_axis_v1");
    let n_actions = spec.policy_logit_count;
    let cfg = SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 44,
        n_simulations: 8,
        leaf_batch_size: 4,
        standard_sims: 0,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        random_opening_plies: 40,
        encoding_name: Some("gnn_axis_v1".to_string()),
        ..Default::default()
    };
    let runner = SelfPlayRunner::new(cfg).expect("gnn runner must construct");
    assert_eq!(
        runner.stats_snapshot().export_offwindow_mass_moves,
        0,
        "the counter must be VISIBLE at 0 when idle (LAW-18)"
    );
    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_graph_producer(runner.graph_producer(), n_actions, served.clone());

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut fired = 0u64;
    while Instant::now() < deadline {
        let snap = runner.stats_snapshot();
        fired = snap.export_offwindow_mass_moves;
        if fired >= 1 && snap.positions_generated >= 3 {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    runner.stop();
    producer.join().expect("producer exits");
    assert!(served.load(Ordering::Relaxed) > 0, "no graph inference served — vacuous drive");
    assert!(
        fired >= 1,
        "export_offwindow_mass_moves never fired on a drive whose exports provably carry \
         overflow mass — the counter's producer is not wired (M-H)"
    );
}
