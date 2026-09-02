// R8 justify: one pin, and the legs only mean anything together — the direct call proves it
// bites, the positive control proves it does not bite everything, and the two end-to-end
// drives prove the SAME predicate holds on both record arms. Split them and a reader can
// green the refusal while its positive control lives in another file; the arm pair in
// particular is one claim ("arm-independent") that reads as two half-claims apart.
//! ⊕ F-816-9 Phase C — the EXPORTER conjunct pin (R275(b) conjunct 2, LAW-14).
//!
//! Subject: `records::refuse_zero_visit_export`. A search that backed up ZERO child visits
//! has no visit distribution to export; every exporter falls back to the (ε-noise-mixed)
//! priors, and the result is indistinguishable in the replay buffer from a real target.
//! On the shakedown regime that prior dump busts the derived visit slot and dies loud —
//! but capacity is derived from the sims regime, so at the prereg PCR 600/75 regime
//! capacity is 674, 192 < 674, and the SAME corrupt target is RECORDED SILENTLY
//! (Phase A §7.1). The refusal below does not depend on capacity at all.
//!
//! FLIP-SET (b): a zero-visit search result handed DIRECTLY to the exporter is refused
//! loud. This bypasses the seam pin entirely — no queue, no inference, no failure — which
//! is what proves the exporter pin bites ALONE.
//!
//! PIN INDEPENDENCE, without a scratch build. `sims = 1` reaches the zero-visit state with
//! every inference HEALTHY: the single sim is consumed by the root expansion and no child
//! is ever visited. The end-to-end legs drive exactly that and assert the run dies with
//! `target_integrity_defects == 1` and `inference_failures_total == 0` — the seam never
//! fired, so the exporter pin is the only thing that stopped it. Its mirror image lives in
//! `search_seam_fatal.rs`, where the seam fires and the target counter stays 0.
//!
//! WHAT THAT DOES **NOT** SHOW (cross-model RED-TEAM correction, kept because the loose
//! claim is the tempting one): independence is not equivalence. Either pin alone stops
//! F-816-9's OBSERVED death — a 192-cell prior dump is the full child set, so the failure
//! had to land on the first post-root batch and leave zero visits. A failure landing LATER
//! leaves a TRUNCATED search with nonzero visits, which passes this pin and is caught only
//! at the seam. The seam is the primary; this is the backstop. See `records.rs`'s
//! `refuse_zero_visit_export` doc for the same statement at the fix site.
//!
//! FLIP-SET (d) — the capacity boundary — is deliberately NOT re-tested here. It is owned
//! by the frozen `target_integrity_postfix.rs::s2b_admits_exactly_128_mass_cells` /
//! `::s2b_refuses_129_mass_cells_with_the_typed_error`, which call `record_position_graph`
//! directly and are therefore untouched by this pin. Duplicating them would put a second
//! authority on a guard the packet requires to stay exactly as it is.
//!
//! Killers: M-ZV-1 (`refuse_zero_visit_export` returns `Ok` unconditionally — the direct
//! and end-to-end legs go RED); M-ZV-2 (sum ROOT visits instead of CHILD visits — the root
//! backs up one visit to itself during its own expansion, so the refusal never fires and
//! both legs go RED; this is the exact off-by-one the defect lived in); M-ZV-3 (call the
//! refusal AFTER the record dispatch — the "nothing reaches the buffer" assert survives but
//! the positive-control leg's ordering claim does not).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_core::board::{Board, BoardGeometry};
use mantis_core::{Cell, Player};
use mantis_encoding::lookup_or_panic;
use mantis_search::{LegalSetPolicy, MCTSTree};
use mantis_selfplay::queues::{DenseQueue, GraphQueue};
use mantis_selfplay::records::{assemble_ls_from_gnn_probs, refuse_zero_visit_export, TargetIntegrityError};
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

const NA: usize = 362; // gnn_axis_v1 policy stride (19*19 + 1)
const TRUNK: i32 = 19;

fn gnn_geometry() -> BoardGeometry {
    let spec = lookup_or_panic("gnn_axis_v1");
    BoardGeometry {
        legal_move_radius: spec.legal_move_radius as i32,
        cluster_threshold: spec.cluster_threshold.unwrap_or(5) as i32,
        cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
    }
}

/// Three well-separated stones → a wide legal set (the frozen bank's `wide_board`).
fn wide_board() -> Board {
    let stones: Vec<((i32, i32), Cell)> =
        vec![((0, 0), Cell::P1), ((8, 0), Cell::P2), ((0, 8), Cell::P1)];
    Board::from_stones(&stones, Player::One, 2, 0, None)
}

/// Uniform prior over the board's legal set, routed dense/overflow by flat index.
fn uniform_prior(board: &Board) -> LegalSetPolicy {
    let legal = board.legal_moves();
    let p = 1.0f32 / legal.len() as f32;
    let mut ls = LegalSetPolicy { dense: vec![0.0; NA], overflow: Default::default() };
    for &(q, r) in &legal {
        let flat = board.window_flat_idx(q, r);
        if flat < NA {
            ls.dense[flat] = p;
        } else {
            ls.overflow.insert((q, r), p);
        }
    }
    ls
}

/// A tree whose root is expanded and whose children carry ZERO visits — the state a
/// `sims = 1` search and a mid-search inference failure both land in.
fn zero_visit_tree(board: &Board) -> MCTSTree {
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board.clone());
    let leaves = tree.select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    assert_eq!(leaves.len(), 1, "construction: the root must be the only pending leaf");
    let centers = vec![board.window_center()];
    tree.expand_and_backup_ls_at(&[uniform_prior(board)], &[0.0f32], &centers, TRUNK);
    tree
}

// ── FLIP-SET (b): the exporter pin, called directly ──────────────────────────────────

#[test]
fn a_zero_visit_search_handed_to_the_exporter_is_refused_loud() {
    let board = wide_board();
    let tree = zero_visit_tree(&board);

    // PRECONDITION, asserted rather than assumed: the root IS expanded and DOES hold
    // children. Without this the refusal below could be passing for the wrong reason.
    let n_children = tree.pool[0].n_children as usize;
    assert!(tree.pool[0].is_expanded() && n_children > 0, "construction: root must be expanded");
    assert!(
        (0..n_children).all(|j| tree.pool[tree.pool[0].first_child as usize + j].n_visits == 0),
        "construction: every child must carry zero visits"
    );

    let err = refuse_zero_visit_export(&tree, 125)
        .expect_err("a search that backed up nothing must not be exportable (M-ZV-1)");
    match err {
        TargetIntegrityError::ZeroVisitSearch { ply_index, n_children: n } => {
            assert_eq!(ply_index, 125, "the ply must ride the error");
            assert_eq!(n, n_children, "the child count must ride the error");
        }
        other => panic!("expected ZeroVisitSearch, got {other}"),
    }
    let text = format!("{err}");
    assert!(text.starts_with("ZeroVisitSearch"), "the variant name must lead the Display: {text}");
    assert!(
        text.contains("did not run"),
        "the Display must say WHAT is wrong, not just that something is: {text}"
    );
}

#[test]
fn a_search_that_backed_up_visits_is_exportable() {
    // The positive control. Without it, `Err(..)` unconditionally would pass the row above
    // and take every self-play run down with it.
    let board = wide_board();
    let mut tree = zero_visit_tree(&board);
    let prior = uniform_prior(&board);
    let mut done = 0;
    while done < 8 {
        let leaves = tree.select_leaves(4)
        .expect("select_leaves: no desync in this fixture");
        if leaves.is_empty() {
            break;
        }
        let n = leaves.len();
        let centers: Vec<(i32, i32)> = leaves.iter().map(mantis_core::Board::window_center).collect();
        tree.expand_and_backup_ls_at(&vec![prior.clone(); n], &vec![0.0f32; n], &centers, TRUNK);
        done += n;
    }

    let backed_up = refuse_zero_visit_export(&tree, 7)
        .expect("a search that visited children must be exportable");
    assert!(backed_up > 0, "the returned total must be the real backed-up count");
}

#[test]
fn an_unexpanded_root_is_refused_with_zero_children() {
    // The degenerate edge: no expansion at all. Unreachable through `play_one_move` (a root
    // that will not expand returns RootExpansionFailed before the exporter), so this is the
    // function's own contract rather than a production drive — stated so the coverage claim
    // is not read as wider than it is.
    let board = wide_board();
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board.clone());
    match refuse_zero_visit_export(&tree, 0) {
        Err(TargetIntegrityError::ZeroVisitSearch { n_children, .. }) => {
            assert_eq!(n_children, 0, "an unexpanded root reports zero children, not garbage");
        }
        other => panic!("expected ZeroVisitSearch on an unexpanded root, got {other:?}"),
    }
}

// ── PIN INDEPENDENCE: the end-to-end drive, with every inference HEALTHY ─────────────

/// Dispersal prefix (the r153 rule — farthest from the window centre), byte-equal to the
/// construction `target_latch_propagation` / `target_wire_carry` use.
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

fn spawn_healthy_graph_producer(
    queue: GraphQueue,
    n_actions: usize,
    served: Arc<AtomicUsize>,
) -> JoinHandle<()> {
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

#[test]
fn the_exporter_pin_stops_a_zero_visit_run_with_the_seam_never_firing() {
    let spec = lookup_or_panic("gnn_axis_v1");
    let n_actions = spec.policy_logit_count;
    let prefix = dispersed_prefix(8);

    // sims=1 + batch=1: the single sim is the root expansion, every child carries 0 visits,
    // and EVERY inference succeeds. Nothing at the seam is wrong — which is the point.
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
    .expect("gnn runner constructs");
    assert!(runner.fatal_defect().is_none(), "fresh runner carries no defect");
    assert_eq!(runner.stats_snapshot().target_integrity_defects, 0);

    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_healthy_graph_producer(runner.graph_producer(), n_actions, served.clone());

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
    let halted = !runner.is_running();
    let snap = runner.stats_snapshot();
    let drained = runner.drain_graph_records();
    runner.stop();
    producer.join().expect("producer exits");

    assert!(served.load(Ordering::Relaxed) > 0, "no graph inference served — vacuous drive");
    let msg = defect.expect(
        "a zero-visit search ran to completion and was exported — the run recorded a target \
         built from priors alone (M-ZV-1/M-ZV-2)",
    );
    assert!(
        msg.contains("ZeroVisitSearch"),
        "the variant name must reach the drain face verbatim: {msg}"
    );
    assert!(halted, "store-then-halt: running must be false once the latch stores (LAW-14)");
    assert_eq!(snap.target_integrity_defects, 1, "the exporter pin's own counter must fire");
    assert_eq!(
        snap.inference_failures_total, 0,
        "the SEAM never fired on this drive — every inference succeeded. A non-zero count \
         here would mean the two pins are not independently reachable, and the isolation \
         this row exists to prove would be an accident of the drive"
    );
    assert!(
        drained.is_empty(),
        "{} record(s) reached the buffer from a search that visited nothing",
        drained.len()
    );
}

/// Dense mock producer: uniform policy rows, always healthy.
fn spawn_healthy_dense_producer(
    queue: DenseQueue,
    stride: usize,
    served: Arc<AtomicUsize>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_batch(4, 5);
        if batch.is_empty() {
            if queue.is_closed() {
                break;
            }
            continue;
        }
        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        let mut flat: Vec<f32> = Vec::with_capacity(ids.len() * stride);
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(ids.len());
        let mut values: Vec<f32> = Vec::with_capacity(ids.len());
        for _ in &ids {
            let start = flat.len();
            flat.extend(std::iter::repeat_n(1.0f32 / stride as f32, stride));
            ranges.push(start..flat.len());
            values.push(0.0);
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        queue.submit_results(&ids, &Arc::new(flat), &ranges, &values);
    })
}

#[test]
fn the_exporter_pin_stops_a_zero_visit_run_on_the_DENSE_arm_too() {
    // The pin's doc claims it is arm-independent, and the call site in `play_one_move` is
    // genuinely ungated — but until this leg that claim was ASSERTED, not driven (cross-model
    // review, NON-BLOCKING 1). The dense arm is the one where a zero-visit export is WORSE:
    // `get_policy` has no prior fallback, so it ships an all-zero row that the dense recorder
    // cannot tell from its legitimate fast-game value-only sentinel. Nothing downstream would
    // ever have caught it.
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 20,
        n_simulations: 1,
        leaf_batch_size: 1,
        standard_sims: 0,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        random_opening_plies: 0,
        encoding_name: Some("v6".to_string()),
        ..Default::default()
    })
    .expect("v6 runner constructs");
    assert_eq!(runner.stats_snapshot().target_integrity_defects, 0);

    let served = Arc::new(AtomicUsize::new(0));
    let producer =
        spawn_healthy_dense_producer(runner.dense_producer(), runner.policy_len(), served.clone());

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
    let halted = !runner.is_running();
    let snap = runner.stats_snapshot();
    runner.stop();
    producer.join().expect("producer exits");

    assert!(served.load(Ordering::Relaxed) > 0, "no dense inference served — vacuous drive");
    let msg = defect.expect(
        "a zero-visit search was exported on the DENSE arm — an all-zero policy row went \
         into the buffer as a value-only sentinel and nothing downstream can distinguish it \
         from a real one",
    );
    assert!(msg.contains("ZeroVisitSearch"), "the variant name must reach the drain face: {msg}");
    assert!(halted, "store-then-halt (LAW-14)");
    assert_eq!(snap.target_integrity_defects, 1, "the exporter pin's own counter must fire");
    assert_eq!(snap.inference_failures_total, 0, "the seam never fired on a healthy drive");
}
