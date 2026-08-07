// R8 >300 justify: the POST-FIX-ONLY oracle bank (DEG x4, S2a, S2b record-level x3, QA,
// O4b latch, CTR gridls producer) shares one construction harness and one feature gate;
// scattering it would scatter the gate IMPL must wire and the freeze audit.
//! ⊕ WP12-R Phase T (TARGET INTEGRITY) — the POST-FIX-ONLY Rust oracle bank.
//! Written at T-2 ORACLE-WRITE, byte-frozen through IMPL.
//!
//! ── GATE (LOUD, enumerated — never a silent skip) ────────────────────────────────────
//! The whole file compiles ONLY under the crate feature `phase_t_postfix`, because it
//! binds the POST-FIX contract that does not exist at HEAD:
//!   * `records::TargetIntegrityError` — enum { MassNotUnity { sum: f64, ply_index,
//!     n_cells }, EmptyTarget { ply_index, n_legal }, VisitSlotsExceeded { n, max,
//!     ply_index } }, Display carrying every field (DESIGN_T §3.3/§3.4);
//!   * `records::record_position_graph(..) -> Result<GraphRecord, TargetIntegrityError>`;
//!   * the runner fatal-defect latch: `SelfPlayRunner::store_fatal_defect(String)` +
//!     `SelfPlayRunner::fatal_defect() -> Option<String>` (store-then-running=false;
//!     the bridge's `collect_graph_data`/drain face raises from this read — §3.4);
//!   * `RunnerStatsSnapshot` LAW-18 counters: `export_offwindow_mass_moves`,
//!     `gridls_zero_policy_rows`, `target_integrity_defects` (§3.6; the third name is
//!     fixed HERE — the design left it unnamed; recorded in ORACLE_NOTES_T.md).
//!
//! IMPL wires the gate by declaring `phase_t_postfix = []` as a DEFAULT feature of
//! mantis-selfplay in the fix commit, so `cargo test --workspace --locked` runs this
//! bank with no invocation change. At HEAD the gate is visibly reported: cargo emits an
//! `unexpected_cfgs` warning naming this exact feature, and ORACLE_NOTES_T.md lists
//! every gated test with pre-fix status "not-compiled-gated".
//!
//! Killers (PREREG_T §3): DEG — M-C; S2a — M-J; S2b record-level — M-D; O4b — M-N;
//! CTR — M-H (per-counter sub-runs).
#![cfg(feature = "phase_t_postfix")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_core::board::Board;
use mantis_core::{Cell, Player};
use mantis_search::{LegalSetPolicy, MCTSTree};
use mantis_selfplay::queues::DenseQueue;
use mantis_selfplay::records::{record_position_graph, TargetIntegrityError};
use mantis_selfplay::replay::hexg::HexgBuffer;

/// Test slot geometry (post-R255: the production value is DERIVED from the sims
/// regime at composition; 128 keeps these frozen oracles' boundary arithmetic
/// unchanged — a test geometry choice, not a shipped tunable).
const MAX_VISITS: usize = 128;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

const NA: usize = 362; // gnn_axis_v1 / v6 policy stride (19*19+1)
const TRUNK: i32 = 19;

/// Three well-separated stones → a wide legal set (>= 140 cells) with a bbox-midpoint
/// window centre; enough cells for the 128/129 boundary constructions.
fn wide_board() -> Board {
    let stones: Vec<((i32, i32), Cell)> =
        vec![((0, 0), Cell::P1), ((8, 0), Cell::P2), ((0, 8), Cell::P1)];
    Board::from_stones(&stones, Player::One, 2, 0, None)
}

/// Two far clusters (records.rs ls_tests::spread_board) — (28,0) is off the global
/// window and covered by cluster-2.
fn spread_board() -> Board {
    let stones: Vec<((i32, i32), Cell)> = (0..5i32)
        .map(|q| ((q, 0), Cell::P1))
        .chain((30..35i32).map(|q| ((q, 0), Cell::P2)))
        .collect();
    Board::from_stones(&stones, Player::One, 2, 0, None)
}

/// ls carrying `masses[i]` at the i-th legal coord (dense/overflow routed by flat).
fn ls_on_first_legal(board: &Board, masses: &[f32]) -> LegalSetPolicy {
    let legal = board.legal_moves();
    assert!(legal.len() >= masses.len(), "board has {} legal, need {}", legal.len(), masses.len());
    let mut ls = LegalSetPolicy { dense: vec![0.0; NA], overflow: Default::default() };
    for (i, &m) in masses.iter().enumerate() {
        let (q, r) = legal[i];
        let flat = board.window_flat_idx(q, r);
        if flat < NA {
            ls.dense[flat] = m;
        } else {
            ls.overflow.insert((q, r), m);
        }
    }
    ls
}

fn record(board: &Board, ls: &LegalSetPolicy) -> Result<mantis_selfplay::replay::hexg::GraphRecord, TargetIntegrityError> {
    record_position_graph(board, ls, TRUNK, 1, 2, 3, true, MAX_VISITS)
}

// ── DEG: the four degenerate constructions, one per §3.3 rev-3 order-arm ─────────────

#[test]
fn deg_4a_half_mass_raises_mass_not_unity() {
    let board = wide_board();
    let ls = ls_on_first_legal(&board, &[0.25, 0.25]); // Σ = 0.5 → order-arm 3
    let err = record(&board, &ls).expect_err("a Σ=0.5 target must be unconstructible");
    match err {
        TargetIntegrityError::MassNotUnity { sum, .. } => {
            assert!((sum - 0.5).abs() < 1e-6, "Display/fields must carry the sum, got {sum}");
        }
        other => panic!("expected MassNotUnity, got {other} — the order-arms drifted"),
    }
    assert!(err_text(&record(&board, &ls)).contains("0.5"), "Display must print the sum");
}

#[test]
fn deg_4b_all_zero_ls_raises_empty_target() {
    let board = wide_board();
    let n_legal = board.legal_moves().len();
    let ls = LegalSetPolicy { dense: vec![0.0; NA], overflow: Default::default() };
    let err = record(&board, &ls).expect_err("an all-zero target must be unconstructible");
    match err {
        TargetIntegrityError::EmptyTarget { n_legal: n, .. } => {
            assert_eq!(n, n_legal, "EmptyTarget must carry the legal-set size");
        }
        other => panic!(
            "expected EmptyTarget (order-arm 2 runs BEFORE unity so the R157 class keeps \
             its OWN variant), got {other}"
        ),
    }
}

#[test]
fn deg_4c_one_nan_among_valid_raises_non_finite_mass_not_unity() {
    let board = wide_board();
    let ls = ls_on_first_legal(&board, &[f32::NAN, 1.0]); // pre-filter scan sees the NaN
    let err = record(&board, &ls)
        .expect_err("a NaN-poisoned target must be unconstructible (order-arm 1)");
    match err {
        TargetIntegrityError::MassNotUnity { sum, .. } => {
            assert!(
                !sum.is_finite(),
                "the PRE-filter legal-scan sum must be poisoned by the NaN the p>0.0 \
                 storage filter would hide (rev-3 N-1), got finite {sum}"
            );
        }
        other => panic!("expected MassNotUnity with a non-finite sum, got {other}"),
    }
    let text = err_text(&record(&board, &ls));
    assert!(text.contains("NaN"), "Display must print the non-finite sum verbatim: {text}");
}

#[test]
fn deg_4c_all_nan_ls_raises_non_finite_mass_not_unity_not_empty_target() {
    let board = wide_board();
    let legal = board.legal_moves();
    let masses = vec![f32::NAN; legal.len().min(64)];
    let ls = ls_on_first_legal(&board, &masses);
    let err = record(&board, &ls).expect_err("an all-NaN target must be unconstructible");
    match err {
        TargetIntegrityError::MassNotUnity { sum, .. } => assert!(!sum.is_finite()),
        other => panic!(
            "expected MassNotUnity (the rev-2 wrong-variant route — empty visits → \
             EmptyTarget — must be gone under the pre-filter basis), got {other}"
        ),
    }
}

fn err_text(res: &Result<mantis_selfplay::replay::hexg::GraphRecord, TargetIntegrityError>) -> String {
    match res {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("expected an error"),
    }
}

// ── S2a: the record carries EVERY ls cell by coord (map equality) ────────────────────

#[test]
fn s2a_record_carries_every_ls_cell_by_coord() {
    let board = spread_board();
    let legal = board.legal_moves();
    assert!(legal.contains(&(28, 0)));
    let in_cell = *legal
        .iter()
        .find(|&&(q, r)| board.window_flat_idx(q, r) < NA)
        .expect("in-window legal cell");
    let mut ls = LegalSetPolicy { dense: vec![0.0; NA], overflow: Default::default() };
    ls.dense[board.window_flat_idx(in_cell.0, in_cell.1)] = 0.3;
    ls.overflow.insert((28, 0), 0.7); // off-window mass — M-J's exact victim

    let rec = record(&board, &ls).expect("a full-mass target must record");
    let got: std::collections::HashMap<(i16, i16), f32> =
        rec.visits.iter().map(|&(q, r, p)| ((q, r), p)).collect();
    assert_eq!(got.len(), 2, "exactly the two nonzero cells stored, got {:?}", rec.visits);
    assert!(
        (got[&(in_cell.0 as i16, in_cell.1 as i16)] - 0.3).abs() < 1e-6,
        "in-window cell mass must ride by coord"
    );
    assert!(
        (got[&(28i16, 0i16)] - 0.7).abs() < 1e-6,
        "off-window cell mass must ride by coord — the M-J read-skip loses exactly this"
    );
}

// ── S2b record-level: the MAX_VISITS boundary (flip-set row 5) ───────────────────────

#[test]
fn s2b_admits_exactly_128_mass_cells() {
    let board = wide_board();
    let masses = vec![1.0f32 / 128.0; 128];
    let ls = ls_on_first_legal(&board, &masses);
    let rec = record(&board, &ls).expect("128 cells == MAX_VISITS must be admitted");
    assert_eq!(rec.visits.len(), 128, "all 128 cells stored, none truncated");
}

#[test]
fn s2b_refuses_129_mass_cells_with_the_typed_error() {
    let board = wide_board();
    let masses = vec![1.0f32 / 129.0; 129];
    let ls = ls_on_first_legal(&board, &masses);
    let err = record(&board, &ls)
        .expect_err("129 cells must raise — the silent top-k truncation is deleted");
    match err {
        TargetIntegrityError::VisitSlotsExceeded { n, max, .. } => {
            assert_eq!(n, 129, "the error must carry the offending count");
            assert_eq!(max, 128, "the error must carry the capacity the record was built against");
        }
        other => panic!("expected VisitSlotsExceeded, got {other}"),
    }
}

#[test]
fn s2b_zero_visit_prior_fallback_on_a_wide_root_raises_visit_slots_exceeded() {
    // The §3.4 stated-reachable raise (predicted RED-as-designed in PREREG): a zero-visit
    // root with > 128 children ships the prior fallback over its FULL child set (§3.1);
    // the record guard must refuse it LOUD — a >128-child prior dump under sims >= 2
    // evidences inference failure, and raising is CORRECT.
    let board = wide_board();
    let legal = board.legal_moves();
    assert!(legal.len() > 192, "need >192 legal so the expand caps at 192 children");
    let p = 1.0f32 / legal.len() as f32;
    let mut prior = LegalSetPolicy { dense: vec![0.0; NA], overflow: Default::default() };
    for &(q, r) in &legal {
        let flat = board.window_flat_idx(q, r);
        if flat < NA {
            prior.dense[flat] = p;
        } else {
            prior.overflow.insert((q, r), p);
        }
    }
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board.clone());
    let leaves = tree.select_leaves(1);
    assert_eq!(leaves.len(), 1);
    let centers = vec![board.window_center()];
    tree.expand_and_backup_ls_at(&[prior], &[0.0f32], &centers, TRUNK);
    assert!(tree.pool[0].n_children as usize > 128, "root must hold >128 children");

    let ls = tree.get_policy_ls(1.0, NA); // zero-visit → prior fallback, >128 cells
    let err = record(&board, &ls).expect_err("a >128-cell prior fallback must raise");
    match err {
        TargetIntegrityError::VisitSlotsExceeded { n, max, .. } => {
            assert!(n > max && max == 128, "n {n} must exceed max {max} == 128");
        }
        other => panic!("expected VisitSlotsExceeded, got {other}"),
    }
}

// ── QA: quick-arm parity — export is flag-independent, flag rides the buffer ─────────
// Construction is IN-WINDOW-ONLY (out of M-J's reach — the T-2 reconciliation: the
// off-window-mass burden is O1r's; asserted as a precondition below).

#[test]
fn qa_is_full_search_flag_rides_and_the_target_is_flag_independent() {
    let stones: Vec<((i32, i32), Cell)> =
        vec![((0, 0), Cell::P1), ((2, 0), Cell::P2), ((0, 2), Cell::P1)];
    let board = Board::from_stones(&stones, Player::One, 2, 0, None);
    let legal = board.legal_moves();
    assert!(
        legal.iter().all(|&(q, r)| board.window_flat_idx(q, r) < NA),
        "QA precondition: compact board must be fully in-window (out of M-J's reach)"
    );
    // A real (tiny) search so the export is a genuine visit distribution.
    let p = 1.0f32 / legal.len() as f32;
    let mut prior = LegalSetPolicy { dense: vec![0.0; NA], overflow: Default::default() };
    for &(q, r) in &legal {
        prior.dense[board.window_flat_idx(q, r)] = p;
    }
    let mut tree = MCTSTree::new(1.5);
    tree.new_game(board.clone());
    let centers = vec![board.window_center()];
    let mut done = 0;
    while done < 16 {
        let boards = tree.select_leaves(4);
        if boards.is_empty() {
            break;
        }
        let n = boards.len();
        let cs: Vec<(i32, i32)> = boards.iter().map(|b| b.window_center()).collect();
        tree.expand_and_backup_ls_at(&vec![prior.clone(); n], &vec![0.0f32; n], &cs, TRUNK);
        done += n;
        let _ = &centers;
    }
    let ls = tree.get_policy_ls(1.0, NA);

    // The SAME export recorded under both arms: get_policy_ls ran before the flag exists
    // (search_drive.rs:709,716-724) — the fix is arm-independent (PROVENANCE_T0 §2).
    let rec_full = record_position_graph(&board, &ls, TRUNK, 1, 2, 3, true, MAX_VISITS)
        .expect("full-arm record");
    let rec_quick = record_position_graph(&board, &ls, TRUNK, 1, 2, 3, false, MAX_VISITS)
        .expect("quick-arm record");
    assert_eq!(rec_full.visits, rec_quick.visits, "the visit target must be flag-independent");
    let sum: f64 = rec_quick.visits.iter().map(|&(_, _, p)| f64::from(p)).sum();
    assert!((sum - 1.0).abs() < 1e-4, "the is_full_search=false row must carry FULL mass");
    assert!(rec_full.is_full_search && !rec_quick.is_full_search);

    // Buffer round-trip: the flag rides push → record_at verbatim.
    let mut buf = HexgBuffer::new(4, "gnn_axis_v1", 128).expect("graph buffer");
    buf.push_record_impl(&rec_full, 1).expect("push full");
    buf.push_record_impl(&rec_quick, 2).expect("push quick");
    assert!(buf.record_at(0).is_full_search);
    assert!(!buf.record_at(1).is_full_search);
    assert_eq!(buf.record_at(0).visits, buf.record_at(1).visits);
}

// ── O4b + CTR(latch): the fatal-defect latch, red-side ───────────────────────────────

#[test]
fn o4b_latch_stores_the_named_variant_and_halts_the_runner() {
    // Random-only runner (drain_shutdown drive-2 shape) so is_running() is genuinely
    // true when the latch fires.
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 50,
        n_simulations: 1,
        leaf_batch_size: 1,
        fast_sims: 1,
        standard_sims: 1,
        quiescence_enabled: false,
        dirichlet_enabled: false,
        random_opening_plies: 50,
        encoding_name: Some("v6".to_string()),
        ..Default::default()
    })
    .expect("runner constructs");
    assert!(runner.fatal_defect().is_none(), "fresh runner carries no defect");
    assert_eq!(
        runner.stats_snapshot().target_integrity_defects,
        0,
        "the latch fire-count must be VISIBLE at 0 when idle (LAW-18)"
    );

    runner.start();
    assert!(runner.is_running());
    let err = TargetIntegrityError::MassNotUnity { sum: 0.5, ply_index: 3, n_cells: 7 };
    runner.store_fatal_defect(err.to_string());

    assert!(
        !runner.is_running(),
        "store_fatal_defect must flip running=false (store-then-halt, §3.4 — a worker \
         panic is NOT sufficient: runner/mod.rs swallows joins)"
    );
    let msg = runner
        .fatal_defect()
        .expect("the drain face must be able to read the stored defect");
    assert!(
        msg.contains("MassNotUnity"),
        "the VARIANT NAME must reach the supervisor-facing surface (M-N kills this): {msg}"
    );
    assert!(msg.contains("0.5"), "the Display fields must ride: {msg}");
    assert_eq!(runner.stats_snapshot().target_integrity_defects, 1, "latch fire-count == 1");
    runner.stop();
}

// ── CTR(gridls): the §3.5 zero-row counter has a live producer (LAW-07) ──────────────

const MOCK_NN_SEED: u64 = 0x4D4F_434B_4E4E_0007;

fn splitmix64_step(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn spawn_dense_producer(queue: DenseQueue, stride: usize, served: Arc<AtomicUsize>) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_batch(2, 5);
        if batch.is_empty() {
            if queue.is_closed() {
                break;
            }
            continue;
        }
        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        let mut flat: Vec<f32> = Vec::new();
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(batch.len());
        let mut values: Vec<f32> = Vec::with_capacity(batch.len());
        for (id, feats) in &batch {
            let mut s = MOCK_NN_SEED ^ *id;
            for &x in feats {
                s ^= u64::from(f32::to_bits(x));
                splitmix64_step(&mut s);
            }
            let start = flat.len();
            for _ in 0..stride {
                let step = splitmix64_step(&mut s);
                flat.push((step >> 40) as f32 / 16_777_216.0_f32);
            }
            ranges.push(start..flat.len());
            values.push(0.0);
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        let arc = Arc::new(flat);
        queue.submit_results(&ids, &arc, &ranges, &values);
    })
}

#[test]
fn ctr_gridls_zero_policy_rows_fires_on_a_dispersed_ls_run() {
    // v6_live2_ls (the R148 control arm): a 40-ply line-dispersed seed prefix gives
    // multi-cluster boards whose far windows see zero visit mass at 8 sims — the §3.5
    // zero-row fill fires on recorded cluster rows and the LAW-18 counter must count it.
    let spec = mantis_encoding::lookup_or_panic("v6_live2_ls");
    let geom = mantis_core::board::BoardGeometry {
        legal_move_radius: spec.legal_move_radius as i32,
        cluster_threshold: spec.cluster_threshold.unwrap_or(5) as i32,
        cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
    };
    let mut b = Board::with_geometry(geom);
    let mut prefix = Vec::new();
    for _ in 0..40 {
        let legal = b.legal_moves();
        if legal.is_empty() {
            break;
        }
        let (cq, cr) = b.window_center();
        let &(q, r) = legal
            .iter()
            .max_by_key(|&&(q, r): &&(i32, i32)| {
                let (dq, dr) = (q - cq, r - cr);
                dq.abs().max(dr.abs()).max((dq + dr).abs())
            })
            .unwrap();
        if b.apply_move(q, r).is_err() {
            break;
        }
        prefix.push((q, r));
    }

    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 46,
        n_simulations: 8,
        leaf_batch_size: 4,
        standard_sims: 0,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        encoding_name: Some("v6_live2_ls".to_string()),
        seed_fraction: 1.0,
        seed_corpus: Some(vec![prefix]),
        ..Default::default()
    })
    .expect("v6_live2_ls runner constructs");
    assert_eq!(
        runner.stats_snapshot().gridls_zero_policy_rows,
        0,
        "the counter must be VISIBLE at 0 before any position records (LAW-18 idle posture)"
    );
    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_dense_producer(runner.dense_producer(), runner.policy_len(), served.clone());

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut fired = 0u64;
    while Instant::now() < deadline {
        let snap = runner.stats_snapshot();
        fired = snap.gridls_zero_policy_rows;
        if fired >= 1 && snap.positions_generated >= 3 {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    runner.stop();
    producer.join().expect("producer exits");
    assert!(served.load(Ordering::Relaxed) > 0, "no inference served — vacuous drive");
    assert!(
        fired >= 1,
        "gridls_zero_policy_rows never fired across the dispersed drive — the §3.5 \
         zero-row producer is not wired to its LAW-18 counter (M-H kills this)"
    );
}
