//! A runner stopped mid-game must NOT push false-draw rows.
//!
//! On `stop()` an in-progress game is DROPPED before finalize, never recorded as an organic
//! draw. A false draw is the signature `terminal_reason == 3` (winner None AND plies <
//! max_moves), which under these regimes can ONLY come from a leaked partial-game finalize.
//! Two drives — full MCTS with a mock inference producer (shutdown with inference in flight),
//! and random-only with no producer — plus a bite proof that the checker flags an injected row.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::SearchKind;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::{GameResultRow, SelfPlayRunner, SelfPlayRunnerConfig};

const ENCODING: &str = "gnn_axis_r8";

/// `terminal_reason` is field 4 of `GameResultRow`; `3` = organic draw.
fn has_false_draw(rows: &[GameResultRow]) -> bool {
    rows.iter().any(|r| r.4 == 3)
}

const MOCK_NN_SEED: u64 = 0x4D4F_434B_4E4E_0006;

fn splitmix64_step(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic policy+value from the leaf's legal coords, NORMALIZED (the segmented-softmax
/// invariant `assemble_ls_from_gnn_probs` checks). The values are irrelevant to the shutdown
/// invariant; only that the producer keeps the search fed so workers are genuinely mid-game.
fn mock_graph_infer(coords: &[(i32, i32)], seed: u64) -> (Vec<f32>, f32) {
    let mut s = seed;
    for &(q, r) in coords {
        s ^= (q as u32) as u64 | ((r as u32) as u64) << 32;
        splitmix64_step(&mut s);
    }
    let mut raw = Vec::with_capacity(coords.len());
    for _ in 0..coords.len() {
        let step = splitmix64_step(&mut s);
        // Strictly positive so the normalization below can never divide by zero.
        raw.push((step >> 40) as f32 / 16_777_216.0_f32 + 1.0e-3);
    }
    let total: f32 = raw.iter().sum();
    let probs: Vec<f32> = raw.iter().map(|p| p / total).collect();
    let vstep = splitmix64_step(&mut s);
    let value = ((vstep % 2_000_001) as i64 - 1_000_000) as f32 / 1_000_000.0_f32;
    (probs, value)
}

/// Spawn a mock producer that serves the graph queue until `stop()` closes it.
///
/// `served` counts requests actually served — strictly positive proves a worker was genuinely
/// mid-MCTS-search, the signal that de-vacuums the drain-shutdown oracle below.
fn spawn_graph_producer(
    queue: GraphQueue,
    n_actions: usize,
    served: Arc<AtomicUsize>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(2, 5);
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
            let (probs, value) = mock_graph_infer(&coords, MOCK_NN_SEED ^ id);
            ids.push(id);
            results.push(
                assemble_ls_from_gnn_probs(n_actions, &probs, &g.policy_scatter_index.0, &coords)
                    .map(|ls| (ls, value)),
            );
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        queue.submit_graph_results(&ids, results);
    })
}

#[test]
fn mcts_drive_with_mock_producer_stop_midgame_no_false_draws() {
    let cfg = SelfPlayRunnerConfig {
        n_workers: 2,
        max_moves_per_game: 30,
        n_simulations: 8,
        leaf_batch_size: 4,
        fast_sims: 8,
        standard_sims: 8,
        search_kind: SearchKind::Puct,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        random_opening_plies: 0,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    };
    let runner = SelfPlayRunner::new(cfg).expect("the graph MCTS runner must construct");
    let n_actions = lookup_or_panic(ENCODING).policy_logit_count;
    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_graph_producer(runner.graph_producer(), n_actions, served.clone());

    runner.start();
    assert!(runner.is_running(), "runner is running after start()");
    // Poll the SERVED counter rather than sleeping a guessed interval: a worker zeroes a 4M-node
    // pool before its first search, so any fixed window lands on the boot. The deadline is a
    // liveness bound, not a tuning knob.
    let deadline = Instant::now() + Duration::from_secs(30);
    while served.load(Ordering::Relaxed) == 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    runner.stop(); // flips running, closes queues (wakes waiters), joins workers

    producer
        .join()
        .expect("mock producer exits once the queue is closed");
    assert!(!runner.is_running(), "runner stopped");

    let drained = runner.drain_game_results();

    // De-vacuum: served batches prove a worker was mid-MCTS-search when `stop()` fired, so the
    // short-circuit was on a LIVE mid-game path rather than an empty drain.
    let inferences_served = served.load(Ordering::Relaxed);
    assert!(
        inferences_served >= 1,
        "no inference was served — workers never reached mid-MCTS-search, so the \
         drain-shutdown oracle would be vacuous (served={inferences_served})",
    );
    assert!(
        !has_false_draw(&drained),
        "P22 violated: a mid-game stop with inference in flight leaked an organic-draw \
         (terminal_reason==3) row",
    );
}

fn random_only_runner(max_moves: usize) -> SelfPlayRunner {
    SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 4,
        max_moves_per_game: max_moves,
        n_simulations: 1,
        leaf_batch_size: 1,
        fast_sims: 1,
        standard_sims: 1,
        quiescence_enabled: false,
        quiescence_blend_2: 0.0,
        dirichlet_enabled: false,
        random_opening_plies: max_moves as u32, // == max_moves → never MCTS
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("random-only runner must construct")
}

#[test]
fn random_only_stop_midgame_no_false_draws() {
    let runner = random_only_runner(50);
    let _baseline = runner.drain_game_results();

    runner.start();

    // De-vacuum: wait (bounded) until a game has COMPLETED, so the runner is provably driving
    // games through the live finalize path; an empty drain would pass `!has_false_draw` vacuously.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut games: Vec<GameResultRow> = Vec::new();
    while Instant::now() < deadline {
        games.extend(runner.drain_game_results());
        if !games.is_empty() {
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert!(
        !games.is_empty(),
        "random-only runner must complete >=1 game (reason-2 ply-cap) — an empty drain \
         would make the false-draw oracle vacuous (games_completed==0)",
    );

    // Workers churn games back-to-back, so a worker is mid-game at the stop instant; without the
    // short-circuit that partial game would finalize as reason==3.
    runner.stop();
    games.extend(runner.drain_game_results());

    assert!(
        !has_false_draw(&games),
        "P22 violated: a mid-game stop pushed an organic-draw (terminal_reason==3) row \
         (partial-game injection). max_moves=50 — a natural reason-3 is impossible under \
         random play (legal moves never empty)",
    );
}

#[test]
fn false_draw_checker_bites_on_injected_reason_3() {
    // A synthetic in-progress finalize: winner=None, plies < max_moves → terminal_reason == 3.
    let injected: GameResultRow = (17, 0, Vec::new(), 0, 3, 0, 0, 0);
    assert!(
        has_false_draw(&[injected]),
        "the false-draw checker MUST flag an injected terminal_reason==3 tuple",
    );
    // A clean set is NOT flagged: the checker is specific to the false-draw signature.
    let clean: Vec<GameResultRow> = vec![
        (10, 0, Vec::new(), 0, 2, 0, 0, 0), // ply-cap
        (11, 1, Vec::new(), 1, 0, 0, 0, 0), // six-in-a-row win
    ];
    assert!(
        !has_false_draw(&clean),
        "the checker must not flag legitimate terminals"
    );
}
