//! P-05 — drain-shutdown: a runner stopped mid-game must NOT push false-draw rows
//! (RE-ANCHOR of `test_drain_shutdown_no_false_draws.rs`, LOCKED DECISION 12).
//!
//! On `stop()`: `running` flips false, BOTH inference queues close (waking blocked
//! waiters with `Err`), workers join. An IN-PROGRESS game hits the §P22
//! short-circuit (`game.rs`: `if !running { return; }` after the move loop) and is
//! DROPPED before finalize — it is NEVER recorded as an organic draw. A false draw
//! is the signature `terminal_reason == 3` (winner None AND plies < max_moves),
//! which under this regime can ONLY appear from a leaked partial-game finalize.
//!
//! Two drives:
//!   1. FULL MCTS with a MOCK inference producer thread (D16 / CAPTURE_LOG C-10):
//!      workers block in `submit_batch_and_wait`; `stop()` closes the queue, wakes
//!      them with `Err`, the worker skips the batch, the move loop sees
//!      `running=false` and breaks, §P22 drops the game. Exercises the realistic
//!      "shutdown with inference in flight" path.
//!   2. RANDOM-ONLY (no producer): the deterministic frozen re-anchor — §P22 fires
//!      for ANY move-loop exit with `running=false`, including a random-opening break.
//!
//! Plus the LAW-07 bite proof: the false-draw checker MUST flag an injected
//! `terminal_reason == 3` tuple (a checker that passes it is a test failure).

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_selfplay::queues::DenseQueue;
use mantis_selfplay::runner::{GameResultRow, SelfPlayRunner, SelfPlayRunnerConfig};

// ── the false-draw checker (the thing under test; the bite proof feeds it) ──────
/// `terminal_reason` is field 4 of `GameResultRow`; `3` = organic draw. A leaked
/// partial-game finalize is the only way it can appear under these regimes.
fn has_false_draw(rows: &[GameResultRow]) -> bool {
    rows.iter().any(|r| r.4 == 3)
}

// ── deterministic MOCK NN (CAPTURE_LOG C-10 / PREREG) ───────────────────────────
const MOCK_NN_SEED: u64 = 0x4D4F_434B_4E4E_0006;

fn splitmix64_step(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic policy+value from the request features (C-10): fold each feature
/// scalar into the splitmix stream, then emit `policy_stride` logits in `[0,1)` and
/// a value in `[-1,1]`. The exact values are irrelevant to the shutdown invariant —
/// only that the producer keeps the search fed so workers are genuinely mid-game.
fn mock_dense_infer(features: &[f32], policy_stride: usize, seed: u64) -> (Vec<f32>, f32) {
    let mut s = seed;
    for &x in features {
        s ^= u64::from(f32::to_bits(x));
        splitmix64_step(&mut s);
    }
    let mut policy = Vec::with_capacity(policy_stride);
    for _ in 0..policy_stride {
        let step = splitmix64_step(&mut s);
        policy.push((step >> 40) as f32 / 16_777_216.0_f32);
    }
    let vstep = splitmix64_step(&mut s);
    let value = ((vstep % 2_000_001) as i64 - 1_000_000) as f32 / 1_000_000.0_f32;
    (policy, value)
}

/// Spawn a mock producer that pops the dense queue and submits C-10 results until
/// the queue is closed (by `stop()`). A small `max` gives a saturation threshold of
/// 1, so the pop returns as soon as a request is present (low latency, no spin).
///
/// `served` counts the inference requests actually served — a strictly-positive
/// value proves a worker was genuinely mid-MCTS-search (a leaf batch in flight), the
/// "worker mid-game" signal that de-vacuums the drain-shutdown oracle below.
fn spawn_dense_producer(queue: DenseQueue, policy_stride: usize, served: Arc<AtomicUsize>) -> JoinHandle<()> {
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
        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(batch.len());
        let mut values: Vec<f32> = Vec::with_capacity(batch.len());
        for (id, feats) in &batch {
            let (policy, value) = mock_dense_infer(feats, policy_stride, MOCK_NN_SEED ^ *id);
            let start = flat.len();
            flat.extend_from_slice(&policy);
            ranges.push(start..flat.len());
            values.push(value);
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        let arc = Arc::new(flat);
        queue.submit_results(&ids, &arc, &ranges, &values);
    })
}

// ── Drive 1: full MCTS + mock producer, stop mid-game ───────────────────────────
#[test]
fn mcts_drive_with_mock_producer_stop_midgame_no_false_draws() {
    let cfg = SelfPlayRunnerConfig {
        n_workers: 2,
        max_moves_per_game: 30,
        n_simulations: 8,
        leaf_batch_size: 4,
        fast_sims: 8,
        standard_sims: 8,
        gumbel_mcts: false,
        dirichlet_enabled: false,
        quiescence_enabled: false,
        completed_q_values: false,
        random_opening_plies: 0,
        encoding_name: Some("v6".to_string()),
        ..Default::default()
    };
    let runner = SelfPlayRunner::new(cfg).expect("v6 MCTS runner must construct");
    let policy_stride = runner.policy_len();
    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_dense_producer(runner.dense_producer(), policy_stride, served.clone());

    runner.start();
    assert!(runner.is_running(), "runner is running after start()");
    // Let workers get well into games (inference in flight) before shutdown.
    thread::sleep(Duration::from_millis(80));
    runner.stop(); // flips running, closes queues (wakes waiters), joins workers

    producer.join().expect("mock producer exits once the queue is closed");
    assert!(!runner.is_running(), "runner stopped");

    let drained = runner.drain_game_results();

    // POSITIVE (de-vacuum): the mock producer served real inference batches, so a
    // worker was genuinely mid-MCTS-search (a leaf batch in flight) when `stop()`
    // fired — the §P22 short-circuit was on a LIVE mid-game path. Were §P22 removed,
    // that in-progress game would finalize as an organic draw (winner None, plies <
    // max_moves ⇒ terminal_reason==3), which the checker below would then flag. An
    // all-ply-cap / empty drain no longer passes this test vacuously.
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

// ── Drive 2: random-only (no producer), the deterministic frozen re-anchor ───────
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
        encoding_name: Some("v6".to_string()),
        ..Default::default()
    })
    .expect("random-only runner must construct")
}

#[test]
fn random_only_stop_midgame_no_false_draws() {
    let runner = random_only_runner(50);
    let _baseline = runner.drain_game_results();

    runner.start();

    // POSITIVE (de-vacuum): wait (bounded) until at least one game has COMPLETED, so
    // the runner is provably driving games through the live finalize path — every
    // completed random-only game is a reason-2 ply-cap. This is deterministic (it
    // WAITS rather than assuming a 30 ms window), unlike the old fixed sleep whose
    // drain could be empty and pass `!has_false_draw` vacuously.
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

    // Workers are now churning games back-to-back, so a worker is mid-game at the stop
    // instant, exercising the §P22 short-circuit (a random-opening break with
    // running=false). Were §P22 removed, that partial game would finalize as reason==3.
    runner.stop();
    games.extend(runner.drain_game_results());

    assert!(
        !has_false_draw(&games),
        "P22 violated: a mid-game stop pushed an organic-draw (terminal_reason==3) row \
         (partial-game injection). max_moves=50 — a natural reason-3 is impossible under \
         random play (legal moves never empty)",
    );
}

// ── LAW-07 bite proof: the checker MUST flag an injected reason-3 tuple ─────────
#[test]
fn false_draw_checker_bites_on_injected_reason_3() {
    // A synthetic in-progress finalize would push winner=None, plies < max_moves →
    // terminal_reason == 3. If the checker passed this, the drain-shutdown oracle
    // would be vacuous.
    let injected: GameResultRow = (17, 0, Vec::new(), 0, 3, 0, 0, 0, 0, 0);
    assert!(
        has_false_draw(&[injected]),
        "the false-draw checker MUST flag an injected terminal_reason==3 tuple",
    );
    // And a clean set (only ply-cap reason 2 + a six-in-a-row win reason 0) is NOT
    // flagged — the checker is specific to the false-draw signature.
    let clean: Vec<GameResultRow> = vec![
        (10, 0, Vec::new(), 0, 2, 0, 0, 0, 0, 0), // ply-cap
        (11, 1, Vec::new(), 1, 0, 0, 0, 0, 0, 0), // six-in-a-row win
    ];
    assert!(!has_false_draw(&clean), "the checker must not flag legitimate terminals");
}
