//! P-03 — inv26 re-anchor: `ply_cap_value` distinct from `draw_reward` (§178),
//! driven end-to-end through the native `start()/drain_game_results()/stop()`
//! (RE-ANCHOR of `inv26_ply_cap_value.rs`).
//!
//! This file drives the runner in RANDOM-ONLY mode (`random_opening_plies ==
//! max_moves_per_game`) so every move is a random opening pick — no MCTS, no
//! inference producer needed (the frozen strategy). `max_moves = 10` makes the
//! ply-cap DETERMINISTIC: a player's 6th stone lands no earlier than ply 10 (the
//! opening single-stone turn then 2-stone compound turns), so with ≤ 5 stones per
//! player a 6-in-a-row is STRUCTURALLY impossible → `winner()` is always `None` →
//! every completed game terminates at the ply-cap (`terminal_reason == 2`). No
//! reliance on the unseeded worker RNG avoiding a lucky win.
//!
//! The drain tuple exposes `terminal_reason` but NOT the per-row `outcome`, so the
//! §178 VALUE branch (`outcome == ply_cap_value`) is pinned by the focused
//! `finalize_game` unit test IN-SRC (`runner/finalize.rs`, mod
//! `inv26_finalize_outcome_tests`) — the branch this drain-tuple oracle cannot see.

use std::time::{Duration, Instant};

use mantis_selfplay::runner::{GameResultRow, SelfPlayRunner, SelfPlayRunnerConfig};

/// Random-only runner: `random_opening_plies == max_moves` → never MCTS.
fn random_only_runner(max_moves: usize, draw_reward: f32, ply_cap_value: f32) -> SelfPlayRunner {
    SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 2,
        max_moves_per_game: max_moves,
        n_simulations: 1, // irrelevant — no MCTS
        leaf_batch_size: 1,
        fast_sims: 1,
        standard_sims: 1,
        draw_reward,
        ply_cap_value,
        quiescence_enabled: false,
        quiescence_blend_2: 0.0,
        dirichlet_enabled: false,
        random_opening_plies: max_moves as u32, // == max_moves → never MCTS
        encoding_name: Some("v6".to_string()),
        ..Default::default()
    })
    .expect("random-only runner must construct")
}

/// Drive until `min_games` complete or `timeout` fires; return every drained
/// `(plies, terminal_reason)`.
fn drive_to_completion(
    runner: &SelfPlayRunner,
    min_games: usize,
    timeout: Duration,
) -> Vec<(usize, u8)> {
    runner.start();
    let deadline = Instant::now() + timeout;
    let mut games = Vec::new();
    while Instant::now() < deadline {
        for row in runner.drain_game_results() {
            games.push((row.0, row.4)); // (plies, terminal_reason)
        }
        if games.len() >= min_games {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    runner.stop();
    for row in runner.drain_game_results() {
        games.push((row.0, row.4));
    }
    games
}

/// Cell 1 — split values reachable; ply-cap path confirmed with `ply_cap_value`
/// DISTINCT from `draw_reward`. Every random-play game hits `terminal_reason == 2`.
#[test]
fn ply_cap_value_distinct_from_draw_reward_every_game_reason_2() {
    let runner = random_only_runner(10, -0.1, -0.5);
    let games = drive_to_completion(&runner, 4, Duration::from_secs(5));
    assert!(
        !games.is_empty(),
        "at least one game must complete (5s, max_moves=10, random-only)",
    );
    for (plies, reason) in &games {
        assert_eq!(
            *reason, 2,
            "random play with max_moves=10 must hit ply-cap (reason 2); got reason={reason} \
             plies={plies}",
        );
    }
}

/// Cell 4 — back-compat: `ply_cap_value == draw_reward` (both `-0.1`) reaches the
/// same ply-cap path. `if terminal_reason == 2 { v } else { v } ≡ v` by
/// construction; the drain confirms the branch is still reached.
#[test]
fn back_compat_when_ply_cap_value_equals_draw_reward() {
    let runner = random_only_runner(10, -0.1, -0.1);
    let games = drive_to_completion(&runner, 4, Duration::from_secs(5));
    assert!(!games.is_empty(), "at least one game must complete");
    for (plies, reason) in &games {
        assert_eq!(
            *reason, 2,
            "back-compat: ply-cap still reached (reason 2); got reason={reason} plies={plies}",
        );
    }
}

/// A random-only ply-cap regime must NEVER produce an organic draw (`reason == 3`)
/// nor a legal-move-exhaustion break — the whole point of the ply-cap oracle is
/// that `reason == 2` is the only terminal it can reach. (This also guards the
/// GameResultRow tuple field order used above: `terminal_reason` is field 4.)
#[test]
fn ply_cap_regime_never_organic_draws() {
    let runner = random_only_runner(10, -0.1, -0.5);
    let games = drive_to_completion(&runner, 4, Duration::from_secs(5));
    for (_plies, reason) in &games {
        assert_ne!(
            *reason, 3,
            "random-only ply-cap regime must not organic-draw"
        );
        assert_ne!(
            *reason, 0,
            "≤5 stones/player cannot 6-in-a-row (no six-in-a-row win)"
        );
        assert_ne!(
            *reason, 1,
            "≤5 stones/player cannot win (no colony win either)"
        );
    }
    // Compile-time anchor: GameResultRow is the drain carrier tuple.
    let _: fn() -> Vec<GameResultRow> = || Vec::new();
}
