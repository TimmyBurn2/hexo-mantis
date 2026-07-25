//! P-01 — inv19 re-anchor: native `SelfPlayRunnerConfig` field→slot one-to-one +
//! distinct-sentinel round-trip (RE-ANCHOR of
//! `inv19_selfplayrunner_config_builder_byte_equivalence.rs`).
//!
//! The frozen inv19 pinned the 38-positional pyo3 ctor. WP6 drops that ctor to
//! WP7 and the config is now a plain-Rust struct; this re-anchor pins the SURVIVING
//! surface:
//!   1. every field maps to exactly one slot — a full struct literal + an
//!      EXHAUSTIVE destructure (no `..`) means a dropped / added / renamed field
//!      fails to COMPILE, and distinct per-field sentinels catch any cross-wire;
//!   2. **NO per-game radius-jitter field** (D7 KILL) and **NO
//!      `feature_len`/`policy_len` override fields** (C-1) — the exhaustive
//!      binding set names exactly the 45 live fields, so a resurrected jitter knob
//!      (or a caller-supplied shape) would break this test loudly;
//!   3. the `Default` impl is TEST-SCAFFOLDING, NOT the config authority (R1 — the
//!      authoritative defaults live in the WP8 Python schema). This file does NOT
//!      assert Default's values are "the" defaults; it asserts the opposite — a
//!      bare `Default` is not a usable config authority (LAW-11).

use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// f32 approx-equality (dodges `clippy::float_cmp`; exact for a literal round-trip).
fn feq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-9
}

/// Every field set to a distinct, non-default sentinel so a swap / alias / drop
/// manifests as a field-equality failure. Constructed as a FULL struct literal (no
/// `..Default::default()`) so a field addition / removal fails to compile.
fn distinct_sentinels() -> SelfPlayRunnerConfig {
    SelfPlayRunnerConfig {
        n_workers: 7,
        max_moves_per_game: 77,
        n_simulations: 88,
        leaf_batch_size: 3,
        c_puct: 2.5,
        fpu_reduction: 0.125,
        fast_prob: 0.375,
        fast_sims: 37,
        standard_sims: 42,
        temp_threshold_compound_moves: 21,
        draw_reward: -0.75,
        ply_cap_value: -0.875, // §178 — DISTINCT from draw_reward sentinel
        quiescence_enabled: false,
        quiescence_blend_2: 0.625,
        temp_min: 0.0625,
        zoi_enabled: true,
        zoi_lookback: 24,
        zoi_margin: 9,
        completed_q_values: true,
        c_visit: 37.5,
        c_scale: 1.25,
        gumbel_mcts: true,
        gumbel_m: 12,
        gumbel_explore_moves: 7,
        dirichlet_alpha: 0.4,
        dirichlet_epsilon: 0.3,
        dirichlet_enabled: false,
        results_queue_cap: 20_000,
        full_search_prob: 0.5,
        n_sims_quick: 50,
        n_sims_full: 100,
        random_opening_plies: 3,
        selfplay_rotation_enabled: true,
        encoding_name: Some("v6w25".to_string()),
        inference_pool_size: Some(4096),
        forced_win_policy_enabled: true,
        forced_win_policy_depth: 5,
        forced_win_policy_weight: 2.5,
        solver_enabled: true,
        solver_depth: 11,
        solver_node_budget: 7777,
        solver_neighbor_dist: 4,
        solver_visit_weight: 0.9,
        seed_fraction: 0.33,
        seed_corpus: Some(vec![vec![(1, 2), (3, 4)]]),
    }
}

/// Test 1 — every field → exactly one slot; the field surface is EXACTLY the 45
/// live fields (no jitter, no feature_len/policy_len). The exhaustive destructure
/// (no `..`) is the compile-time completeness guard; the sentinel asserts are the
/// no-cross-wire guard.
#[test]
fn every_field_maps_to_exactly_one_slot_and_no_killed_fields() {
    let cfg = distinct_sentinels();
    // EXHAUSTIVE destructure — a `..` is DELIBERATELY absent. A resurrected
    // per-game radius-jitter field, a re-added `feature_len`/`policy_len`, or any
    // dropped field breaks this line at COMPILE time.
    let SelfPlayRunnerConfig {
        n_workers,
        max_moves_per_game,
        n_simulations,
        leaf_batch_size,
        c_puct,
        fpu_reduction,
        fast_prob,
        fast_sims,
        standard_sims,
        temp_threshold_compound_moves,
        draw_reward,
        ply_cap_value,
        quiescence_enabled,
        quiescence_blend_2,
        temp_min,
        zoi_enabled,
        zoi_lookback,
        zoi_margin,
        completed_q_values,
        c_visit,
        c_scale,
        gumbel_mcts,
        gumbel_m,
        gumbel_explore_moves,
        dirichlet_alpha,
        dirichlet_epsilon,
        dirichlet_enabled,
        results_queue_cap,
        full_search_prob,
        n_sims_quick,
        n_sims_full,
        random_opening_plies,
        selfplay_rotation_enabled,
        encoding_name,
        inference_pool_size,
        forced_win_policy_enabled,
        forced_win_policy_depth,
        forced_win_policy_weight,
        solver_enabled,
        solver_depth,
        solver_node_budget,
        solver_neighbor_dist,
        solver_visit_weight,
        seed_fraction,
        seed_corpus,
    } = cfg;

    assert_eq!(n_workers, 7);
    assert_eq!(max_moves_per_game, 77);
    assert_eq!(n_simulations, 88);
    assert_eq!(leaf_batch_size, 3);
    assert!(feq(c_puct, 2.5));
    assert!(feq(fpu_reduction, 0.125));
    assert!(feq(fast_prob, 0.375));
    assert_eq!(fast_sims, 37);
    assert_eq!(standard_sims, 42);
    assert_eq!(temp_threshold_compound_moves, 21);
    assert!(feq(draw_reward, -0.75));
    assert!(feq(ply_cap_value, -0.875));
    assert!(!quiescence_enabled);
    assert!(feq(quiescence_blend_2, 0.625));
    assert!(feq(temp_min, 0.0625));
    assert!(zoi_enabled);
    assert_eq!(zoi_lookback, 24);
    assert_eq!(zoi_margin, 9);
    assert!(completed_q_values);
    assert!(feq(c_visit, 37.5));
    assert!(feq(c_scale, 1.25));
    assert!(gumbel_mcts);
    assert_eq!(gumbel_m, 12);
    assert_eq!(gumbel_explore_moves, 7);
    assert!(feq(dirichlet_alpha, 0.4));
    assert!(feq(dirichlet_epsilon, 0.3));
    assert!(!dirichlet_enabled);
    assert_eq!(results_queue_cap, 20_000);
    assert!(feq(full_search_prob, 0.5));
    assert_eq!(n_sims_quick, 50);
    assert_eq!(n_sims_full, 100);
    assert_eq!(random_opening_plies, 3);
    assert!(selfplay_rotation_enabled);
    assert_eq!(encoding_name, Some("v6w25".to_string()));
    assert_eq!(inference_pool_size, Some(4096));
    assert!(forced_win_policy_enabled);
    assert_eq!(forced_win_policy_depth, 5);
    assert!(feq(forced_win_policy_weight, 2.5));
    assert!(solver_enabled);
    assert_eq!(solver_depth, 11);
    assert_eq!(solver_node_budget, 7777);
    assert_eq!(solver_neighbor_dist, 4);
    assert!(feq(solver_visit_weight, 0.9));
    assert!(feq(seed_fraction, 0.33));
    assert_eq!(seed_corpus, Some(vec![vec![(1, 2), (3, 4)]]));
}

/// Test 2 — `SelfPlayRunner::new(config)` accepts the distinct-sentinel config and
/// exposes SPEC-DERIVED shapes (no caller-supplied feature_len/policy_len exists,
/// C-1). Validation order matches the frozen ctor: fast_prob>0 + full_search_prob>0
/// is rejected only at `start()` (the §100 mutex), not at `new()`; with all the
/// sim budgets > 0 the ctor accepts. `start()` is NOT called here.
#[test]
fn distinct_config_constructs_and_exposes_spec_derived_shapes() {
    let runner = SelfPlayRunner::new(distinct_sentinels())
        .expect("ctor must accept the distinct-sentinel config");
    // encoding_name = "v6w25" → state_stride 8×625 = 5000, policy_stride 626.
    assert_eq!(runner.feature_len(), 5000);
    assert_eq!(runner.policy_len(), 626);
    assert!(!runner.is_running());
}

/// Test 3 — the `Default` impl is TEST-SCAFFOLDING, not the config authority.
///
/// We do NOT pin Default's field values as "the" defaults (R1 forbids code-side
/// config authority; the authoritative, minted defaults live in the WP8 Python
/// schema). We assert the CONTRAPOSITIVE: a bare `Default::default()` leaves the
/// identity key unset, so it is NOT a usable config on its own — constructing a
/// runner from it is a native `Err` (LAW-11). That is what "scaffolding, not
/// authority" means operationally.
#[test]
fn default_is_test_scaffolding_not_config_authority() {
    let cfg = SelfPlayRunnerConfig::default();
    assert!(
        cfg.encoding_name.is_none(),
        "Default leaves the identity key unset — scaffolding for `..Default::default()`, \
         not a complete config authority",
    );
    assert!(
        SelfPlayRunner::new(cfg).is_err(),
        "a bare Default::default() must NOT construct a runner — it is not the config \
         authority (LAW-11: absent identity key is an error)",
    );
}
