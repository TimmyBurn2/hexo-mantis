//! `SelfPlayRunnerConfig` field→slot one-to-one, by distinct-sentinel round-trip.
//!
//!   1. Every field maps to exactly one slot: a full struct literal plus an exhaustive
//!      destructure (no `..`) makes a dropped, added or renamed field fail to COMPILE, and
//!      distinct per-field sentinels catch any cross-wire.
//!   2. There is no per-game radius-jitter field and no `feature_len`/`policy_len` override,
//!      so a resurrected knob or a caller-supplied shape breaks this test loudly.
//!   3. The `Default` impl is test scaffolding, not the config authority: a bare `Default`
//!      leaves the identity key unset and cannot construct a runner.

use mantis_search::SearchKind;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// f32 approx-equality, exact for a literal round-trip.
fn feq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-9
}

/// Build a config whose every field is a distinct, non-default sentinel, so a swap, alias
/// or drop manifests as a field-equality failure.
///
/// A full struct literal with no `..Default::default()`, so a field change fails to compile.
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
        ply_cap_value: -0.875, // distinct from the draw_reward sentinel
        quiescence_enabled: false,
        quiescence_blend_2: 0.625,
        temp_min: 0.0625,
        c_visit: 37.5,
        c_scale: 1.25,
        search_kind: SearchKind::Gumbel,
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
        encoding_name: Some("gnn_axis_r8".to_string()),
    }
}

/// Prove every field maps to exactly one slot and no killed field is back.
///
/// The exhaustive destructure is the compile-time completeness guard; the sentinel
/// assertions are the no-cross-wire guard.
#[test]
fn every_field_maps_to_exactly_one_slot_and_no_killed_fields() {
    let cfg = distinct_sentinels();
    // A `..` is deliberately absent: any added, re-added or dropped field breaks this
    // destructure at compile time.
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
        c_visit,
        c_scale,
        search_kind,
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
        encoding_name,
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
    assert!(feq(c_visit, 37.5));
    assert!(feq(c_scale, 1.25));
    assert_eq!(search_kind, SearchKind::Gumbel);
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
    assert_eq!(encoding_name, Some("gnn_axis_r8".to_string()));
}

/// Prove the ctor accepts the sentinel config and exposes spec-derived shapes.
///
/// `fast_prob > 0` with `full_search_prob > 0` is rejected at `start()`, not `new()`, and
/// `start()` is not called here.
#[test]
fn distinct_config_constructs_and_exposes_spec_derived_shapes() {
    let runner = SelfPlayRunner::new(distinct_sentinels())
        .expect("ctor must accept the distinct-sentinel config");
    // Derived from the same spec the runner resolved, never transcribed: the spec decides
    // both numbers, and a caller-supplied shape override would make them disagree with it.
    let spec = mantis_encoding::lookup_or_panic("gnn_axis_r8");
    assert_eq!(runner.feature_len(), spec.state_stride());
    assert_eq!(runner.policy_len(), spec.policy_stride());
    assert_eq!(
        runner.policy_len(),
        362,
        "the graph action space is 19*19 + 1"
    );
    assert!(!runner.is_running());
}

/// Prove `Default` is test scaffolding, not the config authority.
///
/// Its field values are deliberately not pinned as "the" defaults; what is asserted is the
/// contrapositive, that a bare `Default::default()` cannot construct a runner.
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
