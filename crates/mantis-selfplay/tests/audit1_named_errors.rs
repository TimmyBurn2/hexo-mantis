//! FFI-reachable surfaces that return NAMED errors, never a panic: the MCTS node-pool bound
//! (`select_leaves` expands TT-hit leaves outside the batch count, so `n_simulations` alone can
//! overflow the pool past `MAX_ARMED_SIMS`) and `HexgBuffer::new`'s encoding and capacity
//! refusals. The HEXG ring's persist/load refusals live in `replay_hexg.rs`.

use mantis_search::{MAX_ARMED_SIMS, MAX_CHILDREN_PER_NODE, MAX_NODES};
use mantis_selfplay::replay::hexg::{HexgBuffer, HEXG_CAPACITY_CEILING};
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

// The pool bound is derived, and checked at boot

#[test]
fn max_armed_sims_is_derived_from_the_pools_own_two_constants() {
    // Not a tuned number: it is what the pool can serve. If either constant moves, this moves
    // with it — which is the whole reason the bound is not a literal in the schema.
    assert_eq!(MAX_ARMED_SIMS, MAX_NODES / (4 * MAX_CHILDREN_PER_NODE));
    // Printed, not transcribed into a second assert: the value is what a re-mint reads, and
    // an asserted tally has to be re-edited every time either constant moves.
    println!(
        "MAX_ARMED_SIMS = {MAX_ARMED_SIMS} from MAX_NODES {MAX_NODES} / (4 * \
         MAX_CHILDREN_PER_NODE {MAX_CHILDREN_PER_NODE})"
    );
}

fn config_with(sims: usize) -> SelfPlayRunnerConfig {
    // `encoding_name` is REQUIRED: an absent registry spec is an error, not a v6
    // default. Every row below is about a DIFFERENT refusal, so the identity is supplied.
    SelfPlayRunnerConfig {
        n_simulations: sims,
        encoding_name: Some("gnn_axis_v1".into()),
        ..Default::default()
    }
}

#[test]
fn a_sim_budget_the_pool_cannot_serve_is_refused_at_boot() {
    // THE PIN. Before this, a config armed at 2000 booted fine and halted the run at the
    // first move that crossed the bound — a panic inside `finish_expansion`, mid self-play.
    let err = SelfPlayRunner::new(config_with(2000))
        .err()
        .expect("2000 sims is above the pool bound");
    assert!(err.contains("MAX_ARMED_SIMS"), "{err}");
    assert!(
        err.contains("n_simulations"),
        "the error must name the knob: {err}"
    );
}

#[test]
fn the_boundary_is_exactly_the_derived_value() {
    assert!(
        SelfPlayRunner::new(config_with(MAX_ARMED_SIMS)).is_ok(),
        "the bound itself must be servable"
    );
    assert!(
        SelfPlayRunner::new(config_with(MAX_ARMED_SIMS + 1)).is_err(),
        "one past the bound must not be"
    );
}

#[test]
fn the_shipped_sims_regimes_are_all_inside_the_bound() {
    // The control: the bound refuses no sims regime this repo actually runs.
    for sims in [2usize, 50, 150, 320, 600] {
        assert!(
            SelfPlayRunner::new(config_with(sims)).is_ok(),
            "{sims} sims must boot"
        );
    }
}

#[test]
fn a_zero_or_negative_dirichlet_alpha_is_refused_when_the_noise_is_armed() {
    // `sample_dirichlet` refuses such an alpha only at the first noised root, mid-game;
    // boot is where the config key can still be named.
    for bad in [0.0f32, -1.0, f32::NAN] {
        let cfg = SelfPlayRunnerConfig {
            dirichlet_enabled: true,
            dirichlet_alpha: bad,
            ..config_with(50)
        };
        let err = SelfPlayRunner::new(cfg)
            .err()
            .unwrap_or_else(|| panic!("alpha {bad} must be refused"));
        assert!(err.contains("dirichlet_alpha"), "{err}");
    }
    // The control: the noise DISARMED does not care what alpha says.
    let cfg = SelfPlayRunnerConfig {
        dirichlet_enabled: false,
        dirichlet_alpha: 0.0,
        ..config_with(50)
    };
    assert!(
        SelfPlayRunner::new(cfg).is_ok(),
        "a disarmed dirichlet must not be gated on its unused alpha"
    );
}

// The buffer constructor

#[test]
fn an_unknown_encoding_is_an_err_naming_the_registered_set() {
    let err = HexgBuffer::new(8, "nope", 64)
        .err()
        .expect("'nope' is not registered");
    assert!(err.contains("nope"), "{err}");
    assert!(
        err.contains("gnn_axis_v1"),
        "the sorted known list must be in the message: {err}"
    );
}

#[test]
fn a_zero_capacity_is_refused_instead_of_panicking_on_the_first_push() {
    let err = HexgBuffer::new(0, "gnn_axis_v1", 64)
        .err()
        .expect("capacity 0 stores nothing");
    assert!(err.contains("capacity 0"), "{err}");
}

#[test]
fn a_capacity_that_would_wrap_the_slot_geometry_is_refused() {
    // In a release build the product wraps to a small allocation and every later index is
    // wrong; at the allocator it aborts, and an abort is the ONE exit `panic = "unwind"`
    // cannot convert into a Python exception.
    let err = HexgBuffer::new(HEXG_CAPACITY_CEILING + 1, "gnn_axis_v1", 64)
        .err()
        .expect("past the ceiling");
    assert!(err.contains("ceiling"), "{err}");
}

#[test]
fn an_ordinary_buffer_still_builds() {
    // The control for the three rows above.
    assert!(HexgBuffer::new(64, "gnn_axis_v1", 64).is_ok());
}
