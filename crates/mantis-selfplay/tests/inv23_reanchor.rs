//! `encoding_name` end-to-end registry resolution at `SelfPlayRunner::new`:
//!   1. every REGISTERED name resolves to that row's own spec-derived shapes;
//!   2. a deleted grid row does not resolve;
//!   3. an UNKNOWN name → native `Err(String)` naming the bad name + a registry hint;
//!   4. `None` → native `Err` REGARDLESS of any shapes: there is no shape override, and shapes
//!      do not tell one representation from another.
//!
//! Workers are never spawned (`max_moves_per_game = 0`, and `new()` does not
//! start) so no inference producer is needed.

use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Every-default config except the identity key; shapes are spec-derived only, so no shape
/// override can rescue a `None`.
fn cfg_with_encoding(encoding_name: Option<&str>) -> SelfPlayRunnerConfig {
    SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 0,
        n_simulations: 1,
        leaf_batch_size: 1,
        fast_sims: 1,
        standard_sims: 1,
        quiescence_enabled: false,
        quiescence_blend_2: 0.0,
        dirichlet_enabled: false,
        encoding_name: encoding_name.map(str::to_string),
        ..Default::default()
    }
}

/// Test 1 — EVERY registered name resolves, and both shapes come from that row's own
/// spec. Iterated over `all_specs()` rather than naming rows, so a newly registered
/// encoding is covered the day it lands.
#[test]
fn every_registered_encoding_name_resolves_to_its_own_spec_derived_shapes() {
    let mut seen = 0usize;
    for spec in mantis_encoding::all_specs() {
        let runner = SelfPlayRunner::new(cfg_with_encoding(Some(spec.name)))
            .unwrap_or_else(|e| panic!("{} must resolve via the registry: {e}", spec.name));
        assert_eq!(
            runner.feature_len(),
            spec.state_stride(),
            "{}: feature_len must be the row's own state_stride",
            spec.name
        );
        assert_eq!(
            runner.policy_len(),
            spec.policy_stride(),
            "{}: policy_len must be the row's own policy_stride",
            spec.name
        );
        assert!(!runner.is_running(), "{}: runner must not auto-start", spec.name);
        seen += 1;
    }
    assert!(seen > 0, "the registry shipped no encodings, so this test asserted nothing");
}

/// Test 2 — a DELETED grid row does not resolve: a stale config naming one is an error rather
/// than a silent re-resolution onto graph geometry.
#[test]
fn a_deleted_grid_encoding_name_does_not_resolve() {
    for name in ["v6", "v6w25", "v6_live2_ls"] {
        assert!(
            SelfPlayRunner::new(cfg_with_encoding(Some(name))).is_err(),
            "{name} was deleted with the dense path and must not resolve"
        );
    }
}

/// Test 3 — an unknown encoding name is a native `Err(String)` naming the bad name
/// and hinting at the registry source; a plain `String`, no live Python interpreter.
#[test]
fn unknown_encoding_name_is_native_err_naming_the_bad_name() {
    // `SelfPlayRunner` is not `Debug`, so match rather than `expect_err`.
    let err: String = match SelfPlayRunner::new(cfg_with_encoding(Some("not_a_real_encoding"))) {
        Ok(_) => panic!("an unknown encoding_name must be a native Err"),
        Err(e) => e, // statically a `String` — proves the pyo3 error type was stripped (R6)
    };
    assert!(
        err.contains("not_a_real_encoding"),
        "error must name the bad encoding_name; got: {err}",
    );
    assert!(
        err.contains("encoding_name") || err.contains("registry"),
        "error must hint at the registry source; got: {err}",
    );
}

/// Test 4 — `None` is a native `Err` REGARDLESS of any shapes: there is no shape override to
/// supply, so an absent identity key always loud-fails.
#[test]
fn none_encoding_name_is_native_err_regardless_of_shapes() {
    let err: String = match SelfPlayRunner::new(cfg_with_encoding(None)) {
        Ok(_) => panic!("None encoding_name must be a native Err (no dense-by-default)"),
        Err(e) => e,
    };
    assert!(
        err.contains("encoding_name"),
        "error must reference the missing identity key; got: {err}",
    );
}
