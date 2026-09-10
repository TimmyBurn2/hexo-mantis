//! P-02 — inv23 re-anchor: `encoding_name` end-to-end registry resolution
//! (RE-ANCHOR of `inv23_selfplayrunner_encoding_name_e2e.rs`).
//!
//! The runner takes `encoding_name: Option<String>` and resolves the record at
//! `SelfPlayRunner::new` time. This re-anchor pins:
//!   1. every REGISTERED name resolves to that row's own spec-derived shapes;
//!   3. an UNKNOWN name → native `Err(String)` naming the bad name + a registry
//!      hint (NOT a `PyValueError` — the pyo3 shell is WP7, R6);
//!   4. `None` → native `Err` REGARDLESS of any shapes. The frozen config carried
//!      `feature_len`/`policy_len` override kwargs that could rescue a `None`; C-1
//!      DROPS those fields entirely, so `None`+shapes is structurally
//!      unrepresentable — an absent identity key is always an error (LAW-11: shapes
//!      do not tell one representation from another; the frozen `None → v6` fallback is
//!      killed, D2).
//!
//! Workers are never spawned (`max_moves_per_game = 0`, and `new()` does not
//! start) so no inference producer is needed.

use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

/// Every-default config except the identity key. NOTE: there is NO `feature_len` /
/// `policy_len` field to pass — shapes are spec-derived only (C-1), which is
/// exactly why `None` cannot be rescued by a shape override (LAW-11).
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
/// encoding is covered the day it lands (LAW-08) — the two grid rows this used to name by
/// hand went with the dense path and their transcribed 5000/626 and 2888/362 went with
/// them.
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

/// Test 2 — a DELETED grid row does not resolve. The `None → v6` fallback is killed and so
/// is `v6` itself (R346(f)), so a stale config naming one is an error rather than a silent
/// re-resolution onto graph geometry.
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
/// and hinting at the registry source. No live Python interpreter is required (the
/// frozen test needed one to format a `PyValueError`); this is a plain `String`.
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

/// Test 4 — `None` is a native `Err` REGARDLESS of any shapes. There is no shape
/// override to supply (C-1 dropped `feature_len`/`policy_len`), so the frozen
/// "silent v6 fallback for a wider-encoding caller" hazard is UNREPRESENTABLE: an
/// absent identity key always loud-fails (LAW-11 / D2 killed the `None → v6` fallback).
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
