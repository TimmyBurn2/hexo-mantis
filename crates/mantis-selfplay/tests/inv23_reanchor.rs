//! P-02 — inv23 re-anchor: `encoding_name` end-to-end registry resolution
//! (RE-ANCHOR of `inv23_selfplayrunner_encoding_name_e2e.rs`).
//!
//! The runner takes `encoding_name: Option<String>` and resolves the record at
//! `SelfPlayRunner::new` time. This re-anchor pins:
//!   1. `Some("v6w25")` → feature_len 5000 / policy_len 626;
//!   2. `Some("v6")`    → feature_len 2888 / policy_len 362;
//!   3. an UNKNOWN name → native `Err(String)` naming the bad name + a registry
//!      hint (NOT a `PyValueError` — the pyo3 shell is WP7, R6);
//!   4. `None` → native `Err` REGARDLESS of any shapes. The frozen config carried
//!      `feature_len`/`policy_len` override kwargs that could rescue a `None`; C-1
//!      DROPS those fields entirely, so `None`+shapes is structurally
//!      unrepresentable — an absent identity key is always an error (LAW-11: shapes
//!      do not tell Grid vs Graph; the frozen `None → v6` fallback is killed, D2).
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

/// Test 1 — `Some("v6w25")` resolves to v6w25 geometry (state_stride 8×625 = 5000,
/// policy_stride 626).
#[test]
fn encoding_name_v6w25_resolves_to_5000_626() {
    let runner = SelfPlayRunner::new(cfg_with_encoding(Some("v6w25")))
        .expect("v6w25 must resolve via the registry");
    assert_eq!(runner.feature_len(), 5000, "v6w25 state_stride = 8×625");
    assert_eq!(runner.policy_len(), 626, "v6w25 policy_stride");
    assert!(!runner.is_running(), "runner must not auto-start");
}

/// Test 2 — `Some("v6")` resolves to v6 geometry (2888 / 362).
#[test]
fn encoding_name_v6_resolves_to_2888_362() {
    let runner = SelfPlayRunner::new(cfg_with_encoding(Some("v6")))
        .expect("v6 must resolve via the registry");
    assert_eq!(runner.feature_len(), 2888, "v6 state_stride = 8×361");
    assert_eq!(
        runner.policy_len(),
        362,
        "v6 policy_stride = 361 + pass slot"
    );
    assert!(!runner.is_running());
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
