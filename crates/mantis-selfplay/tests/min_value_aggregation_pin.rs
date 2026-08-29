//! CARD-MINPIN (WPSC Phase 4) — parity pin for the K-cluster min value aggregation.
//!
//! `aggregate_cluster_values_min` is the lifted form of the frozen inline loop at
//! `runner/search_drive.rs` (min over per-cluster leaf values). The min/max asymmetry
//! it embodies — value takes the WORST cluster view while policy takes the best-scoring
//! view — is a flagged defect preserved pending the matched-FLOP dense arm
//! (falsified.md F-04 scope note). This pin makes any drive-by "fix" loud:
//! ZERO behavior change is the contract; output must equal the fold-min BIT-FOR-BIT.

use mantis_selfplay::aggregate_cluster_values_min;

fn fold_min(values: &[f32]) -> f32 {
    let mut m = values[0];
    for &v in values {
        if v < m {
            m = v;
        }
    }
    m
}

#[test]
fn k1_degenerate_returns_the_single_value_bit_for_bit() {
    for &v in &[
        -1.0f32,
        -0.999_999_9,
        -0.5,
        -0.0,
        0.0,
        0.25,
        0.999_999_9,
        1.0,
    ] {
        let out = aggregate_cluster_values_min(&[v]);
        assert_eq!(
            out.to_bits(),
            v.to_bits(),
            "K=1 must be identity, got {out} for {v}"
        );
    }
}

#[test]
fn k2_returns_the_smaller_bit_for_bit() {
    let cases: &[([f32; 2], f32)] = &[
        ([-0.75, 0.5], -0.75),
        ([0.5, -0.75], -0.75),
        ([0.123_456_7, 0.123_456_8], 0.123_456_7),
        ([-1.0, 1.0], -1.0),
        ([0.0, 0.0], 0.0),
    ];
    for (input, expected) in cases {
        let out = aggregate_cluster_values_min(input);
        assert_eq!(
            out.to_bits(),
            expected.to_bits(),
            "K=2 min mismatch for {input:?}"
        );
    }
}

#[test]
fn k_many_equals_fold_min_bit_for_bit_on_synthetic_post_tanh_values() {
    // Deterministic synthetic post-tanh spread in [-1, 1]; no RNG (repro discipline).
    let mut values = [0.0f32; 8];
    for (i, slot) in values.iter_mut().enumerate() {
        // K_max = 8 (registry k_max); spread includes negatives, near-±1, near-zero.
        let x = (i as f32) * 0.61 - 2.13;
        *slot = x.tanh();
    }
    let out = aggregate_cluster_values_min(&values);
    assert_eq!(out.to_bits(), fold_min(&values).to_bits());
}

#[test]
fn order_independence_all_rotations_and_reversal_bit_for_bit() {
    let base = [-0.31f32, 0.87, -0.99, 0.02, -0.44, 0.66];
    let expected = aggregate_cluster_values_min(&base);
    for r in 0..base.len() {
        let mut rotated = [0.0f32; 6];
        for i in 0..base.len() {
            rotated[i] = base[(i + r) % base.len()];
        }
        assert_eq!(
            aggregate_cluster_values_min(&rotated).to_bits(),
            expected.to_bits(),
            "rotation {r} changed the aggregate"
        );
    }
    let mut reversed = base;
    reversed.reverse();
    assert_eq!(
        aggregate_cluster_values_min(&reversed).to_bits(),
        expected.to_bits()
    );
}
