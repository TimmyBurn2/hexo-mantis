//! ⊕ GUMBEL-REPAIR-1 — the completed-Q math against Mctx's own numbers.
//!
//! IN-SRC rather than beside `tests/mctx_parity.rs` because the functions under
//! test are `pub(super)`. Widening them to `pub` to reach an integration test
//! would put a transform nothing outside this module may call onto the crate's
//! public surface, which is a worse trade than one extra test module.
//!
//! The fixture is Mctx's own output — see `tools/gen_mctx_parity_fixtures.py`.

use serde_json::Value;

use super::completed_q::{
    mctx_completed_qvalues, mctx_improved_policy_masses, mctx_interior_argmax_input, CqChild,
};

const REL_TOL: f32 = 2e-5;
const ABS_TOL: f32 = 2e-6;

fn fixture() -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/mctx_parity/mctx_parity_v1.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("committed parity fixture at {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the parity fixture is valid JSON")
}

fn f32s(case: &Value, key: &str) -> Vec<f32> {
    case[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture case is missing {key}"))
        .iter()
        .map(|x| x.as_f64().expect("fixture floats are JSON numbers") as f32)
        .collect()
}

fn assert_close(case: &str, field: &str, got: &[f32], want: &[f32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{case}/{field}: length differs from Mctx's"
    );
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = ABS_TOL + REL_TOL * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{case}/{field}[{i}]: {g} vs Mctx {w} (tolerance {tol})"
        );
    }
}

/// `(children, raw_value)` for one fixture case.
fn children_of(case: &Value) -> (Vec<CqChild>, f32) {
    let priors = f32s(case, "prior_probs");
    let qvalues = f32s(case, "qvalues");
    let visits: Vec<u32> = case["visit_counts"]
        .as_array()
        .expect("visit_counts array")
        .iter()
        .map(|x| x.as_u64().expect("visit counts are integers") as u32)
        .collect();
    let children = (0..priors.len())
        .map(|i| CqChild {
            visits: visits[i],
            prior: priors[i],
            // The fixture's `qvalues` are already in root perspective; the negamax
            // flip this repo applies when reading `w_value` is pinned separately
            // (`tests/perspective_parity.rs`) and is not entangled here.
            q_val: if visits[i] > 0 { qvalues[i] } else { 0.0 },
        })
        .collect();
    (
        children,
        case["raw_value"].as_f64().expect("raw_value") as f32,
    )
}

#[test]
fn the_completed_qvalues_match_mctx() {
    let doc = fixture();
    let c_visit = doc["maxvisit_init"].as_f64().expect("maxvisit_init") as f32;
    let c_scale = doc["value_scale"].as_f64().expect("value_scale") as f32;
    let cases = doc["qtransform"].as_array().expect("qtransform section");
    assert!(!cases.is_empty(), "an empty parity section proves nothing");
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let (children, raw_value) = children_of(case);
        let got = mctx_completed_qvalues(&children, raw_value, c_visit, c_scale);
        assert_close(
            name,
            "completed_qvalues",
            &got,
            &f32s(case, "completed_qvalues"),
        );
    }
}

#[test]
fn the_improved_policy_matches_mctx_action_weights() {
    let doc = fixture();
    let c_visit = doc["maxvisit_init"].as_f64().expect("maxvisit_init") as f32;
    let c_scale = doc["value_scale"].as_f64().expect("value_scale") as f32;
    for case in doc["qtransform"].as_array().expect("qtransform section") {
        let name = case["name"].as_str().expect("case name");
        let (children, raw_value) = children_of(case);
        let got = mctx_improved_policy_masses(&children, raw_value, c_visit, c_scale);
        assert_close(name, "action_weights", &got, &f32s(case, "action_weights"));
        let total: f32 = got.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "{name}: the exported target sums to {total}, not 1"
        );
    }
}

/// THE DEVIATION-2 WITNESS. The completion must read the RAW root value, so moving
/// it while every child statistic stays fixed must move the answer. A transform
/// that had kept `W/N` would be constant across this sweep.
#[test]
fn the_completion_reads_the_raw_root_value_and_not_the_backed_up_mean() {
    let doc = fixture();
    let case = doc["qtransform"]
        .as_array()
        .expect("qtransform section")
        .iter()
        .find(|c| c["name"] == "n50_concentrated")
        .expect("the n50_concentrated case");
    let (children, raw_value) = children_of(case);
    let base = mctx_improved_policy_masses(&children, raw_value, 50.0, 0.1);
    let moved = mctx_improved_policy_masses(&children, raw_value + 0.5, 50.0, 0.1);
    assert!(
        base.iter().zip(&moved).any(|(a, b)| (a - b).abs() > 1e-4),
        "moving the raw root value by 0.5 changed no exported mass — the completion \
         is not reading it, which is exactly the deviation this arm repairs"
    );
}

/// The two arms must be DISTINGUISHABLE, or a dispatch wired to the wrong one
/// would pass every parity test above by accident.
#[test]
fn the_mctx_arm_and_the_legacy_arm_disagree() {
    let doc = fixture();
    let case = doc["qtransform"]
        .as_array()
        .expect("qtransform section")
        .iter()
        .find(|c| c["name"] == "n50_concentrated")
        .expect("the n50_concentrated case");
    let (children, raw_value) = children_of(case);

    let mut agg = super::completed_q::CqAgg {
        sum_n: 0,
        max_n: 0,
        visited_prior_sum: 0.0,
        policy_weighted_q: 0.0,
        v_hat: raw_value,
        raw_value,
    };
    for ch in &children {
        agg.sum_n += ch.visits;
        agg.max_n = agg.max_n.max(ch.visits);
        if ch.visits > 0 {
            agg.visited_prior_sum += ch.prior;
            agg.policy_weighted_q += ch.prior * ch.q_val;
        }
    }
    // The legacy arm at its own shipped scale (`c_scale: 1.0`), which is the
    // comparison that matters: this is what every minted config computes today.
    let legacy = super::completed_q::improved_policy_masses(&children, &agg, 50.0, 1.0);
    let mctx = mctx_improved_policy_masses(&children, raw_value, 50.0, 0.1);
    let max_gap = legacy
        .iter()
        .zip(&mctx)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_gap > 1e-3,
        "the two arms produced the same target (max gap {max_gap}) — then either the \
         Mctx arm is not doing anything or the legacy arm was already Mctx, and the \
         parity above proves nothing about which one a config selects"
    );
}

/// THE DEVIATION-4 WITNESS. Interior selection is the improved policy with the
/// visit-count correction, pinned elementwise against Mctx's `_prepare_argmax_input`
/// rather than only at its argmax — many wrong score vectors share an argmax.
#[test]
fn the_interior_selection_score_matches_mctx() {
    let doc = fixture();
    let c_visit = doc["maxvisit_init"].as_f64().expect("maxvisit_init") as f32;
    let c_scale = doc["value_scale"].as_f64().expect("value_scale") as f32;
    let cases = doc["qtransform"].as_array().expect("qtransform section");
    assert!(!cases.is_empty(), "an empty parity section proves nothing");
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let (children, raw_value) = children_of(case);
        let completed = mctx_completed_qvalues(&children, raw_value, c_visit, c_scale);
        let priors: Vec<f32> = children.iter().map(|c| c.prior).collect();
        let visits: Vec<u32> = children.iter().map(|c| c.visits).collect();
        let got = mctx_interior_argmax_input(&priors, &completed, &visits);
        assert_close(
            name,
            "interior_argmax_input",
            &got,
            &f32s(case, "interior_argmax_input"),
        );
    }
}

/// The visit-count correction is what stops the interior selector piling every visit on
/// one child (the paper's §5). Without it the score IS the improved policy and the argmax
/// never moves however many times a child is visited.
#[test]
fn the_interior_score_moves_off_a_child_as_its_visits_accumulate() {
    let priors = vec![0.5f32, 0.3, 0.2];
    let completed = vec![1.0f32, 0.9, 0.8];
    let fresh = mctx_interior_argmax_input(&priors, &completed, &[0, 0, 0]);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv {
                    (i, x)
                } else {
                    (bi, bv)
                }
            })
            .0
    };
    let leader = argmax(&fresh);
    let mut visits = [0u32; 3];
    visits[leader] = 40;
    let after = mctx_interior_argmax_input(&priors, &completed, &visits);
    assert_ne!(
        argmax(&after),
        leader,
        "forty visits on the leading child left it still leading — the \
         `- visits / (1 + sum_visits)` correction is not being applied, and a selector \
         without it visits one child forever"
    );
}
