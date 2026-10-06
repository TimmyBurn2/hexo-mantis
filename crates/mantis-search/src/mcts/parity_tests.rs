//! ⊕ GUMBEL-REPAIR-1 — the completed-Q math against Mctx's own numbers.
//!
//! R8 justify (and why IN-SRC rather than beside `tests/mctx_parity.rs`): the functions under
//! test are `pub(super)` — widening them would put a transform nothing outside this module may
//! call onto the crate's public surface — and both σ arms are pinned on the SAME fixture cases
//! through one `children_of`, so a split forks the reader to save one test module.
//!
//! The fixture is Mctx's own output — see `tools/gen_mctx_parity_fixtures.py`.

use serde_json::Value;

use super::completed_q::{
    mctx_completed_qvalues, mctx_improved_policy_masses, mctx_improved_policy_value,
    mctx_interior_argmax_input, CqChild, QSigma,
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
        let got = mctx_completed_qvalues(
            &children,
            raw_value,
            QSigma {
                c_visit,
                c_scale,
                rescale: true,
            },
        );
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
        let got = mctx_improved_policy_masses(
            &children,
            raw_value,
            QSigma {
                c_visit,
                c_scale,
                rescale: true,
            },
        );
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
    let base = mctx_improved_policy_masses(
        &children,
        raw_value,
        QSigma {
            c_visit: 50.0,
            c_scale: 0.1,
            rescale: true,
        },
    );
    let moved = mctx_improved_policy_masses(
        &children,
        raw_value + 0.5,
        QSigma {
            c_visit: 50.0,
            c_scale: 0.1,
            rescale: true,
        },
    );
    assert!(
        base.iter().zip(&moved).any(|(a, b)| (a - b).abs() > 1e-4),
        "moving the raw root value by 0.5 changed no exported mass — the completion \
         is not reading it, which is exactly the deviation this arm repairs"
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
        let completed = mctx_completed_qvalues(
            &children,
            raw_value,
            QSigma {
                c_visit,
                c_scale,
                rescale: true,
            },
        );
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

/// The once-evaluated score is BIT-identical to the per-pass form it replaced (logit three
/// times, exp twice per child), over seeded random children including floored and huge priors.
#[test]
fn the_interior_score_evaluated_once_is_bit_identical_to_the_per_pass_form() {
    fn per_pass(priors: &[f32], completed: &[f32], visits: &[u32]) -> Vec<f32> {
        let n = priors.len().min(completed.len()).min(visits.len());
        if n == 0 {
            return Vec::new();
        }
        let logit = |j: usize| priors[j].max(1e-8).ln() + completed[j];
        let max_logit = (0..n).map(logit).fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return Vec::new();
        }
        let sum_exp: f32 = (0..n).map(|j| (logit(j) - max_logit).exp()).sum();
        if sum_exp <= 0.0 {
            return Vec::new();
        }
        let denom = 1.0 + visits[..n].iter().sum::<u32>() as f32;
        (0..n)
            .map(|j| (logit(j) - max_logit).exp() / sum_exp - visits[j] as f32 / denom)
            .collect()
    }
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    for case in 0..2_000 {
        let n = 1 + (case % 600);
        let priors: Vec<f32> = (0..n)
            .map(|j| match j % 7 {
                0 => 0.0,
                1 => next() * 1e-9,
                _ => next(),
            })
            .collect();
        let scale = [1.0f32, 50.0, 400.0][case % 3];
        let completed: Vec<f32> = (0..n).map(|_| (next() - 0.5) * scale).collect();
        let visits: Vec<u32> = (0..n).map(|_| (next() * 60.0) as u32).collect();
        let want = per_pass(&priors, &completed, &visits);
        let got = mctx_interior_argmax_input(&priors, &completed, &visits);
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "case {case} (n {n}): the once-evaluated score drifted from the per-pass form"
        );
    }
}

/// The raw arm shares Mctx's completion: min-max of its output, times the visit scale, must
/// be the fixture's rescaled vector (and the all-unvisited constant vector must map to zeros).
#[test]
fn the_no_rescale_arm_carries_the_mctx_completion_under_a_minmax() {
    let doc = fixture();
    let c_visit = doc["maxvisit_init"].as_f64().expect("maxvisit_init") as f32;
    let c_scale = doc["value_scale"].as_f64().expect("value_scale") as f32;
    let raw_sigma = QSigma {
        c_visit,
        c_scale,
        rescale: false,
    };
    let cases = doc["qtransform"].as_array().expect("qtransform section");
    assert!(!cases.is_empty(), "an empty parity section proves nothing");
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let (children, raw_value) = children_of(case);
        let raw = mctx_completed_qvalues(&children, raw_value, raw_sigma);
        let max_n = children.iter().map(|c| c.visits).max().unwrap_or(0);
        let visit_scale = (c_visit + max_n as f32) * c_scale;
        let lo = raw.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let span = hi - lo;
        let rescaled: Vec<f32> = raw
            .iter()
            .map(|&v| {
                if span > 0.0 {
                    (v - lo) / span * visit_scale
                } else {
                    0.0
                }
            })
            .collect();
        assert_close(
            name,
            "minmax(no_rescale)",
            &rescaled,
            &f32s(case, "completed_qvalues"),
        );
    }
}

/// With the rescale off a losing child's completed value KEEPS ITS SIGN — scaled, never
/// shifted onto [0, visit_scale] the way every entry of Mctx's fixture is.
#[test]
fn the_no_rescale_arm_keeps_a_negative_q_negative() {
    let doc = fixture();
    let c_visit = doc["maxvisit_init"].as_f64().expect("maxvisit_init") as f32;
    let cases = doc["qtransform"].as_array().expect("qtransform section");
    let case = cases
        .iter()
        .find(|c| {
            let (children, _) = children_of(c);
            children.iter().any(|ch| ch.visits > 0 && ch.q_val < 0.0)
        })
        .expect("a fixture case with a visited, negative-Q child");
    let (children, raw_value) = children_of(case);
    let got = mctx_completed_qvalues(
        &children,
        raw_value,
        QSigma {
            c_visit,
            c_scale: 1.0,
            rescale: false,
        },
    );
    let expect_negative: Vec<usize> = children
        .iter()
        .enumerate()
        .filter(|(_, ch)| ch.visits > 0 && ch.q_val < 0.0)
        .map(|(i, _)| i)
        .collect();
    for i in expect_negative {
        assert!(
            got[i] < 0.0,
            "child {i} has Q {} yet completed to {} — the min-max rescale is still on",
            children[i].q_val,
            got[i]
        );
    }
    let rescaled = mctx_completed_qvalues(
        &children,
        raw_value,
        QSigma {
            c_visit,
            c_scale: 1.0,
            rescale: true,
        },
    );
    assert!(
        rescaled.iter().all(|&v| v >= 0.0),
        "the rescaled arm is Mctx's [0, scale]"
    );
}

/// With the rescale off an unvisited root completes to a non-zero CONSTANT (`v_mix`), and the
/// exported target must still be the prior — softmax shift invariance, no zero special case.
#[test]
fn the_no_rescale_arm_exports_the_prior_when_nothing_is_visited() {
    let doc = fixture();
    let case = doc["qtransform"]
        .as_array()
        .expect("qtransform section")
        .iter()
        .find(|c| c["name"] == "n5_all_unvisited")
        .expect("the n5_all_unvisited case");
    let (children, raw_value) = children_of(case);
    let sigma = QSigma {
        c_visit: 50.0,
        c_scale: 1.0,
        rescale: false,
    };
    let completed = mctx_completed_qvalues(&children, raw_value, sigma);
    assert!(
        completed.iter().all(|&v| (v - completed[0]).abs() < 1e-6) && completed[0] != 0.0,
        "an unvisited root completes to one non-zero constant under the raw arm: {completed:?}"
    );
    let masses = mctx_improved_policy_masses(&children, raw_value, sigma);
    let priors: Vec<f32> = children.iter().map(|c| c.prior).collect();
    assert_close("n5_all_unvisited", "raw-arm masses", &masses, &priors);
}

/// Two visited children and one unvisited, hand-completed: v_mix = (0.1 + 4 · 0.21875) / 5 = 0.195.
fn hand_case() -> (Vec<CqChild>, f32, QSigma) {
    let children = vec![
        CqChild {
            visits: 3,
            prior: 0.5,
            q_val: 0.5,
        },
        CqChild {
            visits: 1,
            prior: 0.3,
            q_val: -0.25,
        },
        CqChild {
            visits: 0,
            prior: 0.2,
            q_val: 0.0,
        },
    ];
    let sigma = QSigma {
        c_visit: 50.0,
        c_scale: 1.0,
        rescale: false,
    };
    (children, 0.1, sigma)
}

#[test]
fn the_improved_policy_value_is_the_expectation_of_the_completion_in_value_units() {
    let (children, raw, sigma) = hand_case();
    let values = [0.5f64, -0.25, 0.195];
    let logits: Vec<f64> = children
        .iter()
        .zip(values)
        .map(|(ch, q)| f64::from(ch.prior).ln() + 53.0 * q)
        .collect();
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = logits.iter().map(|l| (l - top).exp()).sum();
    let want: f64 = logits
        .iter()
        .zip(values)
        .map(|(l, q)| (l - top).exp() / z * q)
        .sum();
    let got = mctx_improved_policy_value(&children, raw, sigma).expect("three children");
    assert!((f64::from(got) - want).abs() < 1e-6, "{got} against {want}");
    let visit_mean = (3.0 * 0.5 + 1.0 * -0.25) / 4.0;
    assert!(
        (f64::from(got) - visit_mean).abs() > 0.05,
        "{got} reads the visit mean"
    );
    let masses = mctx_improved_policy_masses(&children, raw, sigma);
    let from_masses: f64 = masses
        .iter()
        .zip(values)
        .map(|(&m, q)| f64::from(m) * q)
        .sum();
    assert!(
        (f64::from(got) - from_masses).abs() < 1e-6,
        "π′ is not the exported target's"
    );
}

#[test]
fn an_unsearched_root_values_at_its_raw_value_and_no_children_is_none() {
    let (mut children, raw, sigma) = hand_case();
    for ch in &mut children {
        ch.visits = 0;
    }
    let got = mctx_improved_policy_value(&children, raw, sigma).expect("three children");
    assert!((got - raw).abs() < 1e-7, "{got} against the raw {raw}");
    assert_eq!(mctx_improved_policy_value(&[], raw, sigma), None);
}

#[test]
fn the_value_never_leaves_the_unit_interval() {
    for q in [1.0f32, -1.0] {
        let children = vec![
            CqChild {
                visits: 7,
                prior: 0.6,
                q_val: q,
            },
            CqChild {
                visits: 5,
                prior: 0.4,
                q_val: q,
            },
        ];
        for rescale in [false, true] {
            let sigma = QSigma {
                c_visit: 50.0,
                c_scale: 1.0,
                rescale,
            };
            let got = mctx_improved_policy_value(&children, q, sigma).expect("two children");
            assert!(
                (-1.0..=1.0).contains(&got) && (got - q).abs() < 1e-6,
                "{got} at q {q}"
            );
        }
    }
}
