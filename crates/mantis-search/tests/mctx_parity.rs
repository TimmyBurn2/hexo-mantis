//! ⊕ GUMBEL-REPAIR-1 item 8 — parity against Mctx's OWN outputs.
//!
//! The fixture (`tests/fixtures/mctx_parity/mctx_parity_v1.json`) was minted by
//! running `google-deepmind/mctx` itself; `tools/gen_mctx_parity_fixtures.py`
//! carries the recipe and the reasoning for what is pinned. Nothing here restates
//! the paper — every expected number came out of the reference implementation.
//!
//! WHY UNIT SURFACES AND NOT A WHOLE SEARCH. Mctx searches a `MuZero` latent model
//! over a dense action space; this repo searches a real board over a ragged legal
//! set with batched leaf inference. A whole-search comparison would be comparing
//! two different environments. The surfaces pinned here are the ones where both
//! implementations compute the same function of the same numbers, which is where a
//! deviation would actually live.
//!
//! FLOAT TOLERANCE. Mctx computes in `jnp.float32` through XLA; this crate
//! computes in `f32` with `mul_add`. The two agree to a few ULP but not bitwise,
//! so comparisons are relative with an absolute floor — a bit-exact assertion here
//! would red on a compiler flag rather than on a behaviour change.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use mantis_search::mcts::seq_halving::{considered_visits_sequence, score_considered};

/// Relative tolerance for an Mctx-vs-Rust f32 comparison.
const REL_TOL: f64 = 2e-5;
/// Absolute floor, so a near-zero expected value does not demand infinite relative
/// precision.
const ABS_TOL: f64 = 2e-6;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/mctx_parity/mctx_parity_v1.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "the Mctx parity fixture must be present at {}: {e}. It is a committed \
             artefact, not something a test run mints — see \
             tools/gen_mctx_parity_fixtures.py",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("the parity fixture is valid JSON")
}

/// Assert `got` matches Mctx's `want` elementwise, naming the case and index.
fn assert_close(case: &str, field: &str, got: &[f32], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{case}/{field}: produced {} values against Mctx's {}",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let g = f64::from(g);
        let tol = ABS_TOL + REL_TOL * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{case}/{field}[{i}]: {g} vs Mctx {w} (tolerance {tol}) — this surface is \
             pinned to the reference implementation, so a drift here is a deviation \
             from the algorithm, not a rounding accident"
        );
    }
}

fn floats(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture case is missing the float array {key}"))
        .iter()
        .map(|x| x.as_f64().expect("fixture floats are JSON numbers"))
        .collect()
}

#[test]
fn the_sequential_halving_schedule_matches_mctx_entry_for_entry() {
    let doc = fixture();
    let cases = doc["seq_halving"].as_array().expect("seq_halving section");
    assert!(!cases.is_empty(), "an empty parity section proves nothing");
    for case in cases {
        let m = case["max_num_considered_actions"]
            .as_u64()
            .expect("m is an integer") as usize;
        let n = case["num_simulations"].as_u64().expect("N is an integer") as usize;
        let want: Vec<u32> = case["sequence"]
            .as_array()
            .expect("sequence array")
            .iter()
            .map(|x| x.as_u64().expect("visit counts are integers") as u32)
            .collect();
        let got = considered_visits_sequence(m, n);
        assert_eq!(
            got, want,
            "m={m} N={n}: the considered-visit schedule differs from Mctx's. The \
             schedule IS the budget — its length is the exact-consumption property \
             and its entries are the halving levels."
        );
    }
}

#[test]
fn every_pinned_schedule_consumes_its_whole_budget() {
    // The property the port exists for, asserted against the fixture's own N so it
    // cannot pass by agreeing with a wrong Mctx read.
    let doc = fixture();
    for case in doc["seq_halving"].as_array().expect("seq_halving section") {
        let n = case["num_simulations"].as_u64().expect("N is an integer") as usize;
        assert_eq!(
            case["sequence"].as_array().expect("sequence array").len(),
            n,
            "Mctx's own schedule for N={n} is not N long — the fixture is corrupt"
        );
    }
}

#[test]
fn the_root_score_matches_mctx_including_its_ineligibility_penalty() {
    let doc = fixture();
    let cases = doc["qtransform"].as_array().expect("qtransform section");
    assert!(!cases.is_empty(), "an empty parity section proves nothing");
    let mut eligible_seen = 0usize;
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let logits = floats(case, "prior_logits");
        let gumbel = floats(case, "gumbel");
        let completed = floats(case, "completed_qvalues");
        let visits: Vec<u32> = case["visit_counts"]
            .as_array()
            .expect("visit_counts array")
            .iter()
            .map(|x| x.as_u64().expect("visit counts are integers") as u32)
            .collect();
        let considered = case["considered_visit"].as_u64().expect("considered_visit") as u32;
        // Mctx subtracts `max(logits)` over the whole action set before scoring.
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let want = case["root_score_considered"]
            .as_array()
            .expect("root_score_considered array");
        for (i, w) in want.iter().enumerate() {
            let got = score_considered(
                considered,
                visits[i],
                gumbel[i] as f32,
                logits[i] as f32,
                max_logit as f32,
                completed[i] as f32,
            );
            match (got, w.as_f64()) {
                // `null` is Mctx's `-inf` penalty: the candidate is not at the
                // considered visit level and may not be descended into.
                (None, None) => {}
                (Some(g), Some(w)) => {
                    eligible_seen += 1;
                    assert_close(name, "root_score_considered", &[g], &[w]);
                }
                (g, w) => panic!(
                    "{name}/root_score_considered[{i}]: eligibility disagrees with Mctx \
                     — got {g:?}, Mctx {w:?}. Scoring an ineligible candidate is what \
                     lets Sequential Halving revisit a candidate it has already spent."
                ),
            }
        }
    }
    assert!(
        eligible_seen > 0,
        "every candidate in every case was ruled ineligible — the parity would pass \
         with a scorer that returns None unconditionally"
    );
}
