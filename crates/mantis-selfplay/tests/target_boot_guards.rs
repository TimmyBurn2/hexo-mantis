//! ⊕ WP12-R Phase T (TARGET INTEGRITY) — S2b BOOT-GUARD oracles (DESIGN_T §3.4 guards
//! 1 + 2), written at T-2 ORACLE-WRITE, byte-frozen through IMPL.
//!
//! POST-FIX contract, at the runner's effective-sims resolution seam
//! (`SelfPlayRunner::new`, runner/mod.rs — where zero-on-effective is already rejected):
//!
//!  * GUARD 1 (overshoot-aware temperature-arm bound): refuse to start when
//!    `max(ARMED effective sim counts) + leaf_batch_size - 1 > MAX_VISITS (128)`.
//!    Armed arms: standard (always; effective = standard_sims else n_simulations),
//!    fast iff `fast_prob > 0`, quick/full iff `full_search_prob > 0`. Reads two
//!    EXISTING config values — no new key (R120); armed VALUES never set (R119).
//!  * GUARD 2: refuse `representation == graph && completed_q_values == true` while
//!    `MAX_CHILDREN_PER_NODE (192) > MAX_VISITS (128)`, both constants named in the
//!    message (honest retirement-until-raised; F-1(b)).
//!
//! Every construction below uses a GRAPH encoding for the guard-1 cases (the guard's
//! grounds are the graph record's MAX_VISITS slot; whether IMPL additionally scopes
//! guard 1 by representation is deliberately NOT pinned — ORACLE_NOTES_T.md records
//! the open scoping note).
//!
//! PRE-FIX status at HEAD: the three REFUSE cases are RED (HEAD's ctor admits them);
//! the four ADMIT cases are GREEN (two-sided — they must STAY green post-fix).
//! Killers (PREREG_T §3): M-G (admit-at-128 flips), M-I (guard-2 refuse), M-O
//! (unarmed-arm admit flips), M-D roster context.

use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

const GRAPH_ENC: &str = "gnn_axis_v1";

fn graph_cfg() -> SelfPlayRunnerConfig {
    SelfPlayRunnerConfig {
        encoding_name: Some(GRAPH_ENC.to_string()),
        n_simulations: 50,
        leaf_batch_size: 8,
        standard_sims: 0, // effective standard = n_simulations
        fast_prob: 0.0,
        fast_sims: 50,
        full_search_prob: 0.0,
        n_sims_quick: 0,
        n_sims_full: 0,
        completed_q_values: false,
        ..Default::default()
    }
}

// ── guard 1 ──────────────────────────────────────────────────────────────────────────

#[test]
fn guard1_admits_at_the_exact_boundary_128() {
    // sims 121 + batch 8 - 1 = 128 == MAX_VISITS → ADMIT (the F-2 admit-at-boundary
    // case; M-G's `>` → `>=` flip reds exactly this).
    let cfg = SelfPlayRunnerConfig { n_simulations: 121, ..graph_cfg() };
    assert!(
        SelfPlayRunner::new(cfg).is_ok(),
        "guard 1 must ADMIT at the exact boundary (121 + 8 - 1 == 128)"
    );
}

#[test]
fn guard1_refuses_one_past_the_boundary_129() {
    // sims 122 + batch 8 - 1 = 129 > 128 → REFUSE, naming MAX_VISITS.
    let cfg = SelfPlayRunnerConfig { n_simulations: 122, ..graph_cfg() };
    let err = SelfPlayRunner::new(cfg).err().unwrap_or_else(|| {
        panic!(
            "guard 1 must REFUSE 122 + 8 - 1 = 129 > MAX_VISITS — the silent top-k \
             truncation's replacement boundary (DESIGN_T §3.4)"
        )
    });
    assert!(
        err.contains("MAX_VISITS"),
        "the refusal must name the governing constant MAX_VISITS: {err}"
    );
    assert!(err.contains("128"), "the refusal must name the bound value 128: {err}");
}

#[test]
fn guard1_admits_the_run5_shape() {
    // run5: 50 + 8 - 1 = 57 <= 128 → ADMIT untouched (R119: the guard reads armed
    // values, never sets them; run5.yaml:176,151 shape).
    assert!(
        SelfPlayRunner::new(graph_cfg()).is_ok(),
        "guard 1 must admit the run5 shape (50 sims + batch 8)"
    );
}

#[test]
fn guard1_refuses_through_a_pcr_shaped_armed_arm() {
    // The max-over-ARMED-arms conjunct: full_search_prob > 0 arms n_sims_full = 200;
    // 200 + 8 - 1 = 207 > 128 → REFUSE even though the standard arm (50) fits.
    let cfg = SelfPlayRunnerConfig {
        full_search_prob: 0.5,
        n_sims_quick: 10,
        n_sims_full: 200,
        ..graph_cfg()
    };
    let err = SelfPlayRunner::new(cfg).err().unwrap_or_else(|| {
        panic!("guard 1 must take the MAX over armed arms — an armed 200-sim full arm busts 128")
    });
    assert!(err.contains("MAX_VISITS"), "PCR-arm refusal must name MAX_VISITS: {err}");
}

#[test]
fn guard1_admits_an_unarmed_arm_carrying_an_over_cap_value() {
    // [rev-3, N-2 / M-O] The ARMED filter's admit side: `fast_sims: 500` with
    // `fast_prob: 0.0` is a DEFINED but UNARMED arm — it must NOT enter the max.
    let cfg = SelfPlayRunnerConfig { fast_sims: 500, fast_prob: 0.0, ..graph_cfg() };
    assert!(
        SelfPlayRunner::new(cfg).is_ok(),
        "guard 1 must compute its max over ARMED arms only — an unarmed fast_sims: 500 \
         at fast_prob: 0.0 is inert (M-O's admit case)"
    );
}

// ── guard 2 ──────────────────────────────────────────────────────────────────────────

#[test]
fn guard2_refuses_graph_with_completed_q_naming_both_constants() {
    let cfg = SelfPlayRunnerConfig { completed_q_values: true, ..graph_cfg() };
    let err = SelfPlayRunner::new(cfg).err().unwrap_or_else(|| {
        panic!(
            "guard 2 must refuse representation==graph + completed_q_values=true while \
             MAX_CHILDREN_PER_NODE (192) > MAX_VISITS (128): the post-fix improved \
             exporter is child-count-wide (DESIGN_T §1.3/§3.4, F-1(b))"
        )
    });
    assert!(err.contains("192"), "guard-2 refusal must name MAX_CHILDREN_PER_NODE=192: {err}");
    assert!(err.contains("128"), "guard-2 refusal must name MAX_VISITS=128: {err}");
    assert!(
        err.contains("completed_q"),
        "guard-2 refusal must name the offending key so the operator can act: {err}"
    );
}

#[test]
fn guard2_admits_graph_without_completed_q() {
    // Single-conjunct admit side 1: graph + completed_q=false (run5's own state).
    assert!(
        SelfPlayRunner::new(graph_cfg()).is_ok(),
        "guard 2 must admit graph + completed_q_values=false"
    );
}

#[test]
fn guard2_admits_grid_with_completed_q() {
    // Single-conjunct admit side 2: grid(dense) + completed_q=true — dense-362 records
    // carry no MAX_VISITS; the key stays alive (F-23's history respected).
    let cfg = SelfPlayRunnerConfig {
        encoding_name: Some("v6".to_string()),
        completed_q_values: true,
        n_simulations: 50,
        leaf_batch_size: 8,
        standard_sims: 0,
        ..Default::default()
    };
    assert!(
        SelfPlayRunner::new(cfg).is_ok(),
        "guard 2 must admit a GRID encoding with completed_q_values=true"
    );
}
