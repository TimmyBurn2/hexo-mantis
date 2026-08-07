//! ⊕ WP12-R Phase T (TARGET INTEGRITY) — boot-guard oracles, RE-RULED by
//! R255/ADJ-D34 (the previous byte-frozen S2b suite pinned the `MAX_VISITS = 128`
//! literal this ruling deletes; rewriting these oracles is the ruling's own act).
//!
//! POST-R255 contract, at the runner's effective-sims resolution seam
//! (`SelfPlayRunner::new`, runner/mod.rs):
//!
//!  * The guard's capacity is DERIVED at composition from the configured sims
//!    regime: `max(ARMED effective sim counts) + leaf_batch_size - 1`, via the ONE
//!    authority `replay::hexg::derived_visit_capacity` (shared verbatim with the
//!    mint-time schema validator through the bridge — no second formula).
//!    Armed arms: standard (always; effective = standard_sims else n_simulations),
//!    fast iff `fast_prob > 0`, quick/full iff `full_search_prob > 0`.
//!  * A regime the record format cannot honor (derived capacity past the
//!    `u16` count ceiling `HEXG_VISIT_COUNT_CEILING`) is an ERROR — refused at
//!    mint by the schema twin; this boot-side refusal is the defense-in-depth
//!    line for un-minted constructions (tests, direct API use).
//!  * completed-Q on graph refuses while `MAX_CHILDREN_PER_NODE (192)` exceeds
//!    the DERIVED capacity (the old guard 2, generalized: its firing set shrinks
//!    exactly when the derived slots genuinely cover child-count-wide support —
//!    the "retirement-until-raised" condition realized per-regime).
//!
//! Killers: M-G' (derivation dropped → 600/75 admit reds), M-I' (completed-Q
//! refusal dropped below the child cap), M-O (unarmed-arm filter dropped).

use mantis_selfplay::replay::hexg::{derived_visit_capacity, HEXG_VISIT_COUNT_CEILING};
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

/// The prereg'd PCR 600/75 shape (RUN5_MINT_PREREG SIMS-REGIME row: R160/R163/R165).
fn pcr_600_75_cfg() -> SelfPlayRunnerConfig {
    SelfPlayRunnerConfig {
        full_search_prob: 0.10,
        n_sims_quick: 75,
        n_sims_full: 600,
        ..graph_cfg()
    }
}

// ── the derivation authority ─────────────────────────────────────────────────────────

#[test]
fn derived_capacity_is_max_armed_plus_leaf_overshoot() {
    // standard-only: 50 + 8 - 1 = 57.
    assert_eq!(derived_visit_capacity(50, 0, 0.0, 50, 0.0, 0, 0, 8, false), Ok(57));
    // standard_sims wins over n_simulations when set: 40 + 8 - 1 = 47.
    assert_eq!(derived_visit_capacity(50, 40, 0.0, 50, 0.0, 0, 0, 8, false), Ok(47));
    // PCR-armed: max(50, 75, 600) + 8 - 1 = 607.
    assert_eq!(derived_visit_capacity(50, 0, 0.0, 50, 0.10, 75, 600, 8, false), Ok(607));
    // fast-armed: max(50, 500) + 8 - 1 = 507.
    assert_eq!(derived_visit_capacity(50, 0, 0.5, 500, 0.0, 0, 0, 8, false), Ok(507));
}

#[test]
fn derivation_ignores_a_defined_but_unarmed_arm() {
    // [M-O] `fast_sims: 500` at `fast_prob: 0.0` must NOT enter the max: 50+8-1=57.
    assert_eq!(derived_visit_capacity(50, 0, 0.0, 500, 0.0, 0, 0, 8, false), Ok(57));
    // Quick/full carrying huge values while full_search_prob == 0.0: still 57.
    assert_eq!(
        derived_visit_capacity(50, 0, 0.0, 50, 0.0, 70_000, 70_000, 8, false),
        Ok(57)
    );
}

#[test]
fn derivation_refuses_a_regime_over_the_format_ceiling() {
    // 70_000 + 8 - 1 = 70_007 > u16::MAX (65_535): the record format's `n_visits`
    // count is u16 — no capacity can honor this regime, whatever the config asks.
    let err = derived_visit_capacity(50, 0, 0.0, 50, 0.10, 75, 70_000, 8, false)
        .expect_err("a regime past the u16 count ceiling cannot be honored");
    assert!(
        err.contains(&HEXG_VISIT_COUNT_CEILING.to_string()),
        "the refusal must name the structural ceiling {HEXG_VISIT_COUNT_CEILING}: {err}"
    );
    assert!(
        err.contains("mint"),
        "the refusal must say this is a mint-time error (R255: never a boot surprise): {err}"
    );
}

#[test]
fn the_ceiling_is_the_u16_count_type_not_a_tunable() {
    // Derived from the storage type — if someone re-tunes it as a literal this reds.
    assert_eq!(HEXG_VISIT_COUNT_CEILING, usize::from(u16::MAX));
    // Admit at the exact ceiling: max_armed + lb - 1 == 65_535 → Ok.
    assert_eq!(
        derived_visit_capacity(65_528, 0, 0.0, 50, 0.0, 0, 0, 8, false),
        Ok(HEXG_VISIT_COUNT_CEILING)
    );
    // One past → refuse.
    assert!(derived_visit_capacity(65_529, 0, 0.0, 50, 0.0, 0, 0, 8, false).is_err());
}

// ── boot behavior (the dispatch's pin: 600/75 boots) ────────────────────────────────

#[test]
fn boot_admits_the_prereg_600_75_pcr_regime() {
    // ADJ-D34's exact defect: the minted PCR config refused to boot at ANY
    // leaf_batch_size >= 1 under the 128 literal. R255 pin: it boots.
    assert!(
        SelfPlayRunner::new(pcr_600_75_cfg()).is_ok(),
        "R255: a 600/75-shaped PCR regime must BOOT — the guard's capacity is derived \
         from the regime, not compared against a literal"
    );
}

#[test]
fn boot_admits_the_run5_shape() {
    // run5: 50 + 8 - 1 = 57 → capacity 57, boots (R119: values read, never set).
    assert!(
        SelfPlayRunner::new(graph_cfg()).is_ok(),
        "the guard must admit the run5 shape (50 sims + batch 8)"
    );
}

#[test]
fn boot_refuses_a_regime_over_the_format_ceiling() {
    // Defense-in-depth: the schema twin refuses this at mint; a direct construction
    // must still die loud at boot, with the derivation's own message.
    let cfg = SelfPlayRunnerConfig {
        full_search_prob: 0.10,
        n_sims_quick: 75,
        n_sims_full: 70_000,
        ..graph_cfg()
    };
    let err = SelfPlayRunner::new(cfg)
        .err()
        .expect("a regime past the u16 count ceiling must not boot");
    assert!(
        err.contains(&HEXG_VISIT_COUNT_CEILING.to_string()),
        "the boot refusal must carry the derivation's ceiling message: {err}"
    );
}

// ── completed-Q (old guard 2, generalized against the DERIVED capacity) ─────────────

#[test]
fn completed_q_graph_refused_while_derived_capacity_below_child_cap() {
    // 50 + 8 - 1 = 57 < MAX_CHILDREN_PER_NODE (192): child-count-wide support
    // cannot fit — refuse, naming both values and the offending key.
    let cfg = SelfPlayRunnerConfig { completed_q_values: true, ..graph_cfg() };
    let err = SelfPlayRunner::new(cfg)
        .err()
        .expect("completed-Q on graph must refuse while derived capacity < 192");
    assert!(err.contains("192"), "must name MAX_CHILDREN_PER_NODE=192: {err}");
    assert!(err.contains("57"), "must name the derived capacity 57: {err}");
    assert!(err.contains("completed_q"), "must name the offending key: {err}");
}

#[test]
fn completed_q_graph_admitted_once_derived_capacity_covers_child_cap() {
    // The generalization pin: under the 600/75 regime the derived capacity (607)
    // covers MAX_CHILDREN_PER_NODE (192) — the record physically holds
    // child-count-wide support, so the refusal would be vacuous and must not fire
    // (the old guard's own "retirement-until-raised" condition, realized).
    let cfg = SelfPlayRunnerConfig { completed_q_values: true, ..pcr_600_75_cfg() };
    assert!(
        SelfPlayRunner::new(cfg).is_ok(),
        "completed-Q on graph must ADMIT once the derived capacity covers \
         MAX_CHILDREN_PER_NODE"
    );
}

#[test]
fn completed_q_grid_is_untouched() {
    // Dense-362 records carry no HEXG visit slot; the key stays alive (F-23).
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
        "a GRID encoding with completed_q_values=true is outside this guard"
    );
}
