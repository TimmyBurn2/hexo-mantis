// R8 justify: ONE derivation — the HEXG visit-slot capacity — and every refusal it can
// make, in the file that also drives the BOOT guard against the same numbers. The mint
// surface and the boot surface must agree, and a test that measured only one of them
// could not say so; the two dialects' support bounds are the same claim read twice.
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
//!  * `search.kind: gumbel` on graph does NOT use that formula at all (R347(a)). Its row
//!    is SPARSE — the m visited candidates' exact entries plus one tail mass alpha — so its
//!    slot count is the MINTED `gumbel_m`, and a `gumbel_m` outside
//!    `1..=HEXG_GUMBEL_M_MAX` is the refusal on that arm.
//!
//! Killers: M-G' (derivation dropped → 600/75 admit reds), M-I' (the gumbel m bound
//! dropped), M-O (unarmed-arm filter dropped).

use mantis_search::SearchKind;
use mantis_selfplay::replay::hexg::{
    derived_visit_capacity, HEXG_GUMBEL_M_MAX, HEXG_VISIT_COUNT_CEILING,
};
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
        search_kind: SearchKind::Puct,
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
    assert_eq!(
        derived_visit_capacity(50, 0, 0.0, 50, 0.0, 0, 0, 8, 16, "puct"),
        Ok(57)
    );
    // standard_sims wins over n_simulations when set: 40 + 8 - 1 = 47.
    assert_eq!(
        derived_visit_capacity(50, 40, 0.0, 50, 0.0, 0, 0, 8, 16, "puct"),
        Ok(47)
    );
    // PCR-armed: max(50, 75, 600) + 8 - 1 = 607.
    assert_eq!(
        derived_visit_capacity(50, 0, 0.0, 50, 0.10, 75, 600, 8, 16, "puct"),
        Ok(607)
    );
    // fast-armed: max(50, 500) + 8 - 1 = 507.
    assert_eq!(
        derived_visit_capacity(50, 0, 0.5, 500, 0.0, 0, 0, 8, 16, "puct"),
        Ok(507)
    );
}

#[test]
fn derivation_ignores_a_defined_but_unarmed_arm() {
    // [M-O] `fast_sims: 500` at `fast_prob: 0.0` must NOT enter the max: 50+8-1=57.
    assert_eq!(
        derived_visit_capacity(50, 0, 0.0, 500, 0.0, 0, 0, 8, 16, "puct"),
        Ok(57)
    );
    // Quick/full carrying huge values while full_search_prob == 0.0: still 57.
    assert_eq!(
        derived_visit_capacity(50, 0, 0.0, 50, 0.0, 70_000, 70_000, 8, 16, "puct"),
        Ok(57)
    );
}

#[test]
fn derivation_refuses_a_regime_over_the_format_ceiling() {
    // 70_000 + 8 - 1 = 70_007 > u16::MAX (65_535): the record format's `n_visits`
    // count is u16 — no capacity can honor this regime, whatever the config asks.
    let err = derived_visit_capacity(50, 0, 0.0, 50, 0.10, 75, 70_000, 8, 16, "puct")
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
        derived_visit_capacity(65_528, 0, 0.0, 50, 0.0, 0, 0, 8, 16, "puct"),
        Ok(HEXG_VISIT_COUNT_CEILING)
    );
    // One past → refuse.
    assert!(derived_visit_capacity(65_529, 0, 0.0, 50, 0.0, 0, 0, 8, 16, "puct").is_err());
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
    //
    // AUDIT-1 F-21 CHANGED WHICH AXIS THIS ROW CAN USE — the same interaction its Python twin
    // (`tests/config/test_graph_visit_capacity_relation.py`) records. The mutation used to be
    // `n_sims_full: 70_000`, which is now refused EARLIER by a TIGHTER ceiling:
    // `MAX_ARMED_SIMS` (= `MAX_NODES / (4 * MAX_CHILDREN_PER_NODE)`), because the MCTS
    // node pool overflows from the sim count alone above that. On the SIMS axis the pool
    // bound subsumes the record-format one — at the pool ceiling the derived capacity is
    // still nowhere near 65535. `leaf_batch_size` is the OTHER term the capacity is derived from
    // and carries no such bound, so this row's subject survives on that axis.
    let cfg = SelfPlayRunnerConfig {
        leaf_batch_size: 70_000,
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

#[test]
fn boot_refuses_a_sim_budget_the_node_pool_cannot_serve() {
    // The interaction pinned rather than left for the next reader to rediscover: the old
    // 70_000-sim mutation now reds against the POOL bound, not the record format. Both
    // refusals are correct and a config hitting either cannot boot; what changed is which one
    // names it, and a reader of the row above needs to be able to find that out here.
    let cfg = SelfPlayRunnerConfig {
        full_search_prob: 0.10,
        n_sims_quick: 75,
        n_sims_full: 70_000,
        ..graph_cfg()
    };
    let err = SelfPlayRunner::new(cfg)
        .err()
        .expect("a sim budget past the pool bound must not boot");
    assert!(err.contains("MAX_ARMED_SIMS"), "{err}");
    assert!(
        err.contains("n_sims_full"),
        "the refusal must name the knob: {err}"
    );
}

// ── the density check: which SUPPORT the record has to hold ────────────────────────

/// R347(a) — under `gumbel` the slot count IS the minted m, and the sims regime does not
/// enter; under `puct` it is the derived formula, and m does not enter.
///
/// The two halves are asserted together because the failure this guards against is a
/// SECOND authority: a reader who saw only one arm could reasonably add the other arm's
/// term to it, and the row would then be sized by a quantity that does not bound it.
#[test]
fn the_gumbel_slot_count_is_the_minted_m_and_the_puct_one_is_the_derived_formula() {
    // PUCT: the sims regime decides, at every m.
    for m in [1usize, 8, 16] {
        assert_eq!(
            derived_visit_capacity(50, 0, 0.0, 50, 0.0, 0, 0, 8, m, "puct"),
            Ok(57),
            "m must not enter the PUCT derivation"
        );
    }
    // Gumbel: m decides, at every sims regime.
    for sims in [2usize, 50, 320, 600] {
        assert_eq!(
            derived_visit_capacity(sims, 0, 0.0, sims, 0.0, 0, 0, 8, 16, "gumbel"),
            Ok(16),
            "the sims regime must not enter the Gumbel slot count"
        );
        assert_eq!(
            derived_visit_capacity(sims, 0, 0.0, sims, 0.0, 0, 0, 8, 4, "gumbel"),
            Ok(4)
        );
    }
    println!(
        "sparse-row slot bound: HEXG_GUMBEL_M_MAX = {HEXG_GUMBEL_M_MAX}, derived at m=16 -> {:?}",
        derived_visit_capacity(320, 0, 0.0, 64, 0.0, 0, 0, 8, 16, "gumbel")
    );
}

/// The refusal on the Gumbel arm is m past the MINTED bound — not a sims regime.
#[test]
fn a_gumbel_m_past_the_minted_bound_is_refused() {
    for bad in [0usize, HEXG_GUMBEL_M_MAX + 1, 8192] {
        let err = derived_visit_capacity(320, 0, 0.0, 64, 0.0, 0, 0, 8, bad, "gumbel")
            .expect_err("m outside the minted range must not resolve to a slot count");
        assert!(
            err.contains("gumbel_m") && err.contains(&HEXG_GUMBEL_M_MAX.to_string()),
            "the refusal must name the key and the minted bound: {err}"
        );
    }
    // The bound itself resolves — a bound that refused its own value would be off by one.
    assert_eq!(
        derived_visit_capacity(320, 0, 0.0, 64, 0.0, 0, 0, 8, HEXG_GUMBEL_M_MAX, "gumbel"),
        Ok(HEXG_GUMBEL_M_MAX)
    );
}

// The GRID arm's "outside this guard entirely" row went with the grid rows themselves
// (R346(f)): there is no registered encoding left that this guard does not apply to, so the
// control it provided is now structural rather than testable.

/// The BOOT surface agrees with the mint surface — a graph runner under `gumbel` composes
/// its ring at the minted m, and one past the bound does not boot.
#[test]
fn boot_composes_the_gumbel_graph_ring_at_the_minted_m() {
    let cfg = SelfPlayRunnerConfig {
        search_kind: SearchKind::Gumbel,
        gumbel_m: HEXG_GUMBEL_M_MAX,
        ..graph_cfg()
    };
    assert!(
        SelfPlayRunner::new(cfg).is_ok(),
        "the sparse Gumbel row is the graph arm's format, not a refusal (R347(a))"
    );

    let cfg = SelfPlayRunnerConfig {
        search_kind: SearchKind::Gumbel,
        gumbel_m: HEXG_GUMBEL_M_MAX + 1,
        ..graph_cfg()
    };
    let err = SelfPlayRunner::new(cfg)
        .err()
        .expect("an m past the minted bound must not boot");
    assert!(err.contains("gumbel_m"), "{err}");
}

/// R347(a) — the tail's SHAPE is the recording prior, and the two target injectors write
/// mass by cell after the export, so they cannot be armed on the arm that stores a tail.
#[test]
fn boot_refuses_a_target_injector_on_the_sparse_row_arm() {
    for (forced, solver) in [(true, false), (false, true), (true, true)] {
        let cfg = SelfPlayRunnerConfig {
            search_kind: SearchKind::Gumbel,
            forced_win_policy_enabled: forced,
            solver_enabled: solver,
            ..graph_cfg()
        };
        let err = SelfPlayRunner::new(cfg)
            .err()
            .expect("an injector armed on the sparse-row arm must not boot");
        assert!(
            err.contains("forced_win_policy_enabled") && err.contains("solver_enabled"),
            "the refusal must name both keys so a reader knows which to disarm: {err}"
        );
    }
    // The control: DISARMED injectors leave the arm alone.
    let cfg = SelfPlayRunnerConfig {
        search_kind: SearchKind::Gumbel,
        forced_win_policy_enabled: false,
        solver_enabled: false,
        ..graph_cfg()
    };
    assert!(SelfPlayRunner::new(cfg).is_ok());
}

/// An unknown kind is REFUSED here too, not defaulted — the mint surface and the runner
/// surface must agree about what a kind name means.
#[test]
fn an_unknown_kind_is_refused_by_the_capacity_derivation() {
    let err = derived_visit_capacity(50, 0, 0.0, 50, 0.0, 0, 0, 8, 16, "mctx")
        .expect_err("an unknown kind must not resolve");
    assert!(err.contains("search.kind"), "{err}");
}
