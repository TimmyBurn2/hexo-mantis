//! P-10 — inv_dws3 re-anchor: D-WS3 L1 solver-in-loop + D-WS3V3 seeding/counters
//! wiring source-presence pins (RE-ANCHOR of `inv_dws3_solver_in_loop_wiring.rs`).
//!
//! The full solver-in-loop signal fires ONLY inside self-play (no per-move unit
//! test exercises the wired path end-to-end — same constraint as O1), so a
//! refactor could silently strip any link in the chain with NO behavioural test
//! catching it. These pins assert every load-bearing marker persists, robust to
//! rustfmt re-flow (substring match). Re-anchored `include_str!` targets: the
//! frozen `worker_loop/inner.rs` split PER PHASE into `runner/search_drive.rs`
//! (the per-move solver hook) + `runner/game.rs` (the per-game seeding + the
//! `WorkerParams` destructure); `params.rs` / `config.rs` keep their names.
//!
//! Behavioural correctness lives elsewhere: the solver itself in `mantis-search`
//! (`tactics` tests) and the solver→soft-injection composition in `records.rs`
//! (`dws3_tests`).

const SEARCH: &str = include_str!("../src/runner/search_drive.rs");
const GAME: &str = include_str!("../src/runner/game.rs");
const PARAMS: &str = include_str!("../src/runner/params.rs");
const CONFIG: &str = include_str!("../src/runner/config.rs");

/// Strip Rust line (`//…`) and block (`/* … */`) comments so a source-presence pin
/// verifies LIVE code, not a marker that survives only in a comment. Tracks
/// double-quoted string literals (with `\` escapes) so a `//` or `/*` inside a
/// string is not mistaken for a comment. (The pinned files carry no raw strings or
/// `'"'` char literals, so this minimal scanner is exact for them.)
fn strip_comments(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut in_str = false;
    while i < chars.len() {
        let c = chars[i];
        if in_str {
            out.push(c);
            if c == '\\' && i + 1 < chars.len() {
                out.push(chars[i + 1]);
                i += 2;
                continue;
            }
            if c == '"' {
                in_str = false;
            }
            i += 1;
        } else if c == '"' {
            in_str = true;
            out.push(c);
            i += 1;
        } else if c == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            i += 2;
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i += 2;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// The per-move target-extraction override must (1) gate on the config flag,
/// (2) call the native solver, (3) SOFT-inject the proven win, (4) surface
/// off-window wins (`window_half: None` on the legal_set path), and (5) force the
/// injected row full-search. All five live in `runner/search_drive.rs` now.
#[test]
fn dws3_solver_override_wired_in_search_drive() {
    let search = strip_comments(SEARCH);
    assert!(
        search.contains("ctx.solver_enabled"),
        "solver-in-loop override must be gated on the config-driven enabled flag",
    );
    assert!(
        search.contains("TacticalSolver::new") && search.contains(".prove("),
        "native solver call removed from the training-target extraction path",
    );
    assert!(
        search.contains("ctx.solver_visit_weight"),
        "SOFT visit-injection weight removed (the soft blend is the load-bearing knob)",
    );
    assert!(
        search.contains("window_half: if legal_set"),
        "off-window surfacing removed — the legal_set path must run the solver with \
         window_half=None (D-DECODE); the Dense path keeps the single-window guard",
    );
    assert!(
        search.contains("solver_fired") && search.contains("|| solver_fired"),
        "the solver-injected row must be forced full-search (record_full_search) so PCR's \
         full_search_mask cannot drop the injected policy target",
    );
}

/// The knob must thread end-to-end: config builder → `WorkerParams` sub-bundle →
/// the worker-thread destructure (`runner/game.rs`).
#[test]
fn dws3_solver_knobs_thread_end_to_end() {
    let config = strip_comments(CONFIG);
    let params = strip_comments(PARAMS);
    let game = strip_comments(GAME);
    for field in [
        "solver_enabled",
        "solver_depth",
        "solver_node_budget",
        "solver_neighbor_dist",
        "solver_visit_weight",
    ] {
        assert!(
            config.contains(field),
            "D-WS3 solver config knob `{field}` removed from SelfPlayRunnerConfig",
        );
    }
    assert!(
        params.contains("struct SolverInLoop"),
        "the SolverInLoop WorkerParams sub-bundle was removed (knobs no longer reach workers)",
    );
    assert!(
        game.contains("solver_in_loop: SolverInLoop") || game.contains("SolverInLoop {"),
        "the worker-thread entry no longer destructures the SolverInLoop bundle",
    );
}

/// D-WS3V3 — the in-run fire-rate counters, start-position seeding, and the
/// relative-ply Gumbel-explore gate must stay wired. The per-move counters live in
/// `search_drive.rs`; `seeded_games_started`, the seed-corpus replay hook, and the
/// relative gate span `search_drive.rs` + `game.rs`.
#[test]
fn dws3v3_seeding_and_counters_wired() {
    let search = strip_comments(SEARCH);
    let game = strip_comments(GAME);
    let params = strip_comments(PARAMS);
    let config = strip_comments(CONFIG);
    let search_and_game = format!("{search}\n{game}");
    for marker in [
        "solver_counters.moves_eligible",
        "solver_counters.win_proven",
        "solver_counters.injected",
        "solver_counters.injected_offwindow",
        "solver_counters.budget_exhausted",
        "solver_counters.seeded_games_started",
    ] {
        assert!(
            search_and_game.contains(marker),
            "D-WS3V3 counter increment `{marker}` removed from the worker loop",
        );
    }
    assert!(
        game.contains("seed.corpus") && game.contains("seed.seed_fraction"),
        "the seed-corpus replay hook (rng drawn ONLY when corpus non-empty) was removed",
    );
    assert!(
        search.contains("relative_explore_gate"),
        "the relative-ply Gumbel-explore gate (D-ARGMAX dup-trap fix) was removed",
    );
    assert!(
        params.contains("struct SeedCorpus"),
        "the SeedCorpus WorkerParams sub-bundle was removed (seeding no longer reaches workers)",
    );
    for field in ["seed_fraction", "seed_corpus"] {
        assert!(
            config.contains(field),
            "D-WS3V3 seeding config knob `{field}` removed from SelfPlayRunnerConfig",
        );
    }
}
