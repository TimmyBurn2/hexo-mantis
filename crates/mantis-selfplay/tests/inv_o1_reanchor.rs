//! P-11 — inv_o1 re-anchor: O1 forced-win → one-hot POLICY target wiring
//! source-presence pins (RE-ANCHOR of `inv_o1_forced_win_one_hot_wiring.rs`).
//!
//! O1 hardens the TRAINING policy target to a (near-)one-hot on a proven
//! within-turn forced win the soft visit distribution under-weights. The full
//! signal fires ONLY inside self-play, so a static-analysis "clean up" could strip
//! any link — detection, blend, or the full-search forcing — with NO behavioural
//! test catching it. Substring pins, robust to rustfmt re-flow.
//!
//! Re-anchored `include_str!` targets: the frozen `worker_loop/inner.rs` override
//! moved to `runner/search_drive.rs`; the one-hot blend helper stayed in the
//! selfplay `records.rs`; the detector primitives moved to `mantis-core`
//! (`board/moves.rs`); the config knobs stayed on `runner/config.rs`.
//!
//! Behavioural correctness lives elsewhere: `forced_win_move` in
//! `mantis-core/board/moves.rs` and `apply_forced_win_one_hot` + the
//! aggregate-survival pin in `records.rs`.

const SEARCH: &str = include_str!("../src/runner/search_drive.rs");
const RECORDS: &str = include_str!("../src/records.rs");
const MOVES: &str = include_str!("../../mantis-core/src/board/moves.rs");
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

/// The per-move target-extraction override must (1) detect the forced win,
/// (2) blend the one-hot into the target, (3) force the hardened row full-search,
/// and (4) gate on the config flag. All four live in `runner/search_drive.rs`.
#[test]
fn o1_target_override_wired_in_search_drive() {
    let search = strip_comments(SEARCH);
    assert!(
        search.contains("forced_win_move"),
        "O1 forced-win detection removed from the training-target extraction path",
    );
    assert!(
        search.contains("apply_forced_win_one_hot"),
        "O1 one-hot blend removed from the training-target extraction path",
    );
    assert!(
        search.contains("record_full_search"),
        "O1 must force the hardened row full-search (else PCR's full_search_mask silently \
         drops ~half the forced-win one-hot targets from the policy loss)",
    );
    assert!(
        search.contains("ctx.forced_win_enabled"),
        "O1 override must be gated on the config-driven enabled flag",
    );
}

/// The detector + blend primitives must exist where the override expects them:
/// the forced-win detector in `mantis-core/board/moves.rs`, the one-hot blend in
/// the selfplay `records.rs`.
#[test]
fn o1_primitives_present() {
    let moves = strip_comments(MOVES);
    let records = strip_comments(RECORDS);
    assert!(
        moves.contains("fn forced_win_move") && moves.contains("fn first_winning_move"),
        "O1 forced-win detector primitives removed from mantis-core board/moves.rs",
    );
    assert!(
        records.contains("fn apply_forced_win_one_hot"),
        "O1 one-hot blend helper removed from records.rs",
    );
}

/// The config knobs must remain on the builder so the WP8 schema → runner can
/// drive O1. Source-of-truth discipline: no literal weights/depths in Rust.
#[test]
fn o1_config_knobs_present() {
    let config = strip_comments(CONFIG);
    for field in [
        "forced_win_policy_enabled",
        "forced_win_policy_depth",
        "forced_win_policy_weight",
    ] {
        assert!(
            config.contains(field),
            "O1 config knob `{field}` removed from SelfPlayRunnerConfig",
        );
    }
}
