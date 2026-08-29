//! P-06 — rotation parity: the per-game D6 `sym_idx` at the 3 sites keeps the MCTS
//! tree in the CANONICAL frame while the recorded frame is rotated; the graph path
//! is rotation-free at inference (⊕ NEW).
//!
//! The three sites (D14): (1) input forward-scatter BEFORE inference, (2) policy
//! inverse-scatter AFTER inference, (3) record + aux forward-scatter at finalize.
//! Sites 1 and 2 compose to the identity on the MCTS view: forward by `sym_idx`
//! then inverse by `inv_sym_idx(sym_idx)` restores the canonical frame — so the
//! tree never sees a rotated board. This file proves the identity over the PUBLIC
//! WP5 sym kernels (`apply_symmetry_state` / `rotate_axial`, which the runner's
//! `rotate_*_inplace` wrappers call), pins the 3 sites' wiring by source-presence,
//! and shows the graph build is rotation-free. The actual `pub(crate)`
//! `rotate_*_inplace` wrappers are exercised directly by the in-src parity tests in
//! `runner/rotate.rs`.
//!
//! LAW-07 bite proof: a DESYNCED inverse site (wrong index) MUST diverge — the
//! empty-board / `sym_idx == 0` identity is a fixed point and does not bite on its
//! own, so the desync proves the parity actually couples the sites.

use mantis_selfplay::queues::build_leaf_graph;
use mantis_selfplay::replay::sample::apply_symmetry_state;
use mantis_selfplay::replay::sym::{rotate_axial, SymTables, N_SYMS};

/// The dihedral group inverse (frozen `rotate.rs:54`; a pure function the runner's
/// `inv_sym_idx` wrapper implements). Pure rotations invert to `(6 - s) % 6`;
/// reflective elements are self-inverse.
fn inv_sym(s: usize) -> usize {
    if s < 6 {
        (6 - s) % 6
    } else {
        s
    }
}

fn flat(q: i32, r: i32, bs: i32, half: i32) -> usize {
    ((q + half) * bs + (r + half)) as usize
}

/// Interior coords whose cube-radius stays ≤ the window half (9), so they never
/// rotate out of the window under any of the 12 symmetries → the round-trip is a
/// clean identity on them.
fn interior_coords() -> Vec<(i32, i32)> {
    (-3..=3)
        .flat_map(|q| (-3..=3).map(move |r| (q, r)))
        .collect()
}

/// Site-1 forward-scatter ∘ site-2 inverse-scatter is the identity on the interior
/// (the canonical-view invariant), for every one of the 12 symmetries. Uses `u32`
/// element values (the kernel is generic over `T: Copy`) for exact equality.
#[test]
fn forward_then_inverse_scatter_is_identity_on_interior() {
    let t = SymTables::new(); // v6 shape: 19×19, n_cells = 361
    let bs = t.board_size as i32;
    let half = (bs - 1) / 2;
    let n = t.n_cells;
    let coords = interior_coords();

    let mut buf = vec![0u32; n];
    for (k, &(q, r)) in coords.iter().enumerate() {
        buf[flat(q, r, bs, half)] = (k as u32) + 1;
    }

    for s in 0..N_SYMS {
        let mut fwd = vec![0u32; n];
        apply_symmetry_state(&buf, &mut fwd, s, &t);
        let mut back = vec![0u32; n];
        apply_symmetry_state(&fwd, &mut back, inv_sym(s), &t);
        for &(q, r) in &coords {
            let i = flat(q, r, bs, half);
            assert_eq!(
                back[i], buf[i],
                "sym {s}: interior coord ({q},{r}) not restored by forward∘inverse",
            );
        }
    }
}

/// Coordinate-level inverse: `rotate_axial` composed with its `inv_sym` inverse is
/// the identity for ALL coordinates (no window drop at the coordinate level).
#[test]
fn rotate_axial_forward_inverse_is_identity() {
    for s in 0..N_SYMS {
        for q in -6..=6 {
            for r in -6..=6 {
                let (rq, rr) = rotate_axial(q, r, s);
                let (bq, br) = rotate_axial(rq, rr, inv_sym(s));
                assert_eq!(
                    (bq, br),
                    (q, r),
                    "sym {s}: rotate_axial not inverted by inv_sym({s})"
                );
            }
        }
    }
}

/// `sym_idx == 0` is the identity scatter (empty-board / no-rotation forces
/// identity — the runner short-circuits it).
#[test]
fn sym_idx_zero_is_identity() {
    let t = SymTables::new();
    let n = t.n_cells;
    let buf: Vec<u32> = (0..n as u32).collect();
    let mut out = vec![0u32; n];
    apply_symmetry_state(&buf, &mut out, 0, &t);
    assert_eq!(out, buf, "sym_idx=0 must be the identity scatter");
}

/// LAW-07 bite proof: applying the WRONG inverse index at site 2 (here: the forward
/// scatter AGAIN instead of `inv_sym(s)`) MUST diverge from the canonical frame.
/// This proves the parity bites on a site desync, not merely on the identity
/// fixed point.
#[test]
fn desynced_inverse_site_diverges() {
    let t = SymTables::new();
    let bs = t.board_size as i32;
    let half = (bs - 1) / 2;
    let n = t.n_cells;
    let coords = interior_coords();

    let mut buf = vec![0u32; n];
    for (k, &(q, r)) in coords.iter().enumerate() {
        buf[flat(q, r, bs, half)] = (k as u32) + 1;
    }

    let s = 1usize; // 60° rotation; inv = 5; applying s twice = 120° ≠ identity.
    assert_ne!(inv_sym(s), s, "sym 1 is not self-inverse");
    let mut fwd = vec![0u32; n];
    apply_symmetry_state(&buf, &mut fwd, s, &t);
    let mut wrong = vec![0u32; n];
    apply_symmetry_state(&fwd, &mut wrong, s, &t); // WRONG index (desync)
    assert_ne!(
        wrong, buf,
        "a desynced inverse site must diverge from the canonical frame"
    );
}

/// Graph-path inference is ROTATION-FREE: `build_leaf_graph` takes NO `sym_idx`
/// (v1 coord pre-rotation is WP5 sample-time augmentation, not inference), so the
/// built graph is always the canonical frame — two builds of the same leaf are
/// byte-identical, and no rotation state can perturb the coords/edges.
#[test]
fn graph_build_is_rotation_free_and_deterministic() {
    let stones = vec![
        (0i64, 0, 1),
        (2, 0, 1),
        (0, 3, -1),
        (30, 0, -1),
        (31, 0, -1),
    ];
    let g1 = build_leaf_graph(&stones, 1, 2, 6, 6, 19).expect("leaf builds");
    let g2 = build_leaf_graph(&stones, 1, 2, 6, 6, 19).expect("leaf builds");
    assert_eq!(
        g1.node_coords, g2.node_coords,
        "graph build must be deterministic (rotation-free)"
    );
    assert_eq!(
        g1.edge_index, g2.edge_index,
        "graph edges must be rotation-free"
    );
    assert_eq!(
        g1.window_center, g2.window_center,
        "graph window centre must be rotation-free"
    );
}

// ── source-presence of the 3 rotation sites + graph rotation-free ───────────────
const SEARCH: &str = include_str!("../src/runner/search_drive.rs");
const RECORD: &str = include_str!("../src/runner/record.rs");
const FINALIZE: &str = include_str!("../src/runner/finalize.rs");

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

/// The 3 sites must stay wired, and the graph inference must pass NO `sym_idx` to
/// the builder. Substring pins (robust to rustfmt re-flow). The wiring pins match
/// against comment-STRIPPED source so a marker surviving only in a comment cannot
/// give false assurance; the trailing `rotation-free` doc-marker is an English
/// phrase (never a code token) so it stays matched against the raw source.
#[test]
fn three_rotation_sites_wired_and_graph_is_rotation_free_at_inference() {
    let search = strip_comments(SEARCH);
    let record = strip_comments(RECORD);
    let finalize = strip_comments(FINALIZE);
    // Site 1 — input forward-scatter BEFORE inference (dense path).
    assert!(
        search.contains("rotate_state_inplace(&mut buffer, infer.sym_idx"),
        "site 1 (input forward-scatter before inference) removed from search_drive.rs",
    );
    // Site 2 — policy inverse-scatter AFTER inference (dense path).
    assert!(
        search.contains("rotate_policy_inplace(&mut p, infer.inv_idx"),
        "site 2 (policy inverse-scatter after inference) removed from search_drive.rs",
    );
    // Site 3a — record-time forward-scatter of state/chain/policy.
    assert!(
        record.contains("rotate_state_inplace")
            && record.contains("rotate_chain_inplace")
            && record.contains("rotate_policy_inplace"),
        "site 3 (record-time forward-scatter of state/chain/policy) removed from record.rs",
    );
    // Site 3b — aux forward-scatter at finalize.
    assert!(
        finalize.contains("rotate_aux_inplace"),
        "site 3 (aux forward-scatter at finalize) removed from finalize.rs",
    );
    // Graph rotation-free at inference: builder called with NO sym argument.
    assert!(
        search.contains(
            "build_leaf_graph(&stones, current_player, moves_remaining, win_length, radius, agg_trunk_sz)"
        ),
        "graph build call must pass NO sym_idx (rotation-free at inference)",
    );
    // Doc-marker (English phrase, never a code token) — matched on RAW source, since
    // stripping comments is exactly what would (correctly) remove it.
    assert!(
        SEARCH.contains("rotation-free"),
        "graph rotation-free-at-inference marker removed from search_drive.rs",
    );
}
