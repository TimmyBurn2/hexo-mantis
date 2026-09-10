//! P-06 — rotation parity, GRAPH half: the axis-graph inference path is rotation-free.
//!
//! The dense half of this file (the three window-scatter sites — input forward-scatter
//! before inference, policy inverse-scatter after it, record/aux forward-scatter at
//! finalize — and the `SymTables` scatter kernels they drove) went with the grid path
//! (R346(f)). Their source-presence pin named functions that no longer exist and would
//! have asserted a deleted mechanism into existence, so it is gone rather than relaxed.
//!
//! What remains is the claim that never depended on a window: `build_leaf_graph` takes NO
//! `sym_idx` (coord pre-rotation is HEXG sample-time augmentation, not inference), so the
//! built graph is always the canonical frame, and `rotate_axial` — the primitive the HEXG
//! sample path rotates with — is inverted exactly by `inv_sym`. Proved two ways that must
//! not be separated: numerically, and by reading the builder's ARGUMENT LIST at the call
//! site, because a builder that is rotation-free and unreached passes the numeric half
//! alone.

use mantis_selfplay::queues::build_leaf_graph;
use mantis_selfplay::replay::sym::{rotate_axial, N_SYMS};

/// The dihedral group inverse (a pure function). Pure rotations invert to `(6 - s) % 6`;
/// reflective elements are self-inverse.
fn inv_sym(s: usize) -> usize {
    if s < 6 {
        (6 - s) % 6
    } else {
        s
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

/// Graph-path inference is ROTATION-FREE: `build_leaf_graph` takes NO `sym_idx`, so the
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

/// The argument tokens of the last CALL to `name` in `src`, comma-split at paren
/// depth 0 with whitespace collapsed. `None` when `name` is never called.
///
/// A call is distinguished from the `use` import and from a string literal mentioning
/// the name by requiring the `(` and by taking the LAST occurrence — the import sits
/// above every call site. Nested calls and tuples inside an argument are handled by
/// the depth counter; the pinned call has neither, and a future one that did would
/// still split correctly.
fn call_args(src: &str, name: &str) -> Option<Vec<String>> {
    let open = src.rfind(&format!("{name}("))? + name.len() + 1;
    let bytes: Vec<char> = src[open..].chars().collect();
    let mut depth = 0i32;
    let mut end = None;
    for (i, &c) in bytes.iter().enumerate() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' if depth == 0 => {
                end = Some(i);
                break;
            }
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
    }
    let inner: String = bytes[..end?].iter().collect();
    let mut args = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in inner.chars() {
        match c {
            '(' | '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                args.push(cur.split_whitespace().collect::<Vec<_>>().join(" "));
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    let tail = cur.split_whitespace().collect::<Vec<_>>().join(" ");
    if !tail.is_empty() {
        args.push(tail);
    }
    Some(args)
}

const SEARCH: &str = include_str!("../src/runner/search_drive.rs");

/// Strip Rust line (`//…`) and block (`/* … */`) comments so a source-presence pin
/// verifies LIVE code, not a marker that survives only in a comment. Tracks
/// double-quoted string literals (with `\` escapes) so a `//` or `/*` inside a
/// string is not mistaken for a comment. (The pinned file carries no raw strings or
/// `'"'` char literals, so this minimal scanner is exact for it.)
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

/// The graph inference must pass NO `sym_idx` to the builder.
///
/// Read as an ARGUMENT LIST, not as a source line. The literal one-line form this
/// used to `contains` was a hostage to rustfmt: GUMBEL-REPAIR-1 touched this file,
/// rustfmt split the call across seven lines, and the pin reported the rotation-free
/// property BROKEN while the call was unchanged — the same false red R345 §4d.2
/// recorded on `inv_dws3_reanchor.rs`. `call_args` is insensitive to exactly what
/// formatting moves (whitespace between argument tokens) and to nothing else: a
/// seventh argument, a renamed argument or a reordering all still red.
#[test]
fn graph_build_call_passes_no_sym_idx() {
    let search = strip_comments(SEARCH);
    assert_eq!(
        call_args(&search, "build_leaf_graph"),
        Some(vec![
            "&stones".to_string(),
            "current_player".to_string(),
            "moves_remaining".to_string(),
            "win_length".to_string(),
            "radius".to_string(),
            "agg_trunk_sz".to_string(),
        ]),
        "graph build call must pass NO sym_idx (rotation-free at inference)",
    );
    // Doc-marker (English phrase, never a code token) — matched on RAW source, since
    // stripping comments is exactly what would (correctly) remove it.
    assert!(
        SEARCH.contains("rotation-free"),
        "graph rotation-free-at-inference marker removed from search_drive.rs",
    );
}
