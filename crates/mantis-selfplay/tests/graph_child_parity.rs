// R8 >300 justify: one oracle — the self-play call sequence executed verbatim — plus the
// flat-fixture reader it needs. Splitting the reader out would separate it from the ONE test
// that proves it reads the committed fixture correctly.
//! The SELF-PLAY leg of graph child parity: self-play semantics is THE authority.
//!
//! Runs the self-play graph call sequence VERBATIM —
//!
//!     build_leaf_graph -> logit_rule -> f32 segment softmax
//!       -> assemble_ls_from_gnn_probs -> MCTSTree::new_full / new_game / select_leaves(1)
//!       -> expand_and_backup_ls_at(&[ls], &[v], &[g.window_center], trunk)
//!
//! — and compares the root-child set against a COMMITTED fixture the Python leg binds to as
//! well. One fixture, both sides of the FFI, with the production expand running for real.
//!
//! `expected_children` is a self-play-authored golden, so what the mutations must show is
//! SENSITIVITY: a sum-preserving `legal_probs` swap ACROSS the per-node cap boundary, applied
//! in the SHARED producer, leaves both legs valid and both disagreeing with the constant.

use mantis_core::{Board, BoardGeometry};
use mantis_encoding::lookup;
use mantis_search::{MCTSTree, MAX_CHILDREN_PER_NODE, VIRTUAL_LOSS_PENALTY};
use mantis_selfplay::queues::build_leaf_graph;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;

/// Cross-language prior tolerance (DESIGN §b.3 P-1c). Measured at mint: the largest
/// disagreement between this leg and the torch-f32 producer leg is 7.3e-10.
const PRIOR_TOL: f64 = 1e-5;

/// The ONE flat index boundary between the trunk window and the overflow half:
/// `policy_logit_count` is 362 and `window_flat_idx` returns `usize::MAX` off-window,
/// so `flat >= 361` is exactly "off-window" (the Python leg's `board.to_flat` test).
const OFF_WINDOW_FLAT: usize = 361;

// The fixtures are minted FLAT — every value a number, a short string or a flat array — so
// this reader suffices and the crate needs no JSON dependency.

fn fixture_text(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/eval_selfplay_parity")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture {} unreadable: {e}", path.display()))
}

/// The raw text of the value that follows `"key":`. Panics (never defaults) on absence —
/// a fixture that lost a key must fail loudly, not silently read a zero.
fn value_of<'a>(src: &'a str, key: &str) -> &'a str {
    let needle = format!("\"{key}\":");
    let at = src
        .find(&needle)
        .unwrap_or_else(|| panic!("fixture key {key:?} absent"));
    let rest = src[at + needle.len()..].trim_start();
    if let Some(inner) = rest.strip_prefix('[') {
        let end = inner
            .find(']')
            .unwrap_or_else(|| panic!("unterminated array for {key:?}"));
        &inner[..end]
    } else if let Some(inner) = rest.strip_prefix('"') {
        let end = inner
            .find('"')
            .unwrap_or_else(|| panic!("unterminated string for {key:?}"));
        &inner[..end]
    } else {
        let end = rest.find([',', '\n', '}']).unwrap_or(rest.len());
        rest[..end].trim_end()
    }
}

fn ints(src: &str, key: &str) -> Vec<i64> {
    value_of(src, key)
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<i64>()
                .unwrap_or_else(|e| panic!("fixture {key:?}: {t:?} is not an integer ({e})"))
        })
        .collect()
}

fn floats(src: &str, key: &str) -> Vec<f64> {
    value_of(src, key)
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("fixture {key:?}: {t:?} is not a float ({e})"))
        })
        .collect()
}

fn scalar(src: &str, key: &str) -> i64 {
    let raw = value_of(src, key);
    raw.trim()
        .parse::<i64>()
        .unwrap_or_else(|e| panic!("fixture {key:?}: {raw:?} is not an integer ({e})"))
}

fn text<'a>(src: &'a str, key: &str) -> &'a str {
    value_of(src, key)
}

/// Packed `(q, r)` tie-break key, and the fixture's canonical ordering, so both legs compare
/// aligned `(coord, prior)` sequences rather than sets that print in the same order.
fn packed(q: i32, r: i32) -> u32 {
    (((q + 32768) as u32) << 16) | ((r + 32768) as u32 & 0xFFFF)
}

fn pairs(flat: &[i64]) -> Vec<(i32, i32)> {
    assert_eq!(flat.len() % 2, 0, "coord array must be pairs");
    flat.chunks_exact(2)
        .map(|c| (c[0] as i32, c[1] as i32))
        .collect()
}

/// Run one fixture position through the production self-play path and assert the frozen
/// child set (and, when the fixture carries them, the frozen priors).
fn check_position(src: &str, i: usize, with_priors: bool) {
    let spec = lookup("gnn_axis_v1").expect("gnn_axis_v1 is a registered encoding");
    let key = |suffix: &str| format!("p{i}_{suffix}");
    let id = text(src, &key("id")).to_string();

    // Replay at the SPEC's geometry — the construction `Board.with_encoding_name` performs on
    // the Python side, so the two legs share one board.
    let geom = BoardGeometry {
        legal_move_radius: spec.legal_move_radius as i32,
        cluster_threshold: spec.cluster_threshold.unwrap_or(5) as i32,
        cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
    };
    let mut board = Board::with_geometry(geom);
    for (q, r) in pairs(&ints(src, &key("moves"))) {
        board
            .apply_move(q, r)
            .unwrap_or_else(|e| panic!("{id}: fixture move ({q},{r}) rejected: {e}"));
    }

    // Every recorded field is re-derived from the replayed board, so a drifting fixture fails
    // here rather than shifting the oracle.
    assert_eq!(
        i64::from(board.current_player as i8),
        scalar(src, &key("current_player")),
        "{id}: current_player drift"
    );
    assert_eq!(
        i64::from(board.moves_remaining),
        scalar(src, &key("moves_remaining")),
        "{id}: moves_remaining drift"
    );
    let legal = board.legal_moves();
    assert_eq!(
        legal.len() as i64,
        scalar(src, &key("n_legal")),
        "{id}: n_legal drift"
    );
    let n_off = legal
        .iter()
        .filter(|&&(q, r)| board.window_flat_idx(q, r) >= OFF_WINDOW_FLAT)
        .count();
    assert_eq!(
        n_off as i64,
        scalar(src, &key("n_off_window")),
        "{id}: n_off_window drift"
    );

    // the leaf graph, built from the leaf's own stones
    let stones: Vec<(i64, i64, i64)> = board
        .cells_iter()
        .map(|(&(q, r), &c)| (i64::from(q), i64::from(r), c as i64))
        .collect();
    let g = build_leaf_graph(
        &stones,
        i64::from(board.current_player as i8),
        i64::from(board.moves_remaining),
        spec.win_length.expect("graph spec defines win_length") as u8,
        spec.graph_radius.expect("graph spec defines graph_radius") as u16,
        spec.trunk_size as i32,
    )
    .unwrap_or_else(|e| panic!("{id}: leaf graph build failed: {e}"));
    let wc = pairs(&ints(src, &key("window_center")));
    assert_eq!(g.window_center, wc[0], "{id}: builder window_center drift");

    let legal_coords: Vec<(i32, i32)> = g
        .legal_node_gather
        .iter()
        .map(|&row| {
            (
                g.node_coords[row as usize * 2],
                g.node_coords[row as usize * 2 + 1],
            )
        })
        .collect();

    // `logit_rule` over the BUILDER's legal-node index, then the per-graph segmented softmax in
    // f32 — the same normalisation the Python side forces, which is what licenses the 1e-5 tol.
    let n = legal_coords.len();
    let logits: Vec<f32> = (0..n)
        .map(|i| (((i as i64 * 37) % 101) as f32) / 20.0)
        .collect();
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ex: Vec<f32> = logits.iter().map(|l| (l - mx).exp()).collect();
    let denom: f32 = ex.iter().sum();
    let probs: Vec<f32> = ex.iter().map(|e| e / denom).collect();

    // the SHARED producer; the sensitivity mutation edits exactly this call's input
    let ls = assemble_ls_from_gnn_probs(
        spec.policy_logit_count,
        &probs,
        &g.policy_scatter_index.0,
        &legal_coords,
    )
    .unwrap_or_else(|e| panic!("{id}: assemble_ls_from_gnn_probs failed: {e}"));

    // the expand eval reaches through the bridge door
    let mut tree = MCTSTree::new_full(1.5, VIRTUAL_LOSS_PENALTY, 0.25);
    tree.new_game(board.clone());
    let leaves = tree
        .select_leaves(1)
        .expect("select_leaves: no desync in this fixture");
    assert_eq!(
        leaves.len(),
        1,
        "{id}: fresh root must yield one pending leaf"
    );
    tree.expand_and_backup_ls_at(&[ls], &[0.0f32], &[g.window_center], spec.trunk_size as i32);

    // `get_top_visits` returns zero-visit children too, so this is the COMPLETE child set.
    let mut kids = tree.get_top_visits(MAX_CHILDREN_PER_NODE);
    kids.sort_unstable_by_key(|((q, r), _, _, _)| packed(*q, *r));
    let got: Vec<(i32, i32)> = kids.iter().map(|(c, _, _, _)| *c).collect();
    let want = pairs(&ints(src, &key("expected_children")));

    assert_eq!(
        got, want,
        "{id}: self-play child set diverged from the frozen fixture"
    );
    let got_off = got
        .iter()
        .filter(|&&(q, r)| board.window_flat_idx(q, r) >= OFF_WINDOW_FLAT)
        .count();
    assert_eq!(
        got_off as i64,
        scalar(src, &key("expected_off_window_children")),
        "{id}: off-window child count drift"
    );

    if with_priors {
        let want_priors = floats(src, &key("expected_child_priors"));
        assert_eq!(want_priors.len(), got.len(), "{id}: prior array length");
        for (idx, ((coord, _, prior, _), want_p)) in kids.iter().zip(&want_priors).enumerate() {
            assert!(
                (f64::from(*prior) - *want_p).abs() <= PRIOR_TOL,
                "{id}: child {idx} {coord:?} prior {prior} != frozen {want_p} (tol {PRIOR_TOL})"
            );
        }
    }
}

#[test]
fn rust_leg_matches_fixture() {
    let src = fixture_text("child_parity_v1.json");
    assert_eq!(scalar(&src, "schema"), 1, "fixture schema");
    assert_eq!(text(&src, "encoding"), "gnn_axis_v1", "fixture encoding");
    let n = scalar(&src, "n_positions");
    assert!(n >= 1, "fixture carries no position");
    for i in 0..n as usize {
        check_position(&src, i, true);
    }
}

#[test]
fn rust_leg_matches_dispersed_fixture() {
    let src = fixture_text("dispersed_r6_v1.json");
    assert_eq!(scalar(&src, "schema"), 1, "fixture schema");
    let n = scalar(&src, "n_positions");
    assert_eq!(
        n, 4,
        "the dispersed fixture is pre-registered at 4 positions"
    );
    for i in 0..n as usize {
        // the >361-legal, has-off-window-moves precondition, asserted on this side too
        assert!(
            scalar(&src, &format!("p{i}_n_legal")) > 361,
            "position {i} left the >361-legal regime"
        );
        assert!(
            scalar(&src, &format!("p{i}_n_off_window")) > 0,
            "position {i} has no off-window legal move"
        );
        check_position(&src, i, false);
    }
}
