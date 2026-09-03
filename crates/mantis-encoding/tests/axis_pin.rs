//! AUDIT-1 F-42 — the hex axis basis is ONE ordered table, wherever it is typed.
//!
//! THE DEFECT. The three hex axes were typed SIX times: `mantis_core::board::HEX_AXES`,
//! `mantis_graph::WIN_AXES`, `mantis_selfplay::replay::sym::HEX_BASIS` ("mirroring the board
//! HEX_AXES"), and three Python copies. `validate.rs` pinned `win_axes == WIN_AXES.len()` —
//! the COUNT — and nothing pinned the ORDER, which is what is load-bearing: chain planes are
//! laid out in `HEX_AXES` order, edge one-hots in `WIN_AXES` order, and `axis_perm` is derived
//! from the basis. Swap two entries in one copy and every consumer keeps compiling while the
//! meaning of axis index 1 silently changes on one seam and not the others.
//!
//! WHY A PIN AND NOT ONE OWNER. `mantis-core` and `mantis-graph` are BOTH declared dep-free
//! roots by `repo_design.md` §2 (`mantis-core → (nothing in-workspace)`,
//! `mantis-graph → (nothing; dep-free, wasm32-clean)`), so neither can import the other and
//! one owner is not reachable without a §2 amendment. This crate is the lowest node that sees
//! both, so agreement-or-raise (R104's shape) is the mechanism actually available. The
//! `sym::HEX_BASIS` copy — in a crate that DOES depend on core — is deleted rather than pinned.
//!
//! PLANTED BREAK (verified): swap two entries of either table and
//! `hex_axes_agree_across_crates` reds.

use mantis_core::board::{HEX_AXES, WIN_LENGTH};
use mantis_graph::WIN_AXES;

#[test]
fn hex_axes_agree_across_crates() {
    assert_eq!(
        HEX_AXES, WIN_AXES,
        "mantis_core::board::HEX_AXES and mantis_graph::WIN_AXES are the SAME ordered table. \
         The index is the meaning: chain planes are in HEX_AXES order and edge one-hots in \
         WIN_AXES order, so a reordering on one side relabels one seam and not the other."
    );
}

#[test]
fn the_axis_order_itself_is_pinned_not_just_the_agreement() {
    // Equality alone would stay green if BOTH tables were reordered together — and both are
    // baked into every frozen fixture and every trained checkpoint. This is the value.
    assert_eq!(
        HEX_AXES,
        [(1, 0), (0, 1), (1, -1)],
        "axis 0 = E/W, axis 1 = NE/SW, axis 2 = SE/NW. Every graph-parity golden, every Q13 \
         augmentation table and every trained net was built against THIS order."
    );
}

#[test]
fn the_registry_refuses_a_win_length_the_engine_does_not_play() {
    use mantis_encoding::parse_encoding_toml;

    // AUDIT-1 F-42. The validator used to check `win_length` "present and positive", so a 7
    // loaded cleanly while `Board::player_wins`, `CHAIN_CAP` and the bridge threat scan all
    // went on playing 6.
    let body = graph_body().replace("win_length              = 6", "win_length              = 7");
    let err = parse_encoding_toml("gnn_wlprobe", &body)
        .expect_err("a win_length the engine does not play must be REFUSED");
    assert!(err.contains("win_length"), "the error must name the field: {err}");
    assert!(err.contains(&WIN_LENGTH.to_string()), "…and the value it requires: {err}");
    // The control: the shipped 6 still loads.
    assert!(parse_encoding_toml("gnn_wlok", &graph_body()).is_ok());
}

fn graph_body() -> String {
    r#"
representation          = "graph"
board_size              = 19
trunk_size              = 19
cluster_window_size     = "none"
cluster_threshold       = "none"
legal_move_radius       = 6
n_planes                = 0
plane_layout            = []
policy_logit_count      = 362
has_pass_slot           = true
is_multi_window         = false
value_pool              = "none"
policy_pool             = "none"
sym_table_id            = "size_19"
kept_plane_indices      = []
n_source_planes         = 0
k_max                   = 1
node_feat_dim           = 11
edge_feat_dim           = 5
win_length              = 6
graph_radius            = 6
win_axes                = 3
contract_version        = 1
builder_impl_required   = 1
n_chain_planes          = 6
schema_version          = 4
notes                   = "test"
"#
    .to_string()
}

/// AUDIT-1 F-41. `mantis_graph::BuildParams` used to carry `impl Default`, so a call site
/// writing `BuildParams::default()` meant one specific registry row while reading as "the usual
/// geometry" — and a second graph row (`gnn_axis_r8`) exists that differs in `radius`. The
/// `Default` is gone and `V1_GEOMETRY` names the row; this pins the name to the row's own values.
///
/// It lives HERE because `mantis-graph` is dep-free by `repo_design` §2 and cannot read the
/// registry: this is the lowest crate that sees both. Planted break: change any field of
/// `V1_GEOMETRY` and this reds.
#[test]
fn the_named_build_geometry_equals_the_registry_row_it_names() {
    let spec = mantis_encoding::lookup("gnn_axis_v1").expect("gnn_axis_v1 registered");
    let params = mantis_graph::BuildParams::V1_GEOMETRY;
    assert_eq!(
        params.win_length as usize,
        spec.win_length.expect("a graph row states win_length"),
        "BuildParams::V1_GEOMETRY.win_length disagrees with the row it is named for"
    );
    assert_eq!(
        params.radius as usize,
        spec.graph_radius.expect("a graph row states graph_radius"),
        "BuildParams::V1_GEOMETRY.radius disagrees with the row it is named for"
    );
    assert_eq!(
        params.trunk_size as usize, spec.trunk_size,
        "BuildParams::V1_GEOMETRY.trunk_size disagrees with the row it is named for"
    );
}

/// The other direction: the r8 row is NOT the named constant, so a test that wants r8 must say
/// so. Without this, "V1_GEOMETRY" could drift into meaning whatever the registry's first graph
/// row happens to be, which is the ordering dependence F-41 is about.
#[test]
fn the_named_geometry_is_not_silently_every_graph_row() {
    let r8 = mantis_encoding::lookup("gnn_axis_r8").expect("gnn_axis_r8 registered");
    assert_ne!(
        mantis_graph::BuildParams::V1_GEOMETRY.radius as usize,
        r8.graph_radius.expect("a graph row states graph_radius"),
        "the two registered graph rows now share a radius, so the named constant no longer \
         distinguishes them — re-read whether every `V1_GEOMETRY` site still means v1"
    );
}

/// AUDIT-1 F-49. `mantis-graph`'s dense-parity test replicates the predecessor engine's legal
/// ball at a hand-typed `radius = 5`, deliberately — it is dep-free by `repo_design` §2 and an
/// import would make it the same authority twice. What did not exist is a check that the
/// replicated number still equals the constant it replicates. This is that check, in the lowest
/// crate that sees both. Planted break: change `DEFAULT_LEGAL_MOVE_RADIUS` and this reds with
/// the graph-side literal named.
#[test]
fn the_graph_parity_arms_replicated_radius_still_equals_the_engine_default() {
    assert_eq!(
        mantis_core::board::DEFAULT_LEGAL_MOVE_RADIUS, 5,
        "mantis-graph's `dense_parity` arm replicates this value as a literal `5i32` \
         (crates/mantis-graph/src/lib.rs, `let radius = 5i32`). It is dep-free and cannot \
         import the constant, so moving the constant means moving that literal too."
    );
}

/// The hex-ball formula has ONE home (`mantis_core::board::hex_ball_cells`); this drives it on
/// the radii the suites use, so the formula itself is pinned rather than only its consumers.
#[test]
fn the_hex_ball_formula_is_pinned_at_the_radii_the_suites_use() {
    use mantis_core::board::hex_ball_cells;
    assert_eq!(hex_ball_cells(0), 1);
    assert_eq!(hex_ball_cells(1), 7);
    assert_eq!(hex_ball_cells(4), 61);
    assert_eq!(hex_ball_cells(5), 91);
    assert_eq!(hex_ball_cells(6), 127);
    assert_eq!(hex_ball_cells(8), 217, "the r8 row's radius");
}
