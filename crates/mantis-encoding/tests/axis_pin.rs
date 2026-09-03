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
