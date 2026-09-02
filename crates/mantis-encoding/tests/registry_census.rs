//! R8 justify: one census over one registry, and its rows are meaningful only against each
//! other — the exact-N count, the name set, the deliberately-absent set, the per-field
//! authorities and the malformed-fragment refusals all read the SAME `valid_grid_body`
//! fixture, so a split would leave one file asserting a shape another file's fixture no
//! longer builds.
//! O-4..O-10, O-13 — registry census (pruned set), per-field pins,
//! wire_signature families, spec-derived strides, unknown-key / missing-key /
//! missing-representation parse errors, validator collect-all, n_chain_planes
//! TOML-field authority, registry_sha() export, and the mechanical-delta pins.

#[allow(dead_code)]
mod common;

use std::path::PathBuf;

use mantis_encoding::{
    all_specs, lookup, parse_encoding_toml, registry_sha, registry_sha_hex, PolicyPool,
    Representation, ValuePool, MOVES_REMAINING_PLANE, MY_STONE_PLANE, OPP_STONE_PLANE,
    PLY_PARITY_PLANE,
};

const REGISTERED: [&str; 5] = ["v6", "v6w25", "v6_live2_ls", "gnn_axis_v1", "gnn_axis_r8"];
const ABSENT: [&str; 8] = [
    "v6_live2", "v8", "v8_canvas_realness", "v7full", "v7", "v7e30", "v7mw", "v6tp",
];

// ── O-4: census exact-N + names + ARCH/KILL absence ──────────────────────────
#[test]
fn census_exact_n_and_names() {
    assert_eq!(
        all_specs().count(),
        REGISTERED.len(),
        "the registered set must be exactly the names REGISTERED lists"
    );
    let mut names: Vec<&str> = all_specs().map(|s| s.name).collect();
    names.sort_unstable();
    assert_eq!(names, ["gnn_axis_r8", "gnn_axis_v1", "v6", "v6_live2_ls", "v6w25"]);
    for n in REGISTERED {
        assert!(lookup(n).is_some(), "{n} must be registered");
    }
    for n in ABSENT {
        assert!(lookup(n).is_none(), "{n} must be ABSENT (ARCH/KILL)");
    }
}

// ── O-5: per-field pins ──────────────────────────────────────────────────────
#[test]
fn per_field_pins_v6() {
    let s = lookup("v6").unwrap();
    assert_eq!(s.representation, Representation::Grid);
    assert_eq!(s.board_size, 19);
    assert_eq!(s.trunk_size, 19);
    assert_eq!(s.cluster_window_size, None);
    assert_eq!(s.cluster_threshold, None);
    assert_eq!(s.legal_move_radius, 5);
    assert_eq!(s.n_planes, 8);
    assert_eq!(s.plane_layout.len(), 8);
    assert_eq!(s.policy_logit_count, 362);
    assert!(s.has_pass_slot);
    assert!(!s.is_multi_window);
    assert_eq!(s.value_pool, ValuePool::None);
    assert_eq!(s.policy_pool, PolicyPool::None);
    assert_eq!(s.sym_table_id, "size_19");
    assert_eq!(s.kept_plane_indices, &[0, 1, 2, 3, 8, 9, 10, 11]);
    assert_eq!(s.n_source_planes, 18);
    assert_eq!(s.k_max, 1);
    assert_eq!(s.n_chain_planes, 6);
    assert_eq!(s.schema_version, 3, "schema_version PRESERVED (no bump)");
    assert!(!s.is_graph());
}

#[test]
fn per_field_pins_v6w25() {
    let s = lookup("v6w25").unwrap();
    assert_eq!(s.representation, Representation::Grid);
    assert_eq!(s.board_size, 25);
    assert_eq!(s.trunk_size, 25);
    assert_eq!(s.cluster_window_size, Some(25));
    assert_eq!(s.cluster_threshold, Some(8));
    assert_eq!(s.n_planes, 8);
    assert_eq!(s.policy_logit_count, 626);
    assert!(s.is_multi_window);
    assert_eq!(s.value_pool, ValuePool::Min);
    assert_eq!(s.policy_pool, PolicyPool::ScatterMax);
    assert_eq!(s.sym_table_id, "size_25");
    assert_eq!(s.kept_plane_indices, &[0, 1, 2, 3, 8, 9, 10, 11]);
    assert_eq!(s.k_max, 8);
    assert_eq!(s.n_chain_planes, 6);
    assert_eq!(s.schema_version, 3);
}

#[test]
fn per_field_pins_v6_live2_ls() {
    let s = lookup("v6_live2_ls").unwrap();
    assert_eq!(s.representation, Representation::Grid);
    assert_eq!(s.board_size, 19);
    assert_eq!(s.cluster_window_size, Some(19));
    assert_eq!(s.cluster_threshold, Some(5));
    assert_eq!(s.n_planes, 4);
    assert_eq!(s.policy_logit_count, 362);
    assert!(s.is_multi_window);
    assert_eq!(s.value_pool, ValuePool::Min);
    assert_eq!(s.policy_pool, PolicyPool::LegalSetScatterMax);
    assert_eq!(s.sym_table_id, "size_19");
    assert_eq!(s.kept_plane_indices, &[0, 8, 16, 17]);
    assert_eq!(s.k_max, 8);
    assert_eq!(s.n_chain_planes, 6);
    assert_eq!(s.schema_version, 3);
}

#[test]
fn per_field_pins_gnn_axis_v1() {
    let s = lookup("gnn_axis_v1").unwrap();
    assert_eq!(s.representation, Representation::Graph);
    assert!(s.is_graph());
    assert_eq!(s.board_size, 19);
    assert_eq!(s.n_planes, 0);
    assert!(s.plane_layout.is_empty());
    assert!(s.kept_plane_indices.is_empty());
    assert_eq!(s.n_source_planes, 0);
    assert_eq!(s.policy_logit_count, 362);
    assert!(s.has_pass_slot);
    assert!(!s.is_multi_window);
    assert_eq!(s.node_feat_dim, Some(11));
    assert_eq!(s.edge_feat_dim, Some(5));
    assert_eq!(s.win_length, Some(6));
    assert_eq!(s.graph_radius, Some(6));
    assert_eq!(s.win_axes, Some(3));
    assert_eq!(s.contract_version, Some(1));
    assert_eq!(s.builder_impl_required, Some(1));
    assert_eq!(s.n_chain_planes, 6);
    assert_eq!(s.schema_version, 4, "graph entry schema_version = 4 (preserved)");
    assert_eq!(s.legal_move_radius, 6);
}

// ── O-5: wire_signature families + spec-derived strides ──────────────────────
#[test]
fn wire_signature_families() {
    assert_eq!(lookup("v6").unwrap().wire_signature(), (8, 19, 362, true, "size_19"));
    assert_eq!(lookup("v6w25").unwrap().wire_signature(), (8, 25, 626, true, "size_25"));
    assert_eq!(lookup("v6_live2_ls").unwrap().wire_signature(), (4, 19, 362, true, "size_19"));
    assert_eq!(lookup("gnn_axis_v1").unwrap().wire_signature(), (0, 19, 362, true, "size_19"));
    // v6_live2_ls is the SOLE registered holder of (4,19,362,true,size_19).
    let holders: Vec<&str> = all_specs()
        .filter(|s| s.wire_signature() == (4, 19, 362, true, "size_19"))
        .map(|s| s.name)
        .collect();
    assert_eq!(holders, ["v6_live2_ls"]);
}

#[test]
fn strides_spec_derived() {
    let v6 = lookup("v6").unwrap();
    assert_eq!(v6.n_cells(), 361);
    assert_eq!(v6.state_stride(), 8 * 361);
    assert_eq!(v6.chain_stride(), 6 * 361);
    assert_eq!(v6.aux_stride(), 361);
    assert_eq!(v6.policy_stride(), 362);
    assert_eq!(v6.half(), 9);

    let w = lookup("v6w25").unwrap();
    assert_eq!(w.n_cells(), 625);
    assert_eq!(w.state_stride(), 8 * 625);
    assert_eq!(w.chain_stride(), 6 * 625);
    assert_eq!(w.policy_stride(), 626);
    assert_eq!(w.half(), 12);

    let ls = lookup("v6_live2_ls").unwrap();
    assert_eq!(ls.state_stride(), 4 * 361);
    assert_eq!(ls.chain_stride(), 6 * 361);
    assert_eq!(ls.policy_stride(), 362);

    // Derived slot accessors.
    assert_eq!(v6.cur_stone_slot(), 0);
    assert_eq!(v6.opp_stone_slot(), 4);
    assert_eq!(ls.opp_stone_slot(), 1);
    assert_eq!(ls.turn_phase_planes(), vec![2, 3]);
    assert_eq!(v6.history_planes(), vec![1, 2, 3, 5, 6, 7]);
}

// ── valid TOML body helper (a grid v6-like block) ────────────────────────────
fn valid_grid_body() -> String {
    r#"
representation = "grid"
board_size = 19
trunk_size = 19
cluster_window_size = "none"
cluster_threshold = "none"
legal_move_radius = 5
n_planes = 8
plane_layout = ["a","b","c","d","e","f","g","h"]
policy_logit_count = 362
has_pass_slot = true
is_multi_window = false
value_pool = "none"
policy_pool = "none"
sym_table_id = "size_19"
kept_plane_indices = [0,1,2,3,8,9,10,11]
n_source_planes = 18
k_max = 1
n_chain_planes = 6
schema_version = 3
notes = "test"
"#
    .to_string()
}

// ── O-6: unknown-key reject (collect-all with a co-present missing key) ───────
#[test]
fn unknown_key_rejected_collect_all() {
    // A smuggled unknown key AND a simultaneously-missing required key.
    let body = valid_grid_body()
        .replace("schema_version = 3\n", "schema_version = 3\nsmuggled_key = 7\n")
        .replace("board_size = 19\n", "");
    let err = parse_encoding_toml("v6t", &body).unwrap_err();
    assert!(err.contains("unknown key") && err.contains("smuggled_key"), "unknown-key: {err}");
    assert!(err.contains("board_size"), "co-present missing key must ALSO report: {err}");
}

// ── O-7: missing-key + missing-representation (LAW-11) ────────────────────────
#[test]
fn missing_representation_is_error_not_grid_default() {
    let body = valid_grid_body().replace("representation = \"grid\"\n", "");
    let err = parse_encoding_toml("v6t", &body).unwrap_err();
    assert!(
        err.contains("missing required key") && err.contains("representation"),
        "absent representation must be a named error, NEVER a grid default (LAW-11): {err}"
    );
}

#[test]
fn missing_required_key_is_named_error() {
    let body = valid_grid_body().replace("policy_logit_count = 362\n", "");
    let err = parse_encoding_toml("v6t", &body).unwrap_err();
    assert!(err.contains("policy_logit_count"), "missing key must be named: {err}");
}

// ── O-8: validator collect-all-errors (no short-circuit) ─────────────────────
#[test]
fn validator_collects_all_violations() {
    let mut s = *lookup("v6").unwrap();
    s.n_planes = 9; // len(plane_layout)=8 != 9  AND  len(kept)=8 != 9
    s.has_pass_slot = false; // policy_logit_count 362 != 361
    let err = s.validate().unwrap_err();
    assert!(err.contains("len(plane_layout)"), "violation 1 must appear: {err}");
    assert!(err.contains("policy_logit_count"), "violation 2 must appear: {err}");
    assert!(err.contains("len(kept_plane_indices)"), "violation 3 must appear: {err}");
    // 3 distinct violations in ONE message → collect-all, no short-circuit.
    assert!(err.matches("\n  - ").count() >= 2, "must list multiple violations: {err}");
}

// ── O-9: n_chain_planes TOML-field authority ─────────────────────────────────
#[test]
fn n_chain_planes_is_the_field_authority() {
    for n in REGISTERED {
        assert_eq!(lookup(n).unwrap().n_chain_planes, 6, "{n} ships n_chain_planes=6");
    }
    // The FIELD drives chain_stride (not a source constant / replay reach-through).
    let mut s = *lookup("v6").unwrap();
    s.n_chain_planes = 7;
    assert_eq!(s.chain_stride(), 7 * s.n_cells(), "chain_stride must read the field");
    // The validator rejects 0.
    s.n_chain_planes = 0;
    let err = s.validate().unwrap_err();
    assert!(err.contains("n_chain_planes"), "validator must reject n_chain_planes=0: {err}");
}

// ── O-10: registry_sha() export + stability ──────────────────────────────────
#[test]
fn registry_sha_deterministic_and_matches_on_disk() {
    assert_eq!(registry_sha(), registry_sha(), "registry_sha must be deterministic");
    assert_eq!(registry_sha_hex().len(), 64);
    let on_disk = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/registry.toml");
    let bytes = std::fs::read(&on_disk).unwrap();
    assert_eq!(
        registry_sha_hex(),
        common::sha256_hex(&bytes),
        "registry_sha() must equal sha256 of the on-disk registry.toml"
    );
}

// ── O-13: mechanical-delta pins ──────────────────────────────────────────────
#[test]
fn mechanical_delta_pins() {
    assert_eq!(MY_STONE_PLANE, 0);
    assert_eq!(OPP_STONE_PLANE, 8);
    assert_eq!(MOVES_REMAINING_PLANE, 16);
    assert_eq!(PLY_PARITY_PLANE, 17);
    // board.ply.index() % 2 reproduces the old `ply % 2` on a decoupled state.
    assert_eq!(mantis_core::Ply::new(4).index() % 2, 0, "ply 4 -> ply_val 0");
    assert_eq!(mantis_core::Ply::new(41).index() % 2, 1, "ply 41 -> ply_val 1");
    // chain_stride reads the field (see n_chain_planes_is_the_field_authority).
}

// ── AUDIT-1 F-37: the two registry values `sym_tables_for` would have panicked on ─────
//
// `mantis_selfplay::replay::sym::sym_tables_for` asserts `n_chain_planes == N_CHAIN_PLANES`
// and matches `(sym_table_id, n_planes)` against `("size_19", _) | ("size_25", 8)`, panicking
// on anything else — and it is reached from `SelfPlayRunner::start()`, INCLUDING on a graph
// run where the tables are never read. `spec/validate.rs` checked `n_chain_planes >= 1` and
// never mentioned `sym_table_id`, so a new registry row failed at the first runner start, in
// a worker thread, instead of at load. Leg 3's radius-8 work edits this registry.

#[test]
fn an_unknown_sym_table_id_is_refused_at_LOAD_not_at_the_first_runner_start() {
    let body = valid_grid_body().replace(
        "sym_table_id = \"size_19\"", "sym_table_id = \"size_31\"");
    let err = parse_encoding_toml("v6_symprobe", &body).unwrap_err();
    assert!(err.contains("sym_table_id"), "the error must name the field: {err}");
    assert!(err.contains("size_31"), "…and the value it refused: {err}");
    assert!(err.contains("size_19") && err.contains("size_25"),
        "…and what IS available, or the reader cannot act on it: {err}");
}

#[test]
fn size_25_without_eight_planes_is_refused_because_the_MATCH_is_on_the_pair() {
    // `("size_25", 8)` is one arm. `("size_25", 4)` falls through to the panic — and the
    // pairing is exactly the thing a validator checking each field alone cannot see.
    let body = valid_grid_body()
        .replace("sym_table_id = \"size_19\"", "sym_table_id = \"size_25\"")
        .replace("n_planes = 8", "n_planes = 4")
        .replace(r#"plane_layout = ["a","b","c","d","e","f","g","h"]"#,
                 r#"plane_layout = ["a","b","c","d"]"#);
    let err = parse_encoding_toml("v6_pairprobe", &body).unwrap_err();
    assert!(err.contains("size_25") && err.contains("n_planes"),
        "the error must name the PAIR, not one half of it: {err}");
    // The control: `size_25` with EIGHT planes is a real arm and must still load.
    let ok = valid_grid_body().replace(
        "sym_table_id = \"size_19\"", "sym_table_id = \"size_25\"");
    assert!(parse_encoding_toml("v6_pairok", &ok).is_ok(),
        "the valid (size_25, 8) pair was refused");
}

#[test]
fn a_chain_plane_count_the_D6_tables_cannot_hold_is_refused() {
    let body = valid_grid_body().replace("n_chain_planes = 6", "n_chain_planes = 7");
    let err = parse_encoding_toml("v6_chainprobe", &body).unwrap_err();
    assert!(err.contains("n_chain_planes"), "{err}");
}

#[test]
fn every_SHIPPED_spec_clears_the_two_relations() {
    // The control, and the R98 clean-baseline half: the rule is adopted over a registry that
    // already satisfies it, so nothing here is a green over a tree nobody checked.
    for name in REGISTERED {
        let spec = lookup(name).expect("registered");
        assert!(matches!(spec.sym_table_id, "size_19" | "size_25"),
            "{name}: sym_table_id {:?}", spec.sym_table_id);
        if spec.sym_table_id == "size_25" {
            assert_eq!(spec.n_planes, 8, "{name}: size_25 requires 8 planes");
        }
        assert_eq!(spec.n_chain_planes, 6, "{name}: n_chain_planes");
    }
}
