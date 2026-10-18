// Exceeds the 300-line soft cap (R8): the registered set, each row's field pins and the relations between rows are
// one census, so a row lands or leaves with one edit to one file.

//! O-4..O-10, O-13 — registry census (pruned set), per-field pins on the graph rows and the canvas
//! row, the derived field-diffs that keep `gnn_axis_r8` a one-knob edit of `gnn_axis_v1` and
//! `gnn_axis_r8_pruned` a one-knob edit of `gnn_axis_r8`, and the ring-crossing rule.
//!
//! The three grid rows (`v6`, `v6w25`, `v6_live2_ls`) are pinned here as ABSENT, so a
//! resurrected row fails the census rather than the encoder.

use mantis_encoding::{
    all_specs, lookup, parse_encoding_toml, EmptyEdges, PolicyPool, Representation, ValuePool,
};

const REGISTERED: [&str; 4] = [
    "gnn_axis_v1",
    "gnn_axis_r8",
    "gnn_axis_r8_pruned",
    "hexcanvas_r8",
];
const ABSENT: [&str; 11] = [
    "v6",
    "v6w25",
    "v6_live2_ls",
    "v6_live2",
    "v8",
    "v8_canvas_realness",
    "v7full",
    "v7",
    "v7e30",
    "v7mw",
    "v6tp",
];

#[test]
fn all_specs_iterates_in_name_order() {
    let names: Vec<&str> = all_specs().map(|s| s.name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted, "all_specs must iterate in name order");
}

// O-4: census exact-N + names + ARCH/KILL absence
#[test]
fn census_exact_n_and_names() {
    assert_eq!(
        all_specs().count(),
        REGISTERED.len(),
        "the registered set must be exactly the names REGISTERED lists"
    );
    let mut names: Vec<&str> = all_specs().map(|s| s.name).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "gnn_axis_r8",
            "gnn_axis_r8_pruned",
            "gnn_axis_v1",
            "hexcanvas_r8"
        ]
    );
    for n in REGISTERED {
        assert!(lookup(n).is_some(), "{n} must be registered");
    }
    for n in ABSENT {
        assert!(
            lookup(n).is_none(),
            "{n} must be ABSENT (ARCH/KILL/R346(f))"
        );
    }
}

/// `representation` is the identity key and `"grid"` and `"dense"` are refused BY NAME rather
/// than falling through as unknown, so a stale row says what happened to it.
#[test]
fn representation_parse_refuses_grid_by_name() {
    assert_eq!(Representation::parse("graph"), Ok(Representation::Graph));
    assert_eq!(
        Representation::parse("hexcanvas"),
        Ok(Representation::HexCanvas)
    );
    let err = Representation::parse("grid").expect_err("grid must be refused, never defaulted");
    assert!(
        err.contains("grid") && err.contains("R346(f)"),
        "the refusal must name the deleted representation and the ruling; got {err:?}"
    );
    let err = Representation::parse("dense").expect_err("dense must be refused, never defaulted");
    assert!(
        err.contains("dense") && err.contains("hexcanvas"),
        "the refusal must name the token and the hex-canvas kind's; got {err:?}"
    );
    assert!(
        Representation::parse("canvas").is_err(),
        "an unknown representation is an error, never a default (LAW-11)"
    );
}

// O-5: per-field pins
#[test]
fn per_field_pins_gnn_axis_v1() {
    let s = lookup("gnn_axis_v1").expect("gnn_axis_v1 is a registered row");
    assert_eq!(s.representation, Representation::Graph);
    assert!(s.is_graph());
    assert_eq!(s.board_size, 19);
    assert_eq!(s.trunk_size, 19);
    assert_eq!(s.n_planes, 0);
    assert!(s.plane_layout.is_empty());
    assert!(s.kept_plane_indices.is_empty());
    assert_eq!(s.n_source_planes, 0);
    assert_eq!(s.policy_logit_count, 362);
    assert!(s.has_pass_slot);
    assert!(!s.is_multi_window);
    assert_eq!(s.cluster_window_size, None);
    assert_eq!(s.cluster_threshold, None);
    assert_eq!(s.value_pool, ValuePool::None);
    assert_eq!(s.policy_pool, PolicyPool::None);
    assert_eq!(s.sym_table_id, "size_19");
    assert_eq!(s.k_max, 1);
    assert_eq!(s.node_feat_dim, Some(11));
    assert_eq!(s.edge_feat_dim, Some(5));
    assert_eq!(s.win_length, Some(6));
    assert_eq!(s.graph_radius, Some(6));
    assert_eq!(s.win_axes, Some(3));
    assert_eq!(s.contract_version, Some(1));
    assert_eq!(s.builder_impl_required, Some(1));
    assert_eq!(s.n_chain_planes, 6);
    assert_eq!(s.empty_edges, Some(EmptyEdges::Kept));
    assert_eq!(s.canvas_max_side, None);
    assert_eq!(
        s.schema_version, 6,
        "graph entry schema_version = 6 (the canvas keys)"
    );
    assert_eq!(s.legal_move_radius, 6);
}

/// `gnn_axis_r8`'s own per-field pin: a row with no per-field pin can have its geometry edited
/// without a single assertion moving.
#[test]
fn per_field_pins_gnn_axis_r8() {
    let s = lookup("gnn_axis_r8").expect("gnn_axis_r8 is a registered row");
    assert_eq!(s.representation, Representation::Graph);
    assert!(s.is_graph());
    assert_eq!(s.board_size, 19);
    assert_eq!(s.trunk_size, 19);
    assert_eq!(s.n_planes, 0);
    assert!(s.plane_layout.is_empty());
    assert!(s.kept_plane_indices.is_empty());
    assert_eq!(s.n_source_planes, 0);
    assert_eq!(s.policy_logit_count, 362);
    assert!(s.has_pass_slot);
    assert!(!s.is_multi_window);
    assert_eq!(s.cluster_window_size, None);
    assert_eq!(s.cluster_threshold, None);
    assert_eq!(s.value_pool, ValuePool::None);
    assert_eq!(s.policy_pool, PolicyPool::None);
    assert_eq!(s.sym_table_id, "size_19");
    assert_eq!(s.k_max, 1);
    assert_eq!(s.node_feat_dim, Some(11));
    assert_eq!(s.edge_feat_dim, Some(5));
    assert_eq!(s.win_length, Some(6));
    assert_eq!(
        s.graph_radius,
        Some(8),
        "the row exists to be radius 8 (R328)"
    );
    assert_eq!(s.win_axes, Some(3));
    assert_eq!(s.contract_version, Some(1));
    assert_eq!(s.builder_impl_required, Some(1));
    assert_eq!(s.n_chain_planes, 6);
    assert_eq!(s.empty_edges, Some(EmptyEdges::Kept));
    assert_eq!(s.canvas_max_side, None);
    assert_eq!(
        s.schema_version, 6,
        "graph entry schema_version = 6 (the canvas keys)"
    );
    assert_eq!(
        s.legal_move_radius, 8,
        "the R328 identity: legal_move_radius and graph_radius are ONE number on a graph row, \
         and REPAIR-2's F-18 validator refuses them typed apart"
    );
}

/// The fields two rows' `{:#?}` dumps differ in, DERIVED from the dump so a field added to `RegistrySpec` joins it unasked.
fn differing_fields(a: &str, b: &str) -> Vec<String> {
    let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    assert_eq!(
        la.len(),
        lb.len(),
        "the two dumps have different shapes, so no field-wise diff is possible:\n{a}\n---\n{b}"
    );
    // `{:#?}` breaks an `Option<usize>` over three lines, so a differing line is often the
    // INNER value rather than the field name. Carry the most recent `field:` line seen so a
    // difference is attributed to the field that owns it, and de-duplicate.
    let mut field = String::new();
    let mut differing: Vec<String> = Vec::new();
    for (x, y) in la.iter().zip(lb.iter()) {
        let trimmed = x.trim();
        if let Some((name, _)) = trimmed.split_once(':') {
            if !name.contains(' ') && !name.is_empty() {
                field = name.to_string();
            }
        }
        if x != y && differing.last() != Some(&field) {
            differing.push(field.clone());
        }
    }
    differing.sort_unstable();
    differing.dedup();
    differing
}

fn dump(name: &str) -> String {
    let spec = lookup(name).unwrap_or_else(|| panic!("{name} is registered"));
    let d = format!("{spec:#?}");
    assert!(
        d.lines().count() > 15,
        "the Debug dump collapsed; the field-diff would be vacuous"
    );
    d
}

/// Rust twin of `tests/encoding/test_r8_identity.py::test_r328b_03`: the two graph rows must
/// differ in EXACTLY the radius pair and their identifying strings — `gnn_axis_r8` exists to move
/// ONE knob, so a stray `win_length` or `policy_logit_count` difference makes the r6/r8
/// comparison two comparisons.
#[test]
fn the_two_graph_rows_differ_in_exactly_the_radius_pair() {
    let differing = differing_fields(&dump("gnn_axis_v1"), &dump("gnn_axis_r8"));
    assert_eq!(
        differing,
        vec!["graph_radius", "legal_move_radius", "name", "notes"],
        "the two graph rows differ in {differing:?}. This encoding exists to move ONE knob so \
         the r6/r8 comparison IS a comparison; any other difference makes it two."
    );
}

/// `gnn_axis_r8_pruned` moves ONE knob off `gnn_axis_r8`, so the pruned-v-kept pair compares the edge set alone.
#[test]
fn the_pruned_row_differs_from_r8_in_exactly_its_edge_set() {
    let differing = differing_fields(&dump("gnn_axis_r8"), &dump("gnn_axis_r8_pruned"));
    assert_eq!(
        differing,
        vec!["empty_edges", "name", "notes"],
        "gnn_axis_r8_pruned differs from gnn_axis_r8 in {differing:?}; any field but the edge set \
         makes the pruned-v-kept pair two comparisons"
    );
    assert_eq!(
        lookup("gnn_axis_r8_pruned").map(|s| s.empty_edges),
        Some(Some(EmptyEdges::Pruned))
    );
}

/// Two rows read each other's rings exactly when they differ in nothing but the edges the builder emits.
#[test]
fn rings_cross_between_rows_only_across_the_edge_set() {
    let (v1, r8, pruned) = (
        lookup("gnn_axis_v1").expect("registered"),
        lookup("gnn_axis_r8").expect("registered"),
        lookup("gnn_axis_r8_pruned").expect("registered"),
    );
    assert!(r8.reads_rings_of(r8) && pruned.reads_rings_of(r8) && r8.reads_rings_of(pruned));
    assert!(
        !v1.reads_rings_of(r8) && !r8.reads_rings_of(v1) && !pruned.reads_rings_of(v1),
        "a radius-6 row's legal set is not a radius-8 row's"
    );
}

/// The canvas row's own per-field pin: the graph rows' reply frame, the fence's radius, six planes, the canvas keys.
#[test]
fn per_field_pins_hexcanvas_r8() {
    let s = lookup("hexcanvas_r8").expect("hexcanvas_r8 is a registered row");
    assert_eq!(s.representation, Representation::HexCanvas);
    assert!(!s.is_graph());
    assert_eq!(
        (s.board_size, s.trunk_size, s.policy_logit_count),
        (19, 19, 362)
    );
    assert!(s.has_pass_slot && !s.is_multi_window);
    assert_eq!((s.cluster_window_size, s.cluster_threshold), (None, None));
    assert_eq!(
        (s.value_pool, s.policy_pool),
        (ValuePool::None, PolicyPool::None)
    );
    assert_eq!(
        (s.sym_table_id, s.k_max, s.n_chain_planes),
        ("size_19", 1, 6)
    );
    assert_eq!((s.legal_move_radius, s.win_length), (8, Some(6)));
    assert_eq!(s.n_planes, 6);
    assert_eq!(s.plane_layout, mantis_graph::CANVAS_PLANES);
    assert_eq!(
        (s.kept_plane_indices, s.n_source_planes),
        (&[0, 1, 2, 3, 4, 5][..], 6)
    );
    assert_eq!(
        (s.canvas_side_step, s.canvas_min_side, s.far_halo),
        (Some(8), Some(24), Some(4))
    );
    assert_eq!(
        (s.canvas_max_side, s.far_max_side),
        (Some(112), Some(144)),
        "S_max is DENSE-1 L0's (the smallest multiple of 8 no corpus fence box exceeds) and S_far = S_max + 32"
    );
    assert_eq!(
        (s.node_feat_dim, s.edge_feat_dim, s.graph_radius, s.win_axes),
        (None, None, None, None)
    );
    assert_eq!(
        (s.contract_version, s.builder_impl_required, s.empty_edges),
        (None, None, None)
    );
    assert_eq!(s.schema_version, 6);
}

/// A canvas row reads a graph writer's ring exactly when the radius and win length are its own; no graph row reads a canvas ring.
#[test]
fn a_canvas_row_reads_the_rings_of_a_graph_row_of_its_radius_and_no_other() {
    let (v1, r8, pruned, canvas) = (
        lookup("gnn_axis_v1").expect("registered"),
        lookup("gnn_axis_r8").expect("registered"),
        lookup("gnn_axis_r8_pruned").expect("registered"),
        lookup("hexcanvas_r8").expect("registered"),
    );
    assert!(canvas.reads_rings_of(canvas));
    assert!(
        canvas.reads_rings_of(r8) && canvas.reads_rings_of(pruned),
        "a canvas row reads the radius-8 graph rows' rings"
    );
    assert!(
        !canvas.reads_rings_of(v1),
        "a radius-6 row's legal set is not the canvas's"
    );
    assert!(
        !r8.reads_rings_of(canvas) && !pruned.reads_rings_of(canvas) && !v1.reads_rings_of(canvas),
        "the rule admits a canvas reader of a graph writer, never the reverse"
    );
}

fn canvas_body() -> String {
    r#"
representation          = "hexcanvas"
board_size              = 19
trunk_size              = 19
cluster_window_size     = "none"
cluster_threshold       = "none"
legal_move_radius       = 8
n_planes                = 6
plane_layout            = ["own", "opponent", "legal", "fence", "stones_left_1", "stones_left_2"]
policy_logit_count      = 362
has_pass_slot           = true
is_multi_window         = false
value_pool              = "none"
policy_pool             = "none"
sym_table_id            = "size_19"
kept_plane_indices      = [0, 1, 2, 3, 4, 5]
n_source_planes         = 6
k_max                   = 1
win_length              = 6
n_chain_planes          = 6
canvas_side_step        = 8
canvas_min_side         = 24
canvas_max_side         = 112
far_halo                = 4
far_max_side            = 144
schema_version          = 6
notes                   = "test"
"#
    .to_string()
}

/// The canvas row's keys are required with their relations, and every graph-only key is refused on it.
#[test]
fn a_canvas_row_states_its_keys_and_refuses_the_graph_ones() {
    assert!(
        parse_encoding_toml("canvas_ok", &canvas_body()).is_ok(),
        "the control loads"
    );
    let refused = |label: &str, body: String, needle: &str| {
        let err = parse_encoding_toml(label, &body).expect_err(label);
        assert!(
            err.contains(needle),
            "{label}: the refusal must name {needle:?}: {err}"
        );
    };
    for (key, line) in [
        ("empty_edges", "empty_edges = \"kept\"\n"),
        ("graph_radius", "graph_radius = 8\n"),
        ("node_feat_dim", "node_feat_dim = 11\n"),
        ("contract_version", "contract_version = 1\n"),
    ] {
        refused(key, canvas_body() + line, key);
    }
    refused(
        "no S_max",
        canvas_body().replace("canvas_max_side         = 112\n", ""),
        "canvas_max_side",
    );
    refused(
        "S_far at S_max",
        canvas_body().replace("= 144", "= 112"),
        "far_max_side",
    );
    refused(
        "off the quantum",
        canvas_body().replace("= 112", "= 110"),
        "multiple",
    );
    refused(
        "a halo at the radius",
        canvas_body().replace("far_halo                = 4", "far_halo = 8"),
        "far_halo",
    );
    refused(
        "a short minimum",
        canvas_body().replace("canvas_min_side         = 24", "canvas_min_side = 16"),
        "lone stone",
    );
    refused(
        "another layout",
        canvas_body().replace("\"fence\", ", ""),
        "plane_layout",
    );
    refused(
        "another frame",
        canvas_body().replace("\"size_19\"", "\"size_25\""),
        "size_19",
    );
}

/// A graph row refuses a canvas key, so neither representation carries the other's geometry.
#[test]
fn a_graph_row_refuses_a_canvas_key() {
    let graph = toml_of("gnn_axis_r8");
    assert!(
        parse_encoding_toml("graph_ok", &graph).is_ok(),
        "the control loads"
    );
    let err = parse_encoding_toml("graph_canvas", &(graph + "canvas_max_side = 112\n"))
        .expect_err("a canvas key on a graph row must be refused");
    assert!(
        err.contains("canvas_max_side"),
        "the refusal must name the key: {err}"
    );
}

/// A registered row's own TOML body, read back from the embedded registry.
fn toml_of(name: &str) -> String {
    let table: toml::Table = include_str!("../src/registry.toml")
        .parse()
        .expect("the registry parses");
    let body = &table["encodings"][name];
    toml::to_string(body).expect("a row serialises")
}

/// The field-diff's mutation self-test: the extractor must ATTRIBUTE a difference to the field that
/// owns it, including one buried inside a multi-line `Option`. Driven on two hand-built dumps so
/// the control does not need a mutated registry.
#[test]
fn the_field_diff_attributes_a_nested_difference_to_its_owning_field() {
    let a = "Spec {\n    board_size: 19,\n    win_length: Some(\n        6,\n    ),\n}";
    let b = "Spec {\n    board_size: 19,\n    win_length: Some(\n        7,\n    ),\n}";
    assert_eq!(
        differing_fields(a, b),
        vec!["win_length"],
        "a difference in the INNER line of a multi-line Option must be reported against \
         `win_length`, not against the bare value line — otherwise a real drift is reported \
         under a name no one can find and the assertion above is unreadable"
    );
}
