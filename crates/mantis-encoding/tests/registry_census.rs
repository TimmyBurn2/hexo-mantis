//! O-4..O-10, O-13 — registry census (pruned set), per-field pins on the two graph rows, and
//! the derived field-diff that keeps `gnn_axis_r8` a one-knob edit of `gnn_axis_v1`.
//!
//! The three grid rows (`v6`, `v6w25`, `v6_live2_ls`) went with the dense path (R346(f)); they
//! are pinned here as ABSENT so a resurrected row fails the census rather than the encoder.

use mantis_encoding::{all_specs, lookup, PolicyPool, Representation, ValuePool};

const REGISTERED: [&str; 2] = ["gnn_axis_v1", "gnn_axis_r8"];
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
    assert_eq!(names, ["gnn_axis_r8", "gnn_axis_v1"]);
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

/// `representation` is the identity key (LAW-11) and `"grid"` is refused BY NAME rather than
/// falling through as unknown, so a stale row says what happened to it.
#[test]
fn representation_parse_refuses_grid_by_name() {
    assert_eq!(Representation::parse("graph"), Ok(Representation::Graph));
    let err = Representation::parse("grid").expect_err("grid must be refused, never defaulted");
    assert!(
        err.contains("grid") && err.contains("R346(f)"),
        "the refusal must name the deleted representation and the ruling; got {err:?}"
    );
    assert!(
        Representation::parse("dense").is_err(),
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
    assert_eq!(
        s.schema_version, 4,
        "graph entry schema_version = 4 (preserved)"
    );
    assert_eq!(s.legal_move_radius, 6);
}

/// AUDIT-1 F-41. `gnn_axis_r8` — run6's own identity row — appeared ONLY in this file's name
/// lists; every per-field pin covered the other four rows. A row with no per-field pin is a row
/// whose geometry can be edited without a single assertion moving, and this is the row the next
/// mint runs on.
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
    assert_eq!(
        s.schema_version, 4,
        "graph entry schema_version = 4 (preserved)"
    );
    assert_eq!(
        s.legal_move_radius, 8,
        "the R328 identity: legal_move_radius and graph_radius are ONE number on a graph row, \
         and REPAIR-2's F-18 validator refuses them typed apart"
    );
}

/// The Rust port of `tests/encoding/test_r8_identity.py::test_r328b_03` (AUDIT-1 F-41's repair
/// line, "port the field-diff to Rust"). The two graph rows must differ in EXACTLY the radius
/// pair and their identifying strings — `gnn_axis_r8` exists to move ONE knob, so a stray
/// `win_length` or `policy_logit_count` difference makes the r6/r8 comparison two comparisons.
///
/// DERIVED, not a typed field list. Rust has no reflection, so the walk is over the `{:#?}`
/// dump `#[derive(Debug)]` generates from the struct definition: a field ADDED to
/// `RegistrySpec` joins both dumps automatically, which is the property the Python original
/// has and a hand-enumerated comparison does not — a typed list is edited in the same commit
/// as the drift it would have caught.
#[test]
fn the_two_graph_rows_differ_in_exactly_the_radius_pair() {
    let v1 = format!(
        "{:#?}",
        lookup("gnn_axis_v1").expect("gnn_axis_v1 is registered")
    );
    let r8 = format!(
        "{:#?}",
        lookup("gnn_axis_r8").expect("gnn_axis_r8 is registered")
    );
    let (a, b): (Vec<&str>, Vec<&str>) = (v1.lines().collect(), r8.lines().collect());
    assert_eq!(
        a.len(),
        b.len(),
        "the two dumps have different shapes, so no field-wise diff \
         is possible:\n{v1}\n---\n{r8}"
    );
    assert!(
        a.len() > 15,
        "the Debug dump collapsed to {} lines; this test would be vacuous",
        a.len()
    );

    // `{:#?}` breaks an `Option<usize>` over three lines, so a differing line is often the
    // INNER value rather than the field name. Carry the most recent `field:` line seen so a
    // difference is attributed to the field that owns it, and de-duplicate.
    let mut field = String::new();
    let mut differing: Vec<String> = Vec::new();
    for (x, y) in a.iter().zip(b.iter()) {
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
    assert_eq!(
        differing,
        vec!["graph_radius", "legal_move_radius", "name", "notes"],
        "the two graph rows differ in {differing:?}. This encoding exists to move ONE knob so \
         the r6/r8 comparison IS a comparison; any other difference makes it two."
    );
}

/// LAW-07 for the field-diff above: the extractor must ATTRIBUTE a difference to the field that
/// owns it, including one buried inside a multi-line `Option`. Driven on two hand-built dumps so
/// the control does not need a mutated registry.
#[test]
fn the_field_diff_attributes_a_nested_difference_to_its_owning_field() {
    let a = "Spec {\n    board_size: 19,\n    win_length: Some(\n        6,\n    ),\n}";
    let b = "Spec {\n    board_size: 19,\n    win_length: Some(\n        7,\n    ),\n}";
    let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
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
    assert_eq!(
        differing,
        vec!["win_length"],
        "a difference in the INNER line of a multi-line Option must be reported against \
         `win_length`, not against the bare value line — otherwise a real drift is reported \
         under a name no one can find and the assertion above is unreadable"
    );
}
