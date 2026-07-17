// Exceeds the 300-line soft cap (R8): the single collect-all-errors, multi-pass
// TOML field parser for one encoding entry — the per-field error-collection loop
// is >100 LOC by design, kept whole so one pass reports every field fault at once.
//! TOML field-parser for one `[encodings.<name>]` entry.
//!
//! `parse_one` and its inline macros (`get_int!`, `get_str!`, `get_bool!`) live
//! here; the helpers `leak_str` and `parse_int_or_none` stay in `super`
//! (`registry/mod.rs`).
//!
//! Two sanctioned behavior changes vs the ported original (each oracle-gated):
//!   - UNKNOWN-KEY rejection: any key not in the recognized set is a parse error
//!     (collected, not short-circuited). Kills the old "silently ignore extras".
//!   - `representation` REQUIRED: an absent `representation` is a named parse
//!     error, never a grid/dense default (LAW-11).

use toml::Value;

use super::{leak_str, parse_int_or_none};
use crate::spec::{PolicyPool, RegistrySpec, Representation, ValuePool};

/// The complete recognized key set (grid + graph). A key outside this set is an
/// unknown-key parse error. Graph-only keys present on a grid entry are a
/// distinct `validate()` error ("grid must not set graph-only key"), reported in
/// the same collect-all pass.
const KNOWN_KEYS: &[&str] = &[
    "representation",
    "board_size",
    "trunk_size",
    "cluster_window_size",
    "cluster_threshold",
    "legal_move_radius",
    "n_planes",
    "plane_layout",
    "policy_logit_count",
    "has_pass_slot",
    "is_multi_window",
    "value_pool",
    "policy_pool",
    "sym_table_id",
    "schema_version",
    "notes",
    "kept_plane_indices",
    "n_source_planes",
    "k_max",
    "n_chain_planes",
    "node_feat_dim",
    "edge_feat_dim",
    "win_length",
    "graph_radius",
    "win_axes",
    "contract_version",
    "builder_impl_required",
];

/// Optional non-negative integer key. `None` if absent, `Some(i)` if a valid
/// non-negative integer, and pushes a typed error (returning `None`) if
/// present-but-invalid. Distinct from the required `get_int!` macro.
fn opt_int(table: &toml::Table, key: &str, errs: &mut Vec<String>) -> Option<i64> {
    match table.get(key) {
        None => None,
        Some(v) => match v.as_integer() {
            Some(i) if i >= 0 => Some(i),
            Some(i) => {
                errs.push(format!("{key:?}: negative integer {i}"));
                None
            }
            None => {
                errs.push(format!("{key:?}: present but not an integer"));
                None
            }
        },
    }
}

#[allow(clippy::too_many_lines)]
pub(super) fn parse_one(name: &str, body: &Value) -> Result<RegistrySpec, String> {
    let table = body
        .as_table()
        .ok_or_else(|| format!("[encodings.{name}]: not a table"))?;

    let mut errs: Vec<String> = Vec::new();

    macro_rules! get_int {
        ($key:expr) => {
            match table.get($key).and_then(Value::as_integer) {
                Some(v) => Some(v),
                None => {
                    errs.push(format!(
                        "[encodings.{}]: missing or non-integer key {:?}",
                        name, $key
                    ));
                    None
                }
            }
        };
    }
    macro_rules! get_str {
        ($key:expr) => {
            match table.get($key).and_then(Value::as_str) {
                Some(v) => Some(v),
                None => {
                    errs.push(format!(
                        "[encodings.{}]: missing or non-string key {:?}",
                        name, $key
                    ));
                    None
                }
            }
        };
    }
    macro_rules! get_bool {
        ($key:expr) => {
            match table.get($key).and_then(Value::as_bool) {
                Some(v) => Some(v),
                None => {
                    errs.push(format!(
                        "[encodings.{}]: missing or non-bool key {:?}",
                        name, $key
                    ));
                    None
                }
            }
        };
    }

    // UNKNOWN-KEY rejection (sanctioned change): iterate every present key and
    // flag any outside the recognized set. Runs in the SAME collect-all pass so
    // a smuggled unknown key AND a co-present missing key both report.
    for key in table.keys() {
        if !KNOWN_KEYS.contains(&key.as_str()) {
            errs.push(format!("[encodings.{name}]: unknown key {key:?}"));
        }
    }

    let board_size = get_int!("board_size").map(|v| v as usize);
    let trunk_size = get_int!("trunk_size").map(|v| v as usize);
    let legal_move_radius = get_int!("legal_move_radius").map(|v| v as usize);
    let n_planes = get_int!("n_planes").map(|v| v as usize);
    let policy_logit_count = get_int!("policy_logit_count").map(|v| v as usize);
    let schema_version = get_int!("schema_version").map(|v| v as u32);
    let n_chain_planes = get_int!("n_chain_planes").map(|v| v as usize);
    let has_pass_slot = get_bool!("has_pass_slot");
    let is_multi_window = get_bool!("is_multi_window");

    // cluster_window_size + cluster_threshold: int OR string "none".
    let cluster_window_size = parse_int_or_none(table.get("cluster_window_size"))
        .map_err(|e| format!("[encodings.{name}].cluster_window_size: {e}"))
        .unwrap_or_else(|e| {
            errs.push(e);
            None
        });
    let cluster_threshold = parse_int_or_none(table.get("cluster_threshold"))
        .map_err(|e| format!("[encodings.{name}].cluster_threshold: {e}"))
        .unwrap_or_else(|e| {
            errs.push(e);
            None
        });

    let value_pool_raw = get_str!("value_pool");
    let policy_pool_raw = get_str!("policy_pool");
    let sym_table_id = get_str!("sym_table_id");
    let notes = get_str!("notes");

    // kept_plane_indices: array of integers.
    let kept_plane_indices: Option<Vec<usize>> = match table.get("kept_plane_indices") {
        Some(Value::Array(arr)) => {
            let mut indices: Vec<usize> = Vec::with_capacity(arr.len());
            let mut bad = false;
            for (i, v) in arr.iter().enumerate() {
                match v.as_integer() {
                    Some(n) if n >= 0 => indices.push(n as usize),
                    Some(n) => {
                        errs.push(format!(
                            "[encodings.{name}].kept_plane_indices[{i}]: negative integer {n}"
                        ));
                        bad = true;
                        break;
                    }
                    None => {
                        errs.push(format!(
                            "[encodings.{name}].kept_plane_indices[{i}]: not an integer"
                        ));
                        bad = true;
                        break;
                    }
                }
            }
            if bad {
                None
            } else {
                Some(indices)
            }
        }
        Some(_) => {
            errs.push(format!("[encodings.{name}].kept_plane_indices: not an array"));
            None
        }
        None => {
            errs.push(format!("[encodings.{name}].kept_plane_indices: missing key"));
            None
        }
    };

    let n_source_planes = get_int!("n_source_planes").map(|v| v as usize);
    let k_max = get_int!("k_max").map(|v| v as u32);

    // `representation` is REQUIRED (sanctioned change, LAW-11): an absent key is
    // a named parse error, NOT a Grid/dense default. The graph-only fields stay
    // `None` for grid and are required-when-graph by `validate()`.
    let representation = match table.get("representation") {
        None => {
            errs.push(format!(
                "[encodings.{name}]: missing required key \"representation\""
            ));
            None
        }
        Some(Value::String(s)) => match Representation::parse(s) {
            Ok(r) => Some(r),
            Err(e) => {
                errs.push(format!("[encodings.{name}].representation: {e}"));
                None
            }
        },
        Some(other) => {
            errs.push(format!(
                "[encodings.{name}].representation: must be a string; got {:?}",
                other.type_str()
            ));
            None
        }
    };
    let node_feat_dim = opt_int(table, "node_feat_dim", &mut errs).map(|v| v as usize);
    let edge_feat_dim = opt_int(table, "edge_feat_dim", &mut errs).map(|v| v as usize);
    let win_length = opt_int(table, "win_length", &mut errs).map(|v| v as usize);
    let graph_radius = opt_int(table, "graph_radius", &mut errs).map(|v| v as usize);
    let win_axes = opt_int(table, "win_axes", &mut errs).map(|v| v as usize);
    let contract_version = opt_int(table, "contract_version", &mut errs).map(|v| v as u32);
    let builder_impl_required =
        opt_int(table, "builder_impl_required", &mut errs).map(|v| v as u8);

    // plane_layout: array of strings.
    let plane_layout: Option<Vec<&'static str>> = match table.get("plane_layout") {
        Some(Value::Array(arr)) => {
            let mut planes: Vec<&'static str> = Vec::with_capacity(arr.len());
            let mut bad = false;
            for (i, v) in arr.iter().enumerate() {
                if let Some(s) = v.as_str() {
                    planes.push(leak_str(s));
                } else {
                    errs.push(format!("[encodings.{name}].plane_layout[{i}]: not a string"));
                    bad = true;
                    break;
                }
            }
            if bad {
                None
            } else {
                Some(planes)
            }
        }
        Some(_) => {
            errs.push(format!("[encodings.{name}].plane_layout: not an array"));
            None
        }
        None => {
            errs.push(format!("[encodings.{name}].plane_layout: missing key"));
            None
        }
    };

    let value_pool = value_pool_raw.and_then(|s| match ValuePool::parse(s) {
        Ok(v) => Some(v),
        Err(e) => {
            errs.push(format!("[encodings.{name}].value_pool: {e}"));
            None
        }
    });
    let policy_pool = policy_pool_raw.and_then(|s| match PolicyPool::parse(s) {
        Ok(v) => Some(v),
        Err(e) => {
            errs.push(format!("[encodings.{name}].policy_pool: {e}"));
            None
        }
    });

    if !errs.is_empty() {
        return Err(format!(
            "[encodings.{}]: {} field error(s):\n    - {}",
            name,
            errs.len(),
            errs.join("\n    - ")
        ));
    }

    // All Some at this point.
    // SAFETY: allocated by Box::leak in registry::load();
    // stable for process lifetime — registry is one-shot init.
    let plane_layout: &'static [&'static str] = Box::leak(plane_layout.unwrap().into_boxed_slice());
    // SAFETY: allocated by Box::leak in registry::load();
    // stable for process lifetime — registry is one-shot init.
    let kept_plane_indices: &'static [usize] =
        Box::leak(kept_plane_indices.unwrap().into_boxed_slice());

    Ok(RegistrySpec {
        name: leak_str(name),
        board_size: board_size.unwrap(),
        trunk_size: trunk_size.unwrap(),
        cluster_window_size,
        cluster_threshold,
        legal_move_radius: legal_move_radius.unwrap(),
        n_planes: n_planes.unwrap(),
        plane_layout,
        policy_logit_count: policy_logit_count.unwrap(),
        has_pass_slot: has_pass_slot.unwrap(),
        is_multi_window: is_multi_window.unwrap(),
        value_pool: value_pool.unwrap(),
        policy_pool: policy_pool.unwrap(),
        sym_table_id: leak_str(sym_table_id.unwrap()),
        schema_version: schema_version.unwrap(),
        notes: leak_str(notes.unwrap()),
        kept_plane_indices,
        n_source_planes: n_source_planes.unwrap(),
        k_max: k_max.unwrap(),
        n_chain_planes: n_chain_planes.unwrap(),
        representation: representation.unwrap(),
        node_feat_dim,
        edge_feat_dim,
        win_length,
        graph_radius,
        win_axes,
        contract_version,
        builder_impl_required,
    })
}
