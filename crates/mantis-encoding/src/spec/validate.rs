//! `RegistrySpec::validate` — cross-field invariant checks. Collects ALL
//! violations into one multi-line message; never short-circuits on the first.

use std::collections::BTreeSet;

use super::{PolicyPool, RegistrySpec, Representation, ValuePool};

// Graph invariants are single-sourced against the axis-graph builder's own schema constants,
// so the graph dims, `win_axes` and `builder_impl_required` cannot drift from its output.
use mantis_core::board::WIN_LENGTH;
use mantis_graph::{BUILDER_IMPL_NATIVE, EDGE_FEAT_DIM, NODE_FEAT_DIM, WIN_AXES};

/// Graph-only field must be present and equal the builder-schema constant.
/// Pushes a named error otherwise (missing OR mismatched).
fn require_graph_eq(errs: &mut Vec<String>, key: &str, got: Option<usize>, want: usize) {
    match got {
        Some(v) if v == want => {}
        Some(v) => errs.push(format!(
            "{key}={v} must equal {want} (axis-graph builder schema)"
        )),
        None => errs.push(format!("representation=graph requires {key}")),
    }
}

impl RegistrySpec {
    /// Validate cross-field invariants. Collects ALL violations into a single
    /// multi-line message; never short-circuits on the first.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> Result<(), String> {
        let mut errs: Vec<String> = Vec::new();

        // Action space = board_size² + (pass?1:0), for BOTH representations: a graph plays
        // the identical 19×19+pass board, so policy_logit_count STAYS 362.
        let expected_logits = self.board_size * self.board_size + usize::from(self.has_pass_slot);

        let cw_some = self.cluster_window_size.is_some();
        let ct_some = self.cluster_threshold.is_some();
        if cw_some != self.is_multi_window {
            errs.push(format!(
                "is_multi_window={} != cluster_window_size.is_some()={}",
                self.is_multi_window, cw_some
            ));
        }
        if ct_some != self.is_multi_window {
            errs.push(format!(
                "is_multi_window={} != cluster_threshold.is_some()={}",
                self.is_multi_window, ct_some
            ));
        }

        if self.is_multi_window {
            if !matches!(
                self.value_pool,
                ValuePool::Min | ValuePool::Max | ValuePool::Mean
            ) {
                errs.push(format!(
                    "is_multi_window=true requires value_pool ∈ {{Min,Max,Mean}}; got {:?}",
                    self.value_pool
                ));
            }
            if !matches!(
                self.policy_pool,
                PolicyPool::ScatterMax | PolicyPool::ScatterMean | PolicyPool::LegalSetScatterMax
            ) {
                errs.push(format!(
                    "is_multi_window=true requires policy_pool ∈ {{ScatterMax,ScatterMean,LegalSetScatterMax}}; got {:?}",
                    self.policy_pool
                ));
            }
            if let Some(cw) = self.cluster_window_size {
                if self.trunk_size != cw {
                    errs.push(format!(
                        "is_multi_window=true requires trunk_size==cluster_window_size; \
                         got trunk_size={}, cluster_window_size={}",
                        self.trunk_size, cw
                    ));
                }
            }
        } else {
            if !matches!(self.value_pool, ValuePool::None) {
                errs.push(format!(
                    "is_multi_window=false requires value_pool=None; got {:?}",
                    self.value_pool
                ));
            }
            if !matches!(self.policy_pool, PolicyPool::None) {
                errs.push(format!(
                    "is_multi_window=false requires policy_pool=None; got {:?}",
                    self.policy_pool
                ));
            }
            if self.cluster_window_size.is_some() {
                errs.push(format!(
                    "is_multi_window=false requires cluster_window_size=None; got {:?}",
                    self.cluster_window_size
                ));
            }
            if self.cluster_threshold.is_some() {
                errs.push(format!(
                    "is_multi_window=false requires cluster_threshold=None; got {:?}",
                    self.cluster_threshold
                ));
            }
            if self.trunk_size != self.board_size {
                errs.push(format!(
                    "is_multi_window=false requires trunk_size==board_size; \
                     got trunk_size={}, board_size={}",
                    self.trunk_size, self.board_size
                ));
            }
        }

        if self.legal_move_radius == 0 {
            errs.push("legal_move_radius must be > 0".to_string());
        }

        // Single-window encodings emit exactly 1 cluster view per leaf;
        // multi-window encodings cap at the registry-declared upper bound.
        if self.k_max == 0 {
            errs.push("k_max must be >= 1".to_string());
        }

        // n_chain_planes is the sole authority for chain_stride; must be >= 1
        // (universal — every encoding carries the field).
        if self.n_chain_planes == 0 {
            errs.push("n_chain_planes must be >= 1".to_string());
        }

        // `mantis_selfplay`'s `sym_tables_for` PANICS at runner start (graph runs included) on
        // any other `(sym_table_id, n_chain_planes)`, so the registry refuses it here, at load.
        // Duplicated rather than imported: this crate sits BELOW mantis-selfplay in the DAG, and
        // `registry_census.rs` holds the two equal.
        const SYM_TABLE_IDS: [&str; 2] = ["size_19", "size_25"];
        const SYM_CHAIN_PLANES: usize = 6;
        if !SYM_TABLE_IDS.contains(&self.sym_table_id) {
            errs.push(format!(
                "sym_table_id {:?} is not one of {SYM_TABLE_IDS:?}: `sym_tables_for` has no \
                 table for it and would panic at the first runner start",
                self.sym_table_id
            ));
        }
        if self.sym_table_id == "size_25" && self.n_planes != 8 {
            errs.push(format!(
                "sym_table_id \"size_25\" requires n_planes == 8, got {}: `sym_tables_for` \
                 matches on the PAIR and would panic at the first runner start",
                self.n_planes
            ));
        }
        if self.n_chain_planes != SYM_CHAIN_PLANES {
            errs.push(format!(
                "n_chain_planes must be {SYM_CHAIN_PLANES} (the D6 chain tables' fixed inner \
                 dimension), got {}: `sym_tables_for` asserts this and would panic at the \
                 first runner start",
                self.n_chain_planes
            ));
        }

        // Plane layout sanity: no duplicates, no empty strings.
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for (idx, p) in self.plane_layout.iter().enumerate() {
            if p.is_empty() {
                errs.push(format!("plane_layout[{idx}] is empty"));
            }
            if !seen.insert(p) {
                errs.push(format!("plane_layout[{idx}] duplicate name {p:?}"));
            }
        }

        // Representation-gated: the axis-graph geometry invariants apply to Graph only; the
        // multi-window / legal_move_radius / k_max / n_chain_planes checks above are universal.
        match self.representation {
            Representation::Graph => {
                // action space UNCHANGED (identical 19×19+pass board).
                if self.policy_logit_count != expected_logits {
                    errs.push(format!(
                        "policy_logit_count={} != board_size²+(pass_slot?1:0)={} \
                         (graph plays the identical board; action space unchanged)",
                        self.policy_logit_count, expected_logits
                    ));
                }
                if !self.has_pass_slot {
                    errs.push("representation=graph requires has_pass_slot=true".to_string());
                }
                // node/edge/axis dims single-sourced against the builder schema.
                require_graph_eq(
                    &mut errs,
                    "node_feat_dim",
                    self.node_feat_dim,
                    NODE_FEAT_DIM,
                );
                require_graph_eq(
                    &mut errs,
                    "edge_feat_dim",
                    self.edge_feat_dim,
                    EDGE_FEAT_DIM,
                );
                require_graph_eq(&mut errs, "win_axes", self.win_axes, WIN_AXES.len());
                // `win_length` is the GAME'S rule (`mantis_core::board::WIN_LENGTH`), not a free
                // registry number: "present + positive" would accept a 7 no engine path honours.
                require_graph_eq(&mut errs, "win_length", self.win_length, WIN_LENGTH);
                match self.graph_radius {
                    Some(r) if r >= 1 => {}
                    Some(r) => errs.push(format!("graph_radius={r} must be >= 1")),
                    None => errs.push("representation=graph requires graph_radius".to_string()),
                }
                // THE RELATION, not two positivity checks: the MCTS legal set is built at
                // `legal_move_radius` and the graph's legal nodes at `graph_radius`; if they
                // disagree the ragged policy covers a different cell set than the tree expands.
                if let Some(r) = self.graph_radius {
                    if r != self.legal_move_radius {
                        errs.push(format!(
                            "representation=graph requires graph_radius == legal_move_radius; \
                             got graph_radius={} vs legal_move_radius={} (the MCTS legal set \
                             and the graph's legal nodes would cover different cells)",
                            r, self.legal_move_radius
                        ));
                    }
                }
                // contract + builder handshake the resolver asserts.
                if self.contract_version != Some(1) {
                    errs.push(format!(
                        "representation=graph requires contract_version=1; got {:?}",
                        self.contract_version
                    ));
                }
                if self.builder_impl_required != Some(BUILDER_IMPL_NATIVE) {
                    errs.push(format!(
                        "representation=graph requires builder_impl_required={} (native); got {:?}",
                        BUILDER_IMPL_NATIVE, self.builder_impl_required
                    ));
                }
                // no dense planes (whole-board graph).
                if self.n_planes != 0 {
                    errs.push(format!(
                        "representation=graph requires n_planes=0; got {}",
                        self.n_planes
                    ));
                }
                if !self.plane_layout.is_empty() {
                    errs.push(format!(
                        "representation=graph requires empty plane_layout; got {} entries",
                        self.plane_layout.len()
                    ));
                }
                if !self.kept_plane_indices.is_empty() {
                    errs.push(format!(
                        "representation=graph requires empty kept_plane_indices; got {} entries",
                        self.kept_plane_indices.len()
                    ));
                }
                if self.n_source_planes != 0 {
                    errs.push(format!(
                        "representation=graph requires n_source_planes=0; got {}",
                        self.n_source_planes
                    ));
                }
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "RegistrySpec {:?} validation failed:\n  - {}",
                self.name,
                errs.join("\n  - ")
            ))
        }
    }
}
