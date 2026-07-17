// Exceeds the 300-line soft cap (R8): the full-schema RegistrySpec record + its
// three closed enums + every derived accessor kept together — splitting the type
// from the accessors it owns would fragment one indivisible schema definition.
//! Encoding registry spec — full-schema record per `registry.toml`.
//!
//! The sole encoding record type. Per-Board construction never binds a spec into
//! mantis-core; the dense kernels thread `&RegistrySpec` as a parameter.

mod validate;

/// Value-head pooling mode (multi-window only). `None` for single-window.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValuePool {
    None,
    Min,
    Max,
    Mean,
}

/// Policy-head pooling mode (multi-window only). `None` for single-window.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PolicyPool {
    None,
    ScatterMax,
    ScatterMean,
    /// Scatter-max over K cluster windows WITHOUT the off-window drop — the
    /// aggregated MCTS prior / improved-policy target is a ragged legal-set
    /// (board-coord-keyed), retaining off-global-window cells covered by some
    /// cluster.
    LegalSetScatterMax,
}

/// Input representation discriminant. `Grid` = the dense CNN plane encodings;
/// `Graph` = the axis-graph / GNN encodings. The TOML key `representation` is
/// REQUIRED (absent → parse error, LAW-11 — never a grid/dense default). The
/// grid-only cross-field invariants (`policy_logit_count==bs²+pass`,
/// `len(plane_layout)==n_planes`, kept-plane relationships, trunk==board) are
/// gated on `Grid`; graph encodings carry the `node_feat_dim`/`win_length`/…
/// fields and their own invariants instead.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Representation {
    Grid,
    Graph,
}

impl Representation {
    /// Parse the TOML value string. The identity key is spelled `"grid"` (the
    /// dense CNN encodings) or `"graph"` (the axis-graph encodings).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "grid" => Ok(Representation::Grid),
            "graph" => Ok(Representation::Graph),
            other => Err(format!(
                "representation must be one of [grid,graph]; got {other:?}"
            )),
        }
    }

    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Representation::Grid => "grid",
            Representation::Graph => "graph",
        }
    }

    #[inline]
    #[must_use]
    pub fn is_graph(&self) -> bool {
        matches!(self, Representation::Graph)
    }
}

impl ValuePool {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(ValuePool::None),
            "min" => Ok(ValuePool::Min),
            "max" => Ok(ValuePool::Max),
            "mean" => Ok(ValuePool::Mean),
            other => Err(format!(
                "value_pool must be one of [none,min,max,mean]; got {other:?}"
            )),
        }
    }

    #[must_use]
    pub fn is_some(&self) -> bool {
        !matches!(self, ValuePool::None)
    }
}

impl PolicyPool {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(PolicyPool::None),
            "scatter_max" => Ok(PolicyPool::ScatterMax),
            "scatter_mean" => Ok(PolicyPool::ScatterMean),
            "legal_set_scatter_max" => Ok(PolicyPool::LegalSetScatterMax),
            other => Err(format!(
                "policy_pool must be one of [none,scatter_max,scatter_mean,legal_set_scatter_max]; got {other:?}"
            )),
        }
    }

    #[must_use]
    pub fn is_some(&self) -> bool {
        !matches!(self, PolicyPool::None)
    }
}

/// Full encoding record parsed from `registry.toml`.
///
/// All `&'static str` / `&'static [..]` fields point at heap data leaked at
/// registry init time (`Box::leak`), so addresses are stable for the process
/// lifetime. Cheap to copy — pass by value or `&'static`.
#[derive(Copy, Clone, Debug)]
pub struct RegistrySpec {
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub name: &'static str,
    pub board_size: usize,
    pub trunk_size: usize,
    pub cluster_window_size: Option<usize>,
    pub cluster_threshold: Option<usize>,
    pub legal_move_radius: usize,
    pub n_planes: usize,
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub plane_layout: &'static [&'static str],
    pub policy_logit_count: usize,
    pub has_pass_slot: bool,
    pub is_multi_window: bool,
    pub value_pool: ValuePool,
    pub policy_pool: PolicyPool,
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub sym_table_id: &'static str,
    pub schema_version: u32,
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub notes: &'static str,

    /// Physical source-plane indices retained by this encoding's wire format.
    /// Length == `n_planes`. See `registry.toml` header for the canonical
    /// X+history / O+history block convention.
    ///
    /// &'static after Box::leak in registry::load(); stable for process lifetime.
    pub kept_plane_indices: &'static [usize],
    /// Source tensor plane count *before* the `kept_plane_indices` slice.
    /// Used by the validator for the kept-indices upper bound.
    pub n_source_planes: usize,

    /// Multi-window cluster-count upper bound per position. Single-window
    /// encodings emit exactly 1 view per leaf (`k_max = 1`).
    pub k_max: u32,

    /// Number of chain-length planes (= 6 across all current encodings: 3 hex
    /// axes × 2 players). REQUIRED TOML field; the SOLE authority for
    /// `chain_stride()` (no source constant, no replay reach-through).
    pub n_chain_planes: usize,

    /// `Grid` (dense CNN planes) vs `Graph` (axis-graph GNN). TOML key
    /// `representation` (REQUIRED — absent = error).
    pub representation: Representation,
    /// Per-node feature width (graph only). = 11 for gnn_axis_v1
    /// (= `mantis_graph::NODE_FEAT_DIM`).
    pub node_feat_dim: Option<usize>,
    /// Per-edge feature width (graph only). = 5 (= `mantis_graph::EDGE_FEAT_DIM`).
    pub edge_feat_dim: Option<usize>,
    /// GNN win-length (graph only). = 6.
    pub win_length: Option<usize>,
    /// GNN legal-move / axis-walk radius (graph only). = 6.
    pub graph_radius: Option<usize>,
    /// Number of win axes (graph only). = 3 (= `mantis_graph::WIN_AXES.len()`).
    pub win_axes: Option<usize>,
    /// Ragged-payload contract version this encoding speaks (graph only). = 1.
    pub contract_version: Option<u32>,
    /// Required builder_impl tag the resolver asserts (graph only). = 1 (native;
    /// = `mantis_graph::BUILDER_IMPL_NATIVE`).
    pub builder_impl_required: Option<u8>,
}

impl RegistrySpec {
    /// True for the axis-graph GNN encodings (`representation == Graph`).
    #[inline]
    #[must_use]
    pub fn is_graph(&self) -> bool {
        self.representation.is_graph()
    }

    /// Total cells per trunk input tensor = `trunk_size²`.
    ///
    /// `board_size` is canvas geometry; `trunk_size` is NN input geometry
    /// (= `cluster_window_size` for multi-window, = `board_size` for
    /// single-window).
    #[inline]
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.trunk_size * self.trunk_size
    }

    /// (board_size − 1) / 2 — board half-extent for axial→canvas mapping.
    #[inline]
    #[must_use]
    pub fn half(&self) -> i32 {
        (self.board_size as i32 - 1) / 2
    }

    /// State plane stride = n_planes × n_cells.
    #[inline]
    #[must_use]
    pub fn state_stride(&self) -> usize {
        self.n_planes * self.n_cells()
    }

    /// Chain plane stride = `n_chain_planes` × n_cells. The `n_chain_planes`
    /// TOML field is the authority (the old reach-through into the replay
    /// module's `N_CHAIN_PLANES` constant is severed — DAG).
    #[inline]
    #[must_use]
    pub fn chain_stride(&self) -> usize {
        self.n_chain_planes * self.n_cells()
    }

    /// Aux plane stride = n_cells (single aux plane).
    #[inline]
    #[must_use]
    pub fn aux_stride(&self) -> usize {
        self.n_cells()
    }

    /// Policy stride = `policy_logit_count` (accessor for parity with the strides above).
    #[inline]
    #[must_use]
    pub fn policy_stride(&self) -> usize {
        self.policy_logit_count
    }

    /// Kept-slot index of a source plane within `kept_plane_indices`.
    /// Panics if the source plane is not retained by this encoding.
    #[inline]
    fn kept_slot_of(&self, src_plane: usize) -> usize {
        self.kept_plane_indices
            .iter()
            .position(|&p| p == src_plane)
            .unwrap_or_else(|| {
                panic!(
                    "encoding {:?} does not keep source plane {} (kept={:?})",
                    self.name, src_plane, self.kept_plane_indices
                )
            })
    }

    /// Slice index of the current-player t0 stone plane (source plane 0).
    /// Always 0 today but derived from the registry so a plane-reorder cannot
    /// silently shift it.
    #[inline]
    #[must_use]
    pub fn cur_stone_slot(&self) -> usize {
        self.kept_slot_of(0)
    }

    /// Slice index of the opponent t0 stone plane (source plane 8). Slot 4 for
    /// the v6 family (kept [0,1,2,3,8,…]), slot 1 for the 4-plane live set
    /// (kept [0,8,16,17]).
    #[inline]
    #[must_use]
    pub fn opp_stone_slot(&self) -> usize {
        self.kept_slot_of(8)
    }

    /// Kept-slot indices of the history planes (source 1,2,3 / 9,10,11) the
    /// encoding retains. Empty for the 4-plane live set (history dropped).
    #[must_use]
    pub fn history_planes(&self) -> Vec<usize> {
        const HISTORY_SRC: [usize; 6] = [1, 2, 3, 9, 10, 11];
        self.kept_plane_indices
            .iter()
            .enumerate()
            .filter(|(_, &p)| HISTORY_SRC.contains(&p))
            .map(|(slot, _)| slot)
            .collect()
    }

    /// Kept-slot indices of the turn-phase planes (source 16,17) the encoding
    /// retains. Non-empty only for the 4-plane live set.
    #[must_use]
    pub fn turn_phase_planes(&self) -> Vec<usize> {
        const TURN_PHASE_SRC: [usize; 2] = [16, 17];
        self.kept_plane_indices
            .iter()
            .enumerate()
            .filter(|(_, &p)| TURN_PHASE_SRC.contains(&p))
            .map(|(slot, _)| slot)
            .collect()
    }

    /// Wire-format signature for cross-encoding compatibility checks.
    ///
    /// Two encodings are wire-identical when they produce byte-identical on-disk
    /// rows for the replay-buffer format. The wire layout depends on
    /// `(n_planes, board_size, policy_logit_count, has_pass_slot, sym_table_id)`
    /// — every other registry field affects training semantics but not stored
    /// bytes. Registered families:
    ///   - v6           → (8, 19, 362, true, "size_19")
    ///   - v6w25        → (8, 25, 626, true, "size_25")
    ///   - v6_live2_ls  → (4, 19, 362, true, "size_19")
    ///   - gnn_axis_v1  → (0, 19, 362, true, "size_19")  (graph; no dense rows)
    ///
    /// Derived from existing fields — the TOML source of truth is untouched.
    #[inline]
    #[must_use]
    pub fn wire_signature(&self) -> (usize, usize, usize, bool, &'static str) {
        (
            self.n_planes,
            self.board_size,
            self.policy_logit_count,
            self.has_pass_slot,
            self.sym_table_id,
        )
    }
}
