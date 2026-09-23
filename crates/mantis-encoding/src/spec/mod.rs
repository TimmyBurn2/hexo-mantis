//! Encoding registry spec — full-schema record per `registry.toml`.
//!
//! The sole encoding record type. Per-Board construction never binds a spec into
//! mantis-core; consumers thread `&RegistrySpec` as a parameter.

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
    /// Scatter-max over K cluster windows WITHOUT the off-window drop: the target is a ragged
    /// board-coord-keyed legal set, retaining off-global-window cells some cluster covers.
    LegalSetScatterMax,
}

/// Input representation discriminant; the TOML key is REQUIRED and absent is a parse error.
/// The enum stays a closed one-member type rather than disappearing, because that is what makes
/// an absent or unknown representation an error instead of a fall-through across the FFI.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Representation {
    Graph,
}

impl Representation {
    /// Parse the TOML value string. `"grid"` is refused BY NAME, not merely unknown, so a stale
    /// row says the dense path is gone rather than resolving to graph.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "graph" => Ok(Representation::Graph),
            "grid" => Err(
                "representation=\"grid\" was DELETED with the dense path (R346(f)); \
                 `archive/grid-path` carries the three grid rows"
                    .to_string(),
            ),
            other => Err(format!("representation must be \"graph\"; got {other:?}")),
        }
    }

    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
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
/// All `&'static` fields point at heap data leaked at registry init time, so addresses are
/// stable for the process lifetime. Cheap to copy — pass by value or `&'static`.
#[derive(Copy, Clone, Debug)]
pub struct RegistrySpec {
    pub name: &'static str,
    pub board_size: usize,
    pub trunk_size: usize,
    pub cluster_window_size: Option<usize>,
    pub cluster_threshold: Option<usize>,
    pub legal_move_radius: usize,
    pub n_planes: usize,
    pub plane_layout: &'static [&'static str],
    pub policy_logit_count: usize,
    pub has_pass_slot: bool,
    pub is_multi_window: bool,
    pub value_pool: ValuePool,
    pub policy_pool: PolicyPool,
    pub sym_table_id: &'static str,
    pub schema_version: u32,
    pub notes: &'static str,

    /// Physical source-plane indices retained by this encoding's wire format; length ==
    /// `n_planes`. The `registry.toml` header carries the plane-block convention.
    pub kept_plane_indices: &'static [usize],
    /// Source tensor plane count *before* the `kept_plane_indices` slice.
    /// Used by the validator for the kept-indices upper bound.
    pub n_source_planes: usize,

    /// Multi-window cluster-count upper bound per position. Single-window
    /// encodings emit exactly 1 view per leaf (`k_max = 1`).
    pub k_max: u32,

    /// Number of chain-length planes (3 hex axes × 2 players). Required TOML field, and the
    /// sole authority for `chain_stride()`.
    pub n_chain_planes: usize,

    /// TOML key `representation`; required, and absent is an error.
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

    /// State plane stride = n_planes × n_cells.
    #[inline]
    #[must_use]
    pub fn state_stride(&self) -> usize {
        self.n_planes * self.n_cells()
    }

    /// Chain plane stride = `n_chain_planes` × n_cells, with the TOML field as the authority.
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
}
