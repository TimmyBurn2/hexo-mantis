//! Encoding registry spec — full-schema record per `registry.toml`.
//!
//! The sole encoding record type. Per-Board construction never binds a spec into
//! mantis-core; consumers thread `&RegistrySpec` as a parameter.

mod validate;

pub use mantis_graph::EmptyEdges;

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
/// The enum stays closed, because that is what makes an absent or unknown representation an
/// error instead of a fall-through across the FFI.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Representation {
    Graph,
    /// The fence masked inside its smallest axial box (`mantis_graph::build_canvas`, contract #12).
    HexCanvas,
}

impl Representation {
    /// Parse the TOML value string. `"grid"` and `"dense"` are refused BY NAME, not merely
    /// unknown, so a stale row says the K-window path is gone rather than resolving to a canvas.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "graph" => Ok(Representation::Graph),
            "hexcanvas" => Ok(Representation::HexCanvas),
            "grid" => Err(
                "representation=\"grid\" was DELETED with the dense path (R346(f)); \
                 `archive/grid-path` carries the three grid rows"
                    .to_string(),
            ),
            "dense" => Err(
                "representation=\"dense\" is no token: the dense K-window path was DELETED as \
                 \"grid\" (R346(f)), and the hex-canvas kind is \"hexcanvas\""
                    .to_string(),
            ),
            other => Err(format!(
                "representation must be \"graph\" or \"hexcanvas\"; got {other:?}"
            )),
        }
    }

    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Representation::Graph => "graph",
            Representation::HexCanvas => "hexcanvas",
        }
    }

    #[inline]
    #[must_use]
    pub fn is_graph(&self) -> bool {
        matches!(self, Representation::Graph)
    }
}

/// Parse the TOML `empty_edges` value; there is no default, so anything but the two names is an error.
pub fn parse_empty_edges(s: &str) -> Result<EmptyEdges, String> {
    match s {
        "kept" => Ok(EmptyEdges::Kept),
        "pruned" => Ok(EmptyEdges::Pruned),
        other => Err(format!(
            "empty_edges must be one of [kept,pruned]; got {other:?}"
        )),
    }
}

/// The TOML spelling of an `empty_edges` value.
#[must_use]
pub fn empty_edges_str(e: EmptyEdges) -> &'static str {
    match e {
        EmptyEdges::Kept => "kept",
        EmptyEdges::Pruned => "pruned",
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
#[derive(Copy, Clone, Debug, PartialEq)]
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
    /// The game's win length (graph and hexcanvas): the graph builder walks it, the ring rule compares it. = 6.
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
    /// TOML key `empty_edges` (graph only, required there): whether the builder emits the edges between two empty cells.
    pub empty_edges: Option<EmptyEdges>,
    /// The canvas container's side quantum (hexcanvas only). = 8.
    pub canvas_side_step: Option<usize>,
    /// The canvas container's smallest side (hexcanvas only). = 24.
    pub canvas_min_side: Option<usize>,
    /// S_max: past this smallest square bucket the canvas is in far mode (hexcanvas only).
    pub canvas_max_side: Option<usize>,
    /// The far region's reach around each stone (hexcanvas only). = 4.
    pub far_halo: Option<usize>,
    /// S_far: a far-mode root past this smallest square bucket is refused (hexcanvas only).
    pub far_max_side: Option<usize>,
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

    /// Whether a ring written under `writer` holds this row's records: the rows differ at most in the edge set, or a canvas row reads a graph row of its radius and win length.
    #[must_use]
    pub fn reads_rings_of(&self, writer: &RegistrySpec) -> bool {
        if self.representation == Representation::HexCanvas
            && writer.representation == Representation::Graph
        {
            return self.legal_move_radius == writer.legal_move_radius
                && self.win_length == writer.win_length;
        }
        let as_self = RegistrySpec {
            name: self.name,
            notes: self.notes,
            empty_edges: self.empty_edges,
            ..*writer
        };
        as_self == *self
    }

    /// Policy stride = `policy_logit_count` (accessor for parity with the strides above).
    #[inline]
    #[must_use]
    pub fn policy_stride(&self) -> usize {
        self.policy_logit_count
    }
}
