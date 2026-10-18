//! `WorkerParams` + themed sub-flag bundles + the per-worker `WorkerGeometry`, cloned once per
//! worker spawn and destructured at `game::run_worker_thread` entry, so the per-sim hot path sees
//! local scalars, never a `&RegistrySpec` field access.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use mantis_encoding::{EmptyEdges, RegistrySpec, Representation};
use mantis_graph::CanvasParams;
use mantis_search::{QSigma, SearchKind};

use crate::queues::GraphQueue;
use crate::replay::hexg::GraphRecord;

use super::GameResultRow;

/// Per-worker geometry scalars resolved ONCE from the `RegistrySpec` at spawn
/// time (D2). `Copy` (~32 B); passed by value to `run_worker_thread` and
/// destructured into local scalar bindings at fn entry.

#[derive(Clone, Copy)]
pub struct WorkerGeometry {
    pub policy_stride: usize,
    pub agg_trunk_sz: i32,
    /// The leaf graph builder's win length, resolved at boot so no leaf re-reads the spec.
    pub win_length: u8,
    /// The leaf graph builder's axis-walk radius, resolved at boot likewise.
    pub graph_radius: u16,
    /// Whether the leaf graph builder emits the edges between two empty cells, resolved at boot likewise.
    pub empty_edges: EmptyEdges,
}

/// Why a spec has no builder geometry for the path that asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryError {
    /// A builder key is absent (`value` `None`) or does not fit the builder's integer width.
    Key {
        spec: &'static str,
        key: &'static str,
        value: Option<usize>,
    },
    /// The spec is of another representation than the builder the path runs.
    Representation {
        spec: &'static str,
        representation: &'static str,
        builder: &'static str,
    },
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Key { spec, key, value: None } => write!(f, "spec {spec:?} does not define {key}"),
            Self::Key { spec, key, value: Some(v) } => write!(
                f,
                "spec {spec:?} has {key} = {v}, outside the builder's integer width"
            ),
            Self::Representation { spec, representation, builder } => write!(
                f,
                "spec {spec:?} is representation={representation:?}, and this path runs the {builder} builder"
            ),
        }
    }
}

impl std::error::Error for GeometryError {}

/// Narrow one optional spec key to the builder's width, naming the key on refusal.
fn builder_key<T: TryFrom<usize>>(
    spec: &'static RegistrySpec,
    key: &'static str,
    value: Option<usize>,
) -> Result<T, GeometryError> {
    let refuse = |value| GeometryError::Key {
        spec: spec.name,
        key,
        value,
    };
    let v = value.ok_or_else(|| refuse(None))?;
    T::try_from(v).map_err(|_| refuse(Some(v)))
}

/// The refusal of a spec whose representation is not the one `builder` builds.
fn foreign(spec: &'static RegistrySpec, builder: &'static str) -> GeometryError {
    GeometryError::Representation {
        spec: spec.name,
        representation: spec.representation.as_str(),
        builder,
    }
}

/// Resolve the per-worker geometry ONCE, at `SelfPlayRunner::new`, from the resolved spec.
///
/// The kind dispatch is a **closed** `match spec.representation` with NO `_ =>` arm, and every
/// caller builds axis graphs, so a canvas row is refused here by name.
///
/// # Errors
/// [`GeometryError`] when a builder key is absent or out of width, or the spec is not a graph row.
pub fn resolve_geometry(spec: &'static RegistrySpec) -> Result<WorkerGeometry, GeometryError> {
    match spec.representation {
        Representation::Graph => Ok(WorkerGeometry {
            policy_stride: spec.policy_stride(),
            agg_trunk_sz: spec.trunk_size as i32,
            win_length: builder_key(spec, "win_length", spec.win_length)?,
            graph_radius: builder_key(spec, "graph_radius", spec.graph_radius)?,
            empty_edges: spec.empty_edges.ok_or(GeometryError::Key {
                spec: spec.name,
                key: "empty_edges",
                value: None,
            })?,
        }),
        Representation::HexCanvas => Err(foreign(spec, "axis-graph")),
    }
}

/// A canvas row's builder params, narrowed once; `current_player` and `moves_remaining` are the caller's per position.
/// # Errors
/// [`GeometryError`] when a canvas key is absent or out of width, or the spec is not a canvas row.
pub fn resolve_canvas_geometry(spec: &'static RegistrySpec) -> Result<CanvasParams, GeometryError> {
    match spec.representation {
        Representation::HexCanvas => Ok(CanvasParams {
            radius: builder_key(spec, "legal_move_radius", Some(spec.legal_move_radius))?,
            trunk_size: builder_key(spec, "trunk_size", Some(spec.trunk_size))?,
            side_step: builder_key(spec, "canvas_side_step", spec.canvas_side_step)?,
            min_side: builder_key(spec, "canvas_min_side", spec.canvas_min_side)?,
            max_side: builder_key(spec, "canvas_max_side", spec.canvas_max_side)?,
            far_halo: builder_key(spec, "far_halo", spec.far_halo)?,
            current_player: 1,
            moves_remaining: 2,
        }),
        Representation::Graph => Err(foreign(spec, "hex-canvas")),
    }
}

#[derive(Clone)]
pub(crate) struct SearchFlags {
    pub(crate) quiescence_enabled: bool,
    /// Set on the tree ONCE per worker (`configure_search`), like quiescence — it is
    /// per-worker configuration, not per-move state.
    pub(crate) search_kind: SearchKind,
}

#[derive(Clone)]
pub(crate) struct ExplorationFlags {
    pub(crate) dirichlet_enabled: bool,
}

/// Per-worker channel/queue bundle — the graph inference queue the worker submits to, plus the
/// shared result queues it drains into.
#[derive(Clone)]
pub(crate) struct WorkerChannels {
    pub(crate) graph_queue: GraphQueue,
    pub(crate) recent_game_results: Arc<Mutex<VecDeque<GameResultRow>>>,
    /// Graph-position results queue.
    pub(crate) graph_results_queue: Arc<Mutex<VecDeque<GraphRecord>>>,
}

#[derive(Clone)]
pub(crate) struct WorkerParams {
    pub(crate) max_moves: usize,
    pub(crate) leaf_batch_size: usize,
    pub(crate) c_puct: f32,
    pub(crate) fpu_reduction: f32,
    pub(crate) quiescence_blend_2: f32,
    /// Every searched move's sims, the full arm's when the playout cap is armed.
    pub(crate) n_simulations: usize,
    pub(crate) temp_threshold: usize,
    pub(crate) temp_min: f32,
    pub(crate) sigma: QSigma,
    pub(crate) gumbel_m: usize,
    pub(crate) gumbel_m_quick: usize,
    pub(crate) gumbel_explore_moves: usize,
    pub(crate) dirichlet_alpha: f32,
    pub(crate) dirichlet_epsilon: f32,
    pub(crate) results_queue_cap: usize,
    pub(crate) full_search_prob: f32,
    pub(crate) n_sims_quick: usize,
    pub(crate) random_opening_plies: u32,
    pub(crate) search_stats_every: usize,
    /// DERIVED HEXG visit-slot capacity — composed once in
    /// `SelfPlayRunner::new`, never a default.
    pub(crate) visit_capacity: usize,
    /// Resolved (never `None`) encoding spec, used by the per-game board
    /// construction (`init_per_game_board`).
    pub(crate) registry_spec: &'static RegistrySpec,
    pub(crate) search_flags: SearchFlags,
    pub(crate) exploration_flags: ExplorationFlags,
    /// The tactics block the worker's tree runs; `None` is off.
    pub(crate) tactics: Option<mantis_search::mcts::TacticsConfig>,
    /// The root offence's first decided stone starts a decided tail.
    pub(crate) decided_tails: bool,
}

#[cfg(test)]
mod geometry_tests {
    use super::{resolve_canvas_geometry, resolve_geometry, GeometryError};

    fn spec_with(
        win_length: Option<usize>,
        graph_radius: Option<usize>,
    ) -> &'static mantis_encoding::RegistrySpec {
        let mut spec = *mantis_encoding::lookup_or_panic("gnn_axis_v1");
        spec.win_length = win_length;
        spec.graph_radius = graph_radius;
        Box::leak(Box::new(spec))
    }

    /// The registry's own graph spec resolves to exactly its declared builder geometry.
    #[test]
    fn the_registry_spec_resolves_its_builder_geometry() {
        let spec = mantis_encoding::lookup_or_panic("gnn_axis_v1");
        let g = resolve_geometry(spec).expect("a registry graph spec resolves");
        assert_eq!(Some(usize::from(g.win_length)), spec.win_length);
        assert_eq!(Some(usize::from(g.graph_radius)), spec.graph_radius);
        assert_eq!(Some(g.empty_edges), spec.empty_edges);
        let pruned = mantis_encoding::lookup_or_panic("gnn_axis_r8_pruned");
        let g = resolve_geometry(pruned).expect("the pruned row resolves");
        assert_eq!(g.empty_edges, mantis_encoding::EmptyEdges::Pruned);
    }

    /// An absent or over-wide key is refused at boot by name, never at the first leaf by panic.
    #[test]
    fn an_absent_or_over_wide_key_is_a_named_refusal() {
        let refusal = |key, value| GeometryError::Key {
            spec: "gnn_axis_v1",
            key,
            value,
        };
        let absent = resolve_geometry(spec_with(None, Some(6))).err();
        assert_eq!(absent, Some(refusal("win_length", None)));
        let wide = resolve_geometry(spec_with(Some(6), Some(70_000))).err();
        assert_eq!(wide, Some(refusal("graph_radius", Some(70_000))));
        assert_eq!(
            refusal("win_length", None).to_string(),
            "spec \"gnn_axis_v1\" does not define win_length"
        );
    }

    /// The graph paths refuse a canvas row by name, and the canvas resolver a graph row: no builder runs another's spec.
    #[test]
    fn a_spec_of_the_other_representation_is_refused_by_name() {
        let canvas = mantis_encoding::lookup_or_panic("hexcanvas_r8");
        let refused = resolve_geometry(canvas).err();
        assert!(
            matches!(
                refused,
                Some(GeometryError::Representation {
                    representation: "hexcanvas",
                    ..
                })
            ),
            "the axis-graph geometry admitted a canvas row: {refused:?}"
        );
        let graph = mantis_encoding::lookup_or_panic("gnn_axis_r8");
        let refused = resolve_canvas_geometry(graph).err();
        assert!(
            matches!(
                refused,
                Some(GeometryError::Representation {
                    representation: "graph",
                    ..
                })
            ),
            "the canvas geometry admitted a graph row: {refused:?}"
        );
        let params = resolve_canvas_geometry(canvas).expect("the canvas row resolves");
        assert_eq!(
            (
                usize::from(params.radius),
                usize::from(params.max_side),
                usize::from(params.far_halo)
            ),
            (
                canvas.legal_move_radius,
                canvas.canvas_max_side.unwrap_or(0),
                canvas.far_halo.unwrap_or(0)
            ),
            "the canvas params are not the row's"
        );
    }
}
