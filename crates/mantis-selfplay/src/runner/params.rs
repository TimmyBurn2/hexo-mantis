//! `WorkerParams` + themed sub-flag bundles + the per-worker `WorkerGeometry`, cloned once per
//! worker spawn and destructured at `game::run_worker_thread` entry, so the per-sim hot path sees
//! local scalars, never a `&RegistrySpec` field access.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use mantis_encoding::{RegistrySpec, Representation};
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
}

/// A graph spec whose builder geometry is absent or does not fit the builder's integer width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphGeometryError {
    pub spec: &'static str,
    pub key: &'static str,
    /// The spec's value; `None` when the key is absent.
    pub value: Option<usize>,
}

impl std::fmt::Display for GraphGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value {
            None => write!(f, "graph spec {:?} does not define {}", self.spec, self.key),
            Some(v) => write!(
                f,
                "graph spec {:?} has {} = {v}, outside the builder's integer width",
                self.spec, self.key
            ),
        }
    }
}

impl std::error::Error for GraphGeometryError {}

/// Narrow one optional spec key to the builder's width, naming the key on refusal.
fn builder_key<T: TryFrom<usize>>(
    spec: &'static RegistrySpec,
    key: &'static str,
    value: Option<usize>,
) -> Result<T, GraphGeometryError> {
    let refuse = |value| GraphGeometryError {
        spec: spec.name,
        key,
        value,
    };
    let v = value.ok_or_else(|| refuse(None))?;
    T::try_from(v).map_err(|_| refuse(Some(v)))
}

/// Resolve the per-worker geometry ONCE, at `SelfPlayRunner::new`, from the resolved spec.
///
/// The kind dispatch is a **closed** `match spec.representation` with NO `_ =>` arm, so a new
/// `Representation` variant fails compilation loudly.
///
/// # Errors
/// [`GraphGeometryError`] when `win_length` or `graph_radius` is absent or out of width.
pub fn resolve_geometry(spec: &'static RegistrySpec) -> Result<WorkerGeometry, GraphGeometryError> {
    match spec.representation {
        Representation::Graph => Ok(WorkerGeometry {
            policy_stride: spec.policy_stride(),
            agg_trunk_sz: spec.trunk_size as i32,
            win_length: builder_key(spec, "win_length", spec.win_length)?,
            graph_radius: builder_key(spec, "graph_radius", spec.graph_radius)?,
        }),
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
    pub(crate) fast_prob: f32,
    pub(crate) fast_sims: usize,
    pub(crate) standard_sims: usize,
    pub(crate) temp_threshold: usize,
    pub(crate) temp_min: f32,
    pub(crate) draw_reward: f32,
    /// §178: terminal-via-ply-cap outcome (distinct from `draw_reward`).
    pub(crate) ply_cap_value: f32,
    pub(crate) sigma: QSigma,
    pub(crate) gumbel_m: usize,
    pub(crate) gumbel_explore_moves: usize,
    pub(crate) dirichlet_alpha: f32,
    pub(crate) dirichlet_epsilon: f32,
    pub(crate) results_queue_cap: usize,
    pub(crate) full_search_prob: f32,
    pub(crate) n_sims_quick: usize,
    pub(crate) n_sims_full: usize,
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
}

#[cfg(test)]
mod geometry_tests {
    use super::{resolve_geometry, GraphGeometryError};

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
    }

    /// An absent or over-wide key is refused at boot by name, never at the first leaf by panic.
    #[test]
    fn an_absent_or_over_wide_key_is_a_named_refusal() {
        let refusal = |key, value| GraphGeometryError {
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
            "graph spec \"gnn_axis_v1\" does not define win_length"
        );
    }
}
