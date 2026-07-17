//! HEXG — graph-position replay ring for the GNN training-data path.
//!
//! A PARALLEL ring beside the dense `ReplayBuffer`. It stores a COMPACT
//! whole-board position record — sorted stone list + sparse coord-keyed MCTS
//! visit target + outcome/value_valid + per-game scalars — and rebuilds the axis
//! graph + aligns the policy target AT SAMPLE TIME on the native builder
//! (`mantis_graph::build_axis_graph`). NO dense planes, NO aux, NO K-cluster.
//!
//! Ported from the predecessor engine's `replay_buffer/hexg/` with the
//! FFI-binding strip. The old sample path fused the per-graph builds via the
//! block-diagonal graph-wire fuse (`from_axis_graphs`) and returned that wire +
//! `GraphTargets`; the terminal fuse is deferred to WP6 (the wire type lives in
//! the predecessor inference-bridge module, which routes to WP6). WP5's
//! `sample_graph_batch_impl` returns the buffer-owned `(Vec<AxisGraph>,
//! GraphTargets)` — for a single graph local == global, so the fuse changes no
//! computed value (R-1).
//!
//! ## Sample = rebuild-at-native-builder
//! `sample_graph_batch_impl` weighted-samples record indices, D6-rotates the
//! stored stone coords AND the visit-map keys by one uniform per-sample element
//! (`sym::rotate_axial` — the single source shared with the CNN cell-scatter),
//! rebuilds via `build_axis_graph` (which stamps `builder_impl = 1`), and aligns
//! the rotated visit-keys to the built legal nodes → the per-legal-node policy
//! target. One call emits graph + target together, so a graph/target desync is
//! structurally impossible.

mod persist;
pub mod push;
pub mod sample;
mod storage;

use std::sync::atomic::AtomicU64;

use half::f16;
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::schedule::WeightSchedule;
use mantis_encoding::RegistrySpec;

// ── fixed-slot geometry ────────────────────────────────────────────────────────

/// Max stones per record slot. Over-cap push is a LOUD error.
pub const MAX_STONES: usize = 256;
/// Max sparse visit entries per record slot. Over-cap push is a LOUD error.
pub const MAX_VISITS: usize = 128;

/// HEXG on-disk magic — "HEXG" little-endian (distinct from HEXB `0x48455842`).
pub const HEXG_MAGIC: u32 = 0x4845_5847;
/// HEXG on-disk version. v1.
pub const HEXG_VERSION: u32 = 1;

/// Weight-bucket boundaries mirror `ReplayBuffer::weight_bucket`.
#[inline]
pub(crate) fn weight_bucket(w_bits: u16) -> usize {
    let w = f16::from_bits(w_bits).to_f32();
    if w < 0.30 {
        0
    } else if w < 0.75 {
        1
    } else {
        2
    }
}

/// The single compact graph-position record. Coords are `i16`; the visit target
/// is the sparse coord→prob MCTS distribution over the FULL legal set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphRecord {
    /// Sorted (order irrelevant — the builder re-sorts) stone list `(q, r, ±1)`.
    pub stones: Vec<(i16, i16, i8)>,
    /// Sparse coord-keyed visit target `(q, r, prob)` over legal moves.
    pub visits: Vec<(i16, i16, f32)>,
    /// Side to move (+1 / −1).
    pub current_player: i8,
    /// Moves remaining this turn (0..=255).
    pub moves_remaining: u8,
    /// 0-based ply of this decision.
    pub ply_index: u16,
    /// Move-level playout-cap flag — policy-loss gate.
    pub is_full_search: bool,
    /// Outcome z (placeholder at record time → filled at finalize).
    pub outcome: f32,
    /// 1 = supervise value, 0 = ply-capped row masked.
    pub value_valid: bool,
    /// Completed-game length (compound moves) — sampling weight.
    pub game_length: u16,
}

// ── HexgBuffer ─────────────────────────────────────────────────────────────────

/// Graph-position replay ring (parallel to `ReplayBuffer`). Fixed-slot SoA Vecs;
/// ring overwrite by `head`; weighted rejection sampler + game-length weight
/// schedule lifted verbatim from HEXB. Fields are `pub` for the relocated HEXG
/// oracle suite (`tests/replay_hexg.rs`).
pub struct HexgBuffer {
    pub capacity: usize,
    pub size: usize,
    pub head: usize,

    /// Encoding spec — a `representation == Graph` spec.
    pub encoding: &'static RegistrySpec,
    pub win_length: u8,
    pub radius: u16,
    pub trunk_size: i32,
    pub contract_version: u32,

    // ── fixed-slot record storage (SoA) ──
    pub stones_qr: Vec<i16>,     // flat [cap * MAX_STONES * 2]
    pub stone_players: Vec<i8>,  // flat [cap * MAX_STONES]
    pub n_stones: Vec<u16>,      // [cap]
    pub visit_qr: Vec<i16>,      // flat [cap * MAX_VISITS * 2]
    pub visit_probs: Vec<f32>,   // flat [cap * MAX_VISITS]
    pub n_visits: Vec<u16>,      // [cap]
    pub current_player: Vec<i8>, // [cap]
    pub moves_remaining: Vec<u8>, // [cap]
    pub ply_index: Vec<u16>,     // [cap]
    pub is_full_search: Vec<u8>, // [cap]
    pub outcomes: Vec<f32>,      // [cap]
    pub value_valid: Vec<u8>,    // [cap]
    pub game_length: Vec<u16>,   // [cap]
    pub game_ids: Vec<i64>,      // [cap]; -1 = untagged
    pub weights: Vec<u16>,       // f16 bits; [cap]

    pub weight_schedule: WeightSchedule,
    pub next_game_id: i64,
    pub rng: StdRng,
    pub weight_buckets: [AtomicU64; 3],
}

impl HexgBuffer {
    /// Create a graph-position ring with `capacity` records.
    ///
    /// `encoding` MUST be a `representation == "graph"` spec — the rebuild
    /// `BuildParams` come from its graph fields. A grid encoding is a LOUD error.
    pub fn new(capacity: usize, encoding: &str) -> Result<Self, String> {
        let spec = mantis_encoding::registry::lookup_or_panic(encoding);
        if !spec.is_graph() {
            return Err(format!(
                "HexgBuffer requires a graph encoding; '{encoding}' is representation=grid \
                 (use ReplayBuffer for dense encodings)"
            ));
        }
        let win_length = spec.win_length.expect("validate guarantees win_length for a graph spec") as u8;
        let radius = spec.graph_radius.expect("validate guarantees graph_radius for a graph spec") as u16;
        let contract_version = spec
            .contract_version
            .expect("validate guarantees contract_version for a graph spec");
        let default_w = f16::from_f32(1.0).to_bits();
        Ok(HexgBuffer {
            capacity,
            size: 0,
            head: 0,
            encoding: spec,
            win_length,
            radius,
            trunk_size: spec.trunk_size as i32,
            contract_version,
            stones_qr: vec![0i16; capacity * MAX_STONES * 2],
            stone_players: vec![0i8; capacity * MAX_STONES],
            n_stones: vec![0u16; capacity],
            visit_qr: vec![0i16; capacity * MAX_VISITS * 2],
            visit_probs: vec![0.0f32; capacity * MAX_VISITS],
            n_visits: vec![0u16; capacity],
            current_player: vec![1i8; capacity],
            moves_remaining: vec![2u8; capacity],
            ply_index: vec![0u16; capacity],
            is_full_search: vec![1u8; capacity],
            outcomes: vec![0.0f32; capacity],
            value_valid: vec![1u8; capacity],
            game_length: vec![0u16; capacity],
            game_ids: vec![-1i64; capacity],
            weights: vec![default_w; capacity],
            weight_schedule: WeightSchedule::uniform(),
            next_game_id: 0,
            rng: StdRng::from_rng(&mut rand::rng()),
            weight_buckets: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        })
    }

    /// Fresh monotonic game id.
    pub fn next_game_id(&mut self) -> i64 {
        let id = self.next_game_id;
        self.next_game_id += 1;
        id
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub fn encoding_name(&self) -> &'static str {
        self.encoding.name
    }
}

/// Aligned training targets emitted alongside the per-graph `Vec<AxisGraph>` by
/// `sample_graph_batch_impl`. Plain-Rust struct (the old binding getters move to
/// WP7); `target_argmax_cells` is a pure method.
///
/// * `policy_target` — flat `[Lg]` per-legal-node CE target (graphs concatenated,
///   in `legal_node_gather` order); each graph's segment sums to ~1.
/// * `outcomes` / `value_valid` — `[B]` value target + draw-mask.
/// * `is_full_search` — `[B]` policy-loss gate.
/// * argmax_q/argmax_r/argmax_valid — per-graph max-mass legal node in the
///   ROTATED frame (the AugRoundTrip runtime canary), decoded by
///   `target_argmax_cells`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphTargets {
    pub policy_target: Vec<f32>,
    pub outcomes: Vec<f32>,
    pub value_valid: Vec<u8>,
    pub is_full_search: Vec<u8>,
    pub argmax_q: Vec<i32>,
    pub argmax_r: Vec<i32>,
    pub argmax_valid: Vec<u8>,
}

impl GraphTargets {
    /// `[B]` list of `Optional[(q, r)]` — the collate `target_argmax_cells` arg.
    #[must_use]
    pub fn target_argmax_cells(&self) -> Vec<Option<(i32, i32)>> {
        (0..self.argmax_valid.len())
            .map(|i| {
                if self.argmax_valid[i] != 0 {
                    Some((self.argmax_q[i], self.argmax_r[i]))
                } else {
                    None
                }
            })
            .collect()
    }
}
