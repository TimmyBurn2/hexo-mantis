//! HEXG — graph-position replay ring for the GNN training-data path.
//!
//! R8: >300 LOC by design — the record/buffer types, the slot-geometry
//! constants, and the R255 capacity-derivation authority
//! (`derived_visit_capacity` + its ceiling) are one contract unit: the derivation
//! IS the slot geometry, and splitting it from the struct it sizes would let the
//! two drift apart, which is the exact defect ADJ-D34 closed.
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

// ── slot geometry ──────────────────────────────────────────────────────────────

/// Max stones per record slot. Over-cap push is a LOUD error.
pub const MAX_STONES: usize = 256;

/// Structural ceiling on a record's visit-slot capacity: the per-record
/// `n_visits` counter (SoA field and HEXG on-disk field alike) is `u16`, so no
/// capacity past `u16::MAX` can be stored whatever the sims regime asks.
/// DERIVED from the storage type, never tuned (R255/ADJ-D34: the guard carries
/// no literal — the old `MAX_VISITS = 128` tunable is deleted).
pub const HEXG_VISIT_COUNT_CEILING: usize = u16::MAX as usize;

/// The ONE effective-standard-budget resolution: `standard_sims` wins when set,
/// else `n_simulations`. Shared by [`derived_visit_capacity`] and the runner's
/// own zero-check + budget bake (`SelfPlayRunner::new`) so the guard capacity and
/// the workers' baked budget cannot silently diverge onto two copies of the rule.
#[must_use]
pub fn effective_standard_sims(n_simulations: usize, standard_sims: usize) -> usize {
    if standard_sims == 0 {
        n_simulations
    } else {
        standard_sims
    }
}

/// R255/ADJ-D34 — THE derivation authority for the HEXG visit-slot capacity.
///
/// `capacity = max(ARMED effective sim budgets) + leaf_batch_size − 1` — the
/// production sim loops overshoot by up to `leaf_batch_size − 1` (the uncapped
/// final batch), so this is the largest positive-mass support a graph record
/// can carry. Armed arms: standard (always; effective = `standard_sims` else
/// `n_simulations`), fast iff `fast_prob > 0`, quick/full iff
/// `full_search_prob > 0`.
///
/// Called by BOTH enforcement surfaces — the mint-time schema validator
/// (through the bridge twin `derived_hexg_visit_capacity`) and the
/// `SelfPlayRunner` boot guard — so the two cannot drift onto second formulas.
///
/// SCOPE (R275(a)): this formula is derived from the CURRENT visit-limited target
/// construction, and so are the two F-816-9 pins that sit downstream of it
/// (`records::refuse_zero_visit_export` and `search_drive::InferenceSeamFailure`). If
/// completed-Q-on-graph is adopted at prereg, the capacity AND both pins re-derive from
/// the new construction — a completed-Q export is child-count-wide, not sims-bounded, which
/// is exactly what the second error arm below refuses today (LAW-02: re-derive, never carry
/// the prior across a regime change).
///
/// # Errors
/// * the derived capacity exceeds [`HEXG_VISIT_COUNT_CEILING`] — no slot sizing
///   can honor the regime; the schema twin makes this a MINT-time error, and
///   the boot-side call is defense-in-depth for un-minted constructions;
/// * `completed_q_values` on the graph path while
///   `mantis_search::MAX_CHILDREN_PER_NODE` exceeds the derived capacity — the
///   completed-Q exporter places positive mass on EVERY root child, so its
///   support is child-count-wide, not sims-bounded (WP12-R Phase T guard 2,
///   generalized: the refusal retires per-regime exactly when the derived
///   slots genuinely cover that support).
#[allow(clippy::too_many_arguments)]
pub fn derived_visit_capacity(
    n_simulations: usize,
    standard_sims: usize,
    fast_prob: f32,
    fast_sims: usize,
    full_search_prob: f32,
    n_sims_quick: usize,
    n_sims_full: usize,
    leaf_batch_size: usize,
    completed_q_values: bool,
) -> Result<usize, String> {
    let effective_standard = effective_standard_sims(n_simulations, standard_sims);
    let mut max_armed = effective_standard;
    if fast_prob > 0.0 {
        max_armed = max_armed.max(fast_sims);
    }
    if full_search_prob > 0.0 {
        max_armed = max_armed.max(n_sims_quick).max(n_sims_full);
    }
    let capacity = max_armed + leaf_batch_size.saturating_sub(1);
    if capacity > HEXG_VISIT_COUNT_CEILING {
        return Err(format!(
            "derived HEXG visit capacity {capacity} (max armed sim budget {max_armed} + \
             leaf_batch_size {leaf_batch_size} − 1) exceeds the record-format ceiling \
             {HEXG_VISIT_COUNT_CEILING} (the per-record visit count is u16) — this sims \
             regime cannot be honored by the HEXG record format at any slot sizing; an \
             unsupported regime is a mint-time config error, never a boot surprise \
             (R255/ADJ-D34; keys: selfplay.mcts.n_simulations, selfplay.playout_cap.*, \
             selfplay.leaf_batch_size)"
        ));
    }
    if completed_q_values && mantis_search::MAX_CHILDREN_PER_NODE > capacity {
        return Err(format!(
            "representation==graph with completed_q_values=true is refused while \
             MAX_CHILDREN_PER_NODE ({}) exceeds the derived visit capacity ({capacity}): \
             the completed-Q exporter places positive mass on every root child, so a \
             record's support is child-count-wide and cannot fit the derived HEXG visit \
             slot (WP12-R Phase T, DESIGN_T §3.4; R255) — set completed_q_values=false \
             for graph runs, or raise the armed sim regime until the derived capacity \
             covers it",
            mantis_search::MAX_CHILDREN_PER_NODE,
        ));
    }
    Ok(capacity)
}

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
    /// Per-buffer visit-slot capacity, DERIVED at composition from the sims
    /// regime (`derived_visit_capacity`) — never a literal (R255/ADJ-D34).
    pub visit_capacity: usize,

    // ── fixed-slot record storage (SoA) ──
    pub stones_qr: Vec<i16>,      // flat [cap * MAX_STONES * 2]
    pub stone_players: Vec<i8>,   // flat [cap * MAX_STONES]
    pub n_stones: Vec<u16>,       // [cap]
    pub visit_qr: Vec<i16>,       // flat [cap * visit_capacity * 2]
    pub visit_probs: Vec<f32>,    // flat [cap * visit_capacity]
    pub n_visits: Vec<u16>,       // [cap]
    pub current_player: Vec<i8>,  // [cap]
    pub moves_remaining: Vec<u8>, // [cap]
    pub ply_index: Vec<u16>,      // [cap]
    pub is_full_search: Vec<u8>,  // [cap]
    pub outcomes: Vec<f32>,       // [cap]
    pub value_valid: Vec<u8>,     // [cap]
    pub game_length: Vec<u16>,    // [cap]
    pub game_ids: Vec<i64>,       // [cap]; -1 = untagged
    pub weights: Vec<u16>,        // f16 bits; [cap]

    pub weight_schedule: WeightSchedule,
    pub next_game_id: i64,
    pub rng: StdRng,
    pub weight_buckets: [AtomicU64; 3],
}

impl HexgBuffer {
    /// Create a graph-position ring with `capacity` records and `visit_capacity`
    /// visit slots per record.
    ///
    /// `encoding` MUST be a `representation == "graph"` spec — the rebuild
    /// `BuildParams` come from its graph fields. A grid encoding is a LOUD error.
    /// `visit_capacity` is the DERIVED slot geometry (`derived_visit_capacity` at
    /// the composition site — R255/ADJ-D34: no default, no literal); a value the
    /// format cannot store (`0` or past [`HEXG_VISIT_COUNT_CEILING`]) is a LOUD
    /// error.
    pub fn new(capacity: usize, encoding: &str, visit_capacity: usize) -> Result<Self, String> {
        let spec = mantis_encoding::registry::lookup_or_panic(encoding);
        if !spec.is_graph() {
            return Err(format!(
                "HexgBuffer requires a graph encoding; '{encoding}' is representation=grid \
                 (use ReplayBuffer for dense encodings)"
            ));
        }
        if visit_capacity == 0 || visit_capacity > HEXG_VISIT_COUNT_CEILING {
            return Err(format!(
                "HexgBuffer: visit_capacity {visit_capacity} is outside the record format's \
                 storable range 1..={HEXG_VISIT_COUNT_CEILING} (the per-record visit count \
                 is u16) — derive it from the sims regime via derived_visit_capacity \
                 (R255/ADJ-D34)"
            ));
        }
        let win_length =
            spec.win_length
                .expect("validate guarantees win_length for a graph spec") as u8;
        let radius =
            spec.graph_radius
                .expect("validate guarantees graph_radius for a graph spec") as u16;
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
            visit_capacity,
            stones_qr: vec![0i16; capacity * MAX_STONES * 2],
            stone_players: vec![0i8; capacity * MAX_STONES],
            n_stones: vec![0u16; capacity],
            visit_qr: vec![0i16; capacity * visit_capacity * 2],
            visit_probs: vec![0.0f32; capacity * visit_capacity],
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
