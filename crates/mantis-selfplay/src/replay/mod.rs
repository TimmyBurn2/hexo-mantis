//! ReplayBuffer — pre-allocated ring buffer with vectorized 12-fold hex
//! augmentation. Ported from the predecessor engine's `replay_buffer/` with the
//! FFI-binding STRIP (R6/LAW-17: this crate compiles with ZERO Python/array
//! bindings). The sample-return positional arrays become owned-Rust structs whose
//! field order is the versioned `SAMPLE_ORDER_V1` contract (O-35); array
//! marshaling + the f16-bits reinterpret land in the bridge (WP7).
//!
//! ## Module layout
//!   mod.rs        — `ReplayBuffer` struct + Rust facade (thin delegates)
//!   storage.rs    — resize, dashboard stats, weight schedule, monotonic id
//!   push.rs       — single-position `push_impl`, batched `push_game`/`push_many`
//!   sample.rs     — `sample_batch_core` + weighted-sample + 12-fold apply_sym
//!   persist/      — HEXB v9 save / load (on-disk format)
//!   sym.rs        — 12-fold D6 permutation tables + geometry constants
//!   schedule.rs   — game-length weight schedule
//!   hexg/         — parallel graph-position ring (HEXG v1)
//!
//! ## Memory layout (flat, row-major; strides from the encoding spec)
//!   states       : Vec<u16> — f16 bits, [capacity × spec.state_stride()]
//!   chain_planes : Vec<u16> — f16 bits, [capacity × spec.chain_stride()]
//!   policies     : Vec<f32> — [capacity × spec.policy_stride()]
//!   outcomes     : Vec<f32> — [capacity]
//!   game_ids     : Vec<i64> — [capacity]; -1 = untagged
//!   weights      : Vec<u16> — f16 bits; one sample weight per position
//!   ownership    : Vec<u8>  — [capacity × spec.aux_stride()] (0=P2, 1=empty, 2=P1)
//!   winning_line : Vec<u8>  — [capacity × spec.aux_stride()] binary mask

mod storage;
mod push;
pub mod push_config;
pub mod sample;
mod persist;
pub mod hexg;
pub mod sym;
pub mod schedule;

use half::f16;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::atomic::AtomicU64;

use mantis_encoding::RegistrySpec;
use sym::{sym_tables_for, SymTables};
use schedule::WeightSchedule;

pub use sample::{SampleBatch, SampleBatchWithPos};
pub use hexg::{GraphRecord, GraphTargets, HexgBuffer};

// ── ReplayBuffer ───────────────────────────────────────────────────────────────

/// Ring-buffer replay buffer with 12-fold hexagonal augmentation.
///
/// Construction pre-allocates all storage. No heap allocation happens after
/// `new`. Fields are `pub` so the relocated integration oracle suites
/// (`tests/replay_*`) and the sample bench can populate/inspect ring state — the
/// old in-src `pub(crate)` surface is exposed as `pub` now that the tests live
/// under the single `tests/` root (R5).
pub struct ReplayBuffer {
    pub capacity: usize,
    pub size: usize,
    pub head: usize, // next write slot

    /// Encoding spec — drives all stride / cell-count calculations.
    pub encoding: &'static RegistrySpec,

    pub states: Vec<u16>,       // f16 bits; flat [capacity × spec.state_stride()]
    pub chain_planes: Vec<u16>, // f16 bits; flat [capacity × spec.chain_stride()]
    pub policies: Vec<f32>,     // flat [capacity × spec.policy_stride()]
    pub outcomes: Vec<f32>,     // flat [capacity]
    pub game_ids: Vec<i64>,     // flat [capacity]; -1 = untagged
    pub weights: Vec<u16>,      // f16-as-u16 bits; flat [capacity]

    /// Per-cell ownership of the final board (0=P2, 1=empty, 2=P1).
    pub ownership: Vec<u8>,
    /// Binary mask of the winning 6-in-a-row; all zero on draw.
    pub winning_line: Vec<u8>,

    /// Move-level playout cap flag: 1 = full-search, 0 = quick-search. [capacity].
    pub is_full_search: Vec<u8>,

    /// Per-row value-supervision flag. 1 = supervise, 0 = ply-capped → masked.
    pub value_target_valid: Vec<u8>,

    /// Per-position 0-based ply index within its game. [capacity].
    pub position_indices: Vec<u16>,

    /// Static sym tables for this buffer's encoding.
    pub sym_tables: &'static SymTables,
    pub weight_schedule: WeightSchedule,
    pub next_game_id: i64,
    pub rng: StdRng,

    /// Lock-free weight histogram for O(1) dashboard stats.
    pub weight_buckets: [AtomicU64; 3],
}

impl ReplayBuffer {
    /// Create a new buffer with the given `capacity` (number of positions).
    ///
    /// `encoding` — encoding name from the registry.
    #[must_use]
    pub fn new(capacity: usize, encoding: &str) -> Self {
        let spec = mantis_encoding::registry::lookup_or_panic(encoding);
        Self::build(capacity, spec, StdRng::from_rng(&mut rand::rng()))
    }

    /// Shared constructor body — `new` (registry lookup) and the test-only
    /// `with_encoding` both funnel through here so the field init stays in one
    /// place (R1: the spec is always explicit, no code-side identity default).
    fn build(capacity: usize, spec: &'static RegistrySpec, rng: StdRng) -> Self {
        let default_w = f16::from_f32(1.0).to_bits();
        ReplayBuffer {
            capacity,
            size: 0,
            head: 0,
            encoding: spec,
            states: vec![0u16; capacity * spec.state_stride()],
            chain_planes: vec![0u16; capacity * spec.chain_stride()],
            policies: vec![0.0f32; capacity * spec.policy_stride()],
            outcomes: vec![0.0f32; capacity],
            game_ids: vec![-1i64; capacity],
            weights: vec![default_w; capacity],
            ownership: vec![1u8; capacity * spec.aux_stride()], // 1 = empty default
            winning_line: vec![0u8; capacity * spec.aux_stride()],
            is_full_search: vec![1u8; capacity], // 1 = full-search default
            value_target_valid: vec![1u8; capacity], // 1 = supervise value default
            position_indices: vec![0u16; capacity],
            sym_tables: sym_tables_for(spec), // unread on the graph path (see spawn.rs R28 note)
            weight_schedule: WeightSchedule::uniform(),
            next_game_id: 0,
            rng,
            weight_buckets: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        }
    }

    /// TEST-ONLY constructor injecting an EXPLICIT `&'static RegistrySpec` into
    /// the buffer's `encoding` field — the O-9 P13 accept-on-name-mismatch
    /// witness (C-7). Gated `#[cfg(test)]` so it is NEVER a production surface;
    /// the aliased spec it receives is a leaked test-only local, never registered
    /// (`all_specs()` stays the sole production name source). R1-safe: strictly
    /// more explicit than `new(cap, name)` — no identity default introduced.
    #[cfg(test)]
    pub(crate) fn with_encoding(capacity: usize, spec: &'static RegistrySpec) -> Self {
        Self::build(capacity, spec, StdRng::from_rng(&mut rand::rng()))
    }

    /// Return the encoding name driving this buffer's geometry.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── Plain-Rust helpers (bench + oracle suites) ─────────────────────────────────

impl ReplayBuffer {
    /// Push a zero-filled position with the given outcome, game_length, and
    /// `is_full_search` flag (game_id = -1, position_index = 0). The pub helper
    /// the round-trip / aux oracle suites use to populate the buffer with known
    /// scalar values (the old `#[cfg(test)]` push_raw is subsumed by this — the
    /// relocated integration tests cannot reach a cfg-gated helper, R5).
    pub fn push_for_test(&mut self, outcome: f32, game_length: u16, is_full_search: bool) {
        use std::sync::atomic::Ordering;
        let state_stride = self.encoding.state_stride();
        let chain_stride = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride = self.encoding.aux_stride();
        let slot = self.head;
        if self.size == self.capacity {
            let old_bucket = Self::weight_bucket(self.weights[slot]);
            self.weight_buckets[old_bucket].fetch_sub(1, Ordering::Relaxed);
        }
        let s = slot * state_stride;
        self.states[s..s + state_stride].fill(0);
        let c = slot * chain_stride;
        self.chain_planes[c..c + chain_stride].fill(0);
        let p = slot * policy_stride;
        self.policies[p..p + policy_stride].fill(0.0);
        let a = slot * aux_stride;
        self.ownership[a..a + aux_stride].fill(1);
        self.winning_line[a..a + aux_stride].fill(0);
        self.is_full_search[slot] = is_full_search as u8;
        self.position_indices[slot] = 0;
        self.outcomes[slot] = outcome;
        self.game_ids[slot] = -1;
        self.weights[slot] = if game_length == 0 {
            f16::from_f32(1.0).to_bits()
        } else {
            self.weight_schedule.weight_for(game_length)
        };
        let new_bucket = Self::weight_bucket(self.weights[slot]);
        self.weight_buckets[new_bucket].fetch_add(1, Ordering::Relaxed);
        self.head = (self.head + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
    }

    /// Return the raw `is_full_search` byte at the given buffer slot.
    #[must_use]
    pub fn is_full_search_at(&self, slot: usize) -> u8 {
        self.is_full_search[slot]
    }

    /// Return the raw `value_target_valid` byte at the given buffer slot.
    #[must_use]
    pub fn value_target_valid_at(&self, slot: usize) -> u8 {
        self.value_target_valid[slot]
    }

    /// Return the sampling weight at the given buffer slot as f32.
    #[must_use]
    pub fn weight_at_f32(&self, slot: usize) -> f32 {
        f16::from_bits(self.weights[slot]).to_f32()
    }
}
