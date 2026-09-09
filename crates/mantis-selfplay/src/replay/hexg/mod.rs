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

/// Structural ceiling on the record COUNT a buffer may be asked for (AUDIT-1 F-38).
///
/// DERIVED, never tuned: the widest per-record allocation this buffer makes is
/// `capacity * MAX_STONES * 2` (`stones_qr`), and this is the largest capacity for which that
/// product cannot overflow `usize` — so the arithmetic is safe before any allocator is asked.
/// It is enormously larger than any real ring (run5's is thousands), which is the point: the
/// bound exists to stop a wrap, not to express a policy about buffer sizes.
pub const HEXG_CAPACITY_CEILING: usize = usize::MAX / (MAX_STONES * 2);

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
/// `capacity = max(ARMED effective sim budgets) + leaf_batch_size − 1`, the largest
/// positive-mass support a graph record can carry. Armed arms: standard (always;
/// effective = `standard_sims` else `n_simulations`), fast iff `fast_prob > 0`,
/// quick/full iff `full_search_prob > 0`.
///
/// THE `− 1` TERM IS NOW HEADROOM, NOT A BOUND (R335(c), 2026-09-04). It was derived
/// from the sim loops overshooting by up to `leaf_batch_size − 1` on an uncapped final
/// batch; `search_drive::run_mcts_search` now clamps that batch, so a PUCT search backs
/// up exactly `max_armed` visits and the Gumbel arm backs up fewer. The FORMULA IS
/// DELIBERATELY UNCHANGED: it is a mint-time validator, so tightening it changes which
/// configs mint, which is a ruling's call and not a perf leg's.
///
/// Called by BOTH enforcement surfaces — the mint-time schema validator
/// (through the bridge twin `derived_hexg_visit_capacity`) and the
/// `SelfPlayRunner` boot guard — so the two cannot drift onto second formulas.
///
/// SCOPE (R275(a)): this formula is derived from the CURRENT visit-limited target
/// construction, and so are the two F-816-9 pins that sit downstream of it
/// (`records::refuse_zero_visit_export` and `search_drive::InferenceSeamFailure`).
///
/// # Errors
/// * the derived capacity exceeds [`HEXG_VISIT_COUNT_CEILING`] — no slot sizing
///   can honor the regime; the schema twin makes this a MINT-time error, and
///   the boot-side call is defense-in-depth for un-minted constructions;
/// * `search_kind` is not a kind this build knows;
/// * `search.kind: gumbel` on the graph path, at ANY capacity — that kind's root reaches
///   the full legal set, so the exported target's support IS the legal set, which no sims
///   regime bounds.
///
/// THE CHECK IS A DENSITY CHECK, NOT A VISIT CHECK. The sims regime bounds how many visits
/// a row RECORDS; it says nothing about how many cells the exported distribution puts mass
/// on. Under `puct` the exported target is the visit distribution and the two coincide;
/// under `gumbel` they do not, and the second is unbounded by the config.
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
    search_kind: &str,
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
    // The bound is the exported target's SUPPORT, and which support that is depends on the
    // search kind. The kind is parsed here rather than passed as a derived boolean so the
    // two enforcement surfaces cannot drift onto second formulas — the same reason this
    // whole function has two callers and no second copy.
    let kind = mantis_search::SearchKind::from_config_str(search_kind).ok_or_else(|| {
        format!(
            "search.kind = {search_kind:?} is not a known search kind \
             (expected \"puct\" or \"gumbel\")"
        )
    })?;
    if kind == mantis_search::SearchKind::Gumbel {
        return Err(format!(
            "representation==graph with search.kind=gumbel is refused at ANY derived \
             capacity ({capacity} here): that kind's root reaches its FULL legal set and \
             exports the completed-Q improved policy over it, so the target's support is \
             the legal set itself, which the config bounds nowhere. THE LEGAL SET IS NOT A \
             CONSTANT — it is the union of radius-r balls around every stone minus the \
             occupied cells and GROWS with the stone count, measured at radius 8 to a \
             MEDIAN of 355 and a MAXIMUM of 8142 — so no sims regime can derive a slot \
             count that covers it. A run wanting this kind on the graph path needs a MINTED \
             visit-slot bound, not a derived one, and the ring cost is its subject: at 8 \
             bytes a slot, 8192 slots is ~65 KB per row against today's ~252 B \
             (keys: search.kind, identity.representation)"
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
    /// R345(b)(6) — WHICH GAME this position came from, stamped once per game at
    /// `finalize_game_graph`. `-1` is the untagged sentinel and means the record was built
    /// outside a game (a test, a fixture); a production self-play record always carries a real
    /// id. Before this field every self-play row was pushed with `-1`, so `sample_indices`'s
    /// same-game dedupe — which skips the guard on `-1` — had never once fired on real data,
    /// and a batch could be a dozen positions from one game reported as a dozen samples.
    pub game_id: i64,
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
    //: R345(b)(6) — the LAST sampled batch's composition, written by
    //: `record_batch_composition` and read by the per-batch line the trainer logs. Kept on the
    //: buffer rather than returned from `sample_graph_batch` so the hot path keeps its
    //: signature; a reader asks after the batch it cares about.
    pub last_batch_distinct_games: u32,
    pub last_batch_max_rows_per_game: u32,
    pub last_batch_untagged_rows: u32,
    /// Rows back from the newest, at p50 / p90 / p99.
    pub last_batch_age_quantiles: [u32; 3],
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
        // AUDIT-1 F-38. `lookup_or_panic` ran BEFORE this function's own `Result` checks, so
        // an unknown encoding name reached Python as a `PanicException` while every other
        // refusal here is a named `ValueError` — and `SelfPlayRunner::new` and
        // `PyRegistrySpec::from_registry` both already return the sorted known list.
        let spec = mantis_encoding::registry::lookup(encoding).ok_or_else(|| {
            let mut known: Vec<&str> = mantis_encoding::registry::all_specs()
                .map(|s| s.name)
                .collect();
            known.sort_unstable();
            format!("HexgBuffer: unknown encoding {encoding:?}; registered: {known:?}")
        })?;
        if !spec.is_graph() {
            return Err(format!(
                "HexgBuffer requires a graph encoding; '{encoding}' is representation=grid \
                 (use ReplayBuffer for dense encodings)"
            ));
        }
        // AUDIT-1 F-38. `capacity` was UNBOUNDED at the FFI. Zero panics on the first push
        // (an index out of bounds and a `% 0`), and a huge value wraps the slot-geometry
        // product in release or aborts inside `handle_alloc_error` — the one exit
        // `panic = "unwind"` cannot convert into a Python exception, so it takes the process
        // with it. Both are refused here, by name, before anything is allocated.
        if capacity == 0 {
            return Err(
                "HexgBuffer: capacity 0 stores nothing and panics on the first push \
                        (an index out of bounds and a modulo by zero)"
                    .to_string(),
            );
        }
        if capacity > HEXG_CAPACITY_CEILING {
            return Err(format!(
                "HexgBuffer: capacity {capacity} exceeds the ceiling {HEXG_CAPACITY_CEILING}. \
                 The slot geometry multiplies capacity by MAX_STONES and the per-record \
                 strides; beyond this the product overflows in a release build (no \
                 overflow-checks) or aborts in the allocator, which `panic = \"unwind\"` \
                 cannot convert into a Python exception"
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
            last_batch_distinct_games: 0,
            last_batch_max_rows_per_game: 0,
            last_batch_untagged_rows: 0,
            last_batch_age_quantiles: [0, 0, 0],
            rng: StdRng::from_rng(&mut rand::rng()),
            weight_buckets: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        })
    }

    /// Re-seed the SAMPLER from a caller-supplied seed, replacing the OS-entropy stream
    /// `new` installs.
    ///
    /// `new` seeds from `rand::rng()` because a ring with no declared seed must not pretend
    /// to a reproducible stream. That left production with no way to declare one: two
    /// launches of the same config drew different batch sequences, and no Python-side
    /// `seed_everything` could reach this field. This is that declaration — one method, the
    /// same `StdRng` type, no new dependency, and no per-sample cost (R344(a); the
    /// alternative it was chosen over is costed in `CARD-RING-SAMPLER-SEED`).
    ///
    /// It does NOT make a resumed run continue the pre-stop stream — capturing ChaCha word
    /// position needs rand's private backend, which the crate's `rand` pin exists to keep
    /// this crate away from. What it buys is run-to-run reproducibility from a fixed seed.
    pub fn seed_sampler(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// Fresh monotonic game id.
    /// The `game_id` of the `index`-th record in insertion order (oldest first), or `None`
    /// when `index` is past `size`. R345(b)(6)'s read half.
    pub fn game_id_at(&self, index: usize) -> Option<i64> {
        if index >= self.size {
            return None;
        }
        let slot = (self.head + self.capacity - self.size + index) % self.capacity;
        Some(self.game_ids[slot])
    }

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
