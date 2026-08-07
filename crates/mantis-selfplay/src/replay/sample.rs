//! Read path for `ReplayBuffer` — the owned-Rust sample cores plus the pure
//! scatter kernels that back them. Ported from the predecessor engine's
//! `replay_buffer/sample.rs` with the FFI-binding strip: the positional return
//! arrays become the owned `SampleBatch` / `SampleBatchWithPos` structs whose
//! FIELD ORDER is the versioned `SAMPLE_ORDER_V1` contract (O-35); the unsafe
//! `Vec<u16>→Vec<f16>` reinterpret + the array marshaling + reshape move to WP7.
//! The states/chain carriers stay `Vec<u16>` (raw f16 bits) — no f16→f32→f16
//! round-trip on the data path (O-34).
//!
//! R8: >300 LOC by design — the scatter kernels, the two sample cores, and the
//! O-35 field-order contract (consts + carrier-derived type tags) are one unit;
//! the pin is meaningless split from the structs it pins.

use std::collections::HashSet;

use half::f16;
use rand::RngExt;

use super::sym::{draw_record_sym, SymTables, N_CHAIN_PLANES, N_SYMS};
use super::ReplayBuffer;

// ── ⊕ O-35 tuple-order contract (versioned) ───────────────────────────────────
//
// The dense sample cores return owned structs; their FIELD ORDER is the frozen
// positional contract WP7 emits as the framework arrays (CAPTURE_LOG §A). The 9-form
// SPLICES position_indices at index 7 BEFORE the trailing value_target_valid —
// it is NOT the 8-form with a field appended.

/// Field order of `SampleBatch` (8-tuple), versioned v1.
pub const SAMPLE_ORDER_V1: [&str; 8] = [
    "states",
    "chain",
    "policies",
    "outcomes",
    "ownership",
    "winning_line",
    "is_full_search",
    "value_target_valid",
];

/// Field order of `SampleBatchWithPos` (9-tuple), versioned v1. `position_indices`
/// is spliced at index 7; `value_target_valid` stays last (index 8).
pub const SAMPLE_WITH_POS_ORDER_V1: [&str; 9] = [
    "states",
    "chain",
    "policies",
    "outcomes",
    "ownership",
    "winning_line",
    "is_full_search",
    "position_indices",
    "value_target_valid",
];

/// Carrier element-type tag. `U16` is BOTH the raw f16-bits carrier (states/chain)
/// and the genuine-u16 carrier (position_indices) after the FFI strip — the
/// float16-vs-uint16 framework DTYPE distinction is a WP7 fact (O-34 gates the f16
/// path), NOT representable in the WP5 carrier. The tag pins the WP5-owned carrier
/// type, the only thing that can silently drift here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeTag {
    U8,
    U16,
    F32,
}

/// Sealed per-carrier-element marker so `TypeTag::of` DERIVES the tag from a
/// field's concrete element type (not a hand-written literal): a carrier drift
/// `Vec<u8>`→`Vec<u16>` flips the derived tag and bites O-35.
pub trait CarrierElem {
    const TAG: TypeTag;
}
impl CarrierElem for u8 {
    const TAG: TypeTag = TypeTag::U8;
}
impl CarrierElem for u16 {
    const TAG: TypeTag = TypeTag::U16;
}
impl CarrierElem for f32 {
    const TAG: TypeTag = TypeTag::F32;
}

impl TypeTag {
    /// Derive the carrier tag from a field's concrete element type.
    #[must_use]
    pub fn of<T: CarrierElem>(_field: &[T]) -> TypeTag {
        T::TAG
    }
}

/// The 8-tuple sample-return core output (field order = `SAMPLE_ORDER_V1`).
pub struct SampleBatch {
    /// f16 bits, flat `[B * state_stride]`.
    pub states: Vec<u16>,
    /// f16 bits, flat `[B * chain_stride]`.
    pub chain: Vec<u16>,
    /// flat `[B * policy_stride]`.
    pub policies: Vec<f32>,
    /// `[B]`.
    pub outcomes: Vec<f32>,
    /// flat `[B * aux_stride]`.
    pub ownership: Vec<u8>,
    /// flat `[B * aux_stride]`.
    pub winning_line: Vec<u8>,
    /// `[B]`.
    pub is_full_search: Vec<u8>,
    /// `[B]`.
    pub value_target_valid: Vec<u8>,
    /// For WP7 reshape (shapes derived from `&spec` + `batch_size`).
    pub batch_size: usize,
}

/// The 9-tuple sample-return core output (field order = `SAMPLE_WITH_POS_ORDER_V1`).
pub struct SampleBatchWithPos {
    pub states: Vec<u16>,
    pub chain: Vec<u16>,
    pub policies: Vec<f32>,
    pub outcomes: Vec<f32>,
    pub ownership: Vec<u8>,
    pub winning_line: Vec<u8>,
    pub is_full_search: Vec<u8>,
    /// `[B]` — spliced at index 7 BEFORE `value_target_valid`.
    pub position_indices: Vec<u16>,
    /// `[B]` — index 8 (last).
    pub value_target_valid: Vec<u8>,
    pub batch_size: usize,
}

/// Emit `(name, carrier-derived TypeTag)` per field of `SampleBatch` in struct
/// order via an EXHAUSTIVE destructure — a field rename/removal/addition fails to
/// compile, and each tag is derived from the field's concrete element type.
#[must_use]
pub fn to_ordered_tags(out: &SampleBatch) -> [(&'static str, TypeTag); 8] {
    let SampleBatch {
        states,
        chain,
        policies,
        outcomes,
        ownership,
        winning_line,
        is_full_search,
        value_target_valid,
        batch_size: _,
    } = out;
    [
        ("states", TypeTag::of(states)),
        ("chain", TypeTag::of(chain)),
        ("policies", TypeTag::of(policies)),
        ("outcomes", TypeTag::of(outcomes)),
        ("ownership", TypeTag::of(ownership)),
        ("winning_line", TypeTag::of(winning_line)),
        ("is_full_search", TypeTag::of(is_full_search)),
        ("value_target_valid", TypeTag::of(value_target_valid)),
    ]
}

/// Exhaustive-destructure ordered tags for `SampleBatchWithPos` (9 fields).
#[must_use]
pub fn to_ordered_tags_with_pos(out: &SampleBatchWithPos) -> [(&'static str, TypeTag); 9] {
    let SampleBatchWithPos {
        states,
        chain,
        policies,
        outcomes,
        ownership,
        winning_line,
        is_full_search,
        position_indices,
        value_target_valid,
        batch_size: _,
    } = out;
    [
        ("states", TypeTag::of(states)),
        ("chain", TypeTag::of(chain)),
        ("policies", TypeTag::of(policies)),
        ("outcomes", TypeTag::of(outcomes)),
        ("ownership", TypeTag::of(ownership)),
        ("winning_line", TypeTag::of(winning_line)),
        ("is_full_search", TypeTag::of(is_full_search)),
        ("position_indices", TypeTag::of(position_indices)),
        ("value_target_valid", TypeTag::of(value_target_valid)),
    ]
}

// ── Pure scatter kernels ───────────────────────────────────────────────────────

/// Apply symmetry `sym_idx` to a state tensor (pure coord scatter).
///
/// Plane-count-generic: deduces `n_planes = src.len() / sym_tables.n_cells`.
/// Identical scatter is applied to every plane (state planes do not permute under
/// any hex dihedral symmetry — only cell coordinates do). Generic over `T: Copy`
/// — the internal buffer sampling path calls it with `u16` (f16 bits). Pure
/// scatter; caller zeroes `dst` before invocation.
#[inline]
pub fn apply_symmetry_state<T: Copy>(src: &[T], dst: &mut [T], sym_idx: usize, sym_tables: &SymTables) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(sym_idx < N_SYMS);
    let n_cells = sym_tables.n_cells;
    debug_assert_eq!(
        src.len() % n_cells,
        0,
        "state tensor length {} not a multiple of {} cells",
        src.len(),
        n_cells
    );
    let n_planes = src.len() / n_cells;

    let scatter = &sym_tables.scatter[sym_idx];
    for p in 0..n_planes {
        let base = p * n_cells;
        let src_plane = &src[base..base + n_cells];
        let dst_plane = &mut dst[base..base + n_cells];
        for &(sc, dc) in scatter {
            dst_plane[dc as usize] = src_plane[sc as usize];
        }
    }
}

/// Apply symmetry `sym_idx` to one 6-plane chain-length tensor (coord scatter +
/// axis-plane remap). Caller zeroes `dst` before invocation.
#[inline]
pub fn apply_chain_symmetry<T: Copy>(src: &[T], dst: &mut [T], sym_idx: usize, sym_tables: &SymTables) {
    let n_cells = sym_tables.n_cells;
    debug_assert_eq!(src.len(), N_CHAIN_PLANES * n_cells);
    debug_assert_eq!(dst.len(), N_CHAIN_PLANES * n_cells);
    debug_assert!(sym_idx < N_SYMS);

    let scatter = &sym_tables.scatter[sym_idx];
    let chain_src_lookup = &sym_tables.chain_src_lookup[sym_idx];

    // dst_p indexes the lookup AND drives dst_base arithmetic — kept verbatim.
    #[allow(clippy::needless_range_loop)]
    for dst_p in 0..N_CHAIN_PLANES {
        let src_p = chain_src_lookup[dst_p];
        let src_base = src_p * n_cells;
        let dst_base = dst_p * n_cells;
        let src_plane = &src[src_base..src_base + n_cells];
        let dst_plane = &mut dst[dst_base..dst_base + n_cells];
        for &(sc, dc) in scatter {
            dst_plane[dc as usize] = src_plane[sc as usize];
        }
    }
}

/// Source slice bundle for `ReplayBuffer::apply_sym` (read-only views).
pub struct ApplySymSrc<'a> {
    pub state: &'a [u16],
    pub chain: &'a [u16],
    pub policy: &'a [f32],
    pub own: &'a [u8],
    pub wl: &'a [u8],
}

/// Destination slice bundle for `ReplayBuffer::apply_sym` (mutable views).
pub struct ApplySymDst<'a> {
    pub state: &'a mut [u16],
    pub chain: &'a mut [u16],
    pub policy: &'a mut [f32],
    pub own: &'a mut [u8],
    pub wl: &'a mut [u8],
}

/// Combined slice + sym-tables bundle for `ReplayBuffer::apply_sym`.
pub struct ApplySymSlices<'a> {
    pub src: ApplySymSrc<'a>,
    pub dst: ApplySymDst<'a>,
    pub tables: &'a SymTables,
}

impl ReplayBuffer {
    /// Map an f16 weight (stored as bits) to a histogram bucket index.
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

    /// Sample a single index using rejection sampling on stored weights.
    #[inline]
    pub fn weighted_sample_one(&mut self) -> usize {
        const MAX_REJECT: usize = 32;
        for _ in 0..MAX_REJECT {
            let idx = self.rng.random_range(0..self.size);
            let w = f16::from_bits(self.weights[idx]).to_f32();
            if w >= 1.0 || self.rng.random::<f32>() < w {
                return idx;
            }
        }
        self.rng.random_range(0..self.size)
    }

    /// Sample `batch_size` slot indices, optionally deduplicating by game_id.
    pub(crate) fn sample_indices(&mut self, batch_size: usize, use_dedup: bool) -> Vec<usize> {
        if !use_dedup {
            return (0..batch_size).map(|_| self.weighted_sample_one()).collect();
        }

        const MAX_RETRIES: usize = 8;

        let mut indices: Vec<usize> = (0..batch_size).map(|_| self.weighted_sample_one()).collect();

        let mut seen: HashSet<i64> = HashSet::with_capacity(batch_size);
        for _ in 0..MAX_RETRIES {
            seen.clear();
            let mut all_unique = true;
            for idx in &mut indices {
                let gid = self.game_ids[*idx];
                if gid == -1 || seen.insert(gid) {
                    continue;
                }
                all_unique = false;
                let mut candidate = self.weighted_sample_one();
                for _ in 0..16 {
                    let cgid = self.game_ids[candidate];
                    if cgid == -1 || !seen.contains(&cgid) {
                        break;
                    }
                    candidate = self.weighted_sample_one();
                }
                *idx = candidate;
                let cgid = self.game_ids[candidate];
                if cgid != -1 {
                    seen.insert(cgid);
                }
            }
            if all_unique {
                break;
            }
        }

        indices
    }

    /// Apply symmetry `sym_idx` to one (state, chain, policy, ownership,
    /// winning_line) sample. Aux planes reuse the same scatter table as state.
    #[inline]
    pub fn apply_sym(sym_idx: usize, slices: ApplySymSlices<'_>) {
        let ApplySymSlices {
            src: ApplySymSrc { state: src_state, chain: src_chain, policy: src_policy, own: src_own, wl: src_wl },
            dst: ApplySymDst { state: dst_state, chain: dst_chain, policy: dst_policy, own: dst_own, wl: dst_wl },
            tables,
        } = slices;

        // State planes: pure coordinate scatter (identity plane mapping).
        apply_symmetry_state::<u16>(src_state, dst_state, sym_idx, tables);

        // Chain planes: coord-scatter + axis-plane remap.
        apply_chain_symmetry::<u16>(src_chain, dst_chain, sym_idx, tables);

        let scatter = &tables.scatter[sym_idx];

        // Policy + ownership + winning_line: same hex permutation table. Fuse.
        for &(sc, dc) in scatter {
            let sc_u = sc as usize;
            let dc_u = dc as usize;
            dst_policy[dc_u] = src_policy[sc_u];
            dst_own[dc_u] = src_own[sc_u];
            dst_wl[dc_u] = src_wl[sc_u];
        }
        // Pass action (index n_cells) is always the identity (policy only).
        // Skip for encodings without a pass slot (index n_cells would be one past
        // the end of the policy slice).
        if tables.n_cells < dst_policy.len() {
            dst_policy[tables.n_cells] = src_policy[tables.n_cells];
        }
    }

    /// Sample `batch_size` entries with optional hex-symmetry augmentation, drawn
    /// PER RECORD over the group that is lossless for that record (R245(c) — the
    /// dense scatter drops off-window cells, so a record carrying content there
    /// gets only `sym::WINDOW_PRESERVING_SYMS` while a window-fitting one gets all
    /// 12). Returns the owned `SampleBatch` (field order = `SAMPLE_ORDER_V1`).
    ///
    /// The out-buffers below are NEUTRAL-initialised (`out_ownership` 1-fill =
    /// empty, the rest 0-fill) and `apply_sym` writes only scatter pairs, so a
    /// destination cell no pair reaches keeps its neutral — which is exactly what
    /// the source cell held, for a compact record. No clipped copy is emitted.
    pub fn sample_batch_core(&mut self, batch_size: usize, augment: bool) -> Result<SampleBatch, String> {
        if self.size == 0 {
            return Err("Cannot sample from an empty replay buffer".to_string());
        }

        // Always run the dedup path: `sample_indices` treats the -1 untagged
        // sentinel as "skip this slot" per-sample, so mixed buffers are handled.
        let indices = self.sample_indices(batch_size, true);

        let state_stride = self.encoding.state_stride();
        let chain_stride = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride = self.encoding.aux_stride();

        // States and chain_planes as f16 bits (u16) — no type conversion during scatter.
        let mut out_states = vec![0u16; batch_size * state_stride];
        let mut out_chain = vec![0u16; batch_size * chain_stride];
        let mut out_policies = vec![0.0f32; batch_size * policy_stride];
        let mut out_outcomes = vec![0.0f32; batch_size];
        let mut out_ownership = vec![1u8; batch_size * aux_stride];
        let mut out_winning_line = vec![0u8; batch_size * aux_stride];
        let mut out_is_full_search = vec![0u8; batch_size];
        let mut out_value_valid = vec![0u8; batch_size];

        for (b, &idx) in indices.iter().enumerate() {
            let sym_idx = if augment { draw_record_sym(&mut self.rng, self.compact[idx] != 0) } else { 0 };

            let src_state = &self.states[idx * state_stride..(idx + 1) * state_stride];
            let src_chain = &self.chain_planes[idx * chain_stride..(idx + 1) * chain_stride];
            let src_policy = &self.policies[idx * policy_stride..(idx + 1) * policy_stride];
            let src_own = &self.ownership[idx * aux_stride..(idx + 1) * aux_stride];
            let src_wl = &self.winning_line[idx * aux_stride..(idx + 1) * aux_stride];

            let dst_state = &mut out_states[b * state_stride..(b + 1) * state_stride];
            let dst_chain = &mut out_chain[b * chain_stride..(b + 1) * chain_stride];
            let dst_policy = &mut out_policies[b * policy_stride..(b + 1) * policy_stride];
            let dst_own = &mut out_ownership[b * aux_stride..(b + 1) * aux_stride];
            let dst_wl = &mut out_winning_line[b * aux_stride..(b + 1) * aux_stride];

            Self::apply_sym(
                sym_idx,
                ApplySymSlices {
                    src: ApplySymSrc { state: src_state, chain: src_chain, policy: src_policy, own: src_own, wl: src_wl },
                    dst: ApplySymDst { state: dst_state, chain: dst_chain, policy: dst_policy, own: dst_own, wl: dst_wl },
                    tables: self.sym_tables,
                },
            );

            out_outcomes[b] = self.outcomes[idx];
            out_is_full_search[b] = self.is_full_search[idx];
            out_value_valid[b] = self.value_target_valid[idx];
        }

        Ok(SampleBatch {
            states: out_states,
            chain: out_chain,
            policies: out_policies,
            outcomes: out_outcomes,
            ownership: out_ownership,
            winning_line: out_winning_line,
            is_full_search: out_is_full_search,
            value_target_valid: out_value_valid,
            batch_size,
        })
    }

    /// Extended sampling with per-row `position_indices` for the ply-to-end aux
    /// head. Returns the owned `SampleBatchWithPos` (field order =
    /// `SAMPLE_WITH_POS_ORDER_V1`; position_indices SPLICED before value_target_valid).
    pub fn sample_batch_with_pos_core(
        &mut self,
        batch_size: usize,
        augment: bool,
    ) -> Result<SampleBatchWithPos, String> {
        if self.size == 0 {
            return Err("Cannot sample from an empty replay buffer".to_string());
        }

        let indices = self.sample_indices(batch_size, true);

        let state_stride = self.encoding.state_stride();
        let chain_stride = self.encoding.chain_stride();
        let policy_stride = self.encoding.policy_stride();
        let aux_stride = self.encoding.aux_stride();

        let mut out_states = vec![0u16; batch_size * state_stride];
        let mut out_chain = vec![0u16; batch_size * chain_stride];
        let mut out_policies = vec![0.0f32; batch_size * policy_stride];
        let mut out_outcomes = vec![0.0f32; batch_size];
        let mut out_ownership = vec![1u8; batch_size * aux_stride];
        let mut out_winning_line = vec![0u8; batch_size * aux_stride];
        let mut out_is_full_search = vec![0u8; batch_size];
        let mut out_position_indices = vec![0u16; batch_size];
        let mut out_value_valid = vec![0u8; batch_size];

        for (b, &idx) in indices.iter().enumerate() {
            let sym_idx = if augment { draw_record_sym(&mut self.rng, self.compact[idx] != 0) } else { 0 };

            let src_state = &self.states[idx * state_stride..(idx + 1) * state_stride];
            let src_chain = &self.chain_planes[idx * chain_stride..(idx + 1) * chain_stride];
            let src_policy = &self.policies[idx * policy_stride..(idx + 1) * policy_stride];
            let src_own = &self.ownership[idx * aux_stride..(idx + 1) * aux_stride];
            let src_wl = &self.winning_line[idx * aux_stride..(idx + 1) * aux_stride];

            let dst_state = &mut out_states[b * state_stride..(b + 1) * state_stride];
            let dst_chain = &mut out_chain[b * chain_stride..(b + 1) * chain_stride];
            let dst_policy = &mut out_policies[b * policy_stride..(b + 1) * policy_stride];
            let dst_own = &mut out_ownership[b * aux_stride..(b + 1) * aux_stride];
            let dst_wl = &mut out_winning_line[b * aux_stride..(b + 1) * aux_stride];

            Self::apply_sym(
                sym_idx,
                ApplySymSlices {
                    src: ApplySymSrc { state: src_state, chain: src_chain, policy: src_policy, own: src_own, wl: src_wl },
                    dst: ApplySymDst { state: dst_state, chain: dst_chain, policy: dst_policy, own: dst_own, wl: dst_wl },
                    tables: self.sym_tables,
                },
            );

            out_outcomes[b] = self.outcomes[idx];
            out_is_full_search[b] = self.is_full_search[idx];
            out_position_indices[b] = self.position_indices[idx];
            out_value_valid[b] = self.value_target_valid[idx];
        }

        Ok(SampleBatchWithPos {
            states: out_states,
            chain: out_chain,
            policies: out_policies,
            outcomes: out_outcomes,
            ownership: out_ownership,
            winning_line: out_winning_line,
            is_full_search: out_is_full_search,
            position_indices: out_position_indices,
            value_target_valid: out_value_valid,
            batch_size,
        })
    }
}
