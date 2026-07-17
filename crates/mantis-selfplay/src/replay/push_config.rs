//! Internal config structs for the `ReplayBuffer` push impl methods.
//!
//! Ported from the predecessor engine's `replay_buffer/push_config.rs` with the
//! FFI-binding strip: the borrowed array views become plain borrowed slices and
//! the binding lifetime becomes a data lifetime `'a`. The `.as_slice()?`
//! contiguity check moves to WP7 (the bridge marshals the framework views into
//! these slices). Three structs (not one uniform `PushParams`) because the 3 impls have
//! different array ranks and scalar-vs-array metadata shapes.

use half::f16;

/// Config for `push_impl` — single position with scalar metadata.
pub struct PushSingleConfig<'a> {
    pub state: &'a [f16],
    pub chain_planes: &'a [f16],
    pub policy: &'a [f32],
    pub outcome: f32,
    pub ownership: &'a [u8],
    pub winning_line: &'a [u8],
    pub game_id: i64,
    pub game_length: u16,
    pub is_full_search: bool,
    /// 0-based ply index within game.
    pub position_index: u16,
    /// 1 = supervise value head, 0 = ply-capped → masked.
    pub value_target_valid: bool,
}

/// Config for `push_game_impl` — batched per-game with shared scalar metadata
/// and optional per-row `is_full_search` slice.
pub struct PushGameConfig<'a> {
    pub states: &'a [f16],
    pub chain_planes: &'a [f16],
    pub policies: &'a [f32],
    pub outcomes: &'a [f32],
    pub ownership: &'a [u8],
    pub winning_line: &'a [u8],
    pub game_id: i64,
    pub game_length: u16,
    pub is_full_search: Option<&'a [u8]>,
    /// Per-row 0-based ply index. None ⇒ fills 0.
    pub position_indices: Option<&'a [u16]>,
    /// Per-row value-supervision flag. None ⇒ all-ones (supervise).
    pub value_target_valid: Option<&'a [u8]>,
}

/// Config for `push_many_impl` — batched per-row with all metadata as slices;
/// rows are tagged `game_id = -1`.
pub struct PushManyConfig<'a> {
    pub states: &'a [f16],
    pub chain_planes: &'a [f16],
    pub policies: &'a [f32],
    pub outcomes: &'a [f32],
    pub ownership: &'a [u8],
    pub winning_line: &'a [u8],
    pub game_lengths: &'a [u16],
    pub is_full_search: &'a [u8],
    /// Per-row 0-based ply index. None ⇒ fills zeros.
    pub position_indices: Option<&'a [u16]>,
    /// Per-row value-supervision flag. None ⇒ all-ones (supervise).
    pub value_target_valid: Option<&'a [u8]>,
}
