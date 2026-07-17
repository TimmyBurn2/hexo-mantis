//! Dense encode kernels — FREE FUNCTIONS over mantis-core's public surface.
//!
//! The kernels port from the old `impl Board` methods with zero behavior change:
//! the spec is threaded as a `&RegistrySpec` parameter where the old method read
//! `self.encoding`, and `self.ply % 2` becomes `board.ply.index() % 2`. The
//! encoding-layout plane-index constants live here (they are encoding facts, not
//! board facts — deliberately not in mantis-core).

mod chain;
mod state;

/// Source-plane index of the current player's t0 stones.
pub const MY_STONE_PLANE: usize = 0;
/// Source-plane index of the opponent's t0 stones.
pub const OPP_STONE_PLANE: usize = 8;
/// Source-plane index of the `moves_remaining == 2` broadcast scalar.
pub const MOVES_REMAINING_PLANE: usize = 16;
/// Source-plane index of the ply-parity (`ply % 2`) broadcast scalar.
pub const PLY_PARITY_PLANE: usize = 17;

pub use chain::encode_chain_planes;
pub use state::{
    encode_planes_to_buffer, encode_state_to_buffer, encode_state_to_buffer_channels, to_planes,
    to_planes_channels,
};
