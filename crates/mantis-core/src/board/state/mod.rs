//! Sparse axial hex board — state module.
//!
//! Split into two sub-files:
//! - `core` — types (Board, BoardGeometry, MoveDiff, Player, Cell), consts,
//!   ctors, mutators, Clone, apply/undo, window-coord helpers.
//! - `cluster` — cluster-aware view assembly (`get_cluster_views`,
//!   `get_threat_anchors` + private threat helpers).
//!
//! (The predecessor's tensor-encoder sub-file lives in the encoding crate,
//! not here — this crate carries no encode kernels.)

mod cluster;
mod core;

pub use self::core::{
    Board, BoardGeometry, Cell, MoveDiff, Player,
    BOARD_SIZE, HALF, HEX_AXES, TOTAL_CELLS,
    hex_distance,
};
