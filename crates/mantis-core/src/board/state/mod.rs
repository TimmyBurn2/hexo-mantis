//! Sparse axial hex board — state module.
//!
//! `core` holds the types (Board, BoardGeometry, MoveDiff, Player, Cell), consts, ctors,
//! mutators, Clone, apply/undo and the window-coord helpers.

mod core;

pub use self::core::{
    hex_distance, Board, BoardGeometry, Cell, MoveDiff, Player, BOARD_SIZE, HALF, HEX_AXES,
    TOTAL_CELLS,
};
