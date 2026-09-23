//! mantis-core: board, hex geometry, rules, the Ply vocabulary type.
//!
//! Zero in-workspace dependencies; no Python bindings (the bridge crate owns
//! the FFI surface). Spec/registry resolution never happens here — callers
//! pass plain geometry values (`BoardGeometry`).

pub mod board;
pub mod ply;

pub use board::{Board, BoardGeometry, Cell, MoveDiff, Player};
pub use ply::Ply;
