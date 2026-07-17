//! mantis-core: board, hex geometry, rules, Ply/Turn vocabulary types.
//!
//! Zero in-workspace dependencies; no Python bindings (the bridge crate owns
//! the FFI surface). Spec/registry resolution never happens here — callers
//! pass plain geometry values (`BoardGeometry`).

pub mod board;
pub mod ply;

pub use board::{Board, BoardGeometry, Cell, MoveDiff, Player};
pub use ply::{Ply, Turn};

pub const CRATE_NAME: &str = "mantis-core";

#[cfg(test)]
mod tests {
    #[test]
    fn crate_name_pinned() {
        assert_eq!(super::CRATE_NAME, "mantis-core");
    }
}
