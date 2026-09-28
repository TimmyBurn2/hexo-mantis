//! `tactics` — the exact, net-free tactics module: one turn's facts (`analyze`) and a turn-level strictly forcing
//! threat-space solver (`TurnSolver`, Six's kind), used identically at deploy and in self-play.

pub mod analyze;
pub(crate) mod gen;
pub(crate) mod grid;
pub mod solver;
pub(crate) mod table;

pub use analyze::{analyze, LeafTactics, Terminal};
pub(crate) use grid::{position_key, with_stone};
pub use solver::{Solved, Turn, TurnSolver, Verdict};
