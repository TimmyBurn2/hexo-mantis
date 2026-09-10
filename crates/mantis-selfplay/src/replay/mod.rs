//! Replay rings. The HEXG (graph) ring is [`hexg`]; the HEXB dense ring went with the grid
//! path (R346(f)). [`schedule`] and [`atomic`] are the pieces both rings shared and the
//! graph ring still uses.

pub mod atomic;
pub mod hexg;
pub mod schedule;
pub mod sym;
