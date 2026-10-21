//! mantis-player: the browser bot's deploy driver (contract #13), from the net's batch to the per-stone step machine.

pub mod batch;
pub mod budget;
pub mod cache;
pub mod forward;
pub mod player;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
