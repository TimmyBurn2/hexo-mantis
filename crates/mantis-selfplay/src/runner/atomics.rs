//! `WorkerAtomics` — per-worker control-flag bundle (WP6 D1), ported verbatim
//! from the frozen `worker_loop/atomics.rs`.
//!
//! Two live tunables: `running` (the kill switch flipped by `stop()`) and
//! `radius_override` (§174 curriculum — non-jitter, live). Cloned once per worker
//! spawn; destructured at `game::run_worker_thread` entry.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64};

#[derive(Clone)]
pub(crate) struct WorkerAtomics {
    pub(crate) running: Arc<AtomicBool>,
    pub(crate) radius_override: Arc<AtomicI32>,
    /// Runner-owned model-version snapshot source (frozen
    /// `batcher.current_model_version()`); each move dedup-pushes it into
    /// `version_seen`. Default 0 (no-NN) until WP7 wires the real setter.
    pub(crate) model_version: Arc<AtomicU64>,
}
