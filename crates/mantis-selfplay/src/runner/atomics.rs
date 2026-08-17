//! `WorkerAtomics` — per-worker control-flag bundle (WP6 D1), ported verbatim
//! from the frozen `worker_loop/atomics.rs`.
//!
//! One live tunable: `running` (the kill switch flipped by `stop()`). Cloned
//! once per worker spawn; destructured at `game::run_worker_thread` entry.

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(crate) struct WorkerAtomics {
    pub(crate) running: Arc<AtomicBool>,
    /// Runner-owned model-version snapshot source (frozen
    /// `batcher.current_model_version()`); each move dedup-pushes it into
    /// `version_seen`. Default 0 (no-NN) until WP7 wires the real setter.
    pub(crate) model_version: Arc<AtomicU64>,
    /// WP12-R Phase T fatal-defect latch (DESIGN_T §3.4): the graph-record
    /// dispatch stores a `TargetIntegrityError` message here, counts the fire,
    /// then flips `running=false` (store-then-halt; LAW-14).
    pub(crate) fatal_defect: Arc<Mutex<Option<String>>>,
    pub(crate) target_integrity_defects: Arc<AtomicU64>,
    /// R275(b) SEAM conjunct fire count: leaf inferences that FAILED on an open
    /// queue (LAW-18). Shares `fatal_defect`'s slot and store-then-halt ordering,
    /// keeps its OWN count so the two conjuncts of the F-816-9 class stay
    /// distinguishable in the event stream.
    pub(crate) inference_failures_total: Arc<AtomicU64>,
}
