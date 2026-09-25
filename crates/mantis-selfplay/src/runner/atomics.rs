//! `WorkerAtomics` — per-worker control-flag bundle.
//!
//! One live tunable: `running` (the kill switch flipped by `stop()`). Cloned
//! once per worker spawn; destructured at `game::run_worker_thread` entry.

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(crate) struct WorkerAtomics {
    pub(crate) running: Arc<AtomicBool>,
    /// Runner-owned model-version snapshot source; each move dedup-pushes it into
    /// `version_seen`. Default 0 (no-NN) until the bridge's `set_model_version` sets it.
    pub(crate) model_version: Arc<AtomicU64>,
    /// Fatal-defect latch: the graph-record dispatch stores a `TargetIntegrityError`
    /// message here, counts the fire, then flips `running=false` (store-then-halt).
    pub(crate) fatal_defect: Arc<Mutex<Option<String>>>,
    pub(crate) target_integrity_defects: Arc<AtomicU64>,
    /// SEAM conjunct fire count: leaf inferences that FAILED on an open queue. Shares
    /// `fatal_defect`'s slot and store-then-halt ordering but keeps its OWN count, so the
    /// two conjuncts stay distinguishable in the event stream.
    pub(crate) inference_failures_total: Arc<AtomicU64>,
    /// The runner-wide monotonic GAME id, one `fetch_add` per completed graph game. Its own
    /// counter, not `games_completed`: that STAT may be reset, and an id derived from it collides.
    pub(crate) graph_game_seq: Arc<AtomicU64>,
}
