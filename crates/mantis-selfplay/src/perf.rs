//! DIAGNOSTIC stage timers for the self-play search drive (PERF-BASELINE, 2026-08-29).
//!
//! OFF unless `MANTIS_PERF_STAGES=1` is in the environment when the first timer is
//! reached. When off, every call site pays one relaxed load of a `OnceLock<bool>` and a
//! predicted-not-taken branch; no `Instant::now()` is issued and no atomic is written.
//!
//! This exists to answer §2 B/F of the PERF-BASELINE dispatch — how much of a simulation
//! is search bookkeeping and how much is the inference round trip. It measures; it
//! changes no behaviour.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Stage index into `ACC`. Kept as a plain `usize` const set rather than an enum so the
/// array and the name table cannot drift apart without the compiler noticing.
pub const SELECT_LEAVES: usize = 0;
pub const GRAPH_BUILD: usize = 1;
pub const SUBMIT_WAIT: usize = 2;
pub const EXPAND_BACKUP: usize = 3;
pub const N_STAGES: usize = 4;

pub const STAGE_NAMES: [&str; N_STAGES] = [
    "select_leaves",
    "graph_build",
    "submit_wait",
    "expand_backup",
];

/// Power-of-two buckets for the per-call leaf count, `1` … `2048`.
const N_BUCKETS: usize = 12;

struct StageAcc {
    count: AtomicU64,
    total_ns: AtomicU64,
    max_ns: AtomicU64,
}

impl StageAcc {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            max_ns: AtomicU64::new(0),
        }
    }
}

static ACC: [StageAcc; N_STAGES] = [
    StageAcc::new(),
    StageAcc::new(),
    StageAcc::new(),
    StageAcc::new(),
];

static LEAVES_TOTAL: AtomicU64 = AtomicU64::new(0);
static CALLS: AtomicU64 = AtomicU64::new(0);
#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static LEAF_HIST: [AtomicU64; N_BUCKETS] = [ZERO; N_BUCKETS];

static ENABLED: OnceLock<bool> = OnceLock::new();

/// `true` when `MANTIS_PERF_STAGES=1` was set at process start.
///
/// Read once and cached: a per-call `env::var` on the hot path would itself be the
/// measurement's largest term.
#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("MANTIS_PERF_STAGES").is_ok_and(|v| v == "1"))
}

/// One open timing span. `None` when timing is off, so the whole span costs a branch.
pub struct Span {
    start: Option<Instant>,
    stage: usize,
}

impl Span {
    #[inline]
    #[must_use]
    pub fn start(stage: usize) -> Self {
        Self {
            start: if enabled() {
                Some(Instant::now())
            } else {
                None
            },
            stage,
        }
    }

    /// Close the span and fold its elapsed nanoseconds into the stage accumulator.
    #[inline]
    pub fn stop(self) {
        let Some(t0) = self.start else { return };
        let ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let acc = &ACC[self.stage];
        acc.count.fetch_add(1, Ordering::Relaxed);
        acc.total_ns.fetch_add(ns, Ordering::Relaxed);
        acc.max_ns.fetch_max(ns, Ordering::Relaxed);
    }
}

/// Record one `infer_and_expand_graph` call that carried `n` leaves.
#[inline]
pub fn record_leaf_batch(n: usize) {
    if !enabled() {
        return;
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    LEAVES_TOTAL.fetch_add(n as u64, Ordering::Relaxed);
    let bucket = if n == 0 {
        0
    } else {
        (usize::BITS - n.leading_zeros() - 1) as usize
    };
    LEAF_HIST[bucket.min(N_BUCKETS - 1)].fetch_add(1, Ordering::Relaxed);
}

/// `(stage_name, count, total_ns, max_ns)` for every stage, plus the call/leaf totals.
#[must_use]
pub fn snapshot() -> (
    Vec<(&'static str, u64, u64, u64)>,
    u64,
    u64,
    Vec<(u64, u64)>,
) {
    let stages = (0..N_STAGES)
        .map(|i| {
            (
                STAGE_NAMES[i],
                ACC[i].count.load(Ordering::Relaxed),
                ACC[i].total_ns.load(Ordering::Relaxed),
                ACC[i].max_ns.load(Ordering::Relaxed),
            )
        })
        .collect();
    let hist = (0..N_BUCKETS)
        .map(|b| (1u64 << b, LEAF_HIST[b].load(Ordering::Relaxed)))
        .filter(|(_, c)| *c > 0)
        .collect();
    (
        stages,
        CALLS.load(Ordering::Relaxed),
        LEAVES_TOTAL.load(Ordering::Relaxed),
        hist,
    )
}

/// Zero every accumulator. Used to discard a warm-up window before the measured one.
pub fn reset() {
    for acc in &ACC {
        acc.count.store(0, Ordering::Relaxed);
        acc.total_ns.store(0, Ordering::Relaxed);
        acc.max_ns.store(0, Ordering::Relaxed);
    }
    CALLS.store(0, Ordering::Relaxed);
    LEAVES_TOTAL.store(0, Ordering::Relaxed);
    for b in &LEAF_HIST {
        b.store(0, Ordering::Relaxed);
    }
}
