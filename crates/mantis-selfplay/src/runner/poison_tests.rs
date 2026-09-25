//! A panic that poisoned a runner lock must not turn a Drop-time stop into an abort.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use mantis_core::Board;

use crate::replay::hexg::GraphRecord;

use super::finalize::finalize_game_graph;
use super::search_drive::FatalDefectLatch;
use super::{GameResultRow, SelfPlayRunner, SelfPlayRunnerConfig};

fn runner(n_workers: usize) -> SelfPlayRunner {
    SelfPlayRunner::new(SelfPlayRunnerConfig {
        encoding_name: Some("gnn_axis_v1".to_string()),
        n_workers,
        ..Default::default()
    })
    .expect("gnn_axis_v1 must resolve via the registry")
}

/// Poison `m` the way a worker does: a thread panics while it holds the guard.
fn poison<T: Send>(m: &Mutex<T>) {
    std::thread::scope(|s| {
        let joined = s
            .spawn(|| {
                let _held = m.lock();
                panic!("planted: a worker panics holding the lock");
            })
            .join();
        assert!(joined.is_err(), "the planted panic did not fire");
    });
    assert!(m.is_poisoned(), "the planted panic did not poison the lock");
}

/// The save the unwinding owner still owes: latch the reason, then finalize one finished game.
struct SaveOnUnwind<'a> {
    graph_results: Arc<Mutex<VecDeque<GraphRecord>>>,
    recent: Arc<Mutex<VecDeque<GameResultRow>>>,
    fatal_slot: Arc<Mutex<Option<String>>>,
    saved: &'a AtomicBool,
}

impl Drop for SaveOnUnwind<'_> {
    fn drop(&mut self) {
        let (fires, failures, running) =
            (AtomicU64::new(0), AtomicU64::new(0), AtomicBool::new(true));
        FatalDefectLatch {
            slot: &self.fatal_slot,
            fires: &fires,
            inference_failures: &failures,
            running: &running,
        }
        .store("planted defect".to_string());
        let (games, x, o, d, dropped, seq) = (
            AtomicUsize::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        );
        finalize_game_graph(
            &Board::new(),
            200,
            vec![GraphRecord::default(), GraphRecord::default()],
            vec![],
            vec![],
            None,
            &[0],
            0.0,
            0.0,
            1_000,
            0,
            &self.graph_results,
            &self.recent,
            &games,
            &x,
            &o,
            &d,
            &dropped,
            &seq,
        );
        self.saved.store(true, Ordering::SeqCst);
    }
}

/// The witness: with every lock poisoned, the Drop-time stop mid-unwind must keep the save.
#[test]
fn a_planted_panic_in_drop_during_unwind_keeps_the_save() {
    let saved = AtomicBool::new(false);
    let r = runner(1);
    poison_every_runner_lock(&r);
    let graph_results = r.graph_results.clone();
    let recent = r.recent_game_results.clone();
    let fatal_slot = r.fatal_defect.clone();

    let unwound = catch_unwind(AssertUnwindSafe(|| {
        let _save = SaveOnUnwind {
            graph_results: graph_results.clone(),
            recent: recent.clone(),
            fatal_slot: fatal_slot.clone(),
            saved: &saved,
        };
        let _owned = r;
        panic!("planted: the owner unwinds with the runner in scope");
    }));

    assert!(unwound.is_err(), "the planted unwind did not propagate");
    assert!(
        saved.load(Ordering::SeqCst),
        "the save after the Drop-time stop never ran"
    );
    let rows = graph_results.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(rows.len(), 2, "the finalized game's rows were lost");
    let games = recent.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(games.len(), 1, "the finalized game's metadata row was lost");
    let reason = fatal_slot
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert_eq!(reason.as_deref(), Some("planted defect"));
}

/// Spawn over a poisoned handle list still spawns, and the stop after it still joins every worker.
#[test]
fn start_and_stop_over_a_poisoned_handle_list_join_every_worker() {
    let r = runner(2);
    poison(&r.handles);
    r.start();
    r.stop();
    assert!(!r.is_running());
    let left = r
        .handles
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .len();
    assert_eq!(left, 0, "stop left a worker unjoined");
    assert_eq!(
        r.worker_panics(),
        0,
        "a clean worker was counted as a panic"
    );
}

/// The first-wins latch and its read both survive a poisoned slot.
#[test]
fn the_fatal_latch_stores_and_reads_through_a_poisoned_slot() {
    let r = runner(1);
    poison(&r.fatal_defect);
    r.store_fatal_defect("first".to_string());
    r.store_fatal_defect("second".to_string());
    assert_eq!(r.fatal_defect().as_deref(), Some("first"));
    assert!(!r.is_running());
}

fn poison_every_runner_lock(r: &SelfPlayRunner) {
    poison(&r.handles);
    poison(&r.fatal_defect);
    poison(&r.graph_results);
    poison(&r.recent_game_results);
}
