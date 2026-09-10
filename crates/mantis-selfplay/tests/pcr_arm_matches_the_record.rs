//! ⊕ PCR under BOTH search kinds: the recorded `is_full_search` is the arm that was DRAWN.
//!
//! WHY IT NEEDS A WITNESS AT ALL. Playout-cap randomization draws a per-move arm and the
//! drawn arm decides the sim budget; the row it records carries a BOOLEAN that downstream
//! reads as "this row was searched at the full budget" (the policy-loss gate, and the seam a
//! later ruling would use to discard fast-arm rows entirely). Nothing connected the two: the
//! recorded flag is an OR of the draw with the forced-win hook and the solver hook, so a
//! census of the flag alone cannot say whether the draw fired or a hook did, and the draw
//! itself had no counter. This file adds the counter (LAW-18: a lever under test logs its
//! own fire rate in-run) and measures the two against each other.
//!
//! WHY BOTH KINDS. PCR is drawn in `play_one_move` BEFORE the search kind is dispatched, so
//! it is meant to be kind-independent — and "meant to be" is what a witness is for.
//!
//! WHY THE DENSE PATH. The dense recorder is the one BOTH kinds share, so driving it is
//! what makes the two arms of this witness comparable. (The graph path is no longer refused
//! under `gumbel` — R347(a) gave it the sparse row — but its recorder is graph-only, so a
//! kind-independence claim measured there would be measuring one recorder against itself.)
//!
//! THE RESIDUAL IS STATED, NOT ASSUMED AWAY. A move draws its arm before it searches, and a
//! game's rows reach the drain only when the game FINALIZES — so the counters lead the rows
//! by at most the moves of one unfinished game. The assertions bound that gap by the ply cap
//! rather than pretending it is zero.

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_search::SearchKind;
use mantis_selfplay::queues::DenseQueue;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

const PLY_CAP: usize = 4;
const N_SIMS_QUICK: usize = 8;
const N_SIMS_FULL: usize = 24;

fn spawn_producer(
    queue: DenseQueue,
    policy_stride: usize,
    served: Arc<AtomicUsize>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_batch(1, 5);
        if batch.is_empty() {
            if queue.is_closed() {
                break;
            }
            continue;
        }
        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        let mut flat: Vec<f32> = Vec::new();
        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(batch.len());
        let mut values: Vec<f32> = Vec::with_capacity(batch.len());
        let uniform = 1.0f32 / policy_stride as f32;
        for _ in &batch {
            let start = flat.len();
            flat.extend(std::iter::repeat_n(uniform, policy_stride));
            ranges.push(start..flat.len());
            values.push(0.0);
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        let arc = Arc::new(flat);
        queue.submit_results(&ids, &arc, &ranges, &values);
    })
}

struct Drive {
    full_rows: usize,
    quick_rows: usize,
    pcr_full: u64,
    pcr_quick: u64,
    max_sims: u64,
    single_view_positions: u64,
    total_positions: u64,
}

fn drive(kind: SearchKind, want_rows: usize) -> Drive {
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: PLY_CAP,
        n_simulations: N_SIMS_QUICK,
        leaf_batch_size: 4,
        random_opening_plies: 0,
        dirichlet_enabled: true,
        search_kind: kind,
        quiescence_enabled: false,
        // The lever: a per-move draw between two DIFFERENT budgets. Both hooks that could
        // otherwise set the recorded flag are OFF, so the flag can only come from the draw.
        full_search_prob: 0.5,
        n_sims_quick: N_SIMS_QUICK,
        n_sims_full: N_SIMS_FULL,
        solver_enabled: false,
        forced_win_policy_enabled: false,
        encoding_name: Some("v6".to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let policy_stride = runner.policy_len();
    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_producer(runner.dense_producer(), policy_stride, served);

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut rows = Vec::new();
    while Instant::now() < deadline {
        rows.extend(runner.drain_training_rows());
        if rows.len() >= want_rows || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    rows.extend(runner.drain_training_rows());
    // AFTER the join, so the counters cannot trail the rows they are compared against.
    let snap = runner.stats_snapshot();

    assert!(
        defect.is_none(),
        "{kind:?}: latched a fatal defect: {defect:?}"
    );
    assert!(
        rows.len() >= want_rows,
        "{kind:?}: only {} rows inside the budget — a drive that records nothing cannot \
         speak about which arm was recorded",
        rows.len()
    );

    let full_rows = rows.iter().filter(|r| r.6).count();
    Drive {
        full_rows,
        quick_rows: rows.len() - full_rows,
        pcr_full: snap.pcr_full_moves,
        pcr_quick: snap.pcr_quick_moves,
        max_sims: snap.max_sims_per_search,
        single_view_positions: snap.k_cluster_histogram[0],
        total_positions: snap.k_cluster_histogram.iter().sum(),
    }
}

fn assert_pcr(kind: SearchKind) {
    // WHY 48 AND NOT 8. Assertions (1) and (3) below need BOTH arms to appear, and the arm
    // is a fair coin: at 8 rows the chance that one arm never fires is ~2^-8 per side, and
    // this file was observed red at `full=1 quick=8` before the sample was widened. 48 puts
    // that tail under 2^-48 without making the drive slow — the arms are drawn per move, so
    // the cost is a few more four-ply games.
    let d = drive(kind, 48);
    // Printed, not merely asserted: the QUANTITIES are what a re-mint reads, and a witness
    // that only says "consistent" cannot be quoted (LAW-01, measurement mandatory).
    println!(
        "{kind:?}: drew full={} quick={}; recorded full={} quick={} (max sims/search {})",
        d.pcr_full, d.pcr_quick, d.full_rows, d.quick_rows, d.max_sims
    );

    // (0) THE DRIVE IS ONE ROW PER MOVE, derived rather than assumed: at this ply cap every
    // recorded position expands into exactly ONE cluster view, so a row IS a move and the
    // comparisons below are between comparable units.
    assert_eq!(
        d.single_view_positions,
        d.total_positions,
        "{kind:?}: {} of {} recorded positions expanded into more than one cluster view, so \
         a row is not a move and the row-vs-counter comparison below is not well posed",
        d.total_positions - d.single_view_positions,
        d.total_positions
    );

    // (1) THE LEVER FIRES BOTH WAYS — LAW-18's fire rate, on the counter that sits AT the
    // draw. A lever that only ever drew one arm would make every other assertion vacuous.
    assert!(
        d.pcr_full > 0 && d.pcr_quick > 0,
        "{kind:?}: the playout-cap draw produced only one arm (full={}, quick={}) — at \
         p=0.5 over this many moves that is a stuck draw, not a sample",
        d.pcr_full,
        d.pcr_quick
    );

    // (2) THE RECORD MATCHES THE DRAW. Both hooks that could set the flag independently are
    // off, so every full-flagged row must have a full DRAW behind it, and likewise for
    // quick. The relation is `<=` and not `==` for the reason the header states: the
    // counters lead the rows by the moves of one unfinished game.
    assert!(
        d.full_rows as u64 <= d.pcr_full,
        "{kind:?}: {} rows are flagged full_search but only {} full arms were drawn — the \
         flag is coming from somewhere other than the draw",
        d.full_rows,
        d.pcr_full
    );
    assert!(
        d.quick_rows as u64 <= d.pcr_quick,
        "{kind:?}: {} rows are flagged quick but only {} quick arms were drawn",
        d.quick_rows,
        d.pcr_quick
    );

    // (3) AND BOTH ARMS REACH THE RECORD. A flag that were pinned true would satisfy (2) on
    // the full side and silently lose the quick arm from the training signal.
    assert!(
        d.full_rows > 0 && d.quick_rows > 0,
        "{kind:?}: the recorded rows carry only one arm (full={}, quick={}) — the flag is \
         not travelling with the draw",
        d.full_rows,
        d.quick_rows
    );

    // (4) THE GAP IS THE STATED RESIDUAL and nothing wider: one unfinished game's moves.
    let drawn = d.pcr_full + d.pcr_quick;
    let recorded = (d.full_rows + d.quick_rows) as u64;
    assert!(
        drawn >= recorded && drawn - recorded <= PLY_CAP as u64,
        "{kind:?}: {drawn} arms drawn against {recorded} rows recorded — the gap must be at \
         most one unfinished game ({PLY_CAP} moves); a wider one means draws are happening \
         on moves that never record"
    );

    // (5) THE FULL ARM ACTUALLY SPENDS THE FULL BUDGET. Without this the flag could be
    // correct while both arms searched the same amount, which is a no-op randomization.
    assert_eq!(
        d.max_sims, N_SIMS_FULL as u64,
        "{kind:?}: the widest search served {} leaves against a full arm of {N_SIMS_FULL}",
        d.max_sims
    );
}

#[test]
fn the_puct_kind_records_the_arm_it_drew() {
    assert_pcr(SearchKind::Puct);
}

#[test]
fn the_gumbel_kind_records_the_arm_it_drew() {
    assert_pcr(SearchKind::Gumbel);
}
