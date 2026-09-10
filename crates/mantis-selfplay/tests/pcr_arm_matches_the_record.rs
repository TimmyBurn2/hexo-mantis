//! ⊕ PCR under BOTH search kinds: the recorded `is_full_search` is the arm that was DRAWN.
//!
//! WHY IT NEEDS A WITNESS AT ALL. Playout-cap randomization draws a per-move arm and the
//! drawn arm decides the sim budget; the row it records carries a BOOLEAN that downstream
//! reads as "this row was searched at the full budget" (the policy-loss gate, and the seam a
//! later ruling would use to discard fast-arm rows entirely). Nothing connected the two: the
//! recorded flag WAS an OR of the draw with the forced-win hook and the solver hook, so a
//! census of the flag alone could not say whether the draw fired or a hook did, and the draw
//! itself had no counter. Both hooks went with the dense path (R346(f)); the counter this
//! file added (LAW-18: a lever under test logs its own fire rate in-run) stays, because
//! "the recorded flag is the arm that was drawn" is the claim, not "no hook interfered".
//!
//! WHY BOTH KINDS. PCR is drawn in `play_one_move` BEFORE the search kind is dispatched, so
//! it is meant to be kind-independent — and "meant to be" is what a witness is for.
//!
//! WHICH PATH. The graph one, because R346(f) left exactly one recorder. That makes the
//! two arms of this witness comparable in the way the dense drive used to: the RECORDER is
//! held fixed and only the KIND varies, which is the whole content of a kind-independence
//! claim. (The graph path is not refused under `gumbel` — R347(a) gave it the sparse row.)
//!
//! THE RESIDUAL IS STATED, NOT ASSUMED AWAY. A move draws its arm before it searches, and a
//! game's rows reach the drain only when the game FINALIZES — so the counters lead the rows
//! by at most the moves of one unfinished game. The assertions bound that gap by the ply cap
//! rather than pretending it is zero.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_search::SearchKind;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

const PLY_CAP: usize = 4;
const N_SIMS_QUICK: usize = 8;
const N_SIMS_FULL: usize = 24;
const ENCODING: &str = "gnn_axis_r8";

fn spawn_producer(
    queue: GraphQueue,
    n_actions: usize,
    served: Arc<AtomicUsize>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(4, 5);
        if batch.is_empty() {
            if queue.is_closed() {
                break;
            }
            continue;
        }
        let mut ids = Vec::with_capacity(batch.len());
        let mut results = Vec::with_capacity(batch.len());
        for (id, g) in batch {
            let coords: Vec<(i32, i32)> = g
                .legal_node_gather
                .iter()
                .map(|&row| {
                    (
                        g.node_coords[row as usize * 2],
                        g.node_coords[row as usize * 2 + 1],
                    )
                })
                .collect();
            let n = coords.len();
            let probs = vec![1.0f32 / n.max(1) as f32; n];
            ids.push(id);
            results.push(
                assemble_ls_from_gnn_probs(n_actions, &probs, &g.policy_scatter_index.0, &coords)
                    .map(|ls| (ls, 0.0f32)),
            );
        }
        served.fetch_add(ids.len(), Ordering::Relaxed);
        queue.submit_graph_results(&ids, results);
    })
}

struct Drive {
    full_rows: usize,
    quick_rows: usize,
    pcr_full: u64,
    pcr_quick: u64,
    max_sims: u64,
}

fn drive(kind: SearchKind, want_rows: usize) -> Drive {
    let spec = lookup_or_panic(ENCODING);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: PLY_CAP,
        n_simulations: N_SIMS_QUICK,
        leaf_batch_size: 4,
        random_opening_plies: 0,
        dirichlet_enabled: true,
        search_kind: kind,
        quiescence_enabled: false,
        // The lever: a per-move draw between two DIFFERENT budgets. The two hooks that
        // could otherwise set the recorded flag (the O1 forced-win and the solver
        // injectors) were deleted at R346(f), so the flag can only come from the draw.
        full_search_prob: 0.5,
        n_sims_quick: N_SIMS_QUICK,
        n_sims_full: N_SIMS_FULL,
        encoding_name: Some(ENCODING.to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_producer(
        runner.graph_producer(),
        spec.policy_logit_count,
        served.clone(),
    );

    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut rows = Vec::new();
    while Instant::now() < deadline {
        rows.extend(runner.drain_graph_records());
        if rows.len() >= want_rows || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    rows.extend(runner.drain_graph_records());
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

    let full_rows = rows.iter().filter(|r| r.is_full_search).count();
    Drive {
        full_rows,
        quick_rows: rows.len() - full_rows,
        pcr_full: snap.pcr_full_moves,
        pcr_quick: snap.pcr_quick_moves,
        max_sims: snap.max_sims_per_search,
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

    // (0) THE DRIVE IS ONE ROW PER MOVE, derived rather than assumed. The dense recorder
    // could expand one position into K cluster views, and this used to be read off the
    // k-cluster histogram. The graph recorder emits ONE record per searched move, and the
    // registry says so: `k_max == 1` on this row means whole-board, one graph per leaf. Read
    // from the registry rather than stated, so a future multi-view row reds here instead of
    // silently making the row-vs-counter comparison ill-posed.
    assert_eq!(
        lookup_or_panic(ENCODING).k_max,
        1,
        "{kind:?}: {ENCODING} declares k_max > 1, so a record is no longer a move and the \
         row-vs-counter comparisons below are not well posed"
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

    // (2) THE RECORD MATCHES THE DRAW. No hook can set the flag independently any more, so
    // every full-flagged row must have a full DRAW behind it, and likewise for
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
