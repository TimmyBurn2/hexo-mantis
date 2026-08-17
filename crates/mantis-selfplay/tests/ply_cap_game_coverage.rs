//! ⊕ F-816-9 Phase C — the coverage test the tiers lacked (R274(f), flip-set (c)).
//!
//! THE FINDING THIS EXISTS FOR. 19/19 CI gates were green while the run could not complete
//! one iteration, because no tier plays a game anywhere near the ply cap: every gate drive
//! caps at a few dozen moves, and the whole ply > ~120 regime — where every game lands under
//! R269's degenerate ply-cap flood, and where all five F-816-9 replicates died — was
//! STRUCTURALLY UNOBSERVED. That blindness is part of the defect class, not an aside
//! (Phase A §7.4).
//!
//! Drive: complete games at run5's production self-play parameters (`configs/run5.yaml`
//! selfplay block: 50 sims, leaf_batch 8, 128-move cap, Dirichlet armed, quiescence on,
//! solver and forced-win off, completed-Q off) through the PRODUCTION record path, served by
//! a healthy mock graph producer, driven to the ply cap.
//!
//! THE ACCELERATOR IS DISCLOSED, NOT BURIED — same disclosure Phase A made of the same
//! mechanism. `random_opening_plies` is set HERE, in the harness, never in a config file. It
//! skips MCTS *and* recording for the opening (`game.rs`), so it moves the worker into the
//! ply-120+ regime without touching a single SEARCH parameter: the searches this file
//! measures run at the full production regime, on a real 120-stone board with a legal set in
//! the thousands, and every one of them goes through the production record path. What it does
//! NOT cover is search behaviour at plies 0-119; the existing gate tiers cover short games,
//! and the hole this file exists to close is the deep end.
//!
//! WHY IT IS NEEDED, measured rather than assumed: a full 128-ply game at 50 sims in a DEBUG
//! build (which is what `cargo test` runs) did not complete in 600 s on the dev box. Phase A
//! reported the same shape from the other side — a 4-worker full-length release run made 474
//! moves in 2996 s and completed no game. The accelerator is what makes the deep regime
//! reachable inside a test tier at all.
//!
//! Asserted:
//!   * the game reaches the ply cap — the regime under test, not a proxy for it;
//!   * EVERY recorded position's positive-mass support fits the DERIVED visit capacity, with
//!     the capacity computed by the production derivation rather than transcribed (R255,
//!     R192(e) derive-or-delete). This is the `over_capacity == 0` claim;
//!   * no fatal defect, and BOTH R275(b) counters read 0 — a healthy game must not trip
//!     either pin. This is the negative control for the whole packet: a pin that fires on
//!     healthy play would be worse than the defect it replaces.
//!
//! TIER PLACEMENT: default `cargo test --workspace` tier (CI gate 2), not `#[ignore]`d. The
//! packet expected a gated tier; with the accelerator the drive fits the default tier, and
//! default is strictly better for a coverage hole whose entire history is "nobody ran it".
//! The runtime is stated in the commit, not here — a transcribed duration is a tally that
//! goes stale exactly like a line count (R192(e)).
//!
//! Killer: any change that re-admits an over-capacity or zero-visit export on healthy play
//! reds this file; so does a regression that stops games reaching the cap (the drive would
//! then be measuring a short game and saying nothing about the regime).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_encoding::lookup_or_panic;
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::replay::hexg::derived_visit_capacity;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

// `configs/run5.yaml`, selfplay block. Named here so a drift between this drive and the
// shipped production regime is a one-diff read.
const PROD_SIMS: usize = 50;
const PROD_LEAF_BATCH: usize = 8;
const PROD_PLY_CAP: usize = 128;
const PROD_FAST_SIMS: usize = 50;
const PROD_DIRICHLET_ALPHA: f32 = 0.3;
const PROD_DIRICHLET_EPSILON: f32 = 0.25;
/// HARNESS-ONLY accelerator (see the header). Plies below this are played at random with no
/// search and no recording, so the searched-and-recorded window is exactly the deep regime.
const RANDOM_OPENING_PLIES: u32 = 120;
/// Derived, never transcribed: what the drive above must record per game.
const SEARCHED_PLIES: usize = PROD_PLY_CAP - RANDOM_OPENING_PLIES as usize;

fn spawn_healthy_graph_producer(
    queue: GraphQueue,
    n_actions: usize,
    served: Arc<AtomicUsize>,
) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(PROD_LEAF_BATCH, 5);
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
                .map(|&row| (g.node_coords[row as usize * 2], g.node_coords[row as usize * 2 + 1]))
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

#[test]
fn a_full_ply_cap_game_at_production_parameters_records_within_the_derived_capacity() {
    let spec = lookup_or_panic("gnn_axis_v1");
    let capacity = derived_visit_capacity(
        PROD_SIMS, 0, 0.0, PROD_FAST_SIMS, 0.0, 0, 0, PROD_LEAF_BATCH, false,
    )
    .expect("the production sims regime must have a derivable capacity");

    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: PROD_PLY_CAP,
        n_simulations: PROD_SIMS,
        leaf_batch_size: PROD_LEAF_BATCH,
        fast_sims: PROD_FAST_SIMS,
        standard_sims: 0,
        quiescence_enabled: true,
        quiescence_blend_2: 0.3,
        dirichlet_enabled: true,
        dirichlet_alpha: PROD_DIRICHLET_ALPHA,
        dirichlet_epsilon: PROD_DIRICHLET_EPSILON,
        completed_q_values: false,
        gumbel_mcts: false,
        solver_enabled: false,
        forced_win_policy_enabled: false,
        random_opening_plies: RANDOM_OPENING_PLIES,
        encoding_name: Some("gnn_axis_v1".to_string()),
        ..Default::default()
    })
    .expect("production-parameter gnn runner constructs");

    let served = Arc::new(AtomicUsize::new(0));
    let producer =
        spawn_healthy_graph_producer(runner.graph_producer(), spec.policy_logit_count, served.clone());

    runner.start();
    // Wait on the RECORDS, not on `games_completed`. Two race modes die here rather than
    // becoming a flake: the counter advancing before the finalized rows reach the drain
    // queue (a drain of 0), and a second game finishing between the break and the drain (a
    // drain of 2 games). Only FINALIZED games reach this queue — an in-progress game's rows
    // live in the worker's local vec — so the accumulated length is always a whole number of
    // games.
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut records = Vec::new();
    while Instant::now() < deadline {
        records.extend(runner.drain_graph_records());
        if records.len() >= SEARCHED_PLIES || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let snap = runner.stats_snapshot();
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");

    assert!(served.load(Ordering::Relaxed) > 0, "no graph inference served — vacuous drive");
    assert!(defect.is_none(), "a HEALTHY production game latched a fatal defect: {defect:?}");
    assert!(
        snap.games_completed >= 1,
        "no game completed inside the budget — this drive would then be measuring a partial \
         game and could not speak about the ply-cap regime at all"
    );
    assert!(
        !records.is_empty() && records.len() % SEARCHED_PLIES == 0,
        "one recorded position per SEARCHED ply is the production record path's contract, so \
         the drain must carry a whole number of finalized games' rows ({SEARCHED_PLIES} \
         each); got {}",
        records.len()
    );

    // The regime, asserted rather than assumed: the ply cap is genuinely reached.
    let max_ply = records.iter().map(|r| r.ply_index).max().unwrap_or(0);
    assert_eq!(
        usize::from(max_ply) + 1,
        PROD_PLY_CAP,
        "the game did not reach the ply cap (deepest recorded ply_index {max_ply}) — the \
         ply > ~120 regime is exactly what no gate tier covers, and a short game here would \
         restore that blindness while reporting green"
    );

    // over_capacity == 0, against the DERIVED bound.
    let over: Vec<(u16, usize)> = records
        .iter()
        .map(|r| (r.ply_index, r.visits.len()))
        .filter(|&(_, n)| n > capacity)
        .collect();
    assert!(
        over.is_empty(),
        "positions exceeded the derived visit capacity {capacity} on HEALTHY play: {over:?}"
    );

    // Both R275(b) pins are negative controls here: healthy play must trip neither.
    assert_eq!(
        snap.target_integrity_defects, 0,
        "the exporter pin fired on a healthy full-length game — a pin that refuses real \
         searches is worse than the defect it replaces"
    );
    assert_eq!(
        snap.inference_failures_total, 0,
        "the seam pin fired with every inference served successfully"
    );
}
