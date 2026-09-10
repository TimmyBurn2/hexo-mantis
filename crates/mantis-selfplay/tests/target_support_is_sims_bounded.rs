// >300 justify (R8): ONE claim — what bounds an exported HEXG row's support — measured on a
// real runner, plus the structural facts that make the measurement conclusive.
//! The exported target's support is bounded by the SIM BUDGET, not the root's child count.
//!
//! Under `search.kind: puct` the target is the visit-count distribution, whose three arms are:
//! zero temperature, one-hot, support 1; positive total, where an UNVISITED child contributes
//! zero mass and the recorder keeps an entry only when `p > 0.0`, so the support is the VISITED
//! children; and zero total, the prior fallback over the FULL child set, which IS
//! child-count-wide and is DEAD while recording, since `refuse_zero_visit_export` is its exact
//! complement and runs first. The root cap changes which actions CAN be visited, not how many
//! ARE.
//!
//! A Gumbel row, the other kind, is SPARSE, so its width is the minted `selfplay.gumbel_m` and
//! not the sims regime. Two row kinds, two bounds; this file measures the PUCT one.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mantis_core::board::Cell;
use mantis_core::{Board, Player};
use mantis_encoding::lookup_or_panic;
use mantis_search::{SearchKind, MAX_CHILDREN_PER_NODE};
use mantis_selfplay::queues::GraphQueue;
use mantis_selfplay::records::assemble_ls_from_gnn_probs;
use mantis_selfplay::replay::hexg::GraphRecord;
use mantis_selfplay::runner::{SelfPlayRunner, SelfPlayRunnerConfig};

const LEAF_BATCH: usize = 8;
/// Small enough that the sim budget is FAR below the r8 legal-move count.
const SIMS: usize = 24;

fn spawn_producer(queue: GraphQueue, n_actions: usize, served: Arc<AtomicUsize>) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let batch = queue.pop_graph_batch(LEAF_BATCH, 5);
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

/// Drive one worker under `kind` until `want` graph records are drained.
fn drive(kind: SearchKind, want: usize) -> Vec<GraphRecord> {
    let encoding = "gnn_axis_r8";
    let spec = lookup_or_panic(encoding);
    let runner = SelfPlayRunner::new(SelfPlayRunnerConfig {
        n_workers: 1,
        max_moves_per_game: 4,
        n_simulations: SIMS,
        leaf_batch_size: LEAF_BATCH,
        random_opening_plies: 0,
        dirichlet_enabled: true,
        search_kind: kind,
        encoding_name: Some(encoding.to_string()),
        ..Default::default()
    })
    .expect("runner constructs at the drive's parameters");

    let served = Arc::new(AtomicUsize::new(0));
    let producer = spawn_producer(runner.graph_producer(), spec.policy_logit_count, served);
    runner.start();
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut rows = Vec::new();
    while Instant::now() < deadline {
        rows.extend(runner.drain_graph_records());
        if rows.len() >= want || runner.fatal_defect().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let defect = runner.fatal_defect();
    runner.stop();
    producer.join().expect("producer exits");
    rows.extend(runner.drain_graph_records());

    assert!(
        defect.is_none(),
        "{kind:?} latched a fatal defect: {defect:?}"
    );
    assert!(
        rows.len() >= want,
        "{kind:?}: only {} rows inside the budget — a drive that records nothing cannot \
         speak about row support",
        rows.len()
    );
    rows
}

/// A real driven r8 game: every exported row's support is at most the sim budget.
#[test]
fn an_exported_rows_support_is_the_sim_budget_not_the_child_count() {
    let rows = drive(SearchKind::Puct, 6);

    // One stored entry per VISITED child, and a search spends at most `SIMS` visits.
    for row in &rows {
        assert!(
            !row.visits.is_empty(),
            "an empty visit target is a non-distribution and should have been refused \
             upstream"
        );
        assert!(
            row.visits.len() <= SIMS,
            "a row stored {} entries against a {SIMS}-sim budget. Support is supposed to be \
             the VISITED children — if this fires, unvisited children are reaching the \
             record and the row width is child-count-wide after all",
            row.visits.len()
        );
        assert!(
            row.visits.iter().all(|&(_, _, p)| p > 0.0),
            "a zero-mass entry reached the record — the `p > 0.0` filter in \
             `record_position_graph` is what makes support visit-bounded"
        );
    }

    // The drive must reach the regime where the two bounds diverge, or the rows above are
    // small for a reason that has nothing to do with the claim.
    let widest_legal = rows
        .iter()
        .map(|row| {
            let stones: Vec<((i32, i32), Cell)> = row
                .stones
                .iter()
                .map(|&(q, r, c)| {
                    (
                        (i32::from(q), i32::from(r)),
                        if c > 0 { Cell::P1 } else { Cell::P2 },
                    )
                })
                .collect();
            let mut board = Board::from_stones(&stones, Player::One, row.moves_remaining, 0, None);
            board.set_legal_move_radius(8);
            board.legal_moves().len()
        })
        .max()
        .unwrap_or(0);
    // The child count a row could have had is `min(n_legal, cap)`, and it is that against the
    // sim budget that decides whether the two predictions differ.
    let widest_children = widest_legal.min(MAX_CHILDREN_PER_NODE);
    println!(
        "widest recorded legal set {widest_legal}, per-node cap {MAX_CHILDREN_PER_NODE}, so \
         the widest child count a row could have had is {widest_children} against a \
         {SIMS}-sim budget"
    );
    assert!(
        widest_children > SIMS,
        "the widest child count any recorded root could have had is {widest_children}, not \
         more than the {SIMS}-sim budget — so 'sims-bounded' and 'child-count-bounded' make \
         the same prediction here and the row widths above prove nothing"
    );
}

/// The child-count-wide exporter arm is unreachable while recording: it fires on zero total,
/// and the refusal sums the same visits, is run-fatal at 0, and runs FIRST.
#[test]
fn a_zero_visit_search_is_refused_before_the_wide_exporter_arm_can_run() {
    use mantis_search::MCTSTree;
    use mantis_selfplay::records::refuse_zero_visit_export;

    let mut board = mantis_core::Board::new();
    board.set_legal_move_radius(8);
    board
        .apply_move(0, 0)
        .expect("(0,0) is legal on a fresh board");
    // Grown until the legal set clears the per-node cap: the state needed is a root WIDER than
    // the cap. Stepping outward by one radius keeps every move legal and grows the union.
    let mut q = 0;
    while board.legal_moves().len() <= MAX_CHILDREN_PER_NODE {
        q += 8;
        assert!(
            q <= 8 * 64,
            "the legal set stopped growing at {} before clearing the cap {}",
            board.legal_moves().len(),
            MAX_CHILDREN_PER_NODE
        );
        board
            .apply_move(q, 0)
            .expect("a step of exactly one legal-move radius is legal from the last stone");
    }

    let mut tree = MCTSTree::new(1.5);
    tree.configure_quiescence(false, 0.0);
    tree.configure_search(SearchKind::Gumbel, 50.0, 0.1);
    tree.new_game(board);
    let leaves = tree.select_leaves(1).expect("a fresh root selects itself");
    assert_eq!(leaves.len(), 1);
    let stride = 19 * 19 + 1;
    tree.expand_and_backup(&[vec![1.0f32 / stride as f32; stride]], &[0.25]);

    // Root expanded over the full legal set with NO child visited — the fallback arm's own
    // state.
    assert!(
        tree.root_n_children() > MAX_CHILDREN_PER_NODE,
        "the Gumbel root must be wider than the per-node cap for this to be the \
         interesting state"
    );
    assert!(
        refuse_zero_visit_export(&tree, 0).is_err(),
        "a search whose children carry no visits must be REFUSED, not exported. If this ever \
         passes, the child-count-wide prior-fallback arm becomes reachable and the row-width \
         claim in this file's header stops holding"
    );
}
