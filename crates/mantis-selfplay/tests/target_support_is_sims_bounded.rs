// R8 justify: ONE claim — "what bounds an exported HEXG row's support" — measured on a real
// runner, plus the two structural facts that make the measurement conclusive (the zero-mass
// filter, and the refusal that kills the one child-count-wide exporter arm). The drive and
// the two facts are the same argument at two altitudes; split them and the numbers stop
// being evidence for the claim they were taken for.
//! ⊕ the exported target's support is bounded by the SIM BUDGET, not by the root's child
//! count.
//!
//! WHY THIS FILE EXISTS. `GUMBEL_REPAIR_1_EXIT.md` §4(a) claimed that raising the root cap
//! under the corrected Gumbel dialect could put two ROW KINDS in one replay ring — a
//! truncated-support row and a full-legal-support one — and that a resume across a dialect
//! change was the way they would mix. **That claim was wrong**, and this file is the
//! measurement that retires it rather than a note saying so.
//!
//! THE DERIVATION, which the drive below confirms end to end. Under `search.kind: puct` —
//! the ONLY kind a graph run can boot (see below) — the exported target is `get_policy_ls`,
//! the visit-count distribution. It has three arms and only one is reachable while
//! recording:
//!
//!  * `temperature == 0.0` → one-hot on the most-visited child. Support 1.
//!  * `total > 0.0` → `visits^(1/T) / total` per child. An UNVISITED child contributes
//!    `0^(1/T) = 0`, and `records::record_position_graph` keeps an entry only `if p > 0.0`,
//!    so unvisited children are dropped before the row is built. **Support = the VISITED
//!    children**, which `n_simulations` bounds.
//!  * `total == 0.0` → the prior-fallback distribution over the FULL child set. This arm IS
//!    child-count-wide — and it is DEAD on the recording path:
//!    `records::refuse_zero_visit_export` runs BEFORE the exporter and is its exact
//!    complement (it sums the same children's visits and is run-fatal at 0), so a search
//!    that would take this arm never reaches a record at all.
//!
//! The root cap changes which actions CAN be visited. It does not change how many ARE, and
//! only the visited ones are stored.
//!
//! AND THE COMPLETED TARGET, THE OTHER ROW KIND, IS BOUNDED BY A DIFFERENT QUANTITY.
//! `search.kind: gumbel` on a graph run now boots (R347(a)), and its row is SPARSE: the m
//! sampled candidates' exact entries plus one tail mass, so its width is the minted
//! `selfplay.gumbel_m` and NOT the sims regime. The two row kinds therefore have two
//! bounds, each derived by the same one authority from the kind the run declared, and this
//! file measures the PUCT one. The grid path records fixed-width dense rows and has no
//! variable-length visit vec at all, so the row-kind question does not arise there.

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
/// Small enough that the sim budget is FAR below the r8 legal-move count, which is what
/// makes "sims-bounded" and "child-count-bounded" different predictions.
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

/// THE MEASUREMENT. A real driven r8 game: every exported row's support is at most the sim
/// budget, however wide the root's legal set is.
#[test]
fn an_exported_rows_support_is_the_sim_budget_not_the_child_count() {
    let rows = drive(SearchKind::Puct, 6);

    // The bound the derivation predicts: one stored entry per VISITED child, and a search
    // spends at most `SIMS` visits across the root's children.
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

    // THE DRIVE MUST REACH THE REGIME WHERE THE TWO BOUNDS DIVERGE, or everything above is
    // vacuous: if every recorded root held fewer legal moves than the per-node cap, nothing
    // truncated and the row widths would be small for a reason that has nothing to do with
    // what is being claimed. Rebuild each recorded position and count its legal set.
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
    // The CHILD COUNT a row could have had is `min(n_legal, cap)`, and it is that against
    // the sim budget that decides whether the two predictions differ. Written as the cap
    // alone while the cap sat below every r8 legal set; R347(c) raised it past them, at which
    // point the cap stopped being the binding term and a cap-only guard started asking for a
    // regime this drive cannot reach — while the property it guards was never in doubt.
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

/// The structural half, stated as its own property: the exporter arm that IS child-count-wide
/// is unreachable while recording, because the refusal that guards it is its exact
/// complement.
///
/// `get_policy_ls`'s prior-fallback arm fires on `total == 0.0` — every root child unvisited.
/// `refuse_zero_visit_export` sums those same children's visits and is run-fatal at 0, and it
/// runs FIRST. The two conditions are the same condition, so the wide arm cannot ship a row.
#[test]
fn a_zero_visit_search_is_refused_before_the_wide_exporter_arm_can_run() {
    use mantis_search::MCTSTree;
    use mantis_selfplay::records::refuse_zero_visit_export;

    let mut board = mantis_core::Board::new();
    board.set_legal_move_radius(8);
    board
        .apply_move(0, 0)
        .expect("(0,0) is legal on a fresh board");
    // Grown until the legal set clears the per-node cap, rather than left at one stone: the
    // state this test needs is a Gumbel root WIDER than the cap, and one radius-8 ball stopped
    // being that when R347(c) raised the cap. Stepping outward by exactly one radius keeps
    // every move legal from the stone before it and grows the union monotonically.
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

    // The root is expanded over the full legal set and NO child has been visited — precisely
    // the state the prior-fallback arm exists for.
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
