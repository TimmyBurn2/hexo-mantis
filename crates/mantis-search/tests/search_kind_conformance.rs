// R8 justify: ONE suite-v2 section — "the search kind behind the seam" — and a section is
// a unit. Its detector, its witnesses and its stated floor/envelope are the three halves of
// a single claim, and the envelope's numbers are only meaningful beside the witness that
// spends them.
//! ⊕ SUITE V2 SECTION: the search kind.
//!
//! Suite v2's existing sections (`SEAM_V1_DESIGN.md` §3) are per-ARCH: perf floor, memory
//! envelope, config partition. This is the same shape applied to the SEARCH KIND, which is
//! the other thing behind the seam that a run's strength depends on and that nothing was
//! making state its terms.
//!
//! RUST-SIDE, and the reason is where the object lives. The arch sections are Python because
//! an arch is a Python object; a search kind is a Rust one, reachable from Python only
//! through a config key whose own wire is pinned separately
//! (`tests/config/test_every_key_has_consumer.py`, `tests/selfplay/test_pool_hparams.py`).
//!
//! - **DETECTOR** — the kind. Which search a tree runs is READABLE off the tree, and the
//!   terms that follow from it move with it.
//! - **WITNESS** — the parity vectors live in `mctx_parity.rs` and `mcts/parity_tests.rs`
//!   (Mctx's own numbers). Here: a Gumbel root reaches the FULL legal set, and the
//!   exported target puts mass on all of it and sums to 1.
//! - **FLOOR / ENVELOPE** — the kind's pool terms, stated and derived rather than
//!   transcribed.

use mantis_core::Board;
use mantis_search::{
    MCTSTree, MctxRootState, SearchKind, MAX_ARMED_SIMS, MAX_ARMED_SIMS_GUMBEL,
    MAX_CHILDREN_PER_NODE, MAX_NODES, MAX_ROOT_CHILDREN,
};

/// Board stride for a 19-window encoding with a pass slot — the shape every in-src MCTS
/// fixture uses.
const N_ACTIONS: usize = 19 * 19 + 1;

/// A radius-8 board whose legal set clears the per-node cap, so the two kinds are
/// DISTINGUISHABLE at the root.
///
/// GROWN UNTIL IT CLEARS THE CAP, not a stone list tuned to one. This was two adjacent
/// stones, which put 232 moves in the legal set — comfortably over a 192-wide cap and under
/// the 1024-wide one R347(c) minted, so the fixture stopped being able to tell the kinds
/// apart the moment the constant moved. Stepping outward by one radius keeps every move
/// legal from the stone before it and grows the union of balls monotonically, so the loop
/// terminates against any cap the pool can serve.
fn wide_board() -> Board {
    let mut board = Board::new();
    board.set_legal_move_radius(8);
    board
        .apply_move(0, 0)
        .expect("(0,0) is legal on a fresh board");
    let mut q = 0;
    while board.legal_moves().len() <= MAX_CHILDREN_PER_NODE {
        q += 8;
        assert!(
            q <= 8 * 64,
            "the legal set stopped growing at {} moves before clearing the cap {}",
            board.legal_moves().len(),
            MAX_CHILDREN_PER_NODE
        );
        board
            .apply_move(q, 0)
            .expect("a step of exactly one legal-move radius is legal from the last stone");
    }
    board
}

/// Expand the root once with a uniform policy and a stated leaf value.
///
/// THE SERIALISING LOCK THIS FILE USED TO HOLD IS GONE (R347). The omitted-prior counters
/// were process-global, so cargo running this file's tests on threads of one process meant
/// the omitted-mass witness measured its own expansion plus whichever sibling expanded
/// inside its window — it once read "2 truncating expansions" for one truncating expansion,
/// which is a wrong number rather than a flaky one. The counters are now per-`MCTSTree`, so
/// the window belongs to the tree that opened it and no sibling can enter it.
fn expand_root(kind: SearchKind, value: f32) -> (MCTSTree, Board) {
    expand_root_unlocked(kind, value)
}

/// `expand_root` for a caller that already holds `EXPAND_LOCK`.
fn expand_root_unlocked(kind: SearchKind, value: f32) -> (MCTSTree, Board) {
    let board = wide_board();
    let mut tree = MCTSTree::new(1.5);
    // Quiescence OFF so the backed-up value is the supplied one and the raw-value witness
    // reads what it was handed rather than a corrected version of it.
    tree.configure_quiescence(false, 0.0);
    tree.configure_search(kind, 50.0, 0.1);
    tree.new_game(board.clone());
    let leaves = tree.select_leaves(1).expect("a fresh root selects itself");
    assert_eq!(leaves.len(), 1, "root expansion selects exactly the root");
    let policy = vec![1.0f32 / N_ACTIONS as f32; N_ACTIONS];
    tree.expand_and_backup(&[policy], &[value]);
    (tree, board)
}

// ── DETECTOR ────────────────────────────────────────────────────────────────────

#[test]
fn the_kind_is_readable_off_the_tree_and_carries_its_root_cap() {
    let mut tree = MCTSTree::new(1.5);
    assert_eq!(
        tree.search_kind(),
        SearchKind::Puct,
        "an unconfigured tree runs PUCT: the CONFIG carries no default (an absent \
         `search.kind` is a mint error), and the constructor's value is what a bench or a \
         unit fixture that never calls `configure_search` gets"
    );
    assert_eq!(tree.root_children_cap(), MAX_CHILDREN_PER_NODE);

    tree.configure_search(SearchKind::Gumbel, 50.0, 0.1);
    assert_eq!(tree.search_kind(), SearchKind::Gumbel);
    assert_eq!(
        tree.root_children_cap(),
        MAX_ROOT_CHILDREN,
        "the root cap must follow the kind — a tree that reads `Gumbel` while its root \
         still truncates at the per-node cap is sampling Gumbel-Top-k over a set that \
         cannot contain the action it would have drawn"
    );

    tree.configure_search(SearchKind::Puct, 50.0, 1.0);
    assert_eq!(tree.root_children_cap(), MAX_CHILDREN_PER_NODE, "and back");
}

/// THE PUCT ARM IS A REAL ARM, not the absence of Gumbel. It truncates its root at the
/// per-node cap, keeps no raw values, and pays for neither.
#[test]
fn the_puct_kind_keeps_the_per_node_root_and_no_raw_values() {
    let (puct, _) = expand_root(SearchKind::Puct, 0.25);
    assert_eq!(
        puct.root_n_children(),
        MAX_CHILDREN_PER_NODE,
        "the PUCT root truncates at the per-node cap like every other node"
    );
    assert_eq!(puct.root_raw_value(), 0.0, "and keeps no raw values");
}

#[test]
fn the_config_spelling_round_trips_and_an_unknown_kind_is_refused() {
    for k in [SearchKind::Puct, SearchKind::Gumbel] {
        assert_eq!(SearchKind::from_config_str(k.as_config_str()), Some(k));
    }
    // REFUSED, not defaulted. A kind that falls back to `puct` on a typo is the
    // silent-fallback class LAW-11 closes — and the four keys this one replaces
    // (`gumbel_mcts`, `gumbel_variant`, and a `completed_q_values` on each of two
    // sections) must not parse either.
    for bad in [
        "",
        "PUCT",
        "gumbel ",
        "legacy",
        "mctx",
        "true",
        "gumbel_mcts",
    ] {
        assert_eq!(
            SearchKind::from_config_str(bad),
            None,
            "{bad:?} must not parse as a search kind"
        );
    }
}

/// The TARGET SEMANTICS are the kind's own answer, not a second pair of flags.
#[test]
fn the_kind_is_the_only_authority_over_the_target() {
    assert!(
        SearchKind::Gumbel.completed_q_target(),
        "Gumbel exports the completed-Q improved policy"
    );
    assert!(
        !SearchKind::Puct.completed_q_target(),
        "PUCT exports the visit distribution"
    );
}

// ── WITNESS ─────────────────────────────────────────────────────────────────────

#[test]
fn the_gumbel_root_holds_the_full_legal_set_and_the_puct_root_does_not() {
    let legal = wide_board().legal_moves().len();
    assert!(
        legal > MAX_CHILDREN_PER_NODE,
        "the fixture must have more legal moves than the per-node cap or the two kinds \
         are indistinguishable here (got {legal})"
    );

    let (gumbel, _) = expand_root_unlocked(SearchKind::Gumbel, 0.25);
    let (_, omitted_expansions, _) = gumbel.omitted_prior_stats();
    assert_eq!(
        gumbel.root_n_children(),
        legal,
        "the Gumbel root must reach EVERY legal move: Gumbel-Top-k perturbs every action's \
         logit, so an action outside the sampled set can never be drawn however large its \
         Gumbel value"
    );
    assert_eq!(
        omitted_expansions, 0,
        "the omitted-prior witness, read at the root: no prior mass left the root"
    );

    let (puct, _) = expand_root_unlocked(SearchKind::Puct, 0.25);
    let (mass_micros, omitted_expansions, total_expansions) = puct.omitted_prior_stats();
    println!(
        "omitted-prior at the root: gumbel {:?}, puct {:?}; legal {legal}, cap \
         {MAX_CHILDREN_PER_NODE}",
        gumbel.omitted_prior_stats(),
        puct.omitted_prior_stats()
    );
    assert_eq!(
        puct.root_n_children(),
        MAX_CHILDREN_PER_NODE,
        "the PUCT root truncates at the per-node cap"
    );
    assert!(
        puct.root_n_children() < legal,
        "the PUCT root kept every legal move, so it did not truncate and the comparison \
         above proves nothing"
    );
    assert_eq!(
        total_expansions, 1,
        "the truncating expansion was not counted at all"
    );
    // AND THE DROPPED MASS IS ZERO HERE, WHICH IS A STRUCTURAL FACT AND NOT A GAP. This is
    // the DENSE expand: its policy vector covers the 19-window's 361 cells and nothing else,
    // so an off-window cell's sort prior is 0. Once the per-node cap exceeds 361 — R347(c)
    // made it 1024 — every child the dense picker can drop is off-window and therefore
    // zero-prior, so `mass_micros` cannot be positive on this arm at any board. The mass
    // MECHANISM is witnessed where it can bite, at an explicit sub-window cap, in
    // `mcts/tests.rs::omitted_prior_mass_is_the_tail_the_cap_dropped`.
    assert_eq!(
        (mass_micros, omitted_expansions),
        (0, 0),
        "the DENSE root dropped positive prior mass, which the 361-cell policy vector \
         cannot supply above a {MAX_CHILDREN_PER_NODE}-wide cap"
    );
}

#[test]
fn the_exported_target_covers_the_full_legal_set_and_sums_to_one() {
    let legal = wide_board().legal_moves().len();
    let (tree, _) = expand_root(SearchKind::Gumbel, 0.25);

    // The LEGAL-SET exporter, not the dense one: at radius 8 many legal cells fall outside
    // the 19-window, and the dense export drops `action >= n_actions` by construction. The
    // ragged exporter routes them to `overflow`, which is why every graph run uses it.
    let target = tree.get_improved_policy_ls(N_ACTIONS, 50.0, 0.1);
    let support = target.dense.iter().filter(|&&m| m > 0.0).count() + target.overflow.len();
    assert_eq!(
        support, legal,
        "the exported target must put mass on every legal move — a target whose support is \
         the truncated child set teaches the net that the dropped moves are unplayable"
    );

    let total: f32 = target.dense.iter().sum::<f32>() + target.overflow.values().sum::<f32>();
    assert!(
        (total - 1.0).abs() < 1e-4,
        "the exported target sums to {total}, not 1"
    );
}

#[test]
fn only_the_gumbel_kind_keeps_a_raw_root_value() {
    let (gumbel, _) = expand_root(SearchKind::Gumbel, 0.25);
    assert!(
        (gumbel.root_raw_value() - 0.25).abs() < 1e-6,
        "the Gumbel kind retains the network's own root value, separate from W/N"
    );
    let (puct, _) = expand_root(SearchKind::Puct, 0.25);
    assert_eq!(
        puct.root_raw_value(),
        0.0,
        "the PUCT kind stores NO raw values — that is what keeps its per-worker pool \
         footprint where it was measured"
    );
}

// ── FLOOR / ENVELOPE ────────────────────────────────────────────────────────────

#[test]
fn the_kind_states_its_pool_envelope() {
    // Both bounds DERIVED from the pool's own constants, never transcribed (R98).
    assert_eq!(MAX_ARMED_SIMS, MAX_NODES / (4 * MAX_CHILDREN_PER_NODE));
    assert_eq!(
        MAX_ARMED_SIMS_GUMBEL,
        (MAX_NODES - MAX_ROOT_CHILDREN) / (4 * MAX_CHILDREN_PER_NODE)
    );
    // Compared through locals so clippy does not fold two consts into a literal truth: the
    // assertion is about the RELATION surviving a change to either constant, which is
    // exactly what a const-folded check would stop noticing.
    let (gumbel_ceiling, puct_ceiling) = (MAX_ARMED_SIMS_GUMBEL, MAX_ARMED_SIMS);
    assert!(
        gumbel_ceiling < puct_ceiling,
        "the Gumbel kind spends up to MAX_ROOT_CHILDREN slots on its root, so its ceiling \
         MUST be the lower of the two"
    );

    // The envelope's other term, stated rather than measured: the Gumbel kind allocates one
    // f32 per pool slot for the raw values, and the PUCT kind allocates none.
    assert_eq!(
        std::mem::size_of::<f32>() * MAX_NODES,
        4 * MAX_NODES,
        "the Gumbel kind's per-worker raw-value cost, derived from the pool size"
    );
}

#[test]
fn a_search_at_the_gumbel_ceiling_still_fits_the_pool() {
    // The bound's PURPOSE, checked arithmetically rather than by running a full-ceiling
    // search: the worst case is `4 * sims` expansions at the per-node cap, plus one root at
    // the root cap, and it must fit MAX_NODES. `finish_expansion` panics if it does not.
    let worst_case = 4 * MAX_ARMED_SIMS_GUMBEL * MAX_CHILDREN_PER_NODE + MAX_ROOT_CHILDREN;
    assert!(
        worst_case <= MAX_NODES,
        "a search armed at the Gumbel ceiling would need {worst_case} slots of {MAX_NODES}"
    );
}

/// The interior selector is WIRED, not merely present.
///
/// `mcts/parity_tests.rs` pins the interior score against Mctx's own `_prepare_argmax_input`,
/// which proves the arithmetic and says nothing about whether the descent calls it. This
/// drives the same tree under both kinds with the root pinned to one child — so the only
/// thing that can differ is selection BELOW the root — and requires the two visit
/// distributions to disagree.
#[test]
fn the_interior_selector_changes_where_the_visits_land() {
    fn drive(kind: SearchKind) -> Vec<u32> {
        // A NARROW board, deliberately: this test needs the two selectors to COMPETE over
        // children that get REVISITED, and a fixture wide enough to exercise the per-node cap
        // gives the drive more children than it has simulations — at which point both kinds
        // spend every sim on a fresh unvisited child in prior order and agree by exhaustion
        // rather than by running the same selector. The cap is `wide_board`'s subject; this
        // one's is the descent.
        let mut board = Board::new();
        board.set_legal_move_radius(2);
        board
            .apply_move(0, 0)
            .expect("(0,0) is legal on a fresh board");
        board.apply_move(1, 0).expect("(1,0) is legal beside it");
        let mut tree = MCTSTree::new(1.5);
        tree.configure_quiescence(false, 0.0);
        tree.configure_search(kind, 50.0, 0.1);
        tree.new_game(board);
        // A SKEWED policy: under a uniform one, PUCT and the improved-policy argmax can
        // agree by symmetry and the comparison would pass on a dead wire.
        let policy: Vec<f32> = (0..N_ACTIONS)
            .map(|i| (1.0 + (i % 7) as f32) / (4.0 * N_ACTIONS as f32))
            .collect();
        let leaves = tree.select_leaves(1).expect("root selects itself");
        assert_eq!(leaves.len(), 1);
        tree.expand_and_backup(std::slice::from_ref(&policy), &[0.1]);

        let forced = tree.pool[0].first_child;
        tree.set_forced_root_child(Some(forced))
            .expect("the root's own first child is in range");
        for i in 0..24 {
            let Ok(boards) = tree.select_leaves(1) else {
                break;
            };
            if boards.is_empty() {
                break;
            }
            // Values that vary per simulation, so completed-Q has something to complete
            // with and the two selectors have a reason to disagree.
            let value = 0.4 - 0.05 * (i % 5) as f32;
            tree.expand_and_backup(std::slice::from_ref(&policy), &[value]);
        }
        let node = &tree.pool[forced as usize];
        let first = node.first_child as usize;
        (first..first + node.n_children as usize)
            .map(|i| tree.pool[i].n_visits)
            .collect()
    }

    let puct = drive(SearchKind::Puct);
    let gumbel = drive(SearchKind::Gumbel);
    assert_eq!(
        puct.len(),
        gumbel.len(),
        "both kinds expand the same interior node, so the child counts must match"
    );
    assert!(
        puct.iter().sum::<u32>() > 0,
        "the drive placed no visits below the root — it measures nothing"
    );
    assert_ne!(
        puct, gumbel,
        "the two kinds put their interior visits in the SAME places. Either the Gumbel \
         descent is still running PUCT, or this fixture cannot tell the two selectors apart."
    );
}

/// THE ROUND'S WIDTH IS THE HALVING PHASE'S, exactly — measured on the tree rather than
/// inferred from a driven aggregate.
///
/// The first phase considers `m` candidates, so the first round is `m` descents wide; after
/// those `m` candidates are visited once each the schedule's considered level advances and
/// the next round is `m` wide again, and the widths only start halving when the level runs
/// past what a candidate has. This pins the FIRST two rounds, which is where an
/// off-by-one in the run-length read would show up, and the tail is covered by the driven
/// mean in `mantis-selfplay/tests/gumbel_round_batching.rs`.
#[test]
fn a_gumbel_round_is_exactly_the_halving_phase_wide() {
    const M: usize = 8;
    let board = wide_board();
    let mut tree = MCTSTree::new(1.5);
    tree.configure_quiescence(false, 0.0);
    tree.configure_search(SearchKind::Gumbel, 50.0, 0.1);
    tree.new_game(board);
    let policy = vec![1.0f32 / N_ACTIONS as f32; N_ACTIONS];
    let leaves = tree.select_leaves(1).expect("a fresh root selects itself");
    assert_eq!(leaves.len(), 1);
    tree.expand_and_backup(std::slice::from_ref(&policy), &[0.0]);

    let state = MctxRootState::new_seeded(&tree, M, 64, 20260909);

    let first = state.round_batch(&tree, 50.0, 0.1);
    assert_eq!(
        first.len(),
        M,
        "the first round must descend into every one of the {M} sampled candidates — a \
         narrower round is a run-length read that stopped early, and a wider one is reaching \
         candidates the schedule has not considered yet"
    );
    // Every candidate is a DISTINCT root child: a round that repeated one would spend the
    // phase on a single subtree.
    let mut sorted = first.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), M, "a round must not repeat a candidate");

    let boards = tree
        .select_leaves_forced(&first)
        .expect("no desync under a fresh root");
    assert_eq!(
        boards.len(),
        M,
        "one leaf per candidate: {M} forced descents into disjoint subtrees must yield {M} \
         distinct leaves"
    );
    let policies: Vec<Vec<f32>> = (0..boards.len()).map(|_| policy.clone()).collect();
    tree.expand_and_backup(&policies, &vec![0.0; boards.len()]);

    let second = state.round_batch(&tree, 50.0, 0.1);
    assert_eq!(
        second.len(),
        M,
        "the second round is still the full candidate set — Sequential Halving does not \
         halve until the schedule's considered level advances past a whole pass"
    );
}
