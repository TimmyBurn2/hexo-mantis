// R8 justify: ONE suite-v2 section — "Gumbel as a search kind behind the seam" — and a
// section is a unit. Its detector, its witnesses and its stated floor/envelope are the three
// halves of a single claim, and the envelope's numbers are only meaningful beside the witness
// that spends them.
//! ⊕ GUMBEL-REPAIR-1 item 9 — SUITE V2 SECTION: Gumbel as a search kind.
//!
//! Suite v2's existing sections (`SEAM_V1_DESIGN.md` §3) are per-ARCH: perf floor, memory
//! envelope, config partition. This is the same shape applied to a SEARCH KIND, which is the
//! other thing behind the seam that a run's strength depends on and that nothing was making
//! state its terms.
//!
//! RUST-SIDE, and the reason is where the object lives. The arch sections are Python because
//! an arch is a Python object; a search dialect is a Rust one, reachable from Python only
//! through a config key whose own wire is pinned separately
//! (`tests/config/test_every_key_has_consumer.py`, `tests/selfplay/test_pool_hparams.py`).
//!
//! - **DETECTOR** — the flag. Which dialect a tree runs is READABLE off the tree, and the
//!   terms that follow from it move with it.
//! - **WITNESS** — the parity vectors live in `mctx_parity.rs` and `mcts/parity_tests.rs`
//!   (Mctx's own numbers). Here: a driven root expands its FULL legal set under the Mctx
//!   dialect, and the exported target puts mass on all of it and sums to 1.
//! - **FLOOR / ENVELOPE** — the dialect's pool terms, stated and derived rather than
//!   transcribed.

use std::sync::{Mutex, MutexGuard};

use mantis_core::Board;
use mantis_search::{
    take_omitted_prior_stats, GumbelVariant, MCTSTree, MAX_ARMED_SIMS, MAX_ARMED_SIMS_MCTX,
    MAX_CHILDREN_PER_NODE, MAX_NODES, MAX_ROOT_CHILDREN,
};

/// Board stride for a 19-window encoding with a pass slot — the shape every in-src MCTS
/// fixture uses.
const N_ACTIONS: usize = 19 * 19 + 1;

/// A radius-8 board with two stones: 232 legal moves, comfortably over the 192 per-node cap,
/// so the two dialects are DISTINGUISHABLE at the root. Derived by construction, not
/// asserted as a constant — the count is a property of the geometry, not of this test.
fn wide_board() -> Board {
    let mut board = Board::new();
    board.set_legal_move_radius(8);
    board
        .apply_move(0, 0)
        .expect("(0,0) is legal on a fresh board");
    board.apply_move(1, 0).expect("(1,0) is legal beside it");
    board
}

/// Serialises every root expansion in this binary.
///
/// `take_omitted_prior_stats` reads PROCESS-GLOBAL counters, and cargo runs this file's
/// tests on threads of one process. Without this, the omitted-mass witness measures its own
/// expansion plus whichever sibling happened to expand inside its window — it first read
/// "2 truncating expansions" for one truncating expansion, which is a wrong number rather
/// than a flaky one. The lock is over the WINDOW, not just the call, so the measuring test
/// holds it across take -> expand -> take.
static EXPAND_LOCK: Mutex<()> = Mutex::new(());

fn expand_lock() -> MutexGuard<'static, ()> {
    // A poisoned lock means a sibling test panicked; the counters it left are exactly what
    // the next `take` clears, so recovering is correct and hiding the panic is not — the
    // sibling reports its own failure.
    EXPAND_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Expand the root once with a uniform policy and a stated leaf value.
fn expand_root(variant: GumbelVariant, value: f32) -> (MCTSTree, Board) {
    let _guard = expand_lock();
    expand_root_unlocked(variant, value)
}

/// `expand_root` for a caller that already holds `EXPAND_LOCK`.
fn expand_root_unlocked(variant: GumbelVariant, value: f32) -> (MCTSTree, Board) {
    let board = wide_board();
    let mut tree = MCTSTree::new(1.5);
    // Quiescence OFF so the backed-up value is the supplied one and the raw-value witness
    // reads what it was handed rather than a corrected version of it.
    tree.configure_quiescence(false, 0.0);
    tree.configure_gumbel(variant, true, 50.0, 0.1);
    tree.new_game(board.clone());
    let leaves = tree.select_leaves(1).expect("a fresh root selects itself");
    assert_eq!(leaves.len(), 1, "root expansion selects exactly the root");
    let policy = vec![1.0f32 / N_ACTIONS as f32; N_ACTIONS];
    tree.expand_and_backup(&[policy], &[value]);
    (tree, board)
}

// ── DETECTOR ────────────────────────────────────────────────────────────────────

#[test]
fn the_dialect_is_readable_off_the_tree_and_carries_its_root_cap() {
    let mut tree = MCTSTree::new(1.5);
    assert_eq!(
        tree.gumbel_variant(),
        GumbelVariant::Legacy,
        "a tree that was never configured must run the SHIPPED dialect — a default of \
         `Mctx` would arm the corrected arm by omission, which R345(d) forbids"
    );
    assert_eq!(tree.root_children_cap(), MAX_CHILDREN_PER_NODE);

    tree.configure_gumbel(GumbelVariant::Mctx, true, 50.0, 0.1);
    assert_eq!(tree.gumbel_variant(), GumbelVariant::Mctx);
    assert_eq!(
        tree.root_children_cap(),
        MAX_ROOT_CHILDREN,
        "the root cap must follow the dialect — a dialect that reads `Mctx` while its root \
         still truncates at 192 is the deviation still live under a name that says it is not"
    );

    tree.configure_gumbel(GumbelVariant::Legacy, true, 50.0, 1.0);
    assert_eq!(tree.root_children_cap(), MAX_CHILDREN_PER_NODE, "and back");
}

/// THE DIALECT IS INERT WITHOUT `gumbel_mcts`, and that is the seam the packet draws:
/// *"behind `selfplay.gumbel_mcts` … the PUCT path is untouched"*. A config carrying
/// `gumbel_variant: mctx` with `gumbel_mcts: false` must run PUCT exactly as before —
/// otherwise a key named for Gumbel silently moves the arm run6 is the control for.
#[test]
fn the_dialect_does_nothing_while_gumbel_is_off() {
    let mut tree = MCTSTree::new(1.5);
    tree.configure_gumbel(GumbelVariant::Mctx, false, 50.0, 0.1);
    assert_eq!(
        tree.gumbel_variant(),
        GumbelVariant::Legacy,
        "an inert dialect must READ inert — a tree reporting `Mctx` while Gumbel is off \
         would make every downstream branch take the corrected arm on a PUCT run"
    );
    assert_eq!(tree.root_children_cap(), MAX_CHILDREN_PER_NODE);

    let board = wide_board();
    tree.configure_quiescence(false, 0.0);
    tree.new_game(board);
    let leaves = tree.select_leaves(1).expect("root selects itself");
    assert_eq!(leaves.len(), 1);
    tree.expand_and_backup(&[vec![1.0f32 / N_ACTIONS as f32; N_ACTIONS]], &[0.25]);
    assert_eq!(
        tree.root_n_children(),
        MAX_CHILDREN_PER_NODE,
        "the root still truncates at the per-node cap"
    );
    assert_eq!(tree.root_raw_value(), 0.0, "and keeps no raw values");
}

#[test]
fn the_config_spelling_round_trips_and_an_unknown_dialect_is_refused() {
    for v in [GumbelVariant::Legacy, GumbelVariant::Mctx] {
        assert_eq!(GumbelVariant::from_config_str(v.as_config_str()), Some(v));
    }
    // REFUSED, not defaulted. A dialect that falls back to `legacy` on a typo is the
    // silent-fallback class LAW-11 closes.
    for bad in ["", "MCTX", "mctx ", "gumbel", "legacy_clamped"] {
        assert_eq!(
            GumbelVariant::from_config_str(bad),
            None,
            "{bad:?} must not parse as a dialect"
        );
    }
}

// ── WITNESS ─────────────────────────────────────────────────────────────────────

#[test]
fn the_mctx_root_expands_its_full_legal_set_and_the_legacy_root_does_not() {
    let legal = wide_board().legal_moves().len();
    assert!(
        legal > MAX_CHILDREN_PER_NODE,
        "the fixture must have more legal moves than the per-node cap or the two dialects \
         are indistinguishable here (got {legal})"
    );

    let _guard = expand_lock();
    let _ = take_omitted_prior_stats();
    let (mctx, _) = expand_root_unlocked(GumbelVariant::Mctx, 0.25);
    let (_, omitted_expansions, _) = take_omitted_prior_stats();
    assert_eq!(
        mctx.root_n_children(),
        legal,
        "the Mctx root must hold EVERY legal move: Gumbel-Top-k perturbs every action's \
         logit, so an action outside the root's child set can never be sampled however \
         large its draw"
    );
    assert_eq!(
        omitted_expansions, 0,
        "R345(b)(5)'s witness, read at the root where it is achievable: no prior mass left \
         the root expansion"
    );

    let _ = take_omitted_prior_stats();
    let (legacy, _) = expand_root_unlocked(GumbelVariant::Legacy, 0.25);
    let (mass_micros, omitted_expansions, _) = take_omitted_prior_stats();
    assert_eq!(
        legacy.root_n_children(),
        MAX_CHILDREN_PER_NODE,
        "the legacy root is unchanged — this is the arm run6 runs and R345(d) keeps still"
    );
    assert!(
        omitted_expansions == 1 && mass_micros > 0,
        "the legacy root drops prior mass and the telemetry says so ({omitted_expansions} \
         truncating expansions, {mass_micros} micros) — if it did not, the comparison above \
         would prove nothing"
    );
}

#[test]
fn the_exported_target_covers_the_full_legal_set_and_sums_to_one() {
    let legal = wide_board().legal_moves().len();
    let (tree, _) = expand_root(GumbelVariant::Mctx, 0.25);

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
fn only_the_mctx_dialect_keeps_a_raw_root_value() {
    let (mctx, _) = expand_root(GumbelVariant::Mctx, 0.25);
    assert!(
        (mctx.root_raw_value() - 0.25).abs() < 1e-6,
        "the Mctx dialect retains the network's own root value, separate from W/N"
    );
    let (legacy, _) = expand_root(GumbelVariant::Legacy, 0.25);
    assert_eq!(
        legacy.root_raw_value(),
        0.0,
        "the legacy dialect stores NO raw values — that is what keeps its per-worker pool \
         footprint exactly where run6's control arm measured it"
    );
}

// ── FLOOR / ENVELOPE ────────────────────────────────────────────────────────────

#[test]
fn the_dialect_states_its_pool_envelope() {
    // Both bounds DERIVED from the pool's own constants, never transcribed (R98).
    assert_eq!(MAX_ARMED_SIMS, MAX_NODES / (4 * MAX_CHILDREN_PER_NODE));
    assert_eq!(
        MAX_ARMED_SIMS_MCTX,
        (MAX_NODES - MAX_ROOT_CHILDREN) / (4 * MAX_CHILDREN_PER_NODE)
    );
    // Compared through locals so clippy does not fold two consts into a literal truth: the
    // assertion is about the RELATION surviving a change to either constant, which is
    // exactly what a const-folded check would stop noticing.
    let (mctx_ceiling, legacy_ceiling) = (MAX_ARMED_SIMS_MCTX, MAX_ARMED_SIMS);
    assert!(
        mctx_ceiling < legacy_ceiling,
        "the Mctx dialect spends up to MAX_ROOT_CHILDREN slots on its root, so its ceiling \
         MUST be the lower of the two"
    );
    // The legacy ceiling is UNMOVED. Tightening the bound every existing config validates
    // against, for a dialect nothing arms, would be a mint-surface change.
    assert_eq!(
        MAX_ARMED_SIMS, 1302,
        "the legacy ceiling AUDIT-1 F-21 derived"
    );

    // The envelope's other term, stated rather than measured: the Mctx dialect allocates one
    // f32 per pool slot for the raw values, and the legacy dialect allocates none.
    assert_eq!(
        std::mem::size_of::<f32>() * MAX_NODES,
        4_000_000,
        "the Mctx dialect's per-worker raw-value cost, derived from the pool size"
    );
}

#[test]
fn a_search_at_the_mctx_ceiling_still_fits_the_pool() {
    // The bound's PURPOSE, checked arithmetically rather than by running a 1216-sim search:
    // the worst case is `4 * sims` expansions at the per-node cap, plus one root at the root
    // cap, and it must fit MAX_NODES. `finish_expansion` panics if it does not.
    let worst_case = 4 * MAX_ARMED_SIMS_MCTX * MAX_CHILDREN_PER_NODE + MAX_ROOT_CHILDREN;
    assert!(
        worst_case <= MAX_NODES,
        "a search armed at the Mctx ceiling would need {worst_case} slots of {MAX_NODES}"
    );
}

/// The interior selector is WIRED, not merely present.
///
/// `mcts/parity_tests.rs` pins the interior score against Mctx's own `_prepare_argmax_input`,
/// which proves the arithmetic and says nothing about whether the descent calls it. This
/// drives the same tree under both dialects with the root pinned to one child — so the only
/// thing that can differ is selection BELOW the root — and requires the two visit
/// distributions to disagree.
#[test]
fn the_interior_selector_changes_where_the_visits_land() {
    fn drive(variant: GumbelVariant) -> Vec<u32> {
        let _guard = expand_lock();
        let board = wide_board();
        let mut tree = MCTSTree::new(1.5);
        tree.configure_quiescence(false, 0.0);
        tree.configure_gumbel(variant, true, 50.0, 0.1);
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

    let legacy = drive(GumbelVariant::Legacy);
    let mctx = drive(GumbelVariant::Mctx);
    assert_eq!(
        legacy.len(),
        mctx.len(),
        "both dialects expand the same interior node, so the child counts must match"
    );
    assert!(
        legacy.iter().sum::<u32>() > 0,
        "the drive placed no visits below the root — it measures nothing"
    );
    assert_ne!(
        legacy, mctx,
        "the two dialects put their interior visits in the SAME places. Either the Mctx \
         descent is still running PUCT — the deviation live under a name that says it is \
         not — or this fixture cannot tell the two selectors apart."
    );
}
