//! `tactics` — native Rust in-window-offense tactical proof solver.
//!
//! Ports the AND-OR threat-space proof skeleton onto the native `Board` with ZERO board clone
//! per node. IN-WINDOW-OFFENSE-ONLY: off-window defense belongs to the multi-window decoding
//! fix, so a WIN whose played move lands outside the perception window becomes UNKNOWN.
//!
//! SOUNDNESS INVARIANT: the net value head is NEVER read inside the search — a proof is only a
//! terminal backup or a stone-count shortcut, never a heuristic eval, and `eval.rs` orders moves
//! while reporting UNKNOWN. The `#[cfg(test)]` fuzz cross-checks every LOSS against an
//! independent exhaustive `brute_solve`. The static eval, the TT, net-policy ordering and the
//! quiet-move alpha-beta body are deferred; the proof core is threat-based.

pub mod eval;
pub mod ordering;
pub mod search;
pub mod tt;

use mantis_core::board::Board;

/// 3-valued proof result for the side-to-move (WIN/LOSS/UNKNOWN).
/// UNKNOWN = not determined within depth/budget (NOT a draw, NOT a proof).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// Side-to-move has a proven forced win in the explored (threat) subtree.
    Win,
    /// Side-to-move is in a proven forced loss in the explored subtree.
    Loss,
    /// Unresolved within depth/budget — never a proof.
    Unknown,
}

/// Mate-score base. A proven mate is encoded as `±(MATE - ply)` so a SHORTER forced win scores
/// higher, while the verdict depends only on the magnitude crossing `WIN_THRESHOLD`. Alpha-beta
/// never changes a proven conclusion: the root runs a FULL window, and a LOSS is concluded only
/// on a node whose candidate loop completed with no β-cutoff.
pub(crate) const MATE: i32 = 1_000_000;

/// Scores with magnitude >= this are mate-distance-encoded PROOFS; any bounded score below it is
/// heuristic, never a proof. The 1000-ply band keeps every realisable mate distance inside it.
pub(crate) const WIN_THRESHOLD: i32 = MATE - 1000;

/// Window sentinels strictly outside `[-MATE, MATE]`, so a full window cannot clip a mate score.
pub(crate) const POS_INF: i32 = MATE + 1000;
pub(crate) const NEG_INF: i32 = -(MATE + 1000);

/// Derive the 3-valued proof verdict from a scored-search value: a mate-magnitude score is only
/// ever produced by a sound proof path, so at the ROOT the magnitude alone is a sound verdict.
#[inline]
pub(crate) fn outcome_of(score: i32) -> Outcome {
    if score >= WIN_THRESHOLD {
        Outcome::Win
    } else if score <= -WIN_THRESHOLD {
        Outcome::Loss
    } else {
        Outcome::Unknown
    }
}

impl Outcome {
    /// Flip WIN<->LOSS; UNKNOWN stays UNKNOWN (negamax for HTTT compound turns).
    #[inline]
    pub fn negate(self) -> Self {
        match self {
            Outcome::Win => Outcome::Loss,
            Outcome::Loss => Outcome::Win,
            Outcome::Unknown => Outcome::Unknown,
        }
    }

    /// Int mapping: WIN=1, LOSS=-1, UNKNOWN=0.
    #[inline]
    pub fn to_i32(self) -> i32 {
        match self {
            Outcome::Win => 1,
            Outcome::Loss => -1,
            Outcome::Unknown => 0,
        }
    }
}

/// Node-budget meter (board expansions): `cap` ticks pass, the `cap+1`-th latches `exhausted`.
pub struct Budget {
    cap: u64,
    pub nodes: u64,
    pub exhausted: bool,
    /// Set whenever a node returns at the DEPTH horizon, so iterative deepening can stop on a
    /// search that resolved fully within depth.
    pub hit_horizon: bool,
}

impl Budget {
    pub fn new(cap: u64) -> Self {
        Budget { cap, nodes: 0, exhausted: false, hit_horizon: false }
    }

    /// Charge one node. Returns false (and latches `exhausted`) once over cap.
    #[inline]
    pub fn tick(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes > self.cap {
            self.exhausted = true;
            false
        } else {
            true
        }
    }
}

/// Solver configuration; `window_half`/`cand_cap` default to the 19-window band (9) and 40.
#[derive(Clone, Copy, Debug)]
pub struct TacticalConfig {
    /// Threat-guided candidate cap per node.
    pub cand_cap: usize,
    /// In-window offense guard: `Some(h)` suppresses a WIN whose played move is cheb-distance
    /// > `h` from the window center; `None` gives the full game-theoretic result.
    pub window_half: Option<i32>,
    /// Quiet-move body: `Some(d)` widens the NOT-IN-CHECK candidate set with every empty legal
    /// cell within cheb-distance `d` of a stone; when `d` covers the legal radius the set is the
    /// full legal set, so the LOSS guard's exhaustiveness branch fires. In-check nodes are NOT
    /// widened — there the threat-only set is already complete.
    pub neighbor_dist: Option<i32>,
}

impl Default for TacticalConfig {
    fn default() -> Self {
        TacticalConfig { cand_cap: 40, window_half: Some(9), neighbor_dist: None }
    }
}

/// Result of a `prove` call. `line` is the principal variation, populated for WIN, whose first
/// two entries are the side-to-move's two stones for a 2-stone-turn forcing win.
#[derive(Clone, Debug)]
pub struct ProofResult {
    pub result: Outcome,
    pub line: Vec<(i32, i32)>,
    pub nodes: u64,
    pub budget_exhausted: bool,
}

/// The native tactical solver: NET-FREE proof core, the net only ever ORDERS.
pub struct TacticalSolver {
    config: TacticalConfig,
}

impl TacticalSolver {
    pub fn new(config: TacticalConfig) -> Self {
        TacticalSolver { config }
    }

    /// Try to prove the side-to-move at `board`, cloning it ONCE per call, never per node.
    pub fn prove(&self, board: &Board, max_depth: u32, node_budget: u64) -> ProofResult {
        let mut scratch = board.clone();
        self.prove_in_place(&mut scratch, max_depth, node_budget)
    }

    /// Zero-clone entry for the deploy root hook: searches in place and restores `board`.
    pub fn prove_in_place(
        &self,
        board: &mut Board,
        max_depth: u32,
        node_budget: u64,
    ) -> ProofResult {
        let mut budget = Budget::new(node_budget);
        let mut tt = tt::ProofTt::new();
        let mut ordering = ordering::OrderingState::new();
        // Iterative-deepening + aspiration root driver: each accepted iteration resolves to an
        // EXACT root value, so `outcome_of`'s magnitude mapping is a sound proof.
        let scored = search::solve_root(
            board,
            max_depth as i32,
            &mut budget,
            &self.config,
            &mut tt,
            &mut ordering,
        );

        let mut result = outcome_of(scored.score);
        let mut line = scored.line;

        // BOTH stones of the turn must be in-window, not just the first: the reachability-
        // relevant cell is the COMPLETING stone that lands the win.
        if result == Outcome::Win {
            if let Some(half) = self.config.window_half {
                if line.iter().take(2).any(|&m| is_off_window(board, m, half)) {
                    result = Outcome::Unknown;
                    line = Vec::new();
                }
            }
        }

        ProofResult {
            result,
            line,
            nodes: budget.nodes,
            budget_exhausted: budget.exhausted,
        }
    }
}

/// True if `mv` is off the single GLOBAL perception window: cheb-distance from the
/// bbox-centroid window center exceeds `half`, mirroring the engine's own off-window test.
pub(crate) fn is_off_window(board: &Board, mv: (i32, i32), half: i32) -> bool {
    let (cq, cr) = board.window_center();
    (mv.0 - cq).abs().max((mv.1 - cr).abs()) > half
}
