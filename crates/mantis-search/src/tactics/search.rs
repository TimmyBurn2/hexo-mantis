// Exceeds the 300-line soft cap: the scored α-β proof core, its 3-valued reference oracle and
// the in-src soundness fuzz (which reaches pub(crate) solve*) are one auditable unit.
//! Search core — iterative AND-OR threat-space proof over HTTT compound turns.
//!
//! Flip-aware negamax, zero clone per node: a child result is negated ONLY when the to-move
//! player flipped, which in HTTT is every 2 plies. DEFERRED — the threat/double-threat core
//! only: a position whose only escape or win is a quiet move returns UNKNOWN, never a proof.

use fxhash::FxHashSet;

use mantis_core::board::Board;

use super::eval::heuristic_leaf;
use super::ordering::{candidates, order_moves, OrderingState};
use super::tt::{Bound, ProofTt};
use super::{outcome_of, Budget, Outcome, TacticalConfig, MATE, NEG_INF, POS_INF, WIN_THRESHOLD};

/// A scored node value: a proven mate is `±(MATE - ply)` (magnitude >= `WIN_THRESHOLD`), a
/// heuristic or unresolved node is bounded strictly below it. `line` is the PV, WIN only.
pub struct Scored {
    pub score: i32,
    pub line: Vec<(i32, i32)>,
}

impl Scored {
    /// A bounded leaf value, clamped so `outcome_of` can NEVER read it as a proof.
    #[inline]
    fn heuristic(score: i32) -> Self {
        Scored { score: clamp_heuristic(score), line: Vec::new() }
    }
}

/// Clamp a heuristic strictly inside `(-WIN_THRESHOLD, WIN_THRESHOLD)`, so only `solve`'s
/// proof paths may emit a mate-magnitude score.
#[inline]
pub(crate) fn clamp_heuristic(score: i32) -> i32 {
    score.clamp(-(WIN_THRESHOLD - 1), WIN_THRESHOLD - 1)
}

/// Scored α-β AND-OR threat-space proof for `board.current_player`; the 3-valued verdict is
/// `outcome_of(score)` at the ROOT, full-window.
///
/// SOUNDNESS — α-β prunes ONLY and the proof is NET-FREE: a mate-magnitude score comes only
/// from a sound proof path, a LOSS only from a loop that ran to completion with NO β-cutoff,
/// and the proven-WIN cutoff fires before any sibling is searched with `β <= -WIN_THRESHOLD`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    board: &mut Board,
    depth_left: i32,
    ply: i32,
    mut alpha: i32,
    beta: i32,
    budget: &mut Budget,
    cfg: &TacticalConfig,
    tt: &mut ProofTt,
    ordering: &mut OrderingState,
) -> Scored {
    if !budget.tick() {
        return Scored::heuristic(0); // budget out => UNKNOWN, never a proof
    }

    // (1) Terminal: the engine-owned CF-1 sign is the ONLY proof sign; mate distance = `ply`.
    if board.check_win() {
        let score = if board.terminal_value_to_move() > 0.0 { MATE - ply } else { -(MATE - ply) };
        return Scored { score, line: Vec::new() };
    }

    let stm = board.current_player;
    let opp = stm.other();

    // (2) Immediate-win shortcut: a sound stone-count proof, yielding the winning cell.
    if board.count_winning_moves(stm) >= 1 {
        let line = board.first_winning_move(stm).map_or_else(Vec::new, |m| vec![m]);
        return Scored { score: MATE - ply, line };
    }

    // (3) Double-threat LOSS shortcut (mr==1 only): stm places one stone before the flip and
    //     opp then has >=2 standing win-in-1 cells, so stm blocks at most one.
    if board.moves_remaining == 1 && board.count_winning_moves(opp) >= 2 {
        return Scored { score: -(MATE - ply), line: Vec::new() };
    }

    // (4) TT probe — only a PROVEN LOSS is trusted: a WIN hit has an empty PV and could
    //     truncate the override line.
    let key = (board.zobrist_hash, stm as i8, board.moves_remaining);
    if let Some(score) = tt.get_loss_proof(key, ply) {
        debug_assert!(score <= -WIN_THRESHOLD, "TT proof probe must decode a mate-magnitude LOSS");
        return Scored { score, line: Vec::new() };
    }

    if depth_left <= 0 {
        budget.hit_horizon = true; // a real depth truncation — deepening could help
        return Scored::heuristic(heuristic_leaf(board)); // horizon => non-proof leaf
    }

    // `in_check` is the prune-symmetry premise of the LOSS conclusion below, captured at the
    // node state before descent.
    let in_check = board.count_winning_moves(opp) >= 1;
    let mut moves = candidates(board, stm, opp, cfg.cand_cap, cfg.neighbor_dist);
    if moves.is_empty() {
        return Scored::heuristic(heuristic_leaf(board)); // quiet => cannot prove
    }
    let moves_len = moves.len();
    // Built BEFORE ordering so the recall verify's dropped-move set is order-independent.
    let searched: FxHashSet<(i32, i32)> = moves.iter().copied().collect();
    // Best-first ORDERING — a permutation only: earlier cutoffs, identical conclusions.
    order_moves(&mut moves, tt.get_best_move(key), ply, ordering);

    // Negamax-α-β OR-node. A proven-WIN cutoff fires at `best >= WIN_THRESHOLD` and the
    // generic one at `alpha >= beta`; either returns `best` as a BOUND, skipping the LOSS logic.
    let mut best = NEG_INF;
    let mut best_line: Vec<(i32, i32)> = Vec::new();
    let mut cutoff = false;

    for (idx, &(q, r)) in moves.iter().enumerate() {
        let node_player = board.current_player; // == stm
        let diff = match board.apply_move_tracked(q, r) {
            Ok(d) => d,
            Err(_) => continue,
        };
        // Flip-aware negamax: negate the child value and flip the window ONLY on a flip.
        let flipped = board.current_player != node_player;
        let dchild = depth_left - 1;

        // PVS + LMR, pruning/ordering ONLY. idx 0 is the PV (full window, full depth); later
        // moves get a null-window scout, LMR-reduced when late/quiet/not-in-check.
        let (rc, child_line) = if idx == 0 {
            let (ca, cb) = if flipped { (-beta, -alpha) } else { (alpha, beta) };
            let c = solve(board, dchild, ply + 1, ca, cb, budget, cfg, tt, ordering);
            (if flipped { -c.score } else { c.score }, c.line)
        } else {
            let reduce = lmr_reduction(idx, depth_left, in_check);
            let (na, nb) = if flipped { (-alpha - 1, -alpha) } else { (alpha, alpha + 1) };
            let c = solve(board, dchild - reduce, ply + 1, na, nb, budget, cfg, tt, ordering);
            let mut rc = if flipped { -c.score } else { c.score };
            let mut line = c.line;
            // Re-search at full depth+window whenever the scout could matter. The one reduced
            // value accepted is a reduced-PROVEN result below α, which is verdict-irrelevant.
            let need_full = if reduce > 0 {
                rc > alpha || outcome_of(rc) == Outcome::Unknown
            } else {
                rc > alpha && rc < beta
            };
            if need_full {
                let (fa, fb) = if flipped { (-beta, -alpha) } else { (alpha, beta) };
                let c = solve(board, dchild, ply + 1, fa, fb, budget, cfg, tt, ordering);
                rc = if flipped { -c.score } else { c.score };
                line = c.line;
            }
            (rc, line)
        };

        if rc > best {
            best = rc;
            // PV: same player continued (mr was 2) => append the child line; flipped => the
            // move alone.
            let mut line = vec![(q, r)];
            if !flipped {
                line.extend(child_line.iter().copied());
            }
            best_line = line;
        }
        board.undo_move(diff);

        if best > alpha {
            alpha = best;
        }
        if best >= WIN_THRESHOLD || alpha >= beta {
            // Cutoff move (WIN or β): reward it for ordering future siblings.
            ordering.record_cutoff(ply, (q, r), depth_left);
            cutoff = true;
            break;
        }
    }

    // No candidate could be applied (degenerate) => UNKNOWN, never a false LOSS.
    if best == NEG_INF {
        return Scored::heuristic(0);
    }

    // Proven WIN (`best` is a mate magnitude): exact lower bound is already a win.
    if best >= WIN_THRESHOLD {
        // ORDERING hint only (is_proof=false; never returned as a verdict — a WIN
        // is always reconstructed). The winning move ordered first next visit.
        tt.store_bound(key, best, ply, Bound::Lower, best_line.first().copied(), depth_left);
        return Scored { score: best, line: best_line };
    }
    // β-cutoff with a non-win `best`: a fail-high BOUND, so the LOSS-proof logic below must NOT
    // run. It is strictly above `-WIN_THRESHOLD`, so it cannot be misread as a proven LOSS.
    if cutoff {
        tt.store_bound(key, best, ply, Bound::Lower, best_line.first().copied(), depth_left);
        return Scored { score: best, line: best_line };
    }

    // Loop ran to completion with no cutoff => `best` is the EXACT node value.
    //   best >  -WIN_THRESHOLD  : some candidate did not lose => UNKNOWN.
    //   best <= -WIN_THRESHOLD  : EVERY candidate loses => candidate (R3) logic.
    if outcome_of(best) != Outcome::Loss {
        return Scored::heuristic(best);
    }

    // R3 LOSS-COMPLETENESS GUARD. "Every CANDIDATE loses" proves "every LEGAL move loses" only
    // when the set covered all escapes: `in_check && moves_len < cand_cap` (prune symmetry; the
    // `<` is deliberately PESSIMISTIC, do NOT relax to `<=`) or `moves_len >= legal_move_count()`.
    let loss_complete =
        (in_check && moves_len < cfg.cand_cap) || moves_len >= board.legal_move_count();
    if loss_complete {
        tt.store_loss_proof(key, best, ply, depth_left); // proven, game-theoretic — cache it
        return Scored { score: best, line: Vec::new() };
    }

    // RECALL-PRESERVING VERIFY: certifying a LOSS needs the DROPPED legal moves too, searched
    // with a FULL window (NO α-β pruning); `vbest` tracks the slowest loss for mate distance.
    if cfg.neighbor_dist.is_some() {
        let mut vbest = best;
        for (q, r) in board.legal_moves() {
            if searched.contains(&(q, r)) {
                continue;
            }
            let node_player = board.current_player;
            let diff = match board.apply_move_tracked(q, r) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let flipped = board.current_player != node_player;
            // Full window, full depth, NO PVS/LMR — the verify preserves EXACT
            // recall for the LOSS certification (the soundness-critical path).
            let child = solve(board, depth_left - 1, ply + 1, NEG_INF, POS_INF, budget, cfg, tt, ordering);
            let rc = if flipped { -child.score } else { child.score };
            if rc >= WIN_THRESHOLD {
                let mut line = vec![(q, r)];
                if !flipped {
                    line.extend(child.line.iter().copied());
                }
                board.undo_move(diff);
                return Scored { score: rc, line };
            }
            board.undo_move(diff);
            if outcome_of(rc) != Outcome::Loss {
                return Scored::heuristic(0); // a dropped move escapes the loss
            }
            if rc > vbest {
                vbest = rc;
            }
        }
        // Every reduced candidate AND every dropped legal move loses => certified.
        tt.store_loss_proof(key, vbest, ply, depth_left);
        return Scored { score: vbest, line: Vec::new() };
    }

    Scored::heuristic(0) // candidate set incomplete, verify disabled -> cannot prove LOSS
}

/// LMR depth reduction for a late (`idx >= 6`) non-PV candidate at `depth_left >= 4`, never in
/// check. VERDICT-EXACT: `solve` re-searches at full depth whenever a reduced child matters.
#[inline]
fn lmr_reduction(idx: usize, depth_left: i32, in_check: bool) -> i32 {
    if !in_check && idx >= 6 && depth_left >= 4 {
        1
    } else {
        0
    }
}

/// Iterative-deepening + aspiration ROOT driver over `1..=max_depth`, reusing TT and ordering
/// state and stopping at a proven mate or an exhausted budget. Every accepted iteration
/// resolves to an EXACT value, so `outcome_of(score)` is a sound verdict.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_root(
    board: &mut Board,
    max_depth: i32,
    budget: &mut Budget,
    cfg: &TacticalConfig,
    tt: &mut ProofTt,
    ordering: &mut OrderingState,
) -> Scored {
    let mut result = Scored::heuristic(0);
    let mut have = false;
    let mut last = 0i32;
    let mut depth = 1;
    while depth <= max_depth {
        budget.hit_horizon = false; // track whether THIS iteration was depth-truncated
        let s = aspiration_search(board, depth, last, have, budget, cfg, tt, ordering);
        if budget.exhausted {
            if !have {
                result = s; // first iteration starved: best effort
            }
            break;
        }
        result = s;
        last = result.score;
        have = true;
        if outcome_of(result.score) != Outcome::Unknown {
            break; // proven WIN/LOSS — final; deeper search cannot change it
        }
        if !budget.hit_horizon {
            break; // tree fully resolved within depth — deepening cannot change it
        }
        depth += 1;
    }
    result
}

/// One aspiration-windowed root search at `depth`; a fail-low/high widens that side to ±∞ and
/// re-searches, so the RETURNED value is exact and never a clipped bound.
#[allow(clippy::too_many_arguments)]
fn aspiration_search(
    board: &mut Board,
    depth: i32,
    last: i32,
    have: bool,
    budget: &mut Budget,
    cfg: &TacticalConfig,
    tt: &mut ProofTt,
    ordering: &mut OrderingState,
) -> Scored {
    const W: i32 = 64; // aspiration half-width (heuristic units)
    let (mut alpha, mut beta) = (NEG_INF, POS_INF);
    if have && last.abs() < WIN_THRESHOLD {
        alpha = (last - W).max(NEG_INF);
        beta = (last + W).min(POS_INF);
    }
    loop {
        let s = solve(board, depth, 0, alpha, beta, budget, cfg, tt, ordering);
        if budget.exhausted {
            return s; // starved: caller keeps the last completed result
        }
        if s.score <= alpha && alpha > NEG_INF {
            alpha = NEG_INF; // fail low: widen down, re-search exact
            continue;
        }
        if s.score >= beta && beta < POS_INF {
            beta = POS_INF; // fail high: widen up, re-search exact
            continue;
        }
        return s; // strictly in-window => exact value
    }
}

/// 3-VALUED REFERENCE ORACLE (test-only): the pre-increment-1 proof core, verbatim, kept as
/// the verdict-invariance target — α-β may never change a verdict.
#[cfg(test)]
pub(crate) struct Solved3 {
    pub outcome: Outcome,
    pub line: Vec<(i32, i32)>,
}

#[cfg(test)]
impl Solved3 {
    #[inline]
    fn unknown() -> Self {
        Solved3 { outcome: Outcome::Unknown, line: Vec::new() }
    }
}

#[cfg(test)]
pub(crate) fn solve_3valued(
    board: &mut Board,
    depth_left: i32,
    budget: &mut Budget,
    cfg: &TacticalConfig,
    tt: &mut ProofTt,
) -> Solved3 {
    if !budget.tick() {
        return Solved3::unknown();
    }
    if board.check_win() {
        let outcome =
            if board.terminal_value_to_move() > 0.0 { Outcome::Win } else { Outcome::Loss };
        return Solved3 { outcome, line: Vec::new() };
    }
    let stm = board.current_player;
    let opp = stm.other();
    if board.count_winning_moves(stm) >= 1 {
        let line = board.first_winning_move(stm).map_or_else(Vec::new, |m| vec![m]);
        return Solved3 { outcome: Outcome::Win, line };
    }
    if board.moves_remaining == 1 && board.count_winning_moves(opp) >= 2 {
        return Solved3 { outcome: Outcome::Loss, line: Vec::new() };
    }
    let key = (board.zobrist_hash, stm as i8, board.moves_remaining);
    if tt.get_loss_proof(key, 0).is_some() {
        return Solved3 { outcome: Outcome::Loss, line: Vec::new() };
    }
    if depth_left <= 0 {
        return Solved3::unknown();
    }
    let in_check = board.count_winning_moves(opp) >= 1;
    let moves = candidates(board, stm, opp, cfg.cand_cap, cfg.neighbor_dist);
    if moves.is_empty() {
        return Solved3::unknown();
    }
    let moves_len = moves.len();
    let searched: FxHashSet<(i32, i32)> = moves.iter().copied().collect();
    let mut saw_unknown = false;
    for &(q, r) in &moves {
        let node_player = board.current_player;
        let diff = match board.apply_move_tracked(q, r) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let child = solve_3valued(board, depth_left - 1, budget, cfg, tt);
        let flipped = board.current_player != node_player;
        let rc = if flipped { child.outcome.negate() } else { child.outcome };
        if rc == Outcome::Win {
            let mut line = vec![(q, r)];
            if !flipped {
                line.extend(child.line.iter().copied());
            }
            board.undo_move(diff);
            return Solved3 { outcome: Outcome::Win, line };
        }
        board.undo_move(diff);
        if rc == Outcome::Unknown {
            saw_unknown = true;
        }
    }
    if saw_unknown {
        return Solved3::unknown();
    }
    let loss_complete =
        (in_check && moves_len < cfg.cand_cap) || moves_len >= board.legal_move_count();
    if loss_complete {
        tt.store_loss_proof(key, -MATE, 0, depth_left);
        return Solved3 { outcome: Outcome::Loss, line: Vec::new() };
    }
    if cfg.neighbor_dist.is_some() {
        for (q, r) in board.legal_moves() {
            if searched.contains(&(q, r)) {
                continue;
            }
            let node_player = board.current_player;
            let diff = match board.apply_move_tracked(q, r) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let child = solve_3valued(board, depth_left - 1, budget, cfg, tt);
            let flipped = board.current_player != node_player;
            let rc = if flipped { child.outcome.negate() } else { child.outcome };
            if rc == Outcome::Win {
                let mut line = vec![(q, r)];
                if !flipped {
                    line.extend(child.line.iter().copied());
                }
                board.undo_move(diff);
                return Solved3 { outcome: Outcome::Win, line };
            }
            board.undo_move(diff);
            if rc == Outcome::Unknown {
                return Solved3::unknown();
            }
        }
        tt.store_loss_proof(key, -MATE, 0, depth_left);
        return Solved3 { outcome: Outcome::Loss, line: Vec::new() };
    }
    Solved3::unknown()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mantis_core::board::{Cell, Player};

    const WIN: Outcome = Outcome::Win;
    const LOSS: Outcome = Outcome::Loss;

    fn solver() -> super::super::TacticalSolver {
        // window_half=None for the core proof tests (offense-guard tested
        // separately); cand_cap matches the reference candidate cap.
        super::super::TacticalSolver::new(TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None })
    }

    // Independent exhaustive oracle: ALL legal moves, no TT, early-exit on WIN.
    fn brute_solve(board: &mut Board, depth: i32, budget: &mut Budget) -> Outcome {
        if !budget.tick() {
            return Outcome::Unknown;
        }
        if board.check_win() {
            return if board.terminal_value_to_move() > 0.0 { WIN } else { LOSS };
        }
        let stm = board.current_player;
        if board.count_winning_moves(stm) >= 1 {
            return WIN;
        }
        if depth <= 0 {
            return Outcome::Unknown;
        }
        let mut saw_unknown = false;
        for (q, r) in board.legal_moves() {
            let node_player = board.current_player;
            let diff = match board.apply_move_tracked(q, r) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let raw = brute_solve(board, depth - 1, budget);
            let flipped = board.current_player != node_player;
            let rc = if flipped { raw.negate() } else { raw };
            board.undo_move(diff);
            if rc == WIN {
                return WIN;
            }
            if rc == Outcome::Unknown {
                saw_unknown = true;
            }
        }
        if saw_unknown { Outcome::Unknown } else { LOSS }
    }

    // Deterministic, dependency-free PRNG (matches board/mod.rs test style).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            // splitmix64
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next() as usize) % (hi - lo + 1)
        }
    }

    /// Crossing open-fives fork: 4 P2 winning cells, P1 to move at mr=2 blocks 2 -> LOSS.
    fn build_fork() -> Board {
        let mut b = Board::new();
        b.apply_move(0, 0).unwrap(); // P1 opener
        let p2_order = [
            (3, 3), (2, 3), (4, 3), (1, 3), (5, 3), (3, 2), (3, 4), (3, 1), (3, 5),
        ];
        let p1_fillers = [
            (-3, -3), (-3, -2), (-2, -3), (-3, -4), (-4, -3), (-2, -2),
            (-4, -4), (-4, -2), (-2, -4), (-5, -3), (-3, -5), (-5, -4),
        ];
        let (mut p2i, mut p1i) = (0usize, 0usize);
        let mut turn: i32 = -1;
        let mut guard = 0;
        while p2i < p2_order.len() && guard < 50 {
            guard += 1;
            let mut placed = 0;
            if turn == -1 {
                while placed < 2 && p2i < p2_order.len() {
                    let (q, r) = p2_order[p2i];
                    b.apply_move(q, r).unwrap();
                    p2i += 1;
                    placed += 1;
                }
            } else {
                while placed < 2 && p1i < p1_fillers.len() {
                    let (q, r) = p1_fillers[p1i];
                    b.apply_move(q, r).unwrap();
                    p1i += 1;
                    placed += 1;
                }
            }
            turn *= -1;
        }
        // Normalise to P1-to-move (mr=2) if a P2 partial turn left P2 on move.
        let extra = [(-3, 8), (-3, 7), (-4, 8), (-4, 7)];
        let mut ei = 0;
        while b.current_player == Player::Two && ei < extra.len() {
            let (q, r) = extra[ei];
            b.apply_move(q, r).unwrap();
            ei += 1;
        }
        b
    }

    /// Compact double-threat: 4 P2 winning cells, P1 to move at mr=2 -> proven LOSS. Radius 2
    /// keeps the EXHAUSTIVE brute oracle cheap; `build_fork`'s ~300-cell set would be ~300^2.
    fn compact_double_threat(off_q: i32, off_r: i32, vertical: bool) -> Board {
        let mut stones: Vec<((i32, i32), Cell)> = Vec::new();
        for i in 0..5i32 {
            if vertical {
                stones.push(((off_q, off_r + i), Cell::P2));
                stones.push(((off_q + 2, off_r + i), Cell::P2));
            } else {
                stones.push(((off_q + i, off_r), Cell::P2));
                stones.push(((off_q + i, off_r + 2), Cell::P2));
            }
        }
        let mut b = static_board(&stones, Player::One, 2);
        b.set_legal_move_radius(2);
        b
    }

    /// NOT-IN-CHECK position (two parallel P2 open-FOURS: threats but no win-in-1) — the
    /// surface the R3 guard protects, where a LOSS needs the full legal candidate set.
    fn compact_double_open_four(off_q: i32, off_r: i32, vertical: bool) -> Board {
        let mut stones: Vec<((i32, i32), Cell)> = Vec::new();
        for i in 0..4i32 {
            if vertical {
                stones.push(((off_q, off_r + i), Cell::P2));
                stones.push(((off_q + 2, off_r + i), Cell::P2));
            } else {
                stones.push(((off_q + i, off_r), Cell::P2));
                stones.push(((off_q + i, off_r + 2), Cell::P2));
            }
        }
        let mut b = static_board(&stones, Player::One, 2);
        b.set_legal_move_radius(2);
        b
    }

    /// Direct-construction static board for the off-window guard test, where a far stone
    /// shifts the window centre. `ply = stones.len()`, `last_move = None`.
    fn static_board(stones: &[((i32, i32), Cell)], player: Player, mr: u8) -> Board {
        Board::from_stones(stones, player, mr, stones.len() as u32, None)
    }

    #[test]
    fn test1_immediate_win_is_win() {
        // P1 builds 0..4 on r=0 with an immediate win available -> WIN, not a proven loss.
        let mut c = Board::new();
        for &(q, r) in &[
            (0, 0),
            (0, 9), (0, 8),
            (1, 0), (2, 0),
            (-3, 9), (-3, 8),
            (3, 0), (4, 0),
            (-1, 9), (-2, 9),
        ] {
            c.apply_move(q, r).unwrap();
        }
        assert_eq!(c.current_player, Player::One, "expected P1 to move");
        assert!(c.count_winning_moves(Player::One) >= 1, "P1 should have an immediate win");
        let r = solver().prove(&c, 20, 10_000);
        assert_eq!(r.result, WIN, "P1-with-immediate-win must be WIN, got {:?}", r.result);
        assert!(!r.line.is_empty(), "WIN must carry the move line");
        // line[0] must actually complete a 6.
        let mut c2 = c.clone();
        c2.apply_move(r.line[0].0, r.line[0].1).unwrap();
        assert!(c2.check_win(), "line[0] must complete 6, got {:?}", r.line[0]);
    }

    #[test]
    fn test2_quiet_position_not_loss() {
        // A quiet early position must NOT be a proven LOSS via the threat search.
        let mut e = Board::new();
        for &(q, r) in &[(0, 0), (3, 3), (3, 4), (0, 1), (1, 0)] {
            e.apply_move(q, r).unwrap();
        }
        let r = solver().prove(&e, 20, 5_000);
        assert_ne!(r.result, LOSS, "quiet position must not be a proven LOSS, got {:?}", r.result);
    }

    #[test]
    fn test4_fork_is_proven_loss() {
        // The spread-out fork (radius-5 legal set): the threat-pruned solver proves LOSS at two
        // depths. Brute confirmation lives in `test5`, where the tree is not ~300^2 (~60 s).
        let f = build_fork();
        assert_eq!(f.current_player, Player::One, "fork: expected P1 to move");
        assert_eq!(f.count_winning_moves(Player::Two), 4, "fork: expected 4 P2 threats");

        let rt = solver().prove(&f, 30, 300_000);
        let rm = solver().prove(&f, 12, 300_000);
        assert_eq!(rt.result, LOSS, "depth-30 must prove the fork LOSS, got {:?}", rt);
        assert_eq!(rm.result, LOSS, "depth-12 must prove the fork LOSS, got {:?}", rm);
    }

    #[test]
    fn test5_compact_loss_brute_confirmed() {
        // POSITIVE forced-loss detection: solver LOSS, brute AGREES, and TSS is far cheaper.
        let s = solver();
        let cases = [(0, 0, false), (-3, 4, false), (2, -2, true), (-6, -1, true)];
        for &(q, r, vert) in &cases {
            let b = compact_double_threat(q, r, vert);
            assert_eq!(
                b.count_winning_moves(Player::Two),
                4,
                "compact case {:?}: expected 4 P2 threats",
                (q, r, vert)
            );
            let res = s.prove(&b, 12, 200_000);
            assert_eq!(res.result, LOSS, "solver must prove compact LOSS, case {:?}", (q, r, vert));

            let mut bb = Budget::new(2_000_000);
            let brute = brute_solve(&mut b.clone(), 12, &mut bb);
            assert_eq!(brute, LOSS, "brute oracle disagrees (soundness), case {:?}", (q, r, vert));
            assert!(
                res.nodes < bb.nodes,
                "TSS ({}) not cheaper than brute ({}), case {:?}",
                res.nodes,
                bb.nodes,
                (q, r, vert)
            );
        }
    }

    #[test]
    fn soundness_fuzz_zero_false_loss() {
        // SOUNDNESS: every LOSS claim is cross-checked by the exhaustive brute_solve. Two
        // streams, because random near-terminal play alone yields ~0 forced losses: (A) random
        // control, (B) constructed double-threats. Assert 0 false-LOSS AND claims > 0.
        let mut rng = Lcg(0x0D_D5_01_5E_12_34_56_78);
        // Stream (C) adds NOT-IN-CHECK open-four doubles. This `solver()` is threat-only, so
        // the verify is disabled and such a position falls through to UNKNOWN: C exercises the
        // candidate-generation SURFACE only, the `verify_*` tests cover the certified LOSS.
        let s = solver();
        let mut checked = 0usize;
        let mut bad = 0usize;
        let mut nic_checked = 0usize;

        let cross_check = |bd: &Board| -> (bool, bool) {
            // returns (claimed_loss, refuted_by_brute)
            let res = s.prove(bd, 20, 4_000);
            if res.result != LOSS {
                return (false, false);
            }
            let mut bb = Budget::new(40_000);
            (true, brute_solve(&mut bd.clone(), 12, &mut bb) == WIN)
        };

        // (A) random near-terminal control.
        for _ in 0..60 {
            let mut bd = Board::new();
            let mut ok = true;
            let plies = rng.range(8, 24);
            for _ in 0..plies {
                let lm = bd.legal_moves();
                if lm.is_empty() {
                    ok = false;
                    break;
                }
                let (q, r) = lm[rng.range(0, lm.len() - 1)];
                if bd.apply_move(q, r).is_err() || bd.check_win() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let (claimed, refuted) = cross_check(&bd);
            if claimed {
                checked += 1;
                if refuted {
                    bad += 1;
                }
            }
        }

        // (B) constructed double-threats over offsets/orientations — reliable true LOSSes.
        for off_q in -8..=4i32 {
            for &(off_r, vert) in &[(0, false), (3, true), (-2, false)] {
                let bd = compact_double_threat(off_q, off_r, vert);
                let (claimed, refuted) = cross_check(&bd);
                if claimed {
                    checked += 1;
                    if refuted {
                        bad += 1;
                    }
                }
            }
        }

        // (C) constructed NOT-IN-CHECK open-four doubles; the count proves the surface ran.
        for off_q in -8..=4i32 {
            for &(off_r, vert) in &[(0, false), (3, true), (-2, false)] {
                let bd = compact_double_open_four(off_q, off_r, vert);
                assert_eq!(
                    bd.count_winning_moves(Player::Two),
                    0,
                    "stream C must be NOT-IN-CHECK (no P2 win-in-1), case {:?}",
                    (off_q, off_r, vert)
                );
                nic_checked += 1;
                let (claimed, refuted) = cross_check(&bd);
                if claimed {
                    checked += 1;
                    if refuted {
                        bad += 1;
                    }
                }
            }
        }

        assert_eq!(bad, 0, "SOUNDNESS: {bad}/{checked} LOSS claims refuted by exhaustive oracle");
        assert!(checked > 0, "soundness fuzz vacuous: no LOSS claims exercised");
        assert!(nic_checked > 0, "not-in-check surface not exercised (R3 guard untested)");
        eprintln!(
            "soundness fuzz: {checked} LOSS claims (all brute-confirmed), {nic_checked} not-in-check positions exercised"
        );
    }

    #[test]
    fn in_window_guard_suppresses_offwindow_win() {
        // P1 has 0..4 on r=0 (immediate win at (-1,0)/(5,0)); a far P2 stone at
        // (20,20) shifts the window center to ~(10,10) so the winning cell is
        // off-window (cheb > 9). Guard ON -> suppressed (UNKNOWN); OFF -> WIN.
        let mut stones: Vec<((i32, i32), Cell)> =
            (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        stones.push(((20, 20), Cell::P2));
        let b = static_board(&stones, Player::One, 2);
        assert!(b.count_winning_moves(Player::One) >= 1, "P1 should have an immediate win");

        let off = super::super::TacticalSolver::new(TacticalConfig { cand_cap: 40, window_half: Some(9), neighbor_dist: None });
        let on = super::super::TacticalSolver::new(TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None });

        let guarded = off.prove(&b, 8, 10_000);
        let unguarded = on.prove(&b, 8, 10_000);
        assert_eq!(unguarded.result, WIN, "no guard: must be WIN");
        assert!(super::super::is_off_window(&b, unguarded.line[0], 9), "test setup: win cell must be off-window");
        assert_eq!(guarded.result, Outcome::Unknown, "in-window guard must suppress the off-window WIN");
        assert!(guarded.line.is_empty(), "suppressed proof carries no line");
    }

    /// A genuine P1 WIN whose only winning move is a `threat_move` that `cand_cap` truncation
    /// drops: the P2 double-threat puts P1 IN CHECK while P1's far line reaches six via
    /// (4,0)+(5,0). With `cand_cap=1` an UNGUARDED search says FALSE LOSS; R3 must say UNKNOWN.
    fn fork_with_p1_counter() -> Board {
        // P2 compact double-threat far from the origin (radius-2 legal set).
        let mut stones: Vec<((i32, i32), Cell)> = Vec::new();
        for i in 0..5i32 {
            stones.push(((10 + i, 10), Cell::P2));
            stones.push(((10 + i, 12), Cell::P2));
        }
        // P1 four-in-a-row on r=0 (the counter line): (4,0)+(5,0) completes six.
        for q in 0..4i32 {
            stones.push(((q, 0), Cell::P1));
        }
        let mut b = static_board(&stones, Player::One, 2);
        b.set_legal_move_radius(2);
        b
    }

    #[test]
    fn neighbor_dist_widens_not_in_check_candidates_to_full_legal() {
        // Quiet-widening: at a NOT-IN-CHECK node `neighbor_dist=Some(d)` covering the legal
        // radius makes the candidate set the FULL legal set. `None` stays threat-only.
        let b = compact_double_open_four(0, 0, false); // P2 open-fours, P1 to move
        let (stm, opp) = (Player::One, Player::Two);
        assert_eq!(b.count_winning_moves(opp), 0, "setup: P1 is NOT in check");
        let nlegal = b.legal_move_count();

        let threat_only = super::candidates(&b, stm, opp, 1000, None);
        let widened = super::candidates(&b, stm, opp, 1000, Some(5));

        assert!(
            threat_only.len() < nlegal,
            "threat-only set ({}) must be a strict subset of legal ({nlegal})",
            threat_only.len()
        );
        assert_eq!(
            widened.len(),
            nlegal,
            "neighbor_dist covering the radius must yield the full legal set"
        );
        // Widening is additive: every threat-only candidate is still present.
        for m in &threat_only {
            assert!(widened.contains(m), "widened set dropped threat candidate {m:?}");
        }
    }

    #[test]
    fn off_window_completing_stone_suppressed() {
        // Coherence: the A1 override places line[0] AND the cached completing line[1], so BOTH
        // must be in-window. Here line[0] is in-window (cheb 3) and line[1] is not (cheb 4).
        let stones = vec![
            ((0, 0), Cell::P1), ((1, 0), Cell::P1), ((2, 0), Cell::P1), ((3, 0), Cell::P1),
            ((-1, 0), Cell::P2),
        ];
        let mut b = static_board(&stones, Player::One, 2);
        b.set_legal_move_radius(3);

        let raw = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 40, window_half: None, neighbor_dist: None,
        });
        let ru = raw.prove(&b, 12, 80_000);
        assert_eq!(ru.result, WIN, "unguarded must find the rightward win");
        assert!(ru.line.len() >= 2, "expected a 2-stone win, got {:?}", ru.line);
        assert!(!super::super::is_off_window(&b, ru.line[0], 3), "setup: line[0] must be IN-window, got {:?}", ru.line[0]);
        assert!(super::super::is_off_window(&b, ru.line[1], 3), "setup: completing line[1] must be OFF-window, got {:?}", ru.line[1]);

        let guarded = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 40, window_half: Some(3), neighbor_dist: None,
        });
        assert_eq!(
            guarded.prove(&b, 12, 80_000).result,
            Outcome::Unknown,
            "an off-window COMPLETING stone must suppress the override"
        );
    }

    #[test]
    fn spread_multicluster_no_false_proof() {
        // IMMUNITY (measured, not asserted): a flat [140][140]+70 array rep produces phantom
        // mates past |coord|~63; this HashMap/run-length solver must emit none in that regime.
        let s = solver();

        // (a) a brute-confirmed forced P1 LOSS translated past the OOB boundary (coord 64-90).
        for &(oq, orr) in &[(70, 70), (-80, 5), (64, -88)] {
            let b = compact_double_threat(oq, orr, false);
            assert!(
                b.cells_iter().any(|(&(q, r), _)| q.abs().max(r.abs()) > 63),
                "setup: cluster must exceed the |coord|>63 OOB boundary"
            );
            assert_eq!(
                s.prove(&b, 12, 400_000).result,
                LOSS,
                "native must prove the forced loss unchanged past the OOB boundary, coord {:?}",
                (oq, orr)
            );
        }

        // (b) a MULTI-CLUSTER non-winning board at large coords, kept within a BOUNDED bbox:
        //     clusters 150 cells apart make the brute oracle's legal rebuild O(bbox).
        let multi: Vec<((i32, i32), Cell)> = vec![
            ((70, 70), Cell::P1), ((71, 70), Cell::P1), ((70, 71), Cell::P2),
            ((78, 72), Cell::P2), ((79, 72), Cell::P2), ((78, 73), Cell::P1),
            ((73, 78), Cell::P1), ((74, 78), Cell::P2), ((73, 79), Cell::P1),
        ];
        let mut mb = static_board(&multi, Player::One, 2);
        mb.set_legal_move_radius(1); // tight legal set -> the brute oracle stays cheap
        let res = s.prove(&mb, 6, 100_000);
        let mut bb = Budget::new(80_000);
        let brute = brute_solve(&mut mb.clone(), 6, &mut bb);
        if res.result == WIN {
            assert_ne!(brute, LOSS, "native FALSE WIN on multi-cluster board");
        }
        if res.result == LOSS {
            assert_ne!(brute, WIN, "native FALSE LOSS on multi-cluster board");
        }
    }

    #[test]
    fn widened_solver_stays_sound() {
        // SOUNDNESS of the quiet-move widening: with the WIDE candidate set the verdict must
        // stay consistent with the exhaustive oracle. Non-vacuous (>=1 LOSS proven).
        let s = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1000,
            window_half: None,
            neighbor_dist: Some(5),
        });
        let cases = [(0, 0, false), (-3, 4, false), (2, -2, true), (-6, -1, true)];
        let mut loss_claims = 0;
        for &(q, r, vert) in &cases {
            let b = compact_double_threat(q, r, vert);
            let res = s.prove(&b, 12, 1_000_000);
            assert_ne!(res.result, WIN, "widened FALSE WIN on a forced loss, case {:?}", (q, r, vert));
            if res.result == LOSS {
                loss_claims += 1;
                let mut bb = Budget::new(2_000_000);
                assert_eq!(
                    brute_solve(&mut b.clone(), 12, &mut bb),
                    LOSS,
                    "widened FALSE LOSS (brute disagrees), case {:?}",
                    (q, r, vert)
                );
            }
        }
        assert!(loss_claims > 0, "widened solver proved 0 LOSSes (vacuous) — raise budget/depth");
    }

    #[test]
    fn verify_recovers_truncated_win() {
        // Recall-preserving verify: cand_cap=1 truncates away P1's winning counter, so the
        // verify must search the dropped legal set and return WIN where R3 alone says UNKNOWN.
        let b = fork_with_p1_counter();
        let no_verify = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: None,
        });
        let verify = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: Some(2),
        });
        assert_eq!(no_verify.prove(&b, 20, 200_000).result, Outcome::Unknown, "no-verify: conservative UNKNOWN");
        let res = verify.prove(&b, 20, 400_000);
        assert_eq!(res.result, WIN, "verify must recover the truncated winning counter");
        // The recovered line's first move must actually win the position.
        assert!(!res.line.is_empty(), "WIN carries a line");
    }

    #[test]
    fn verify_certifies_truncated_loss() {
        // The other verify branch: the dropped legal moves ALSO all lose, certifying the LOSS.
        let b = compact_double_threat(0, 0, false);
        let no_verify = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: None,
        });
        let verify = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: Some(2),
        });
        assert_eq!(no_verify.prove(&b, 12, 400_000).result, Outcome::Unknown, "no-verify: conservative UNKNOWN");
        assert_eq!(verify.prove(&b, 12, 1_000_000).result, LOSS, "verify must certify the truncated LOSS");
        let mut bb = Budget::new(2_000_000);
        assert_eq!(brute_solve(&mut b.clone(), 12, &mut bb), LOSS, "brute confirms the LOSS (soundness)");
    }

    #[test]
    fn verify_path_is_sound_over_grid() {
        // cand_cap=1 truncates at EVERY node, so every LOSS routes through the full verify.
        let s = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: Some(2),
        });
        let cases = [(0, 0, false), (-3, 4, false), (2, -2, true), (-6, -1, true)];
        for &(q, r, vert) in &cases {
            let b = compact_double_threat(q, r, vert);
            let res = s.prove(&b, 12, 1_500_000);
            assert_ne!(res.result, WIN, "verify FALSE WIN on a forced loss, case {:?}", (q, r, vert));
            if res.result == LOSS {
                let mut bb = Budget::new(2_000_000);
                assert_eq!(
                    brute_solve(&mut b.clone(), 12, &mut bb),
                    LOSS,
                    "verify FALSE LOSS (brute disagrees), case {:?}",
                    (q, r, vert)
                );
            }
        }
    }

    // RED-TEAM soundness attacks: produce a FALSE proof or prove it cannot. Heavy ones are
    // `#[ignore]`.

    /// RED-TEAM: the verify path on NOT-IN-CHECK positions with a truncating cand_cap plus
    /// neighbor widening, cross-checking EVERY proof against the full-width brute oracle.
    ///
    /// SLOW (`#[ignore]`): 96 budget-1.5M solves, minutes to hours, and the only test
    /// exercising a not-in-check ROOT LOSS certified by the verify. VERIFIED 0 false proofs,
    /// 2026-06-29 (19/19, this sweep incl.).
    #[test]
    #[ignore = "exhaustive full-width verify sweep — minutes; run on-demand (--ignored)"]
    fn redteam_verify_grid_no_false_proof() {
        let mut loss_claims = 0usize;
        let mut win_claims = 0usize;
        let mut nic_loss = 0usize;
        let mut brute_unknown_on_loss = 0usize;
        for &cap in &[1usize, 3] {
            for &nd in &[2i32] {
                let s = super::super::TacticalSolver::new(TacticalConfig {
                    cand_cap: cap,
                    window_half: None,
                    neighbor_dist: Some(nd),
                });
                for off_q in -4..=3i32 {
                    for &(off_r, vert) in &[(0, false), (3, true), (-2, false)] {
                        for builder in 0..2 {
                            let b = if builder == 0 {
                                compact_double_open_four(off_q, off_r, vert)
                            } else {
                                compact_double_threat(off_q, off_r, vert)
                            };
                            let in_check = b.count_winning_moves(Player::Two) >= 1;
                            let res = s.prove(&b, 12, 1_500_000);
                            if res.result == LOSS {
                                loss_claims += 1;
                                if !in_check {
                                    nic_loss += 1;
                                }
                                let mut bb = Budget::new(1_500_000);
                                let brute = brute_solve(&mut b.clone(), 12, &mut bb);
                                assert_ne!(
                                    brute, WIN,
                                    "FALSE LOSS: solver LOSS but brute finds an escape-to-WIN \
                                     (cap={cap}, nd={nd}, in_check={in_check}, case {:?})",
                                    (off_q, off_r, vert, builder)
                                );
                                if brute == Outcome::Unknown {
                                    brute_unknown_on_loss += 1;
                                }
                            } else if res.result == WIN {
                                win_claims += 1;
                                let mut bb = Budget::new(1_500_000);
                                let brute = brute_solve(&mut b.clone(), 12, &mut bb);
                                assert_ne!(
                                    brute, LOSS,
                                    "FALSE WIN: solver WIN but brute proves LOSS \
                                     (cap={cap}, nd={nd}, case {:?})",
                                    (off_q, off_r, vert, builder)
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(loss_claims > 0, "vacuous: no LOSS claims exercised");
        eprintln!(
            "redteam grid: {loss_claims} LOSS ({nic_loss} not-in-check, {brute_unknown_on_loss} \
             brute-unresolved), {win_claims} WIN — all brute-consistent, 0 false proofs"
        );
    }

    /// RED-TEAM random stream: random COMPACT positions under a truncating cand_cap, every
    /// claim cross-checked bidirectionally. SLOW (`#[ignore]`): 250 verify-config solves plus
    /// per-claim brute. VERIFIED 0 false proofs, 2026-06-29.
    #[test]
    #[ignore = "exhaustive random verify sweep — run on-demand (--ignored)"]
    fn redteam_verify_random_compact_no_false_proof() {
        let mut rng = Lcg(0xBADC_0FFE_E0DD_F00D);
        let s = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 2,
            window_half: None,
            neighbor_dist: Some(2),
        });
        let mut loss_claims = 0usize;
        let mut nic_seen = 0usize;
        let mut win_claims = 0usize;
        let mut samples = 0usize;
        let mut attempt = 0usize;
        // Random legal moves kept inside a small box, so brute stays exhaustive-cheap.
        while samples < 250 && attempt < 4000 {
            attempt += 1;
            let mut bd = Board::new();
            let plies = rng.range(6, 18);
            let mut ok = true;
            for _ in 0..plies {
                // restrict to a compact box so brute stays cheap + dense tactics
                let lm: Vec<(i32, i32)> = bd
                    .legal_moves()
                    .into_iter()
                    .filter(|&(q, r)| q.abs() <= 3 && r.abs() <= 3)
                    .collect();
                if lm.is_empty() {
                    ok = false;
                    break;
                }
                let (q, r) = lm[rng.range(0, lm.len() - 1)];
                if bd.apply_move(q, r).is_err() || bd.check_win() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            bd.set_legal_move_radius(2);
            if bd.legal_move_count() > 30 {
                continue; // keep brute exhaustive-cheap
            }
            samples += 1;
            let in_check = bd.count_winning_moves(bd.current_player.other()) >= 1;
            let res = s.prove(&bd, 12, 1_000_000);
            if res.result == LOSS {
                loss_claims += 1;
                if !in_check {
                    nic_seen += 1;
                }
                let mut bb = Budget::new(1_500_000);
                let brute = brute_solve(&mut bd.clone(), 12, &mut bb);
                assert_ne!(
                    brute, WIN,
                    "FALSE LOSS (random): solver LOSS but brute escapes-to-WIN, in_check={in_check}"
                );
            } else if res.result == WIN {
                win_claims += 1;
                let mut bb = Budget::new(1_500_000);
                let brute = brute_solve(&mut bd.clone(), 12, &mut bb);
                assert_ne!(brute, LOSS, "FALSE WIN (random): solver WIN but brute proves LOSS");
            }
        }
        eprintln!(
            "redteam random: {samples} compact positions, {loss_claims} LOSS ({nic_seen} \
             not-in-check), {win_claims} WIN — 0 false proofs"
        );
    }

    /// RED-TEAM: flip-aware negamax SIGN. Asserts WIN on a 2-stone forcing win, then REALIZES
    /// the line — independent of the brute oracle, which shares the flip logic.
    #[test]
    fn redteam_flip_sign_two_stone_win_realized() {
        let stones: Vec<((i32, i32), Cell)> =
            (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = static_board(&stones, Player::One, 2);
        assert_eq!(b.count_winning_moves(Player::One), 0, "setup: no single-stone win (needs 2)");
        let res = solver().prove(&b, 12, 200_000);
        assert_eq!(res.result, WIN, "2-stone forcing win must be WIN (flip-sign), got {:?}", res.result);
        // Realize the same-turn line: both stones are P1's (mr=2 -> 1, not flipped),
        // so line carries P1's two placements. Replaying must complete a 6.
        assert!(res.line.len() >= 2, "same-turn win must carry both P1 stones, got {:?}", res.line);
        let mut c = b.clone();
        c.apply_move(res.line[0].0, res.line[0].1).unwrap();
        c.apply_move(res.line[1].0, res.line[1].1).unwrap();
        assert!(c.check_win(), "realized WIN line must produce a real 6, got line {:?}", res.line);
        // And the SAME position must never be called a LOSS by the verify config.
        let v = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 2,
            window_half: None,
            neighbor_dist: Some(2),
        });
        assert_ne!(v.prove(&b, 12, 500_000).result, LOSS, "winnable position must never be a LOSS");
    }

    /// RED-TEAM: a budget-starved target must return UNKNOWN, never a manufactured proof.
    #[test]
    fn redteam_budget_exhaustion_no_false_proof() {
        let v = super::super::TacticalSolver::new(TacticalConfig {
            cand_cap: 1,
            window_half: None,
            neighbor_dist: Some(2),
        });
        // A brute-confirmed forced LOSS, so the full-budget result is LOSS.
        let b = compact_double_threat(0, 0, false);
        let mut bb = Budget::new(2_000_000);
        assert_eq!(brute_solve(&mut b.clone(), 12, &mut bb), LOSS, "setup: truly a forced LOSS");
        // Find a budget large enough to prove it, then starve below that.
        let full = v.prove(&b, 12, 1_000_000);
        assert_eq!(full.result, LOSS, "full budget proves the LOSS");
        for budget in 1u64..=full.nodes.saturating_sub(1).min(300) {
            let r = v.prove(&b, 12, budget);
            assert_ne!(
                r.result, WIN,
                "budget {budget}: starved search manufactured a WIN"
            );
            // A LOSS is permitted ONLY if the search actually completed (not exhausted).
            if r.result == LOSS {
                assert!(
                    !r.budget_exhausted,
                    "budget {budget}: EXHAUSTED search still returned a LOSS proof (unsound)"
                );
            }
        }
    }

    #[test]
    fn neighbor_dist_does_not_widen_in_check_nodes() {
        // IN CHECK the threat-only set is already complete, so widening must not change it.
        let b = compact_double_threat(0, 0, false);
        let (stm, opp) = (Player::One, Player::Two);
        assert!(b.count_winning_moves(opp) >= 1, "setup: P1 IS in check");
        let plain = super::candidates(&b, stm, opp, 1000, None);
        let widened = super::candidates(&b, stm, opp, 1000, Some(5));
        assert_eq!(plain, widened, "in-check candidate set must not widen");
    }

    #[test]
    fn r3_guard_suppresses_truncated_false_loss() {
        // SOUNDNESS (R3): `cand_cap=1` drops P1's winning counter, so unguarded means FALSE LOSS.
        let b = fork_with_p1_counter();
        // Test-setup invariants: P1 is in check (4 P2 threats), has no win-in-1,
        // but the position is truly a WIN (full-legal brute finds the counter).
        assert_eq!(b.count_winning_moves(Player::Two), 4, "setup: 4 P2 threats (P1 in check)");
        assert_eq!(b.count_winning_moves(Player::One), 0, "setup: P1 has no immediate win");
        let mut bb = Budget::new(2_000_000);
        assert_eq!(brute_solve(&mut b.clone(), 12, &mut bb), WIN, "setup: position is truly a P1 WIN");

        let trunc = super::super::TacticalSolver::new(TacticalConfig { cand_cap: 1, window_half: None, neighbor_dist: None });
        let res = trunc.prove(&b, 20, 200_000);
        assert_ne!(
            res.result, LOSS,
            "R3: truncated candidate set must not yield a false LOSS (got {:?})",
            res.result
        );
    }

    /// Drive the scored α-β core directly (full root window) and return verdict + score + PV.
    fn run_scored(
        b: &Board,
        cfg: &TacticalConfig,
        depth: i32,
        budget: u64,
    ) -> (Outcome, i32, Vec<(i32, i32)>) {
        let mut board = b.clone();
        let mut bud = Budget::new(budget);
        let mut tt = super::super::tt::ProofTt::new();
        let mut ordering = super::OrderingState::new();
        let s = super::solve(&mut board, depth, 0, NEG_INF, POS_INF, &mut bud, cfg, &mut tt, &mut ordering);
        (outcome_of(s.score), s.score, s.line)
    }

    /// The pre-increment-1 3-valued oracle's verdict — the invariance target.
    fn run_3valued(b: &Board, cfg: &TacticalConfig, depth: i32, budget: u64) -> Outcome {
        let mut board = b.clone();
        let mut bud = Budget::new(budget);
        let mut tt = super::super::tt::ProofTt::new();
        super::solve_3valued(&mut board, depth, &mut bud, cfg, &mut tt).outcome
    }

    #[test]
    fn clamp_heuristic_never_reaches_proof_region() {
        // SOUNDNESS: a heuristic leaf can NEVER masquerade as a mate. Clamp pins
        // any eval (incl. ±∞-ish) strictly inside (-WIN_THRESHOLD, WIN_THRESHOLD).
        for &v in &[i32::MIN, -MATE, -WIN_THRESHOLD, -1, 0, 1, WIN_THRESHOLD, MATE, i32::MAX] {
            let c = super::clamp_heuristic(v);
            assert!(c.abs() < WIN_THRESHOLD, "clamp({v}) = {c} leaked into the proof region");
            assert_ne!(outcome_of(c), WIN, "clamped heuristic must never read as WIN");
            assert_ne!(outcome_of(c), LOSS, "clamped heuristic must never read as LOSS");
        }
    }

    #[test]
    fn scored_mate_distance_prefers_shorter_win() {
        // Mate-distance encoding: a SHORTER forced win scores higher. An immediate
        // (1-stone) win at the root is exactly `MATE - 0 = MATE`; a 2-stone forcing
        // win must score strictly less (it lands a ply deeper) but still proven.
        let cfg = TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None };

        // Immediate win: P1 has 0..4 on r=0 (a single stone completes six).
        let imm: Vec<((i32, i32), Cell)> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let imm_b = static_board(&imm, Player::One, 2);
        assert!(imm_b.count_winning_moves(Player::One) >= 1, "setup: immediate win");
        let (o1, s1, _) = run_scored(&imm_b, &cfg, 12, 50_000);
        assert_eq!(o1, WIN, "immediate win must be WIN");
        assert_eq!(s1, MATE, "immediate win scores MATE - 0 = MATE, got {s1}");

        // 2-stone forcing win: P1 has 0..3 on r=0 (needs (4,0) then (5,0)).
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let two_b = static_board(&two, Player::One, 2);
        assert_eq!(two_b.count_winning_moves(Player::One), 0, "setup: no 1-stone win");
        let (o2, s2, _) = run_scored(&two_b, &cfg, 12, 200_000);
        assert_eq!(o2, WIN, "2-stone forcing win must be WIN");
        assert!(s2 >= WIN_THRESHOLD, "2-stone win must be proven, got {s2}");
        assert!(s2 < s1, "deeper mate must score lower: 2-stone {s2} !< immediate {s1}");
    }

    #[test]
    fn scored_loss_is_mate_magnitude_negative() {
        // A proven forced LOSS carries a mate-magnitude NEGATIVE score (the
        // mate-distance loss encoding), and the verdict matches the oracle.
        let cfg = TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None };
        let b = compact_double_threat(0, 0, false);
        let (o, s, _) = run_scored(&b, &cfg, 12, 200_000);
        assert_eq!(o, LOSS, "compact double-threat is a forced LOSS");
        assert!(s <= -WIN_THRESHOLD, "LOSS must carry a mate-magnitude negative score, got {s}");
    }

    #[test]
    fn verdict_invariance_scored_matches_3valued() {
        // THE INVARIANCE GATE: scored α-β must reproduce the 3-valued verdict everywhere.
        type Case = (Board, TacticalConfig, i32, u64, &'static str);
        let cfg = |cand_cap, neighbor_dist| TacticalConfig {
            cand_cap,
            window_half: None,
            neighbor_dist,
        };
        let imm: Vec<((i32, i32), Cell)> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let mut quiet = Board::new();
        for &(q, r) in &[(0, 0), (3, 3), (3, 4), (0, 1), (1, 0)] {
            quiet.apply_move(q, r).unwrap();
        }
        let cases: Vec<Case> = vec![
            (static_board(&imm, Player::One, 2), cfg(40, None), 12, 50_000, "immediate-win"),
            (static_board(&two, Player::One, 2), cfg(40, None), 12, 200_000, "2-stone-win"),
            (quiet, cfg(40, None), 20, 20_000, "quiet-not-loss"),
            (build_fork(), cfg(40, None), 30, 300_000, "fork-loss-d30"),
            (build_fork(), cfg(40, None), 12, 300_000, "fork-loss-d12"),
            (compact_double_threat(0, 0, false), cfg(40, None), 12, 200_000, "compact-loss-a"),
            (compact_double_threat(-3, 4, false), cfg(40, None), 12, 200_000, "compact-loss-b"),
            (compact_double_threat(2, -2, true), cfg(40, None), 12, 200_000, "compact-loss-c"),
            (compact_double_open_four(0, 0, false), cfg(40, None), 12, 80_000, "open4-threatonly-unknown"),
            // verify (neighbor_dist) + truncating cand_cap surfaces:
            (fork_with_p1_counter(), cfg(1, None), 20, 200_000, "trunc-r3-unknown"),
            (fork_with_p1_counter(), cfg(1, Some(2)), 20, 400_000, "trunc-verify-win"),
            (compact_double_threat(0, 0, false), cfg(1, Some(2)), 12, 1_000_000, "trunc-verify-loss"),
            (compact_double_threat(0, 0, false), cfg(1, None), 12, 400_000, "trunc-noverify-unknown"),
        ];
        for (b, c, depth, budget, name) in cases {
            let (scored, _, _) = run_scored(&b, &c, depth, budget);
            let reference = run_3valued(&b, &c, depth, budget);
            assert_eq!(
                scored, reference,
                "VERDICT INVARIANCE BROKEN ({name}): scored α-β = {scored:?}, 3-valued oracle = {reference:?}"
            );
        }
    }

    #[test]
    fn scored_win_pv_is_realizable() {
        // α-β must not corrupt the winning PV (the A1 override plays line[0..2]).
        // The 2-stone forcing win's line must replay to a real 6-in-a-row.
        let cfg = TacticalConfig { cand_cap: 40, window_half: None, neighbor_dist: None };
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = static_board(&two, Player::One, 2);
        let (o, _, line) = run_scored(&b, &cfg, 12, 200_000);
        assert_eq!(o, WIN, "must be WIN");
        assert!(line.len() >= 2, "2-stone win carries both stones, got {line:?}");
        let mut c = b.clone();
        c.apply_move(line[0].0, line[0].1).unwrap();
        c.apply_move(line[1].0, line[1].1).unwrap();
        assert!(c.check_win(), "PV must realize a real 6, got {line:?}");
    }

    #[test]
    #[allow(clippy::nonminimal_bool)] // the explicit "not (WIN,LOSS) and not (LOSS,WIN)" form is intentional (VERBATIM)
    fn verdict_invariance_fuzz_scored_matches_3valued() {
        // The fixed 13-case invariance test widened to a RANDOMIZED stream, stressing the
        // fail-soft mate-bound corners. Scored and oracle share candidate set + budget, so they
        // must never contradict and must be EQUAL when both are conclusive; a
        // scored-conclusive / oracle-UNKNOWN split is an allowed pruning win, asserted sound.
        let mut rng = Lcg(0xF17E_55ED_2026_0629);
        // THREAT-ONLY configs: the recall VERIFY uses no PVS/LMR and is covered elsewhere.
        let cfgs = [(1usize, None), (2, None), (40, None)];
        let (mut win, mut loss, mut unknown, mut checked, mut exact_agree) = (0, 0, 0, 0, 0);

        let check = |b: &Board, cand: usize, nd: Option<i32>, depth: i32, budget: u64,
                     win: &mut i32, loss: &mut i32, unknown: &mut i32, checked: &mut i32, exact: &mut i32| {
            let cfg = TacticalConfig { cand_cap: cand, window_half: None, neighbor_dist: nd };
            let (scored, _, _) = run_scored(b, &cfg, depth, budget);
            let reference = run_3valued(b, &cfg, depth, budget);
            assert!(
                !(scored == WIN && reference == LOSS) && !(scored == LOSS && reference == WIN),
                "FUZZ CONTRADICTION: scored {scored:?} vs oracle {reference:?} (cand={cand}, nd={nd:?})"
            );
            if scored != Outcome::Unknown && reference != Outcome::Unknown {
                assert_eq!(scored, reference, "FUZZ INVARIANCE: both conclusive but unequal (cand={cand}, nd={nd:?})");
                *exact += 1;
            }
            // Brute-confirm any scored verdict; radius-2 boards keep it cheap.
            if scored == LOSS || scored == WIN {
                let mut bb = Budget::new(200_000);
                let brute = brute_solve(&mut b.clone(), 12, &mut bb);
                if scored == LOSS {
                    assert_ne!(brute, WIN, "FUZZ FALSE LOSS: scored LOSS but brute escapes to WIN");
                } else {
                    assert_ne!(brute, LOSS, "FUZZ FALSE WIN: scored WIN but brute proves LOSS");
                }
            }
            match scored {
                Outcome::Win => *win += 1,
                Outcome::Loss => *loss += 1,
                Outcome::Unknown => *unknown += 1,
            }
            *checked += 1;
        };

        // (A) random COMPACT boards, one random threat-only config each.
        let (mut samples, mut attempt) = (0, 0);
        while samples < 36 && attempt < 4000 {
            attempt += 1;
            let mut bd = Board::new();
            let mut ok = true;
            let plies = rng.range(5, 10);
            for _ in 0..plies {
                let lm: Vec<(i32, i32)> = bd
                    .legal_moves()
                    .into_iter()
                    .filter(|&(q, r)| q.abs() <= 2 && r.abs() <= 2)
                    .collect();
                if lm.is_empty() {
                    ok = false;
                    break;
                }
                let (q, r) = lm[rng.range(0, lm.len() - 1)];
                if bd.apply_move(q, r).is_err() || bd.check_win() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            bd.set_legal_move_radius(2);
            if bd.legal_move_count() > 36 {
                continue;
            }
            samples += 1;
            let (cand, nd) = cfgs[rng.range(0, cfgs.len() - 1)];
            check(&bd, cand, nd, 6, 40_000, &mut win, &mut loss, &mut unknown, &mut checked, &mut exact_agree);
        }

        // (B) constructed double-threats guaranteeing forced-LOSS verdicts under truncation.
        for off_q in -4..=3i32 {
            for &(off_r, vert) in &[(0, false), (3, true)] {
                for builder in 0..2 {
                    let b = if builder == 0 {
                        compact_double_threat(off_q, off_r, vert)
                    } else {
                        compact_double_open_four(off_q, off_r, vert)
                    };
                    let (cand, nd) = cfgs[rng.range(0, cfgs.len() - 1)];
                    check(&b, cand, nd, 8, 100_000, &mut win, &mut loss, &mut unknown, &mut checked, &mut exact_agree);
                }
            }
        }

        // (C) constructed WIN positions over a grid: an open FIVE proves WIN instantly for ANY
        //     cand_cap, and radius-2 stops the rare truncation path expanding a wide tree.
        for off in -4..=3i32 {
            let stones: Vec<((i32, i32), Cell)> =
                (0..5).map(|q| ((q, off), Cell::P1)).collect();
            let mut b = static_board(&stones, Player::One, 2);
            b.set_legal_move_radius(2);
            let (cand, nd) = cfgs[rng.range(0, cfgs.len() - 1)];
            check(&b, cand, nd, 8, 80_000, &mut win, &mut loss, &mut unknown, &mut checked, &mut exact_agree);
        }

        assert!(checked > 40, "fuzz too small ({checked} positions)");
        assert!(win > 0, "fuzz vacuous: no WIN verdict exercised");
        assert!(loss > 0, "fuzz vacuous: no LOSS verdict exercised");
        assert!(unknown > 0, "fuzz should include UNKNOWN positions (mate-bound corners)");
        eprintln!(
            "verdict-invariance fuzz: {checked} positions ({win} WIN, {loss} LOSS, {unknown} UNKNOWN), \
             {exact_agree} both-conclusive agreements, 0 contradictions, 0 brute-refuted verdicts"
        );
    }

    #[test]
    fn iterative_deepening_proves_short_mate_and_is_idempotent() {
        // The ID + aspiration driver must return the SAME verdict regardless of max_depth.
        let s = solver();
        let b = compact_double_threat(0, 0, false); // forced P1 LOSS in a few plies
        let shallow = s.prove(&b, 6, 400_000);
        let deep = s.prove(&b, 40, 400_000);
        assert_eq!(shallow.result, LOSS, "ID must prove the short forced LOSS at a small cap");
        assert_eq!(deep.result, LOSS, "ID verdict idempotent under a deeper cap");
        assert!(
            deep.nodes < 400_000 && !deep.budget_exhausted,
            "mate-early-stop must prove the shallow mate well within budget (nodes={}, exhausted={})",
            deep.nodes,
            deep.budget_exhausted
        );
    }

    #[test]
    fn iterative_deepening_win_pv_is_realizable() {
        // ID/aspiration must not corrupt the root WIN PV; the replayed line must complete a 6.
        let s = solver();
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let b = static_board(&two, Player::One, 2);
        let r = s.prove(&b, 20, 200_000);
        assert_eq!(r.result, WIN, "ID must prove the 2-stone forcing WIN");
        assert!(r.line.len() >= 2, "WIN carries both stones, got {:?}", r.line);
        let mut c = b.clone();
        c.apply_move(r.line[0].0, r.line[0].1).unwrap();
        c.apply_move(r.line[1].0, r.line[1].1).unwrap();
        assert!(c.check_win(), "ID WIN PV must realize a real 6, got {:?}", r.line);
    }

    /// Drive the scored core with an OPTIONAL `PolicyPrior` wired into ordering.
    fn run_scored_pol(
        b: &Board,
        cfg: &TacticalConfig,
        depth: i32,
        budget: u64,
        policy: Option<Box<dyn super::super::ordering::PolicyPrior>>,
    ) -> (Outcome, Vec<(i32, i32)>) {
        let mut board = b.clone();
        let mut bud = Budget::new(budget);
        let mut tt = super::super::tt::ProofTt::new();
        let mut ordering = match policy {
            Some(p) => super::OrderingState::with_policy(p),
            None => super::OrderingState::new(),
        };
        let s = super::solve(&mut board, depth, 0, NEG_INF, POS_INF, &mut bud, cfg, &mut tt, &mut ordering);
        (outcome_of(s.score), s.line)
    }

    #[test]
    fn verdict_invariant_to_net_policy_ordering() {
        // The CRITICAL property: net policy in candidate ORDERING must NEVER change a verdict.
        // An ADVERSARIAL prior maximally permutes the order; verdict and PV must be invariant.
        use super::super::ordering::PolicyPrior;
        struct Adversarial;
        impl PolicyPrior for Adversarial {
            fn prior(&self, mv: (i32, i32)) -> f32 {
                // splitmix-style hash of the coords -> a scattered value in [-1, 1].
                let mut z = (mv.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (mv.1 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 31;
                ((z & 0xFFFF) as f32) / 32768.0 - 1.0
            }
        }
        let cfg = |cand_cap, neighbor_dist| TacticalConfig { cand_cap, window_half: None, neighbor_dist };
        let imm: Vec<((i32, i32), Cell)> = (0..5).map(|q| ((q, 0), Cell::P1)).collect();
        let two: Vec<((i32, i32), Cell)> = (0..4).map(|q| ((q, 0), Cell::P1)).collect();
        let mut quiet = Board::new();
        for &(q, r) in &[(0, 0), (3, 3), (3, 4), (0, 1), (1, 0)] {
            quiet.apply_move(q, r).unwrap();
        }
        type Case = (Board, TacticalConfig, i32, u64, &'static str);
        let cases: Vec<Case> = vec![
            (static_board(&imm, Player::One, 2), cfg(40, None), 12, 50_000, "immediate-win"),
            (static_board(&two, Player::One, 2), cfg(40, None), 12, 200_000, "2-stone-win"),
            (quiet, cfg(40, None), 12, 40_000, "quiet-unknown"),
            (build_fork(), cfg(40, None), 12, 300_000, "fork-loss"),
            (compact_double_threat(0, 0, false), cfg(40, None), 12, 200_000, "compact-loss"),
            (compact_double_threat(-3, 4, false), cfg(40, None), 12, 200_000, "compact-loss-b"),
            (fork_with_p1_counter(), cfg(1, Some(2)), 20, 400_000, "trunc-verify-win"),
            (compact_double_threat(0, 0, false), cfg(1, Some(2)), 12, 1_000_000, "trunc-verify-loss"),
            (compact_double_threat(0, 0, false), cfg(1, None), 12, 400_000, "trunc-noverify-unknown"),
        ];
        for (b, c, depth, budget, name) in cases {
            let (base, base_line) = run_scored_pol(&b, &c, depth, budget, None);
            let (perm, perm_line) = run_scored_pol(&b, &c, depth, budget, Some(Box::new(Adversarial)));
            assert_eq!(base, perm, "net-policy ordering CHANGED the verdict ({name}): {base:?} -> {perm:?}");
            // A WIN PV must still realize a real win under the reordered search.
            if perm == WIN {
                let mut bd = b.clone();
                for &(q, r) in perm_line.iter().take(2) {
                    bd.apply_move(q, r).unwrap();
                }
                assert!(bd.check_win(), "policy-reordered WIN PV must realize a real 6 ({name})");
            }
            let _ = &base_line;
        }
    }
}

