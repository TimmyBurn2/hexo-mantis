// >300 justify (R8): the offence, both audit forms and the tests that build pool trees to drive them are one unit.
//! The root at deploy: the offence plays a decided stone before the search, the audit vets the search's turn after.

use mantis_core::board::Board;

use super::node::Node;
use super::tactics_wiring::{AuditConfig, AuditMode, TacticsError, TacticsState};
use super::MCTSTree;
use crate::tactics::{analyze, position_key, with_stone, Terminal, Verdict};

/// The opponent's standing after our turn: no proof (the turn holds), a proof in this many turns, or no budget left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Proof {
    Holds,
    Lost(u8),
    Exhausted,
}

/// A first stone's audit: a second stone that holds, none to try, its longest-resisting loss, or no budget left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FirstStone {
    HeldBy((i32, i32)),
    Untried,
    Lost(u8, (i32, i32)),
    Exhausted,
}

/// The row a decided root's stone counts in.
#[derive(Clone, Copy)]
enum Decided {
    Finish,
    StoredProof,
    Proof,
}

/// `board` with `cells` placed, or `None` when one is occupied.
fn after(board: &Board, cells: &[(i32, i32)]) -> Option<Board> {
    let mut b = board.clone();
    for &(q, r) in cells {
        b.apply_move(q, r).ok()?;
    }
    Some(b)
}

/// A node's children as pool indices: most visits first, then the higher prior, then pool order.
fn ranked_children(pool: &[Node], node: u32) -> Vec<u32> {
    let n = &pool[node as usize];
    if !n.is_expanded() {
        return Vec::new();
    }
    let mut kids: Vec<u32> = (n.first_child..n.first_child + u32::from(n.n_children)).collect();
    kids.sort_by(|&a, &b| {
        let (x, y) = (&pool[a as usize], &pool[b as usize]);
        y.n_visits
            .cmp(&x.n_visits)
            .then(y.prior.total_cmp(&x.prior))
            .then(a.cmp(&b))
    });
    kids
}

/// The candidate whose opponent proof needs the most turns, the first of equals (the kind's order).
fn best_hold<T: Copy>(lost: &[(u8, T)]) -> Option<T> {
    let mut best: Option<(u8, T)> = None;
    for &(n, x) in lost {
        if best.is_none_or(|(b, _)| n > b) {
            best = Some((n, x));
        }
    }
    best.map(|(_, x)| x)
}

impl TacticsState {
    /// The opponent's proof at `post` (the opponent to move after our turn) within the audit's call and total budgets.
    fn opponent_proof(&mut self, audit: AuditConfig, post: &Board, left: &mut u64) -> Proof {
        if post.check_win() {
            return Proof::Holds;
        }
        let facts = analyze(post);
        match facts.terminal {
            Some(Terminal::Win) => return Proof::Lost(0),
            Some(Terminal::Loss) => return Proof::Holds,
            None if !facts.forced.is_empty() => return Proof::Holds,
            None => {}
        }
        if *left == 0 {
            return Proof::Exhausted;
        }
        let nodes = audit.nodes.min(*left);
        self.counters.audit_calls += 1;
        let solved = self.solver.search_quiet(post, audit.turns, nodes);
        *left -= solved.nodes.min(*left);
        match solved.verdict {
            Verdict::Win { turns, .. } => Proof::Lost(turns),
            // The total, not the call's own budget, cut this call short.
            Verdict::Unknown { exhausted: true } if nodes < audit.nodes => Proof::Exhausted,
            _ => Proof::Holds,
        }
    }

    /// First stone `c`'s audit over its top `m` second stones: its children by rank, then the forced blocks after it.
    fn first_stone(
        &mut self,
        audit: AuditConfig,
        root: &Board,
        pool: &[Node],
        c: u32,
        left: &mut u64,
    ) -> FirstStone {
        let Some(after_c) = after(root, &[pool[c as usize].cell()]) else {
            return FirstStone::Untried;
        };
        let mut seconds: Vec<(i32, i32)> = ranked_children(pool, c)
            .into_iter()
            .map(|i| pool[i as usize].cell())
            .collect();
        for cell in analyze(&after_c).forced {
            if !seconds.contains(&cell) {
                seconds.push(cell);
            }
        }
        seconds.truncate(audit.m as usize);
        let mut lost = Vec::new();
        for d2 in seconds {
            let Some(post) = after(&after_c, &[d2]) else {
                continue;
            };
            match self.opponent_proof(audit, &post, left) {
                Proof::Holds => return FirstStone::HeldBy(d2),
                Proof::Exhausted => return FirstStone::Exhausted,
                Proof::Lost(n) => lost.push((n, (n, d2))),
            }
        }
        best_hold(&lost).map_or(FirstStone::Untried, |(n, d2)| FirstStone::Lost(n, d2))
    }

    /// One stone left: `chosen`, or on its proven loss the stored hold stone, the alternatives, then the best hold.
    fn hold_last_stone(
        &mut self,
        audit: AuditConfig,
        root: &Board,
        chosen: (i32, i32),
        rest: &[(i32, i32)],
        left: &mut u64,
    ) -> (i32, i32) {
        let key = position_key(root);
        let hold = self
            .next_hold_stone
            .take()
            .filter(|&(k, s)| k == key && s != chosen && root.legal_moves_set().contains(&s))
            .map(|(_, s)| s);
        let Some(post) = after(root, &[chosen]) else {
            return chosen;
        };
        let n = match self.opponent_proof(audit, &post, left) {
            Proof::Holds => return chosen,
            Proof::Exhausted => {
                self.counters.audit_exhausted += 1;
                return chosen;
            }
            Proof::Lost(n) => n,
        };
        self.counters.root_vetoes += 1;
        let mut lost = vec![(n, chosen)];
        let walk = hold
            .into_iter()
            .chain(rest.iter().copied().filter(|&c| Some(c) != hold));
        for cell in walk {
            let Some(post) = after(root, &[cell]) else {
                continue;
            };
            match self.opponent_proof(audit, &post, left) {
                Proof::Holds => return cell,
                Proof::Exhausted => {
                    self.counters.audit_exhausted += 1;
                    return chosen;
                }
                Proof::Lost(n) => lost.push((n, cell)),
            }
        }
        self.counters.best_holds += 1;
        best_hold(&lost).unwrap_or(chosen)
    }

    /// Two stones left: `chosen` unless no second stone holds it, then the alternatives; a hold's second is stored.
    fn hold_first_stone(
        &mut self,
        audit: AuditConfig,
        root: &Board,
        pool: &[Node],
        cands: (u32, &[u32]),
        left: &mut u64,
    ) -> (i32, i32) {
        let (chosen_idx, rest) = cands;
        let chosen = pool[chosen_idx as usize].cell();
        let key = position_key(root);
        let mover = root.current_player;
        let mut lost = Vec::new();
        for (i, &c) in std::iter::once(&chosen_idx).chain(rest).enumerate() {
            let cell = pool[c as usize].cell();
            match self.first_stone(audit, root, pool, c, left) {
                FirstStone::HeldBy(d2) => {
                    self.next_hold_stone = Some((with_stone(key, cell, mover), d2));
                    return cell;
                }
                FirstStone::Untried => return cell,
                FirstStone::Exhausted => {
                    self.counters.audit_exhausted += 1;
                    return chosen;
                }
                FirstStone::Lost(n, d2) => {
                    if i == 0 {
                        self.counters.root_vetoes += 1;
                    }
                    lost.push((n, (cell, d2)));
                }
            }
        }
        self.counters.best_holds += 1;
        let Some((cell, d2)) = best_hold(&lost) else {
            return chosen;
        };
        self.next_hold_stone = Some((with_stone(key, cell, mover), d2));
        cell
    }

    /// The known-bad: the first candidate (`chosen` first) whose turn allows a proven opponent win, else `chosen`.
    fn invert(
        &mut self,
        audit: AuditConfig,
        root: &Board,
        pool: &[Node],
        cands: (u32, &[u32]),
        left: &mut u64,
    ) -> (i32, i32) {
        let (chosen_idx, rest) = cands;
        let chosen = pool[chosen_idx as usize].cell();
        for (i, &c) in std::iter::once(&chosen_idx).chain(rest).enumerate() {
            let cell = pool[c as usize].cell();
            let allows = if root.moves_remaining == 1 {
                match after(root, &[cell]) {
                    Some(post) => match self.opponent_proof(audit, &post, left) {
                        Proof::Lost(_) => Some(true),
                        Proof::Holds => Some(false),
                        Proof::Exhausted => None,
                    },
                    None => Some(false),
                }
            } else {
                match self.first_stone(audit, root, pool, c, left) {
                    FirstStone::Lost(..) => Some(true),
                    FirstStone::Exhausted => None,
                    FirstStone::HeldBy(_) | FirstStone::Untried => Some(false),
                }
            };
            match allows {
                Some(true) => {
                    if i > 0 {
                        self.counters.root_vetoes += 1;
                    }
                    return cell;
                }
                Some(false) => {}
                None => {
                    self.counters.audit_exhausted += 1;
                    return chosen;
                }
            }
        }
        chosen
    }
}

impl MCTSTree {
    /// Before the search: a finish, the last call's proof stone or a new proof's first stone (unsearched), else `None`.
    pub fn root_offence(&mut self) -> Result<Option<(i32, i32)>, TacticsError> {
        let root = &self.root_board;
        let Some(t) = self.tactics.as_deref_mut() else {
            return Ok(None);
        };
        let key = position_key(root);
        let stored = t
            .next_proof_stone
            .take()
            .filter(|&(k, _)| k == key)
            .map(|(_, s)| s);
        if root.check_win() {
            return Ok(None);
        }
        let facts = analyze(root);
        let decided = match facts.terminal {
            Some(Terminal::Win) => facts.finish.first().map(|&s| (s, Decided::Finish)),
            Some(Terminal::Loss) => {
                t.counters.decided_lost += 1;
                None
            }
            None if !facts.forced.is_empty() => None,
            None => match stored {
                Some(s) => Some((s, Decided::StoredProof)),
                None if root.moves_remaining == 2 && t.config.root_nodes > 0 => {
                    let solved =
                        t.solver
                            .search_quiet(root, t.config.root_turns, t.config.root_nodes);
                    match solved.verdict {
                        Verdict::Win { first, .. } => match *first.stones() {
                            [a, b] => {
                                t.next_proof_stone =
                                    Some((with_stone(key, a, root.current_player), b));
                                Some((a, Decided::Proof))
                            }
                            [a] => Some((a, Decided::Proof)),
                            _ => None,
                        },
                        Verdict::Unknown { exhausted: true } => {
                            t.counters.root_solver_exhausted += 1;
                            None
                        }
                        _ => None,
                    }
                }
                None => None,
            },
        };
        let Some(((q, r), row)) = decided else {
            return Ok(None);
        };
        if !root.legal_moves_set().contains(&(q, r)) {
            t.counters.proof_stone_illegal += 1;
            t.next_proof_stone = None;
            return Err(TacticsError::ProofStoneIllegal { q, r });
        }
        match row {
            Decided::Finish => t.counters.finishes_played += 1,
            Decided::StoredProof => t.counters.proof_stones_played += 1,
            Decided::Proof => t.counters.root_proofs_found += 1,
        }
        Ok(Some((q, r)))
    }

    /// After the search: `chosen`, or the audit's hold for a turn allowing a proven opponent win; `order` ranks the root.
    pub fn root_audit(&mut self, chosen: (i32, i32), order: Option<&[u32]>) -> (i32, i32) {
        let (root, pool) = (&self.root_board, &self.pool[..]);
        let Some(t) = self.tactics.as_deref_mut() else {
            return chosen;
        };
        let Some(audit) = t.config.audit else {
            return chosen;
        };
        let r0 = &pool[0];
        let is_child = |i: &u32| {
            r0.is_expanded()
                && (r0.first_child..r0.first_child + u32::from(r0.n_children)).contains(i)
        };
        let ranked = order.map_or_else(|| ranked_children(pool, 0), <[u32]>::to_vec);
        let Some(first) = ranked
            .iter()
            .copied()
            .find(|i| is_child(i) && pool[*i as usize].cell() == chosen)
        else {
            return chosen;
        };
        let rest: Vec<u32> = ranked
            .iter()
            .copied()
            .filter(|i| *i != first && is_child(i))
            .take(audit.k as usize)
            .collect();
        let mut left = audit.total_nodes;
        match (audit.mode, root.moves_remaining) {
            (AuditMode::Inverted, _) => t.invert(audit, root, pool, (first, &rest), &mut left),
            (AuditMode::Hold, 1) => {
                let cells: Vec<(i32, i32)> =
                    rest.iter().map(|&i| pool[i as usize].cell()).collect();
                t.hold_last_stone(audit, root, chosen, &cells, &mut left)
            }
            (AuditMode::Hold, _) => {
                t.hold_first_stone(audit, root, pool, (first, &rest), &mut left)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::node::pack_cell;
    use crate::mcts::tactics_fixtures::{played, FIX219, P1_FOUR, P2_FOUR_ONE_LEFT};
    use crate::mcts::{TacticsConfig, TacticsCounters};

    /// Small audit budgets: FIX219's one-turn wins prove in one node, so every call here is cheap.
    const AUDIT: AuditConfig = AuditConfig {
        turns: 2,
        nodes: 256,
        k: 4,
        m: 2,
        total_nodes: 1 << 16,
        mode: AuditMode::Hold,
    };
    const ARMED: TacticsConfig = TacticsConfig {
        leaf_turns: 2,
        leaf_nodes: 64,
        root_turns: 2,
        root_nodes: 256,
        audit: Some(AUDIT),
    };
    /// P1 to move with ONE stone at `FIX219[..16]`: of 636 legal cells only these five stop P2's one-turn win.
    const HOLDS_16: [(i32, i32); 5] = [(-4, -12), (-4, -11), (-4, -10), (-3, -12), (-2, -13)];

    fn armed(board: &Board, config: TacticsConfig) -> MCTSTree {
        let mut tree = MCTSTree::new(1.5);
        tree.configure_tactics(Some(config));
        tree.new_game(board.clone());
        tree
    }

    fn with_audit(audit: AuditConfig) -> TacticsConfig {
        TacticsConfig {
            audit: Some(audit),
            ..ARMED
        }
    }

    /// Expand `parent` over `cells`, visits falling by one a rank from `top`; the children's pool indices.
    fn expand(tree: &mut MCTSTree, parent: u32, cells: &[(i32, i32)], top: u32) -> Vec<u32> {
        let first = tree.next_free;
        tree.pool[parent as usize].first_child = first;
        tree.pool[parent as usize].n_children = cells.len() as u16;
        for (j, &(q, r)) in cells.iter().enumerate() {
            tree.pool[first as usize + j] = Node {
                parent,
                action_idx: pack_cell(q, r),
                n_visits: top - j as u32,
                prior: 1.0 / cells.len() as f32,
                ..Node::uninit()
            };
        }
        tree.next_free = first + cells.len() as u32;
        (first..tree.next_free).collect()
    }

    fn counters(tree: &MCTSTree) -> TacticsCounters {
        tree.tactics_counters()
    }

    #[test]
    fn a_proven_root_plays_its_first_stone_and_the_next_call_the_stored_second_with_no_search() {
        let board = played(&FIX219);
        let mut tree = armed(&board, ARMED);
        assert_eq!(tree.root_offence(), Ok(Some((-4, -11))));
        assert_eq!(counters(&tree).root_proofs_found, 1);
        let mut next = board.clone();
        next.apply_move(-4, -11).expect("empty");
        tree.new_game(next);
        assert_eq!(tree.root_offence(), Ok(Some((-4, -10))));
        let c = counters(&tree);
        assert_eq!((c.proof_stones_played, c.root_proofs_found), (1, 0));
        assert_eq!(c.descents, 0, "a decided root spends no descent");
    }

    #[test]
    fn a_stored_stone_is_valid_on_its_own_position_only() {
        let board = played(&FIX219);
        let mut tree = armed(&board, ARMED);
        assert_eq!(tree.root_offence(), Ok(Some((-4, -11))));
        let mut elsewhere = board.clone();
        elsewhere.apply_move(-4, -10).expect("empty");
        tree.new_game(elsewhere);
        assert_ne!(tree.root_offence(), Ok(Some((-4, -10))));
        assert_eq!(counters(&tree).proof_stones_played, 0);
    }

    #[test]
    fn a_finish_is_played_at_both_calls_and_a_lost_cover_is_counted_and_searched() {
        let board = played(&P1_FOUR);
        let mut tree = armed(&board, ARMED);
        assert_eq!(tree.root_offence(), Ok(Some((-2, 0))));
        let mut next = board.clone();
        next.apply_move(-2, 0).expect("empty");
        tree.new_game(next);
        assert_eq!(tree.root_offence(), Ok(Some((-1, 0))));
        assert_eq!(counters(&tree).finishes_played, 1);
        let mut lost = armed(&played(&P2_FOUR_ONE_LEFT), ARMED);
        assert_eq!(lost.root_offence(), Ok(None));
        assert_eq!(counters(&lost).decided_lost, 1);
    }

    #[test]
    fn a_root_solve_out_of_nodes_is_counted_and_the_root_searched() {
        let config = TacticsConfig {
            root_nodes: 1,
            ..ARMED
        };
        let mut tree = armed(&played(&FIX219[..15]), config);
        assert_eq!(tree.root_offence(), Ok(None));
        assert_eq!(counters(&tree).root_solver_exhausted, 1);
    }

    #[test]
    fn an_illegal_decided_stone_is_refused_counted_and_not_played() {
        let root = played(&FIX219[..16]);
        let mut tree = armed(&root, ARMED);
        let key = position_key(&root);
        if let Some(t) = tree.tactics.as_deref_mut() {
            t.next_proof_stone = Some((key, (60, 60)));
        }
        assert_eq!(
            tree.root_offence(),
            Err(TacticsError::ProofStoneIllegal { q: 60, r: 60 })
        );
        assert_eq!(counters(&tree).proof_stone_illegal, 1);
    }

    /// `FIX219[..16]` with the chosen `(-10, 0)` (it lets P2 win) most visited, then `rest` in order.
    fn last_stone_root(rest: &[(i32, i32)], config: TacticsConfig) -> MCTSTree {
        let root = played(&FIX219[..16]);
        let mut tree = armed(&root, config);
        let cells: Vec<(i32, i32)> = std::iter::once((-10, 0))
            .chain(rest.iter().copied())
            .collect();
        expand(&mut tree, 0, &cells, 20);
        tree
    }

    #[test]
    fn a_turn_that_allows_a_proven_win_is_swapped_for_the_first_that_holds_in_the_kinds_order() {
        // PLANTED BREAK 4: the audit keeping the vetoed move reds this test.
        let mut tree = last_stone_root(&[(-20, 0), (-21, 0), HOLDS_16[0], HOLDS_16[1]], ARMED);
        assert_eq!(tree.root_audit((-10, 0), None), HOLDS_16[0]);
        let c = counters(&tree);
        assert_eq!((c.root_vetoes, c.best_holds, c.audit_calls), (1, 0, 4));
    }

    #[test]
    fn a_chosen_turn_that_holds_stands_and_with_no_hold_in_k_the_best_hold_is_the_first_of_equals()
    {
        let mut tree = last_stone_root(&[HOLDS_16[0]], ARMED);
        assert_eq!(tree.root_audit(HOLDS_16[0], None), HOLDS_16[0]);
        assert_eq!(counters(&tree).root_vetoes, 0);
        let narrow = with_audit(AuditConfig { k: 2, ..AUDIT });
        let mut tree = last_stone_root(&[(-20, 0), (-21, 0), HOLDS_16[0]], narrow);
        assert_eq!(tree.root_audit((-10, 0), None), (-10, 0));
        let c = counters(&tree);
        assert_eq!((c.root_vetoes, c.best_holds, c.audit_calls), (1, 1, 3));
    }

    #[test]
    fn best_hold_takes_the_longest_proof_and_the_first_of_equals() {
        assert_eq!(
            best_hold(&[(1, 'a'), (2, 'b'), (2, 'c'), (0, 'd')]),
            Some('b')
        );
        assert_eq!(best_hold(&[(1, 'a'), (1, 'b')]), Some('a'));
        assert_eq!(best_hold::<char>(&[]), None);
    }

    #[test]
    fn the_total_budget_spent_leaves_the_kinds_choice_standing() {
        let tight = with_audit(AuditConfig {
            total_nodes: 1,
            ..AUDIT
        });
        let mut tree = last_stone_root(&[HOLDS_16[0]], tight);
        assert_eq!(tree.root_audit((-10, 0), None), (-10, 0));
        let c = counters(&tree);
        assert_eq!((c.root_vetoes, c.audit_exhausted), (1, 1));
    }

    #[test]
    fn at_two_stones_left_a_first_stone_holds_through_its_top_m_second_stones_and_stores_the_hold()
    {
        let root = played(&FIX219[..15]);
        for (m, want, second) in [(2, (-4, -11), (-10, 0)), (3, (-13, 0), (-4, -10))] {
            let mut tree = armed(&root, with_audit(AuditConfig { m, ..AUDIT }));
            let firsts = expand(&mut tree, 0, &[(-13, 0), (-12, 0), (-4, -11)], 10);
            expand(&mut tree, firsts[0], &[(-10, 0), (-12, 1), (-4, -10)], 3);
            expand(&mut tree, firsts[1], &[(-10, 0), (-12, 1)], 2);
            expand(&mut tree, firsts[2], &[(-10, 0)], 1);
            assert_eq!(tree.root_audit((-13, 0), None), want, "m = {m}");
            let mut after_first = root.clone();
            after_first.apply_move(want.0, want.1).expect("empty");
            let stored = tree.tactics.as_deref().and_then(|t| t.next_hold_stone);
            assert_eq!(
                stored,
                Some((position_key(&after_first), second)),
                "m = {m}"
            );
            assert_eq!(counters(&tree).root_vetoes, u64::from(m == 2));
        }
    }

    #[test]
    fn the_next_call_tries_the_stored_hold_stone_before_the_candidates() {
        let root = played(&FIX219[..15]);
        let mut tree = armed(&root, with_audit(AuditConfig { m: 3, ..AUDIT }));
        let firsts = expand(&mut tree, 0, &[(-13, 0)], 10);
        expand(&mut tree, firsts[0], &[(-10, 0), (-12, 1), (-4, -10)], 3);
        assert_eq!(tree.root_audit((-13, 0), None), (-13, 0));
        // The second call, a fresh search on the same tree: its pick loses, and the stored stone is tried first.
        let mut root16 = root.clone();
        root16.apply_move(-13, 0).expect("empty");
        tree.new_game(root16);
        expand(&mut tree, 0, &[(-10, 0), (-20, 0), HOLDS_16[0]], 20);
        assert_eq!(tree.root_audit((-10, 0), None), (-4, -10));
        assert_eq!(counters(&tree).audit_calls, 2);
    }

    #[test]
    fn the_inverted_known_bad_plays_into_a_proven_win_when_its_candidates_allow_one() {
        let inverted = with_audit(AuditConfig {
            mode: AuditMode::Inverted,
            ..AUDIT
        });
        let root = played(&FIX219[..16]);
        let mut tree = armed(&root, inverted);
        expand(&mut tree, 0, &[HOLDS_16[0], HOLDS_16[1], (-20, 0)], 20);
        assert_eq!(tree.root_audit(HOLDS_16[0], None), (-20, 0));
        assert_eq!(counters(&tree).root_vetoes, 1);
        let mut tree = last_stone_root(&[HOLDS_16[0]], inverted);
        assert_eq!(tree.root_audit((-10, 0), None), (-10, 0));
    }

    #[test]
    fn the_audit_walks_the_order_it_is_given_and_leaves_a_move_that_is_not_a_root_child() {
        let mut tree = last_stone_root(&[HOLDS_16[0], HOLDS_16[1]], ARMED);
        let children = ranked_children(&tree.pool, 0);
        let given = [children[0], children[2], children[1]];
        assert_eq!(tree.root_audit((-10, 0), Some(&given)), HOLDS_16[1]);
        let mut tree = last_stone_root(&[HOLDS_16[0]], ARMED);
        assert_eq!(tree.root_audit((5, 5), None), (5, 5));
        assert_eq!(counters(&tree).audit_calls, 0);
        let mut off = MCTSTree::new(1.5);
        off.new_game(played(&FIX219[..16]));
        assert_eq!(off.root_audit((-10, 0), None), (-10, 0));
        assert_eq!(off.root_offence(), Ok(None));
    }
}
