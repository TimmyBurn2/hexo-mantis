// >300 justify (R8): the leaf decision, its block and rows, and the tests that need pool internals are one unit.
//! The tactics module on the search path: a decided leaf is a terminal backed up inline, a counted descent.

use mantis_core::board::Board;

use super::MCTSTree;
use crate::tactics::{analyze, LeafTactics, Terminal, TurnSolver, Verdict};

/// Every tree's solver table: 2^18 slots, as Six's 8 MiB.
pub const TACTICS_TABLE_ENTRIES: usize = 1 << 18;

pub use crate::tactics::solver::MIN_TACTICS_RADIUS;

/// What the defence audit does with a proven opponent win: hold (the lever), or the known-bad that plays into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditMode {
    Hold,
    Inverted,
}

/// The defence audit's budgets: per call, candidates walked, second stones tried, and one total for a move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuditConfig {
    pub turns: u8,
    pub nodes: u64,
    pub k: u32,
    pub m: u32,
    pub total_nodes: u64,
    pub mode: AuditMode,
}

/// The tactics block a tree runs; a tree without one searches exactly as before the module existed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TacticsConfig {
    pub leaf_turns: u8,
    pub leaf_nodes: u64,
    pub root_turns: u8,
    pub root_nodes: u64,
    pub audit: Option<AuditConfig>,
}

/// A refusal on the tactics path, each a state the wiring will not run through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TacticsError {
    /// A board whose legal-move radius leaves some window empty illegal: the solver's stones could be unplayable.
    RadiusBelowFive { radius: i32 },
    /// A proof, next-proof or finishing stone that is not in the root board's legal set.
    ProofStoneIllegal { q: i32, r: i32 },
}

impl std::fmt::Display for TacticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RadiusBelowFive { radius } => write!(
                f,
                "TacticsError::RadiusBelowFive: the board's legal-move radius is {radius}, below \
                 {MIN_TACTICS_RADIUS}; a window's empties may then be illegal cells, so tactics cannot run"
            ),
            Self::ProofStoneIllegal { q, r } => write!(
                f,
                "TacticsError::ProofStoneIllegal: the solver's stone ({q}, {r}) is not in the root's legal \
                 set; it is not played, and the search's own move stands"
            ),
        }
    }
}

impl std::error::Error for TacticsError {}

/// One search's in-run rows, reset by `new_game`; the select calls count the descents.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TacticsCounters {
    pub descents: u64,
    pub served_leaves: u64,
    pub terminal_win1: u64,
    pub terminal_lost_on_cover: u64,
    pub terminal_strict_win: u64,
    pub terminal_six: u64,
    pub terminal_revisits: u64,
    pub forced_restrictions: u64,
    pub leaf_solver_calls: u64,
    pub leaf_solver_exhausted: u64,
    pub root_proofs_found: u64,
    pub proof_stones_played: u64,
    pub finishes_played: u64,
    pub root_solver_exhausted: u64,
    pub decided_lost: u64,
    pub root_vetoes: u64,
    pub best_holds: u64,
    pub audit_calls: u64,
    pub audit_exhausted: u64,
    pub proof_stone_illegal: u64,
}

impl TacticsCounters {
    /// Solver terminals over every kind: the descents that ended at a leaf the module decided.
    #[must_use]
    pub fn solver_terminals(&self) -> u64 {
        self.terminal_win1
            + self.terminal_lost_on_cover
            + self.terminal_strict_win
            + self.terminal_six
    }
}

/// The tree's tactics state: its block, its solver (allocated once), this search's rows and the stored stones.
pub(crate) struct TacticsState {
    pub(crate) config: TacticsConfig,
    pub(crate) solver: TurnSolver,
    pub(crate) counters: TacticsCounters,
}

/// How a descent's leaf is spent: backed up inline, or evaluated by the net with the facts its expansion reads.
pub(crate) enum LeafCall {
    Inline,
    Evaluate(Option<LeafTactics>),
}

/// A decided leaf's backed-up value for its side to move, and the row it counts in.
enum Decided {
    Revisit(f32),
    Terminal(f32, fn(&mut TacticsCounters)),
    Evaluate(LeafTactics),
}

impl MCTSTree {
    /// Arm or disarm the tactics block, once per worker or player. Pure state set, surviving `new_game`.
    pub fn configure_tactics(&mut self, config: Option<TacticsConfig>) {
        self.tactics = config.map(|config| {
            Box::new(TacticsState {
                config,
                solver: TurnSolver::new(TACTICS_TABLE_ENTRIES),
                counters: TacticsCounters::default(),
            })
        });
    }

    /// The armed block, or `None` when tactics are off.
    #[must_use]
    pub fn tactics_config(&self) -> Option<TacticsConfig> {
        self.tactics.as_ref().map(|t| t.config)
    }

    /// This search's tactics rows; all zero with tactics off.
    #[must_use]
    pub fn tactics_counters(&self) -> TacticsCounters {
        self.tactics
            .as_ref()
            .map(|t| t.counters)
            .unwrap_or_default()
    }

    /// Descents the last `select_leaves` / `select_leaves_forced` call backed up inline, beside the boards it returned.
    #[must_use]
    pub fn last_inline_descents(&self) -> usize {
        self.inline_descents
    }

    /// Refuse, as `TacticsError::RadiusBelowFive`, a board whose legal-move radius the armed block cannot run on.
    pub fn check_tactics_board(&self, board: &Board) -> Result<(), TacticsError> {
        let radius = board.legal_move_radius();
        if self.tactics.is_some() && radius < MIN_TACTICS_RADIUS {
            return Err(TacticsError::RadiusBelowFive { radius });
        }
        Ok(())
    }

    /// The per-search reset `new_game` runs: this search's rows, and the solver's table.
    pub(crate) fn reset_tactics_search(&mut self) {
        if let Some(t) = self.tactics.as_deref_mut() {
            t.counters = TacticsCounters::default();
            t.solver.clear();
        }
    }

    /// The tactics decision at a descent's leaf; `Evaluate(None)` with tactics off. The root is never decided.
    pub(crate) fn tactics_leaf(&mut self, leaf_idx: u32, board: &Board) -> LeafCall {
        let node = self.pool[leaf_idx as usize];
        let Some(t) = self.tactics.as_deref_mut() else {
            return LeafCall::Evaluate(None);
        };
        let decided = if node.is_terminal {
            Decided::Revisit(node.terminal_value)
        } else if board.check_win() {
            // CF-1: `mr == 1` means the side that made six is still to move.
            let tv = if board.moves_remaining == 1 {
                1.0
            } else {
                -1.0
            };
            Decided::Terminal(tv, |c| c.terminal_six += 1)
        } else {
            let facts = analyze(board);
            match facts.terminal {
                _ if leaf_idx == 0 => Decided::Evaluate(facts),
                Some(Terminal::Win) => Decided::Terminal(1.0, |c| c.terminal_win1 += 1),
                Some(Terminal::Loss) => Decided::Terminal(-1.0, |c| c.terminal_lost_on_cover += 1),
                None if facts.forced.is_empty()
                    && board.moves_remaining == 2
                    && t.config.leaf_nodes > 0 =>
                {
                    t.counters.leaf_solver_calls += 1;
                    let solved =
                        t.solver
                            .search_quiet(board, t.config.leaf_turns, t.config.leaf_nodes);
                    match solved.verdict {
                        Verdict::Win { .. } => {
                            Decided::Terminal(1.0, |c| c.terminal_strict_win += 1)
                        }
                        Verdict::Unknown { exhausted: true } => {
                            t.counters.leaf_solver_exhausted += 1;
                            Decided::Evaluate(facts)
                        }
                        _ => Decided::Evaluate(facts),
                    }
                }
                None => Decided::Evaluate(facts),
            }
        };
        match decided {
            Decided::Revisit(tv) => {
                t.counters.terminal_revisits += 1;
                self.backup(leaf_idx, tv);
                LeafCall::Inline
            }
            Decided::Terminal(tv, count) => {
                count(&mut t.counters);
                let n = &mut self.pool[leaf_idx as usize];
                n.is_terminal = true;
                n.terminal_value = tv;
                self.backup(leaf_idx, tv);
                LeafCall::Inline
            }
            Decided::Evaluate(facts) => LeafCall::Evaluate(Some(facts)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::node::{pack_cell, Node};
    use crate::tactics::Terminal;

    /// The design's leaf budgets with no root offence or audit: what the leaf wiring alone reads.
    const LEAF_ONLY: TacticsConfig = TacticsConfig {
        leaf_turns: 2,
        leaf_nodes: 64,
        root_turns: 8,
        root_nodes: 0,
        audit: None,
    };

    /// T2's `fix219@17`: P2 to move with two stones holds a one-turn strict win at (-4, -11), (-4, -10).
    const FIX219: [(i32, i32); 17] = [
        (-1, -8),
        (-4, -8),
        (-4, -9),
        (-5, -8),
        (0, -10),
        (1, -13),
        (-6, -9),
        (-5, -7),
        (3, -13),
        (-6, -12),
        (-5, -10),
        (-6, -15),
        (4, -16),
        (-1, -14),
        (2, -15),
        (-13, 0),
        (-10, 0),
    ];

    /// P1's four (0..3, 0) and P2's scatter; P1 is to move with two stones after the last.
    const P1_FOUR: [(i32, i32); 11] = [
        (0, 0),
        (20, 20),
        (22, 20),
        (1, 0),
        (2, 0),
        (20, 24),
        (22, 24),
        (3, 0),
        (-9, 9),
        (30, 30),
        (32, 30),
    ];

    /// P2's open four at (10..13, 5); after the last stone P1 is to move with ONE stone.
    const P2_FOUR_ONE_LEFT: [(i32, i32); 8] = [
        (0, 0),
        (10, 5),
        (11, 5),
        (1, 0),
        (2, 0),
        (12, 5),
        (13, 5),
        (-5, 0),
    ];

    fn played(seq: &[(i32, i32)]) -> Board {
        let mut b = Board::new();
        b.set_legal_move_radius(8);
        for &(q, r) in seq {
            b.apply_move(q, r)
                .expect("a test sequence places on empty cells");
        }
        b
    }

    /// A tactics-armed tree at `seq[..root]` with one node path down the rest: the tree, its leaf and the leaf's board.
    fn path_to_leaf(
        seq: &[(i32, i32)],
        root: usize,
        config: TacticsConfig,
    ) -> (MCTSTree, u32, Board) {
        let mut tree = MCTSTree::new(1.5);
        tree.configure_tactics(Some(config));
        let mut board = played(&seq[..root]);
        tree.new_game(board.clone());
        let mut parent = 0u32;
        for (depth, &(q, r)) in seq[root..].iter().enumerate() {
            board.apply_move(q, r).expect("the path's stone is empty");
            let idx = depth as u32 + 1;
            tree.pool[parent as usize].first_child = idx;
            tree.pool[parent as usize].n_children = 1;
            tree.pool[idx as usize] = Node {
                parent,
                action_idx: pack_cell(q, r),
                prior: 1.0,
                moves_remaining: board.moves_remaining,
                ..Node::uninit()
            };
            parent = idx;
        }
        tree.next_free = parent + 1;
        (tree, parent, board)
    }

    #[test]
    fn a_quiet_leaf_whose_mover_wins_strictly_is_a_plus_one_terminal_with_no_net_call() {
        let (mut tree, leaf, board) = path_to_leaf(&FIX219, 15, LEAF_ONLY);
        assert!(matches!(tree.tactics_leaf(leaf, &board), LeafCall::Inline));
        let node = tree.pool[leaf as usize];
        assert!(node.is_terminal);
        assert_eq!(
            node.terminal_value, 1.0,
            "+1 for the leaf's own side to move"
        );
        let c = tree.tactics_counters();
        assert_eq!((c.leaf_solver_calls, c.terminal_strict_win), (1, 1));
        // P1 at the root placed both stones of the turn that let P2 win: -1 from the root's side.
        assert_eq!((tree.pool[0].n_visits, tree.pool[0].w_value), (1, -1.0));
    }

    #[test]
    fn a_finishing_mover_is_plus_one_and_a_lost_cover_minus_one() {
        let (mut tree, leaf, board) = path_to_leaf(&P1_FOUR, 9, LEAF_ONLY);
        assert!(matches!(tree.tactics_leaf(leaf, &board), LeafCall::Inline));
        assert_eq!(tree.pool[leaf as usize].terminal_value, 1.0);
        assert_eq!(tree.tactics_counters().terminal_win1, 1);

        let (mut tree, leaf, board) = path_to_leaf(&P2_FOUR_ONE_LEFT, 6, LEAF_ONLY);
        assert_eq!(board.moves_remaining, 1);
        assert!(matches!(tree.tactics_leaf(leaf, &board), LeafCall::Inline));
        assert_eq!(tree.pool[leaf as usize].terminal_value, -1.0);
        assert_eq!(tree.tactics_counters().terminal_lost_on_cover, 1);
    }

    #[test]
    fn a_revisited_terminal_is_backed_up_inline_and_counted() {
        let (mut tree, leaf, board) = path_to_leaf(&FIX219, 15, LEAF_ONLY);
        assert!(matches!(tree.tactics_leaf(leaf, &board), LeafCall::Inline));
        assert!(matches!(tree.tactics_leaf(leaf, &board), LeafCall::Inline));
        assert_eq!(tree.tactics_counters().terminal_revisits, 1);
        assert_eq!(tree.pool[0].n_visits, 2);
    }

    #[test]
    fn the_root_is_never_decided_and_carries_its_facts() {
        let mut tree = MCTSTree::new(1.5);
        tree.configure_tactics(Some(LEAF_ONLY));
        let root = played(&P1_FOUR);
        tree.new_game(root.clone());
        match tree.tactics_leaf(0, &root) {
            LeafCall::Evaluate(Some(facts)) => assert_eq!(facts.terminal, Some(Terminal::Win)),
            _ => panic!("the root is evaluated, whatever its facts"),
        }
        assert!(!tree.pool[0].is_terminal);
    }

    fn uniform() -> Vec<f32> {
        let n = mantis_core::board::BOARD_SIZE * mantis_core::board::BOARD_SIZE + 1;
        vec![1.0 / n as f32; n]
    }

    fn root_children(tree: &MCTSTree) -> Vec<((i32, i32), f32)> {
        let root = tree.pool[0];
        (root.first_child..root.first_child + u32::from(root.n_children))
            .map(|i| (tree.pool[i as usize].cell(), tree.pool[i as usize].prior))
            .collect()
    }

    /// Expand the root once with a uniform policy; `select_leaves(1)` must hand the root itself to the net.
    fn expand_root(tree: &mut MCTSTree) {
        let leaves = tree.select_leaves(1).expect("no desync");
        assert_eq!(
            leaves.len(),
            1,
            "the root goes to the net, whatever its facts"
        );
        tree.expand_and_backup(&[uniform()], &[0.0]);
    }

    #[test]
    fn a_forced_block_restricts_the_children_to_its_cells_with_priors_renormalised() {
        let root = played(&P2_FOUR_ONE_LEFT[..7]);
        assert_eq!(root.moves_remaining, 2);
        let mut tree = MCTSTree::new(1.5);
        tree.configure_tactics(Some(LEAF_ONLY));
        tree.new_game(root.clone());
        expand_root(&mut tree);
        let children = root_children(&tree);
        let mut cells: Vec<(i32, i32)> = children.iter().map(|c| c.0).collect();
        cells.sort_unstable();
        assert_eq!(cells, vec![(8, 5), (9, 5), (14, 5), (15, 5)]);
        let mass: f32 = children.iter().map(|c| c.1).sum();
        assert!(
            (mass - 1.0).abs() < 1e-5,
            "priors renormalised over the forced cells: {mass}"
        );
        assert_eq!(tree.tactics_counters().forced_restrictions, 1);

        let mut plain = MCTSTree::new(1.5);
        plain.new_game(root.clone());
        expand_root(&mut plain);
        assert_eq!(
            root_children(&plain).len(),
            root.legal_move_count(),
            "tactics off expands the legal set"
        );
    }

    #[test]
    fn a_decided_root_is_expanded_over_its_legal_set() {
        for (seq, why) in [
            (&P1_FOUR[..], "a finish"),
            (&P2_FOUR_ONE_LEFT[..], "a lost cover"),
        ] {
            let root = played(seq);
            let mut tree = MCTSTree::new(1.5);
            tree.configure_tactics(Some(LEAF_ONLY));
            tree.new_game(root.clone());
            expand_root(&mut tree);
            assert!(
                !tree.pool[0].is_terminal,
                "{why}: the root is never a terminal"
            );
            assert_eq!(
                root_children(&tree).len(),
                root.legal_move_count(),
                "{why}: every legal child"
            );
        }
    }

    /// Drive `budget` descents the way the budget loops do, boards and inline descents both counted.
    fn drive(tree: &mut MCTSTree, budget: usize, batch: usize) -> usize {
        let mut done = 0;
        while done < budget {
            let boards = tree
                .select_leaves(batch.min(budget - done))
                .expect("no desync");
            let inline = tree.last_inline_descents();
            assert!(
                boards.len() + inline <= batch.min(budget - done),
                "a select call overspent its n"
            );
            if boards.is_empty() && inline == 0 {
                break;
            }
            let policies = vec![uniform(); boards.len()];
            tree.expand_and_backup(&policies, &vec![0.0; boards.len()]);
            done += boards.len() + inline;
        }
        done
    }

    #[test]
    fn a_search_spends_exactly_its_descents_and_the_rows_add_up() {
        // P1 to move against an open four: four forced children, so the descents reach decided leaves fast.
        let mut tree = MCTSTree::new(1.5);
        tree.configure_tactics(Some(LEAF_ONLY));
        tree.new_game(played(&P2_FOUR_ONE_LEFT[..7]));
        expand_root(&mut tree);
        let done = 1 + drive(&mut tree, 511, 8);
        assert_eq!(done, 512);
        let c = tree.tactics_counters();
        assert_eq!(c.descents, 512, "every counted descent is in the row");
        assert_eq!(
            c.served_leaves + c.solver_terminals() + c.terminal_revisits,
            c.descents
        );
        assert!(
            c.solver_terminals() > 0,
            "a position this sharp decides some leaves: {c:?}"
        );
    }

    /// P1's five (0..4, 0) and P2's scatter; P1 is to move with two stones after the last.
    const P1_FIVE: [(i32, i32); 11] = [
        (0, 0),
        (20, 20),
        (22, 20),
        (1, 0),
        (2, 0),
        (20, 24),
        (22, 24),
        (3, 0),
        (4, 0),
        (30, 30),
        (32, 30),
    ];

    #[test]
    fn a_six_is_plus_one_while_its_maker_holds_a_stone_and_minus_one_once_the_turn_has_passed() {
        let first: Vec<(i32, i32)> = P1_FIVE.iter().copied().chain([(5, 0)]).collect();
        let second: Vec<(i32, i32)> = P1_FIVE.iter().copied().chain([(-9, 9), (5, 0)]).collect();
        for (seq, want) in [(first, 1.0), (second, -1.0)] {
            let (mut tree, leaf, board) = path_to_leaf(&seq, P1_FIVE.len(), LEAF_ONLY);
            assert!(matches!(tree.tactics_leaf(leaf, &board), LeafCall::Inline));
            assert_eq!(tree.pool[leaf as usize].terminal_value, want);
            assert_eq!(tree.tactics_counters().terminal_six, 1);
            // P1 made the six from the root: a win from the root's side, whichever stone made it.
            assert_eq!((tree.pool[0].n_visits, tree.pool[0].w_value), (1, 1.0));
        }
    }

    #[test]
    fn a_leaf_solve_out_of_nodes_is_counted_and_the_leaf_evaluated() {
        let config = TacticsConfig {
            leaf_nodes: 1,
            ..LEAF_ONLY
        };
        let (mut tree, leaf, board) = path_to_leaf(&FIX219[..15], 13, config);
        assert!(matches!(
            tree.tactics_leaf(leaf, &board),
            LeafCall::Evaluate(Some(_))
        ));
        let c = tree.tactics_counters();
        assert_eq!((c.leaf_solver_calls, c.leaf_solver_exhausted), (1, 1));
        assert!(!tree.pool[leaf as usize].is_terminal);
    }

    #[test]
    fn the_legal_set_path_restricts_a_forced_leaf_to_its_blocks_as_the_dense_path_does() {
        let root = played(&P2_FOUR_ONE_LEFT[..7]);
        let mut tree = MCTSTree::new(1.5);
        tree.configure_tactics(Some(LEAF_ONLY));
        tree.new_game(root.clone());
        assert_eq!(tree.select_leaves(1).expect("no desync").len(), 1);
        let ls = crate::LegalSetPolicy {
            dense: uniform(),
            ..crate::LegalSetPolicy::default()
        };
        let trunk = root.cluster_window_size() as i32;
        tree.expand_and_backup_ls_at(&[ls], &[0.0], &[root.window_center()], trunk);
        let children = root_children(&tree);
        let mut cells: Vec<(i32, i32)> = children.iter().map(|c| c.0).collect();
        cells.sort_unstable();
        assert_eq!(cells, vec![(8, 5), (9, 5), (14, 5), (15, 5)]);
        let mass: f32 = children.iter().map(|c| c.1).sum();
        assert!((mass - 1.0).abs() < 1e-5, "renormalised: {mass}");
        assert_eq!(tree.tactics_counters().forced_restrictions, 1);
    }

    #[test]
    fn with_tactics_off_a_decided_leaf_goes_to_the_net_as_before() {
        let (mut tree, leaf, board) = path_to_leaf(&FIX219, 15, LEAF_ONLY);
        tree.configure_tactics(None);
        assert!(matches!(
            tree.tactics_leaf(leaf, &board),
            LeafCall::Evaluate(None)
        ));
        assert!(!tree.pool[leaf as usize].is_terminal);
        assert_eq!(tree.tactics_counters(), TacticsCounters::default());
    }
}
