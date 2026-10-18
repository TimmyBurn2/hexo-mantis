//! Within-turn tree reuse: the next search starts on a root child's kept subtree instead of an empty root.

use super::{MCTSTree, Node, SearchKind};
use mantis_core::board::{Board, Cell};

impl MCTSTree {
    /// `new_game(board)`, keeping the subtree of a turn's first stone's child that is exactly `board`; the visits kept.
    pub fn reroot_or_new_game(&mut self, board: Board) -> u32 {
        let kept = self.child_reaching(&board).map(|child| self.subtree(child));
        self.new_game(board);
        let Some(nodes) = kept else {
            return 0;
        };
        self.pool[..nodes.len()].copy_from_slice(&nodes);
        self.next_free = nodes.len() as u32;
        self.pool[0].n_visits
    }

    /// The expanded, undecided child reaching `board` from a PUCT turn's first stone (Gumbel halving reads root visits).
    fn child_reaching(&self, board: &Board) -> Option<u32> {
        let root = &self.pool[0];
        if self.kind != SearchKind::Puct
            || self.root_board.moves_remaining != 2
            || !root.is_expanded()
        {
            return None;
        }
        let first = root.first_child;
        (first..first + u32::from(root.n_children)).find(|&i| {
            let child = &self.pool[i as usize];
            let (q, r) = child.cell();
            child.is_expanded()
                && !child.is_terminal
                && board.get(q, r) != Cell::Empty
                && self
                    .root_board_plus(q, r)
                    .is_some_and(|next| same_position(&next, board))
        })
    }

    fn root_board_plus(&self, q: i32, r: i32) -> Option<Board> {
        let mut next = self.root_board.clone();
        next.apply_move(q, r).ok().map(|()| next)
    }

    /// `child`'s subtree from slot 0, breadth first, each node's children one contiguous block.
    fn subtree(&self, child: u32) -> Vec<Node> {
        let mut nodes = vec![Node {
            parent: u32::MAX,
            ..self.pool[child as usize]
        }];
        let mut at = 0;
        while at < nodes.len() {
            let node = nodes[at];
            if node.is_expanded() {
                let first = node.first_child as usize;
                nodes[at].first_child = nodes.len() as u32;
                for i in first..first + usize::from(node.n_children) {
                    nodes.push(Node {
                        parent: at as u32,
                        ..self.pool[i]
                    });
                }
            }
            at += 1;
        }
        nodes
    }
}

/// The exact eval-cache key's inputs, compared outright: the stones, the side to move, the stones left, the geometry.
fn same_position(a: &Board, b: &Board) -> bool {
    a.current_player == b.current_player
        && a.moves_remaining == b.moves_remaining
        && a.geometry() == b.geometry()
        && a.cells_iter().count() == b.cells_iter().count()
        && a.cells_iter().all(|(&(q, r), &cell)| b.get(q, r) == cell)
}

#[cfg(test)]
mod tests {
    use super::super::{QSigma, SearchKind};
    use super::*;
    use mantis_core::board::BOARD_SIZE;

    const STRIDE: usize = BOARD_SIZE * BOARD_SIZE + 1;

    /// A deterministic, position-keyed net: a peaked prior over the window and a value in [-0.5, 0.5].
    fn net(board: &Board) -> (Vec<f32>, f32) {
        let key = board.zobrist_hash;
        let mut policy = vec![0.0; STRIDE];
        for (q, r) in board.legal_moves() {
            let flat = board.window_flat_idx(q, r);
            if flat < STRIDE {
                let mix = key ^ (u128::from(flat as u32) * 0x9E37_79B9_7F4A_7C15);
                policy[flat] = 1.0 + (mix % 97) as f32;
            }
        }
        let total: f32 = policy.iter().sum();
        policy.iter_mut().for_each(|p| *p /= total);
        (policy, ((key % 2001) as f32 / 1000.0 - 1.0) * 0.5)
    }

    fn search(tree: &mut MCTSTree, descents: usize, batch: usize) {
        let mut spent = 0;
        while spent < descents {
            let leaves = tree
                .select_leaves(batch.min(descents - spent))
                .expect("select_leaves: no desync in this fixture");
            spent += leaves.len() + tree.last_inline_descents() + tree.last_tt_hits();
            let (policies, values): (Vec<_>, Vec<_>) = leaves.iter().map(net).unzip();
            tree.expand_and_backup(&policies, &values);
        }
    }

    /// A mid-game position with the side to move holding both its stones.
    fn two_stone_board() -> Board {
        let mut board = Board::new();
        for (q, r) in [(0, 0), (1, 0), (0, 1), (-1, 2), (2, -1)] {
            board.apply_move(q, r).expect("an empty cell");
        }
        assert_eq!(board.moves_remaining, 2);
        board
    }

    fn tree(kind: SearchKind) -> MCTSTree {
        let mut tree = MCTSTree::new_full(1.5, 1.0, 0.25);
        tree.configure_search(
            kind,
            QSigma {
                c_visit: 50.0,
                c_scale: 1.0,
                rescale: true,
            },
        );
        tree.new_game(two_stone_board());
        tree
    }

    /// Every field of every node under `idx`, depth first in child order, floats as bits.
    fn snapshot(tree: &MCTSTree, idx: u32, out: &mut Vec<[u64; 9]>) {
        let n = tree.pool[idx as usize];
        out.push([
            u64::from(n.action_idx),
            u64::from(n.n_visits),
            u64::from(n.w_value.to_bits()),
            u64::from(n.prior.to_bits()),
            u64::from(n.n_children),
            u64::from(n.moves_remaining),
            u64::from(n.is_terminal),
            u64::from(n.terminal_value.to_bits()),
            u64::from(n.virtual_loss_count),
        ]);
        if n.is_expanded() {
            for c in n.first_child..n.first_child + u32::from(n.n_children) {
                assert_eq!(
                    tree.pool[c as usize].parent, idx,
                    "a child points at its parent"
                );
                snapshot(tree, c, out);
            }
        }
    }

    fn most_visited_child(tree: &MCTSTree) -> u32 {
        let root = tree.pool[0];
        (root.first_child..root.first_child + u32::from(root.n_children))
            .max_by_key(|&i| tree.pool[i as usize].n_visits)
            .expect("an expanded root")
    }

    fn board_after(tree: &MCTSTree, child: u32) -> Board {
        let (q, r) = tree.pool[child as usize].cell();
        tree.root_board_plus(q, r)
            .expect("a root child is an empty cell")
    }

    #[test]
    fn a_reroot_keeps_the_chosen_subtree_bit_for_bit() {
        // PLANTED BREAK: push the copied children without remapping `parent` and the snapshot's parent check reds.
        let mut tree = tree(SearchKind::Puct);
        search(&mut tree, 256, 8);
        let child = most_visited_child(&tree);
        let mut before = Vec::new();
        snapshot(&tree, child, &mut before);
        let visits = tree.pool[child as usize].n_visits;
        assert!(
            visits > 1 && before.len() > 1,
            "the chosen child kept a subtree"
        );

        let inherited = tree.reroot_or_new_game(board_after(&tree, child));
        let mut after = Vec::new();
        snapshot(&tree, 0, &mut after);
        assert_eq!(inherited, visits, "the visits kept are the child's");
        assert_eq!(
            after, before,
            "the re-rooted tree is the child's subtree, bit for bit"
        );
        assert!(
            after.iter().all(|n| n[8] == 0),
            "a finished search left no virtual loss"
        );
        assert_eq!(tree.pool[0].parent, u32::MAX);
        assert_eq!(
            tree.next_free as usize,
            after.len(),
            "the pool holds the subtree alone"
        );
        assert_eq!(
            tree.transposition_table.len(),
            0,
            "the table starts empty, as a fresh search's"
        );
    }

    #[test]
    fn a_reroot_then_search_is_the_subtree_searched_in_place() {
        let mut in_place = tree(SearchKind::Puct);
        search(&mut in_place, 192, 8);
        let child = most_visited_child(&in_place);
        let mut rerooted = tree(SearchKind::Puct);
        search(&mut rerooted, 192, 8);
        assert!(rerooted.reroot_or_new_game(board_after(&in_place, child)) > 1);

        in_place.transposition_table.clear();
        in_place
            .set_forced_root_child(Some(child))
            .expect("a root child");
        search(&mut in_place, 128, 1);
        search(&mut rerooted, 128, 1);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        snapshot(&in_place, child, &mut a);
        snapshot(&rerooted, 0, &mut b);
        assert_eq!(
            b, a,
            "128 more descents leave the same statistics on the same node set"
        );
    }

    #[test]
    fn a_board_no_kept_child_reaches_starts_fresh() {
        let mut tree = tree(SearchKind::Puct);
        search(&mut tree, 64, 8);
        let child = most_visited_child(&tree);
        let next = board_after(&tree, child);

        let mut crossed = next.clone();
        let (q, r) = crossed.legal_moves()[0];
        crossed.apply_move(q, r).expect("a legal cell");
        let same = tree.root_board.clone();
        let mut unvisited = tree.root_board.clone();
        let root = tree.pool[0];
        let cold = (root.first_child..root.first_child + u32::from(root.n_children))
            .find(|&i| !tree.pool[i as usize].is_expanded())
            .expect("64 descents leave a child unexpanded");
        let (q, r) = tree.pool[cold as usize].cell();
        unvisited.apply_move(q, r).expect("an empty cell");

        for (label, board) in [
            ("same position", same),
            ("one stone more", crossed),
            ("cold child", unvisited),
        ] {
            let mut t = tree_copy(&tree);
            assert_eq!(t.reroot_or_new_game(board), 0, "{label}");
            assert!(
                !t.pool[0].is_expanded() && t.next_free == 1,
                "{label}: a fresh root"
            );
        }
        let mut decided = tree_copy(&tree);
        decided.pool[child as usize].is_terminal = true;
        assert_eq!(
            decided.reroot_or_new_game(next.clone()),
            0,
            "a decided child"
        );

        // A turn's second stone: its child is the other side's position, the next turn's.
        let mut second = tree_copy(&tree);
        second.new_game(next);
        search(&mut second, 64, 8);
        let child = most_visited_child(&second);
        assert_eq!(
            second.reroot_or_new_game(board_after(&second, child)),
            0,
            "a stone 2 root"
        );

        let mut gumbel = self::tree(SearchKind::Gumbel);
        search(&mut gumbel, 64, 8);
        let child = most_visited_child(&gumbel);
        assert_eq!(
            gumbel.reroot_or_new_game(board_after(&gumbel, child)),
            0,
            "a Gumbel tree"
        );
    }

    fn tree_copy(tree: &MCTSTree) -> MCTSTree {
        let mut copy = MCTSTree::new_full(1.5, 1.0, 0.25);
        copy.root_board = tree.root_board.clone();
        copy.pool[..tree.next_free as usize].copy_from_slice(&tree.pool[..tree.next_free as usize]);
        copy.next_free = tree.next_free;
        copy
    }
}
