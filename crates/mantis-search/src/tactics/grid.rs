//! The solver's dense board: Six's window layout, per-side window lists with O(1) removal, a 128-bit key.
//!
//! A cell's index is `(q - oq + 128) * 256 + (r - or + 128)`, so index order is `(q, r)` order.

use mantis_core::board::zobrist::ZobristTable;
use mantis_core::board::{Board, Cell};

/// Cells per grid side (Six's width).
pub(crate) const SIZE: i32 = 256;
const CELLS: usize = (SIZE * SIZE) as usize;
/// Index steps of the axes (1, 0), (0, 1), (1, -1): `HEX_AXES` in index space.
pub(crate) const AXIS_STEP: [i32; 3] = [SIZE, 1, SIZE - 1];
/// A stone within this many cells of the grid's edge ends the solve.
const EDGE_MARGIN: i32 = 16;
const WIN: i32 = 6;
const ABSENT: u32 = u32::MAX;

/// A side's list slot: `P1` moves first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Side {
    P1 = 0,
    P2 = 1,
}

impl Side {
    pub(crate) fn other(self) -> Self {
        match self {
            Side::P1 => Side::P2,
            Side::P2 => Side::P1,
        }
    }

    pub(crate) fn of(player: mantis_core::Player) -> Self {
        match player {
            mantis_core::Player::One => Side::P1,
            mantis_core::Player::Two => Side::P2,
        }
    }

    fn zobrist_index(self) -> usize {
        self as usize
    }
}

/// Which list a pure window sits in: four or more of a side's stones, exactly three, exactly two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Threat = 0,
    Three = 1,
    Two = 2,
}

/// A stone the grid cannot hold: within `EDGE_MARGIN` of its edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GridOverflow;

/// Window ids with O(1) removal: `pos[id]` is the id's slot in `ids`, `ABSENT` when it has none.
struct WindowList {
    ids: Vec<u32>,
    pos: Vec<u32>,
}

impl WindowList {
    fn new() -> Self {
        WindowList {
            ids: Vec::new(),
            pos: vec![ABSENT; 3 * CELLS],
        }
    }

    fn add(&mut self, id: u32) {
        // At most 3 * CELLS windows exist, so a slot never reaches ABSENT.
        self.pos[id as usize] = self.ids.len() as u32;
        self.ids.push(id);
    }

    fn remove(&mut self, id: u32) {
        let at = self.pos[id as usize] as usize;
        if let Some(moved) = self.ids.pop() {
            if at < self.ids.len() {
                self.ids[at] = moved;
                self.pos[moved as usize] = at as u32;
            }
        }
        self.pos[id as usize] = ABSENT;
    }
}

pub(crate) struct Grid {
    origin: (i32, i32),
    /// 0 empty, 1 P1, 2 P2.
    cells: Vec<u8>,
    /// Per axis and start index: P1's count in the low nibble, P2's in the high.
    windows: [Vec<u8>; 3],
    lists: [[WindowList; 3]; 2],
    placed: Vec<u32>,
    key: u128,
}

impl Grid {
    pub(crate) fn new() -> Self {
        Grid {
            origin: (0, 0),
            cells: vec![0; CELLS],
            windows: [vec![0; CELLS], vec![0; CELLS], vec![0; CELLS]],
            lists: [
                [WindowList::new(), WindowList::new(), WindowList::new()],
                [WindowList::new(), WindowList::new(), WindowList::new()],
            ],
            placed: Vec::new(),
            key: 0,
        }
    }

    /// Place `board`'s stones around their bounding box's centre; the grid must be empty.
    pub(crate) fn load(&mut self, board: &Board) -> Result<(), GridOverflow> {
        debug_assert!(
            self.placed.is_empty(),
            "load on a grid still holding stones"
        );
        let mut stones: Vec<((i32, i32), Side)> = board
            .cells_iter()
            .filter_map(|(&cell, &c)| match c {
                Cell::P1 => Some((cell, Side::P1)),
                Cell::P2 => Some((cell, Side::P2)),
                Cell::Empty => None,
            })
            .collect();
        stones.sort_unstable_by_key(|&(cell, _)| cell);
        let (mut lq, mut hq, mut lr, mut hr) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
        for &((q, r), _) in &stones {
            (lq, hq, lr, hr) = (lq.min(q), hq.max(q), lr.min(r), hr.max(r));
        }
        // Six's truncating centre: `midpoint` rounds a negative odd sum the other way.
        self.origin = if stones.is_empty() {
            (0, 0)
        } else {
            ((lq + hq) / 2, (lr + hr) / 2)
        };
        for ((q, r), side) in stones {
            match self.index_of(q, r) {
                Some(idx) => self.place(idx, side),
                None => {
                    self.unload();
                    return Err(GridOverflow);
                }
            }
        }
        Ok(())
    }

    /// Remove every stone, newest first: the grid is empty again with no memset.
    pub(crate) fn unload(&mut self) {
        while !self.placed.is_empty() {
            self.undo();
        }
    }

    /// The index of `(q, r)`, or `None` within `EDGE_MARGIN` of the edge.
    pub(crate) fn index_of(&self, q: i32, r: i32) -> Option<u32> {
        let lq = q - self.origin.0 + SIZE / 2;
        let lr = r - self.origin.1 + SIZE / 2;
        let inside = |x: i32| (EDGE_MARGIN..SIZE - EDGE_MARGIN).contains(&x);
        (inside(lq) && inside(lr)).then(|| (lq * SIZE + lr) as u32)
    }

    /// The axial cell of an index.
    pub(crate) fn cell_at(&self, idx: u32) -> (i32, i32) {
        let idx = idx as i32;
        (
            idx / SIZE - SIZE / 2 + self.origin.0,
            idx % SIZE - SIZE / 2 + self.origin.1,
        )
    }

    /// Whether a placement at `idx` stays inside the margin.
    pub(crate) fn placeable(idx: u32) -> bool {
        let (lq, lr) = (idx as i32 / SIZE, idx as i32 % SIZE);
        let inside = |x: i32| (EDGE_MARGIN..SIZE - EDGE_MARGIN).contains(&x);
        inside(lq) && inside(lr)
    }

    /// 0 empty, 1 P1, 2 P2.
    pub(crate) fn at(&self, idx: u32) -> u8 {
        self.cells[idx as usize]
    }

    pub(crate) fn key(&self) -> u128 {
        self.key
    }

    pub(crate) fn list(&self, side: Side, kind: Kind) -> &[u32] {
        &self.lists[side as usize][kind as usize].ids
    }

    /// The six cell indices of window `id` (`axis * CELLS + start`).
    pub(crate) fn window_cells(id: u32) -> [u32; 6] {
        let axis = id as usize / CELLS;
        let start = (id as usize % CELLS) as i32;
        let step = AXIS_STEP[axis];
        std::array::from_fn(|i| (start + i as i32 * step) as u32)
    }

    /// Place a stone of `side` at `idx`, which the caller has checked `placeable` and empty.
    pub(crate) fn place(&mut self, idx: u32, side: Side) {
        debug_assert!(Self::placeable(idx) && self.cells[idx as usize] == 0);
        self.cells[idx as usize] = side as u8 + 1;
        self.placed.push(idx);
        let (q, r) = self.cell_at(idx);
        self.key ^= ZobristTable::get_for_pos(q, r, side.zobrist_index());
        self.shift_windows(idx, side, true);
    }

    /// Take back the newest stone.
    pub(crate) fn undo(&mut self) {
        let Some(idx) = self.placed.pop() else {
            return;
        };
        let side = if self.cells[idx as usize] == 1 {
            Side::P1
        } else {
            Side::P2
        };
        self.cells[idx as usize] = 0;
        let (q, r) = self.cell_at(idx);
        self.key ^= ZobristTable::get_for_pos(q, r, side.zobrist_index());
        self.shift_windows(idx, side, false);
    }

    fn shift_windows(&mut self, idx: u32, side: Side, add: bool) {
        let unit: u8 = if side == Side::P1 { 1 } else { 0x10 };
        for (axis, &step) in AXIS_STEP.iter().enumerate() {
            for k in 0..WIN {
                let start = (idx as i32 - k * step) as usize;
                let before = self.windows[axis][start];
                let after = if add { before + unit } else { before - unit };
                self.windows[axis][start] = after;
                self.relist((axis * CELLS + start) as u32, before, after);
            }
        }
    }

    fn relist(&mut self, id: u32, before: u8, after: u8) {
        for side in [Side::P1, Side::P2] {
            let was = kind_of(before, side);
            let is = kind_of(after, side);
            if was == is {
                continue;
            }
            if let Some(kind) = was {
                self.lists[side as usize][kind as usize].remove(id);
            }
            if let Some(kind) = is {
                self.lists[side as usize][kind as usize].add(id);
            }
        }
    }
}

/// The list a packed window belongs to for `side`: pure (none of the other side's stones) and 2, 3 or 4+ own.
fn kind_of(packed: u8, side: Side) -> Option<Kind> {
    let (own, other) = match side {
        Side::P1 => (packed & 0xF, packed >> 4),
        Side::P2 => (packed >> 4, packed & 0xF),
    };
    match (other, own) {
        (0, 2) => Some(Kind::Two),
        (0, 3) => Some(Kind::Three),
        (0, n) if n >= 4 => Some(Kind::Threat),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mantis_core::board::zobrist::splitmix64_next;
    use mantis_core::board::HEX_AXES;
    use std::collections::BTreeSet;

    /// Clustered random play, each stone within distance 2 of one of the last six.
    fn position(seed: u64, stones: usize) -> Board {
        let mut state = seed;
        let mut b = Board::new();
        b.set_legal_move_radius(8);
        let mut placed: Vec<(i32, i32)> = Vec::new();
        while placed.len() < stones {
            let (q, r) = if placed.is_empty() {
                (3, -7)
            } else {
                let tail = &placed[placed.len().saturating_sub(6)..];
                let (aq, ar) = tail[(splitmix64_next(&mut state) as usize) % tail.len()];
                let dq = (splitmix64_next(&mut state) % 5) as i32 - 2;
                let dr = (splitmix64_next(&mut state) % 5) as i32 - 2;
                (aq + dq, ar + dr)
            };
            if b.get(q, r) == Cell::Empty && b.apply_move(q, r).is_ok() {
                placed.push((q, r));
            }
        }
        b
    }

    /// Every pure window of `side` by a brute-force scan of the board, as (axis, start cell, kind).
    fn scanned(board: &Board, side: Side) -> BTreeSet<(usize, (i32, i32), u8)> {
        let own = if side == Side::P1 { Cell::P1 } else { Cell::P2 };
        let mut out = BTreeSet::new();
        for (&(sq, sr), _) in board.cells_iter() {
            for (axis, &(dq, dr)) in HEX_AXES.iter().enumerate() {
                for k in 0..6 {
                    let start = (sq - k * dq, sr - k * dr);
                    let (mut mine, mut theirs) = (0u8, 0u8);
                    for i in 0..6 {
                        match board.get(start.0 + i * dq, start.1 + i * dr) {
                            Cell::Empty => {}
                            c if c == own => mine += 1,
                            _ => theirs += 1,
                        }
                    }
                    let kind = match (theirs, mine) {
                        (0, 2) => Some(Kind::Two as u8),
                        (0, 3) => Some(Kind::Three as u8),
                        (0, n) if n >= 4 => Some(Kind::Threat as u8),
                        _ => None,
                    };
                    if let Some(kind) = kind {
                        out.insert((axis, start, kind));
                    }
                }
            }
        }
        out
    }

    fn listed(grid: &Grid, side: Side) -> BTreeSet<(usize, (i32, i32), u8)> {
        let mut out = BTreeSet::new();
        for kind in [Kind::Threat, Kind::Three, Kind::Two] {
            for &id in grid.list(side, kind) {
                let axis = id as usize / CELLS;
                let start = grid.cell_at((id as usize % CELLS) as u32);
                assert!(
                    out.insert((axis, start, kind as u8)),
                    "window {id} listed twice"
                );
            }
        }
        out
    }

    #[test]
    fn the_lists_are_every_pure_window_of_a_brute_force_scan() {
        let mut grid = Grid::new();
        for seed in 0..40u64 {
            let board = position(0x6_71d0 + seed, 10 + (seed as usize % 50));
            grid.load(&board).expect("a compact position fits the grid");
            for side in [Side::P1, Side::P2] {
                assert_eq!(
                    listed(&grid, side),
                    scanned(&board, side),
                    "seed {seed} side {side:?}"
                );
            }
            assert_eq!(
                grid.key(),
                board.zobrist_hash,
                "the key is the board's own Zobrist hash"
            );
            grid.unload();
        }
    }

    #[test]
    fn unload_leaves_an_empty_grid_and_a_place_undo_pair_restores_every_list() {
        let mut grid = Grid::new();
        let board = position(0xfeed, 40);
        grid.load(&board).expect("fits");
        let before: Vec<Vec<u32>> = [Side::P1, Side::P2]
            .iter()
            .flat_map(|&s| [Kind::Threat, Kind::Three, Kind::Two].map(|k| grid.list(s, k).to_vec()))
            .collect();
        let key = grid.key();
        let empty = (0..CELLS as u32).find(|&i| {
            Grid::placeable(i) && grid.at(i) == 0 && {
                let (q, r) = grid.cell_at(i);
                board
                    .cells_iter()
                    .any(|(&(sq, sr), _)| mantis_core::board::hex_distance(q, r, sq, sr) == 1)
            }
        });
        let idx = empty.expect("an empty neighbour of a stone");
        grid.place(idx, Side::P2);
        grid.undo();
        let after: Vec<Vec<u32>> = [Side::P1, Side::P2]
            .iter()
            .flat_map(|&s| [Kind::Threat, Kind::Three, Kind::Two].map(|k| grid.list(s, k).to_vec()))
            .collect();
        let as_sets = |v: &Vec<Vec<u32>>| {
            v.iter()
                .map(|l| l.iter().copied().collect::<BTreeSet<u32>>())
                .collect::<Vec<_>>()
        };
        assert_eq!(as_sets(&after), as_sets(&before));
        assert_eq!(grid.key(), key);
        grid.unload();
        assert!(grid.cells.iter().all(|&c| c == 0));
        assert!(grid.windows.iter().all(|w| w.iter().all(|&p| p == 0)));
        assert!(grid
            .lists
            .iter()
            .flatten()
            .all(|l| l.ids.is_empty() && l.pos.iter().all(|&p| p == ABSENT)));
        assert_eq!(grid.key(), 0);
    }

    #[test]
    fn index_order_is_axial_order_and_the_edge_is_refused() {
        let mut grid = Grid::new();
        grid.load(&position(1, 12)).expect("fits");
        let a = grid.index_of(2, 9).expect("inside");
        let b = grid.index_of(3, -9).expect("inside");
        assert!(a < b, "(2, 9) precedes (3, -9) in (q, r) order");
        assert_eq!(grid.cell_at(a), (2, 9));
        let (oq, or) = grid.origin;
        assert!(grid.index_of(oq + SIZE / 2 - EDGE_MARGIN, or).is_none());
        assert!(grid.index_of(oq, or - SIZE / 2 + EDGE_MARGIN - 1).is_none());
        assert!(grid.index_of(oq + SIZE / 2 - EDGE_MARGIN - 1, or).is_some());
    }
}
