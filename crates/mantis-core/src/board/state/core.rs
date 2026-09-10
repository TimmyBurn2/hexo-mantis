// Exceeds the 300-line soft cap: the Board state core (types, ctors, mutators,
// window helpers, Clone) ports as one line-auditable unit with its in-file tests.
use std::cell::{Cell as StdCell, UnsafeCell};
use fxhash::{FxHashMap, FxHashSet};
use super::super::zobrist::ZobristTable;
use crate::ply::Ply;

/// Captures everything mutated by one `apply_move_tracked` call so `undo_move` can reverse it
/// in O(1) with no HashMap scan. All fields are private: the only constructor is
/// `apply_move_tracked` and the only consumer is `undo_move`.
#[derive(Debug, Clone)]
pub struct MoveDiff {
    pub(crate) q: i32,
    pub(crate) r: i32,
    pub(crate) player: Player,
    // Previous full Zobrist hash state.
    prev_zobrist_hash: u128,
    // Turn-structure state before the move.
    prev_moves_remaining: u8,
    prev_current_player: Player,
    prev_ply: Ply,
    // Win-detection state before the move.
    prev_last_move: Option<(i32, i32)>,
    // Bounding-box state before the move (needed for O(1) bbox undo).
    prev_min_q: i32,
    prev_max_q: i32,
    prev_min_r: i32,
    prev_max_r: i32,
    prev_has_stones: bool,
    // Action anchors state before the move.
    prev_action_anchors: [(i32, i32); 4],
    prev_action_anchors_count: usize,
}

/// Board size (cells per axis of the view window).
pub const BOARD_SIZE: usize = 19;
/// Half-width: window covers [-HALF, HALF] relative to its centre.
pub const HALF: i32 = (BOARD_SIZE as i32 - 1) / 2; // 9
/// Total cells in the 19×19 view window.
pub const TOTAL_CELLS: usize = BOARD_SIZE * BOARD_SIZE; // 361

/// The three hex axis directions (positive direction only; win scan uses ±).
pub fn hex_distance(q1: i32, r1: i32, q2: i32, r2: i32) -> i32 {
    ((q1 - q2).abs() + (q1 + r1 - q2 - r2).abs() + (r1 - r2).abs()) / 2
}

pub const HEX_AXES: [(i32, i32); 3] = [
    (1, 0),  // E / W
    (0, 1),  // NE / SW
    (1, -1), // SE / NW
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Player {
    One = 1,
    Two = -1,
}

impl Player {
    pub fn other(self) -> Self {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i8)]
pub enum Cell {
    #[default]
    Empty = 0,
    P1 = 1,
    P2 = -1,
}

/// Plain geometry values for a Board. Spec/registry resolution is NOT this crate's job —
/// callers resolve names to values and pass the values in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoardGeometry {
    /// Hex-ball radius for legal-move expansion.
    pub legal_move_radius: i32,
    /// Max `hex_distance` at which two stones share a cluster.
    pub cluster_threshold: i32,
    /// Cluster-view side length; odd, >= 7.
    pub cluster_window_size: usize,
}

/// Sparse game board — all state needed to continue a game from any position.
///
/// `Send + !Sync`: the legal-move cache uses `UnsafeCell` with deliberately NO
/// `unsafe impl Sync`, so a `&Board` cannot cross threads at the type level.
#[derive(Debug)]
pub struct Board {
    /// Sparse stone map: (q, r) → Cell.
    pub(crate) cells: FxHashMap<(i32, i32), Cell>,
    /// Whose turn it is.
    pub current_player: Player,
    /// How many moves the current player still has to place this turn: 1 on ply 0, then 2.
    /// A within-turn count — deliberately a bare `u8`, not a Ply/Turn index.
    pub moves_remaining: u8,
    /// Total half-moves placed so far.
    pub ply: Ply,
    /// Incremental Zobrist hash.
    pub zobrist_hash: u128,
    /// The move most recently applied (used for fast win detection).
    pub(crate) last_move: Option<(i32, i32)>,
    /// Bounding box of all placed stones (maintained incrementally).
    pub(crate) min_q: i32,
    pub(crate) max_q: i32,
    pub(crate) min_r: i32,
    pub(crate) max_r: i32,
    /// True once at least one stone has been placed.
    pub(crate) has_stones: bool,
    /// Last 4 stones placed (q, r).
    pub(crate) action_anchors: [(i32, i32); 4],
    pub(crate) action_anchors_count: usize,
    /// Lazily-maintained set of all currently legal moves, behind interior mutability so
    /// `legal_moves_set(&self)` can rebuild without `&mut self`. When `cache_dirty` is false
    /// the cache is correct; when true, `legal_moves_set()` rebuilds it.
    ///
    /// # Safety invariants (INV-1..INV-4)
    ///
    /// The crate's three `unsafe` expressions (two in `board::moves::legal_moves_set`, one in
    /// `Clone::clone` below) rest on:
    ///
    /// - **INV-1.** The only `&mut` into `legal_cache` is created inside `legal_moves_set`'s
    ///   rebuild block, entered iff `cache_dirty == true`, and dropped before the shared
    ///   return borrow is created.
    /// - **INV-2.** `cache_dirty` goes false->true only in `&mut self` methods, or on boards
    ///   not yet shared. Enforced by privacy: the field is private to `state::core` and the
    ///   sole crate-visible true-setter is `mark_cache_dirty(&mut self)`.
    /// - **INV-3.** Every `&FxHashSet` returned by `legal_moves_set` borrows `*self`, so the
    ///   borrow checker rejects any `&mut self` call — hence any rebuild — while it is live.
    /// - **INV-4.** The rebuild block calls NO Board method and reads nothing through the
    ///   cache. This is what excludes re-entrancy: `cache_dirty` is also true DURING a
    ///   rebuild, so a call issued from inside the block could create a second borrow
    ///   aliasing the live exclusive one with no other invariant violated. Adding ANY call
    ///   inside the rebuild block is a review failure.
    ///
    /// The soundness boundary spans `state::core` (owns the flag and its privacy — a NEW
    /// `&self` write of `cache_dirty` here is a review failure) and `board::moves` (holds the
    /// unsafe blocks). `pub(crate)` is harmless: `UnsafeCell::get` yields a raw pointer.
    pub(crate) legal_cache: UnsafeCell<FxHashSet<(i32, i32)>>,
    /// Set true by any mutating operation, cleared by `legal_moves_set()` after a rebuild.
    /// PRIVATE to `state::core`: the false->true transition is gated behind `&mut self`.
    cache_dirty: StdCell<bool>,
    /// Per-board legal-move radius override; `legal_moves_set()` rebuilds by hex-ball
    /// expansion at this radius. Default `moves::DEFAULT_LEGAL_MOVE_RADIUS` (5).
    pub(crate) legal_move_radius: i32,
    /// Per-board cluster connectivity threshold: two stones share a cluster iff their
    /// `hex_distance` is <= this. Default 5 (v6 wire format); wide-window corpora use 8.
    pub(crate) cluster_threshold: i32,
    /// Per-board cluster window side length, used by `get_cluster_views()` for its 2-plane
    /// snapshots. Default `BOARD_SIZE` (19, v6 wire format).
    pub(crate) cluster_window_size: usize,
}

impl Board {
    /// Create an empty board ready for the first move. The baked constants (radius 5,
    /// threshold 5, window 19) are game-rules constants, not config defaults.
    pub fn new() -> Self {
        // Pre-populated with the 5x5 region at (0,0), restricting the first move to 25 cells
        // so branching stays ~24 for the whole game. The rules make every cell legal on an
        // empty board, but hundreds of root children cost evaluation for no strategic gain.
        let mut init_cache = FxHashSet::default();
        init_cache.reserve(50);
        for dq in -2i32..=2 {
            for dr in -2i32..=2 {
                init_cache.insert((dq, dr));
            }
        }

        Board {
            cells: FxHashMap::default(),
            current_player: Player::One,
            moves_remaining: 1,
            ply: Ply::ZERO,
            zobrist_hash: 0,
            last_move: None,
            min_q: 0,
            max_q: 0,
            min_r: 0,
            max_r: 0,
            has_stones: false,
            action_anchors: [(0, 0); 4],
            action_anchors_count: 0,
            legal_cache: UnsafeCell::new(init_cache),
            cache_dirty: StdCell::new(false),
            legal_move_radius: super::super::moves::DEFAULT_LEGAL_MOVE_RADIUS,
            cluster_threshold: super::super::moves::DEFAULT_CLUSTER_THRESHOLD,
            cluster_window_size: BOARD_SIZE,
        }
    }

    /// Construct a Board from plain geometry values — the sole non-default ctor. Callers
    /// resolve names to values BEFORE calling; this crate never sees names.
    pub fn with_geometry(g: BoardGeometry) -> Board {
        debug_assert!(
            g.cluster_window_size >= 7 && g.cluster_window_size % 2 == 1,
            "cluster_window_size must be odd and >= 7; got {}",
            g.cluster_window_size
        );
        let mut b = Board::new();
        b.cluster_window_size = g.cluster_window_size;
        b.cluster_threshold = g.cluster_threshold;
        b.legal_move_radius = g.legal_move_radius;
        b.cache_dirty.set(true);
        b
    }

    /// The Board's current geometry values (the introspection surface).
    pub fn geometry(&self) -> BoardGeometry {
        BoardGeometry {
            legal_move_radius: self.legal_move_radius,
            cluster_threshold: self.cluster_threshold,
            cluster_window_size: self.cluster_window_size,
        }
    }

    /// Override the cluster connectivity threshold. Affects only `get_clusters()` /
    /// `get_cluster_views()`; legal-move expansion is unchanged.
    pub fn set_cluster_threshold(&mut self, threshold: i32) {
        self.cluster_threshold = threshold;
    }

    /// Current cluster threshold (default 5 = v6 wire-format).
    pub fn cluster_threshold(&self) -> i32 {
        self.cluster_threshold
    }

    /// Override the cluster window side length, used by `get_cluster_views()` to size the
    /// 2-plane snapshot. Caller must use an odd value >= 7; enforced by debug_assert.
    pub fn set_cluster_window_size(&mut self, size: usize) {
        debug_assert!(
            size >= 7 && size % 2 == 1,
            "cluster_window_size must be odd and >= 7; got {size}"
        );
        self.cluster_window_size = size;
    }

    /// Current cluster window side length (default 19 = v6 wire-format).
    pub fn cluster_window_size(&self) -> usize {
        self.cluster_window_size
    }

    /// Override the legal-move radius, marking `legal_cache` dirty. No Rust-level guard: any
    /// boundary-layer guard lives at the boundary.
    pub fn set_legal_move_radius(&mut self, radius: i32) {
        self.legal_move_radius = radius;
        self.cache_dirty.set(true);
    }

    /// Current legal-move radius (default 5).
    pub fn legal_move_radius(&self) -> i32 {
        self.legal_move_radius
    }

    // The ONLY crate-visible surface over `cache_dirty`; the `legal_cache` field doc
    // (INV-1..INV-4) carries the invariants this protocol enforces.

    /// Whether the legal-move cache needs a rebuild.
    pub(crate) fn cache_is_dirty(&self) -> bool {
        self.cache_dirty.get()
    }

    /// Mark the cache clean. False-only setter, safe through `&self`: a false transition can
    /// never arm a rebuild.
    pub(crate) fn clear_cache_dirty(&self) {
        self.cache_dirty.set(false);
    }

    /// Invalidate the legal-move cache. The sole crate-visible true-setter; requires
    /// `&mut self` (INV-2), so it cannot run while a `&FxHashSet` is live (INV-3). `pub` and
    /// not `pub(crate)` because the `compile_fail,E0502` doctest compiles as an external
    /// crate and must reach it, so the pin fails for E0502 and not a private-method error.
    pub fn mark_cache_dirty(&mut self) {
        self.cache_dirty.set(true);
    }

    /// Centre of the trunk-sized view window: bbox centroid, `(0, 0)` when empty. Truncating
    /// `(a+b)/2` is deliberate — `i32::midpoint` floors toward -inf and would shift the NN
    /// window by <=1 cell on negative-odd bbox sums, breaking legacy anchor calibration.
    #[allow(clippy::manual_midpoint)]
    pub fn window_center(&self) -> (i32, i32) {
        if !self.has_stones {
            return (0, 0);
        }
        let cq = (self.min_q + self.max_q) / 2;
        let cr = (self.min_r + self.max_r) / 2;
        (cq, cr)
    }

    /// Window-relative flat index for (q, r), `usize::MAX` when out of window. Dispatches via
    /// `self.cluster_window_size`, the NN-input frame geometry, not the canvas size.
    #[inline]
    pub fn window_flat_idx(&self, q: i32, r: i32) -> usize {
        let (cq, cr) = self.window_center();
        let trunk_sz = self.cluster_window_size as i32;
        let half = (trunk_sz - 1) / 2;
        Self::window_flat_idx_at_geom(q, r, cq, cr, trunk_sz, half)
    }

    /// Flat index at a specific centre, legacy default geometry (19/9). A non-19 trunk must
    /// go through `window_flat_idx_at_geom` with `(trunk_sz, half)` threaded from the boundary.
    #[inline]
    pub fn window_flat_idx_at(q: i32, r: i32, cq: i32, cr: i32) -> usize {
        Self::window_flat_idx_at_geom(q, r, cq, cr, BOARD_SIZE as i32, HALF)
    }

    /// Flat-index kernel with caller-threaded geometry: hot-loop callers pre-extract
    /// `(trunk_sz, half)` once, and `#[inline]` folds the bounds check into the caller.
    #[inline]
    pub fn window_flat_idx_at_geom(
        q: i32, r: i32, cq: i32, cr: i32, trunk_sz: i32, half: i32,
    ) -> usize {
        let wq = q - cq + half;
        let wr = r - cr + half;
        if wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz {
            (wq as usize * trunk_sz as usize) + wr as usize
        } else {
            usize::MAX
        }
    }

    /// Returns the cell at (q, r).
    pub fn get_cell(&self, q: i32, r: i32) -> Cell {
        self.cells.get(&(q, r)).copied().unwrap_or(Cell::Empty)
    }

    /// Axial coordinates (q, r) from a window-relative flat index; dispatches via
    /// `self.cluster_window_size` so non-default windows decode correctly.
    #[inline]
    pub fn window_coords(&self, flat: usize) -> (i32, i32) {
        let (cq, cr) = self.window_center();
        let trunk_sz = self.cluster_window_size;
        let half = ((trunk_sz as i32) - 1) / 2;
        let wq = (flat / trunk_sz) as i32;
        let wr = (flat % trunk_sz) as i32;
        (wq - half + cq, wr - half + cr)
    }

    /// Whether (q, r) is inside the current trunk-sized view window.
    #[inline]
    pub fn in_window(&self, q: i32, r: i32) -> bool {
        let (cq, cr) = self.window_center();
        let trunk_sz = self.cluster_window_size as i32;
        let half = (trunk_sz - 1) / 2;
        let wq = q - cq + half;
        let wr = r - cr + half;
        wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz
    }

    /// Iterator over all occupied cells: yields `(&(q, r), &Cell)` pairs.
    pub fn cells_iter(&self) -> impl Iterator<Item = (&(i32, i32), &Cell)> {
        self.cells.iter()
    }

    /// Cell at (q, r).  Returns Empty for unoccupied or out-of-window cells.
    #[inline]
    pub fn get(&self, q: i32, r: i32) -> Cell {
        self.cells.get(&(q, r)).copied().unwrap_or(Cell::Empty)
    }

    /// Apply a move at (q, r) for the current player.
    ///
    /// `Err` only if the cell is occupied: the board is conceptually infinite and this does no
    /// window or radius check, so those constraints are the caller's. `moves_remaining`
    /// decrements, and at 0 the turn passes — player flips, count resets to 2.
    pub fn apply_move(&mut self, q: i32, r: i32) -> Result<(), &'static str> {
        if self.cells.contains_key(&(q, r)) {
            return Err("cell already occupied");
        }

        // Bounding box FIRST so `window_flat_idx` sees the final bbox, which keeps the
        // Zobrist hash position-deterministic (same stone set -> same centre -> same hash).
        if self.has_stones {
            if q < self.min_q { self.min_q = q; }
            if q > self.max_q { self.max_q = q; }
            if r < self.min_r { self.min_r = r; }
            if r > self.max_r { self.max_r = r; }
        } else {
            self.min_q = q;
            self.max_q = q;
            self.min_r = r;
            self.max_r = r;
            self.has_stones = true;
        }

        let player_idx = match self.current_player { Player::One => 0, Player::Two => 1 };

        let cell = match self.current_player {
            Player::One => Cell::P1,
            Player::Two => Cell::P2,
        };
        self.cells.insert((q, r), cell);

        // Lazy invalidation: the search hot path calls apply_move ~2D times per simulation
        // but needs `legal_moves_set()` once, at leaf expansion.
        self.cache_dirty.set(true);

        // Update action anchors (last 4 stones).
        if self.action_anchors_count < 4 {
            self.action_anchors[self.action_anchors_count] = (q, r);
            self.action_anchors_count += 1;
        } else {
            self.action_anchors[0] = self.action_anchors[1];
            self.action_anchors[1] = self.action_anchors[2];
            self.action_anchors[2] = self.action_anchors[3];
            self.action_anchors[3] = (q, r);
        }

        // Use absolute (q, r) for Zobrist — position-independent, no window dependency.
        self.zobrist_hash ^= ZobristTable::get_for_pos(q, r, player_idx);
        self.ply = self.ply.next();
        self.last_move = Some((q, r));

        // Advance turn structure
        self.moves_remaining -= 1;
        if self.moves_remaining == 0 {
            self.current_player = self.current_player.other();
            self.moves_remaining = 2;
        }

        Ok(())
    }

    /// Apply a move and return a reversible state diff for O(1) undo.
    pub fn apply_move_tracked(&mut self, q: i32, r: i32) -> Result<MoveDiff, &'static str> {
        let diff = MoveDiff {
            q,
            r,
            player: self.current_player,
            prev_zobrist_hash: self.zobrist_hash,
            prev_moves_remaining: self.moves_remaining,
            prev_current_player: self.current_player,
            prev_ply: self.ply,
            prev_last_move: self.last_move,
            prev_min_q: self.min_q,
            prev_max_q: self.max_q,
            prev_min_r: self.min_r,
            prev_max_r: self.max_r,
            prev_has_stones: self.has_stones,
            prev_action_anchors: self.action_anchors,
            prev_action_anchors_count: self.action_anchors_count,
        };

        self.apply_move(q, r)?;
        Ok(diff)
    }

    /// Undo a move previously applied by `apply_move_tracked`.
    pub fn undo_move(&mut self, diff: MoveDiff) {
        if let Some(cell) = self.cells.remove(&(diff.q, diff.r)) {
            debug_assert_eq!(
                cell,
                match diff.player {
                    Player::One => Cell::P1,
                    Player::Two => Cell::P2,
                },
                "undo_move removed a stone with mismatched player",
            );
        } else {
            debug_assert!(false, "undo_move expected placed stone to exist");
        }

        // Lazy invalidation for the same reason as apply_move: undo runs ~D times per sim
        // during selection traversal, and `legal_moves_set()` is not called until expansion.
        self.cache_dirty.set(true);

        self.zobrist_hash = diff.prev_zobrist_hash;
        self.moves_remaining = diff.prev_moves_remaining;
        self.current_player = diff.prev_current_player;
        self.ply = diff.prev_ply;
        self.last_move = diff.prev_last_move;

        self.min_q = diff.prev_min_q;
        self.max_q = diff.prev_max_q;
        self.min_r = diff.prev_min_r;
        self.max_r = diff.prev_max_r;
        self.has_stones = diff.prev_has_stones;

        self.action_anchors = diff.prev_action_anchors;
        self.action_anchors_count = diff.prev_action_anchors_count;
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Board {
    fn clone(&self) -> Self {
        // legal_cache contents are NOT copied: rebuilding N entries dominates clone cost on
        // the search hot path, and `cache_dirty = true` makes the clone rebuild from `cells`.
        // SAFETY: shared read of the cache's len through the UnsafeCell. Clone takes `&self`,
        // so per INV-1 no exclusive borrow can be live — the only `&mut` exists inside
        // `legal_moves_set`'s rebuild block, which per INV-4 calls no Board method.
        let cap = unsafe { (*self.legal_cache.get()).len() };
        Board {
            cells: self.cells.clone(),
            current_player: self.current_player,
            moves_remaining: self.moves_remaining,
            ply: self.ply,
            zobrist_hash: self.zobrist_hash,
            last_move: self.last_move,
            min_q: self.min_q,
            max_q: self.max_q,
            min_r: self.min_r,
            max_r: self.max_r,
            has_stones: self.has_stones,
            action_anchors: self.action_anchors,
            action_anchors_count: self.action_anchors_count,
            legal_cache: UnsafeCell::new(FxHashSet::with_capacity_and_hasher(cap, Default::default())),
            cache_dirty: StdCell::new(true),
            legal_move_radius: self.legal_move_radius,
            cluster_threshold: self.cluster_threshold,
            cluster_window_size: self.cluster_window_size,
        }
    }
}

// Deliberately NO `unsafe impl Sync for Board`: the `UnsafeCell` cache makes Board auto-!Sync
// while it stays auto-Send. The crate's three `unsafe` expressions each rest on INV-1..INV-4;
// a fourth touching the cache is a review failure.

// Test-fixture builder, feature `test-fixtures`, OFF by default, so production behaviour is
// byte-untouched. Downstream test/bench targets use it for positions the public `apply_move`
// cadence cannot reach.
#[cfg(feature = "test-fixtures")]
impl Board {
    /// Test-only static-position builder: plants `stones`, recomputes the bbox, marks the
    /// cache dirty and sets the turn-structure fields explicitly. `last_move` is `Some(..)`
    /// only for a terminal-win fixture, since `check_win` reads `last_move` alone.
    pub fn from_stones(
        stones: &[((i32, i32), Cell)],
        to_move: Player,
        moves_remaining: u8,
        ply: u32,
        last_move: Option<(i32, i32)>,
    ) -> Board {
        let mut b = Board::new();
        let (mut lq, mut hq, mut lr, mut hr) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
        for &((q, r), c) in stones {
            b.cells.insert((q, r), c);
            lq = lq.min(q);
            hq = hq.max(q);
            lr = lr.min(r);
            hr = hr.max(r);
        }
        if !stones.is_empty() {
            b.has_stones = true;
            b.min_q = lq;
            b.max_q = hq;
            b.min_r = lr;
            b.max_r = hr;
        }
        b.mark_cache_dirty();
        b.current_player = to_move;
        b.moves_remaining = moves_remaining;
        b.ply = Ply::new(ply);
        b.last_move = last_move;
        b
    }
}

#[cfg(all(test, feature = "test-fixtures"))]
mod from_stones_tests {
    use super::*;
    use crate::board::WIN_LENGTH;

    #[test]
    fn from_stones_sets_expected_state() {
        // A P1 3-in-a-row along the E axis, off-origin so the bbox is non-trivial.
        let stones = [
            ((2, 1), Cell::P1),
            ((3, 1), Cell::P1),
            ((4, 1), Cell::P1),
        ];
        let b = Board::from_stones(&stones, Player::Two, 2, 7, Some((4, 1)));

        // cells present.
        assert_eq!(b.get(2, 1), Cell::P1);
        assert_eq!(b.get(3, 1), Cell::P1);
        assert_eq!(b.get(4, 1), Cell::P1);
        assert_eq!(b.cells.len(), 3);

        // bbox == stone min/max; has_stones set.
        assert!(b.has_stones);
        assert_eq!((b.min_q, b.max_q, b.min_r, b.max_r), (2, 4, 1, 1));

        // turn-structure fields set as passed.
        assert_eq!(b.current_player, Player::Two);
        assert_eq!(b.moves_remaining, 2);
        assert_eq!(b.ply, Ply::new(7));
        assert_eq!(b.last_move, Some((4, 1)));

        // mark_cache_dirty => legal_moves_set rebuilds against the planted stones.
        let legal = b.legal_moves_set();
        assert!(!legal.is_empty(), "legal set must rebuild from planted stones");
        assert!(!legal.contains(&(2, 1)), "occupied cell is not legal");
        assert!(legal.contains(&(5, 1)), "empty neighbour must be legal");
    }

    #[test]
    fn from_stones_terminal_win_reads_last_move() {
        // check_win() reads last_move only, so a 6-in-a-row with it on the line wins.
        let six: Vec<((i32, i32), Cell)> =
            (0..WIN_LENGTH as i32).map(|q| ((q, 0), Cell::P1)).collect();
        let win = Board::from_stones(&six, Player::One, 1, 11, Some((5, 0)));
        assert!(win.check_win(), "6-in-a-row with last_move on the line is a win");

        let no_last = Board::from_stones(&six, Player::One, 1, 11, None);
        assert!(!no_last.check_win(), "check_win reads last_move; None => not a win");
    }

    #[test]
    fn from_stones_empty_leaves_default_bbox() {
        let b = Board::from_stones(&[], Player::One, 1, 0, None);
        assert!(!b.has_stones);
        assert_eq!(b.cells.len(), 0);
    }
}

#[cfg(test)]
mod geometry_tests {
    //! Re-anchored geometry-ctor pins: 2 of the predecessor's 9 spec-ctor tests survive the
    //! registry decoupling.
    use super::*;

    /// Asymmetric values (radius 4, threshold 7, window 9) so a `with_geometry` transcription
    /// bug swapping the two same-typed i32 fields cannot pass.
    #[test]
    fn with_geometry_propagates_fields() {
        let b = Board::with_geometry(BoardGeometry {
            legal_move_radius: 4,
            cluster_threshold: 7,
            cluster_window_size: 9,
        });
        assert_eq!(b.legal_move_radius(), 4);
        assert_eq!(b.cluster_threshold(), 7);
        assert_eq!(b.cluster_window_size(), 9);
        assert_eq!(
            b.geometry(),
            BoardGeometry { legal_move_radius: 4, cluster_threshold: 7, cluster_window_size: 9 }
        );
    }

    #[test]
    fn clone_preserves_geometry() {
        let a = Board::with_geometry(BoardGeometry {
            legal_move_radius: 8,
            cluster_threshold: 8,
            cluster_window_size: 25,
        });
        let b = a.clone();
        assert_eq!(a.geometry(), b.geometry());
    }
}
