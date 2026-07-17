// Exceeds the 300-line soft cap: the Board state core (types, ctors, mutators,
// window helpers, Clone) ports as one line-auditable unit with its in-file tests.
use std::cell::{Cell as StdCell, UnsafeCell};
use fxhash::{FxHashMap, FxHashSet};
use super::super::zobrist::ZobristTable;
use crate::ply::Ply;

// ── MoveDiff ──────────────────────────────────────────────────────────────────

/// Captures everything mutated by one `apply_move_tracked` call so that
/// `undo_move` can reverse it in O(1) without any HashMap scan.
///
/// All fields are private; the only way to construct a `MoveDiff` is through
/// `apply_move_tracked`, and the only way to consume it is through `undo_move`.
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

// ── Player ────────────────────────────────────────────────────────────────────

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

// ── Cell ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i8)]
pub enum Cell {
    #[default]
    Empty = 0,
    P1 = 1,
    P2 = -1,
}

// ── BoardGeometry ────────────────────────────────────────────────────────────

/// Plain geometry values for a Board. Spec/registry resolution is NOT this
/// crate's job — callers (upstream layers) resolve names to values and pass
/// the values in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoardGeometry {
    /// Hex-ball radius for legal-move expansion.
    pub legal_move_radius: i32,
    /// Max `hex_distance` at which two stones share a cluster.
    pub cluster_threshold: i32,
    /// Cluster-view side length; odd, >= 7.
    pub cluster_window_size: usize,
}

// ── Board ─────────────────────────────────────────────────────────────────────

/// Sparse game board.  All state needed to continue a game from any position.
///
/// # Thread safety
///
/// `Board` is `Send + !Sync`: the lazily-rebuilt legal-move cache uses
/// `UnsafeCell` with deliberately NO `unsafe impl Sync`, so a `&Board`
/// cannot cross threads at the type level. No engine caller shares one Board
/// between threads (workers own their boards); any FFI wrapper that needs
/// `Sync` supplies its own synchronization at its own layer.
#[derive(Debug)]
pub struct Board {
    /// Sparse stone map: (q, r) → Cell.
    pub(crate) cells: FxHashMap<(i32, i32), Cell>,
    /// Whose turn it is.
    pub current_player: Player,
    /// How many moves the current player still has to place this turn.
    /// Starts at 1 on ply 0 (P1's single first move), then 2 for every turn.
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
    /// Lazily-maintained set of all currently legal moves.
    ///
    /// Uses interior mutability so that `legal_moves_set(&self)` can rebuild
    /// on demand without requiring `&mut self` (callers legitimately hold
    /// `&Board` references at rebuild time, e.g. search leaf expansion).
    ///
    /// Invariant: when `cache_dirty` is false, `legal_cache` is correct.
    /// When `cache_dirty` is true, `legal_moves_set()` rebuilds it.
    ///
    /// # Safety invariants (INV-1..INV-4)
    ///
    /// The crate's three `unsafe` expressions (two in
    /// `board::moves::legal_moves_set`, one in `Clone::clone` below) rest on:
    ///
    /// - **INV-1.** The only `&mut` into `legal_cache` is created inside
    ///   `legal_moves_set`'s rebuild block, entered iff `cache_dirty == true`,
    ///   and is dropped before the shared return borrow is created.
    /// - **INV-2.** `cache_dirty` transitions false→true only inside methods
    ///   taking `&mut self` (`apply_move`, `undo_move`,
    ///   `set_legal_move_radius`, `override_legal_move_radius`,
    ///   `mark_cache_dirty`) or on boards not yet shared (`with_geometry`'s
    ///   local, Clone's freshly built value). Enforced by privacy, not
    ///   comments: the field is private to `state::core`; the sole
    ///   crate-visible true-setter is `mark_cache_dirty(&mut self)`.
    /// - **INV-3.** Every `&FxHashSet` returned by `legal_moves_set` borrows
    ///   `*self`, so the borrow checker rejects any `&mut self` call — hence
    ///   any dirty-true transition, hence any rebuild — while such a
    ///   reference is live.
    /// - **INV-4.** The rebuild block — everything up to (but not including)
    ///   its terminal `clear_cache_dirty()` call — calls NO Board method and
    ///   reads NOTHING through the cache: its body touches only plain fields
    ///   (`cells`, the bbox fields, `legal_move_radius`) plus the local
    ///   `&mut` it owns. (`clear_cache_dirty()` itself touches only the
    ///   `Cell` flag, never the cache.) This excludes re-entrancy:
    ///   `cache_dirty` is also true DURING a rebuild (it is cleared only at
    ///   the end), so without INV-4 a call issued from inside the rebuild
    ///   block (`legal_moves_set` re-entry, or `clone()`'s shared cap read)
    ///   could create a second borrow aliasing the live exclusive one with no
    ///   other invariant violated. Adding ANY call inside the rebuild block
    ///   is a review failure.
    ///
    /// The soundness boundary spans TWO modules and is policed by review:
    /// `state::core` (owns the flag and its privacy — safe code HERE can
    /// still set the flag through `&self`, so any NEW `&self` write of
    /// `cache_dirty` in this module is a review failure) and `board::moves`
    /// (holds the unsafe blocks that rely on these invariants).
    ///
    /// `pub(crate)` is harmless: misuse from safe code is impossible
    /// (`UnsafeCell::get` yields a raw pointer).
    pub(crate) legal_cache: UnsafeCell<FxHashSet<(i32, i32)>>,
    /// Set to true by any mutating operation (apply_move / undo_move).
    /// Cleared by `legal_moves_set()` after a full rebuild.
    ///
    /// PRIVATE to `state::core`: the false→true transition is gated behind
    /// `&mut self` (`mark_cache_dirty`) — see the `legal_cache` safety
    /// invariants (INV-2).
    cache_dirty: StdCell<bool>,
    /// Per-board legal-move radius override.
    ///
    /// `legal_moves_set()` rebuilds by hex-ball expansion at this radius.
    /// Default is the canonical `moves::DEFAULT_LEGAL_MOVE_RADIUS` (5).
    pub(crate) legal_move_radius: i32,
    /// Per-board cluster connectivity threshold.
    ///
    /// Two stones share a cluster iff their `hex_distance` is ≤ this value.
    /// Default is `moves::DEFAULT_CLUSTER_THRESHOLD` (5, matching the v6
    /// wire format). Wider-window corpus generation overrides to 8.
    pub(crate) cluster_threshold: i32,
    /// Per-board cluster window side length.
    ///
    /// `get_cluster_views()` emits 2-plane snapshots of this side length.
    /// Default is `BOARD_SIZE` (19, v6 wire format).
    pub(crate) cluster_window_size: usize,
}

impl Board {
    /// Create an empty board ready for the first move.
    ///
    /// The baked constants (radius 5, threshold 5, window 19) are game-rules
    /// constants, not config defaults — config-identity resolution happens
    /// upstream; this legacy ctor stays byte-exact.
    pub fn new() -> Self {
        // Pre-populate legal_cache with the 5×5 region centred at (0,0).
        // This restricts the first move to a 25-cell neighbourhood, keeping
        // the search branching factor at ~24 for the entire game (matching the
        // bbox+2 semantics used after every stone is placed).
        //
        // The rules say "all cells legal for empty board", but hundreds of root
        // children would multiply per-sim evaluation cost for no strategic
        // benefit — the first move's location is arbitrary.
        //
        // cache_dirty starts false — no rebuild needed until a stone is placed.
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

    /// Construct a Board from plain geometry values.
    ///
    /// The sole non-default Board ctor. Callers resolve any named
    /// configuration to values BEFORE calling — this crate never sees names.
    ///
    /// Marks `cache_dirty=true` (the legal cache built in `new()` is for the
    /// default radius 5; non-default radii require a rebuild on the first
    /// `legal_moves_set()` call). Debug-asserts the window is odd and >= 7.
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

    /// Override the cluster connectivity threshold for this Board. Affects
    /// only `get_clusters()` / `get_cluster_views()`; legal-move expansion is
    /// unchanged.
    pub fn set_cluster_threshold(&mut self, threshold: i32) {
        self.cluster_threshold = threshold;
    }

    /// Current cluster threshold (default 5 = v6 wire-format).
    pub fn cluster_threshold(&self) -> i32 {
        self.cluster_threshold
    }

    /// Override the cluster window side length. Used by `get_cluster_views()`
    /// to size the 2-plane snapshot. Caller must use an odd value (>= 7);
    /// panic enforced by debug_assert.
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

    /// Override the legal-move radius for this Board.
    ///
    /// Marks `legal_cache` dirty so the next `legal_moves_set()` rebuilds at
    /// the new radius. Used by self-play callers that vary the radius per
    /// game; this setter has no Rust-level guard (any boundary-layer guard
    /// lives at the boundary).
    pub fn set_legal_move_radius(&mut self, radius: i32) {
        self.legal_move_radius = radius;
        self.cache_dirty.set(true);
    }

    /// Explicit curriculum radius override (training-time radius scheduling).
    ///
    /// Identical mechanics to `set_legal_move_radius`; kept as a distinct
    /// named entry point so curriculum call sites stay greppable.
    pub fn override_legal_move_radius(&mut self, radius: i32) {
        self.legal_move_radius = radius;
        self.cache_dirty.set(true);
    }

    /// Current legal-move radius (default 5, may be overridden via
    /// `set_legal_move_radius`).
    pub fn legal_move_radius(&self) -> i32 {
        self.legal_move_radius
    }

    // ── Legal-cache dirty-flag protocol ───────────────────────────────────────
    // The ONLY crate-visible surface over `cache_dirty`; see the `legal_cache`
    // field doc (INV-1..INV-4) for the invariants this protocol enforces.

    /// Whether the legal-move cache needs a rebuild.
    pub(crate) fn cache_is_dirty(&self) -> bool {
        self.cache_dirty.get()
    }

    /// Mark the legal-move cache clean (rebuild complete). False-only setter;
    /// safe through `&self` — a false transition can never arm a rebuild.
    pub(crate) fn clear_cache_dirty(&self) {
        self.cache_dirty.set(false);
    }

    /// Invalidate the legal-move cache: the next `legal_moves_set()` call
    /// performs a full rebuild. The sole crate-visible true-setter; requires
    /// `&mut self` (INV-2), so it cannot be called while a `&FxHashSet`
    /// returned by `legal_moves_set()` is live (INV-3).
    ///
    /// `pub` (not `pub(crate)`): the `compile_fail,E0502` doctest pinned on
    /// `legal_moves_set` compiles as an external crate and must reach this
    /// method so the pin fails for the right reason (E0502, not a
    /// private-method error).
    pub fn mark_cache_dirty(&mut self) {
        self.cache_dirty.set(true);
    }

    // ── Window ────────────────────────────────────────────────────────────────

    /// Centre of the trunk-sized view window: centroid of the bounding
    /// box. Defaults to (0, 0) on an empty board.
    ///
    /// Uses truncating-toward-zero integer division `(a+b)/2` to preserve
    /// frame calibration for legacy checkpoints trained against this
    /// semantic. Migrating to `i32::midpoint` (floor toward -∞) would shift
    /// the absolute NN window by ≤1 cell on negative-odd bbox sums — see the
    /// falsified register (midpoint row) and the inv18 pin tests.
    // Truncate-toward-zero semantics preserves anchor calibration for legacy
    // checkpoints (predecessor forensic record; pinned by inv18/inv18b).
    #[allow(clippy::manual_midpoint)]
    pub fn window_center(&self) -> (i32, i32) {
        if !self.has_stones {
            return (0, 0);
        }
        let cq = (self.min_q + self.max_q) / 2;
        let cr = (self.min_r + self.max_r) / 2;
        (cq, cr)
    }

    /// Window-relative flat index for axial (q, r) — geometry-aware.
    ///
    /// Result is in [0, trunk_sz²). Returns usize::MAX for out-of-window coords.
    ///
    /// Dispatches via `self.cluster_window_size` (the NN-input frame geometry;
    /// window indexing uses it, not the canvas size).
    #[inline]
    pub fn window_flat_idx(&self, q: i32, r: i32) -> usize {
        let (cq, cr) = self.window_center();
        let trunk_sz = self.cluster_window_size as i32;
        let half = (trunk_sz - 1) / 2;
        Self::window_flat_idx_at_geom(q, r, cq, cr, trunk_sz, half)
    }

    /// Window-relative flat index for axial (q, r) at a specific center —
    /// legacy default-geometry (19/9) associated fn.
    ///
    /// Callers that need a non-19 trunk must use `window_flat_idx_at_geom`
    /// and thread `(trunk_sz, half)` from values extracted at the boundary.
    /// This wrapper keeps byte-exact behaviour for default-geometry call sites.
    #[inline]
    pub fn window_flat_idx_at(q: i32, r: i32, cq: i32, cr: i32) -> usize {
        Self::window_flat_idx_at_geom(q, r, cq, cr, BOARD_SIZE as i32, HALF)
    }

    /// Window-relative flat index kernel — caller-threaded geometry.
    ///
    /// Scalar-only API: per-hot-loop callers pre-extract `(trunk_sz, half)`
    /// once at their boundary and pass the integer pair in. Marked
    /// `#[inline]` so the compiler can fold the bounds check + index math
    /// into the caller.
    ///
    /// `trunk_sz`: per-cluster NN input side length (= `Board::cluster_window_size`
    ///   cached on Board for self-dispatch).
    /// `half`:     `(trunk_sz - 1) / 2` — pre-computed by caller.
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

    /// Axial coordinates (q, r) from a window-relative flat index.
    ///
    /// Dispatches via `self.cluster_window_size` so non-default windows
    /// decode correctly.
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
    ///
    /// Dispatches via `self.cluster_window_size`.
    #[inline]
    pub fn in_window(&self, q: i32, r: i32) -> bool {
        let (cq, cr) = self.window_center();
        let trunk_sz = self.cluster_window_size as i32;
        let half = (trunk_sz - 1) / 2;
        let wq = q - cq + half;
        let wr = r - cr + half;
        wq >= 0 && wq < trunk_sz && wr >= 0 && wr < trunk_sz
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Iterator over all occupied cells: yields `(&(q, r), &Cell)` pairs.
    pub fn cells_iter(&self) -> impl Iterator<Item = (&(i32, i32), &Cell)> {
        self.cells.iter()
    }

    /// Cell at (q, r).  Returns Empty for unoccupied or out-of-window cells.
    #[inline]
    pub fn get(&self, q: i32, r: i32) -> Cell {
        self.cells.get(&(q, r)).copied().unwrap_or(Cell::Empty)
    }

    // ── Move application ──────────────────────────────────────────────────────

    /// Apply a move at (q, r) for the current player.
    ///
    /// Returns `Err` only if the cell is already occupied. The board is
    /// conceptually infinite — `apply_move` performs no window or radius
    /// check, and any previously-empty (q, r) is accepted. Window / radius /
    /// bbox-margin constraints are the caller's responsibility
    /// (`legal_moves_set` for search, `legal_move_radius` for self-play,
    /// etc.); this entry point is the unconditional cell-write primitive.
    ///
    /// After a successful move:
    /// - `moves_remaining` decrements.
    /// - When it reaches 0 the turn passes: `current_player` flips and
    ///   `moves_remaining` resets to 2.
    pub fn apply_move(&mut self, q: i32, r: i32) -> Result<(), &'static str> {
        if self.cells.contains_key(&(q, r)) {
            return Err("cell already occupied");
        }

        // Update bounding box FIRST so that window_flat_idx uses the final
        // bounding box — this keeps the Zobrist hash position-deterministic
        // (same stone set → same bbox → same centre → same hash).
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

        // Mark legal cache dirty — legal_moves_set() will rebuild lazily.
        // This avoids 24+ HashSet operations on every apply_move (the search
        // hot path calls apply_move ~2D times per simulation during traversal
        // and reconstruction, but legal_moves_set() is only needed once per
        // sim at leaf expansion).
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

        // Mark legal cache dirty — it will be rebuilt lazily on next access.
        // This avoids O(24) HashSet operations per undo (undo is called ~D
        // times per sim during selection traversal but legal_moves_set() is
        // not called until leaf expansion).
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
        // Skip copying legal_cache contents — rebuilding a HashSet of N entries
        // is O(N) allocation and dominates clone cost on the search hot path
        // (every leaf expansion reconstructs a board via clone + apply_move*).
        //
        // We set cache_dirty = true unconditionally so that the first
        // legal_moves_set() call on the clone rebuilds from `cells` (which IS
        // correctly copied).  This is safe even when diffs is empty (root node
        // expansion) because the rebuild is always correct given a valid `cells`.
        // SAFETY: shared read of the cache's len through the UnsafeCell.
        // Clone takes `&self`, so per INV-1 no exclusive borrow can be live:
        // the only `&mut` ever created exists inside `legal_moves_set`'s
        // rebuild block, which per INV-4 calls no Board method — `clone()`
        // can never run while it exists. Coexisting shared reads (a live
        // `&FxHashSet` from `legal_moves_set`) alias this read harmlessly.
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

// NOTE: deliberately NO `unsafe impl Sync for Board` — the `UnsafeCell` cache
// makes Board auto-`!Sync` (no impl to write; a `&Board` cannot cross threads
// at the type level), while Board stays auto-`Send` (worker threads own their
// boards). The crate carries exactly three `unsafe` expressions (two in
// `board::moves::legal_moves_set`, one in `Clone::clone` above), each resting
// on INV-1..INV-4 (see the `legal_cache` field doc). A fourth `unsafe` block
// touching the cache is a review failure.

// ── Test-fixture builder (feature `test-fixtures`, OFF by default) ─────────────
//
// A public stone-planting builder that reproduces the frozen `static_board` /
// `fwm_board` construction exactly. Additive + feature-gated: when the feature
// is OFF (the default for `cargo build`) this block is not compiled and no
// existing path can reference it, so production behaviour is byte-untouched.
// Consumed only by test/bench targets in downstream crates (e.g. the search
// crate's tactics soundness fuzz + mcts unit suite) that need non-legal-cadence
// positions the public `apply_move` cadence cannot reach.
#[cfg(feature = "test-fixtures")]
impl Board {
    /// Test-only static-position builder. Plants `stones` (bbox recomputed from
    /// their min/max), marks the legal cache dirty (so `legal_moves_set()`
    /// rebuilds on demand), and sets the turn-structure fields explicitly.
    ///
    /// `ply` is explicit because consumers vary it (`static_board` passes
    /// `stones.len()`; the mcts quiescence/CF-1 fixtures pass a specific ply).
    /// `last_move` is `Some(..)` only for a terminal-win fixture — `check_win`
    /// reads `last_move` alone, so a fixture asserting `check_win()` must supply
    /// the completing cell, and one asserting `!check_win()` leaves it `None`.
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

        // mark_cache_dirty => legal_moves_set rebuilds against the planted stones
        // (non-empty; excludes the occupied cells; radius-5 ball around them).
        let legal = b.legal_moves_set();
        assert!(!legal.is_empty(), "legal set must rebuild from planted stones");
        assert!(!legal.contains(&(2, 1)), "occupied cell is not legal");
        assert!(legal.contains(&(5, 1)), "empty neighbour must be legal");
    }

    #[test]
    fn from_stones_terminal_win_reads_last_move() {
        // A 6-in-a-row with last_move on the line makes check_win() true (it reads
        // last_move only); no last_move leaves it false.
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
    //! Re-anchored geometry-ctor pins (2 of the predecessor's 9 spec-ctor
    //! tests survive the registry decoupling; the rest die with the spec
    //! binding or defer to the encoding crate).
    use super::*;

    /// Asymmetric values (radius 4, threshold 7, window 9) so a `with_geometry`
    /// transcription bug swapping the two same-typed i32 fields cannot pass.
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
