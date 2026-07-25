// Exceeds the 300-line soft cap (R8): the full PyBoard surface (~40 methods)
// plus the inlined threat-viewer scanner (dropped from mantis-core — it is a
// viewer-only pure function with no core/search/selfplay consumer) port as one
// line-auditable unit; splitting them would need a 7th module file (out of the
// R1 write scope) for one private helper.
//! Python-visible Board wrapper over `mantis_core::Board`.
//!
//! The new `mantis_core::Board` carries plain geometry and NO encoding ref (the
//! DAG severed spec resolution from core), so this wrapper HOLDS the encoding
//! binding itself (`Option<&'static RegistrySpec>`): `with_encoding_name` sets it
//! via `mantis_encoding::lookup`; `to_tensor` routes through it +
//! `mantis_encoding::to_planes`; `size` and the radius/cluster guards read it.
//! `Board` is `Send + !Sync` (deliberately no `unsafe impl Sync`) — the bridge
//! brings single-thread Python ownership via `#[pyclass(unsendable)]` (LOCKED #3).

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArrayMethods};

use mantis_core::{Board as RustBoard, BoardGeometry, Cell, Player};
use mantis_core::board::{BOARD_SIZE, HALF, DEFAULT_CLUSTER_THRESHOLD};
use mantis_encoding::RegistrySpec;

/// Return tuple of `get_cluster_views`: a list of `(2, S, S)` view arrays
/// (current-player + opponent stones) paired with the axial (q, r) centre
/// of each cluster window.
type ClusterViewsOut = (Vec<Py<PyArray3<f32>>>, Vec<(i32, i32)>);

/// Map a Python player id (1 = P1, -1 = P2) to the Rust `Player` enum.
/// Used by the forcing-move primitive bindings. `ValueError` on any other value.
fn player_from_i8(player: i8) -> PyResult<Player> {
    match player {
        1 => Ok(Player::One),
        -1 => Ok(Player::Two),
        other => Err(PyValueError::new_err(format!(
            "player must be 1 (P1) or -1 (P2); got {other}"
        ))),
    }
}

/// A Hex Tac Toe board.
///
/// Coordinate system: axial (q, r) with -9 ≤ q, r ≤ 9 for a 19×19 grid.
///
/// Turn structure:
///   - Player 1 opens with exactly 1 move (ply 0).
///   - After that, each player places 2 stones per turn.
///
/// `unsendable` (LOCKED #3): `Board` is `Send + !Sync` (UnsafeCell legal-cache +
/// Cell dirty-flag, no `unsafe impl Sync`); single-thread Python ownership is the
/// bridge's synchronization. F-42: `module = "mantis._engine"`.
#[pyclass(name = "Board", module = "mantis._engine", unsendable)]
pub struct PyBoard {
    inner: RustBoard,
    /// Bridge-held encoding binding (relocated from core — see the module doc).
    /// `None` = a deliberately encoding-less board (`Board.new()`).
    encoding: Option<&'static RegistrySpec>,
}

impl Default for PyBoard {
    /// Equivalent to `PyBoard::new()` — empty board, no encoding bound.
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyBoard {
    /// Create a new empty board (no encoding bound).
    #[new]
    pub fn new() -> Self {
        PyBoard { inner: RustBoard::new(), encoding: None }
    }

    /// Registry-resolved Board ctor. Looks the encoding up by name in
    /// `crates/mantis-encoding/src/registry.toml` and binds the resulting
    /// `RegistrySpec` (its geometry drives the Board; the spec ref is held by
    /// the wrapper for `to_tensor`/`size`/guards). The registry path is the
    /// single supported entry point for non-default encoding construction.
    ///
    /// Raises `ValueError` if `name` is not a registered encoding.
    #[staticmethod]
    pub fn with_encoding_name(name: &str) -> PyResult<Self> {
        let spec = mantis_encoding::lookup(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown encoding {name:?}; see crates/mantis-encoding/src/registry.toml"
            ))
        })?;
        let geom = BoardGeometry {
            legal_move_radius: spec.legal_move_radius as i32,
            cluster_threshold: spec
                .cluster_threshold
                .unwrap_or(DEFAULT_CLUSTER_THRESHOLD as usize) as i32,
            cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
        };
        Ok(PyBoard {
            inner: RustBoard::with_geometry(geom),
            encoding: Some(spec),
        })
    }

    /// Place a stone at (q, r) for the current player.
    /// Raises ValueError if the move is illegal.
    pub fn apply_move(&mut self, q: i32, r: i32) -> PyResult<()> {
        self.inner
            .apply_move(q, r)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns True if either player has 6 in a row.
    pub fn check_win(&self) -> bool {
        self.inner.check_win()
    }

    /// Returns the winning player (1 or -1) or None.
    pub fn winner(&self) -> Option<i8> {
        self.inner.winner().map(|p| match p {
            Player::One => 1,
            Player::Two => -1,
        })
    }

    /// Returns the 6 cells forming the winning line, or empty list if no win.
    ///
    /// Scans all stones when last_move doesn't yield a 6-line, so a win found
    /// by `winner()`'s fallback path (HTT 2-moves-per-turn shifting last_move
    /// off the line) still surfaces the winning cells.
    pub fn find_winning_line(&self) -> Vec<(i32, i32)> {
        self.inner.find_winning_line()
    }

    /// CF-1 terminal value from the side-to-move's perspective, valid at a
    /// `check_win()` leaf: `+1.0` when `moves_remaining == 1` (first-stone win,
    /// winner still to move), `-1.0` when `moves_remaining == 2` (turn-final
    /// win, flipped to the loser).
    pub fn terminal_value_to_move(&self) -> f32 {
        self.inner.terminal_value_to_move()
    }

    // ── Forcing-move primitives (offline TSS/minimax oracle) ────────────────────
    // Read-only queries over the tested Rust win-detection logic. Exposed for an
    // offline tactical-search probe: a threat-space search / shallow MCTS-Solver
    // minimax needs the engine's OWN winning-move oracle to enumerate threats +
    // defenses and prove forced losses. `player`: 1 = P1, -1 = P2.

    /// Count empty legal cells that complete a 6-in-a-row for `player` (1 / -1).
    /// `>= 3` ⇒ provably forced win next turn (opponent blocks at most 2/turn).
    pub fn count_winning_moves(&self, player: i8) -> PyResult<u32> {
        Ok(self.inner.count_winning_moves(player_from_i8(player)?))
    }

    /// All empty legal cells that complete a 6-in-a-row for `player`, sorted.
    /// The threat/defense enumeration primitive: own-threats (player) and the
    /// cells the opponent must block (call with the opponent's id).
    pub fn winning_moves(&self, player: i8) -> PyResult<Vec<(i32, i32)>> {
        Ok(self.inner.winning_moves(player_from_i8(player)?))
    }

    /// Lexicographically-first 6-completing cell for `player`, or None.
    pub fn first_winning_move(&self, player: i8) -> PyResult<Option<(i32, i32)>> {
        Ok(self.inner.first_winning_move(player_from_i8(player)?))
    }

    /// True if `player` has ≥ `min_len` consecutive stones along any hex axis.
    pub fn has_player_long_run(&self, player: i8, min_len: usize) -> PyResult<bool> {
        Ok(self.inner.has_player_long_run(player_from_i8(player)?, min_len))
    }

    /// Cells that, if `player` plays them, create ≥1 immediate winning move (an
    /// open-4 → win-in-1) — the threat-creating move set for a threat-space search.
    /// Computed in-engine so a Python caller pays ONE FFI hop, not a clone per cell.
    pub fn threat_moves(&self, player: i8) -> PyResult<Vec<(i32, i32)>> {
        Ok(self.inner.threat_moves(player_from_i8(player)?))
    }

    /// The immediate move for the SIDE TO MOVE that proves a within-turn forced
    /// win, or None. depth≥1: a 6-completing move now; depth≥2 (only at
    /// moves_remaining==2): a first placement that wins on the same turn's 2nd
    /// stone. Reads turn-phase from moves_remaining (CF-1 discipline).
    pub fn forced_win_move(&self, depth: u8) -> Option<(i32, i32)> {
        self.inner.forced_win_move(depth)
    }

    /// List of all legal moves as list of (q, r) tuples.
    pub fn legal_moves(&self) -> Vec<(i32, i32)> {
        self.inner.legal_moves()
    }

    /// Number of legal moves (number of empty cells).
    pub fn legal_move_count(&self) -> usize {
        self.inner.legal_move_count()
    }

    /// Returns the cell value at (q, r): 0=empty, 1=P1, -1=P2.
    pub fn get(&self, q: i32, r: i32) -> i8 {
        match self.inner.get(q, r) {
            Cell::Empty => 0,
            Cell::P1 => 1,
            Cell::P2 => -1,
        }
    }

    /// Current player: 1 for player 1, -1 for player 2.
    #[getter]
    pub fn current_player(&self) -> i8 {
        match self.inner.current_player {
            Player::One => 1,
            Player::Two => -1,
        }
    }

    /// How many moves the current player still has to place this turn.
    #[getter]
    pub fn moves_remaining(&self) -> u8 {
        self.inner.moves_remaining
    }

    /// Total half-moves played (stones placed).
    #[getter]
    pub fn ply(&self) -> u32 {
        self.inner.ply.index()
    }

    /// Override the per-Board legal-move radius cap.
    ///
    /// Raises `ValueError` when the board was constructed via
    /// `Board.with_encoding_name` (encoding bound). Callers should use the
    /// registry entry instead of overriding post-construction.
    pub fn set_legal_move_radius(&mut self, radius: i32) -> PyResult<()> {
        if self.encoding.is_some() {
            return Err(PyValueError::new_err(
                "set_legal_move_radius after with_encoding_name is not supported; \
                 use registry (Board.with_encoding_name) instead of overriding post-construction"
            ));
        }
        self.inner.set_legal_move_radius(radius);
        Ok(())
    }

    /// Read the current per-Board legal-move radius cap.
    pub fn legal_move_radius(&self) -> i32 {
        self.inner.legal_move_radius()
    }

    /// Incremental Zobrist hash of the current position.
    pub fn zobrist_hash(&self) -> u128 {
        self.inner.zobrist_hash
    }

    /// Encode the board as a flat list of floats for the 18 tensor planes
    /// (shape conceptually [18, board_size, board_size] where board_size comes
    /// from the bound encoding — default v6 wire geometry, 19 → flat 18×361=6498).
    ///   plane 0: current player's stones
    ///   plane 8: opponent's stones
    ///   plane 16: moves_remaining == 2 ? 1.0 : 0.0
    ///   plane 17: ply % 2
    ///   (chain-length planes moved to the replay-buffer aux sub-buffer.)
    ///
    /// Panics for multi-window encodings (v6w25 etc.) — with `panic = "unwind"`
    /// that panic crosses the FFI as a catchable `PanicException`; use
    /// `get_cluster_views()` for those encodings.
    ///
    /// Zero-copy return via `IntoPyArray`: the returned array is a NumPy view
    /// over the Vec the encode kernel just allocated. Python callers spell:
    ///   `board.to_tensor().reshape(18, board.size, board.size)`.
    ///
    /// Panics for an encoding-less board (`Board.new()`) — no v6 default
    /// (R28, LAW-11); construct via `Board.with_encoding_name(...)` first.
    /// With `panic = "unwind"` that panic crosses the FFI as a catchable
    /// `PanicException`, matching this function's own multi-window panic
    /// arm above.
    pub fn to_tensor<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let spec = match self.encoding {
            Some(s) => s,
            None => panic!(
                "Board::to_tensor called on an encoding-less Board (Board.new()); \
                 construct via Board.with_encoding_name(...) first — no v6 default \
                 (R28, LAW-11)"
            ),
        };
        mantis_encoding::to_planes(&self.inner, spec).into_pyarray(py)
    }

    /// Returns a tuple of (list of NumPy arrays, list of (q, r) centers) for each cluster.
    ///
    /// Each NumPy array has shape `(2, S, S)` where `S = self.cluster_window_size`
    /// (default 19 = v6 wire format; v6w25 callers `set_cluster_window_size(25)`).
    /// Plane 0 = current player's stones, plane 1 = opponent's stones. Arrays are
    /// created via zero-copy transfer from Rust allocations.
    pub fn get_cluster_views(
        &self,
        py: Python<'_>,
    ) -> PyResult<ClusterViewsOut> {
        let window_size = self.inner.cluster_window_size();
        let (views, centers) = self.inner.get_cluster_views();
        let py_views: PyResult<Vec<_>> = views
            .into_iter()
            .map(|v| {
                // Transfer Vec ownership to NumPy (zero-copy), then reshape.
                PyArray1::from_vec(py, v)
                    .reshape([2_usize, window_size, window_size])
                    .map(pyo3::Bound::unbind)
            })
            .collect();
        Ok((py_views?, centers))
    }

    /// Set the cluster connectivity threshold (default 5). Used by v6w25 corpus
    /// generation to widen cluster reach to 8. Affects only `get_clusters()` /
    /// `get_cluster_views()`; legal-move expansion is independent.
    ///
    /// Raises `ValueError` when the board was constructed via
    /// `Board.with_encoding_name` (encoding bound). Use registry entry instead.
    pub fn set_cluster_threshold(&mut self, threshold: i32) -> PyResult<()> {
        if self.encoding.is_some() {
            return Err(PyValueError::new_err(
                "set_cluster_threshold after with_encoding_name is not supported; \
                 use registry (Board.with_encoding_name) instead of overriding post-construction"
            ));
        }
        self.inner.set_cluster_threshold(threshold);
        Ok(())
    }

    /// Current cluster threshold (default 5).
    pub fn cluster_threshold(&self) -> i32 {
        self.inner.cluster_threshold()
    }

    /// Set the cluster window side length (default 19). Used by v6w25 corpus
    /// generation to produce 25×25 cluster windows. Caller must use an odd value
    /// >= 7. Returns ValueError on bad input.
    ///
    /// Raises `ValueError` when the board was constructed via
    /// `Board.with_encoding_name` (encoding bound). Use registry entry instead.
    pub fn set_cluster_window_size(&mut self, size: usize) -> PyResult<()> {
        if self.encoding.is_some() {
            return Err(PyValueError::new_err(
                "set_cluster_window_size after with_encoding_name is not supported; \
                 use registry (Board.with_encoding_name) instead of overriding post-construction"
            ));
        }
        if size < 7 || size.is_multiple_of(2) {
            return Err(PyValueError::new_err(format!(
                "cluster_window_size must be odd and >= 7; got {size}"
            )));
        }
        self.inner.set_cluster_window_size(size);
        Ok(())
    }

    /// Current cluster window side length (default 19).
    pub fn cluster_window_size(&self) -> usize {
        self.inner.cluster_window_size()
    }

    /// Window-relative flat index for axial (q, r).
    /// Used by selfplay workers to convert legal-move coords to policy indices.
    pub fn to_flat(&self, q: i32, r: i32) -> usize {
        self.inner.window_flat_idx(q, r)
    }

    /// Board size (cells per axis). Default 19 (v6 wire format); honors the
    /// encoding bound at construction via `with_encoding_name` (e.g. 25 for
    /// v6w25). A raw geometry default on a deliberately encoding-less board —
    /// NOT the identity-resolution path (that hard-errors on an unknown name).
    #[getter]
    pub fn size(&self) -> usize {
        self.encoding.map_or(BOARD_SIZE, |s| s.board_size)
    }

    /// Returns threat cells as list of (q, r, level, player) tuples.
    /// Threats are EMPTY cells within threatening windows. Viewer only.
    pub fn get_threats(&self) -> Vec<(i32, i32, u8, u8)> {
        let mut stones = std::collections::HashMap::new();
        for (&(q, r), &cell) in self.inner.cells_iter() {
            let player = match cell {
                Cell::P1 => 0u8,
                Cell::P2 => 1u8,
                Cell::Empty => continue,
            };
            stones.insert((q, r), player);
        }
        threats::get_threats(&stones)
            .into_iter()
            .map(|t| (t.q, t.r, t.level, t.player))
            .collect()
    }

    /// Returns a list of all stones on the board as (q, r, player).
    pub fn get_stones(&self) -> Vec<(i32, i32, i8)> {
        self.inner
            .cells_iter()
            .map(|(&(q, r), &cell)| {
                let p = match cell {
                    Cell::Empty => 0,
                    Cell::P1 => 1,
                    Cell::P2 => -1,
                };
                (q, r, p)
            })
            .collect()
    }

    /// Return a deep clone of this board (carries the encoding binding).
    pub fn clone(&self) -> PyBoard {
        PyBoard { inner: self.inner.clone(), encoding: self.encoding }
    }

    /// Python copy.copy() support.
    pub fn __copy__(&self) -> PyBoard {
        PyBoard { inner: self.inner.clone(), encoding: self.encoding }
    }

    /// Python copy.deepcopy() support.
    pub fn __deepcopy__(&self, _memo: Py<PyAny>) -> PyBoard {
        PyBoard { inner: self.inner.clone(), encoding: self.encoding }
    }

    pub fn __repr__(&self) -> String {
        let mut s = format!(
            "Board(ply={}, player={}, moves_remaining={})\n",
            self.inner.ply.index(),
            match self.inner.current_player { Player::One => 1, Player::Two => -1 },
            self.inner.moves_remaining,
        );
        let (cq, cr) = self.inner.window_center();
        // wr=18 is top row visually; wq=0 is left column
        for wr in (0..BOARD_SIZE).rev() {
            for wq in 0..BOARD_SIZE {
                let q = wq as i32 - HALF + cq;
                let r = wr as i32 - HALF + cr;
                let c = match self.inner.get(q, r) {
                    Cell::Empty => '.',
                    Cell::P1 => 'X',
                    Cell::P2 => 'O',
                };
                s.push(c);
                s.push(' ');
            }
            s.push('\n');
        }
        s
    }
}

impl PyBoard {
    /// Construct a PyBoard directly from a Rust Board (used by PyMCTSTree
    /// leaf marshaling). The new `mantis_core::Board` carries no encoding ref, so
    /// the wrapper's encoding is `None` — leaf boards are pure geometry (the
    /// encoding is a bridge-only concern bound solely via `with_encoding_name`).
    pub(crate) fn from_inner(inner: RustBoard) -> Self {
        PyBoard { inner, encoding: None }
    }

    /// Crate-internal accessor for the wrapped Rust Board. Used by PyMCTSTree /
    /// PyTacticalSolver (sibling modules) to read the underlying board across
    /// the PyO3 boundary — `inner` is private.
    pub(crate) fn inner_ref(&self) -> &RustBoard {
        &self.inner
    }
}

// ── Threat-viewer scanner (inlined; dropped from mantis-core) ────────────────
//
// Scans all three hex axes for length-6 windows where one player has N stones
// (N >= 3) and the rest are empty (no opponent blocking). The highlighted cells
// are the EMPTY cells within the window. Never called from MCTS or training —
// viewer only. Ported verbatim from the predecessor's `board::threats`, which
// the new mantis-core omits (no core/search/selfplay consumer).
mod threats {
    use std::collections::HashMap;

    use mantis_core::board::HEX_AXES;

    const WIN_LEN: usize = 6;

    /// Endpoint-bounded line for `scan_line` (axes (1,0) and (0,1)).
    #[derive(Clone, Copy, Debug)]
    struct ScanLineParams {
        start: (i32, i32),
        end: (i32, i32),
        axis: (i32, i32),
    }

    /// Bbox-walking line for `scan_line_general` (axis (1,-1)).
    #[derive(Clone, Copy, Debug)]
    struct ScanLineGeneralParams {
        start: (i32, i32),
        axis: (i32, i32),
        bbox_min: (i32, i32),
        bbox_max: (i32, i32),
    }

    /// A single threat cell: an empty cell within a threatening window.
    #[derive(Debug, Clone, Copy)]
    pub(super) struct ThreatCell {
        pub q: i32,
        pub r: i32,
        pub level: u8, // 3=warning, 4=forced, 5=critical
        pub player: u8, // 0 or 1
    }

    /// Scan the board for threat cells. `stones` maps (q, r) -> player (0 or 1).
    pub(super) fn get_threats<S: ::std::hash::BuildHasher>(
        stones: &HashMap<(i32, i32), u8, S>,
    ) -> Vec<ThreatCell> {
        if stones.is_empty() {
            return Vec::new();
        }

        // Compute bounding box, extended by WIN_LEN in each direction.
        let mut min_q = i32::MAX;
        let mut max_q = i32::MIN;
        let mut min_r = i32::MAX;
        let mut max_r = i32::MIN;
        for &(q, r) in stones.keys() {
            if q < min_q { min_q = q; }
            if q > max_q { max_q = q; }
            if r < min_r { min_r = r; }
            if r > max_r { max_r = r; }
        }
        let margin = WIN_LEN as i32;
        min_q -= margin;
        max_q += margin;
        min_r -= margin;
        max_r += margin;

        // Track best threat level per (q, r, player).
        let mut best: HashMap<(i32, i32, u8), u8> = HashMap::new();

        for &(dq, dr) in &HEX_AXES {
            if dq == 1 && dr == 0 {
                // Lines indexed by r. For each r, slide window over q.
                for r in min_r..=max_r {
                    scan_line(stones, &mut best, ScanLineParams {
                        start: (min_q, r),
                        end: (max_q, r),
                        axis: (dq, dr),
                    });
                }
            } else if dq == 0 && dr == 1 {
                // Lines indexed by q. For each q, slide window over r.
                for q in min_q..=max_q {
                    scan_line(stones, &mut best, ScanLineParams {
                        start: (q, min_r),
                        end: (q, max_r),
                        axis: (dq, dr),
                    });
                }
            } else {
                // (1, -1): lines indexed by q+r. For constant s = q+r,
                // q ranges and r = s - q.
                let min_s = min_q + min_r;
                let max_s = max_q + max_r;
                for s in min_s..=max_s {
                    let start_q = min_q;
                    let start_r = s - start_q;
                    scan_line_general(stones, &mut best, ScanLineGeneralParams {
                        start: (start_q, start_r),
                        axis: (dq, dr),
                        bbox_min: (min_q, min_r),
                        bbox_max: (max_q, max_r),
                    });
                }
            }
        }

        best.into_iter()
            .map(|((q, r, player), level)| ThreatCell { q, r, level, player })
            .collect()
    }

    /// Scan a line along direction (dq, dr) starting at (start_q, start_r).
    fn scan_line<S: ::std::hash::BuildHasher>(
        stones: &HashMap<(i32, i32), u8, S>,
        best: &mut HashMap<(i32, i32, u8), u8>,
        params: ScanLineParams,
    ) {
        let ScanLineParams { start: (start_q, start_r), end: (end_q, end_r), axis: (dq, dr) } = params;

        let (line_start_q, line_start_r, steps) = if dq == 1 && dr == 0 {
            (start_q, start_r, (end_q - start_q + 1) as usize)
        } else {
            (start_q, start_r, (end_r - start_r + 1) as usize)
        };

        if steps < WIN_LEN {
            return;
        }

        for w in 0..=(steps - WIN_LEN) {
            let wq = line_start_q + (w as i32) * dq;
            let wr = line_start_r + (w as i32) * dr;
            check_window(stones, best, wq, wr, dq, dr);
        }
    }

    /// General line scanner for the (1,-1) direction.
    fn scan_line_general<S: ::std::hash::BuildHasher>(
        stones: &HashMap<(i32, i32), u8, S>,
        best: &mut HashMap<(i32, i32, u8), u8>,
        params: ScanLineGeneralParams,
    ) {
        let ScanLineGeneralParams {
            start: (start_q, start_r),
            axis: (dq, dr),
            bbox_min: (min_q, min_r),
            bbox_max: (max_q, max_r),
        } = params;

        // Count how many steps we can take from start in direction (dq, dr)
        // while staying within bounds.
        let mut steps = 0usize;
        loop {
            let q = start_q + (steps as i32) * dq;
            let r = start_r + (steps as i32) * dr;
            if q < min_q || q > max_q || r < min_r || r > max_r {
                break;
            }
            steps += 1;
        }

        if steps < WIN_LEN {
            return;
        }

        for w in 0..=(steps - WIN_LEN) {
            let wq = start_q + (w as i32) * dq;
            let wr = start_r + (w as i32) * dr;
            check_window(stones, best, wq, wr, dq, dr);
        }
    }

    /// Check a single window of WIN_LEN cells starting at (wq, wr) in direction (dq, dr).
    fn check_window<S: ::std::hash::BuildHasher>(
        stones: &HashMap<(i32, i32), u8, S>,
        best: &mut HashMap<(i32, i32, u8), u8>,
        wq: i32,
        wr: i32,
        dq: i32,
        dr: i32,
    ) {
        let mut p0_count = 0u8;
        let mut p1_count = 0u8;
        let mut empties: [(i32, i32); WIN_LEN] = [(0, 0); WIN_LEN];
        let mut n_empties = 0usize;

        for i in 0..WIN_LEN {
            let cq = wq + (i as i32) * dq;
            let cr = wr + (i as i32) * dr;
            match stones.get(&(cq, cr)) {
                Some(&0) => p0_count += 1,
                Some(&1) => p1_count += 1,
                None => {
                    empties[n_empties] = (cq, cr);
                    n_empties += 1;
                }
                _ => {}
            }
        }

        // Threat for player 0.
        if p1_count == 0 && p0_count >= 3 {
            let level = p0_count; // 3=warning, 4=forced, 5=critical
            for &(eq, er) in empties.iter().take(n_empties) {
                let entry = best.entry((eq, er, 0)).or_insert(0);
                if level > *entry {
                    *entry = level;
                }
            }
        }

        // Threat for player 1.
        if p0_count == 0 && p1_count >= 3 {
            let level = p1_count;
            for &(eq, er) in empties.iter().take(n_empties) {
                let entry = best.entry((eq, er, 1)).or_insert(0);
                if level > *entry {
                    *entry = level;
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn empty_board_no_threats() {
            let stones: HashMap<(i32, i32), u8> = HashMap::new();
            assert!(get_threats(&stones).is_empty());
        }

        #[test]
        fn threat_forced_two_gaps() {
            // Line: O O O _ O along axis (1,0) at r=0 — 4 P1 stones at q=0,1,2,4.
            let mut stones = HashMap::new();
            stones.insert((0, 0), 1u8);
            stones.insert((1, 0), 1u8);
            stones.insert((2, 0), 1u8);
            stones.insert((4, 0), 1u8);

            let threats = get_threats(&stones);
            let forced: Vec<_> = threats.iter().filter(|t| t.level == 4 && t.player == 1).collect();
            assert!(forced.iter().any(|t| t.q == 3 && t.r == 0));
            assert!(forced.iter().any(|t| t.q == 5 && t.r == 0));
            assert!(!forced.iter().any(|t| t.q == 0 && t.r == 0));
        }

        #[test]
        fn critical_threat_five_in_row() {
            let mut stones = HashMap::new();
            for q in 0..5 {
                stones.insert((q, 0), 0u8);
            }
            let threats = get_threats(&stones);
            let critical: Vec<_> = threats.iter().filter(|t| t.level == 5 && t.player == 0).collect();
            assert!(critical.iter().any(|t| t.q == 5 && t.r == 0));
            assert!(critical.iter().any(|t| t.q == -1 && t.r == 0));
        }

        #[test]
        fn nw_axis_threats() {
            // 4 stones along NW axis (1,-1): (0,0),(1,-1),(2,-2),(3,-3) for player 0.
            let mut stones = HashMap::new();
            for i in 0..4 {
                stones.insert((i, -i), 0u8);
            }
            let threats = get_threats(&stones);
            let forced: Vec<_> = threats.iter().filter(|t| t.level == 4 && t.player == 0).collect();
            assert!(!forced.is_empty());
            for f in &forced {
                assert!(!stones.contains_key(&(f.q, f.r)));
            }
        }
    }
}

/// Register the `Board` pyclass into `_engine`. Called by Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_board_is_encoding_less() {
        let b = PyBoard::new();
        assert!(b.encoding.is_none());
        assert_eq!(b.size(), BOARD_SIZE, "encoding-less board size defaults to v6 geometry");
        assert_eq!(b.current_player(), 1);
        assert_eq!(b.moves_remaining(), 1);
        assert_eq!(b.ply(), 0);
    }

    #[test]
    fn with_encoding_name_binds_geometry_and_size() {
        let b = PyBoard::with_encoding_name("v6").expect("v6 registered");
        assert_eq!(b.size(), 19);
        assert!(b.encoding.is_some());
        let w25 = PyBoard::with_encoding_name("v6w25").expect("v6w25 registered");
        assert_eq!(w25.size(), 25, "bound v6w25 board size = 25");
        assert_eq!(w25.cluster_window_size(), 25);
    }

    #[test]
    fn with_encoding_name_unknown_errors() {
        assert!(PyBoard::with_encoding_name("not_a_real_encoding").is_err());
    }

    #[test]
    fn radius_and_cluster_guards_fire_when_bound() {
        let mut b = PyBoard::with_encoding_name("v6").unwrap();
        assert!(b.set_legal_move_radius(4).is_err());
        assert!(b.set_cluster_threshold(8).is_err());
        assert!(b.set_cluster_window_size(25).is_err());
    }

    #[test]
    fn set_cluster_window_size_bounds_on_encoding_less() {
        let mut b = PyBoard::new();
        assert!(b.set_cluster_window_size(6).is_err(), "even rejected");
        assert!(b.set_cluster_window_size(5).is_err(), "< 7 rejected");
        assert!(b.set_cluster_window_size(25).is_ok());
        assert_eq!(b.cluster_window_size(), 25);
    }

    #[test]
    fn apply_move_and_win_primitives() {
        let mut b = PyBoard::new();
        assert!(b.apply_move(0, 0).is_ok());
        assert!(b.apply_move(0, 0).is_err(), "occupied cell rejected");
        assert_eq!(b.current_player(), -1);
        assert!(!b.check_win());
        assert_eq!(b.winner(), None);
        // get_stones reflects the single placed stone.
        let stones = b.get_stones();
        assert_eq!(stones, vec![(0, 0, 1)]);
    }

    #[test]
    fn clone_preserves_encoding_binding() {
        let b = PyBoard::with_encoding_name("v6w25").unwrap();
        let c = b.clone();
        assert_eq!(c.size(), 25);
        assert!(c.encoding.is_some());
    }

    #[test]
    fn from_inner_is_encoding_less() {
        let inner = RustBoard::new();
        let b = PyBoard::from_inner(inner);
        assert!(b.encoding.is_none());
        assert_eq!(b.inner_ref().moves_remaining, 1);
    }

    #[test]
    fn get_threats_surfaces_open_line() {
        // Build a clean P1 3-in-a-row along E via the 2-stone-turn cadence
        // (apply_move is the unconditional cell-write primitive — P2 fillers
        // placed far off the E line so they never share a length-6 window).
        let mut b = PyBoard::new();
        b.apply_move(0, 0).unwrap(); // P1 opening single -> P2 turn (mr 2)
        b.apply_move(0, 12).unwrap(); // P2 filler
        b.apply_move(1, 12).unwrap(); // P2 filler -> P1 turn (mr 2)
        b.apply_move(1, 0).unwrap(); // P1
        b.apply_move(2, 0).unwrap(); // P1 -> P1 now has (0,0),(1,0),(2,0)
        // get_threats maps the viewer scan; P1 is player id 0. A 3-in-a-row in
        // an open length-6 window is at least a level-3 warning.
        let empty = PyBoard::new();
        assert!(empty.get_threats().is_empty(), "empty board has no threats");
        let threats = b.get_threats();
        assert!(threats.iter().any(|&(_, _, level, player)| level >= 3 && player == 0),
            "expected a P1 (id 0) warning/forced threat, got {threats:?}");
        // Every returned threat cell must be EMPTY (viewer contract).
        for &(q, r, _, _) in &threats {
            assert_eq!(b.get(q, r), 0, "threat cell ({q},{r}) must be empty");
        }
    }
}
