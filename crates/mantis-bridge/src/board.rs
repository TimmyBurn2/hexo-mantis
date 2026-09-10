// Exceeds the 300-line soft cap (R8): the PyBoard surface plus the inlined threat-viewer
// scanner are one auditable unit; splitting them adds a module file for one private helper.
//! Python-visible Board wrapper over `mantis_core::Board`, which carries plain geometry and NO
//! encoding ref — so this wrapper HOLDS the encoding binding and `with_encoding_name` sets it.
//! `Board` is `Send + !Sync`; single-thread Python ownership is the bridge's synchronization.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_core::board::{BOARD_SIZE, DEFAULT_CLUSTER_THRESHOLD, HALF};
use mantis_core::{Board as RustBoard, BoardGeometry, Cell, Player};
use mantis_encoding::RegistrySpec;

/// Map a Python player id (1 = P1, -1 = P2) to the Rust `Player` enum; `ValueError` otherwise.
fn player_from_i8(player: i8) -> PyResult<Player> {
    match player {
        1 => Ok(Player::One),
        -1 => Ok(Player::Two),
        other => Err(PyValueError::new_err(format!(
            "player must be 1 (P1) or -1 (P2); got {other}"
        ))),
    }
}

/// A Hex Tac Toe board in axial (q, r). Player 1 opens with one move; then 2 stones per turn.
#[pyclass(name = "Board", module = "mantis._engine", unsendable)]
pub struct PyBoard {
    inner: RustBoard,
    /// Bridge-held encoding binding; `None` = a deliberately encoding-less board.
    encoding: Option<&'static RegistrySpec>,
}

impl Default for PyBoard {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyBoard {
    #[new]
    pub fn new() -> Self {
        PyBoard {
            inner: RustBoard::new(),
            encoding: None,
        }
    }

    /// Registry-resolved Board ctor: binds the named spec, whose geometry drives the Board.
    /// `ValueError` on an unregistered name.
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
                .unwrap_or(DEFAULT_CLUSTER_THRESHOLD as usize)
                as i32,
            cluster_window_size: spec.cluster_window_size.unwrap_or(spec.board_size),
        };
        Ok(PyBoard {
            inner: RustBoard::with_geometry(geom),
            encoding: Some(spec),
        })
    }

    /// Place a stone at (q, r) for the current player; `ValueError` if the move is illegal.
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

    /// The 6 cells forming the winning line, or an empty list, including the all-stones fallback
    /// `winner()` uses when last_move does not lie on the line.
    pub fn find_winning_line(&self) -> Vec<(i32, i32)> {
        self.inner.find_winning_line()
    }

    /// CF-1 terminal value from the side-to-move's perspective at a `check_win()` leaf: `+1.0`
    /// when `moves_remaining == 1` (winner still to move), `-1.0` when it is 2 (flipped to loser).
    pub fn terminal_value_to_move(&self) -> f32 {
        self.inner.terminal_value_to_move()
    }

    // Forcing-move primitives over the tested win-detection logic. `player`: 1 = P1, -1 = P2.

    /// Count empty legal cells completing a 6 for `player`; `>= 3` is a forced win next turn.
    pub fn count_winning_moves(&self, player: i8) -> PyResult<u32> {
        Ok(self.inner.count_winning_moves(player_from_i8(player)?))
    }

    /// All empty legal cells completing a 6 for `player`, sorted — own threats, or the
    /// opponent's must-block cells when called with the opponent's id.
    pub fn winning_moves(&self, player: i8) -> PyResult<Vec<(i32, i32)>> {
        Ok(self.inner.winning_moves(player_from_i8(player)?))
    }

    /// Lexicographically-first 6-completing cell for `player`, or None.
    pub fn first_winning_move(&self, player: i8) -> PyResult<Option<(i32, i32)>> {
        Ok(self.inner.first_winning_move(player_from_i8(player)?))
    }

    /// True if `player` has ≥ `min_len` consecutive stones along any hex axis.
    pub fn has_player_long_run(&self, player: i8, min_len: usize) -> PyResult<bool> {
        Ok(self
            .inner
            .has_player_long_run(player_from_i8(player)?, min_len))
    }

    /// Cells that, if `player` plays them, create ≥1 immediate winning move. In-engine, so a
    /// Python caller pays ONE FFI hop.
    pub fn threat_moves(&self, player: i8) -> PyResult<Vec<(i32, i32)>> {
        Ok(self.inner.threat_moves(player_from_i8(player)?))
    }

    /// The immediate move for the SIDE TO MOVE that proves a within-turn forced win, or None.
    /// depth≥1 completes a 6 now; depth≥2 (only at `moves_remaining == 2`) wins on the 2nd stone.
    pub fn forced_win_move(&self, depth: u8) -> Option<(i32, i32)> {
        self.inner.forced_win_move(depth)
    }

    /// List of all legal moves as list of (q, r) tuples.
    pub fn legal_moves(&self) -> Vec<(i32, i32)> {
        self.inner.legal_moves()
    }

    pub fn legal_move_count(&self) -> usize {
        self.inner.legal_move_count()
    }

    /// Is `(q, r)` in the board's authoritative legal-move set? Reads the SAME `FxHashSet`
    /// `legal_moves` collects from. `apply_move` deliberately does NOT consult it: it replays
    /// moves already known legal.
    pub fn is_legal(&self, q: i32, r: i32) -> bool {
        self.inner.legal_moves_set().contains(&(q, r))
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

    /// Override the per-Board legal-move radius cap; `ValueError` on an encoding-bound board.
    pub fn set_legal_move_radius(&mut self, radius: i32) -> PyResult<()> {
        if self.encoding.is_some() {
            return Err(PyValueError::new_err(
                "set_legal_move_radius after with_encoding_name is not supported; \
                 use registry (Board.with_encoding_name) instead of overriding post-construction",
            ));
        }
        self.inner.set_legal_move_radius(radius);
        Ok(())
    }

    pub fn legal_move_radius(&self) -> i32 {
        self.inner.legal_move_radius()
    }

    /// Incremental Zobrist hash of the current position.
    pub fn zobrist_hash(&self) -> u128 {
        self.inner.zobrist_hash
    }

    /// Window-relative flat index for axial (q, r), converting coords to policy indices.
    pub fn to_flat(&self, q: i32, r: i32) -> usize {
        self.inner.window_flat_idx(q, r)
    }

    /// Board size (cells per axis), honouring the encoding bound; the default is raw geometry on
    /// an encoding-less board, NOT the identity-resolution path.
    #[getter]
    pub fn size(&self) -> usize {
        self.encoding.map_or(BOARD_SIZE, |s| s.board_size)
    }

    /// Threat cells as (q, r, level, player) tuples — EMPTY cells within threatening windows.
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
        PyBoard {
            inner: self.inner.clone(),
            encoding: self.encoding,
        }
    }

    pub fn __copy__(&self) -> PyBoard {
        PyBoard {
            inner: self.inner.clone(),
            encoding: self.encoding,
        }
    }

    pub fn __deepcopy__(&self, _memo: Py<PyAny>) -> PyBoard {
        PyBoard {
            inner: self.inner.clone(),
            encoding: self.encoding,
        }
    }

    pub fn __repr__(&self) -> String {
        let mut s = format!(
            "Board(ply={}, player={}, moves_remaining={})\n",
            self.inner.ply.index(),
            match self.inner.current_player {
                Player::One => 1,
                Player::Two => -1,
            },
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
    /// Construct a PyBoard from a Rust Board (leaf marshaling); encoding is `None`, because leaf
    /// boards are pure geometry.
    pub(crate) fn from_inner(inner: RustBoard) -> Self {
        PyBoard {
            inner,
            encoding: None,
        }
    }

    /// Crate-internal accessor for the wrapped Rust Board, since `inner` is private.
    pub(crate) fn inner_ref(&self) -> &RustBoard {
        &self.inner
    }
}

// Threat-viewer scanner (inlined; dropped from mantis-core): scans all three hex axes for
// length-6 windows where one player has N >= 3 stones and the rest are empty. Viewer only.
mod threats {
    use std::collections::HashMap;

    // The window length IS the game's win length and the axis set IS the board's, so a literal
    // here would drift from the rule `Board::player_wins` enforces.
    use mantis_core::board::{HEX_AXES, WIN_LENGTH as WIN_LEN};

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
        pub level: u8,  // 3=warning, 4=forced, 5=critical
        pub player: u8, // 0 or 1
    }

    /// Scan the board for threat cells. `stones` maps (q, r) -> player (0 or 1).
    pub(super) fn get_threats<S: ::std::hash::BuildHasher>(
        stones: &HashMap<(i32, i32), u8, S>,
    ) -> Vec<ThreatCell> {
        if stones.is_empty() {
            return Vec::new();
        }

        let mut min_q = i32::MAX;
        let mut max_q = i32::MIN;
        let mut min_r = i32::MAX;
        let mut max_r = i32::MIN;
        for &(q, r) in stones.keys() {
            if q < min_q {
                min_q = q;
            }
            if q > max_q {
                max_q = q;
            }
            if r < min_r {
                min_r = r;
            }
            if r > max_r {
                max_r = r;
            }
        }
        let margin = WIN_LEN as i32;
        min_q -= margin;
        max_q += margin;
        min_r -= margin;
        max_r += margin;

        let mut best: HashMap<(i32, i32, u8), u8> = HashMap::new();

        for &(dq, dr) in &HEX_AXES {
            if dq == 1 && dr == 0 {
                for r in min_r..=max_r {
                    scan_line(
                        stones,
                        &mut best,
                        ScanLineParams {
                            start: (min_q, r),
                            end: (max_q, r),
                            axis: (dq, dr),
                        },
                    );
                }
            } else if dq == 0 && dr == 1 {
                for q in min_q..=max_q {
                    scan_line(
                        stones,
                        &mut best,
                        ScanLineParams {
                            start: (q, min_r),
                            end: (q, max_r),
                            axis: (dq, dr),
                        },
                    );
                }
            } else {
                // (1, -1): lines indexed by s = q+r, with q ranging and r = s - q.
                let min_s = min_q + min_r;
                let max_s = max_q + max_r;
                for s in min_s..=max_s {
                    let start_q = min_q;
                    let start_r = s - start_q;
                    scan_line_general(
                        stones,
                        &mut best,
                        ScanLineGeneralParams {
                            start: (start_q, start_r),
                            axis: (dq, dr),
                            bbox_min: (min_q, min_r),
                            bbox_max: (max_q, max_r),
                        },
                    );
                }
            }
        }

        best.into_iter()
            .map(|((q, r, player), level)| ThreatCell {
                q,
                r,
                level,
                player,
            })
            .collect()
    }

    fn scan_line<S: ::std::hash::BuildHasher>(
        stones: &HashMap<(i32, i32), u8, S>,
        best: &mut HashMap<(i32, i32, u8), u8>,
        params: ScanLineParams,
    ) {
        let ScanLineParams {
            start: (start_q, start_r),
            end: (end_q, end_r),
            axis: (dq, dr),
        } = params;

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

        if p1_count == 0 && p0_count >= 3 {
            let level = p0_count; // 3=warning, 4=forced, 5=critical
            for &(eq, er) in empties.iter().take(n_empties) {
                let entry = best.entry((eq, er, 0)).or_insert(0);
                if level > *entry {
                    *entry = level;
                }
            }
        }

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
            let forced: Vec<_> = threats
                .iter()
                .filter(|t| t.level == 4 && t.player == 1)
                .collect();
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
            let critical: Vec<_> = threats
                .iter()
                .filter(|t| t.level == 5 && t.player == 0)
                .collect();
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
            let forced: Vec<_> = threats
                .iter()
                .filter(|t| t.level == 4 && t.player == 0)
                .collect();
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
        assert_eq!(
            b.size(),
            BOARD_SIZE,
            "encoding-less board size falls back to the raw geometry default"
        );
        assert_eq!(b.current_player(), 1);
        assert_eq!(b.moves_remaining(), 1);
        assert_eq!(b.ply(), 0);
    }

    #[test]
    fn with_encoding_name_binds_geometry_and_size() {
        // The two registered rows differ only in RADIUS, read from the registry, so a move reds.
        for name in ["gnn_axis_v1", "gnn_axis_r8"] {
            let spec = mantis_encoding::lookup_or_panic(name);
            let b = PyBoard::with_encoding_name(name).expect("a registered encoding");
            assert!(b.encoding.is_some());
            assert_eq!(b.size(), spec.board_size, "{name}: bound board size");
            assert_eq!(
                b.legal_move_radius(),
                spec.legal_move_radius as i32,
                "{name}: the spec's radius must drive the bound Board"
            );
        }
        assert_ne!(
            PyBoard::with_encoding_name("gnn_axis_v1")
                .expect("registered")
                .legal_move_radius(),
            PyBoard::with_encoding_name("gnn_axis_r8")
                .expect("registered")
                .legal_move_radius(),
            "the two rows exist to differ in radius; if they stop, this test is vacuous"
        );
    }

    #[test]
    fn a_deleted_grid_encoding_name_is_refused() {
        // The grid rows are gone from the registry, so binding one is the unknown-encoding error.
        for name in ["v6", "v6w25", "v6_live2_ls"] {
            assert!(
                PyBoard::with_encoding_name(name).is_err(),
                "{name} was deleted with the dense path and must not resolve"
            );
        }
    }

    #[test]
    fn with_encoding_name_unknown_errors() {
        assert!(PyBoard::with_encoding_name("not_a_real_encoding").is_err());
    }

    #[test]
    fn radius_guard_fires_when_bound() {
        // The radius override is the guard that remains, and an encoding-bound board refuses it.
        let mut b = PyBoard::with_encoding_name("gnn_axis_v1").expect("registered");
        assert!(b.set_legal_move_radius(4).is_err());
        let mut free = PyBoard::new();
        assert!(
            free.set_legal_move_radius(4).is_ok(),
            "the guard is about the BINDING, not the value — an unbound board still accepts it"
        );
    }

    #[test]
    fn apply_move_and_win_primitives() {
        let mut b = PyBoard::new();
        assert!(b.apply_move(0, 0).is_ok());
        assert!(b.apply_move(0, 0).is_err(), "occupied cell rejected");
        assert_eq!(b.current_player(), -1);
        assert!(!b.check_win());
        assert_eq!(b.winner(), None);
        let stones = b.get_stones();
        assert_eq!(stones, vec![(0, 0, 1)]);
    }

    #[test]
    fn clone_preserves_encoding_binding() {
        let b = PyBoard::with_encoding_name("gnn_axis_r8").expect("registered");
        let c = b.clone();
        assert_eq!(c.size(), b.size());
        assert_eq!(
            c.legal_move_radius(),
            8,
            "the r8 row's radius survives the clone"
        );
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
        // A clean P1 3-in-a-row along E, with P2 fillers far off the line.
        let mut b = PyBoard::new();
        b.apply_move(0, 0).unwrap(); // P1 opening single -> P2 turn (mr 2)
        b.apply_move(0, 12).unwrap(); // P2 filler
        b.apply_move(1, 12).unwrap(); // P2 filler -> P1 turn (mr 2)
        b.apply_move(1, 0).unwrap(); // P1
        b.apply_move(2, 0).unwrap(); // P1 -> P1 now has (0,0),(1,0),(2,0)
                                     // P1 is player id 0; an open-window 3-in-a-row is level 3+.
        let empty = PyBoard::new();
        assert!(empty.get_threats().is_empty(), "empty board has no threats");
        let threats = b.get_threats();
        assert!(
            threats
                .iter()
                .any(|&(_, _, level, player)| level >= 3 && player == 0),
            "expected a P1 (id 0) warning/forced threat, got {threats:?}"
        );
        for &(q, r, _, _) in &threats {
            assert_eq!(b.get(q, r), 0, "threat cell ({q},{r}) must be empty");
        }
    }
}
