// Exceeds the 300-line soft cap (R8): the PyBoard pymethod surface ports as one auditable unit
// with its tests; splitting a pyclass's methods across files scatters the Python face.
//! Python-visible Board wrapper over `mantis_core::Board`, which carries plain geometry and NO
//! encoding ref — so this wrapper HOLDS the encoding binding and `with_encoding_name` sets it.
//! `Board` is `Send + !Sync`; single-thread Python ownership is the bridge's synchronization.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_core::board::{BOARD_SIZE, HALF};
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
}
