"""GameState — immutable snapshot of a Hex Tac Toe board position."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from mantis._engine import Board
from mantis.util.constants import HISTORY_LEN


@dataclass(frozen=True)
class GameState:
    """Position identity plus its bounded move history.

    The dense `to_tensor()` encode, the 18-plane source layout and the K-cluster views it
    read went with the grid path (R346(f)); what remains is the identity a replay driver
    needs to step a `Board` and hash the positions it visits.
    """

    current_player: int
    moves_remaining: int
    zobrist_hash: int
    ply: int
    # deque(maxlen=HISTORY_LEN): most-recent state is at the right (index -1).
    move_history: deque[GameState] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LEN)
    )

    @staticmethod
    def from_board(
        rust_board: Board,
        history: deque[GameState] | None = None,
    ) -> GameState:
        """Snapshot the engine board.

        Args:
            rust_board: the live engine board to read.
            history: the prior states, or `None` for a fresh bounded deque.

        Returns:
            The snapshot.
        """
        if history is None:
            history = deque(maxlen=HISTORY_LEN)
        return GameState(
            current_player=rust_board.current_player,
            moves_remaining=rust_board.moves_remaining,
            zobrist_hash=rust_board.zobrist_hash(),
            ply=rust_board.ply,
            move_history=history,
        )

    def apply_move(self, rust_board: Board, q: int, r: int) -> GameState:
        """Play `(q, r)` on the board and snapshot the result.

        Args:
            rust_board: the live engine board, MUTATED in place.
            q: axial q of the cell to play.
            r: axial r of the cell to play.

        Returns:
            The snapshot after the move, carrying this state in its history.

        Raises:
            ValueError: the engine refused the move as illegal.
        """
        rust_board.apply_move(q, r)
        new_history: deque[GameState] = deque(self.move_history, maxlen=HISTORY_LEN)
        new_history.append(self)
        return GameState.from_board(rust_board, history=new_history)

    def __hash__(self) -> int:
        # zobrist_hash is u128; Python's hash() reduces large ints to Py_hash_t width.
        return hash(self.zobrist_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return self.zobrist_hash == other.zobrist_hash and self.ply == other.ply
