"""Corpus replay (v6w25 encoding): K-cluster windows at R=8 perception + 25Ã25
cluster windows + cluster_threshold=8.

Wire format identical to v6 (8 spec.kept_plane_indices planes + pass-slot policy)
but spatial extent is 25Ã25 instead of 19Ã19. Replays raw game moves on a Board
configured for v6w25 via ``Board.with_encoding_name("v6w25")`` and emits per-ply
(8, 25, 25) tensors aligned with the played move's cluster window â same
alignment policy as v6.
"""
from __future__ import annotations

import numpy as np

from mantis._engine import Board
from mantis.data._log import get_logger
from mantis.data.loss_counters import REPLAY_COUNTERS
from mantis.encoding import EncodingSpec
from mantis.encoding import lookup as _lookup_encoding
from mantis.env.game_state import (
    N_CHAIN_PLANES,
    GameState,
    _compute_chain_planes,  # pyright: ignore[reportPrivateUsage]  # ported chain-plane kernel
)
from mantis.monitor.best_effort import best_effort

log = get_logger(__name__)

# Registry spec for v6w25 â use spec fields instead of hardcoded consts.
_V6W25_SPEC: EncodingSpec = _lookup_encoding("v6w25")

BOARD_SIZE_V6W25: int = _V6W25_SPEC.board_size  # 25
HALF_V6W25: int = (BOARD_SIZE_V6W25 - 1) // 2  # 12
N_PLANES_V6W25: int = _V6W25_SPEC.n_planes  # 8
N_CELLS_V6W25: int = _V6W25_SPEC.n_cells  # 625
N_ACTIONS_V6W25: int = _V6W25_SPEC.n_actions  # 626 (cells + pass; v6 wire-format compat)
CLUSTER_THRESHOLD_V6W25: int | None = _V6W25_SPEC.cluster_threshold  # 8
LEGAL_MOVE_RADIUS_V6W25: int = _V6W25_SPEC.legal_move_radius  # 8


def _make_v6w25_board() -> Board:
    """Construct a fresh Board configured for v6w25 cluster encoding."""
    return Board.with_encoding_name("v6w25")


def replay_game_to_triples_v6w25(
    moves: list[tuple[int, int]],
    winner: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay a move sequence and return v6w25 training arrays.

    Args:
        moves:  Ordered (q, r) sequence for the complete game.
        winner: +1 if player 1 won, -1 if player 2 won.

    Returns:
        4-tuple ``(states, chain_planes, policies, outcomes)``:
        states:       float16 array of shape (T, 8, 25, 25) â spec.kept_plane_indices
                      slice of the full 18-plane tensor.
        chain_planes: float16 array of shape (T, 6, 25, 25) â Q13 chain planes.
        policies:     float32 array of shape (T, 626) â one-hot on move played.
        outcomes:     float32 array of shape (T,)     â Â±1 from current player's POV.
    """
    max_len = len(moves)
    states = np.zeros(
        (max_len, N_PLANES_V6W25, BOARD_SIZE_V6W25, BOARD_SIZE_V6W25),
        dtype=np.float16,
    )
    chain_planes = np.zeros(
        (max_len, N_CHAIN_PLANES, BOARD_SIZE_V6W25, BOARD_SIZE_V6W25), dtype=np.float16
    )
    policies = np.zeros((max_len, N_ACTIONS_V6W25), dtype=np.float32)
    outcomes = np.zeros(max_len, dtype=np.float32)
    t = 0

    board = _make_v6w25_board()
    state = GameState.from_board(board)

    for q, r in moves:
        full_tensor, centers = state.to_tensor()  # (K, 18, 25, 25) float16
        target_k = -1
        target_idx = -1
        for k, (cq, cr) in enumerate(centers):
            wq = q - cq + HALF_V6W25
            wr = r - cr + HALF_V6W25
            if 0 <= wq < BOARD_SIZE_V6W25 and 0 <= wr < BOARD_SIZE_V6W25:
                target_k = k
                target_idx = wq * BOARD_SIZE_V6W25 + wr
                break

        if target_k >= 0:
            # Slice 18 â 8 planes (spec.kept_plane_indices, v6 wire format).
            states[t] = full_tensor[target_k, list(_V6W25_SPEC.kept_plane_indices)]
            chain_planes[t] = (
                _compute_chain_planes(
                    full_tensor[target_k, 0].astype(np.float32),
                    full_tensor[target_k, 8].astype(np.float32),
                ).astype(np.float16)
                / 6.0
            )
            policies[t, target_idx] = 1.0
            outcomes[t] = 1.0 if state.current_player == winner else -1.0
            t += 1
        else:
            # Off-window DROP: this ply has no representable dense target and emits NO
            # training row. Not an exception — a silent supervision loss (LAW-18).
            REPLAY_COUNTERS.increment("data.replay.v6w25.off_window_ply_dropped")

        ok, next_state = best_effort(
            "data.replay.v6w25.illegal_move_truncated_game",
            # default-bound (ruff B023): the closure runs inside THIS iteration only.
            lambda s=state, mq=q, mr=r: s.apply_move(board, mq, mr),
            counters=REPLAY_COUNTERS,
        )
        if not ok or next_state is None:
            break  # engine raised on an illegal move; the rest of the game is lost
        state = next_state

    return states[:t], chain_planes[:t], policies[:t], outcomes[:t]
