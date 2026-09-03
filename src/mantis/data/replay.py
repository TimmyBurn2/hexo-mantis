"""Corpus replay â training triples: per-encoding replayers + registry dispatch.

Routes ``(moves, winner)`` replay to the per-encoding implementation by
``encoding_spec.name`` and returns a frozen :class:`ReplayTriples` whose optional
fields capture the per-encoding shape asymmetry:

  - ``v6``          â :func:`replay_game_to_triples_v6`
                      â ``(states[T,18,19,19], chain_planes, policies, outcomes)``
  - ``v6w25``       â :func:`replay_game_to_triples_v6w25`
                      â ``(states[T,8,25,25], chain_planes, policies, outcomes)``
  - ``v6_live2_ls`` â :func:`replay_game_to_triples_ls` (reachable legal-set path)
                      â ``(states[R,4,19,19], policies, outcomes, ply_index)`` â NO
                        chain_planes (the LS per-cluster-row path).

The v8 dataset and the ``v8``/``v8_canvas_realness`` dispatch keys are SEVERED
(v8 never crosses; registry has no v8).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mantis._engine import Board
from mantis.data._log import get_logger
from mantis.data.loss_counters import REPLAY_COUNTERS
from mantis.data.replay_v6w25 import replay_game_to_triples_v6w25
from mantis.encoding import EncodingSpec
from mantis.env.game_state import (
    BOARD_SIZE,
    N_CHAIN_PLANES,
    GameState,
    _compute_chain_planes,  # pyright: ignore[reportPrivateUsage]  # ported chain-plane kernel
)
from mantis.monitor.best_effort import best_effort
from mantis.util.constants import HISTORY_LEN

log = get_logger(__name__)

_POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1  # 362


@dataclass(frozen=True)
class ReplayTriples:
    """Unified return shape for per-encoding replay.

    Fields:
        states:       float16 array (T, n_planes, board, board).
        policies:     float32 array (T, n_actions) â one-hot on move played.
        outcomes:     float32 array (T,) â Â±1 from current player's POV.
        chain_planes: float16 (T, 6, board, board) â v6/v6w25 only; None on the
                      v6_live2_ls legal-set path.
        ply_index:    int32 (T,) â original ply index per row; v6_live2_ls only
                      (its per-cluster-row path has no positional row/ply identity).
    """

    states: np.ndarray
    policies: np.ndarray
    outcomes: np.ndarray
    chain_planes: np.ndarray | None = None
    ply_index: np.ndarray | None = None


def replay_game_to_triples_v6(
    moves: list[tuple[int, int]],
    winner: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay a move sequence and return v6 pre-allocated training arrays.

    Args:
        moves:  Ordered (q, r) sequence for the complete game.
        winner: +1 if player 1 won, -1 if player 2 won.

    Returns:
        states:       float16 array of shape (T, 18, 19, 19) â full tensor (the
                      slice to 8 kept planes happens at dataset-load time).
        chain_planes: float16 array of shape (T, 6, 19, 19)
        policies:     float32 array of shape (T, 362)  â one-hot on move played
        outcomes:     float32 array of shape (T,)       â Â±1 from current player's POV
    """
    max_len = len(moves)
    states       = np.zeros((max_len, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    chain_planes = np.zeros((max_len, N_CHAIN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    policies     = np.zeros((max_len, _POLICY_SIZE),               dtype=np.float32)
    outcomes     = np.zeros(max_len,                               dtype=np.float32)
    t = 0

    board = Board()
    state = GameState.from_board(board)

    for q, r in moves:
        tensor, centers = state.to_tensor()  # (K, 18, 19, 19) float16
        target_k = target_idx = -1
        for k, (cq, cr) in enumerate(centers):
            wq = q - cq + 9
            wr = r - cr + 9
            if 0 <= wq < BOARD_SIZE and 0 <= wr < BOARD_SIZE:
                target_k, target_idx = k, wq * BOARD_SIZE + wr
                break

        if target_k >= 0:
            states[t]               = tensor[target_k]   # direct write, no .copy()
            # Chain planes: from most-recent stone planes (plane 0 = cur, plane 8 = opp).
            chain_planes[t]         = _compute_chain_planes(
                tensor[target_k, 0].astype(np.float32),
                tensor[target_k, HISTORY_LEN].astype(np.float32),
            ).astype(np.float16) / 6.0
            policies[t, target_idx] = 1.0
            outcomes[t]             = 1.0 if state.current_player == winner else -1.0
            t += 1
        else:
            # Off-window DROP (documented in the module docstring): this ply has no
            # representable dense target and emits NO training row. Not an exception —
            # a silent supervision loss, which is why it is COUNTED here (LAW-18).
            REPLAY_COUNTERS.increment("data.replay.v6.off_window_ply_dropped")

        ok, next_state = best_effort(
            "data.replay.v6.illegal_move_truncated_game",
            # default-bound (ruff B023): the closure runs inside THIS iteration only.
            lambda s=state, mq=q, mr=r: s.apply_move(board, mq, mr),
            counters=REPLAY_COUNTERS,
        )
        if not ok or next_state is None:
            break  # engine raised on an illegal move; the rest of the game is lost
        state = next_state

    return states[:t], chain_planes[:t], policies[:t], outcomes[:t]


def replay_game_to_triples_ls(
    moves: list[tuple[int, int]],
    winner: int,
    *,
    kept_plane_indices: Sequence[int],
    policy_size: int,
    k_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legal-set (``v6_live2_ls``) per-cluster-row replay â NO off-window drop.

    Scatter the played move across ALL containing windows â one dense-
    ``policy_size`` one-hot row per cluster window that geometrically contains
    the move. Corpus-side mirror of the Rust ``legal_set_scatter_max``
    per-cluster-local-362 buffer rows.

    Semantic delta vs :func:`replay_game_to_triples_v6` (the v6 path): the v6
    path emits ONE row per ply â the FIRST containing window â and drops every
    other window's view of the move (off-window-DROP supervision). Here a move
    near/beyond one window's extent still gets probability mass in whichever
    window(s) contain it. A ply whose move lies outside ALL cluster windows is
    still skipped (no representable dense target â matches the v6 path).

    Args:
        moves:  Ordered (q, r) sequence for the complete game.
        winner: +1 if player 1 won, -1 if player 2 won.
        kept_plane_indices: registry ``kept_plane_indices`` â slice of the
            18-plane tensor to emit (v6_live2_ls = [0, 8, 16, 17]).
        policy_size: registry ``policy_logit_count`` (v6_live2_ls = 362).
        k_max: registry ``k_max`` â cap on cluster views considered per ply
            (v6_live2_ls = 8).

    Returns:
        states:    float16 (R, len(kept_plane_indices), S, S) â one row per
                   (ply, containing-window) pair.
        policies:  float32 (R, policy_size) â one-hot at the window-LOCAL cell.
        outcomes:  float32 (R,) â Â±1 from current player's POV.
        ply_index: int32   (R,) â ORIGINAL ply index of each row (rows for the
                   same ply are consecutive).
    """
    kept = list(kept_plane_indices)
    states_rows: list[np.ndarray] = []
    target_rows: list[int] = []
    outcome_rows: list[float] = []
    ply_rows: list[int] = []

    board = Board()
    state = GameState.from_board(board)

    for ply, (q, r) in enumerate(moves):
        tensor, centers = state.to_tensor()  # (K, 18, S, S) float16
        _, _, H, W = tensor.shape
        half = (H - 1) // 2
        outcome = 1.0 if state.current_player == winner else -1.0
        rows_before = len(states_rows)
        for k, (cq, cr) in enumerate(centers[:k_max]):
            wq = q - cq + half
            wr = r - cr + half
            if 0 <= wq < H and 0 <= wr < W:
                states_rows.append(tensor[k][kept])  # slice 18âlen(kept) planes
                target_rows.append(wq * W + wr)
                outcome_rows.append(outcome)
                ply_rows.append(ply)
        if len(states_rows) == rows_before:
            # A ply outside ALL cluster windows emits no row at all (docstring above) —
            # an un-excepted supervision DROP, counted for the same LAW-18 reason.
            REPLAY_COUNTERS.increment("data.replay.ls.off_window_ply_dropped")

        ok, next_state = best_effort(
            "data.replay.ls.illegal_move_truncated_game",
            # default-bound (ruff B023): the closure runs inside THIS iteration only.
            lambda s=state, mq=q, mr=r: s.apply_move(board, mq, mr),
            counters=REPLAY_COUNTERS,
        )
        if not ok or next_state is None:
            break  # engine raised on an illegal move; the rest of the game is lost
        state = next_state

    n_rows = len(states_rows)
    if n_rows == 0:
        s_dim = BOARD_SIZE
        return (
            np.zeros((0, len(kept), s_dim, s_dim), dtype=np.float16),
            np.zeros((0, policy_size), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    states = np.stack(states_rows).astype(np.float16, copy=False)
    policies = np.zeros((n_rows, policy_size), dtype=np.float32)
    policies[np.arange(n_rows), np.asarray(target_rows, dtype=np.int64)] = 1.0
    outcomes = np.asarray(outcome_rows, dtype=np.float32)
    ply_index = np.asarray(ply_rows, dtype=np.int32)
    return states, policies, outcomes, ply_index


# Registry-name-keyed dispatch table. v8 / v8_canvas_realness SEVERED (v8 never
# crosses; registry has no v8). v6_live2_ls routes to the legal-set replayer and
# forwards its spec-driven (kept_plane_indices, policy_size, k_max).
_SUPPORTED = ("v6", "v6w25", "v6_live2_ls")


def replay_game_to_triples(
    moves: list[tuple[int, int]],
    winner: int,
    encoding_spec: EncodingSpec,
) -> ReplayTriples:
    """Dispatch ``(moves, winner)`` replay to the per-encoding implementation.

    Routes by ``encoding_spec.name`` over the registered corpus encodings
    (``v6``, ``v6w25``, ``v6_live2_ls``). Raises ``ValueError`` for any name
    without a registered replayer (v8 keys are absent by construction).
    """
    name = encoding_spec.name
    if name == "v6":
        s, c, p, o = replay_game_to_triples_v6(moves, winner)
        return ReplayTriples(states=s, policies=p, outcomes=o, chain_planes=c)
    if name == "v6w25":
        s, c, p, o = replay_game_to_triples_v6w25(moves, winner)
        return ReplayTriples(states=s, policies=p, outcomes=o, chain_planes=c)
    if name == "v6_live2_ls":
        s, p, o, pi = replay_game_to_triples_ls(
            moves,
            winner,
            kept_plane_indices=encoding_spec.kept_plane_indices,
            policy_size=encoding_spec.policy_logit_count,
            k_max=encoding_spec.k_max,
        )
        return ReplayTriples(states=s, policies=p, outcomes=o, ply_index=pi)
    raise ValueError(
        f"replay_game_to_triples: no replayer for encoding {name!r} "
        f"(supported: {_SUPPORTED})"
    )
