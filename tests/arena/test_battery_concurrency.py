"""G games in flight must match the serial default byte-for-byte, in the same order.

Identity is provable on the CPU arm only: `index_add_` is nondeterministic on CUDA, so a CUDA
arm could assert aggregate equivalence at best, and no CUDA arm is run here.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from mantis._engine import Board
from mantis.arena.match import DEFAULT_MAX_PLIES, play_paired_match
from mantis.arena.regime import RegimeKey

_ENCODING = "gnn_axis_v1"


@dataclass(frozen=True)
class _Opening:
    opening_id: str
    moves: list


class _DeterministicBot:
    """Pick a legal move by a per-game counter; stateful, so sharing one interleaves games."""

    def __init__(self, stride: int) -> None:
        self._stride = stride
        self._i = 0
        self.games = 0

    def new_game(self) -> None:
        self._i = 0
        self.games += 1

    def select_move(self, board):
        legal = board.legal_moves()
        mv = legal[(self._i * self._stride) % len(legal)]
        self._i += 1
        return mv

    def name(self) -> str:
        return f"deterministic_{self._stride}"


def _board_factory():
    return Board.with_encoding_name(_ENCODING)


def _regime_key() -> RegimeKey:
    return RegimeKey(
        bot="candidate", variant="test", model_sims=1, opponent_spec="fixed",
        opening_book="test_book", deploy_matched=False, encoding=_ENCODING,
    )


def _openings(n: int = 6) -> list[_Opening]:
    """Build `n` distinct four-ply openings derived from the engine's own legal set."""
    openings: list[_Opening] = []
    for i in range(n):
        board = _board_factory()
        moves: list[tuple[int, int]] = []
        for ply in range(4):
            legal = sorted(board.legal_moves())
            move = legal[(i * 7 + ply * 3) % len(legal)]
            board.apply_move(*move)
            moves.append(move)
        openings.append(_Opening(opening_id=f"op{i}", moves=moves))
    return openings


def _pair():
    return (_DeterministicBot(3), _DeterministicBot(5))


def _play(concurrency: int, sink: list | None = None):
    cand, opp = _pair()
    return play_paired_match(
        cand, opp, _openings(),
        regime_key=_regime_key(), board_factory=_board_factory,
        max_plies=DEFAULT_MAX_PLIES, record_sink=None if sink is None else sink.append,
        player_factory=_pair, concurrency=concurrency,
    )


@pytest.mark.parametrize("concurrency", [2, 3, 4, 12])
def test_cpu_arm_trajectory_identity_and_stable_game_index(concurrency: int) -> None:
    """G-in-flight and serial produce byte-identical records, in the same order.

    Order is half the property: the eval ladder indexes into this list, so the game index is
    stable only if the concurrent arm reassembles in loop order.
    """
    serial = _play(1)
    parallel = _play(concurrency)
    assert [(r.opening_id, r.colors["candidate"]) for r in serial] == [
        (r.opening_id, r.colors["candidate"]) for r in parallel
    ], f"concurrency={concurrency}: the game index moved — records did not reassemble in loop order"
    assert serial == parallel, (
        f"concurrency={concurrency}: a record differs between the serial and G-in-flight arms. "
        "Every field is compared, `trajectory_hash` included — LAW-04's dedupe input."
    )


def test_law_04_dedupe_still_counts_distinct_games() -> None:
    """Concurrency must not manufacture duplicate games: the distinct-hash count is unmoved.

    The paired law plays each opening twice with colors swapped, and those are genuinely
    different trajectories, so distinct hashes must equal the game count on distinct openings.
    """
    serial = _play(1)
    parallel = _play(4)
    assert len({r.trajectory_hash for r in serial}) == len(
        {r.trajectory_hash for r in parallel}
    ), "the distinct-game count moved between arms"
    assert len({r.trajectory_hash for r in parallel}) == len(parallel), (
        "a trajectory hash repeated inside one G-in-flight match — either two games shared a "
        "player's state, or the dedupe input stopped discriminating"
    )


def test_record_sink_fires_in_loop_order_under_concurrency() -> None:
    """The sink sees the same sequence it would have seen serially."""
    serial_sink: list = []
    _play(1, serial_sink)
    parallel_sink: list = []
    _play(4, parallel_sink)
    assert serial_sink == parallel_sink


def test_each_worker_thread_gets_its_own_player_pair() -> None:
    """The factory is called once per worker thread, not once per game."""
    made: list[int] = []
    lock = threading.Lock()

    def factory():
        with lock:
            made.append(1)
        return _pair()

    cand, opp = _pair()
    concurrency = 3
    records = play_paired_match(
        cand, opp, _openings(), regime_key=_regime_key(),
        board_factory=_board_factory, max_plies=DEFAULT_MAX_PLIES,
        player_factory=factory, concurrency=concurrency,
    )
    assert len(records) == 12
    assert 1 <= len(made) <= concurrency, (
        f"the factory was called {len(made)} times for {len(records)} games at "
        f"concurrency={concurrency}; it must be called at most once per worker thread"
    )


def test_serial_default_ignores_the_factory_entirely() -> None:
    """`concurrency=1` runs on the objects the caller passed; the factory is never consulted.

    The pinned `new_game()` count proves the caller's own objects are the ones that played.
    """
    calls: list[int] = []
    cand, opp = _pair()
    records = play_paired_match(
        cand, opp, _openings(), regime_key=_regime_key(),
        board_factory=_board_factory, max_plies=DEFAULT_MAX_PLIES,
        player_factory=lambda: (calls.append(1), _pair())[1],
    )
    assert calls == [], "the serial arm consulted the factory"
    assert cand.games == len(records) == 12, "the caller's own player did not play every game"


def test_refusals_are_named() -> None:
    """A concurrency > 1 with no factory, and a concurrency < 1, both refuse LOUDLY."""
    cand, opp = _pair()
    kw = dict(regime_key=_regime_key(), board_factory=_board_factory,
              max_plies=DEFAULT_MAX_PLIES)
    with pytest.raises(ValueError, match="needs a `player_factory`"):
        play_paired_match(cand, opp, _openings(), concurrency=2, **kw)
    with pytest.raises(ValueError, match="must be >= 1"):
        play_paired_match(cand, opp, _openings(), concurrency=0, **kw)
