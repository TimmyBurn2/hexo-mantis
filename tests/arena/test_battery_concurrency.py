"""⊕ R335(e) Leg 3 — `S-BATTERY-G`: G games in flight, UNARMED, serial default byte-for-byte.

WHAT LANDED AND WHAT DID NOT. `play_paired_match` gains `player_factory` and `concurrency`.
`concurrency=1` — the default, and what every shipped config runs, because no config key
touches this — is the loop that was there before. **No value is armed by this leg**: the
concurrency figure is an operator prereg row (R335(e)), so what exists here is the CAPABILITY
and its determinism proof, never a battery-rate claim.

WHY A RATE CLAIM WOULD BE WORTHLESS HERE ANYWAY. Tranche-2 §10.1 measured every one of eight
battery games running to the 128-ply cap, because an untrained candidate cannot beat `RandomBot`
inside it — **a battery rate is a statement about the CAP × the sim budget until a candidate can
end games**. The rate is measured at the mint's own battery under the minted value, not here.

THE RISK IS DETERMINISM, NOT SPEED (SCOUT §5 P3 risk 3). Two properties must survive: per-game
trajectory identity, and a STABLE GAME INDEX — records must reassemble in loop order however
the threads finish. Both are asserted below on the CPU arm, which is the only arm on which
identity is provable: `index_add_` is nondeterministic on CUDA, so the CUDA arm can assert
aggregate equivalence only, and that limitation is disclosed here rather than discovered later.
There is no CUDA in this workspace (torch is `2.11.0+cpu`), so the CUDA arm is NOT run and is
NOT reported as passing.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from mantis._engine import Board
from mantis.arena.match import DEFAULT_MAX_PLIES, play_paired_match
from mantis.arena.regime import RegimeKey

_ENCODING = "v6_live2_ls"


@dataclass(frozen=True)
class _Opening:
    opening_id: str
    moves: list


class _DeterministicBot:
    """Picks a legal move by a per-game counter — stateful, so sharing one across concurrent
    games would interleave two games' counters and is exactly what `player_factory` prevents."""

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
    return [
        _Opening(opening_id=f"op{i}", moves=[(i, 0), (i, 1), (i + 1, 0), (i + 1, 1)])
        for i in range(n)
    ]


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
    """WITNESS (i): G-in-flight and serial produce byte-identical records, IN THE SAME ORDER.

    Compared as whole records, not as a set: the ORDER is half the property. A consumer that
    indexes into this list — and the eval ladder does — sees a stable game index only if the
    concurrent arm reassembles in loop order.
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
    """WITNESS (iii): concurrency must not become a way to manufacture duplicate games.

    LAW-04 counts DISTINCT games by trajectory hash. The paired law plays each opening twice
    with colors swapped, and those two games are genuinely different trajectories; the count of
    distinct hashes must therefore be identical under both arms, and equal to the game count
    when the openings are distinct.
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
    """The factory is called once per WORKER THREAD, not once per game.

    Sharing one pair across threads would interleave two games on one search tree — the
    mechanism `player_factory` exists to prevent — and building a pair per game would pay the
    construction cost G times over.
    """
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
    """`concurrency=1` runs on the objects the CALLER passed — the factory is never consulted.

    This is what 'the default is today's serial loop byte-for-byte' means operationally: a
    caller that passes a factory but leaves concurrency alone gets the old behaviour on the old
    objects, and the pinned `new_game()` count proves the old objects are the ones that played.
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
