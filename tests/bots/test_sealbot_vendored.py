# >300 justify (R8): the four Tier-2 oracles are ONE claim — the REAL vendored engine is the bar
# the rungs say it is — over one precondition ladder. A split forks that ladder into copies, and
# the point of the file is that its skip is a RESULT with one reason vocabulary, not four.
"""Tier-2 SealBot oracles: `not_run` in CI, by ruling rather than by gap.

Resolution is CI-verifiable; LIVENESS is a property of a built C++/pybind11 extension
executing, which has no runnable producer in CI. Every row reports `not_run` — a RESULT, never
coverage — and names the box measurement that will produce a real one: O-A13 (Tier 2a, box M-1)
that mantis and the vendored engine agree on the GAME; O-A11 (2b, M-2) that `last_depth ==
depth`; O-A12 (2b, M-3) that the bar is reproducible; O-A14 (2b, M-4) that the compound-turn
buffer holds over a real game. O-A13 sits behind the CHEAPEST precondition because disagreement
about the game means every SealBot number measures a different one.

The box rider selects these with `pytest -k`, making the substrings `rules_agreement`,
`depth_receipt` and `determinism` a CONTRACT on these function names: a `-k` matching nothing
exits 5, neither pass nor fail.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import Board

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_VENDOR = _REPO / "vendor" / "external" / "sealbot"
_GAME_PY = _VENDOR / "game.py"
_CURRENT = _VENDOR / "current"

_ENC = "gnn_axis_v1"
_BOOK = "book_v1_s20260625_p4"

#: DESIGN_A §5: >=200 positions for the rules differential, >=20 for the depth receipt.
_RULES_POSITIONS = 200
_DEPTH_POSITIONS = 20

#: The cross-process determinism probe, a module constant so the command the box pastes into its
#: log is readable. It reaches the adapter through the SAME public entry points production uses.
_CROSS_PROCESS_PROBE = (
    "from mantis._engine import Board;"
    "from mantis.arena.books import paired_openings;"
    "from mantis.bots.sealbot import SealBotAdapter, load_sealbot_modules;"
    "mm, gm = load_sealbot_modules();"
    "b = Board.with_encoding_name('gnn_axis_v1');"
    "[b.apply_move(*m) for m in paired_openings("
    "'book_v1_s20260625_p4', n_pairs=1, seed=20260625)[0].moves];"
    "a = SealBotAdapter(depth=5, minimax_module=mm, game_module=gm);"
    "a.new_game();"
    "print(a.select_move(b))"
)


def _require_vendored_game() -> Any:
    """Tier 2a. LOUD skip with the box counterpart named — never a silent one."""
    if not _GAME_PY.is_file():
        pytest.skip(
            "not_run (Tier 2a) — the vendored `game.py` is absent; run `make vendor`. "
            "Box counterpart: M-1 (rules agreement), DESIGN_A §3.5.4."
        )
    from mantis.bots.sealbot import install_game_module

    return install_game_module(_GAME_PY)


def _require_built_extension() -> Any:
    """Tier 2b. LOUD skip naming the exact build command, with its OWN preconditions rather than
    a call through `_require_vendored_game`, which would name the wrong box counterpart."""
    tier_2b_boxes = "Box counterparts: M-2 (depth receipt), M-3 (determinism), M-4 (liveness)."
    if not _GAME_PY.is_file():
        pytest.skip(f"not_run (Tier 2b) — nothing vendored; run `make vendor`. {tier_2b_boxes}")
    if not sorted(_CURRENT.glob("minimax_cpp*.so")):
        pytest.skip(
            "not_run (Tier 2b) — the vendored extension is not built; run "
            "`bash tools/vendor_build_sealbot.sh` from the repo root, which verifies the "
            "pinned sha and the applied patch and then runs `python setup.py build_ext "
            f"--inplace` inside vendor/external/sealbot/current/. {tier_2b_boxes}"
        )
    from mantis.bots.sealbot import load_sealbot_modules

    return load_sealbot_modules()


def _book_positions(n: int) -> list[Board]:
    """Return `n` distinct positions replayed from the sha-pinned book, never a synthetic corpus."""
    from mantis.arena.books import paired_openings

    openings = paired_openings(_BOOK, n_pairs=n, seed=20260625)
    assert len(openings) >= n, (
        f"the book yielded {len(openings)} openings for a sample of {n}; a differential over "
        f"fewer positions than the design registers is a weaker row reported as the stronger one"
    )
    boards: list[Board] = []
    for opening in openings[:n]:
        board = Board.with_encoding_name(_ENC)
        for move in opening.moves:
            board.apply_move(*move)
        boards.append(board)
    return boards


#: The three hex axes as `game.py` declares them — used ONLY to walk a position forward, never
#: to decide a winner, which would be the vacuity this grant exists to remove.
_HEX_DIRECTIONS = ((1, 0), (0, 1), (1, -1))

#: Plies to drive a book opening forward before giving up. MEASURED, not chosen: at 40 all 200
#: openings are already decided and 60 and 80 return the identical set.
_DRIVE_PLY_CAP = 40


def _line_extending_move(board: Board, mover: int) -> tuple[int, int]:
    """A legal cell adjacent to one of `mover`'s own stones, deterministically chosen."""
    legal = set(board.legal_moves())
    for q, r in sorted((q, r) for q, r, p in board.get_stones() if int(p) == mover):
        for dq, dr in _HEX_DIRECTIONS:
            for sign in (1, -1):
                candidate = (q + dq * sign, r + dr * sign)
                if candidate in legal:
                    return candidate
    return sorted(legal)[0]


def _decided_positions(n: int) -> list[Board]:
    """Return `n` positions that actually have a WINNER, driven forward from the pinned book.

    The book's openings are all 4-ply while six-in-a-row needs six stones, so `_book_positions`
    can contain NO decided position — 0 of 200 — and the winner-identity row's `checked > 0`
    guard fired rather than certifying anything. The drive uses only mantis's engine: take the
    winning move when one is offered, else extend the mover's line. Measured: 200 of 200 openings
    decide within `_DRIVE_PLY_CAP` and both identities occur (189 for `+1`, 11 for `-1`).
    """
    decided: list[Board] = []
    for board in _book_positions(n):
        for _ply in range(_DRIVE_PLY_CAP):
            if board.winner() is not None:
                break
            mover = int(board.current_player)
            winning = board.first_winning_move(mover)
            board.apply_move(*(winning if winning is not None else _line_extending_move(board, mover)))
        if board.winner() is not None:
            decided.append(board)
    assert decided, (
        f"driving {n} book openings {_DRIVE_PLY_CAP} plies produced NO decided position; the "
        f"sample this row asserts over is empty and the row would certify nothing (R81/R86)"
    )
    return decided


def _shadow_game(game_module: Any, board: Board) -> Any:
    """Rebuild the vendored `HexGame` from a mantis `Board` the way the adapter does."""
    from mantis.bots.sealbot import build_shadow_game

    return build_shadow_game(board, game_module=game_module)


def test_rules_agreement_win_and_legality_differential() -> None:
    """O-A13 arm 1: the axial basis and the three win directions must coincide, measured over the
    book plus 200 driven-to-decided positions — the WIN half was vacuous over the book alone."""
    game_module = _require_vendored_game()
    disagreements: list[str] = []
    for board in [*_book_positions(_RULES_POSITIONS), *_decided_positions(_RULES_POSITIONS)]:
        shadow = _shadow_game(game_module, board)
        mantis_win = bool(board.check_win())
        vendored_win = shadow.winner is not game_module.Player.NONE
        if mantis_win != vendored_win:
            disagreements.append(f"win {board.get_stones()}: mantis={mantis_win} vendored={vendored_win}")
        mantis_legal = set(board.legal_moves())
        occupied = set(shadow.board.keys())
        if mantis_legal & occupied:
            disagreements.append(f"legality {sorted(mantis_legal & occupied)}")
    assert disagreements == [], (
        "mantis and the vendored engine disagree about the GAME; every SealBot number would "
        "be measuring a different one. STOP — do not proceed to M-2..M-4:\n"
        + "\n".join(disagreements[:20])
    )


def test_rules_agreement_turn_structure_parity_over_a_replayed_opening() -> None:
    """O-A13 arm 2: `moves_remaining` and `moves_left_in_turn` must agree at EVERY ply, because
    a one-ply offset would make the adapter play the wrong side's half."""
    game_module = _require_vendored_game()
    board = _book_positions(1)[0]
    offsets: list[str] = []
    for ply in range(20):
        shadow = _shadow_game(game_module, board)
        if int(board.moves_remaining) != int(shadow.moves_left_in_turn):
            offsets.append(
                f"ply {ply}: mantis={board.moves_remaining} vendored={shadow.moves_left_in_turn}"
            )
        legal = board.legal_moves()
        if not legal or board.winner() is not None:
            break
        board.apply_move(*legal[0])
    assert offsets == [], "turn-structure parity broke:\n" + "\n".join(offsets)


def test_rules_agreement_winner_identity_under_the_declared_player_map() -> None:
    """O-A13 arm 3: agreeing that SOMEONE won is not agreement — the `+1 -> Player.A` map must
    make both implementations name the SAME winner, or every rung's win rate is the opponent's.
    The `checked > 0` guard is retained, because it is what made the earlier vacuity visible."""
    game_module = _require_vendored_game()
    checked = 0
    mismatches: list[str] = []
    for board in _decided_positions(_RULES_POSITIONS):
        winner = board.winner()
        if winner is None:
            continue
        checked += 1
        shadow = _shadow_game(game_module, board)
        expected = game_module.Player.A if winner == 1 else game_module.Player.B
        if shadow.winner is not expected:
            mismatches.append(f"{board.get_stones()}: mantis={winner} vendored={shadow.winner}")
    assert checked > 0, (
        "no decided position appeared in the sample, so this row asserted nothing; widen the "
        "sample rather than recording it green (R81/R86)"
    )
    assert mismatches == [], "winner identity disagrees:\n" + "\n".join(mismatches[:20])


@pytest.mark.parametrize("depth", [5, 6])
def test_depth_receipt_holds_on_every_move_at_the_configured_depth(depth: int) -> None:
    """O-A11 -> box M-2, parametrized over run5's two minted sealbot depths: a receipt that holds
    at 5 and truncates at 6 is what a single-depth row would report as a pass."""
    minimax_module, game_module = _require_built_extension()
    from mantis.bots.sealbot import SealBotAdapter

    adapter = SealBotAdapter(
        depth=depth, minimax_module=minimax_module, game_module=game_module
    )
    adapter.new_game()
    reached: list[int] = []
    for board in _book_positions(_DEPTH_POSITIONS):
        move = adapter.select_move(board)
        assert move in board.legal_moves(), f"the engine returned an illegal move {move}"
        reached.append(int(adapter.last_reached_depth))

    truncated = [d for d in reached if d != depth]
    assert truncated == [], (
        f"depth receipt below 100% at depth {depth}: reached {truncated}. The bar is not the "
        f"bar it claims (F-20). Record the failing positions; do NOT raise the time limit "
        f"and do NOT lower the depth to make this pass (PREREG_A §8 abort 5)."
    )


class _TimeCutRestored:
    """The REAL vendored engine with the wall-clock cut PUT BACK, for one row only.

    The row asks for `depth=99` against an adapter whose required contract sets an unreachable
    time limit; `search.h` then loops to depth 99 with its only other exits the `TimeUp` catch
    that contract removes and a proven-win break that cannot fire on a four-ply opening — so the
    sealed call had NO terminating path and would have HUNG at the box, not raised.

    Nothing is stubbed: `get_move` is the real C++ search, `last_depth` the real receipt, and
    exactly one write (`time_limit`) is intercepted through the seam the Tier-1 rows use. The
    raise is derived, not measured — 0.05 s cannot complete 99 plies and a four-ply opening holds
    no immediate win — and if that is wrong the row reports DID NOT RAISE.
    """

    def __init__(self, real_module: Any, seconds: float) -> None:
        object.__setattr__(self, "_real_module", real_module)
        object.__setattr__(self, "_seconds", seconds)

    def MinimaxBot(self) -> Any:  # noqa: N802 — vendored name
        return _TimeCutBot(self._real_module.MinimaxBot(), self._seconds)

    def __getattr__(self, name: str) -> Any:
        """Everything else is the REAL module's, `WIN_THRESHOLD` above all: hiding the
        patch-exported mate-distance constant makes the adapter refuse to construct."""
        return getattr(object.__getattribute__(self, "_real_module"), name)


class _TimeCutBot:
    """Delegates everything to the real engine; forces `time_limit` back to a binding value."""

    def __init__(self, engine: Any, seconds: float) -> None:
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_seconds", seconds)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._engine, name, self._seconds if name == "time_limit" else value)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_engine"), name)

    def get_move(self, game: Any) -> Any:
        return self._engine.get_move(game)


#: A depth the engine cannot reach inside the bounded search below.
_UNREACHABLE_DEPTH = 99

#: The vendored default — the cut the adapter neutralises in production and this row restores,
#: so the truncation the receipt guards against actually happens.
_BINDING_TIME_LIMIT_SEC = 0.05


def test_depth_receipt_raises_rather_than_reporting_a_shallower_bar() -> None:
    """O-A11's second half: a receipt that is READ but not ACTED ON is an assurance. The engine
    reaches the adapter through `_TimeCutRestored`, which puts the wall-clock cut back."""
    minimax_module, game_module = _require_built_extension()
    from mantis.bots.sealbot import SealBotAdapter, SealBotDepthError

    adapter = SealBotAdapter(
        depth=_UNREACHABLE_DEPTH,
        minimax_module=_TimeCutRestored(minimax_module, _BINDING_TIME_LIMIT_SEC),
        game_module=game_module,
    )
    adapter.new_game()
    board = _book_positions(1)[0]

    with pytest.raises(SealBotDepthError) as exc:
        adapter.select_move(board)

    assert str(_UNREACHABLE_DEPTH) in str(exc.value), (
        f"a depth the engine cannot reach within the game must raise NAMING the configured "
        f"depth, so the round records the rung as broken rather than reporting a weaker "
        f"opponent under the rung's name: {exc.value}"
    )
    assert int(adapter.last_reached_depth) < _UNREACHABLE_DEPTH, (
        f"the receipt must have READ a truncated depth, not merely raised: "
        f"last_reached_depth={adapter.last_reached_depth}"
    )


def test_determinism_across_five_fresh_instances_in_one_process() -> None:
    """O-A12 -> box M-3, in-process half. The `_rng`-is-dead grep is not the evidence; this is."""
    minimax_module, game_module = _require_built_extension()
    from mantis.bots.sealbot import SealBotAdapter

    board = _book_positions(1)[0]
    moves: list[tuple[int, int]] = []
    for _ in range(5):
        adapter = SealBotAdapter(depth=5, minimax_module=minimax_module, game_module=game_module)
        adapter.new_game()
        moves.append(adapter.select_move(board))
    assert len(set(moves)) == 1, (
        f"five fresh instances at the same position and depth chose {moves}. A "
        f"non-deterministic fixed-depth bar is not reproducible, and F-20 is in the register "
        f"precisely so that is not re-derived at cost."
    )


def test_determinism_across_two_processes() -> None:
    """O-A12's cross-process half: in-process determinism can be an artefact of warm state inside
    one loaded extension, and only a second interpreter distinguishes deterministic from cached."""
    _require_built_extension()
    runs = [
        subprocess.run(
            [sys.executable, "-c", _CROSS_PROCESS_PROBE], cwd=str(_REPO),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert runs[0] != "", "the cross-process probe printed nothing; it asserted nothing"
    assert runs[0] == runs[1], (
        f"two interpreters chose different moves at the same position and depth: {runs}"
    )


def test_the_compound_turn_buffer_holds_over_a_real_game() -> None:
    """O-A14's real arm -> box M-4(f): the Tier-1 arms drive a SCRIPTED double, and only a real
    game exercises the assumption that no opponent stone lands between the halves of one turn."""
    minimax_module, game_module = _require_built_extension()
    from mantis.bots.random_bot import RandomBot
    from mantis.bots.sealbot import SealBotAdapter

    adapter = SealBotAdapter(depth=5, minimax_module=minimax_module, game_module=game_module)
    adapter.new_game()
    opponent = RandomBot(seed=20260802)
    board = Board.with_encoding_name(_ENC)
    seat = int(board.current_player)

    illegal: list[tuple[int, int]] = []
    for _ply in range(60):
        if board.winner() is not None or not board.legal_moves():
            break
        mover = adapter if int(board.current_player) == seat else opponent
        move = mover.select_move(board)
        if move not in board.legal_moves():
            illegal.append(move)
            break
        board.apply_move(*move)

    assert illegal == [], f"the adapter played an illegal move into a real game: {illegal}"


def test_a_book_opening_leaves_no_stale_half_buffered() -> None:
    """The `moves_remaining > 1` invariant, from a BOOK OPENING — the producer it did not have.

    O-A14's other real arms start from an empty board, where the wrapper short-circuits and
    returns ONE move, so the defect's condition never arises: measured, deleting the invariant
    leaves both tiers green. The precondition is asserted, not assumed — a four-ply opening
    leaves the first mover exactly ONE half due, measured on all four openings. The observation
    is a determinism differential deliberately: with the invariant deleted, 2 of 4 openings put
    the stale half on an occupied cell and 2 played it silently onto a legal one, and the silent
    half is the dangerous one.
    """
    minimax_module, game_module = _require_built_extension()
    from mantis.bots.random_bot import RandomBot
    from mantis.bots.sealbot import SealBotAdapter

    board = _book_positions(1)[0]
    seat = int(board.current_player)
    assert int(board.moves_remaining) == 1, (
        f"this row's whole subject is the ONE-half-due turn a four-ply opening leaves; the "
        f"book handed it {board.moves_remaining} halves, so the condition is unreachable and "
        f"the row would certify nothing (R81/R86)"
    )

    adapter = SealBotAdapter(depth=5, minimax_module=minimax_module, game_module=game_module)
    adapter.new_game()
    board.apply_move(*adapter.select_move(board))

    opponent = RandomBot(seed=20260802)
    for _ply in range(4):
        if int(board.current_player) == seat or board.winner() is not None:
            break
        board.apply_move(*opponent.select_move(board))
    assert int(board.current_player) == seat and board.winner() is None, (
        "the opponent did not hand the turn back; this row needs a SECOND adapter move"
    )

    reference = SealBotAdapter(
        depth=5, minimax_module=minimax_module, game_module=game_module
    )
    reference.new_game()
    expected = reference.select_move(board)

    assert adapter.select_move(board) == expected, (
        "the adapter's second move disagrees with a FRESH search of the same position, so it "
        "consumed a half buffered on a turn that had only one — it answered a position it "
        "never searched. `_search` must not fill the buffer when `moves_remaining == 1`."
    )
    assert adapter.illegal_buffer_discards == 0, (
        f"a stale half reached consumption and was discarded {adapter.illegal_buffer_discards} "
        f"times; the buffer's own invariant is the fix, not a wider discard net"
    )


def test_the_illegal_buffer_counter_reads_zero_over_a_real_game() -> None:
    """O-A14's counter half: a non-zero count means the invariant broke and the adapter re-searched."""
    minimax_module, game_module = _require_built_extension()
    from mantis.bots.random_bot import RandomBot
    from mantis.bots.sealbot import SealBotAdapter

    adapter = SealBotAdapter(depth=5, minimax_module=minimax_module, game_module=game_module)
    adapter.new_game()
    opponent = RandomBot(seed=20260802)
    board = Board.with_encoding_name(_ENC)
    seat = int(board.current_player)

    for _ply in range(60):
        if board.winner() is not None or not board.legal_moves():
            break
        mover = adapter if int(board.current_player) == seat else opponent
        board.apply_move(*mover.select_move(board))

    assert adapter.illegal_buffer_discards == 0, (
        f"the compound-turn buffer was discarded {adapter.illegal_buffer_discards} times in "
        f"one game; the invariant DESIGN_A §2.5.3 rests on does not hold"
    )
