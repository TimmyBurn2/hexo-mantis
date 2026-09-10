# >300 justify (R8): the three properties are ONE claim — the adapter translates honestly and
# REFUSES rather than degrading — over one seam. They share the recording double and the two board
# constructions; a split forks those into copies that drift while both stay green.
"""The SealBot adapter: it translates honestly, and REFUSES rather than degrading.

THE FIXED-DEPTH RECEIPT. The vendored search makes the configured depth a CEILING and unwinds a
time-out by RETURNING the last completed depth's move, so a bar named `sealbot_d5` can silently
play at depth 3 and still report a number. A fourth arm exists because the receipt check alone
cannot distinguish "checks correctly" from "always raises".

THE PLAYER MAPPING. The engine decides identity with an `is` test, so a wrong mapping does not
error — it silently plays the other colour. Driven where it is player -1's turn, because on a
FRESH board an inverted map is unobservable.

THE COMPOUND-TURN BUFFER. `get_move` returns 1 OR 2 moves against a one-half-ply protocol; three
arms hold whether or not the legality re-check exists, which is why the forced-illegal-buffer arm
is the only one that can see it. The discard is COUNTED (LAW-18).

THE MODULE INSTALL. The vendored C++ imports the top-level name `game`, so the loader installs a
module NAME without writing `sys.path`, and a foreign `game` is REFUSED and left in place.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import Board

_REPO = Path(__file__).resolve().parents[2]
_SCAN_ROOTS = (_REPO / "src", _REPO / "tools")

#: `int64(time_limit * 1e6)` at 1e6 s is 1e12 microseconds, four orders below int64 max — provably
#: non-overflowing AND provably unreachable within a game. Asserted as the LITERAL, never through a
#: symbol the adapter could redefine underneath the oracle.
_NON_BINDING_TIME_LIMIT = 1e6

_ENC = "gnn_axis_v1"

#: Absence sentinel for the double's constructor — the OPPOSITE of a default: it is what lets the
#: double record "the adapter passed nothing" distinguishably from "the adapter passed the
#: vendored default".
_UNSET = object()


class _FakePlayer:
    """Stand-in for the vendored `game.Player`. IDENTITY is the whole contract — the engine uses an
    `is` test — so these are two distinct sentinels."""

    A = object()
    B = object()


class _FakeGameModule:
    Player = _FakePlayer


class _FakeMinimax:
    """A recording double shaped like the vendored `MinimaxBot`. `writes` records ONLY what the
    ADAPTER set, so a dropped `time_limit` assignment is a MISSING KEY, not a wrong value."""

    def __init__(self) -> None:
        object.__setattr__(self, "writes", {})
        object.__setattr__(self, "payloads", [])
        object.__setattr__(self, "last_depth", 0)
        object.__setattr__(self, "_script", [])

    def __setattr__(self, name: str, value: Any) -> None:
        self.writes[name] = value
        object.__setattr__(self, name, value)

    def program(self, moves: list[tuple[int, int]], *, last_depth: int) -> None:
        """Queue ONE `get_move` response, through `object.__setattr__` so programming the double
        never pollutes the `writes` record the oracle reads."""
        object.__setattr__(self, "_script", [*self._script, (list(moves), last_depth)])

    def get_move(self, game: Any) -> list[tuple[int, int]]:
        self.payloads.append(game)
        moves, depth = self._script.pop(0)
        object.__setattr__(self, "last_depth", depth)
        return list(moves)


class _FakeMinimaxModule:
    def __init__(self) -> None:
        self.instances: list[_FakeMinimax] = []

    def MinimaxBot(self, time_limit: Any = _UNSET) -> _FakeMinimax:  # noqa: N802 — vendored name
        bot = _FakeMinimax()
        if time_limit is not _UNSET:
            bot.writes["time_limit"] = time_limit
        self.instances.append(bot)
        return bot


class _StubBoard:
    """A duck-typed board, permitted explicitly by the bot protocol, used by ONE row: the
    illegal-buffer condition needs a board whose legal set can be edited under the adapter's feet,
    which is the point of the row — the adapter must not trust its own buffer."""

    def __init__(self, *, stones: list[tuple[int, int, int]], current_player: int,
                 moves_remaining: int, legal: list[tuple[int, int]]) -> None:
        self._stones = list(stones)
        self.current_player = current_player
        self.moves_remaining = moves_remaining
        self._legal = list(legal)

    def get_stones(self) -> list[tuple[int, int, int]]:
        return list(self._stones)

    def legal_moves(self) -> list[tuple[int, int]]:
        return list(self._legal)

    def occupy(self, coord: tuple[int, int]) -> None:
        self._stones.append((coord[0], coord[1], self.current_player))
        self._legal.remove(coord)


def _adapter(depth: int) -> tuple[Any, _FakeMinimax]:
    from mantis.bots.sealbot import SealBotAdapter

    module = _FakeMinimaxModule()
    adapter = SealBotAdapter(depth=depth, minimax_module=module, game_module=_FakeGameModule)
    adapter.new_game()
    assert len(module.instances) == 1, (
        f"the adapter must construct exactly one MinimaxBot per game; got "
        f"{len(module.instances)}"
    )
    return adapter, module.instances[0]


def _fresh_board() -> Board:
    return Board.with_encoding_name(_ENC)


def _board_mid_compound_turn() -> Board:
    """A board where the mover has TWO halves left — the only state in which a 2-move `get_move`
    return is well-formed. Measured: a fresh board is `+1` with one half; after one move it is
    `-1` with two."""
    board = _fresh_board()
    board.apply_move(*board.legal_moves()[0])
    assert int(board.current_player) == -1 and board.moves_remaining == 2, (
        f"turn structure moved: current_player={board.current_player}, "
        f"moves_remaining={board.moves_remaining}"
    )
    return board


def _board_with_a_winning_move() -> tuple[Board, tuple[int, int]]:
    """A position where the mover has an immediate win, built by driving the real engine so a
    geometry change fails loudly instead of silently degrading the row it feeds."""
    board = _fresh_board()
    line = [(q, 0) for q in range(8)]
    placed = 0
    for _ in range(40):
        legal = board.legal_moves()
        if int(board.current_player) == 1:
            win = board.first_winning_move(1)
            if win is not None:
                return board, win
            move = line[placed]
            placed += 1
            assert move in legal, f"the line construction left the legal window at {move}"
        else:
            off_line = [c for c in legal if c[1] >= 3]
            move = off_line[-1]
        board.apply_move(*move)
    raise AssertionError("could not construct a position with an immediate win for +1")


def test_adapter_drives_max_depth_and_builds_the_declared_player_payload() -> None:
    """The adapter drives `max_depth` and builds the declared player payload. FIRING ORDER matters:
    the `max_depth` assertion runs FIRST, so an inverted mapping fails on the payload and a dropped
    write fails on the membership, and each kill is attributable."""
    adapter, instance = _adapter(depth=5)
    board = _board_mid_compound_turn()  # it is player -1's turn here, by construction

    instance.program(board.legal_moves()[:1], last_depth=5)
    adapter.select_move(board)

    assert "max_depth" in instance.writes, (
        f"the adapter never wrote max_depth; unset, `bot.h:29` leaves it at 200 and the "
        f"rung is time-bounded only — the bar is then not the bar it claims. "
        f"writes={instance.writes}"
    )
    assert instance.writes["max_depth"] == 5
    payload = instance.payloads[0]
    assert payload.current_player is _FakePlayer.B, (
        "mantis player -1 must map to the vendored Player.B sentinel BY IDENTITY: "
        "`minimax_bot.cpp:25` is an `is` test, so a wrong mapping plays the other colour "
        "silently and every rung number is then the opponent's"
    )


def test_adapter_neutralises_the_time_cut() -> None:
    """The adapter neutralises the time cut. MEMBERSHIP first, then VALUE: an adapter that performs
    no write at all would kill a bare value comparison by `KeyError` — an ERROR-mode kill for a row
    whose stated mechanism is an assertion."""
    adapter, instance = _adapter(depth=5)
    board = _fresh_board()
    instance.program(board.legal_moves()[:1], last_depth=5)
    adapter.select_move(board)

    assert "time_limit" in instance.writes, (
        f"the adapter never set time_limit; SealBot keeps its 0.05 s default "
        f"(`minimax_bot.cpp:82`) and `search.h:41-42`'s deadline then truncates the search, "
        f"which is F-20's exact mechanism. writes={instance.writes}"
    )
    assert instance.writes["time_limit"] == _NON_BINDING_TIME_LIMIT


def test_depth_receipt_below_the_configured_depth_raises_a_named_error() -> None:
    """A depth receipt below the configured depth raises a named error. The `raises` block contains
    ONLY the call: an assertion inside it would be unreached under the mutation that drops the
    raise, leaving the message content with no killing mutation at all."""
    from mantis.bots.sealbot import SealBotDepthError

    adapter, instance = _adapter(depth=5)
    board = _fresh_board()
    instance.program(board.legal_moves()[:1], last_depth=4)

    with pytest.raises(SealBotDepthError) as exc:
        adapter.select_move(board)

    message = str(exc.value)
    assert "5" in message and "4" in message, (
        f"the receipt violation must name the CONFIGURED depth and the REACHED one — the "
        f"bar the rung claims versus the bar it played: {message}"
    )


def test_a_winning_move_below_the_configured_depth_does_not_raise() -> None:
    """The engine breaks early on a proven win, so a short receipt is CORRECT there: a solved
    position is not a truncated search. Without this arm the receipt check is satisfied by an
    adapter that raises unconditionally, and every rung dies."""
    adapter, instance = _adapter(depth=5)
    board, winning = _board_with_a_winning_move()
    instance.program([winning], last_depth=3)

    move = adapter.select_move(board)
    assert move == winning


def test_compound_turn_buffer_consumes_the_second_move_without_re_searching() -> None:
    """A 2-move `get_move` return is one compound TURN against a protocol that is one half-ply per
    call: the second half must come from the buffer, be LEGAL at consumption, and cost no second
    search."""
    adapter, instance = _adapter(depth=5)
    board = _board_mid_compound_turn()
    pair = board.legal_moves()[:2]
    instance.program(pair, last_depth=5)

    first = adapter.select_move(board)
    board.apply_move(*first)
    second = adapter.select_move(board)

    assert [first, second] == pair
    assert len(instance.payloads) == 1, (
        f"the buffered half re-entered the search: {len(instance.payloads)} get_move calls "
        f"for one compound turn"
    )
    assert second in board.legal_moves()
    assert adapter.illegal_buffer_discards == 0


def test_a_one_move_get_move_return_needs_no_special_case() -> None:
    """SealBot returns ONE move when the first half already wins; the buffer-empty rule must cover
    it with no branch."""
    adapter, instance = _adapter(depth=5)
    board = _board_mid_compound_turn()
    legal = board.legal_moves()
    instance.program([legal[0]], last_depth=5)
    instance.program([legal[1]], last_depth=5)

    first = adapter.select_move(board)
    board.apply_move(*first)
    second = adapter.select_move(board)

    assert [first, second] == [legal[0], legal[1]]
    assert len(instance.payloads) == 2, "a 1-move return must trigger a fresh search next call"
    assert adapter.illegal_buffer_discards == 0


def test_an_illegal_buffered_move_is_discarded_re_searched_and_counted() -> None:
    """The ONLY row that can see a missing legality re-check. On a real `Board`, occupying the
    buffered coordinate necessarily consumes the mover's remaining half and flips the turn, which
    routes the adapter down the turn-changed branch instead of the illegal one."""
    adapter, instance = _adapter(depth=5)
    legal = [(0, 0), (1, 0), (2, 0), (3, 0)]
    board = _StubBoard(stones=[], current_player=1, moves_remaining=2, legal=legal)
    first, buffered, replacement = legal[0], legal[1], legal[2]
    instance.program([first, buffered], last_depth=5)
    instance.program([replacement], last_depth=5)

    played = adapter.select_move(board)
    assert played == first
    board.occupy(played)
    board.occupy(buffered)  # the buffered coordinate is now OCCUPIED, same side still to move
    board.moves_remaining = 1

    second = adapter.select_move(board)
    assert second in board.legal_moves(), (
        f"the adapter played the occupied coordinate {second} into a scored game"
    )
    assert adapter.illegal_buffer_discards == 1, (
        "LAW-18/R164: the discard is an in-run counted event, never a silent `if` — a "
        "reader must be able to see it fire WHILE the run is going"
    )


def _sys_path_writes(tree: ast.AST) -> list[int]:
    """Lines that MUTATE `sys.path`: a call to insert/append/extend on it, or an assignment whose
    target is it."""
    def _is_sys_path(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        )

    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"insert", "append", "extend"} and _is_sys_path(node.func.value):
                lines.append(node.lineno)
        elif isinstance(node, ast.Assign) and any(_is_sys_path(t) for t in node.targets):
            lines.append(node.lineno)
        elif isinstance(node, ast.AugAssign) and _is_sys_path(node.target):
            lines.append(node.lineno)
    return sorted(lines)


def test_no_sys_path_write_anywhere_under_src_or_tools() -> None:
    """LAW-17/R5, by AST walk rather than grep: this repo carries prose about `sys.path`, and a
    text scan would false-positive on it. The detector is self-tested inline so the row cannot
    pass by walking nothing."""
    probe = ast.parse("import sys\nsys.path.insert(0, 'x')\nsys.path = []\n")
    assert _sys_path_writes(probe) == [2, 3], "the detector itself must fire"

    offenders: list[str] = []
    for root in _SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            offenders += [f"{path.relative_to(_REPO)}:{n}" for n in _sys_path_writes(tree)]
    assert offenders == [], f"LAW-17 is not negotiable; sys.path is written at {offenders}"


def test_a_foreign_game_module_is_refused_and_left_intact(tmp_path: Path) -> None:
    """The adapter REFUSES to overwrite a squat on `sys.modules["game"]`. TWO mutations, TWO
    failure points: dropping the raise fails at the `pytest.raises` block exit and leaves the
    identity assertion unreached, while raising AFTER overwriting fails the identity assertion."""
    from mantis.bots.sealbot import SealBotModuleCollisionError, install_game_module

    sentinel = object()
    game_py = tmp_path / "game.py"
    game_py.write_text("VENDORED = True\n")

    had_previous = "game" in sys.modules
    previous = sys.modules["game"] if had_previous else None
    sys.modules["game"] = sentinel  # type: ignore[assignment]
    try:
        with pytest.raises(SealBotModuleCollisionError) as exc:
            install_game_module(game_py)
        assert sys.modules["game"] is sentinel, (
            "the loader clobbered a caller's `game` module while refusing — the refusal must "
            "cost the caller nothing (DESIGN_A §2.5.2)"
        )
        assert "game" in str(exc.value)
    finally:
        if had_previous:
            sys.modules["game"] = previous  # type: ignore[assignment]
        else:
            del sys.modules["game"]
