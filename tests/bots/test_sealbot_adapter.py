# >300 justify (R8).
# O-A7/O-A10/O-A14 are ONE claim — the adapter translates honestly and REFUSES rather than
# degrading — over one seam (`bots/sealbot.py`). They share the `_FakeMinimax` recording
# double (SR-1/SR-2) and the two board constructions; a split forks those into copies that
# drift while both stay green. Executable content is a minority; the rest is the per-row
# "what defect is this the only witness to" rationale LAW-07 asks each row to carry, and
# the reachability note R166 asks each drive to state.
"""⊕ WP12-R Phase A / O-A7, O-A10, O-A14(fake) (DESIGN_A §2.4/§2.5, PREREG_A §1).

The adapter is this phase's only new production module and it carries three properties no
existing test in the tree can see, because none of them has an external opponent to
translate to.

The defect each row is the ONLY witness to:

- **O-A10 (a)-(d)** — F-20, live. `search.h:157` makes the configured depth a CEILING,
  reached only if `time_limit` allows, and `search.h:179-197` unwinds a time-out by
  RETURNING the last completed depth's move. So a bar named `sealbot_d5` can silently play
  at depth 3 and still report a number. The adapter must (a) DRIVE `max_depth`, (b)
  neutralise the time cut and (c) CHECK the `last_depth` receipt rather than assure it.
  Arm (d) exists because (c) alone cannot distinguish "checks correctly" from "always
  raises" — an adapter that raised on every move satisfies (c) and destroys every rung.
- **O-A10 (payload)** — the `+1 -> Player.A` mapping. `minimax_bot.cpp:25` decides player
  identity with an `is` test against the vendored `game.Player` objects, so a wrong mapping
  does not error: it silently plays the other colour and every SealBot number is then the
  opponent's. Driven on a board where it is player -1's turn, because on a FRESH board
  `current_player` is +1 and an inverted map is unobservable there.
- **O-A14 (a)-(d)** — the compound-turn buffer. `get_move` returns 1 OR 2 moves against a
  protocol that is one half-ply per call. Arms (a)-(c) hold whether or not the legality
  re-check exists, which is exactly why arm (d) — a forced-illegal buffer — is the only row
  that can see M-A8. LAW-18/R164: the discard is COUNTED in-run, never a silent `if`.
- **O-A7 (a),(b)** — LAW-17. The vendored extension's C++ does `py::module_::import("game")`
  (`minimax_bot.cpp:16`), so the loader must install a top-level module NAME without ever
  writing `sys.path`. Arm (b) pins the half a reviewer would not think to ask for: a
  pre-existing foreign `game` is REFUSED, not shadowed, and is left in place while refusing.

SEAM (frozen here, ORACLE-FIRST — IMPL builds to it or files a grant):
  * `SealBotAdapter(*, depth: int, minimax_module: Any, game_module: Any)` — SR-1's
    constructor-parameter form, so a Tier-1 row hands it a double with no vendor tree.
    Satisfies `BotProtocol` (`name()`, `new_game()`, `select_move(board)`).
  * `SealBotDepthError` — SR-5's named, importable receipt-violation type.
  * `SealBotModuleCollisionError` — the `sys.modules["game"]` refusal type.
  * `adapter.illegal_buffer_discards: int` — LAW-18's in-run counter, per adapter.
  * `mantis.bots.sealbot.install_game_module(path)` — the refuse-don't-overwrite installer.
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

#: DESIGN_A §2.4: `int64(time_limit * 1e6)` at 1e6 s is 1e12 microseconds, four orders below
#: int64 max — provably non-overflowing AND provably unreachable within a game. Asserted as
#: the LITERAL, never through a symbol the adapter could redefine underneath the oracle.
_NON_BINDING_TIME_LIMIT = 1e6

_ENC = "gnn_axis_v1"

#: Absence sentinel for the double's constructor — the OPPOSITE of a default: it is what
#: lets the double record "the adapter passed nothing" distinguishably from "the adapter
#: passed the vendored default 0.05" (`minimax_bot.cpp:82`).
_UNSET = object()


class _FakePlayer:
    """Stand-in for the vendored `game.Player`. IDENTITY is the whole contract
    (`minimax_bot.cpp:25` is an `is` test), so these are two distinct sentinels."""

    A = object()
    B = object()


class _FakeGameModule:
    Player = _FakePlayer


class _FakeMinimax:
    """SR-2's recording double, shaped like `minimax_cpp.MinimaxBot`.

    `writes` records ONLY what the ADAPTER set — never a constructor default — which is
    what makes M-A5 (drop the `time_limit` assignment) observable as a MISSING KEY rather
    than as a wrong value.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "writes", {})
        object.__setattr__(self, "payloads", [])
        object.__setattr__(self, "last_depth", 0)
        object.__setattr__(self, "_script", [])

    def __setattr__(self, name: str, value: Any) -> None:
        self.writes[name] = value
        object.__setattr__(self, name, value)

    def program(self, moves: list[tuple[int, int]], *, last_depth: int) -> None:
        """Queue ONE `get_move` response. Goes through `object.__setattr__` so programming
        the double never pollutes the `writes` record the oracle reads."""
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
    """A duck-typed board, permitted explicitly by `protocol.py:16-18`.

    Used by ONE row (O-A14 arm (d)) and only there: the illegal-buffer condition is
    unreachable on a real `Board`, because occupying the buffered coordinate necessarily
    consumes the mover's remaining half and flips the turn. Constructing it needs a board
    whose legal set can be edited under the adapter's feet — which is the whole point of
    the row: the adapter must not trust its own buffer.
    """

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
    """A board where the mover has TWO halves left — the only state in which a 2-move
    `get_move` return is well-formed. Measured at ORACLE-WRITE: a fresh board is `+1` with
    `moves_remaining == 1`; after one move it is `-1` with `moves_remaining == 2`."""
    board = _fresh_board()
    board.apply_move(*board.legal_moves()[0])
    assert int(board.current_player) == -1 and board.moves_remaining == 2, (
        f"turn structure moved: current_player={board.current_player}, "
        f"moves_remaining={board.moves_remaining}"
    )
    return board


def _board_with_a_winning_move() -> tuple[Board, tuple[int, int]]:
    """A position where the mover (+1) has an immediate win. Built by driving the real
    engine rather than by transcribing coordinates, so a geometry change fails loudly here
    instead of silently degrading the row it feeds."""
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


# ── O-A10 ───────────────────────────────────────────────────────────────────────────────
def test_adapter_drives_max_depth_and_builds_the_declared_player_payload() -> None:
    """O-A10 arm (a) + the payload arm (M-A9's observer).

    FIRING ORDER, stated because two mutations target this one function: the `max_depth`
    assertion runs FIRST and the payload assertion SECOND. Under M-A9 (inverted mapping)
    `max_depth` still holds, so the first failure is the payload assertion and the kill is
    attributable. Under a mutation that dropped the `max_depth` write, the first failure is
    the membership assertion and the payload line is `[unreached]`.
    """
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
    """O-A10 arm (b). MEMBERSHIP first, then VALUE (PREREG_A C-14): under M-A5 the adapter
    performs no write at all, and a bare `writes["time_limit"] == 1e6` would kill by
    `KeyError` — an ERROR-mode kill for a row whose stated mechanism is an assertion."""
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
    """O-A10 arm (c). The `raises` block contains ONLY the call (SR-6): an assertion inside
    it would be `[unreached]` under M-A6, leaving the message content with no killing
    mutation at all."""
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
    """O-A10 arm (d). `search.h:178` breaks early on a proven win, so `last_depth < depth`
    is CORRECT there: a solved position is not a truncated search. Without this arm, (c) is
    satisfied by an adapter that raises unconditionally and every rung dies."""
    adapter, instance = _adapter(depth=5)
    board, winning = _board_with_a_winning_move()
    instance.program([winning], last_depth=3)

    move = adapter.select_move(board)
    assert move == winning


# ── O-A14, the Tier-1 (fake-module) arms ────────────────────────────────────────────────
def test_compound_turn_buffer_consumes_the_second_move_without_re_searching() -> None:
    """O-A14 arms (a)+(b). A 2-move `get_move` return is one compound TURN against a
    protocol that is one half-ply per call: the second half must come from the buffer, be
    LEGAL at consumption, and cost no second search."""
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
    """O-A14 arm (c). SealBot returns ONE move when the first half already wins
    (`minimax_bot.cpp:44-60`); the buffer-empty rule must cover it with no branch."""
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
    """O-A14 arm (d) — the ONLY row that can see M-A8. See `_StubBoard`'s docstring for why
    the condition needs a duck-typed board: on a real `Board` occupying the buffered
    coordinate necessarily consumes the mover's remaining half and flips the turn, which
    would route the adapter down the turn-changed branch instead of the illegal branch."""
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


# ── O-A7 ────────────────────────────────────────────────────────────────────────────────
def _sys_path_writes(tree: ast.AST) -> list[int]:
    """Lines that MUTATE `sys.path`: a call to insert/append/extend on it, or an assignment
    whose target is it."""
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
    """O-A7 arm (a) / LAW-17 / R5, re-run with the new module present. An AST walk, not a
    grep: this repo carries prose about `sys.path` in `bots/`, and a text scan would
    false-positive on it. Detector self-tested inline so the row cannot pass by walking
    nothing (R81/R86)."""
    probe = ast.parse("import sys\nsys.path.insert(0, 'x')\nsys.path = []\n")
    assert _sys_path_writes(probe) == [2, 3], "the detector itself must fire"

    offenders: list[str] = []
    for root in _SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            offenders += [f"{path.relative_to(_REPO)}:{n}" for n in _sys_path_writes(tree)]
    assert offenders == [], f"LAW-17 is not negotiable; sys.path is written at {offenders}"


def test_a_foreign_game_module_is_refused_and_left_intact(tmp_path: Path) -> None:
    """O-A7 arm (b). `sys.modules["game"]` is a squat on a very common name; the adapter
    REFUSES rather than overwrites.

    TWO mutations, TWO failure points, and the split is load-bearing: M-A17 (drop the raise)
    fails at the `pytest.raises` block exit and the identity assertion below is `[unreached]`
    — SR-6 does NOT rescue a post-condition from a mutation that kills the raise. M-A17b
    (raise AFTER overwriting) passes the block and fails the identity assertion. Neither
    observes the other's subject, which is why both rows exist.
    """
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
