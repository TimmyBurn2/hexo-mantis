# >300 justify (R8): ONE seam — the vendored engine's translation boundary — whose three
# properties (the depth receipt, stateless position reconstruction, the `sys.modules` install)
# share the loader, the player map and the shadow view; splitting them forks that shared state
# into copies that drift while both halves stay green.
"""SealBot adapter — the vendored fixed-depth external bar.

Pure Python; no pyo3. Nothing here names a host path or an endpoint: the ONE external string this
phase permits is the public vendor URL, in `vendor/pins.toml`.

A FIXED-DEPTH BAR THAT IS ACTUALLY FIXED. The vendored search is iterative deepening with the
configured depth as a CEILING, and a time-out unwinds by returning the LAST COMPLETED depth's
move, so a rung named `sealbot_d5` can silently play at depth 3 and still report a number. The
adapter drives `max_depth`, neutralises the time cut with an unreachable sentinel, and reads
`last_depth` back as a RECEIPT. A violation raises `SealBotDepthError`, which `eval/worker.py`
does not catch, so it ends the whole eval round — loud and final, because a round that dies is
recoverable while a rung that quietly reports a bar it did not play is not.

POSITION RECONSTRUCTION, NEVER INCREMENTAL TRACKING. The arena applies the opening book straight
to the board without passing it through either bot, so an adapter tracking state from its own
`select_move` calls would start every game four plies behind.

A MODULE NAME INSTALLED WITHOUT TOUCHING `sys.path` (LAW-17/R5). The vendored C++ imports the
top-level name `game`, so `install_game_module` writes `sys.modules` and REFUSES rather than
overwrites: `game` is a very common name and a caller's module is not this adapter's to clobber.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import mantis
from mantis.bots.protocol import RungUnresolvable

#: An arithmetic discharge rather than a knob: the vendored deadline is
#: `now + microseconds(int64(time_limit * 1e6))`, so 1e6 seconds is 1e12 microseconds — four
#: orders of magnitude below int64 max, hence no overflow, and ~11.6 days, hence unreachable
#: inside a game. A TUNABLE time limit is exactly what the falsified register killed.
NON_BINDING_TIME_LIMIT = 1e6

#: The top-level module name the vendored extension imports (`minimax_bot.cpp:16`).
GAME_MODULE_NAME = "game"

#: Path components below the vendor root, kept as separate segments so no path-shaped string
#: literal enters `src/mantis/bots/` at all.
_VENDOR_SEALBOT = ("external", "sealbot")
_BUILT_DIR = "current"
_EXTENSION_GLOB = "minimax_cpp*.so"

#: Skip-reason class markers, beside the strings they mark so there is ONE authority for "which
#: class is this refusal": the in-run counter classifies by importing them rather than by
#: re-transcribing wording that would then drift out of the classifier's reach.
VENDOR_ABSENT_MARKER = "sealbot vendor tree not located"
BUILD_ABSENT_MARKER = "sealbot extension not built"
LOAD_FAILED_MARKER = "sealbot vendored modules failed to load"

#: The refusal reasons, built once from the markers above. Each names EXACTLY ITS OWN missing
#: step: a reason naming both commands would be a checklist, not a diagnosis.
VENDOR_ABSENT_REASON = (
    f"{VENDOR_ABSENT_MARKER}: no ancestor of the installed package holds vendor/pins.toml, "
    "so there is nowhere for the pinned engine to live; run `make vendor` from the repo root"
)
#: The ONE tracked step that builds the extension. The reason names the SCRIPT, not the raw
#: invocation, so a box runs the repair from the tree rather than from a record.
BUILD_SCRIPT = "tools/vendor_build_sealbot.sh"
BUILD_ABSENT_REASON = (
    f"{BUILD_ABSENT_MARKER}: no {_EXTENSION_GLOB} under vendor/external/sealbot/current/; "
    f"fetch the pin, then run `bash {BUILD_SCRIPT}` from the repo root (it verifies the pinned "
    "sha and the applied patch, then runs `python setup.py build_ext --inplace` in that "
    "directory)"
)

#: The module object this loader installed at `sys.modules["game"]`, or None. Identity, not a
#: name test: it is what lets the loader tell ITS OWN module from a foreign squatter.
_INSTALLED_GAME: Any = None


class SealBotDepthError(RuntimeError):
    """The `last_depth` receipt did not match the configured fixed depth. Named and importable so
    a caller can discriminate a receipt violation from an unrelated failure: it means the bar the
    rung CLAIMS to be is not the bar it PLAYED."""


class SealBotModuleCollisionError(RuntimeError):
    """`sys.modules["game"]` was already occupied by something this loader did not install."""


def find_vendor_root() -> Path | None:
    """The `vendor/` directory of the repo the package is installed from, or None. Returns None,
    never a default path and never an env-provided one: an endpoint that can point anywhere is a
    host-path channel wearing a disguise."""
    package_file = mantis.__file__
    if package_file is None:  # namespace package: nothing to walk up from
        return None
    for ancestor in Path(package_file).resolve().parents:
        if (ancestor / "vendor" / "pins.toml").is_file():
            return ancestor / "vendor"
    return None


def install_game_module(path: Path) -> Any:
    """Install the vendored `game.py` at `sys.modules["game"]`, refusing to overwrite. `sys.path`
    is untouched, which is why this goes by explicit file path; a pre-existing occupant this
    loader did not install raises and is left exactly as it was found."""
    global _INSTALLED_GAME
    if GAME_MODULE_NAME in sys.modules:
        occupant = sys.modules[GAME_MODULE_NAME]
        if occupant is not _INSTALLED_GAME:
            raise SealBotModuleCollisionError(
                f"sys.modules[{GAME_MODULE_NAME!r}] is already occupied by {occupant!r}, which "
                f"this loader did not install. The sealbot adapter refuses to shadow a "
                f"caller's module rather than silently overwriting it (LAW-17, DESIGN_A §2.5.2)."
            )
        return occupant

    spec = importlib.util.spec_from_file_location(GAME_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no import spec for the vendored game module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[GAME_MODULE_NAME] = module
    _INSTALLED_GAME = module
    return module


def _load_extension(so_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("minimax_cpp", so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no import spec for the vendored extension at {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sealbot_modules() -> tuple[Any, Any]:
    """`(minimax_module, game_module)`, or `RungUnresolvable` naming the ONE missing step. Probed
    EAGERLY by the resolver: a factory that resolved and then failed mid-round would make every
    skip oracle green while the rung died in a scored round."""
    root = find_vendor_root()
    if root is None:
        raise RungUnresolvable(rung="sealbot", reason=VENDOR_ABSENT_REASON)
    sealbot_dir = root.joinpath(*_VENDOR_SEALBOT)
    built = sorted((sealbot_dir / _BUILT_DIR).glob(_EXTENSION_GLOB))
    if not built:
        raise RungUnresolvable(rung="sealbot", reason=BUILD_ABSENT_REASON)
    game_module = install_game_module(sealbot_dir / f"{GAME_MODULE_NAME}.py")
    return _load_extension(built[0]), game_module


def _vendored_player(mantis_player: int, player_enum: Any) -> Any:
    """Map mantis `+1 -> Player.A`, `-1 -> Player.B`; anything else is a loud failure.

    Mantis `+1` is always the side that moves first, which is the side the vendored rules give one
    stone on turn one. The vendored engine decides identity with an `is` test, so a wrong mapping
    does not error: it silently plays the other colour and every number is then the opponent's.
    """
    if mantis_player == 1:
        return player_enum.A
    if mantis_player == -1:
        return player_enum.B
    raise ValueError(f"unmappable mantis player {mantis_player!r}; expected +1 or -1")


class _ShadowGame:
    """The vendored `HexGame` view the C++ extractor reads, rebuilt from a mantis board.

    The extractor reads exactly four attributes, so those four are eager. `winner` is LAZY: eager
    would require a vendored surface the engine never touches, and deriving it from mantis's OWN
    detector would make the rules differential vacuous.
    """

    __slots__ = ("_game_module", "board", "current_player", "move_count", "moves_left_in_turn")

    def __init__(self, *, board: dict, current_player: Any, moves_left_in_turn: int,
                 move_count: int, game_module: Any) -> None:
        self.board = board
        self.current_player = current_player
        self.moves_left_in_turn = moves_left_in_turn
        self.move_count = move_count
        self._game_module = game_module

    @property
    def winner(self) -> Any:
        probe = self._game_module.HexGame()
        probe.board = dict(self.board)
        for cell in self.board:
            if probe._check_win(*cell):  # noqa: SLF001 — the vendored rule is the point
                return self.board[cell]
        return self._game_module.Player.NONE


def build_shadow_game(board: Any, *, game_module: Any) -> _ShadowGame:
    """Rebuild the vendored view from a mantis `Board`. Stateless, every call."""
    player_enum = game_module.Player
    stones = {
        (int(q), int(r)): _vendored_player(int(p), player_enum)
        for q, r, p in board.get_stones()
    }
    return _ShadowGame(
        board=stones,
        current_player=_vendored_player(int(board.current_player), player_enum),
        moves_left_in_turn=int(board.moves_remaining),
        move_count=len(stones),
        game_module=game_module,
    )


class SealBotAdapter:
    """`BotProtocol` over the vendored engine: one half-ply per `select_move` call."""

    def __init__(self, *, depth: int, minimax_module: Any, game_module: Any) -> None:
        self._depth = depth
        self._game_module = game_module
        self._engine = minimax_module.MinimaxBot()
        self._win_threshold = self._resolve_score_channel(minimax_module)
        #: The compound-turn discard is a COUNTED in-run event, never a silent `if`: a reader has
        #: to be able to see it fire while the run is going (LAW-18).
        self.illegal_buffer_discards = 0
        #: The receipt the last search returned, kept readable so the box rider can record it.
        self.last_reached_depth = 0
        self._buffer: list[tuple[int, int]] = []
        self._buffer_seat: int | None = None
        self._configure()

    def _resolve_score_channel(self, minimax_module: Any) -> float | None:
        """The engine's own mate-distance threshold, or `None` when there is no score channel.

        Two honest answers and no third: the extension exposes `last_score` AND, through the vendor
        patch, `WIN_THRESHOLD`; or a collaborator with no `last_score` has no score to compare. One
        that reports a score but no threshold is a MIS-BUILT vendor tree and refuses — falling back
        to the weaker board-derived proof is how a receipt stops catching what it exists for.
        """
        if not hasattr(self._engine, "last_score"):
            return None
        if not hasattr(minimax_module, "WIN_THRESHOLD"):
            raise RungUnresolvable(
                rung="sealbot",
                reason=(
                    f"{BUILD_ABSENT_MARKER}: the vendored engine reports `last_score` but the "
                    f"extension exposes no `WIN_THRESHOLD`, so `vendor/patches/sealbot.patch` "
                    f"did not apply. Without the engine's own mate-distance threshold the depth "
                    f"receipt cannot tell a legitimate early break from a silent truncation, and "
                    f"guessing is what F-20 is in the register for. Re-run `make vendor`, then "
                    f"rebuild."
                ),
            )
        return float(minimax_module.WIN_THRESHOLD)

    def _configure(self) -> None:
        self._engine.max_depth = self._depth
        self._engine.time_limit = NON_BINDING_TIME_LIMIT

    def name(self) -> str:
        return f"sealbot_d{self._depth}"

    def new_game(self) -> None:
        """Clear the compound-turn buffer and re-assert the two levers. No board state is held
        between calls, so there is nothing else to reset."""
        self._buffer.clear()
        self._buffer_seat = None
        self._configure()

    def select_move(self, board: Any) -> tuple[int, int]:
        seat = int(board.current_player)
        buffered = self._take_buffered(board, seat)
        if buffered is not None:
            return buffered
        return self._search(board, seat)

    def _take_buffered(self, board: Any, seat: int) -> tuple[int, int] | None:
        """The second half of a compound turn, or None if a fresh search is owed.

        `get_move` returns one OR two moves against a protocol that is one half-ply per call, and
        the load-bearing assumption is that no opponent stone lands between the two halves. Two
        ways it can fail, and they are DIFFERENT events: the turn changed under us, so the buffer
        is dropped without a count; or the buffered move is no longer legal, which should be
        impossible, so it IS counted and the adapter re-searches.
        """
        if not self._buffer:
            return None
        candidate = self._buffer.pop(0)
        if self._buffer_seat != seat:
            self._buffer_seat = None
            return None
        if candidate in board.legal_moves():
            return candidate
        self.illegal_buffer_discards += 1
        self._buffer_seat = None
        return None

    def _search(self, board: Any, seat: int) -> tuple[int, int]:
        game = build_shadow_game(board, game_module=self._game_module)
        moves = [(int(q), int(r)) for q, r in self._engine.get_move(game)]
        if not moves:
            raise SealBotDepthError(
                f"the vendored engine returned no move at configured depth {self._depth}"
            )
        self.last_reached_depth = int(self._engine.last_depth)
        self._check_receipt(board, seat, game, moves[0])
        # Buffer the second half ONLY when a second half is actually DUE. `get_move` can return a
        # pair on a turn with one half left — a 4-ply book opening leaves the first mover exactly
        # ONE half — and buffering it there parks a move belonging to no turn, which re-surfaces
        # at the adapter's NEXT turn. Measured before this guard: `illegal_buffer_discards == 1`
        # in 4 of 6 real games from book openings.
        if len(moves) > 1 and int(board.moves_remaining) > 1:
            self._buffer = [moves[1]]
            self._buffer_seat = seat
        return moves[0]

    def _check_receipt(self, board: Any, seat: int, game: Any, move: tuple[int, int]) -> None:
        """Raise unless the short search had a reason the ENGINE ITSELF would recognise.

        The receipt catches ONE thing — a silent time-out truncation reporting a bar the rung did
        not play — so it must model every legitimate reason `last_depth` comes back below the
        ceiling, or it rejects correct play. Four, each read off the vendored source: no truncation;
        the engine never searched (an empty board returns before the search, so `last_depth` keeps
        its initial `0`, and a search that DID run always leaves `>= 1`); the engine broke out on
        `std::abs(last_score) >= WIN_THRESHOLD`, verbatim through the exported constant; and the
        returned move completes six, the board-derived proof a collaborator with no score channel
        can still offer.

        The score arm is not optional, measured: with only the board-derived proof the real engine
        raised in 6 of 6 games within 4-11 plies, at `last_score` `+99999996` and `-99999995`, the
        winning turn a PAIR so `moves[0]` was never the completing cell.
        """
        reached = self.last_reached_depth
        if reached == self._depth:
            return
        if reached == 0 and not game.board:
            return
        if self._engine_proved_a_terminal_line():
            return
        if move in board.winning_moves(seat):
            return
        raise SealBotDepthError(
            f"sealbot depth receipt violated: configured max_depth={self._depth}, engine "
            f"reported last_depth={reached}, the position was searched, the engine proved no "
            f"terminal line, and the returned move {move} does not end the game. The only "
            f"remaining explanation is a truncated search, so the rung would report a bar it "
            f"did not play (F-20). This ends the eval round — `worker.py:349-356` catches only "
            f"`RungUnresolvable` — rather than passing off a weaker opponent under the rung's "
            f"name."
        )

    def _engine_proved_a_terminal_line(self) -> bool:
        """The deepening loop's own break condition, on the engine's own number. `False` with no
        score channel."""
        if self._win_threshold is None:
            return False
        return abs(float(self._engine.last_score)) >= self._win_threshold


__all__ = [
    "BUILD_ABSENT_MARKER",
    "BUILD_ABSENT_REASON",
    "BUILD_SCRIPT",
    "LOAD_FAILED_MARKER",
    "NON_BINDING_TIME_LIMIT",
    "VENDOR_ABSENT_MARKER",
    "VENDOR_ABSENT_REASON",
    "SealBotAdapter",
    "SealBotDepthError",
    "SealBotModuleCollisionError",
    "build_shadow_game",
    "find_vendor_root",
    "install_game_module",
    "load_sealbot_modules",
]
