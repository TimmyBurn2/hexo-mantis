# >300 justify (R8). ONE seam — the vendored engine's translation boundary —
# whose three properties (the depth receipt, stateless position reconstruction, the
# `sys.modules` install) share the loader, the player map and the shadow view; splitting them
# forks that shared state into copies that drift while both halves stay green. Executable
# content is a minority: the rest is the per-surface "which register row is this the answer
# to" rationale, which DESIGN_A §7's projection did not count.
"""SealBot adapter — the vendored fixed-depth external bar (WP12-R Phase A, DESIGN_A §2.4/§2.5).

Pure Python; no pyo3 (R6/LAW-17 hold trivially — the bridge is untouched). Nothing in this
module names a host path, an operator directory or an endpoint: the ONE external string this
phase permits is the public vendor URL, and it lives in `vendor/pins.toml` (rule 7).

THE THREE PROPERTIES THIS MODULE EXISTS FOR, each with the register row it answers:

1. **A fixed-depth bar that is actually fixed (F-20).** The vendored search is iterative
   deepening with the configured depth as a CEILING (`current/engine/search.h:157`), and a
   time-out unwinds by returning the LAST COMPLETED depth's move (`search.h:179-197`). So a
   rung named `sealbot_d5` can silently play at depth 3 and still report a number. The
   adapter therefore drives `max_depth`, neutralises the time cut with a provably
   non-overflowing, provably unreachable sentinel, and — crucially — reads `last_depth` back
   as a RECEIPT rather than assuming the ceiling was reached. A violation raises
   `SealBotDepthError`. **What that costs, stated exactly rather than hopefully:**
   `src/mantis/eval/worker.py:349-356` catches `RungUnresolvable` and nothing else, so this
   propagates out of the rung block and ends the whole eval round. That is loud and it is
   final — it is NOT a per-rung "record it as broken", and this module said so wrongly until
   REVIEW-impl measured it. A round that dies is recoverable; a rung that quietly reports a
   bar it did not play is not, which is why the raise stays. Making it per-rung is a
   `worker.py` change and is queued, not taken here.

   **The receipt models every LEGITIMATE short search, and it took a built extension to learn
   what they are.** `search.h:178` is `if (std::abs(last_score) >= WIN_THRESHOLD) break;` —
   note the `std::abs`: the engine stops deepening on a proven win, a proven LOSS, and any
   mate inside the band (`engine/constants.h:51-52`, "mate-distance detection"). Reconstructing
   that as "the returned move completes six" is strictly narrower on all three axes, and
   measured against the real engine at run5's minted depth 5 it raised in **6 of 6 games**
   within 4-11 plies. So the receipt now reads the engine's OWN signal instead of a
   board-derived proxy for it, and `vendor/patches/sealbot.patch` exports the threshold so the
   number is never re-typed into Python.
2. **Position reconstruction, never incremental tracking.** The arena applies the opening
   book straight to the board without passing it through either bot
   (`src/mantis/arena/match.py:71-73`), so an adapter that tracked state from its own
   `select_move` calls would start every game four plies behind. `build_shadow_game` rebuilds
   the vendored view from `board.get_stones()` on every call; the adapter holds no board
   state, so there is no state to desynchronise.
3. **A module NAME installed without touching `sys.path` (LAW-17/R5).** The vendored C++
   does `py::module_::import("game")` (`current/minimax_bot.cpp:16`), so the name `game` must
   exist at top level. `install_game_module` writes `sys.modules`, never `sys.path`, and it
   REFUSES rather than overwrites: `game` is a very common name and a caller's module is not
   this adapter's to clobber. The blast radius is bounded by process — eval runs in a spawned
   child (`src/mantis/eval/pipeline.py:168`, `mp_ctx_name="spawn"`).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import mantis
from mantis.bots.protocol import RungUnresolvable

#: DESIGN_A §2.4, and it is an arithmetic discharge rather than a knob (PREREG_A W-4): the
#: vendored deadline is `now + microseconds(int64(time_limit * 1e6))` (`search.h:41-42`). At
#: 1e6 seconds that is 1e12 microseconds — four orders of magnitude below int64 max (~9.2e18),
#: so it provably cannot overflow, and ~11.6 days, so it provably cannot be reached inside a
#: game. A *tunable* time limit is exactly what F-20 falsified, which is why this is not a
#: schema field: a reviewer who judges otherwise gets a queue row, not a config key.
NON_BINDING_TIME_LIMIT = 1e6

#: The top-level module name the vendored extension imports (`minimax_bot.cpp:16`).
GAME_MODULE_NAME = "game"

#: Path components below the vendor root. Kept as separate segments so no path-shaped string
#: literal enters `src/mantis/bots/` at all.
_VENDOR_SEALBOT = ("external", "sealbot")
_BUILT_DIR = "current"
_EXTENSION_GLOB = "minimax_cpp*.so"

#: Skip-reason class markers. These live HERE, beside the strings they mark, so there is ONE
#: authority for "which class is this refusal" — `mantis.eval.pipeline`'s in-run counter
#: (LAW-18/R164) classifies by importing them rather than by re-transcribing wording that
#: would then be free to drift out of the classifier's reach.
VENDOR_ABSENT_MARKER = "sealbot vendor tree not located"
BUILD_ABSENT_MARKER = "sealbot extension not built"
LOAD_FAILED_MARKER = "sealbot vendored modules failed to load"

#: The refusal reasons, built once from the markers above. Each names EXACTLY ITS OWN missing
#: step: a reason naming both commands would be a checklist, not a diagnosis, and the operator
#: could not tell which step to run.
VENDOR_ABSENT_REASON = (
    f"{VENDOR_ABSENT_MARKER}: no ancestor of the installed package holds vendor/pins.toml, "
    "so there is nowhere for the pinned engine to live; run `make vendor` from the repo root"
)
#: The ONE tracked step that builds the extension. The reason names the SCRIPT, not the raw
#: invocation, so a box runs the repair from the tree rather than from a record (R324(c)).
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
    """The `last_depth` receipt did not match the configured fixed depth (SR-5).

    A named, importable type on purpose: `pytest.raises` — and any caller — must be able to
    discriminate a receipt violation from an unrelated failure. This is not a warning. It
    means the bar the rung CLAIMS to be is not the bar it PLAYED, which is F-20 live.
    """


class SealBotModuleCollisionError(RuntimeError):
    """`sys.modules["game"]` was already occupied by something this loader did not install."""


def find_vendor_root() -> Path | None:
    """The `vendor/` directory of the repo the package is installed from, or None.

    Walks up from `mantis.__file__` for an ancestor holding `vendor/pins.toml` — the ONE
    vendoring authority (CLAUDE.md "Deliberately absent") — and returns that ancestor's
    `vendor` directory. Returns **None**, never a default path and never an env-provided one:
    an endpoint that can point anywhere is a host-path channel wearing a disguise, which is
    why DESIGN_A §2.2(2) deleted the three per-kind environment keys instead of re-using one
    of them here. The names themselves survive only in the oracle, as strings a refusal reason
    may never speak.
    """
    package_file = mantis.__file__
    if package_file is None:  # namespace package: nothing to walk up from
        return None
    for ancestor in Path(package_file).resolve().parents:
        if (ancestor / "vendor" / "pins.toml").is_file():
            return ancestor / "vendor"
    return None


def install_game_module(path: Path) -> Any:
    """Install the vendored `game.py` at `sys.modules["game"]`, refusing to overwrite.

    `sys.modules` is written; **`sys.path` is not** — that is the letter of LAW-17 and it is
    also why this is by explicit file path rather than by making a directory importable.
    Squatting a name as common as `game` is stated plainly rather than glossed, and the
    refusal is the mitigation: a pre-existing occupant this loader did not install raises,
    and is left exactly as it was found. A refusal must cost the caller nothing.
    """
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
    """`(minimax_module, game_module)`, or `RungUnresolvable` naming the ONE missing step.

    Probed EAGERLY by the resolver, deliberately: a factory that resolved and then failed
    mid-round would make every skip oracle green while the rung died in a scored round.
    """
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
    """mantis `+1 -> Player.A`, `-1 -> Player.B`, anything else is a loud failure.

    Not a convention chosen for convenience: `match.py:114-115` starts every game with the
    board's own first mover and swaps only the CANDIDATE's colour, so mantis `+1` is always
    the side that moves first — which is exactly the side the vendored rules give one stone on
    turn one (`game.py:42`). It matters because `minimax_bot.cpp:26` decides identity with an
    `is` test: a wrong mapping does not error, it silently plays the other colour and every
    SealBot number is then the opponent's.
    """
    if mantis_player == 1:
        return player_enum.A
    if mantis_player == -1:
        return player_enum.B
    raise ValueError(f"unmappable mantis player {mantis_player!r}; expected +1 or -1")


class _ShadowGame:
    """The vendored `HexGame` view the C++ extractor reads, rebuilt from a mantis board.

    `minimax_bot.cpp:19,30-33` reads exactly four attributes — `board`, `current_player`,
    `moves_left_in_turn`, `move_count` — so those four are eager. `winner` is a LAZY property
    and both halves of that choice are load-bearing: eagerly computing it would make the
    adapter require a vendored surface the engine never touches (breaking every Tier-1 row,
    which hands in a double exposing only `Player`), and computing it from mantis's OWN
    detector would make the rules differential vacuous — it would be comparing mantis to
    itself. It is therefore derived from the VENDORED win rule, on demand.
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
        #: LAW-18/R164: the compound-turn discard is a COUNTED in-run event, never a silent
        #: `if`. A reader has to be able to see it fire while the run is going.
        self.illegal_buffer_discards = 0
        #: The receipt the last search returned, kept readable so the box rider can record it.
        self.last_reached_depth = 0
        self._buffer: list[tuple[int, int]] = []
        self._buffer_seat: int | None = None
        self._configure()

    def _resolve_score_channel(self, minimax_module: Any) -> float | None:
        """The engine's own mate-distance threshold, or `None` when there is no score channel.

        A capability question with exactly two honest answers and no third:

        * the vendored extension exposes `MinimaxBot.last_score` (`minimax_bot.cpp:92-94`) AND,
          through `vendor/patches/sealbot.patch`, the module constant `WIN_THRESHOLD`
          (`engine/constants.h:51-52`). The threshold is returned, and `_check_receipt` reads
          `search.h:178`'s condition VERBATIM rather than reconstructing it;
        * a collaborator with no `last_score` at all — the Tier-1 recording double — has no
          score to compare, and `None` says precisely that. It still has the board-derived
          proof, which is the strongest evidence available to something that does not search.

        **A collaborator that reports a score but no threshold is a MIS-BUILT vendor tree** —
        the patch did not apply — and it refuses. Falling back to the weaker board-derived
        proof there is exactly how a receipt stops catching the thing it exists for, silently.
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
        """Clears the compound-turn buffer and re-asserts the two levers. No board state is
        held between calls, so there is nothing else to reset."""
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

        `get_move` returns one OR two moves (`minimax_bot.cpp:55-59`) against a protocol that
        is one half-ply per call. The load-bearing assumption is that no opponent stone can be
        placed between the two halves of one turn: `match.py:80-81` selects the mover from
        `board.current_player`, and `moves_remaining` does not fall to zero until the compound
        turn completes, so the same mover is called twice in succession. Two ways the
        assumption can fail, and they are DIFFERENT events:

        * the turn changed under us — the buffer belongs to a turn that is over, so it is
          dropped without a count (nothing is broken; there is simply nothing to consume);
        * the buffered move is no longer legal — that should be impossible, so it IS counted,
          and the adapter re-searches rather than playing an occupied cell into a scored game.
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
        # Buffer the second half ONLY when a second half is actually DUE. `get_move` can return
        # a pair on a turn with one half left — rung games start from a 4-ply book opening
        # (`match.py:71-73`), which leaves the first mover exactly ONE half, and the engine
        # still answers with a full turn. Buffering it there parks a move that belongs to no
        # turn: the opponent moves next, and the stale half then re-surfaces at the adapter's
        # NEXT turn, where `_take_buffered` sees the same seat and can only catch it if the
        # opponent happens to have taken that very cell. Measured before this guard existed:
        # `illegal_buffer_discards == 1` in 4 of 6 real games from book openings. The buffer's
        # own invariant is the fix, not a wider discard net.
        if len(moves) > 1 and int(board.moves_remaining) > 1:
            self._buffer = [moves[1]]
            self._buffer_seat = seat
        return moves[0]

    def _check_receipt(self, board: Any, seat: int, game: Any, move: tuple[int, int]) -> None:
        """Raise unless the short search had a reason the ENGINE ITSELF would recognise.

        The receipt exists to catch ONE thing: a silent time-out truncation reporting a bar the
        rung did not play (F-20). It must therefore model every legitimate reason `last_depth`
        can come back below the ceiling, or it rejects correct play. Three, each read off the
        vendored source rather than inferred:

        1. **No truncation at all** — `reached == depth`.
        2. **The engine never searched.** `minimax_bot.cpp:47-50` returns `[(0,0)]` on an empty
           board *before* invoking the search, so `last_depth` is never written and keeps its
           initial `0`. Both conjuncts are load-bearing and narrow: a search that DID run
           always leaves `last_depth >= 1`, because the deepening loop starts at 1.
        3. **The engine broke out on its own mate-distance signal** — `search.h:178`,
           `if (std::abs(last_score) >= WIN_THRESHOLD) break;`, read verbatim through the
           exported constant rather than reconstructed. Note the `std::abs`: this covers a
           proven win, a proven LOSS, and any mate inside the band.
        4. **The returned move completes six** — the board-derived proof, which is all a
           collaborator with no score channel (the Tier-1 double) can offer. On the real engine
           it is subsumed by (3); it is kept because it is sound and because a double that
           cannot report a score can still prove this much.

        **Why (3) is not optional, measured rather than argued:** until REVIEW-impl built the
        extension, this method carried only (1) and (4) — `DESIGN_A` §2.4 quoted the
        `std::abs(...)` line correctly and then substituted "the returned move immediately
        wins", which is strictly narrower on all three of win / loss / mate-distance. Against
        the real engine at run5's minted depth 5 that raised in **6 of 6 games** within 4-11
        plies, at `last_score` values of `+99999996` (proven win) and `-99999995` (proven loss),
        with the winning turn a PAIR so `moves[0]` was never the completing cell. A receipt that
        fires on correct play is not a stricter receipt; it is a broken rung.
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
        """`search.h:178`'s condition, on the engine's own number. `False` with no score channel."""
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
