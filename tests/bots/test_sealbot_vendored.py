# >300 justify (R8). The four Tier-2 oracles are ONE claim — the
# REAL vendored engine is the bar the rungs say it is — over one precondition ladder
# (`_require_vendored_game` / `_require_built_extension`). A split forks that ladder into
# copies, and the whole point of the file is that its skip is a RESULT with one reason
# vocabulary, not four ad-hoc ones.
"""⊕ WP12-R Phase A / O-A11, O-A12, O-A13, O-A14(real) — Tier 2, `not_run` in CI (R169).

**These rows do not run in CI and that is the ruled outcome, not a gap.** R169 splits
"RESOLVES" from "IS LIVE": resolution is a property of mantis code plus the local
filesystem and is CI-verifiable; liveness is a property of a built C++/pybind11 extension
executing, and it has no runnable producer in CI. Every row here reports **`not_run`** —
which is a RESULT, never coverage — and names the box measurement that will produce a real
one (DESIGN_A §3.5.4). Reporting one of these as `covered` without having executed it is
PREREG_A §8 abort 6.

| oracle | tier | box counterpart | what it would prove |
|---|---|---|---|
| O-A13 | 2a — `make vendor` only | **M-1** | mantis and the vendored engine agree on the GAME |
| O-A11 | 2b — vendor + build | **M-2** | `last_depth == depth`; the bar is the bar it claims |
| O-A12 | 2b | **M-3** | same position + depth -> same move; the bar is reproducible |
| O-A14 | 2b | **M-4** | the compound-turn buffer holds over a real game |

The defect each row is the ONLY witness to:

- **O-A13** — a rules mismatch. The MOST SERIOUS outcome in the whole rider (DESIGN_A
  §3.5.5): if the two implementations disagree about the win condition, the legal set or
  the turn structure, then every SealBot number measures a DIFFERENT GAME and no amount of
  determinism or depth discipline rescues it. It runs one tier earlier than the extension
  rows because it needs only the vendored `game.py` — deliberately putting the
  highest-risk correctness surface behind the CHEAPEST precondition.
- **O-A11** — F-20 reproduced against the real engine. The Tier-1 pin (O-A10) proves the
  adapter CHECKS the receipt; only this row proves the receipt HOLDS when a real search
  runs. Anything below 100% is a FAIL — and the failure is reported, never tuned by raising
  the time limit or lowering the depth.
- **O-A12** — an irreproducible bar. `_rng` is dead by grep (DESIGN_A §1.10), but a grep
  over headers the design did not read is EVIDENCE, not proof; this row is the evidence,
  and PREREG_A §1 says so explicitly.
- **O-A14 real** — the buffer invariant over a real game rather than a scripted double.

SR-7 (PREREG_A §0): the box rider selects these with `pytest -k`, so the substrings
`rules_agreement`, `depth_receipt` and `determinism` are a CONTRACT on these function names.
A `-k` string matching nothing exits **5** — neither a pass nor a fail — which DESIGN_A
§3.5.5 maps to `not_run — selector matched nothing`, a RIDER defect wearing a SealBot
defect's clothes.
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

#: The cross-process determinism probe, a module constant so the command the box pastes into
#: its log is readable rather than assembled inline. It reaches the adapter through the SAME
#: public entry points production uses.
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
    """Tier 2b. LOUD skip naming the exact build command DESIGN_A §2.6 fixes.

    Its OWN preconditions, not a call through `_require_vendored_game`: a Tier-2b row that
    skipped with Tier 2a's reason would name box counterpart M-1 for a measurement that is
    M-2/M-3/M-4, and the box would record the wrong row as `not_run`.
    """
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
    """`n` distinct positions replayed from the sha-pinned book — the same openings the
    rungs actually play, never a synthetic corpus."""
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


#: G-A1 re-point. The three hex axes, as `game.py:21` declares them — used ONLY to walk a
#: position forward, never to decide a winner (deciding one here would be the vacuity this
#: grant exists to remove).
_HEX_DIRECTIONS = ((1, 0), (0, 1), (1, -1))

#: Plies to drive a book opening forward before giving up on it. MEASURED at the re-point,
#: not chosen: at 40 every one of the 200 openings is already decided, and 60 and 80 return
#: the identical set, so the cap is well clear of the boundary rather than tuned to it.
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
    """`n` positions that actually have a WINNER, driven forward from the pinned book.

    **G-A1 (WP12-R Phase A IMPL, granted): a DOMAIN correction, not a weakening.** Every
    assertion in the two rows below is unchanged; only the set of positions they are applied
    to moves. The subject — "the two implementations name the SAME winner under the declared
    `+1 -> Player.A` map" — is identical.

    Why it was owed, MEASURED rather than argued: the book's openings are all **4-ply**
    (`paired_openings(...)` returns 200 openings, every one 4 moves), and six-in-a-row needs
    six stones, so `_book_positions` cannot contain a decided position — **0 of 200**. Arm 3
    therefore never incremented `checked`, and its own `assert checked > 0` guard fired: the
    row asserted nothing and said so, which is the guard working exactly as R81/R86 intend.
    Left alone it would have returned a **false `FAILED — rules mismatch`** at box
    measurement M-1 and routed the decision line to STATE 3b on a falsehood.

    The drive is deterministic and uses only mantis's own engine: take the winning move when
    the engine offers one (`first_winning_move`), otherwise extend the mover's own line.
    **Measured at the re-point: 200 of 200 openings reach a decided position within
    `_DRIVE_PLY_CAP`, and BOTH winner identities occur (189 for `+1`, 11 for `-1`)** — so the
    row discriminates an inverted map in both directions, which a single-winner sample could
    not do. Positions carry 11-14 stones.

    The `assert` below is deliberately kept: if a geometry or turn-structure change ever made
    the drive stop producing decided positions, this row must go loud rather than quietly
    shrink back to the vacuous sample it was rescued from.
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


# ── O-A13 (Tier 2a) -> box M-1 ──────────────────────────────────────────────────────────
def test_rules_agreement_win_and_legality_differential() -> None:
    """O-A13 arm 1. The axial basis and the three win directions must coincide, measured
    over the book rather than argued from `game.py:21`.

    **G-A1 re-point (domain only, assertions untouched):** the sample is now the 200 book
    positions **plus** 200 driven-to-decided ones. The legality half was always real over the
    book; the WIN half was vacuous there — 0 of 200 book positions is decided, so `mantis_win`
    and `vendored_win` were both `False` on every row and the comparison could not disagree.
    Adding decided positions is what makes the win half able to fail at all.
    """
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
    """O-A13 arm 2. `moves_remaining` and `moves_left_in_turn` must agree at EVERY ply: the
    compound-turn structure is what the adapter's buffer depends on, and a one-ply offset
    would make the adapter play the wrong side's half."""
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
    """O-A13 arm 3. Agreeing that SOMEONE won is not agreement: the `+1 -> Player.A` map has
    to make the two implementations name the SAME winner, or every rung's win rate is the
    opponent's.

    **G-A1 re-point (domain only, assertions untouched):** `_decided_positions` replaces
    `_book_positions`. See that helper for the measurement — 0 of 200 book positions can be
    decided, so this row's `checked > 0` guard fired and the row certified nothing. Both
    winner identities occur in the new sample, so an inverted map is caught in either
    direction. The guard below is retained deliberately: it is the reason the defect was
    visible rather than silent, and it must stay able to fire.
    """
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


# ── O-A11 (Tier 2b) -> box M-2 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("depth", [5, 6])
def test_depth_receipt_holds_on_every_move_at_the_configured_depth(depth: int) -> None:
    """O-A11 -> box M-2. Parametrized over run5's two minted sealbot depths, because a
    receipt that holds at 5 and truncates at 6 is precisely the failure a single-depth row
    would report as a pass."""
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
    """The REAL vendored engine, with the wall-clock cut PUT BACK — for one row only.

    **G-A2 (WP12-R Phase A IMPL, granted): a DOMAIN correction, not a weakening.** The
    assertions below are byte-for-byte what they were; what moves is the condition under
    which the receipt is asked to fire.

    Why it was owed, DERIVED at source: the row asked for `depth=99` against an adapter whose
    REQUIRED contract (DESIGN_A §2.4) sets a time limit that provably cannot be reached inside
    a game. `search.h:157` then loops `for depth = 1; depth <= 99; depth++` and its only other
    exits are the `TimeUp` catch — which that contract removes, deliberately — and the
    proven-win break at `:178`, which cannot fire on a four-ply book opening. So the call had
    no terminating path at all: the row would have HUNG at the box, not raised.

    What this wrapper does NOT do: stub the search, fake `last_depth`, or touch a private
    attribute of the adapter. `get_move` is the real C++ search and `last_depth` is the real
    receipt. It is injected through `minimax_module`, the same SR-1 seam the Tier-1 rows use,
    and it intercepts exactly one write — `time_limit` — restoring the cut the adapter
    neutralises. That makes the domain **F-20's actual failure mode**: a time-truncated search
    returning the last completed depth's move while the rung claims a deeper bar. Which is the
    condition the receipt exists to catch, so the re-pointed row tests the receipt on the case
    that motivated it rather than on an unreachable one.

    Why the raise is guaranteed rather than hoped for, stated so RED-TEAM can check it: 0.05 s
    cannot complete 99 plies of alpha-beta on an unbounded board, so `last_depth < 99`; and a
    four-ply opening cannot contain an immediate win, so the adapter's proven-win exemption
    (`move in board.winning_moves(seat)`) cannot apply. Both conjuncts of `_check_receipt`'s
    "do not raise" path therefore fail, and it raises. **If that derivation is ever wrong the
    row fails LOUD** — `pytest.raises` reports `DID NOT RAISE`; it cannot pass quietly.

    Honest limit: unlike G-A1, this re-point is **derived, not measured**. The extension is not
    built in the environment this phase ran in (MS-4 is `not_run`), so the first execution of
    this row is the box's. That is stated rather than glossed.
    """

    def __init__(self, real_module: Any, seconds: float) -> None:
        object.__setattr__(self, "_real_module", real_module)
        object.__setattr__(self, "_seconds", seconds)

    def MinimaxBot(self) -> Any:  # noqa: N802 — vendored name
        return _TimeCutBot(self._real_module.MinimaxBot(), self._seconds)

    def __getattr__(self, name: str) -> Any:
        """Everything else is the REAL module's, `WIN_THRESHOLD` above all.

        Without this the wrapper hides the patch-exported mate-distance constant and the
        adapter refuses to construct — correctly, since a score channel with no threshold is a
        mis-built vendor tree. Measured: this row was the single failure in an otherwise green
        Tier-2 battery, and the adapter's own guard is what caught it.
        """
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


#: A depth the engine cannot reach inside the bounded search below. Unchanged from the sealed
#: row: what changed is that the search now terminates.
_UNREACHABLE_DEPTH = 99

#: The vendored default (`minimax_bot.cpp:82`) — the cut the adapter neutralises in production
#: and this row restores, so the truncation the receipt guards against actually happens.
_BINDING_TIME_LIMIT_SEC = 0.05


def test_depth_receipt_raises_rather_than_reporting_a_shallower_bar() -> None:
    """O-A11's second half. A receipt that is READ but not ACTED ON is an assurance, and the
    whole of DESIGN_A §2.4 is that the adapter carries a receipt instead.

    **G-A2 re-point:** the engine is handed to the adapter through `_TimeCutRestored`, which
    puts the wall-clock cut back. See that class for why the sealed form could not terminate
    and why this one must raise. Assertions unchanged.
    """
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


# ── O-A12 (Tier 2b) -> box M-3 ──────────────────────────────────────────────────────────
def test_determinism_across_five_fresh_instances_in_one_process() -> None:
    """O-A12 -> box M-3, in-process half. The `_rng`-is-dead grep is NOT the evidence; this
    row is (PREREG_A §1)."""
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
    """O-A12's cross-process half. In-process determinism can be an artefact of warm state
    inside one loaded extension; only a second interpreter distinguishes a deterministic
    ENGINE from a cached one."""
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


# ── O-A14 real arm (Tier 2b) -> box M-4 ─────────────────────────────────────────────────
def test_the_compound_turn_buffer_holds_over_a_real_game() -> None:
    """O-A14's real arm -> box M-4 item (f). The Tier-1 arms drive a SCRIPTED double; only a
    real game exercises the assumption that no opponent stone can be placed between the two
    halves of one turn."""
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
    """⊕ G-A4 / RED-TEAM F-RT-2 — the `moves_remaining > 1` invariant, from a BOOK OPENING.

    **The producer this invariant did not have.** O-A14's two real arms start from
    `Board.with_encoding_name(_ENC)` — an empty board with no book — and from there the
    wrapper short-circuits (`minimax_bot.cpp:47-50`) and returns ONE move, so nothing is ever
    buffered on a turn with one half due and the defect's condition never arises. Measured by
    RED-TEAM: delete the invariant and the entire suite, both tiers, stays green.

    That is the same shape as the defect it guards — *a condition arising only from a book
    opening, tested only from an empty board* — so this row reaches it the way the defect was
    actually found: through the sha-pinned book the rungs really play.

    **The precondition is asserted, not assumed.** A four-ply opening leaves the first mover
    exactly ONE half due; measured across the first four openings, `moves_remaining == 1` on
    every one. If a future book changed that, this row would stop reaching its subject, and it
    must say so rather than pass.

    **The observation is a determinism differential, and that is deliberate.** Asserting only
    `illegal_buffer_discards == 0` would catch just the loud half: measured with the invariant
    deleted, 2 of 4 openings put the stale half on an occupied cell (counted) while the other 2
    played it silently onto a legal one — and the silent half is the dangerous one, since the
    adapter would be answering a position it never searched. SealBot is deterministic at a
    fixed depth (O-A12), so a fresh adapter on the same position is an exact reference: if the
    live adapter agrees with it, it searched; if it consumed a stale half, it cannot.
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
    """O-A14's real arm -> box M-4 item (f), the counter half. A non-zero count means the
    invariant broke and the adapter re-searched: no illegal move was played, but the
    assumption the buffer rests on is false and DESIGN_A §3.5.5 records it as
    `FAILED — compound-turn buffer defect`."""
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
