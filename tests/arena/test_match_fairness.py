"""⊕ WP11-A arena — paired-match fairness law (design §a.2 match.py, §b test_match_fairness.py).

RED-at-import until IMPL writes `mantis.arena.match`. Pins: argmax-only (no temperature
token anywhere in arena/eval sources — dispatch item 7), the paired color-swap law (every
opening is played exactly twice, colors swapped — deploy_strength_eval.py `_play_pair`
:331-360 parity), and every `GameRecord`'s stamped fields (design §a.2:
`regime_key, opening_id, colors, trajectory_hash, winner, plies`).

ORACLE-CHOSEN SEAM: `GameRecord.colors` is a `{"candidate": <player int>, "opponent": <player
int>}` mapping (player ints ``1``/``-1``, engine convention) — the minimal shape that lets a
paired-color-swap oracle assert without inventing a bespoke enum; the design names the field
but not its exact shape. Openings are constructed locally as a tiny duck-typed
`(opening_id, moves)` namespace so this suite does not couple to `mantis.arena.books`'s
suite (tests/arena/test_books.py) landing in lockstep.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mantis._engine import Board
from mantis.arena.match import DEFAULT_MAX_PLIES, play_paired_match
from mantis.arena.regime import RegimeKey

_REPO = Path(__file__).resolve().parents[2]
_ENCODING = "v6_live2_ls"


@dataclass(frozen=True)
class _Opening:
    opening_id: str
    moves: list


class _FixedMoveBot:
    """Plays a pre-scripted sequence of moves, then falls back to the first legal move."""

    def __init__(self, scripted: list[tuple[int, int]]) -> None:
        self._scripted = list(scripted)
        self._i = 0

    def new_game(self) -> None:
        self._i = 0

    def select_move(self, board):
        if self._i < len(self._scripted):
            mv = self._scripted[self._i]
            self._i += 1
            if mv in board.legal_moves():
                return mv
        return board.legal_moves()[0]

    def name(self) -> str:
        return "fixed_move_bot"


def _board_factory():
    return Board.with_encoding_name(_ENCODING)


def _regime_key(*, deploy_matched: bool) -> RegimeKey:
    return RegimeKey(
        bot="candidate", variant="test", model_sims=1, opponent_spec="fixed",
        opening_book="test_book", deploy_matched=deploy_matched, encoding=_ENCODING,
    )


def _openings() -> list[_Opening]:
    return [
        _Opening(opening_id="op0", moves=[(0, 0), (1, 1), (0, 1), (1, 0)]),
        _Opening(opening_id="op1", moves=[(2, 2), (3, 3), (2, 3), (3, 2)]),
    ]


def test_argmax_only_no_temperature_token_in_arena_or_eval():
    for pkg in ("arena", "eval"):
        pkg_dir = _REPO / "src" / "mantis" / pkg
        if not pkg_dir.exists():
            continue
        for path in pkg_dir.rglob("*.py"):
            text = path.read_text()
            assert "temperature" not in text.lower(), (
                f"{path} carries a 'temperature' token — no eval/arena surface may have one "
                "(dispatch item 7: not schema-representable for eval)"
            )


def test_paired_openings_swap_colors_exactly():
    candidate = _FixedMoveBot([(5, 5), (6, 6)])
    opponent = _FixedMoveBot([(7, 7), (8, 8)])
    records = play_paired_match(
        candidate, opponent, _openings(),
        regime_key=_regime_key(deploy_matched=True),
        board_factory=_board_factory, max_plies=DEFAULT_MAX_PLIES, record_sink=None,
    )
    assert len(records) == len(_openings()) * 2, "each opening must be played exactly twice"
    by_opening: dict[str, list] = {}
    for rec in records:
        by_opening.setdefault(rec.opening_id, []).append(rec)
    for opening_id, recs in by_opening.items():
        assert len(recs) == 2, f"opening {opening_id} must appear exactly twice"
        cand_colors = {rec.colors["candidate"] for rec in recs}
        assert cand_colors == {1, -1}, (
            f"opening {opening_id}: candidate must play BOTH colors across its pair, got {cand_colors}"
        )


def test_records_stamp_regime_key_opening_and_trajectory_hash():
    candidate = _FixedMoveBot([(5, 5)])
    opponent = _FixedMoveBot([(7, 7)])
    regime_key = _regime_key(deploy_matched=True)
    records = play_paired_match(
        candidate, opponent, _openings(),
        regime_key=regime_key, board_factory=_board_factory, max_plies=DEFAULT_MAX_PLIES, record_sink=None,
    )
    assert len(records) > 0
    for rec in records:
        assert rec.regime_key == regime_key
        assert rec.opening_id in {"op0", "op1"}
        assert isinstance(rec.trajectory_hash, str) and len(rec.trajectory_hash) == 64, (
            "trajectory_hash must be a sha256 hex digest"
        )
        assert rec.winner in ("candidate", "opponent", "draw")
        assert isinstance(rec.plies, int) and rec.plies > 0


def test_trajectory_hash_is_move_order_sensitive_and_deterministic():
    candidate = _FixedMoveBot([(5, 5), (6, 6)])
    opponent = _FixedMoveBot([(7, 7), (8, 8)])
    regime_key = _regime_key(deploy_matched=True)
    records_a = play_paired_match(
        candidate, opponent, [_Opening(opening_id="op0", moves=[(0, 0), (1, 1), (0, 1), (1, 0)])],
        regime_key=regime_key, board_factory=_board_factory, max_plies=DEFAULT_MAX_PLIES, record_sink=None,
    )
    records_b = play_paired_match(
        _FixedMoveBot([(5, 5), (6, 6)]), _FixedMoveBot([(7, 7), (8, 8)]),
        [_Opening(opening_id="op0", moves=[(0, 0), (1, 1), (0, 1), (1, 0)])],
        regime_key=regime_key, board_factory=_board_factory, max_plies=DEFAULT_MAX_PLIES, record_sink=None,
    )
    assert [r.trajectory_hash for r in records_a] == [r.trajectory_hash for r in records_b], (
        "identical scripted play from an identical opening must reproduce identical hashes"
    )
    records_diff_opening = play_paired_match(
        _FixedMoveBot([(5, 5), (6, 6)]), _FixedMoveBot([(7, 7), (8, 8)]),
        [_Opening(opening_id="op1", moves=[(2, 2), (3, 3), (2, 3), (3, 2)])],
        regime_key=regime_key, board_factory=_board_factory, max_plies=DEFAULT_MAX_PLIES, record_sink=None,
    )
    hashes_a = {r.trajectory_hash for r in records_a}
    hashes_diff = {r.trajectory_hash for r in records_diff_opening}
    assert hashes_a.isdisjoint(hashes_diff), (
        "a different opening (different move order) must produce different trajectory hashes"
    )


def test_deploy_matched_flag_travels_from_rung_to_record():
    candidate = _FixedMoveBot([(5, 5)])
    opponent = _FixedMoveBot([(7, 7)])
    matched_key = _regime_key(deploy_matched=True)
    unmatched_key = _regime_key(deploy_matched=False)
    matched_records = play_paired_match(
        candidate, opponent, _openings(), regime_key=matched_key,
        board_factory=_board_factory, max_plies=DEFAULT_MAX_PLIES, record_sink=None,
    )
    unmatched_records = play_paired_match(
        _FixedMoveBot([(5, 5)]), _FixedMoveBot([(7, 7)]), _openings(), regime_key=unmatched_key,
        board_factory=_board_factory, max_plies=DEFAULT_MAX_PLIES, record_sink=None,
    )
    assert all(r.regime_key.deploy_matched is True for r in matched_records)
    assert all(r.regime_key.deploy_matched is False for r in unmatched_records)
