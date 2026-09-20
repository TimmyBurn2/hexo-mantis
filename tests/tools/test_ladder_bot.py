"""`tools/ladder_bot.py` (LADDER-1): the entry point — token from the environment only, the time control parsed, and `--replay` as the determinism + budget witness over a receipt's own recorded positions."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_NET = "a9a46c55bd1ceadb38145fef6527254d77d37900aebea4332def55ca75f56bfc"
#: An opening whose plies 2–3 are the recorded opponent's and whose ply 4 is the receipt's first recorded stone.
_OPENING = {"book": "book_v1_s20260625_p4", "index": 0, "opening_id": "0", "relative": [[0, 0], [1, 0], [0, 1], [2, 0]],
            "off_book_at": None}


@pytest.fixture(scope="module")
def bot_mod(ladder):
    path = _REPO / "tools" / "ladder_bot.py"
    spec = importlib.util.spec_from_file_location("ladder_bot_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text, want", [
    ("unlimited", {"mode": "unlimited"}),
    ("turn:45000", {"mode": "turn", "turnTimeMs": 45000}),
    ("match:300000:2000", {"mode": "match", "mainTimeMs": 300000, "incrementMs": 2000}),
])
def test_the_time_control_spelling_becomes_the_servers_union(bot_mod, text: str, want: dict[str, Any]) -> None:
    assert bot_mod.parse_time_control(text) == want


@pytest.mark.parametrize("text", ["", "turn", "turn:x", "match:1", "blitz:100", "unlimited:5"])
def test_a_time_control_the_server_would_refuse_is_refused_here_first(bot_mod, text: str) -> None:
    with pytest.raises(ValueError):
        bot_mod.parse_time_control(text)


def test_a_missing_token_is_a_named_exit_not_a_401_later(bot_mod, monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.delenv("HEXO_TOKEN", raising=False)
    rc = bot_mod.main(["--backend", "strix", "--server", "http://127.0.0.1:9", "--work-dir", str(tmp_path)])
    assert rc == 2
    assert "HEXO_TOKEN" in capsys.readouterr().err


class _ReplayBackend:
    """Deterministic scripted turns keyed by stone count; `flip` makes the second call disagree."""

    backend, name, net_hash, sims, encoding, search, seed = "mantis", f"mantis:{_NET[:8]}", _NET, 4, "gnn_axis_v1", {}, None

    def __init__(self, flip: bool = False, sims_per_turn: int = 8) -> None:
        self.flip, self.sims_per_turn, self.calls = flip, sims_per_turn, 0

    def new_game(self, game_id: str) -> None:
        return None

    def select_turn(self, board: Any, forced=()):
        from ladder.backends import TurnResult  # noqa: PLC0415

        self.calls += 1
        stones = len(board.get_stones())
        turns = {3: ((2, 0), (3, 0)), 7: ((-1, 0), (-2, 0))}
        first, second = turns[stones]
        if self.flip and stones == 7:
            first, second = second, first
        placements = tuple(forced) + (first, second)[len(forced):]
        return TurnResult(placements=(placements[0], placements[1]), sims=self.sims_per_turn, ms=2.0,
                          book_stones=len(forced))

    def close(self) -> None:
        return None


def _receipt(ladder, tmp_path: Path) -> Path:
    r = ladder.receipt.GameReceipt(
        server="s", game_id="g_7Qm2Kx",
        bot={"name": f"mantis:{_NET[:8]}", "backend": "mantis", "net_hash": _NET, "display_name": "M", "profile_id": "p"},
        opponent={"display_name": "S", "profile_id": "q", "elo": 1000}, side="x", time_control={"mode": "unlimited"},
        rated=False, sims_configured=4, search={}, started=0.0, opening=_OPENING)
    r.add_move(request_id=1, stones=3, time_limit=None, placements=((2, 0), (3, 0)), sims=8, ms=2.0, server_date=None,
               book_stones=1)
    r.add_move(request_id=2, stones=7, time_limit=None, placements=((-1, 0), (-2, 0)), sims=8, ms=2.0, server_date=None,
               book_stones=0)
    # HeXO x,y for wire q,r: x = q + r, y = -r.
    wire = [(0, 0, "a"), (1, 0, "b"), (0, 1, "b"), (2, 0, "a"), (3, 0, "a"), (1, -1, "b"), (2, -1, "b"),
            (-1, 0, "a"), (-2, 0, "a"), (4, 0, "b"), (5, 0, "b")]
    moves = [{"moveNumber": i + 1, "playerId": p, "x": q + r, "y": -r, "timestamp": i} for i, (q, r, p) in enumerate(wire)]
    body = r.finish(winner="o", reason="six-in-a-row", finished=1.0, finished_game={
        "startedAt": 0, "finishedAt": 1, "moveCount": len(moves), "moves": moves,
        "gameResult": {"reason": "six-in-a-row", "winningPlayerId": "b", "abortedByPlayerId": None}, "players": []})
    return ladder.receipt.write_receipt(tmp_path, body)


def test_a_replay_that_reproduces_every_move_at_the_configured_budget_passes(bot_mod, ladder, tmp_path: Path) -> None:
    report = bot_mod.replay_receipt(ladder.receipt.read_receipt(_receipt(ladder, tmp_path)), _ReplayBackend())
    assert report.passed and report.moves == 2 and report.mismatches == [] and report.budget_misses == []
    assert report.book_misses == [] and report.below_budget == []


def test_the_replay_derives_the_book_stones_from_the_receipts_opening_and_names_a_disagreement(bot_mod, ladder, tmp_path: Path) -> None:
    """R363(c): the forced stones are re-derived from (opening, position), never trusted from the recorded placements."""
    body = ladder.receipt.read_receipt(_receipt(ladder, tmp_path))
    body["opening"]["relative"][3] = [5, 5]
    report = bot_mod.replay_receipt(body, _ReplayBackend())
    assert report.book_misses == [{"request_id": 1, "recorded": [[2, 0]], "derived": [[5, 5]]}]
    assert not report.passed
    body["opening"]["relative"][3] = [2, 0]
    body["moves"][0]["book_stones"] = 0
    report = bot_mod.replay_receipt(body, _ReplayBackend())
    assert report.book_misses == [{"request_id": 1, "recorded": [], "derived": [[2, 0]]}]


def test_a_replay_that_diverges_names_the_move(bot_mod, ladder, tmp_path: Path) -> None:
    report = bot_mod.replay_receipt(ladder.receipt.read_receipt(_receipt(ladder, tmp_path)), _ReplayBackend(flip=True))
    assert not report.passed
    assert report.mismatches == [{"request_id": 2, "recorded": [[-1, 0], [-2, 0]], "replayed": [[-2, 0], [-1, 0]]}]


def test_a_replay_whose_backend_spends_a_different_budget_than_recorded_fails(bot_mod, ladder, tmp_path: Path) -> None:
    report = bot_mod.replay_receipt(ladder.receipt.read_receipt(_receipt(ladder, tmp_path)), _ReplayBackend(sims_per_turn=7))
    assert not report.passed and report.mismatches == []
    assert report.budget_misses == [{"request_id": 1, "sims": 7, "recorded": 8}, {"request_id": 2, "sims": 7, "recorded": 8}]


def test_a_turn_recorded_below_the_configured_budget_is_reported_and_must_replay_exactly(bot_mod, ladder, tmp_path: Path) -> None:
    """Both heads stop early on a DECIDED position (measured live 2026-09-19: strix's solver 3 of 512, mantis's PUCT tree 427 of 512 once every path hit a terminal) — reported, and the replay must reproduce it."""
    body = ladder.receipt.read_receipt(_receipt(ladder, tmp_path))
    body["moves"][1]["sims"] = 3
    report = bot_mod.replay_receipt(body, _ReplayBackend())  # the replay spends 8 where 3 was recorded
    assert report.budget_misses == [{"request_id": 2, "sims": 8, "recorded": 3}] and not report.passed
    assert report.below_budget == [{"request_id": 2, "sims": 3, "configured": 8}]

    class _Short(_ReplayBackend):
        def select_turn(self, board, forced=()):
            turn = super().select_turn(board, forced)
            return turn if len(board.get_stones()) != 7 else type(turn)(placements=turn.placements, sims=3, ms=turn.ms)

    report = bot_mod.replay_receipt(body, _Short())
    assert report.passed and report.budget_misses == []
    assert report.below_budget == [{"request_id": 2, "sims": 3, "configured": 8}]


def test_a_receipt_without_the_servers_move_list_cannot_be_replayed_and_says_so(bot_mod, ladder, tmp_path: Path) -> None:
    body = ladder.receipt.read_receipt(_receipt(ladder, tmp_path))
    body["moves_full"] = None
    with pytest.raises(ValueError, match="moves_full"):
        bot_mod.replay_receipt(body, _ReplayBackend())
