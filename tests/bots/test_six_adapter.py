"""The Six rung adapter: the protocol parsed, the position sent in order, forfeits counted, the pin and provider refused."""
# >300 justify (R8): one adapter pinned three ways — its protocol parser, the bot over a recording double, the bot over
# a fake engine process; split, a refusal's test would drift from the move path it guards.
from __future__ import annotations

import hashlib
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from _six_fake import fake_vendor

from mantis._engine import Board
from mantis.arena.adjudicate import TERMINAL_FORFEIT
from mantis.arena.match import _play_one_game
from mantis.bots.protocol import BotProtocol, RungUnresolvable
from mantis.bots.random_bot import RandomBot
from mantis.bots.six import (
    CLOSE_LOG_MARKER,
    FINDING_LOG_MARKER,
    PROVIDER_MARKER,
    SHA_MISMATCH_MARKER,
    VENDOR_ABSENT_MARKER,
    SixBot,
    SixEngine,
    SixSyncError,
    locate_six,
    parse_reply,
    provider_of,
    resolve_six,
    six_availability,
)
from mantis.util.hashing import sha256_file

_ENCODING = "gnn_axis_r8"
_OPENING = [(0, 0), (1, 0), (0, 1), (2, -1)]  # X single, O pair, X's first: X to move, 1 left
_FALLBACK = ["CUDA is not available: libcudnn.so.9: cannot open shared object file", "using the CPU"]


def _board_after(moves: Sequence[tuple[int, int]]) -> Any:
    board = Board.with_encoding_name(_ENCODING)
    for q, r in moves:
        board.apply_move(q, r)
    return board


class _Engine:
    """A recording engine double: each `search` answers the next scripted reply's lines."""

    def __init__(self, replies: list[list[str]], provider: str = "cuda") -> None:
        self.replies = list(replies)
        self.searches: list[tuple[list[tuple[int, int]], int, int]] = []
        self.new_games = 0
        self.provider = provider
        self.closed = False

    def new_game(self) -> None:
        self.new_games += 1

    def search(self, moves: Sequence[tuple[int, int]], *, radius: int, nodes: int) -> list[str]:
        self.searches.append((list(moves), radius, nodes))
        return self.replies.pop(0)

    def close(self) -> None:
        self.closed = True


def _bot(replies: list[list[str]], nodes: int = 16) -> tuple[SixBot, _Engine]:
    engine = _Engine(replies)
    return SixBot(engine, name="six_test", nodes=nodes), engine


def _observed(bot: SixBot, moves: Sequence[tuple[int, int]]) -> Any:
    bot.new_game()
    for q, r in moves:
        bot.observe_move(q, r)
    return _board_after(moves)



def test_a_two_stone_bestmove_parses_in_order() -> None:
    reply = parse_reply(["info depth 1 score 3 nodes 12 time 4 pv 1 2 3 4", "bestmove 1 2 -3 4"])
    assert reply.stones == ((1, 2), (-3, 4)) and reply.failure is None


def test_a_one_stone_bestmove_is_a_whole_answer() -> None:
    assert parse_reply(["bestmove 5 -5"]).stones == ((5, -5),)


@pytest.mark.parametrize("lines, why", [
    (["bestmove none"], "bestmove none"),
    (["info string search failed: x; playing next to the newest stones", "bestmove 1 1 2 2"], "search failed"),
    (["error illegal move 3", "bestmove 1 1"], "error illegal move 3"),
])
def test_a_reply_that_is_not_the_engines_own_search_is_a_failure(lines: list[str], why: str) -> None:
    reply = parse_reply(lines)
    assert reply.failure is not None and why in reply.failure


@pytest.mark.parametrize("lines", [["bestmove 1 2 3"], ["bestmove a b"], ["info depth 1"], ["bestmove --1 2"],
                                   ["bestmove 99999999999 0"]])
def test_a_malformed_answer_is_a_failure_not_a_crash(lines: list[str]) -> None:
    reply = parse_reply(lines)
    assert reply.failure is not None and reply.stones == ()


@pytest.mark.parametrize("stderr, device, provider", [
    (["2026 [W:onnxruntime] shape ops on CPU"], "cuda", "cuda"),
    (_FALLBACK, "cuda", "cpu"),
    (["CUDA is not available: x", "using WebGPU (Vulkan)"], "cuda", "webgpu"),
    ([], "cpu", "cpu"),
])
def test_the_provider_is_read_off_the_engines_stderr(stderr: list[str], device: str, provider: str) -> None:
    assert provider_of(stderr, device) == provider



def test_the_bot_satisfies_the_protocol_and_observes() -> None:
    bot, _ = _bot([])
    assert isinstance(bot, BotProtocol) and bot.name() == "six_test" and callable(bot.observe_move)


def test_the_position_goes_in_observed_order_at_the_boards_radius_and_the_turn_is_played_whole() -> None:
    opening = _OPENING[:3]  # X single, O pair: X to move with 2 left
    bot, engine = _bot([["info depth 1 nodes 12", "bestmove 3 3 4 4"]])
    board = _observed(bot, opening)
    assert bot.select_move(board) == (3, 3)
    assert engine.searches == [(opening, board.legal_move_radius(), 16)]
    bot.observe_move(3, 3)
    board.apply_move(3, 3)
    assert bot.select_move(board) == (4, 4), "the turn's second stone comes from the same search"
    assert len(engine.searches) == 1 and bot.stale_pending == 0


def test_the_second_stone_waits_for_exactly_the_board_it_was_chosen_for() -> None:
    """MUTATION THAT REDS IT: dropping the stone-count guard, so a later board with one stone left replays the stale stone."""
    bot, engine = _bot([["bestmove 3 3 4 4"], ["bestmove 5 5"]])
    assert bot.select_move(_observed(bot, _OPENING[:3])) == (3, 3)
    later = [*_OPENING[:3], (3, 3), (-3, 3), (-2, 2)]  # the cached stone's board has passed: 6 stones, 1 left
    for q, r in later[3:]:
        bot.observe_move(q, r)
    assert bot.select_move(_board_after(later)) == (5, 5), "the cached stone was chosen for 4 stones, not 6"
    assert len(engine.searches) == 2 and bot.stale_pending == 1


def test_new_game_resets_the_log_and_tells_the_engine() -> None:
    bot, engine = _bot([["bestmove 3 3 4 4"], ["bestmove 5 5 6 6"]])
    bot.select_move(_observed(bot, _OPENING[:3]))
    board = _observed(bot, _OPENING[:3])
    bot.select_move(board)
    assert engine.new_games == 2 and engine.searches[1][0] == _OPENING[:3]


def test_a_log_that_is_not_the_boards_stones_is_refused() -> None:
    bot, _ = _bot([["bestmove 3 3"]])
    bot.new_game()
    bot.observe_move(0, 0)
    with pytest.raises(SixSyncError):
        bot.select_move(_board_after(_OPENING[:3]))


def test_a_failed_search_forfeits_through_the_arena_and_is_counted(caplog: pytest.LogCaptureFixture) -> None:
    bot, _ = _bot([["info string search failed: boom; playing next to the newest stones", "bestmove 9 9 9 8"]])
    caplog.set_level(logging.WARNING)
    winner, _plies, moves, terminal, *_ = _play_one_game(
        RandomBot(seed=1), bot, _OPENING[:3], candidate_color=-1, board_factory=lambda: Board.with_encoding_name(_ENCODING),
        max_plies=20, opening_id="o", adjudicator=None)
    assert terminal == TERMINAL_FORFEIT and winner == "candidate" and list(moves) == _OPENING[:3]
    assert bot.forfeits == {"failed": 1, "illegal": 0}
    assert any(FINDING_LOG_MARKER in r.getMessage() and "search failed" in r.getMessage() for r in caplog.records)


def test_an_illegal_stone_is_returned_unchanged_for_the_arena_to_forfeit() -> None:
    bot, _ = _bot([["bestmove 0 0 7 7"]])  # (0, 0) is occupied
    winner, _plies, _moves, terminal, *_ = _play_one_game(
        RandomBot(seed=1), bot, _OPENING[:3], candidate_color=-1, board_factory=lambda: Board.with_encoding_name(_ENCODING),
        max_plies=20, opening_id="o", adjudicator=None)
    assert terminal == TERMINAL_FORFEIT and winner == "candidate"
    assert bot.forfeits == {"failed": 0, "illegal": 1} and bot.findings



def test_the_located_assets_are_the_pins_engine_and_network(tmp_path: Path) -> None:
    root = fake_vendor(tmp_path)
    assets = locate_six(root, "gen0030")
    assert assets.engine.name == "sixengine" and assets.net.name == "gen-0030.onnx"
    assert assets.net_sha256 == sha256_file(assets.net) and assets.commit == "a" * 40


def test_a_planted_wrong_network_hash_is_REFUSED(tmp_path: Path) -> None:
    """MUTATION THAT REDS IT: `locate_six` returning the net without comparing its sha256 to the pin's."""
    root = fake_vendor(tmp_path, net_sha="0" * 64)
    with pytest.raises(RungUnresolvable) as exc:
        locate_six(root, "gen0030")
    assert SHA_MISMATCH_MARKER in exc.value.reason and "gen-0030.onnx" in exc.value.reason


def test_a_pinned_runtime_is_re_hashed_at_load(tmp_path: Path) -> None:
    runtime = hashlib.sha256(b"a runtime").hexdigest()
    assert locate_six(fake_vendor(tmp_path / "a", runtime_sha=runtime), "gen0030").runtime_sha256 == runtime
    with pytest.raises(RungUnresolvable, match="libonnxruntime.so.1"):
        locate_six(fake_vendor(tmp_path / "b", runtime_sha="0" * 64), "gen0030")


def test_an_unfetched_asset_names_make_vendor_six(tmp_path: Path) -> None:
    root = fake_vendor(tmp_path)
    (root / "external" / "six-assets" / "gen-0030.onnx").unlink()
    with pytest.raises(RungUnresolvable) as exc:
        locate_six(root, "gen0030")
    assert VENDOR_ABSENT_MARKER in exc.value.reason and "make vendor.six" in exc.value.reason


def test_a_network_the_pin_does_not_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RungUnresolvable, match="gen0030"):
        locate_six(fake_vendor(tmp_path), "gen9999")


def test_the_engine_process_gets_the_net_absolute_the_cache_off_and_reports_its_provider(tmp_path: Path) -> None:
    assets = locate_six(fake_vendor(tmp_path, stderr=_FALLBACK), "gen0030")
    engine = SixEngine(assets, device="cuda")
    try:
        assert engine.provider == "cpu"
        lines = engine.search([(0, 0)], radius=8, nodes=16)
        assert lines == ["bestmove none"]
    finally:
        engine.close()
    received = (assets.engine.parent / "received.txt").read_text(encoding="utf-8").splitlines()
    assert received[0] == f"--net {assets.net.resolve()}"
    assert "setoption cacheEntries 0" in received and "position radius 8 moves 0 0" in received
    assert "go nodes 16" in received


def test_the_cpu_device_launches_the_engine_on_the_cpu(tmp_path: Path) -> None:
    assets = locate_six(fake_vendor(tmp_path), "gen0030")
    engine = SixEngine(assets, device="cpu")
    engine.close()
    received = (assets.engine.parent / "received.txt").read_text(encoding="utf-8").splitlines()
    assert received[0] == f"--net {assets.net.resolve()} --cpu" and engine.provider == "cpu"


def test_a_cuda_rung_whose_engine_fell_back_is_REFUSED(tmp_path: Path) -> None:
    factory = resolve_six(opponent_sims=16, variant="gen0030", device="cuda",
                          vendor_root=fake_vendor(tmp_path, stderr=_FALLBACK))
    with pytest.raises(RungUnresolvable) as exc:
        factory()
    assert PROVIDER_MARKER in exc.value.reason and "cpu" in exc.value.reason
    assert "make build.cuda" in exc.value.reason


def test_a_cuda_rung_whose_engine_is_on_cuda_plays(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    factory = resolve_six(opponent_sims=16, variant="gen0030", device="cuda:0", vendor_root=fake_vendor(tmp_path))
    bot = factory()
    try:
        assert isinstance(bot, SixBot) and bot.name() == "six_gen0030_n16" and bot.provider == "cuda"
    finally:
        bot.close()
    assert any("provider=cuda" in r.getMessage() for r in caplog.records)


def test_an_engine_that_cannot_start_is_a_recorded_refusal(tmp_path: Path) -> None:
    factory = resolve_six(opponent_sims=16, variant="gen0030", device="cpu",
                          vendor_root=fake_vendor(tmp_path, stderr=["could not load the network: bad"], dies=True))
    with pytest.raises(RungUnresolvable, match="could not load the network"):
        factory()


def test_an_engine_that_cannot_be_executed_is_a_recorded_refusal(tmp_path: Path) -> None:
    root = fake_vendor(tmp_path)
    engine = root / "external" / "six-assets" / "release" / "engine" / "sixengine"
    engine.chmod(0o644)
    factory = resolve_six(opponent_sims=16, variant="gen0030", device="cpu", vendor_root=root)
    with pytest.raises(RungUnresolvable, match="did not start"):
        factory()


def test_a_failed_search_on_an_empty_board_forfeits_off_the_board() -> None:
    bot, _ = _bot([["bestmove none"]])
    board = _observed(bot, [])
    move = bot.select_move(board)
    assert not board.is_legal(*move) and bot.forfeits == {"failed": 1, "illegal": 0}


def test_the_engine_finds_the_releases_own_runtime_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/planted/inherited")
    assets = locate_six(fake_vendor(tmp_path), "gen0030")
    SixEngine(assets, device="cpu").close()
    env = (assets.engine.parent / "env.txt").read_text(encoding="utf-8").split(":")
    assert env == [str(assets.engine.parent), "/planted/inherited"]


def test_an_unwritable_temp_dir_is_a_recorded_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = resolve_six(opponent_sims=16, variant="gen0030", device="cpu", vendor_root=fake_vendor(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "no-such-dir"))
    with pytest.raises(RungUnresolvable, match="did not start"):
        factory()


def test_an_engine_playing_without_a_network_is_refused(tmp_path: Path) -> None:
    factory = resolve_six(opponent_sims=16, variant="gen0030", device="cpu",
                          vendor_root=fake_vendor(tmp_path, ident="HexBot Baseline"))
    with pytest.raises(RungUnresolvable, match="HexBot Baseline"):
        factory()


def test_the_bot_logs_its_counters_when_it_closes(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    bot, engine = _bot([["bestmove 3 3 4 4"]])
    bot.select_move(_observed(bot, _OPENING[:3]))
    bot.close()
    assert engine.closed
    assert any(CLOSE_LOG_MARKER in r.getMessage() and "searches=1" in r.getMessage() and "stale_pending=0"
               in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("kwargs, why", [
    ({"opponent_sims": None, "device": "cuda"}, "nodes"),
    ({"opponent_sims": 16, "device": None}, "device"),
    ({"opponent_sims": 16, "device": "mps"}, "mps"),
    ({"opponent_sims": 16, "device": "cuda:1"}, "cuda:1"),
])
def test_a_rung_without_nodes_or_a_known_device_is_refused(tmp_path: Path, kwargs: dict, why: str) -> None:
    with pytest.raises(RungUnresolvable, match=why):
        resolve_six(variant="gen0030", vendor_root=fake_vendor(tmp_path), **kwargs)



@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="LOUD SKIP — the Six rung plays on CUDA")
def test_the_vendored_engine_plays_two_games_on_cuda() -> None:
    """The landed rung's smoke: gen 30 at 16 nodes, two games through the arena, no forfeit, CUDA reported."""
    available, why = six_availability("gen0030")
    if not available:
        pytest.skip(f"LOUD SKIP — Six is not vendored here: {why}")
    bot = resolve_six(opponent_sims=16, variant="gen0030", device="cuda")()
    terminals = []
    try:
        for color in (1, -1):
            _w, _p, _m, terminal, *_ = _play_one_game(
                RandomBot(seed=3), bot, _OPENING, candidate_color=color,
                board_factory=lambda: Board.with_encoding_name(_ENCODING), max_plies=80,
                opening_id="smoke", adjudicator=None)
            terminals.append(terminal)
    finally:
        bot.close()
    assert bot.provider == "cuda" and TERMINAL_FORFEIT not in terminals, terminals
    assert bot.forfeits == {"failed": 0, "illegal": 0} and bot.findings == [] and bot.searches > 0
