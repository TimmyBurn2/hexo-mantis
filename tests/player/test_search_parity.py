"""Search parity (W3, in the protected set): the Rust player and the Python deploy head, fed one stub's outputs, play the same stones and count the same rows on the goldens at concurrency 1."""
from __future__ import annotations

import copy
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import mantis.arena.deploy_head as head_mod
import mantis.selfplay.inference_server as server_mod
from mantis import _engine
from mantis._engine import Board, Player
from mantis.arena.eval_cache import MAX_ENTRIES
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
from mantis.config.resolve.inference_batching import resolve_inference_batching
from mantis.config.resolve.puct import resolve_puct_constants
from mantis.config.resolve.tactics import resolve_deploy_tactics
from mantis.encoding import lookup
from mantis.eval.worker import build_candidate_player
from mantis.model import ARCH_KINDS, build_net
from mantis.selfplay.inference_local import LocalInferenceEngine
from mantis.selfplay.served_graphs import SMALL_POP_FLOOR_NODES
from mantis.util.loadpkg import load_tools_package

REPO = Path(__file__).resolve().parents[2]
GOLDENS = REPO / "tests" / "fixtures" / "graph_build" / "positions_r8.txt"
ENC = "gnn_axis_r8"
SIMS = 128
FEAT = 11
TURNS = 3
# P1 holds four in a row, open at both ends, with two stones to play: the root solver decides both stones unsearched.
FORCED_WIN = [(0, 0), (0, 5), (5, -5), (1, 0), (2, 0), (-5, 0), (-3, 5), (3, 0), (0, -3), (5, 3), (-2, -4)]

pytestmark = pytest.mark.xdist_group("search_parity")
CAPTURED: list[list[np.ndarray]] = []


def _config() -> dict[str, Any]:
    """A production config of the census on the player's row with its deploy tactics armed: the shipped posture."""
    dumps = [load_config(path).model_dump() for path in production_configs(REPO)]
    return copy.deepcopy(next(d for d in dumps
                              if d["identity"]["encoding"] == ENC and resolve_deploy_tactics(d) is not None))


def _golden_moves(n: int) -> list[list[tuple[int, int]]]:
    """Each golden position's move line, in the order its stones were played."""
    out = []
    for line in GOLDENS.read_text(encoding="utf-8").splitlines():
        if line.startswith("M "):
            v = [int(f) for f in line.split()[1:]]
            out.append([(v[1 + 2 * i], v[2 + 2 * i]) for i in range(v[0])])
            if len(out) == n:
                break
    return out


def stub_outputs(graphs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """One net for both sides: each legal row's prior from a hash of its features, each graph's value from their sum."""
    probs, values = [], []
    for rows in graphs:
        c = np.array([1 + zlib.crc32(np.ascontiguousarray(row).tobytes()) % 97 for row in rows], dtype=np.float32)
        total = c.sum(dtype=np.float32)
        probs.append(c / total)
        values.append(((int(total) % 2001) / 1000.0 - 1.0) * 0.5)
    return np.concatenate(probs), np.asarray(values, dtype=np.float32)


def graph_batch_rows(batch: Any) -> list[np.ndarray]:
    """The head's adapter: a GraphBatch's legal rows' features, per graph."""
    x, legal, offsets = batch.x.numpy(), batch.legal_node_gather.numpy(), batch.legal_offsets.numpy()
    return [x[legal[lo:hi]] for lo, hi in zip(offsets[:-1], offsets[1:], strict=True)]


def web_batch_rows(step: dict[str, Any]) -> list[np.ndarray]:
    """The player's adapter: a WebBatch's legal rows' features, per graph."""
    x, legal, graphs = step["x"].reshape(-1, FEAT), step["legal"], step["legal_graph"]
    return [x[legal[graphs == g]] for g in range(step["n_graphs"])]


@pytest.fixture(scope="module")
def stubbed() -> Iterator[None]:
    """The deploy path's served outputs replaced by the stub, read off the GraphBatch it is handed."""
    served = server_mod.served_outputs

    def fake(_model: Any, batch: Any, _vocab: Any, *, trunk: Any, amp_dtype: Any) -> tuple[torch.Tensor, torch.Tensor]:
        rows = graph_batch_rows(batch)
        CAPTURED.append(rows)
        probs, values = stub_outputs(rows)
        return torch.from_numpy(probs), torch.from_numpy(values)

    server_mod.served_outputs = fake
    try:
        yield
    finally:
        server_mod.served_outputs = served


@pytest.fixture(scope="module")
def engine(stubbed: None) -> Iterator[LocalInferenceEngine]:
    config, spec = _config(), lookup(ENC)
    arch = ARCH_KINDS[config["identity"]["arch_kind"]](in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, hidden=8,
                                                       num_layers=1, policy_hidden=8, value_hidden=8)
    engine = LocalInferenceEngine(
        build_net(arch), torch.device("cpu"), encoding_spec=spec, fused_graph_caps=resolve_fused_graph_caps(config),
        inference_batching=resolve_inference_batching(config), max_in_flight=int(config["selfplay"]["leaf_batch_size"]),
        submitters=1, leaf_build_threads=1, bucket_floor_nodes=SMALL_POP_FLOOR_NODES)
    try:
        yield engine
    finally:
        engine.close()


def _head(engine: LocalInferenceEngine, *, early_stop: bool = True, tree_reuse: bool = True) -> Any:
    config = _config()
    sp = config["selfplay"]
    return build_candidate_player(
        engine, SIMS, spec=lookup(ENC), leaf_batch_size=int(sp["leaf_batch_size"]), c_visit=sp["c_visit"],
        c_scale=sp["c_scale"], q_rescale=sp["q_rescale"], search_kind="puct", gumbel_m=sp["gumbel_m"], gumbel_seed=0,
        tactics=resolve_deploy_tactics(config), puct=resolve_puct_constants(config), early_stop=early_stop,
        tree_reuse=tree_reuse)


def _player() -> Player:
    config = _config()
    deploy = {"search_kind": "puct", **resolve_puct_constants(config).tree_kwargs(),
              "leaf_batch_size": int(config["selfplay"]["leaf_batch_size"]), "tactics": resolve_deploy_tactics(config)}
    return Player(deploy, ENC, cpu_path=True)


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {k: after[k] - before.get(k, 0) for k in after}


def _board(moves: list[tuple[int, int]]) -> Any:
    board = Board.with_encoding_name(ENC)
    for q, r in moves:
        board.apply_move(q, r)
    return board


Turn = tuple[list[tuple], dict[str, int]]


def head_turns(head: Any, games: list[list[tuple[int, int]]]) -> list[list[Turn]]:
    """Per game: up to `TURNS` turns to a win, the head on both sides; each turn's stones and its rows over the turn."""
    out = []
    for moves in games:
        head.new_game()
        board, turns = _board(moves), []
        while len(turns) < TURNS and not board.check_win():
            before, stones = head.search_rows(), []
            for _ in range(board.moves_remaining):
                cell = head.select_move(board)
                stones.append((tuple(cell), int(head.last_sims or 0), bool(head.last_stopped), head.last_tactics))
                board.apply_move(*cell)
                if board.check_win():
                    break
            turns.append((stones, _delta(head.search_rows(), before)))
        out.append(turns)
    return out


def player_turns(player: Player, games: list[list[tuple[int, int]]]) -> tuple[list[list[Turn]], int]:
    """The same from the Rust player through its pull/push, and how many of its stones the root solver decided."""
    out, decided = [], 0
    for moves in games:
        player.new_game(moves, sims=SIMS)
        board, turns = _board(moves), []
        while len(turns) < TURNS and not board.check_win():
            before = player.rows()
            player.begin_turn(0.0, sims=SIMS)
            while "turn" not in (step := player.next_batch(0.0)):
                player.submit(*stub_outputs(web_batch_rows(step)), 0.0)
            for s in step["turn"]:
                board.apply_move(*s["cell"])
                decided += s["decided"]
            stones = [(tuple(s["cell"]), s["sims"], s["stopped"], s["tactics"]) for s in step["turn"]]
            turns.append((stones, _delta(player.rows(), before)))
        out.append(turns)
    return out, decided


def _mismatches(head: list[list[Turn]], player: list[list[Turn]]) -> list[int]:
    return [i for i, (h, p) in enumerate(zip(head, player, strict=True))
            if len(h) != len(p) or any(ht[0] != pt[0] or any(pt[1].get(k) != v for k, v in ht[1].items())
                                       for ht, pt in zip(h, p, strict=False))]


@pytest.fixture(scope="module")
def default_games() -> list[list[tuple[int, int]]]:
    return [FORCED_WIN, *_golden_moves(32)]


@pytest.fixture(scope="module")
def played(engine: LocalInferenceEngine, default_games: list) -> tuple[list, list, int]:
    return head_turns(_head(engine), default_games), *player_turns(_player(), default_games)


def _witnessed(head: list[list[Turn]], player: list[list[Turn]], decided: int) -> None:
    """Identical games, and every arm W3 holds the player to actually ran: reuse, the cache's hits, a decided stone."""
    assert _mismatches(head, player) == [], "the player and the head part on these games"
    arms = ("reuse_reroots", "cache_hits", "cache_all_hit_calls")  # evictions need 8 192 entries: cache.rs's own test
    total = {k: sum(rows[k] for turns in head for _, rows in turns) for k in arms}
    assert all(total[k] >= 1 for k in arms), f"an arm went unread: {total}"
    assert decided >= 1, "no stone was decided by the root solver, so that arm went unread"


def test_the_player_plays_the_heads_stones_and_counts_its_rows(played: tuple[list, list, int]) -> None:
    _witnessed(*played)


def test_the_two_adapters_give_the_same_rows_on_the_goldens(engine: LocalInferenceEngine) -> None:
    """The head's GraphBatch and the player's WebBatch of the same goldens hand the stub the same rows, graph by graph."""
    positions = []
    for moves in _golden_moves(16):
        board = Board.with_encoding_name(ENC)
        for q, r in moves:
            board.apply_move(q, r)
        positions.append((board.get_stones(), 1 if board.current_player == 1 else -1, board.moves_remaining))
    CAPTURED.clear()
    engine.infer_positions_ls(positions)
    head_rows = [rows for call in CAPTURED for rows in call]
    web_rows = web_batch_rows({**_engine.web_batch(positions, ENC)})
    assert len(head_rows) == len(web_rows) == len(positions)
    assert all(np.array_equal(h, w) for h, w in zip(head_rows, web_rows, strict=True))


def test_a_head_without_reuse_or_stopping_at_a_tie_is_told_apart(engine: LocalInferenceEngine,
                                                                 played: tuple[list, list, int], default_games: list,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAKS: a head that re-roots by a new game, and one whose stop compares `>= left`, each part from the player."""
    _head_rows, player, _decided = played
    assert _mismatches(head_turns(_head(engine, tree_reuse=False), default_games), player), "reuse went unread"
    monkeypatch.setattr(head_mod, "_leader_fixed", lambda tree, left: (top := tree.get_top_visits(2)) and (
        len(top) == 1 or top[0][1] - top[1][1] >= left))
    assert _mismatches(head_turns(_head(engine), default_games), player), "the stop's tie went unread"


@pytest.mark.integration
def test_the_player_plays_the_heads_stones_on_every_golden(engine: LocalInferenceEngine) -> None:
    games = [FORCED_WIN, *_golden_moves(512)]
    _witnessed(head_turns(_head(engine), games), *player_turns(_player(), games))


def test_the_players_constants_are_the_ladders_and_the_caches() -> None:
    """The player is the second consumer of the early stop and reuse: its switches are the ladder's, its cache the head's."""
    ladder = load_tools_package("ladder")
    backends = __import__(f"{ladder.__name__}.backends", fromlist=["LADDER_EARLY_STOP"])
    constants = _engine.player_constants()
    constants.pop("virtual_loss")  # the search crate's own constant, read by both trees; no Python source to pin it to
    assert constants == {"early_stop": backends.LADDER_EARLY_STOP, "tree_reuse": backends.LADDER_TREE_REUSE,
                         "cache_entries": MAX_ENTRIES}
