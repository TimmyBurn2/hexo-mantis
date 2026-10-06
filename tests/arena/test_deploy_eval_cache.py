"""The deploy head's per-game exact evaluation cache: one per player, keyed as self-play's cache keys a leaf, cleared by
the player's new game, a key repeated inside one batch evaluated once."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import mantis.arena.eval_cache as ec
from mantis._engine import Board
from mantis.arena.deploy_head import DeployHeadPlayer
from mantis.arena.eval_cache import GameEvalCache
from mantis.arena.match import _record_one
from mantis.monitor.game_record import eval_record
from mantis.util.puct import PuctConstants


@dataclass(frozen=True)
class _Board:
    stones: tuple[tuple[int, int, int], ...]
    current_player: int = 1
    moves_remaining: int = 2

    def get_stones(self):
        return list(self.stones)


class _Engine:
    """Answers a position with outputs derived from its key, and records every position it is asked to evaluate."""

    def __init__(self) -> None:
        self.submitted: list[list] = []

    @staticmethod
    def positions_of(boards):
        return [(list(b.get_stones()), int(b.current_player), int(b.moves_remaining)) for b in boards]

    @staticmethod
    def leaf_keys(positions):
        return [repr((sorted(map(tuple, s)), p, m)) for s, p, m in positions]

    def infer_positions_ls(self, positions):
        self.submitted.append(list(positions))
        keys = self.leaf_keys(positions)
        dense = [[float(len(k) % 7), 0.5] for k in keys]
        overflow = [[((len(k), 1), 0.25)] for k in keys]
        values = [float(len(k)) / 100.0 for k in keys]
        centers = [(len(k), -len(k)) for k in keys]
        return dense, overflow, values, centers


A = _Board(((0, 0, 1),))
B = _Board(((0, 0, 1), (1, 0, -1)))
A_ONE_LEFT = _Board(((0, 0, 1),), moves_remaining=1)


def test_a_position_seen_earlier_in_the_game_is_served_without_a_second_evaluation():
    engine = _Engine()
    cache = GameEvalCache(engine)
    first = cache.infer_batch_ls([A, B])
    again = cache.infer_batch_ls([B, A])
    assert engine.submitted == [engine.positions_of([A, B])], "the second call evaluated nothing"
    assert again == tuple(list(reversed(col)) for col in first)
    assert cache.counters() == {"calls": 2, "positions": 4, "hits": 2, "served": 2, "in_batch_repeats": 0,
                                "all_hit_calls": 1, "evicted": 0}


def test_a_key_repeated_inside_one_batch_is_evaluated_once():
    engine = _Engine()
    cache = GameEvalCache(engine)
    dense, overflow, values, centers = cache.infer_batch_ls([A, B, A])
    assert engine.submitted == [engine.positions_of([A, B])]
    assert dense[0] == dense[2] and overflow[0] == overflow[2] and values[0] == values[2] and centers[0] == centers[2]
    assert cache.counters()["in_batch_repeats"] == 1


def test_the_key_is_exact_so_the_stones_to_place_split_a_position():
    engine = _Engine()
    cache = GameEvalCache(engine)
    cache.infer_batch_ls([A])
    cache.infer_batch_ls([A_ONE_LEFT])
    assert len(engine.submitted) == 2


def test_a_new_game_starts_empty():
    engine = _Engine()
    cache = GameEvalCache(engine)
    cache.infer_batch_ls([A])
    cache.new_game()
    cache.infer_batch_ls([A])
    assert len(engine.submitted) == 2


def test_a_hit_returns_what_the_evaluation_returned_bit_for_bit():
    engine = _Engine()
    cache = GameEvalCache(engine)
    fresh = engine.infer_positions_ls(engine.positions_of([A, B]))
    engine.submitted.clear()
    cache.infer_batch_ls([A, B])
    assert cache.infer_batch_ls([A, B]) == fresh


def test_the_deploy_head_clears_its_cache_at_a_new_game_and_reports_each_moves_rows():
    engine = _Engine()
    cache = GameEvalCache(engine)
    cache.infer_batch_ls([A])
    head = DeployHeadPlayer(expand_fn=lambda tree, leaves: None, n_sims=8, leaf_batch_size=8, c_visit=50.0,
                            c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16, gumbel_seed=0, tactics=None,
                            puct=PuctConstants(c_puct=1.5, fpu_reduction=0.0, quiescence_enabled=False,
                                               quiescence_blend_2=0.0), eval_cache=cache)
    head.new_game()
    assert cache.infer_batch_ls([A]) is not None and len(engine.submitted) == 2, "the new game emptied the cache"




def test_past_its_cap_the_oldest_entry_goes_and_is_evaluated_again(monkeypatch):
    monkeypatch.setattr(ec, "MAX_ENTRIES", 2)
    engine = _Engine()
    cache = GameEvalCache(engine)
    c = _Board(((0, 0, 1), (5, 5, -1)))
    first = cache.infer_batch_ls([A, B, c])
    assert cache.counters()["evicted"] == 1
    assert cache.infer_batch_ls([B, c]) == tuple(col[1:] for col in first) and len(engine.submitted) == 1
    assert cache.infer_batch_ls([A]) == tuple(col[:1] for col in first) and len(engine.submitted) == 2


def test_an_off_window_half_and_a_centre_come_back_as_the_engine_gave_them():
    class _Wide(_Engine):
        def infer_positions_ls(self, positions):
            dense, _overflow, values, centers = super().infer_positions_ls(positions)
            overflow = [[((-9, 4), 0.125), ((7, -3), 0.0625)] for _ in positions]
            return dense, overflow, values, centers

    engine = _Wide()
    cache = GameEvalCache(engine)
    fresh = cache.infer_batch_ls([A])
    assert cache.infer_batch_ls([A]) == fresh and fresh[1] == [[((-9, 4), 0.125), ((7, -3), 0.0625)]]

class _CachedLine:
    """Plays its cells in order; its cumulative cache rows grow by one hit and one served position a move."""

    def __init__(self, cells: list[tuple[int, int]]) -> None:
        self._cells, self._n = cells, 0
        self.rows = {"calls": 0, "positions": 0, "hits": 0, "served": 0, "in_batch_repeats": 0, "all_hit_calls": 0}

    def new_game(self) -> None:
        self._n = 0

    def search_rows(self) -> dict[str, int]:
        return {f"cache_{k}": v for k, v in self.rows.items()}

    def select_move(self, board):
        for key, by in (("calls", 1), ("positions", 2), ("hits", 1), ("served", 1)):
            self.rows[key] += by
        cell = self._cells[self._n]
        self._n += 1
        return cell


def test_a_game_record_carries_that_games_cache_rows_and_the_eval_record_writes_them():
    candidate = _CachedLine([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)])
    candidate.rows["hits"] = 40  # rows from an earlier game on the same player
    opponent = _CachedLine([(0, 3), (1, 3), (2, 3), (3, 3), (-3, 3), (-4, 3)])
    record = _record_one(candidate, opponent, SimpleNamespace(moves=[], opening_id="t"), 1, regime_key=None,
                         board_factory=Board, max_plies=64, adjudicator=None)
    assert record.candidate_search == {"cache_calls": 6, "cache_positions": 12, "cache_hits": 6, "cache_served": 6,
                                       "cache_in_batch_repeats": 0, "cache_all_hit_calls": 0}, "this game's moves only"
    kw = dict(game_id="g", run_id="r", step=1, channel="external", rung="six", phase="rung", game_index=1,
              moves=[(0, 0)], result="p1", plies=1, termination="win", candidate_color=1, seed=1, served_sims=256)
    assert "candidate_search" not in eval_record(**kw)
    assert eval_record(**kw, candidate_search={"cache_hits": 1})["candidate_search"] == {"cache_hits": 1}
