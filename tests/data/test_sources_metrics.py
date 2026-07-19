"""O4d — corpus sources + metrics smoke.

GameRecord / CorpusSource contract; HumanGameSource yields the expected records from a
tiny frozen human-JSON fixture; CorpusMetrics counters; analyse_* / run_analysis return
the expected stat-dict key sets on synthetic (real, replayable) input.
"""
from __future__ import annotations

import json

import pytest
from _frozen_games import FROZEN_GAMES

from mantis.data import corpus_metrics as cm
from mantis.data.pipeline_metrics import CorpusMetrics, SourceMetrics
from mantis.data.sources.base import CorpusSource, GameRecord
from mantis.data.sources.human import HumanGameSource


# --------------------------------------------------------------------------- #
# GameRecord / CorpusSource contract
# --------------------------------------------------------------------------- #
def test_gamerecord_fields() -> None:
    rec = GameRecord(game_id_str="g", moves=[(0, 0), (1, 0)], winner=1, source="human")
    assert rec.metadata == {}
    assert rec.moves == [(0, 0), (1, 0)]


def test_corpussource_is_abstract() -> None:
    with pytest.raises(TypeError):
        CorpusSource()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# HumanGameSource
# --------------------------------------------------------------------------- #
def _valid_human_game() -> dict:
    moves = [{"x": q, "y": r, "anon_player": "p1" if i % 2 == 0 else "p2"}
             for i, (q, r) in enumerate(FROZEN_GAMES["g2"][0])]
    return {
        "gameOptions": {"rated": True},
        "moveCount": len(moves),
        "gameResult": {"reason": "six-in-a-row", "anon_winner": "p1"},
        "moves": moves,
        "players": [
            {"anon_profile_id": "p1", "elo": 1200},
            {"anon_profile_id": "p2", "elo": 1100},
        ],
    }


def test_human_source_yields_valid_record(tmp_path) -> None:
    (tmp_path / "aaa.json").write_text(json.dumps(_valid_human_game()))
    # A game that fails the ingestion filter (unrated) must be skipped.
    bad = _valid_human_game()
    bad["gameOptions"]["rated"] = False
    (tmp_path / "bbb.json").write_text(json.dumps(bad))

    src = HumanGameSource(tmp_path)
    assert src.name() == "human"
    records = list(src)
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "human"
    assert rec.winner == 1  # anon_winner == p1 == moves[0].anon_player
    assert rec.moves == FROZEN_GAMES["g2"][0]
    assert rec.metadata["elo_p1"] == 1200
    assert rec.metadata["elo_p2"] == 1100


def test_human_source_filters(tmp_path) -> None:
    short = _valid_human_game()
    short["moveCount"] = 5
    (tmp_path / "short.json").write_text(json.dumps(short))
    not_win = _valid_human_game()
    not_win["gameResult"]["reason"] = "resignation"
    (tmp_path / "resign.json").write_text(json.dumps(not_win))
    assert list(HumanGameSource(tmp_path)) == []


# --------------------------------------------------------------------------- #
# CorpusMetrics pipeline counters
# --------------------------------------------------------------------------- #
def test_corpus_metrics_counters() -> None:
    m = CorpusMetrics(flush_interval=100)
    m.record_game("human", 22)
    m.record_game("human", 30, colony_bug=True)
    m.record_duplicate("human")
    m.flush()
    summ = m.summary()["human"]
    assert summ == {
        "games_processed": 2,
        "games_duplicated": 1,
        "positions_pushed": 52,
        "colony_bug_games": 1,
        "value_flip_games": 0,
    }
    assert SourceMetrics().positions_per_hour() == 0.0


# --------------------------------------------------------------------------- #
# analyse_* stat-dict key sets
# --------------------------------------------------------------------------- #
def _synthetic_records() -> list[GameRecord]:
    recs = []
    for i, (gid, (moves, winner)) in enumerate(FROZEN_GAMES.items()):
        recs.append(GameRecord(
            game_id_str=gid, moves=list(moves), winner=winner, source="human",
            metadata={"elo_p1": 1200 - 50 * i, "elo_p2": 1100 + 40 * i},
        ))
    return recs


def test_run_analysis_key_sets() -> None:
    recs = _synthetic_records()
    res = cm.run_analysis(recs, "test", cluster_sample=20)
    assert set(res) == {
        "game_count", "total_positions", "game_lengths", "win_rates",
        "move_entropy", "opening_diversity", "cluster_counts", "ply_coverage",
    }
    assert set(res["game_lengths"]) == {
        "median", "mean", "std", "min", "max", "p10_threshold", "p90",
    }
    assert set(res["win_rates"]) == {
        "overall_p1_win_rate", "p1_advantage_flag", "by_elo_band",
        "worst_band_rate", "worst_band_label",
    }
    assert set(res["move_entropy"]) == {
        "mean_entropy_nats", "std_entropy_nats",
        "low_info_games_below_0.5", "low_info_fraction",
    }
    assert {"unique_at_move_3", "unique_at_move_10", "dupe_rate_first_10",
            "first_move_entropy"} <= set(res["opening_diversity"])
    assert set(res["cluster_counts"]) == {
        "median_cluster_count", "mean_cluster_count", "max_cluster_count",
        "frac_k_gt2", "distribution", "sample_size",
    }
    assert set(res["ply_coverage"]) == {
        "total_positions", "late_game_positions", "late_game_fraction",
        "late_game_flag", "ply_histogram",
    }


def test_quality_scores_and_elo_stratified() -> None:
    recs = _synthetic_records()
    entropies = cm._compute_per_game_entropies(recs)
    scores = cm.compute_quality_scores(recs, entropies)
    assert len(scores) == len(recs)
    for row in scores.values():
        assert set(row) == {"source", "elo", "game_length", "mean_entropy", "quality_score"}
    qstats = cm.analyse_quality_distribution(scores)
    assert {"mean_score", "median_score", "mean_per_source",
            "frac_below_0.3", "frac_above_0.7"} == set(qstats)

    elo = cm.analyse_elo_stratified(recs)
    assert set(elo) == set(cm.MANIFEST_BAND_ORDER)
    for band in elo.values():
        assert set(band) == {"game_count", "median_compound_moves", "top_openings"}
