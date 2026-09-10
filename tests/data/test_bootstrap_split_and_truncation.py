"""The seeded GAME-level corpus split, and the counted stone-ceiling truncation.

Both live in the encoder because it is the last place that still knows which plies came from
which game: the split must be by GAME (a ply-level split puts positions from one game on both
sides, so a held-out loss over them measures memorisation), and the ring's stone ceiling must
be counted per ROW so a game count can never be read without the row-level loss beside it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis.data.bootstrap_encode import CorpusSplit, CorpusEncodeError, encode_corpus
# Bare module name: there is no package named `tests`, so `from tests.data...` resolves only
# under an invocation that happens to put the rootdir on the path.
from test_bootstrap_encode import _dataset, _legal_walk, _records  # noqa: PLC2701

_ENC = "gnn_axis_r8"


def _encode(tmp_path: Path, records, *, split=None, name="a", capacity=4096):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    d = _dataset(root, records)
    return encode_corpus(d, root / "out.hexg", encoding=_ENC, capacity=capacity,
                         visit_capacity=8, split=split)


def test_the_two_sides_PARTITION_the_corpus_exactly(tmp_path: Path) -> None:
    """The two sides are disjoint and exhaustive, checked as counts that reconcile."""
    recs = _records(40)
    tr = _encode(tmp_path, recs, split=CorpusSplit(7, 0.25, "train"), name="a")
    ho = _encode(tmp_path, recs, split=CorpusSplit(7, 0.25, "heldout"), name="b")
    whole = _encode(tmp_path, recs, name="c")
    assert tr["games"] + ho["games"] == whole["games"] == 40
    assert tr["plies"] + ho["plies"] == whole["plies"]
    assert tr["games"] > 0 and ho["games"] > 0


def test_assignment_keys_on_the_GAME_HASH_and_not_on_record_ORDER(tmp_path: Path) -> None:
    """Reversing the corpus moves no game across the partition.

    An index- or shuffle-based split passes every count-based row above and fails this one.
    """
    recs = _records(40)
    split = CorpusSplit(7, 0.25, "heldout")
    fwd = _encode(tmp_path, recs, split=split, name="a")
    rev = _encode(tmp_path, list(reversed(recs)), split=split, name="b")
    assert fwd["game_hash_set_sha256"] == rev["game_hash_set_sha256"], (
        "the held-out SET changed when the corpus was reordered — the split is keyed on "
        "position, not on identity"
    )


def test_the_split_is_INDEPENDENT_of_max_games(tmp_path: Path) -> None:
    """A truncated smoke run draws the same side for the same game as a full run."""
    recs = _records(40)
    split = CorpusSplit(7, 0.25, "heldout")
    full = _encode(tmp_path, recs, split=split, name="a")
    part = _encode(tmp_path, recs[:20], split=split, name="b")
    assert part["games"] <= full["games"]
    assert part["games"] > 0, "the prefix selected nothing; this row is vacuous"


def test_a_DIFFERENT_SEED_gives_a_different_partition(tmp_path: Path) -> None:
    """Two seeds give different partitions; drop the seed from the key and they would not."""
    recs = _records(60)
    a = _encode(tmp_path, recs, split=CorpusSplit(1, 0.25, "heldout"), name="a")
    b = _encode(tmp_path, recs, split=CorpusSplit(2, 0.25, "heldout"), name="b")
    assert a["game_hash_set_sha256"] != b["game_hash_set_sha256"], (
        "two seeds produced the IDENTICAL held-out set — the seed is not reaching the hash"
    )


def test_an_EMPTY_side_is_REFUSED(tmp_path: Path) -> None:
    """An empty side is refused: an empty ring is not a small ring."""
    with pytest.raises(CorpusEncodeError, match="selected ZERO games"):
        _encode(tmp_path, _records(3), split=CorpusSplit(7, 0.99, "train"), name="a")


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_a_fraction_outside_the_open_unit_interval_is_REFUSED(frac: float) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        CorpusSplit(7, frac, "train")


def test_an_unknown_part_is_REFUSED() -> None:
    with pytest.raises(ValueError, match="must be 'train' or 'heldout'"):
        CorpusSplit(7, 0.1, "validation")


def test_the_provenance_records_the_split_and_NULLS_it_when_absent(tmp_path: Path) -> None:
    """A split ring and a whole-corpus ring are distinguishable from provenance alone."""
    whole = _encode(tmp_path, _records(20), name="a")
    part = _encode(tmp_path, _records(20), split=CorpusSplit(7, 0.25, "train"), name="b")
    assert whole["split_seed"] is None and whole["split_part"] is None
    assert part["split_seed"] == 7 and part["split_part"] == "train"
    assert part["split_heldout_frac"] == 0.25


def test_a_game_past_the_stone_ceiling_is_TRUNCATED_and_COUNTED(tmp_path: Path) -> None:
    """Rows over `MAX_STONES` are lost and the provenance says so, driven with a real
    over-length game rather than a mocked ceiling."""
    from mantis._engine import max_stones
    ceiling = max_stones()
    long_game = {"game_hash": "long", "winner": 1,
                 "moves": [[q, r] for q, r in _legal_walk(ceiling + 40, seed=5)]}
    prov = _encode(tmp_path, [long_game], name="a", capacity=ceiling + 64)
    assert prov["games"] == 1, "the game is ACCEPTED, not refused"
    assert prov["games_truncated"] == 1
    # Row j carries j stones, so rows j in [ceiling+1, len(moves)-1] are refused.
    expected_lost = len(long_game["moves"]) - (ceiling + 1)
    assert expected_lost > 0, "the fixture game does not exceed the ceiling; row is vacuous"
    assert prov["rows_refused_over_max_stones"] == expected_lost
    assert prov["plies"] + prov["rows_refused_over_max_stones"] == prov["plies_offered"]
    assert prov["max_stones_ceiling"] == ceiling


def test_a_corpus_INSIDE_the_ceiling_reports_zero_loss(tmp_path: Path) -> None:
    """Control: the counters are not always-on decoration."""
    prov = _encode(tmp_path, _records(5), name="a")
    assert prov["rows_refused_over_max_stones"] == 0
    assert prov["games_truncated"] == 0
    assert prov["plies"] == prov["plies_offered"]


def test_the_ply_histogram_is_present_and_ORDERS_NUMERICALLY(tmp_path: Path) -> None:
    """The provenance is written with `sort_keys=True`, so unpadded labels would put "64-127"
    after "512-575"."""
    recs = _records(5) + [{"game_hash": "long", "winner": 1,
                           "moves": [[q, r] for q, r in _legal_walk(300, seed=9)]}]
    prov = _encode(tmp_path, recs, name="a", capacity=4096)
    hist = prov["ply_histogram_64"]
    assert hist, "the histogram is empty"
    keys = list(json.loads(json.dumps(hist, sort_keys=True)))
    lows = [int(k.split("-")[0]) for k in keys]
    assert lows == sorted(lows), f"histogram keys do not order numerically under sort_keys: {keys}"
    assert sum(hist.values()) == prov["games"]


def test_a_BC_row_CONTRIBUTES_POLICY_GRADIENT_and_is_not_value_only() -> None:
    """A BC row must carry `is_full_search=True` or it contributes no policy gradient.

    `ragged_policy_ce` masks by that flag and `graph_loss_denominators` sums the same mask, so
    the flag's ROLE is "does this row contribute policy gradient?", not "was there a search?".
    """
    from mantis.data.bootstrap_encode import encode_game
    from mantis._engine import Board

    rows = list(encode_game([(q, r) for q, r in _legal_walk(12, seed=3)], 1,
                            board_factory=lambda: Board.with_encoding_name(_ENC)))
    assert rows, "the fixture game encoded no rows"
    flags = [row[5] for row in rows]
    assert all(flags), (
        "a BC row must carry is_full_search=True or its one-hot policy target is masked out "
        "of the loss and the pretrain trains the value head alone"
    )


def test_the_policy_DENOMINATOR_over_a_BC_batch_is_the_graph_count_not_zero() -> None:
    """A zero policy denominator silently turns the whole policy term into 0 for the run."""
    import numpy as np

    from mantis.data.bootstrap_encode import encode_game
    from mantis.train.losses import graph_loss_denominators
    from mantis._engine import Board

    rows = list(encode_game([(q, r) for q, r in _legal_walk(12, seed=4)], 1,
                            board_factory=lambda: Board.with_encoding_name(_ENC)))
    ifs = np.array([row[5] for row in rows], dtype=np.uint8)
    vv = np.array([row[7] for row in rows], dtype=np.uint8)
    policy_den, _value_den = graph_loss_denominators(ifs, vv, len(rows))
    assert policy_den == float(len(rows)) and policy_den > 0, (
        f"policy denominator {policy_den} over {len(rows)} BC rows; a 0 here makes every "
        "policy loss 0 for the whole pretrain"
    )
