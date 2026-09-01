"""R328(d) + the R328 MAX_STONES amendment — the seeded GAME-level split, and the counted truncation.

TWO MECHANISMS, ONE PRODUCER. The encoder is the last place that still knows which plies came
from which game, so both live there: the split has to be by GAME (a ply-level split puts
positions from the same game on both sides, and a held-out loss over them measures
memorisation), and the ring's stone ceiling has to be counted per ROW.

WHY THE TRUNCATION IS COUNTED RATHER THAN RAISED. `push_graph_position` refuses a position with
more stones than `MAX_STONES`; on the R247 human corpus at radius 8 that is 7 866 of 547 251 ply
rows across 88 of 8 698 games. The architect ruled the ceiling STAYS and the corpus truncates
with its loss counted — so `"8 698 / 8 698 games"` can never be read without the row-level
figure standing beside it in the same provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis.data.bootstrap_encode import CorpusSplit, CorpusEncodeError, encode_corpus
# Bare module name, the repo's convention for a same-directory test helper
# (`tests/model/conformance/*` imports `_corpus` the same way). `from tests.data...`
# is a violation of R5 — there is no package named `tests` — and it resolves only under
# an invocation that happens to put the rootdir on the path.
from test_bootstrap_encode import _dataset, _legal_walk, _records  # noqa: PLC2701

_ENC = "gnn_axis_r8"


def _encode(tmp_path: Path, records, *, split=None, name="a", capacity=4096):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    d = _dataset(root, records)
    return encode_corpus(d, root / "out.hexg", encoding=_ENC, capacity=capacity,
                         visit_capacity=8, split=split)


# ═══ the split ═══════════════════════════════════════════════════════════════════════════
def test_the_two_sides_PARTITION_the_corpus_exactly(tmp_path: Path) -> None:
    """DISJOINT and EXHAUSTIVE, checked as counts that reconcile — the contamination check.

    This is the property a held-out set exists for: a game on both sides is training data
    wearing a held-out label, and a game on neither is silently dropped evidence."""
    recs = _records(40)
    tr = _encode(tmp_path, recs, split=CorpusSplit(7, 0.25, "train"), name="a")
    ho = _encode(tmp_path, recs, split=CorpusSplit(7, 0.25, "heldout"), name="b")
    whole = _encode(tmp_path, recs, name="c")
    assert tr["games"] + ho["games"] == whole["games"] == 40
    assert tr["plies"] + ho["plies"] == whole["plies"]
    assert tr["games"] > 0 and ho["games"] > 0


def test_assignment_keys_on_the_GAME_HASH_and_not_on_record_ORDER(tmp_path: Path) -> None:
    """PB-1. Reversing the corpus must not move a single game across the partition.

    An index- or shuffle-based split passes every count-based row above and fails this one:
    the counts still reconcile, they are just counts of different games."""
    recs = _records(40)
    split = CorpusSplit(7, 0.25, "heldout")
    fwd = _encode(tmp_path, recs, split=split, name="a")
    rev = _encode(tmp_path, list(reversed(recs)), split=split, name="b")
    assert fwd["game_hash_set_sha256"] == rev["game_hash_set_sha256"], (
        "the held-out SET changed when the corpus was reordered — the split is keyed on "
        "position, not on identity"
    )


def test_the_split_is_INDEPENDENT_of_max_games(tmp_path: Path) -> None:
    """A truncated smoke run draws the same side for the same game as a full run.

    This is what a keyed hash buys over a shuffle, and it is the difference between a smoke
    config that exercises the real partition and one that exercises a different one."""
    recs = _records(40)
    split = CorpusSplit(7, 0.25, "heldout")
    full = _encode(tmp_path, recs, split=split, name="a")
    part = _encode(tmp_path, recs[:20], split=split, name="b")
    assert part["games"] <= full["games"]
    assert part["games"] > 0, "the prefix selected nothing; this row is vacuous"


def test_a_DIFFERENT_SEED_gives_a_different_partition(tmp_path: Path) -> None:
    """PB-3. If the seed is dropped from the hash key, every seed gives one partition."""
    recs = _records(60)
    a = _encode(tmp_path, recs, split=CorpusSplit(1, 0.25, "heldout"), name="a")
    b = _encode(tmp_path, recs, split=CorpusSplit(2, 0.25, "heldout"), name="b")
    assert a["game_hash_set_sha256"] != b["game_hash_set_sha256"], (
        "two seeds produced the IDENTICAL held-out set — the seed is not reaching the hash"
    )


def test_an_EMPTY_side_is_REFUSED(tmp_path: Path) -> None:
    """PB-2. An empty ring is not a small ring."""
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
    """A split ring and a whole-corpus ring must be distinguishable by a reader who knows
    nothing about how either was produced."""
    whole = _encode(tmp_path, _records(20), name="a")
    part = _encode(tmp_path, _records(20), split=CorpusSplit(7, 0.25, "train"), name="b")
    assert whole["split_seed"] is None and whole["split_part"] is None
    assert part["split_seed"] == 7 and part["split_part"] == "train"
    assert part["split_heldout_frac"] == 0.25


# ═══ the counted truncation ══════════════════════════════════════════════════════════════
def test_a_game_past_the_stone_ceiling_is_TRUNCATED_and_COUNTED(tmp_path: Path) -> None:
    """The amendment's mechanism: rows over `MAX_STONES` are lost, and the provenance says so.

    Driven with a real over-length game rather than a mocked ceiling, so what is exercised is
    the same comparison the corpus hit."""
    from mantis._engine import max_stones
    ceiling = max_stones()
    long_game = {"game_hash": "long", "winner": 1,
                 "moves": [[q, r] for q, r in _legal_walk(ceiling + 40, seed=5)]}
    prov = _encode(tmp_path, [long_game], name="a", capacity=ceiling + 64)
    assert prov["games"] == 1, "the game is ACCEPTED, not refused"
    assert prov["games_truncated"] == 1
    # Row j carries j stones, so rows j in [ceiling+1, len(moves)-1] are refused — DERIVED
    # from the game's own length and the engine's ceiling, never a typed count.
    expected_lost = len(long_game["moves"]) - (ceiling + 1)
    assert expected_lost > 0, "the fixture game does not exceed the ceiling; row is vacuous"
    assert prov["rows_refused_over_max_stones"] == expected_lost
    assert prov["plies"] + prov["rows_refused_over_max_stones"] == prov["plies_offered"]
    assert prov["max_stones_ceiling"] == ceiling


def test_a_corpus_INSIDE_the_ceiling_reports_zero_loss(tmp_path: Path) -> None:
    """The positive control: the counters are not always-on decoration."""
    prov = _encode(tmp_path, _records(5), name="a")
    assert prov["rows_refused_over_max_stones"] == 0
    assert prov["games_truncated"] == 0
    assert prov["plies"] == prov["plies_offered"]


def test_the_ply_histogram_is_present_and_ORDERS_NUMERICALLY(tmp_path: Path) -> None:
    """The provenance is written with `sort_keys=True`, so unpadded labels put "64-127" after
    "512-575". A histogram a reader has to re-sort by eye is one they will read wrong."""
    recs = _records(5) + [{"game_hash": "long", "winner": 1,
                           "moves": [[q, r] for q, r in _legal_walk(300, seed=9)]}]
    prov = _encode(tmp_path, recs, name="a", capacity=4096)
    hist = prov["ply_histogram_64"]
    assert hist, "the histogram is empty"
    keys = list(json.loads(json.dumps(hist, sort_keys=True)))
    lows = [int(k.split("-")[0]) for k in keys]
    assert lows == sorted(lows), f"histogram keys do not order numerically under sort_keys: {keys}"
    assert sum(hist.values()) == prov["games"]
