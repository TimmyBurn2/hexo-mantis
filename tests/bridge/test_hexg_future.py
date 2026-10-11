"""The HEXG v4 future-occupancy field through the engine face: old goldens sample byte-equal, labels, refusals."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mantis._engine import HexgBuffer

from test_hexg_root_value import _digest, _sampled_arrays

_REPLAY = Path(__file__).resolve().parents[1] / "fixtures" / "replay"
_ENC = "gnn_axis_v1"
_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
#: The v4 golden's record 0 field: its game's eleven remaining stones (ply 3, a 7-turn game).
_ELEVEN = [(2, 0, 1), (0, 2, 1), (-1, 0, -1), (1, 1, -1), (2, -1, 1), (-1, 2, 1), (3, 0, -1), (0, 3, -1),
           (-2, 1, 1), (1, -2, 1), (2, 2, -1)]


@pytest.mark.parametrize("fixture", ["hexg_v3_golden.hexg", "hexg_v4_golden.hexg"])
def test_the_v3_and_v4_goldens_sample_byte_equal_to_the_v2_tree(fixture: str) -> None:
    """The same two rows under each later layout: every pre-v3 array of a seeded sample is the v2 tree's."""
    golden = json.loads((_REPLAY / "hexg_v2_sample_golden.json").read_text(encoding="utf-8"))
    buf = HexgBuffer(16, _ENC, 128)
    assert buf.load_from_path(str(_REPLAY / fixture)) == 2
    arrays, targets = _sampled_arrays(buf, 20261001, 8)
    moved = [k for k, v in arrays.items() if _digest(v) != golden["arrays"][k]]
    assert moved == [], f"{fixture}: arrays no longer byte-equal to the v2 tree: {moved}"
    valid = np.asarray(targets.future_valid)
    label = np.asarray(targets.future_target)
    assert label.shape == np.asarray(targets.policy_target).shape, "one label per legal node"
    if fixture == "hexg_v3_golden.hexg":
        assert valid.tolist() == [0] * 8 and not label.any(), "a v3 row carries no field and labels nothing"
    else:
        outcomes = np.asarray(targets.outcomes)
        assert valid.tolist() == [int(o == 1.0) for o in outcomes], "record 0 (outcome +1) alone carries it"


def _labels(buf: HexgBuffer) -> tuple[np.ndarray, np.ndarray]:
    _wire, targets = buf.sample_graph_batch(1, augment=False, n_threads=1)
    return np.asarray(targets.future_target), np.asarray(targets.future_valid)


def test_a_pushed_field_labels_the_movers_stones_one_and_the_opponents_two() -> None:
    """The eleven stones in the mover's frame: six of the mover's (+1), five of the opponent's, the rest empty."""
    for mover, mine, theirs in ((1, 6, 5), (-1, 5, 6)):
        buf = HexgBuffer(4, _ENC, 128)
        buf.push_graph_position(_STONES, [(2, 0, 1.0)], mover, 2, 3, True, 1.0, True, 7, game_id=0,
                                future_stones=_ELEVEN)
        label, valid = _labels(buf)
        assert valid.tolist() == [1]
        assert (int((label == 1).sum()), int((label == 2).sum())) == (mine, theirs)
        assert set(np.unique(label).tolist()) <= {0, 1, 2}


def test_a_row_pushed_without_the_field_samples_invalid_and_empty() -> None:
    buf = HexgBuffer(4, _ENC, 128)
    buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, True, 1.0, True, 7, game_id=0)
    label, valid = _labels(buf)
    assert valid.tolist() == [0] and not label.any()


@pytest.mark.parametrize(("future", "game_length"), [
    (_ELEVEN + [(5, 5, 1)] * 10, 40),  # over the horizon
    (_ELEVEN[:4] + [(1, 0, 1)], 4),     # a stone on one the row holds
    (_ELEVEN, 9),                       # short of a longer game
])
def test_a_field_that_is_not_its_games_next_placements_is_refused_by_name(future: list[Any],
                                                                          game_length: int) -> None:
    buf = HexgBuffer(4, _ENC, 128)
    with pytest.raises(ValueError, match="future-occupancy"):
        buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, True, 1.0, True, game_length, game_id=0,
                                future_stones=future)
    assert buf.size == 0
