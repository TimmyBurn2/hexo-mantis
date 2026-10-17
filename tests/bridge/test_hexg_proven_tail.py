"""The HEXG v5 proven-tail flag through the engine face: old goldens sample byte-equal, the flag, its refusal."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mantis._engine import HexgBuffer

from test_hexg_root_value import _digest, _sampled_arrays

_REPLAY = Path(__file__).resolve().parents[1] / "fixtures" / "replay"
_ENC = "gnn_axis_v1"
_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]


@pytest.mark.parametrize("fixture", ["hexg_v4_golden.hexg", "hexg_v5_golden.hexg"])
def test_the_v4_and_v5_goldens_sample_byte_equal_to_the_v2_tree(fixture: str) -> None:
    """The same two rows under each layout: every pre-v3 array is the v2 tree's, and v5 flags record 1 alone."""
    golden = json.loads((_REPLAY / "hexg_v2_sample_golden.json").read_text(encoding="utf-8"))
    buf = HexgBuffer(16, _ENC, 128)
    assert buf.load_from_path(str(_REPLAY / fixture)) == 2
    arrays, targets = _sampled_arrays(buf, 20261001, 8)
    moved = [k for k, v in arrays.items() if _digest(v) != golden["arrays"][k]]
    assert moved == [], f"{fixture}: arrays no longer byte-equal to the v2 tree: {moved}"
    flags, outcomes = np.asarray(targets.proven_tail), np.asarray(targets.outcomes)
    assert flags.dtype == np.uint8 and set(outcomes.tolist()) == {1.0, -1.0}, "both records drawn"
    want = [0] * 8 if fixture == "hexg_v4_golden.hexg" else [int(o == -1.0) for o in outcomes]
    assert flags.tolist() == want, "record 1 (outcome -1) is the v5 golden's one flagged row"


def _flags(buf: HexgBuffer) -> list[int]:
    _wire, targets = buf.sample_graph_batch(1, augment=False, n_threads=1)
    return np.asarray(targets.proven_tail).tolist()


def test_a_pushed_flag_rides_its_row_and_a_row_pushed_without_it_samples_unflagged() -> None:
    flagged, plain = HexgBuffer(4, _ENC, 128), HexgBuffer(4, _ENC, 128)
    flagged.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, False, 1.0, True, 7, game_id=0, proven_tail=True)
    plain.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, False, 1.0, True, 7, game_id=0)
    assert (_flags(flagged), _flags(plain)) == ([1], [0])


def test_a_flagged_row_claiming_a_full_search_is_refused_by_name() -> None:
    buf = HexgBuffer(4, _ENC, 128)
    with pytest.raises(ValueError, match="proven_tail"):
        buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, True, 1.0, True, 7, game_id=0, proven_tail=True)
    assert buf.size == 0


def test_the_flag_survives_a_v5_save_and_load(tmp_path: Path) -> None:
    """A flagged quick row (outcome +1) and an unflagged full-search row (outcome -1) re-load with their flags."""
    buf = HexgBuffer(4, _ENC, 128)
    buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, False, 1.0, True, 7, game_id=0, proven_tail=True)
    buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, True, -1.0, True, 7, game_id=1)
    path = tmp_path / "ring.hexg"
    buf.save_to_path(str(path))
    assert path.read_bytes()[4:8] == (5).to_bytes(4, "little")
    back = HexgBuffer(4, _ENC, 128)
    assert back.load_from_path(str(path)) == 2
    back.seed_sampler(3)
    _wire, targets = back.sample_graph_batch(16, augment=False, n_threads=1)
    outcomes = np.asarray(targets.outcomes)
    assert set(outcomes.tolist()) == {1.0, -1.0}
    assert np.asarray(targets.proven_tail).tolist() == [int(o == 1.0) for o in outcomes]
