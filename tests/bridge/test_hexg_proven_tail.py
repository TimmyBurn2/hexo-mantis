"""The HEXG v5 proven-tail flag through the engine face: old goldens sample byte-equal, the flag in the saved ring, its refusal."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis._engine import HexgBuffer
from mantis.diagnostics.ring_reader import load_ring

from test_hexg_root_value import _digest, _sampled_arrays

_REPLAY = Path(__file__).resolve().parents[1] / "fixtures" / "replay"
_ENC = "gnn_axis_v1"
_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]


@pytest.mark.parametrize(("fixture", "flags"), [("hexg_v4_golden.hexg", [0, 0]), ("hexg_v5_golden.hexg", [0, 1])])
def test_the_v4_and_v5_goldens_sample_byte_equal_to_the_v2_tree(fixture: str, flags: list[int]) -> None:
    """The same two rows under each layout: every sampled array is the v2 tree's, and v5 flags record 1 alone."""
    golden = json.loads((_REPLAY / "hexg_v2_sample_golden.json").read_text(encoding="utf-8"))
    buf = HexgBuffer(16, _ENC, 128)
    assert buf.load_from_path(str(_REPLAY / fixture)) == 2
    arrays, _targets = _sampled_arrays(buf, 20261001, 8)
    moved = [k for k, v in arrays.items() if _digest(v) != golden["arrays"][k]]
    assert moved == [], f"{fixture}: arrays no longer byte-equal to the v2 tree: {moved}"
    assert load_ring(_REPLAY / fixture).proven_tail.tolist() == flags


def _saved_flags(buf: HexgBuffer, path: Path) -> list[int]:
    buf.save_to_path(str(path))
    return load_ring(path).proven_tail.tolist()


def test_a_pushed_flag_rides_its_row_into_the_saved_ring_and_a_plain_push_saves_none(tmp_path: Path) -> None:
    flagged, plain = HexgBuffer(4, _ENC, 128), HexgBuffer(4, _ENC, 128)
    flagged.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, False, 1.0, True, 7, game_id=0, proven_tail=True)
    plain.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, False, 1.0, True, 7, game_id=0)
    assert (_saved_flags(flagged, tmp_path / "f.hexg"), _saved_flags(plain, tmp_path / "p.hexg")) == ([1], [0])


def test_a_flagged_row_claiming_a_full_search_is_refused_by_name() -> None:
    buf = HexgBuffer(4, _ENC, 128)
    with pytest.raises(ValueError, match="proven_tail"):
        buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, True, 1.0, True, 7, game_id=0, proven_tail=True)
    assert buf.size == 0


def test_the_flag_survives_an_engine_load_and_resave(tmp_path: Path) -> None:
    """A flagged quick row and an unflagged full-search row, saved, loaded back into the engine and saved again."""
    buf = HexgBuffer(4, _ENC, 128)
    buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, False, 1.0, True, 7, game_id=0, proven_tail=True)
    buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, True, -1.0, True, 7, game_id=1)
    first = tmp_path / "ring.hexg"
    buf.save_to_path(str(first))
    assert first.read_bytes()[4:8] == (5).to_bytes(4, "little")
    back = HexgBuffer(4, _ENC, 128)
    assert back.load_from_path(str(first)) == 2
    assert _saved_flags(back, tmp_path / "again.hexg") == [1, 0]
