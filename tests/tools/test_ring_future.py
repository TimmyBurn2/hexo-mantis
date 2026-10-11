"""`tools/ring_future.py`: a ring's rows get their game's next placements from the run's records, settled by game."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from mantis._engine import Board, HexgBuffer, future_horizon
from mantis.diagnostics.ring_reader import load_ring

_ENC = "gnn_axis_v1"
_SHARED = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
#: Two recorded games sharing their first six placements, and one the records never saw.
_GAMES = {
    0: _SHARED + [(2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2), (2, -2), (2, 1), (1, 2), (-1, 3), (-3, 1)],
    1: _SHARED + [(2, -1), (-2, 1), (1, 1), (3, 0), (0, 3), (-3, 3), (-1, -1), (1, -2), (3, -2), (2, 2)],
    2: [(1, 1), (2, 0), (0, 2), (-1, 2), (3, 0), (0, 3), (-2, 2), (2, 1), (1, 2), (-1, 3)],
}


def _tool() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools" / "ring_future.py"
    spec = importlib.util.spec_from_file_location("ring_future_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _placed(moves: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    board, out = Board.with_encoding_name(_ENC), []
    for q, r in moves:
        out.append((q, r, board.current_player))
        board.apply_move(q, r)
    return out


def _write_inputs(tmp_path: Path, records: dict[int, list[tuple[int, int]]], *, copies: int = 1) -> tuple[Path, Path]:
    """A ring of every game's plies 1.. without the field, and a shard of `records`."""
    buf = HexgBuffer(256, _ENC, 16)
    for gid, moves in _GAMES.items():
        placed, board = _placed(moves), Board.with_encoding_name(_ENC)
        board.apply_move(*moves[0])
        for ply in range(1, len(moves)):
            buf.push_graph_position(placed[:ply], [(9, 9, 1.0)], board.current_player, board.moves_remaining, ply,
                                    True, 1.0, True, (len(moves) + 1) // 2, game_id=gid)
            board.apply_move(*moves[ply])
    ring = tmp_path / "ring.hexg"
    buf.save_to_path(str(ring))
    shard = tmp_path / "games_0_seg0.jsonl"
    shard.write_text("".join(json.dumps({"channel": "selfplay", "run_id": "t", "step": 5, "game_id": f"g{g}",
                                         "moves": [list(m) for m in moves], "result": "p1"}) + "\n"
                             for g, moves in records.items() for _ in range(copies)), encoding="utf-8")
    return ring, shard


def _run(tmp_path: Path, records: dict[int, list[tuple[int, int]]], *extra: str, copies: int = 1,
         min_settled: str = "0.5") -> tuple[int, Any]:
    ring, shard = _write_inputs(tmp_path, records, copies=copies)
    out = tmp_path / "out.hexg"
    rc = _tool().main(["--ring", str(ring), "--games", str(shard), "--run-id", "t", "--step-lo", "0",
                       "--step-hi", "9", "--out", str(out), "--min-settled", min_settled, *extra])
    return rc, load_ring(out) if out.exists() else None


def test_each_recorded_games_rows_get_its_next_placements_and_the_rest_none(tmp_path: Path) -> None:
    rc, out = _run(tmp_path, {0: _GAMES[0], 1: _GAMES[1]})
    assert rc == 0
    horizon = future_horizon()
    for i in range(out.header.size):
        gid, ply = int(out.game_id[i]), int(out.ply_index[i])
        want = None if gid == 2 else _placed(_GAMES[gid])[ply:ply + horizon]
        assert out.row_future_stones(i) == want, f"game {gid} ply {ply}"
    prov = json.loads((tmp_path / "out.hexg.provenance.json").read_text(encoding="utf-8"))
    assert prov["unsettled_rows"]["no_record"] == len(_GAMES[2]) - 1
    assert prov["rings"][str(tmp_path / "out.hexg")]["pairs_contradicting"] == 0


def test_a_ring_game_whose_record_contradicts_its_rows_is_given_no_field(tmp_path: Path) -> None:
    """PLANTED BREAK: game 0's record with two later placements swapped no longer holds its rows' positions."""
    altered = list(_GAMES[0])
    altered[9], altered[11] = altered[11], altered[9]
    rc, out = _run(tmp_path, {0: altered, 1: _GAMES[1]}, min_settled="0.3")
    assert rc == 0
    assert not out.future_valid[out.game_id == 0].any()
    assert out.future_valid[out.game_id == 1].all()


def test_the_heldout_split_is_by_whole_game(tmp_path: Path) -> None:
    rc, out = _run(tmp_path, {0: _GAMES[0], 1: _GAMES[1]}, "--heldout-rows", "5", "--heldout-seed", "3",
                   "--heldout-out", str(tmp_path / "held.hexg"))
    held = load_ring(tmp_path / "held.hexg")
    assert rc == 0 and held.header.size >= 5
    assert set(np.unique(held.game_id).tolist()).isdisjoint(np.unique(out.game_id).tolist())
    assert held.header.size + out.header.size == sum(len(m) - 1 for m in _GAMES.values())


def test_copies_of_a_record_settle_as_the_one_game_they_are(tmp_path: Path) -> None:
    """The mirror holds copies of shards: a record twice is one game, not two candidates."""
    rc, out = _run(tmp_path, {0: _GAMES[0], 1: _GAMES[1]}, copies=2)
    assert rc == 0 and out.future_valid[out.game_id != 2].all()
    prov = json.loads((tmp_path / "out.hexg.provenance.json").read_text(encoding="utf-8"))
    assert (prov["records"], prov["records_duplicate"]) == (2, 2)


def test_a_conversion_that_settles_too_few_rows_is_refused_and_keeps_no_ring_under_the_name(tmp_path: Path) -> None:
    """PLANTED BREAK: one recorded game of three settles a third of the rows, under the stated share; the ring stays `.partial`."""
    rc, out = _run(tmp_path, {0: _GAMES[0]}, min_settled="0.5")
    assert rc == 1 and out is None
    assert (tmp_path / "out.hexg.partial").exists()
    prov = json.loads((tmp_path / "out.hexg.partial.provenance.json").read_text(encoding="utf-8"))
    assert prov["refused"] and prov["settled_share"] < 0.5


def test_heldout_rows_without_an_output_path_is_a_usage_error(tmp_path: Path) -> None:
    ring, shard = _write_inputs(tmp_path, {})
    with pytest.raises(SystemExit) as exc:
        _tool().main(["--ring", str(ring), "--games", str(shard), "--run-id", "t", "--step-lo", "0", "--step-hi", "9",
                      "--out", str(tmp_path / "o.hexg"), "--min-settled", "0.5", "--heldout-rows", "3"])
    assert exc.value.code == 2
