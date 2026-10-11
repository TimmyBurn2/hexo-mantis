"""Through the production drain every pushed row carries its game's next placements, read against the game record."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from mantis._engine import Board, HexgBuffer, future_horizon
from mantis.diagnostics import ring_audit
from mantis.diagnostics.ring_reader import load_ring
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.monitor.game_record import iter_run_games
from mantis.monitor.game_recorder import GameRecorder
from mantis.selfplay.pool import WorkerPool
from test_search_stats_end_to_end import _TIMEOUT_S, _cfg

_ENCODING = "gnn_axis_v1"
_PLY_CAP = 30


def _placements(moves: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """The game record's stones with their players, read off a board the moves are replayed on."""
    board, out = Board.with_encoding_name(_ENCODING), []
    for q, r in moves:
        out.append((q, r, board.current_player))
        board.apply_move(q, r)
    return out


@pytest.mark.integration
def test_every_pushed_row_carries_its_games_next_placements(tmp_path: Path) -> None:
    spec = lookup(_ENCODING)
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim), hidden=16, num_layers=1)
    recorder = GameRecorder(record_dir=tmp_path, run_id="e2e", seed=1)
    ring = HexgBuffer(capacity=1024, encoding=_ENCODING, visit_capacity=128)
    cfg = _cfg()
    cfg["selfplay"]["max_game_moves"] = _PLY_CAP
    pool = WorkerPool(build_net(arch), cfg, torch.device("cpu"), ring, arch=arch, recorder=recorder)
    pool.start()
    try:
        deadline = time.monotonic() + _TIMEOUT_S
        while time.monotonic() < deadline and recorder.games_written < 2:
            pool.check_producer_health()
            time.sleep(0.2)
    finally:
        pool.stop()
    pool.check_producer_health()
    path = tmp_path / "ring.hexg"
    ring.save_to_path(str(path))
    rows = load_ring(path)
    assert rows.header.size > 0, "the pool pushed no row inside the budget"
    assert rows.future_valid.tolist() == [1] * rows.header.size, "a self-play row carries its future field"

    games = [_placements([tuple(m) for m in g["moves"]]) for g in iter_run_games(tmp_path, "e2e")]
    horizon, matched = future_horizon(), 0
    for i in range(rows.header.size):
        ply = int(rows.ply_index[i])
        stones = {(int(s["q"]), int(s["r"]), int(s["p"])) for s in rows.row_stones(i)}
        owners = [g for g in games if len(g) > ply and set(g[:ply]) == stones]
        if not owners:
            continue  # a game still in flight at the stop: its rows reached the ring, its record did not
        matched += 1
        field = rows.row_future_stones(i)
        assert any(field == g[ply:ply + horizon] for g in owners), f"row {i} (ply {ply}): {field}"
    assert matched >= 2 * horizon, f"only {matched} rows matched a recorded game"
    assert ring_audit.future_disagreements(rows)[1].size == 0, "the ring's own rows contradict each other"
