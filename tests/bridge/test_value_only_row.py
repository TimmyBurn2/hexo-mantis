"""The quick arm's value-only row (no explicit entry, no tail) through the ring: saved, loaded, sampled and re-pushed."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from mantis._engine import HexgBuffer
from mantis.diagnostics.ring_reader import Ring, load_ring
from mantis.util.loadpkg import load_tools_package

load_tools_package("run_monitor")
rings = importlib.import_module("run_monitor.rings")

_ENC = "gnn_axis_v1"
_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]


def _push_value_only(buf: HexgBuffer, game_id: int) -> None:
    buf.push_graph_position(_STONES + [(2, 0, -1)], [], 1, 1, 4, False, 1.0, True, 4, game_id=game_id,
                            tail_mass=0.0, root_value=1.0, root_value_valid=True)


def _saved_ring(path: Path) -> Path:
    """Game 0: a searched row, then its decided root's value-only row; game 1: one searched row."""
    buf = HexgBuffer(8, _ENC, 128)
    buf.push_graph_position(_STONES, [(2, 0, 0.6), (1, 1, 0.4)], 1, 2, 3, True, 1.0, True, 4, game_id=0)
    _push_value_only(buf, 0)
    buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 3, True, -1.0, True, 4, game_id=1)
    buf.save_to_path(str(path))
    return path


def _assert_value_only_at(ring: Ring, i: int) -> None:
    assert int(ring.n_visits[i]) == 0 and float(ring.tail_mass[i]) == 0.0
    assert int(ring.is_full_search[i]) == 0 and int(ring.value_valid[i]) == 1
    assert float(ring.root_value[i]) == 1.0 and int(ring.root_value_valid[i]) == 1


def test_a_value_only_row_survives_save_and_load_and_samples_with_no_target(tmp_path: Path) -> None:
    """The row reads back as written and samples as an all-zero target with no argmax cell, its value intact."""
    path = _saved_ring(tmp_path / "r.hexg")
    _assert_value_only_at(load_ring(path), 1)
    loaded = HexgBuffer(8, _ENC, 128)
    assert loaded.load_from_path(str(path)) == 3

    only = HexgBuffer(4, _ENC, 128)
    _push_value_only(only, 7)
    _wire, targets = only.sample_graph_batch(2, augment=True, n_threads=1)
    assert float(np.asarray(targets.policy_target).sum()) == 0.0
    assert int(np.asarray(targets.explicit_mask).sum()) == 0
    assert np.asarray(targets.tail_mass).tolist() == [0.0, 0.0]
    assert np.asarray(targets.is_full_search).tolist() == [0, 0]
    assert np.asarray(targets.root_value).tolist() == [1.0, 1.0]
    assert targets.target_argmax_cells == [None, None]


def test_the_unseen_ring_re_pushes_a_value_only_row(tmp_path: Path) -> None:
    """PLANTED BREAK: refuse every empty row at the push face and the re-push raises EmptyTarget."""
    path = _saved_ring(tmp_path / "r.hexg")
    meta = rings.unseen_ring(path, -1, tmp_path / "u.hexg")
    assert meta["rows"] == 3 and meta["games"] == 2
    _assert_value_only_at(load_ring(tmp_path / "u.hexg"), 1)
