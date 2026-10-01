"""The HEXG v3 root value through the engine face: v2 rows byte-equal, the field round-trips, bad values refused."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mantis._engine import HexgBuffer
from mantis.selfplay.graph_collate import graph_wire_from_rust

_REPLAY = Path(__file__).resolve().parents[1] / "fixtures" / "replay"
_ENC = "gnn_axis_v1"
_STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]


def _digest(arr: Any) -> dict[str, Any]:
    a = np.asarray(arr)
    raw = np.ascontiguousarray(a).tobytes()  # lifts a 0-d scalar to 1-d; the shape is read before it
    return {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": hashlib.sha256(raw).hexdigest()}


def _sampled_arrays(buf: HexgBuffer, seed: int, batch: int) -> tuple[dict[str, Any], Any]:
    """Every wire and pre-v3 target array of one seeded augmented sample, by the golden's names."""
    buf.seed_sampler(seed)
    wire, targets = buf.sample_graph_batch(batch, augment=True, n_threads=1)
    payload = graph_wire_from_rust(wire)
    arrays = {"wire_" + f.name: np.asarray(getattr(payload, f.name)) for f in dataclasses.fields(payload)}
    for name in ("policy_target", "explicit_mask", "tail_mass", "outcomes", "value_valid", "is_full_search"):
        arrays["targets_" + name] = np.asarray(getattr(targets, name))
    cells = targets.target_argmax_cells
    arrays["targets_argmax_valid"] = np.array([c is not None for c in cells], dtype=np.uint8)
    arrays["targets_argmax_qr"] = np.array([c if c is not None else (0, 0) for c in cells], dtype=np.int32)
    return arrays, targets


def test_the_v2_golden_samples_byte_equal_to_the_v2_tree_and_carries_no_root_value() -> None:
    """A seeded sample of the v2 golden is the v2 tree's, array for array, with no root value on any row."""
    golden = json.loads((_REPLAY / "hexg_v2_sample_golden.json").read_text(encoding="utf-8"))
    buf = HexgBuffer(16, _ENC, 128)
    assert buf.load_from_path(str(_REPLAY / "hexg_v2_golden.hexg")) == 2
    arrays, targets = _sampled_arrays(buf, 20261001, 8)
    assert sorted(arrays) == sorted(golden["arrays"]), "the golden names every array the sample emits"
    moved = [k for k, v in arrays.items() if _digest(v) != golden["arrays"][k]]
    assert moved == [], f"arrays that are no longer byte-equal to the v2 tree: {moved}"
    assert np.asarray(targets.root_value_valid).tolist() == [0] * 8
    assert np.asarray(targets.root_value).view(np.uint32).tolist() == [0] * 8, "invalid rows carry +0.0"


def test_a_pushed_root_value_rides_its_row_to_the_sampled_targets() -> None:
    """Each sampled row carries its own pushed value and flag; a row pushed without one samples as none."""
    buf = HexgBuffer(8, _ENC, 128)
    for i in range(6):
        kw: dict[str, Any] = {} if i % 3 == 0 else {"root_value": i / 8.0, "root_value_valid": True}
        buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, i, True, float(i), True, 9, game_id=i, **kw)
    buf.seed_sampler(5)
    _wire, t = buf.sample_graph_batch(6, augment=True, n_threads=1)
    outcomes = np.asarray(t.outcomes)
    valid = np.asarray(t.root_value_valid)
    value = np.asarray(t.root_value)
    assert valid.dtype == np.uint8 and value.dtype == np.float32
    for k, z in enumerate(outcomes.tolist()):
        i = int(z)
        assert int(valid[k]) == int(i % 3 != 0), k
        assert float(value[k]) == (i / 8.0 if i % 3 else 0.0), k


@pytest.mark.parametrize(("value", "valid"), [
    (float("nan"), True), (1.5, True), (-1.25, True), (0.25, False),
])
def test_the_push_face_refuses_a_root_value_outside_its_contract(value: float, valid: bool) -> None:
    """Outside [-1, 1], or non-zero without the flag, is a ValueError naming the field; the ring stays empty."""
    buf = HexgBuffer(4, _ENC, 128)
    with pytest.raises(ValueError, match="root_value"):
        buf.push_graph_position(_STONES, [(2, 0, 1.0)], 1, 2, 0, True, 1.0, True, 9,
                                root_value=value, root_value_valid=valid)
    assert buf.size == 0
