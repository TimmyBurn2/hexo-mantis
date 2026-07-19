"""O2 — 234-probe value-health harness (test oracle).

O2a: the frozen M1–M4 metric definitions reproduce the committed golden from the
committed decoded-v arrays (no checkpoint). O2b: the scoring path (model construction
+ dist65 decode + argmin-K min-pool) exercised on a synthetic net. O2c: probe-fixture
SHA tamper → loud. Real-anchor end-to-end M1–M4 DEFERS to WP11.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import _value_health as vh
import numpy as np
import pytest
import torch

from mantis.model import CnnArch, build_net

_FIX = Path(__file__).resolve().parents[1] / "fixtures" / "value_probes"


# ── O2a metric-math golden ────────────────────────────────────────────────────


def test_o2a_metrics_reproduce_golden_to_8dp() -> None:
    d = np.load(_FIX / "decoded_v.npz")
    want = json.loads((_FIX / "metrics.json").read_text())
    got = vh.metrics_from_decoded(
        "scalar", d["loss_v"], d["safe_v"], d["loss_tail"], d["safe_tail"]
    )
    for key, want_key in [
        ("mean_v_on_losses", "M1_mean_v_on_losses"),
        ("ece", "M2_ece"),
        ("decoded_auc", "M3_decoded_auc"),
        ("false_pessimism", "M4_false_pessimism"),
    ]:
        val = got[key]
        assert val is not None, f"scalar arm: {key} must be populated"
        assert abs(val - want[want_key]) <= 5e-9
    assert got["tail_mass_auc"] is None  # scalar arm


# ── O2b synthetic-net scoring ─────────────────────────────────────────────────


def _synthetic_cluster_tensors(k: int, c: int, hw: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(k, c, hw, hw, generator=g, dtype=torch.float32) for _ in range(3)]


def _score_set(model, arm, tensors):
    return [vh.score_cluster_tensor(model, arm, t) for t in tensors]


def test_o2b_dist65_scoring_schema_and_routing() -> None:
    torch.manual_seed(0)
    model = build_net(CnnArch(board_size=19, in_channels=4, filters=16, res_blocks=1,
                              value_head_type="dist65"))
    model.eval()
    loss = _score_set(model, "dist65", _synthetic_cluster_tensors(3, 4, 19, 1))
    safe = _score_set(model, "dist65", _synthetic_cluster_tensors(3, 4, 19, 2))
    metrics = vh.compute_metrics("dist65", loss, safe)
    row = vh.build_row("dist65", "deadbeef", "v6_live2_ls", metrics, len(loss), len(safe))
    assert tuple(row.keys()) == vh.ROW_KEYS
    assert row["n_loss"] == 3 and row["n_safe"] == 3
    # dist65 arm → tail_mass_auc populated, decoded_auc null.
    assert metrics["tail_mass_auc"] is not None
    assert metrics["decoded_auc"] is None
    ece = metrics["ece"]
    assert ece is not None and 0.0 <= ece <= 1.0
    assert all(s["tail_mass"] is not None for s in loss + safe)


def test_o2b_scalar_scoring_routing_inverse() -> None:
    torch.manual_seed(0)
    model = build_net(CnnArch(board_size=19, in_channels=8, filters=16, res_blocks=1,
                              value_head_type="scalar"))
    model.eval()
    loss = _score_set(model, "scalar", _synthetic_cluster_tensors(3, 8, 19, 3))
    safe = _score_set(model, "scalar", _synthetic_cluster_tensors(3, 8, 19, 4))
    metrics = vh.compute_metrics("scalar", loss, safe)
    assert metrics["decoded_auc"] is not None
    assert metrics["tail_mass_auc"] is None
    assert all(s["tail_mass"] is None for s in loss + safe)


def test_o2b_scoring_is_deterministic() -> None:
    torch.manual_seed(0)
    model = build_net(CnnArch(board_size=19, in_channels=4, filters=16, res_blocks=1,
                              value_head_type="dist65"))
    model.eval()
    t = _synthetic_cluster_tensors(3, 4, 19, 7)
    assert _score_set(model, "dist65", t) == _score_set(model, "dist65", t)


def test_o2b_arm_mismatch_raises() -> None:
    torch.manual_seed(0)
    dist = build_net(CnnArch(board_size=19, in_channels=4, filters=16, res_blocks=1,
                             value_head_type="dist65"))
    scal = build_net(CnnArch(board_size=19, in_channels=8, filters=16, res_blocks=1,
                             value_head_type="scalar"))
    with pytest.raises(ValueError):
        vh.assert_arm_matches(dist, "scalar")
    with pytest.raises(ValueError):
        vh.assert_arm_matches(scal, "dist65")
    # matched arms are fine.
    vh.assert_arm_matches(dist, "dist65")
    vh.assert_arm_matches(scal, "scalar")


# ── O2c SHA-tamper → loud ─────────────────────────────────────────────────────


def test_o2c_default_path_sha_guard_holds_on_frozen_fixtures() -> None:
    # The committed fixtures match the pinned SHAs → the default-path guard passes.
    vh.verify_probe_shas(str(_FIX / "probe_set_v1.jsonl"), str(_FIX / "negatives_v1.jsonl"),
                         is_default=True)


def test_o2c_tampered_default_path_raises(tmp_path: Path) -> None:
    probe = tmp_path / "probe_set_v1.jsonl"
    neg = tmp_path / "negatives_v1.jsonl"
    shutil.copy(_FIX / "probe_set_v1.jsonl", probe)
    shutil.copy(_FIX / "negatives_v1.jsonl", neg)
    with probe.open("ab") as f:
        f.write(b"tamper\n")
    with pytest.raises(RuntimeError):
        vh.verify_probe_shas(str(probe), str(neg), is_default=True)


def test_o2c_custom_path_bypasses_guard(tmp_path: Path) -> None:
    probe = tmp_path / "probe_set_v1.jsonl"
    neg = tmp_path / "negatives_v1.jsonl"
    probe.write_text("mutated\n")
    neg.write_text("mutated\n")
    # is_default=False → guard skipped, no raise even though SHAs differ.
    vh.verify_probe_shas(str(probe), str(neg), is_default=False)
