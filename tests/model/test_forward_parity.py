"""O_bench (A-committed, Tier-2) — small-net forward drift guard.

Pins the ported nets' forward on fixed seed+input+weights to committed KB-scale
goldens (`value_probes/forward/*.pt`), covering GNN + CNN (scalar/dist65) + the
aux-flags-ON case. These goldens are NEW-generated (a self-contained regression
reference for the ported code, NOT the old-side parity reference — that Tier-1 leg
loads `wp/WP9/oldside/` and is recorded in IMPL_NOTES). Guards future drift.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mantis.model import CnnArch, GnnArch, GnnNet, build_net

_FWD = Path(__file__).resolve().parents[1] / "fixtures" / "value_probes" / "forward"
_TOL = 1e-6


def _load(name: str):
    return torch.load(_FWD / f"{name}.pt", map_location="cpu", weights_only=False)


def _assert_match(got, expected) -> None:
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        g = g.to(torch.float32)
        e = e.to(torch.float32)
        assert g.shape == e.shape, (g.shape, e.shape)
        max_abs = float((g - e).abs().max().item()) if g.numel() else 0.0
        assert max_abs <= _TOL, f"forward drift: max_abs_diff={max_abs:.3e} > {_TOL}"


@pytest.mark.parametrize("name", ["small_cnn_scalar", "small_cnn_dist65", "small_cnn_aux_chain"])
def test_cnn_forward_golden(name: str) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    payload = _load(name)
    net = build_net(CnnArch(**payload["arch"]))
    net.load_state_dict(payload["state_dict"], strict=True)
    net.eval()
    flags = payload["flags"]
    with torch.no_grad():
        out = net.forward(payload["inputs"]["x"], aux=flags["aux"], chain=flags["chain"])
    _assert_match(list(out), payload["outputs"])


def test_gnn_forward_single_golden() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    payload = _load("small_gnn")
    net = build_net(GnnArch(**payload["arch"]))
    assert isinstance(net, GnnNet)
    net.load_state_dict(payload["state_dict"], strict=True)
    net.eval()
    i = payload["inputs"]
    with torch.no_grad():
        out = net.forward_single(
            i["x"], i["edge_index"], i["edge_attr"], i["legal_mask"], i["stone_mask"]
        )
    _assert_match(list(out), payload["outputs"])
