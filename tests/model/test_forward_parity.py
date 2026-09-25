"""O_bench (A-committed, Tier-2) — small-net forward drift guard.

Pins the ported net's forward on fixed seed+input+weights to a committed KB-scale
golden (`value_probes/forward/small_gnn.pt`). The golden is NEW-generated (a
self-contained regression reference for the ported code, NOT the old-side parity
reference — that Tier-1 leg loads `wp/WP9/oldside/` and is recorded in IMPL_NOTES).
The three `small_cnn_*` legs went with the dense path. Guards future drift.
"""
from __future__ import annotations

from pathlib import Path

import torch

from _determinism import deterministic_algorithms
from mantis.model import GnnArch, GnnNet, build_net

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


def test_gnn_forward_single_golden() -> None:
    """Deterministic mode and one thread for this test only: both are process-global, and leaked they re-number every later test."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with deterministic_algorithms():
            _check_small_gnn()
    finally:
        torch.set_num_threads(threads)


def _check_small_gnn() -> None:
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
