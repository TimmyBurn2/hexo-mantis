"""Recomputing each trunk layer in backward changes no number: outputs and every gradient equal the kept-activation run."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from _gine_oracle import synthetic_batch
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net
from mantis.model.gine import RepresentationNetwork

_REPO = Path(__file__).resolve().parents[2]


def _step(net: torch.nn.Module, b: dict[str, torch.Tensor], device: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    net.zero_grad(set_to_none=True)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        logits, value, bins = net.forward_batch(b["x"], b["edge_index"], b["edge_attr"], b["legal_index"],
                                                b["stone_mask"], b["node_offsets"])
    (logits.float().square().mean() + value.float().sum() + bins.float().square().mean()).backward()
    return logits.detach(), [p.grad.clone() for p in net.parameters() if p.grad is not None]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LOUD SKIP — the bf16 autocast training path needs a GPU"))])
def test_recompute_equals_kept_activations(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same net, batch and seed, with and without the per-layer recompute: bit-identical logits and gradients."""
    config = load_config(production_configs(_REPO)[0]).model_dump()
    torch.manual_seed(20260925)
    net = build_net(arch_from_spec_and_config(lookup(config["identity"]["encoding"]), config)).to(device).train()
    b = synthetic_batch(device, n_graphs=4 if device == "cpu" else 16)
    recomputed = _step(net, b, device)
    monkeypatch.setattr(RepresentationNetwork, "_layer", RepresentationNetwork._conv)
    kept = _step(net, b, device)
    assert torch.equal(recomputed[0], kept[0])
    assert len(recomputed[1]) == len(kept[1]) > 0
    assert all(torch.equal(x, y) for x, y in zip(recomputed[1], kept[1], strict=True))
