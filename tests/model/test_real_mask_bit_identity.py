"""`GnnNetV2.real_mask_from_batch` is bit-identical to the scalar-assignment form it replaced, on every device."""
from __future__ import annotations

import pytest
import torch

from mantis.model.gnn_v2 import GnnNetV2

_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LOUD SKIP — the synced form only syncs on CUDA"))]


def _assignment_form(stone_mask: torch.Tensor, legal_index: torch.Tensor) -> torch.Tensor:
    """The oracle: the pre-L1 expression, whose pageable scalar H2D synced every CUDA forward."""
    real = stone_mask.clone()
    real[legal_index] = True
    return real


@pytest.mark.parametrize("device", _DEVICES)
def test_the_mask_equals_the_assignment_form_on_random_batches(device: str) -> None:
    """Random stone masks and strictly ascending legal rows, empty legal sets included."""
    gen = torch.Generator().manual_seed(20260925)
    for n, n_legal in [(1, 0), (7, 0), (7, 3), (64, 64), (5000, 1200), (40_000, 9000)]:
        stone = (torch.rand(n, generator=gen) < 0.3).to(device)
        legal = torch.randperm(n, generator=gen)[:n_legal].sort().values.to(device)
        before = stone.clone()
        got = GnnNetV2.real_mask_from_batch(stone, legal)
        assert got.dtype is torch.bool and got.device == stone.device
        assert torch.equal(got, _assignment_form(stone, legal)), f"n={n} legal={n_legal} on {device}"
        assert torch.equal(stone, before), "the input stone mask was mutated"
