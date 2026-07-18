"""Smoke net for mantis.util.device (net addition — the module had no old test).

device.py imports torch at module top (DESIGN §c.7), so the entire module is
skipped when torch is absent (the torch-dependent leg). When torch IS present:
`best_device()` returns a torch.device and prefers CUDA > MPS > CPU under
monkeypatched availability.
"""
from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed (torch-dependent leg)",
)


def test_best_device_returns_torch_device():
    import torch

    from mantis.util.device import best_device

    d = best_device()
    assert isinstance(d, torch.device)


def test_best_device_prefers_cuda(monkeypatch):
    import torch

    from mantis.util import device as devmod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert devmod.best_device().type == "cuda"


def test_best_device_prefers_mps_when_no_cuda(monkeypatch):
    import torch

    from mantis.util import device as devmod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert devmod.best_device().type == "mps"
    else:
        pytest.skip("torch build has no mps backend")


def test_best_device_falls_back_to_cpu(monkeypatch):
    import torch

    from mantis.util import device as devmod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert devmod.best_device().type == "cpu"
