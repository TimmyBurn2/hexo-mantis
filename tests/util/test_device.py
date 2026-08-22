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


# ── WORKER-SWEEP (R309(g)) — the card-level sink and the per-round boundary ──────────────
# Two readings the caching-allocator counters do not carry, added HERE because
# `mantis.util.device` owns `torch.cuda` for the paths `tests/eval/test_pipeline_isolation.py`
# fences off — NOT for the whole repo, which is measurably false (`selfplay/graph_collate.py`,
# `selfplay/inference_server.py`, `train/subsystems.py` and `diagnostics/fusion_calibrate.py` all
# touch it directly). The narrower claim is the one that survives a grep, and it is still the
# reason these two live here rather than in the sweep. The card-level one is the sink
# that measured 15 342 MiB against the allocator's own figure at matched config on the
# 2026-08-22 host (RECAL_EXIT_2026-08-22.md §2/§8.3); where the two disagree the LARGER
# governs, which is a rule no reader can apply against one number.


def test_cuda_device_used_bytes_is_total_minus_free(monkeypatch):
    import torch

    from mantis.util import device as devmod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (400, 1000))
    assert devmod.cuda_device_used_bytes("cuda:0") == 600


def test_cuda_device_used_bytes_refuses_a_device_with_no_cuda(monkeypatch):
    """A counter read that silently returned zeros would be a measurement of NOTHING
    reported as a measurement — `cuda_memory_counters`' own stated posture, and the reason
    the 2026-08-22 sitting's `peaks.py` produced 1 392 GiB on a 16 GiB card."""
    import torch

    from mantis.util import device as devmod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="cpu"):
        devmod.cuda_device_used_bytes("cpu")


def test_cuda_device_used_bytes_refuses_when_cuda_is_unavailable(monkeypatch):
    import torch

    from mantis.util import device as devmod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no CUDA"):
        devmod.cuda_device_used_bytes("cuda:0")


def test_reset_cuda_peak_counters_calls_through_to_the_allocator(monkeypatch):
    import torch

    from mantis.util import device as devmod

    seen: list = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda dev: seen.append(dev))
    devmod.reset_cuda_peak_counters("cuda:0")
    assert seen == ["cuda:0"]


def test_reset_cuda_peak_counters_refuses_a_device_with_no_cuda(monkeypatch):
    """The reset is a MEASUREMENT BOUNDARY. A no-op on a device with no counters would make
    a caller believe it had opened a fresh window it never opened."""
    import torch

    from mantis.util import device as devmod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no CUDA"):
        devmod.reset_cuda_peak_counters("cuda:0")


def test_cuda_device_total_bytes_returns_the_TOTAL_half_not_the_free_half(monkeypatch):
    """A TRANSPOSED UNPACK (`total_bytes, free_bytes = mem_get_info(...)`) would report free
    bytes as the card total, and every headroom calculation downstream would be wrong with the
    whole suite green. `mem_get_info` returns `(free, total)`, and this row is what pins which
    end of the pair the field named `card_total_bytes` carries."""
    import torch

    from mantis.util import device as devmod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (400, 1000))
    assert devmod.cuda_device_total_bytes("cuda:0") == 1000
    assert devmod.cuda_device_used_bytes("cuda:0") == 600, (
        "used and total must come off the same pair in the same order"
    )
