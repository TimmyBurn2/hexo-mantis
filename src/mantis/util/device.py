"""Device selection utilities for cross-platform support (CUDA, MPS, CPU).

Imports ``torch`` at module top (DESIGN §c.7: torch-at-module-top KEPT). The
package init ``mantis.util.__init__`` MUST NOT import this module, so that
``mantis.util`` and the torch-free leaves (``cpu_budget``, ``coordinates``,
``constants``) stay importable without torch. Only explicit torch-consumers do
``from mantis.util.device import best_device``.
"""

from __future__ import annotations

import torch


def best_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def release_cuda_cache() -> None:
    """Release the CUDA caching allocator's freed blocks (CARD_VRAM_ACCUMULATION, VERDICT-A).

    The graph path generates variable-size tensor batches per MCTS leaf; the caching
    allocator cannot reuse blocks of mismatched sizes and accumulates them. Without this
    release, reserved VRAM grows monotonically; with it, reserved stays bounded. No-op
    when CUDA is unavailable.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
