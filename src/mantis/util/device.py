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
