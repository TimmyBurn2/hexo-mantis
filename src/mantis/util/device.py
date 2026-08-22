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


def cuda_counters_available(device: str) -> bool:
    """True iff `device` names a CUDA device whose allocator counters can be read.

    HERE, and not in `mantis.eval.child_memory` where its one caller lives, and the reason is
    an isolation law rather than tidiness: `tests/eval/test_pipeline_isolation.py::
    test_parent_side_eval_modules_have_no_inference_surface` bans every `.cuda` attribute in
    `src/mantis/eval/*.py` except the child entry point, because an in-process CUDA eval path
    on the parent side is what that law makes unrepresentable. The eval child's memory probe
    needs four counters; wording the guard around it would be the shape this repo refuses, so
    the torch access lives in the module that already owns `torch.cuda` for the whole repo.
    """
    if str(device).split(":", 1)[0].strip().lower() != "cuda":
        return False
    return bool(torch.cuda.is_available())


def cuda_memory_counters(device: str) -> dict[str, int]:
    """The caching allocator's four numbers for `device`: two high-water, two instantaneous.

    BOTH pairs, because a boundary sample and a high-water are different instruments and the
    box block's standing rule is that where they disagree the LARGER governs — a rule no
    reader can apply against one number. Caller's responsibility to have checked
    `cuda_counters_available` first; this raises on a device with no CUDA, which is correct:
    a counter read that silently returned zeros would be a measurement of nothing reported as
    a measurement.
    """
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def release_cuda_cache() -> None:
    """Release the CUDA caching allocator's freed blocks (CARD_VRAM_ACCUMULATION, VERDICT-A).

    The graph path generates variable-size tensor batches per MCTS leaf; the caching
    allocator cannot reuse blocks of mismatched sizes and accumulates them. Without this
    release, reserved VRAM grows monotonically; with it, reserved stays bounded. No-op
    when CUDA is unavailable.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
