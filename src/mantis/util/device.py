"""Device selection utilities for cross-platform support (CUDA, MPS, CPU).

Imports ``torch`` at module top, so ``mantis.util.__init__`` MUST NOT import this module — the
torch-free leaves stay importable without torch, and only explicit torch-consumers import here.
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

    It lives here rather than in `mantis.eval.child_memory` because an isolation test bans every
    `.cuda` attribute in `src/mantis/eval/*.py` outside the child entry point: this module owns
    `torch.cuda` for the paths that guard fences off, not for the whole repo.
    """
    if str(device).split(":", 1)[0].strip().lower() != "cuda":
        return False
    return bool(torch.cuda.is_available())


def cuda_memory_counters(device: str) -> dict[str, int]:
    """The caching allocator's four numbers for `device`: two high-water, two instantaneous.

    BOTH pairs, because a boundary sample and a high-water are different instruments and where
    they disagree the LARGER governs — a rule no reader can apply against one number. Raises on
    a device with no CUDA: zeros would be a measurement of nothing reported as a measurement.
    """
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def release_cuda_cache() -> None:
    """Release the CUDA caching allocator's freed blocks; no-op when CUDA is unavailable.

    The graph path generates variable-size tensor batches per MCTS leaf, and the allocator
    cannot reuse mismatched blocks, so without this release reserved VRAM grows monotonically.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_cuda(device: str) -> str:
    """Return `device` iff it names a CUDA device on a torch build that has one, else raise —
    the refusal, and not a zero, is the point: a reader that guessed at a shape it did not
    recognise once reported 1 392 GiB of high-water on a 16 GiB card."""
    if str(device).split(":", 1)[0].strip().lower() != "cuda":
        raise ValueError(
            f"device {device!r} is not a CUDA device: there is no caching allocator and no "
            "card reading to take. Refusing rather than returning zeros."
        )
    if not torch.cuda.is_available():
        raise ValueError(
            f"device {device!r} names CUDA but this process has no CUDA: nothing to measure. "
            "Refusing rather than returning zeros."
        )
    return str(device)


def cuda_device_used_bytes(device: str) -> int:
    """CARD-level used bytes for `device` — `total - free` from `torch.cuda.mem_get_info`.

    A different instrument from the allocator counters, not a second view of them: this reports
    what the CARD has committed, including the CUDA context, fragmentation and any co-resident
    process. On the 2026-08-22 host the two disagreed by 3.62 GiB of high-water at matched
    config and duration, and where they disagree the LARGER governs.
    """
    dev = _require_cuda(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
    return int(total_bytes) - int(free_bytes)


def cuda_device_total_bytes(device: str) -> int:
    """Total bytes on `device` — the `total` half of `torch.cuda.mem_get_info`, carried beside
    the used reading because a peak without the capacity it was taken against is a number a
    reader cannot size."""
    dev = _require_cuda(device)
    _free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
    return int(total_bytes)


def reset_cuda_peak_counters(device: str) -> None:
    """Open a fresh high-water window on `device` (`torch.cuda.reset_peak_memory_stats`).

    A MEASUREMENT BOUNDARY, which is why it refuses on a device with no counters: a silent no-op
    would leave a caller believing it had opened a window it never opened. A reset is legitimate
    exactly when the caller owns the window and declares its boundaries, which is why the eval
    child never calls this and the worker sweep does.
    """
    dev = _require_cuda(device)
    torch.cuda.reset_peak_memory_stats(dev)
