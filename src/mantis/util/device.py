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

    HERE, and not in `mantis.eval.child_memory` — the caller this placement was decided for.
    (AUDIT-1 F-52: the line read "where its ONE caller lives"; `diagnostics/worker_sweep.py`
    reaches it at four sites too. The placement argument below is unaffected — it is about
    which module may own `torch.cuda`, not about how many callers there are — but a count
    stated in passing is still read as a census.) The reason is
    an isolation law rather than tidiness: `tests/eval/test_pipeline_isolation.py::
    test_parent_side_eval_modules_have_no_inference_surface` bans every `.cuda` attribute in
    `src/mantis/eval/*.py` except the child entry point, because an in-process CUDA eval path
    on the parent side is what that law makes unrepresentable. The eval child's memory probe
    needs four counters; wording the guard around it would be the shape this repo refuses, so
    the torch access lives here — in the module that owns `torch.cuda` FOR THE PATHS A GUARD
    FENCES OFF. That is the true width of the claim and it is narrower than it used to read:
    this module is NOT the repo's only `torch.cuda` consumer (`selfplay/graph_collate.py`,
    `selfplay/inference_server.py`, `train/subsystems.py` and
    `diagnostics/fusion_calibrate.py` all touch it directly), and a sentence claiming the
    whole repo is the overclaiming class this tree keeps finding.
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


def _require_cuda(device: str) -> str:
    """Return `device` iff it names a CUDA device on a torch build that has one, else raise.

    The refusal, and not a zero, is the point. `cuda_memory_counters` above states the rule
    this shares: *a counter read that silently returned zeros would be a measurement of
    nothing reported as a measurement*. The 2026-08-22 re-calibration sitting has the measured
    instance — a reader that guessed at a shape it did not recognise reported 1 392 GiB of
    high-water on a 16 GiB card, and a plausible-looking wrong number would have been minted
    against.
    """
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

    THE SECOND SINK, and it is a different instrument from the allocator counters rather than
    a second view of them. `max_memory_allocated` reports what THIS process's caching
    allocator handed out; this reports what the CARD has committed, which includes the CUDA
    context, fragmentation the allocator holds but has not handed out, and any co-resident
    process. On the 2026-08-22 host the two disagreed by 3.62 GiB of high-water at matched
    config and duration across allocator postures (`RECAL_EXIT_2026-08-22.md` §2/§8.3), and
    the sitting's own falsifier was decided on the larger of the pair.

    The box block's standing rule — already stated on `cuda_memory_counters` — is that where a
    boundary sample and a high-water disagree the LARGER GOVERNS and the disagreement is a
    finding. That rule needs both numbers; this is the one the repo did not yet expose.
    """
    dev = _require_cuda(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
    return int(total_bytes) - int(free_bytes)


def cuda_device_total_bytes(device: str) -> int:
    """Total bytes on `device` — the `total` half of `torch.cuda.mem_get_info`.

    Carried beside the used reading because a peak without the capacity it was taken against is
    a number a reader cannot size. The 2026-08-22 record is the case: "15 342 MiB" means one
    thing on this card and another on a larger one, and the sitting's own falsifier was an
    inequality against the card total.
    """
    dev = _require_cuda(device)
    _free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
    return int(total_bytes)


def reset_cuda_peak_counters(device: str) -> None:
    """Open a fresh high-water window on `device` (`torch.cuda.reset_peak_memory_stats`).

    A MEASUREMENT BOUNDARY, which is why it refuses on a device with no counters instead of
    doing nothing: a silent no-op would leave a caller believing it had opened a window it
    never opened, and every later figure would then be a fragment of some earlier one.

    `mantis.eval.child_memory` warns about the inverse hazard — a FOREIGN reset mid-round
    turns a later figure into a fragment, and a figure that FELL reads as memory released. The
    two are the same rule from opposite sides: a reset is legitimate exactly when the caller
    owns the window and says where its boundaries are. The eval child owns no boundaries and
    never calls this; `mantis.diagnostics.worker_sweep` owns per-round boundaries, declares
    them in its report, and does.
    """
    dev = _require_cuda(device)
    torch.cuda.reset_peak_memory_stats(dev)
