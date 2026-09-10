"""CPU thread budget detection + per-library env defaults — TEST-TREE, NOT WIRED.

Relocated out of ``src/mantis/util/``: an AST census found ZERO consumers outside this module
and its own test. The code is preserved because the failure mode it addresses is real and
measured; the wiring is what is absent. Wiring ``apply_auto_thread_budget`` into an entry point
sets ``OMP_NUM_THREADS`` and its siblings process-wide, moving every timing measurement on this
machine and confounding the perf lane's banked before-side — a scheduling reason, re-openable
once the re-bench readout lands.

PyTorch / NumPy / OpenBLAS / MKL read these vars at native-runtime initialisation, during
import, so they must be set before ``import numpy`` or ``import torch``. Without it, on a
container with N threads carved out of an M-thread host, every BLAS op grabs M threads against
the N-slot cgroup: 100 % container CPU, ~60 % GPU util, self-play workers starved. Stdlib-only.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_INTEROP_THREADS",
)


def detect_cpu_budget() -> int:
    """Return the smallest of (host nproc, sched affinity, cgroup quotas), or 1 if every
    detection fails."""
    candidates: list[int] = []
    n = os.cpu_count()
    if n:
        candidates.append(n)
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    # cgroup v2: /sys/fs/cgroup/cpu.max  ("max <period>" or "<quota> <period>")
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if parts and parts[0] != "max":
            q, p = int(parts[0]), int(parts[1])
            if q > 0 and p > 0:
                candidates.append(max(1, math.ceil(q / p)))
    except (FileNotFoundError, ValueError, IndexError, PermissionError):
        pass
    # cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_{quota,period}_us
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read().strip())
        if q > 0 and p > 0:
            candidates.append(max(1, math.ceil(q / p)))
    except (FileNotFoundError, ValueError, PermissionError):
        pass
    return min(candidates) if candidates else 1


def derive_per_lib(budget: int, n_workers: int | None) -> int:
    """Return the per-library thread count for a detected budget and worker count.

    ``clamp(budget // (4 + n_workers // 8), 1, 8)``, divisor 4 when ``n_workers`` is absent: each
    batch of 8 self-play workers adds roughly one more concurrent BLAS caller, and the cap of 8
    stops a 128-vCPU host granting 32-thread BLAS for tiny ops."""
    if n_workers is None or n_workers <= 0:
        divisor = 4
    else:
        divisor = 4 + n_workers // 8
    return max(1, min(budget // max(1, divisor), 8))


def apply_auto_thread_budget(
    *,
    n_workers: int | None = None,
    log_prefix: str = "[mantis]",
    silent: bool = False,
) -> dict[str, Any]:
    """Set per-library thread caps in os.environ from the detected cgroup budget.

    Idempotent via ``_MANTIS_THREAD_BUDGET_APPLIED``; then ``MANTIS_THREAD_BUDGET=N``; then any
    pre-existing per-var env, filled with ``setdefault`` so only unset vars are touched.
    ``n_workers`` shrinks the per-lib slice for self-play workloads."""
    if "_MANTIS_THREAD_BUDGET_APPLIED" in os.environ:
        return {
            "cpu_budget": detect_cpu_budget(),
            "applied": False,
            **{v: os.environ.get(v, "") for v in _THREAD_ENV_VARS},
        }

    budget = detect_cpu_budget()
    forced = os.environ.get("MANTIS_THREAD_BUDGET")
    per_lib = int(forced) if forced else derive_per_lib(budget, n_workers)
    for v in _THREAD_ENV_VARS:
        os.environ.setdefault(v, str(per_lib))
    os.environ["_MANTIS_THREAD_BUDGET_APPLIED"] = "1"

    if not silent:
        nw = "n/a" if n_workers is None else str(n_workers)
        msg = (
            f"{log_prefix} cpu_budget={budget} n_workers={nw} per_lib={per_lib} "
            f"omp={os.environ['OMP_NUM_THREADS']} mkl={os.environ['MKL_NUM_THREADS']} "
            f"interop={os.environ['TORCH_INTEROP_THREADS']}"
        )
        try:
            print(msg, file=sys.stderr)
        except (OSError, ValueError):
            pass

    return {
        "cpu_budget": budget,
        "per_lib": per_lib,
        "applied": True,
        **{v: os.environ[v] for v in _THREAD_ENV_VARS},
    }


def apply_torch_interop_cap() -> None:
    """Apply ``torch.set_num_interop_threads`` from env after torch is imported — inter-op
    threads have no env hook, so they must be set before any parallel torch work. No-op if no
    relevant env var is set."""
    n = int(os.environ.get("TORCH_INTEROP_THREADS", os.environ.get("OMP_NUM_THREADS", "0")))
    if n <= 0:
        return
    try:
        import torch as _torch  # local: callers have already imported torch
        _torch.set_num_interop_threads(n)
    except (RuntimeError, ImportError):
        # RuntimeError: the parallel runtime already started; ImportError: torch absent.
        pass
