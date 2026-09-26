"""Per-worker resources under pytest-xdist: a share of the cores, and repo probe paths of its own."""
from __future__ import annotations

import os
from collections.abc import MutableMapping

#: The pools a real boot's children size from the environment: torch/OpenMP, MKL, and rayon.
_POOL_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")


def thread_share(workers: int | None, cpus: int) -> int | None:
    """The cores one of `workers` xdist workers may use, or None outside xdist."""
    if workers is None:
        return None
    return max(1, cpus // workers)


def apply_thread_share(env: MutableMapping[str, str], cpus: int | None = None) -> int | None:
    """Export this worker's share to every pool its children read, never over an explicit value."""
    count = env.get("PYTEST_XDIST_WORKER_COUNT")
    share = thread_share(int(count) if count else None, cpus or os.cpu_count() or 1)
    if share is not None:
        for var in _POOL_VARS:
            env.setdefault(var, str(share))
    return share


def worker_suffix(env: MutableMapping[str, str]) -> str:
    """A suffix that keeps one worker's repo-root probe paths apart from every other worker's."""
    worker = env.get("PYTEST_XDIST_WORKER")
    return f".{worker}" if worker else ""
