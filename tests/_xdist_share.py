"""Per-worker resources under pytest-xdist: a share of the cores, and repo probe paths of its own."""
from __future__ import annotations

from collections.abc import MutableMapping

from _cpu_budget import _THREAD_ENV_VARS, detect_cpu_budget


def thread_share(workers: int | None, cpus: int) -> int | None:
    """The cores one of `workers` xdist workers may use, or None outside xdist."""
    if workers is None:
        return None
    return max(1, cpus // workers)


def apply_thread_share(env: MutableMapping[str, str], cpus: int | None = None) -> int | None:
    """Export this worker's share of the container-aware CPU budget to the BLAS/OpenMP pools, never over an explicit value."""
    count = env.get("PYTEST_XDIST_WORKER_COUNT")
    share = thread_share(int(count) if count else None, cpus or detect_cpu_budget())
    if share is not None:
        for var in _THREAD_ENV_VARS:
            env.setdefault(var, str(share))
    return share


def worker_suffix(env: MutableMapping[str, str]) -> str:
    """A suffix that keeps one worker's repo-root probe paths apart from every other worker's."""
    worker = env.get("PYTEST_XDIST_WORKER")
    return f".{worker}" if worker else ""
