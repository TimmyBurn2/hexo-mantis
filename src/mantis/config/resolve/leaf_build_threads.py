"""THE one derivation for the EVAL leaf build's width.

Armed against a profile that put 95.3 % of the eval game loop inside the serial leaf-graph
build, an N-sweep at a real 64-move board separating it into a 5.2 ms/leaf slope against a
2.4 ms round-trip intercept. A derived prediction, not a measured optimum.

The reservation is IDENTICAL to the ring rebuild's, so this DELEGATES to `resolve_sample_threads`
rather than restating it: one place where the reservation can be wrong, one name per consumer.
Self-play workers are deliberately NOT covered — each is already one of `n_workers` threads
building its own leaves, so widening one takes threads from the others and double-counts the
reservation. The eval child is the case this exists for: one calling thread on an idle card.
A host its caller asserts runs no self-play reserves nothing: `resolve_standalone_leaf_build_threads`.
"""
import os
from collections.abc import Mapping
from typing import Any

from mantis.config.resolve.sample_threads import (
    MissingSampleThreadsInputError,
    resolve_sample_threads,
)

#: The inference-server thread every engine starts.
_SERVER_THREADS = 1


def resolve_leaf_build_threads(full_config: Any, *, cpu_count: int | None = None) -> int:
    """Return the number of OS threads the eval leaf-graph build may use. Always >= 1.

    Args:
        full_config: the whole validated config mapping (`RunConfig.model_dump()`).
        cpu_count: the host's usable core count; `None` reads `os.cpu_count()`. Injectable so
            a test can state a machine rather than assert about the one it runs on.

    Returns:
        The same budget the ring rebuild gets — `max(1, cpu_count - n_workers - 1)`. `1` is
        the serial path and the exact-parity control, never a state meaning "no threads".

    Raises:
        MissingSampleThreadsInputError: an input the reservation derives from is absent. The
            delegate's error is deliberately NOT re-wrapped — it names `selfplay.n_workers`,
            the level that is actually missing.
    """
    return resolve_sample_threads(full_config, cpu_count=cpu_count)


def resolve_standalone_leaf_build_threads(
    full_config: Any, *, concurrency: int, cpu_count: int | None = None,
) -> int:
    """The cores less the serving thread, shared by `concurrency` games in flight, at most one leaf batch. Always >= 1.

    Raises:
        MissingSampleThreadsInputError: the config carries no `selfplay.leaf_batch_size`.
        ValueError: `concurrency` is below 1.
    """
    if int(concurrency) < 1:
        raise ValueError(f"resolve_standalone_leaf_build_threads: concurrency={concurrency} games in flight")
    section = full_config.get("selfplay") if isinstance(full_config, Mapping) else None
    if not isinstance(section, Mapping) or "leaf_batch_size" not in section:
        raise MissingSampleThreadsInputError(
            "selfplay.leaf_batch_size is absent, so the standalone build width has no ceiling (LAW-11)")
    cores = int(cpu_count) if cpu_count is not None else (os.cpu_count() or 1)
    per_game = (cores - _SERVER_THREADS) // int(concurrency)
    return max(1, min(int(section["leaf_batch_size"]), per_game))


__all__ = ["resolve_leaf_build_threads", "resolve_standalone_leaf_build_threads"]
