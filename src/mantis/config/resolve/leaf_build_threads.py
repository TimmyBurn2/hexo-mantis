"""THE one derivation for the EVAL leaf build's width.

Armed against a profile that put 95.3 % of the eval game loop inside the serial leaf-graph
build, an N-sweep at a real 64-move board separating it into a 5.2 ms/leaf slope against a
2.4 ms round-trip intercept. A derived prediction, not a measured optimum.

The reservation is IDENTICAL to the ring rebuild's, so this DELEGATES to `resolve_sample_threads`
rather than restating it: one place where the reservation can be wrong, one name per consumer.
Self-play workers are deliberately NOT covered — each is already one of `n_workers` threads
building its own leaves, so widening one takes threads from the others and double-counts the
reservation. The eval child is the case this exists for: one calling thread on an idle card.
"""
from typing import Any

from mantis.config.resolve.sample_threads import resolve_sample_threads


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


__all__ = ["resolve_leaf_build_threads"]
