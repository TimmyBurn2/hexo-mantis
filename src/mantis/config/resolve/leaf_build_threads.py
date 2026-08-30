"""`resolve_leaf_build_threads` — THE one derivation for the EVAL leaf build's width.

NIGHTRUN-1 E1, against the Leg 1 profile. `submit_graphs_and_wait_ls` built its leaf graphs
in a serial loop on the calling thread; the profile puts **95.3 % of the whole eval game loop**
inside that call, and an N-sweep at a real 64-move board separates it into a 5.2 ms/leaf slope
against a 2.4 ms round-trip intercept. The loop is embarrassingly parallel, so it is
parallelised — and this module decides how wide.

ONE ARITHMETIC, TWO NAMED CONSUMERS. The reservation is IDENTICAL to the ring rebuild's
(`resolve_sample_threads`) and this function DELEGATES to it rather than restating it: both
run on a box that has already promised `selfplay.n_workers` threads to self-play and one to
the inference-server thread, and both must take only what is left. Restating the same
arithmetic under a second name is the duplicate-authority class R79 names; a thin delegating
sibling keeps ONE place where the reservation can be wrong, while letting each consumer carry
its own reason.

WHY A SEPARATE NAME AT ALL, rather than calling the ring's resolver directly at the eval site.
The ring resolver's name says what it is for, and a second consumer reading it makes the name
a lie for one of them. The next reader who changes the ring's reservation would have no way to
know an eval path moved with it. The delegation is visible here; the coupling is not implicit.

WHAT THIS DOES **NOT** COVER, stated because the omission is deliberate: the SELF-PLAY worker's
own `LocalInferenceEngine` stays at the serial width. Each self-play worker is already one of
`n_workers` threads building its own leaves; widening one worker's build takes its threads from
the others, so the pool as a whole gains nothing and the reservation above would be
double-counted. The eval child is the case this exists for — one calling thread, on a box whose
card is otherwise idle for the duration of the build.

**This is a DERIVED prediction, not a measured optimum**, exactly as the ring's is. What it is
measured against is the eval single-stream arm at 64 moves, whose reading belongs beside this
docstring the day it changes.
"""
from typing import Any

from mantis.config.resolve.sample_threads import resolve_sample_threads


def resolve_leaf_build_threads(full_config: Any, *, cpu_count: int | None = None) -> int:
    """The number of OS threads the eval leaf-graph build may use. Always >= 1.

    Args:
        full_config: the whole validated config mapping (`RunConfig.model_dump()`).
        cpu_count: the host's usable core count; `None` reads `os.cpu_count()`. Injectable so
            a test can state a machine rather than assert about the one it runs on.

    Returns:
        The same budget the ring rebuild gets — `max(1, cpu_count - n_workers - 1)`. `1` is
        the serial path and the exact-parity control, never a state meaning "no threads".

    Raises:
        MissingSampleThreadsInputError: an input the reservation derives from is absent. The
            error is the delegate's and is deliberately NOT re-wrapped: it names
            `selfplay.n_workers`, which is the level that is actually missing.
    """
    return resolve_sample_threads(full_config, cpu_count=cpu_count)


__all__ = ["resolve_leaf_build_threads"]
