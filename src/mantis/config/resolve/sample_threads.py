"""`resolve_sample_threads` — THE one derivation for the ring sample's rebuild width.

PERF-TRANCHE-1 B1, against ledger §10.5 line #1. `sample_graph_batch` is the trainer's
single largest line, and this tranche's M-2 measurement split it: **1 221 ms of
`build_axis_graph` against 163 ms of fuse and 2 ms of align** per step at run5 shape. The
rebuild is a serial loop over independent items on a 24-thread box, so it is parallelised —
and this module decides how wide.

WHY A DERIVATION AND NOT A CONFIG KEY. A thread budget is a property of the HOST and of what
the run has already committed elsewhere, and both inputs already exist: `os.cpu_count()` is
the host, and `selfplay.n_workers` is what the run has promised to self-play. Minting a
third number that has to be kept consistent with those two by hand is the second-authority
shape R1 exists to refuse — and it would go stale the first time the run moved to a
different box. Nothing here is a code-side DEFAULT: there is no value standing in for one
the operator did not supply, only arithmetic over values they did.

THE RESERVATION IS THE POINT. During a training step the self-play workers are still
running, so a pool sized to the whole machine takes its threads from them: the step gets
faster while the run gets slower, which is the only level that matters (the research
packet's JC-4). The budget therefore reserves `n_workers` threads for self-play and one for
the inference-server thread, and takes what is left.

**This is a DERIVED prediction, not a measured optimum.** The contended arm — trainer
stepping while the workers run — is what decides whether the reservation is right, and its
reading belongs beside this docstring the day it exists.
"""
import os
from collections.abc import Mapping
from typing import Any

_KEY = "selfplay.n_workers"
#: The inference-server thread. One, not a fudge factor: `WorkerPool` starts exactly one
#: (`pool.py`, `self._inference_server`), and the PERF-BASELINE ledger measures it at 97.8 %
#: occupancy in the contended regime, so it is a whole thread and not a share of one.
_SERVER_THREADS = 1


class MissingSampleThreadsInputError(ValueError):
    """An input the thread budget derives from is absent. Names the missing level."""


def resolve_sample_threads(full_config: Any, *, cpu_count: int | None = None) -> int:
    """The number of OS threads the ring sample's rebuild may use. Always >= 1.

    Args:
        full_config: the whole validated config mapping (`RunConfig.model_dump()`).
        cpu_count: the host's usable core count; `None` reads `os.cpu_count()`. Injectable
            so a test can state a machine rather than assert about the one it runs on.

    Returns:
        `max(1, cpu_count - n_workers - 1)` — never 0, because 1 is the serial path and a
        budget of "no threads at all" is not a state this loop can be in.

    Raises:
        MissingSampleThreadsInputError: the config is not a mapping, has no `selfplay`
            section, or that section carries no `n_workers`.
    """
    if not isinstance(full_config, Mapping):
        raise MissingSampleThreadsInputError(
            f"{_KEY}: the config is not a mapping ({type(full_config).__name__}), so the "
            "self-play reservation cannot be read and the budget would silently take the "
            "whole machine from the workers"
        )
    if "selfplay" not in full_config:
        raise MissingSampleThreadsInputError(
            f"{_KEY}: the config has no `selfplay` section. Absent is an ERROR, never a "
            "default (LAW-11): a reservation that silently became zero hands the trainer "
            "every core the self-play workers are using."
        )
    section = full_config["selfplay"]
    if not isinstance(section, Mapping):
        raise MissingSampleThreadsInputError(
            f"{_KEY}: the `selfplay` section is not a mapping ({type(section).__name__})"
        )
    if "n_workers" not in section:
        raise MissingSampleThreadsInputError(
            f"{_KEY} is absent. The key is REQUIRED by the schema, so a config that reaches "
            "here without it was not built through the one loader (R1)."
        )
    n_workers = int(section["n_workers"])
    cores = int(cpu_count) if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, cores - n_workers - _SERVER_THREADS)


__all__ = ["MissingSampleThreadsInputError", "resolve_sample_threads"]
