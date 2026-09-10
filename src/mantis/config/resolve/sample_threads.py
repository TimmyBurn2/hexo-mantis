"""THE one derivation for the ring sample's rebuild width.

Armed against a measurement that split `sample_graph_batch`, the trainer's largest line, into
1 221 ms of `build_axis_graph` against 163 ms of fuse and 2 ms of align per step at run5 shape.
A DERIVED prediction, not a measured optimum.

Derived rather than minted as a config key: both inputs already exist (`os.cpu_count()` and
`selfplay.n_workers`), and a third hand-synced number would go stale on the next box. Nothing
here is a code-side default — only arithmetic over values the operator supplied.

THE RESERVATION IS THE POINT: self-play workers are still running during a training step, so a
pool sized to the whole machine takes its threads from them and the run gets slower while the
step gets faster.
"""
import os
from collections.abc import Mapping
from typing import Any

_KEY = "selfplay.n_workers"
#: The inference-server thread. One, not a fudge factor: `WorkerPool` starts exactly one, and it
#: was measured at 97.8 % occupancy contended, so it is a whole thread and not a share of one.
_SERVER_THREADS = 1


class MissingSampleThreadsInputError(ValueError):
    """An input the thread budget derives from is absent. Names the missing level."""


def resolve_sample_threads(full_config: Any, *, cpu_count: int | None = None) -> int:
    """Return the number of OS threads the ring sample's rebuild may use. Always >= 1.

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
