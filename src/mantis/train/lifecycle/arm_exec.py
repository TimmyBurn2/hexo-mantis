"""Arm `PR_SET_PDEATHSIG`, then BECOME the child.

``python -m mantis.train.lifecycle.arm_exec -- PROG ARG...``

The arming is cleared across `fork` and PRESERVED across `execve`, so arming and then `execvp`ing
hands it to a program that knows nothing about it — the only way to arm a wrapper this repo does
not own. A DIRECT exec is therefore armed from the run interpreter's first instruction, including
through `import torch`; a WRAPPED launch whose wrapper FORKS starts unarmed until the run's own
gate runs, and what the trampoline gives that case is the death of the WRAPPER.

It lives here rather than in `mantis.monitor` because it must import this package's arming
authority and `monitor -> train` is an illegal edge. Deliberately tiny and torch-free.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from mantis.train.lifecycle.signals import arm_parent_death_signal

#: The POSIX "command not found" convention, used when the `execvp` itself fails. Deliberately
#: outside the reserved diagnosis band: this process never became the program it was asked to be.
EXEC_FAILED_EXIT_CODE: int = 127

_USAGE = (
    "usage: python -m mantis.train.lifecycle.arm_exec -- PROG ARG...  "
    "(the program after `--` is required)"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Arm, then `execvp` into the program after `--`.

    The failure path is an `os._exit`: returning would let a Python traceback stand in for the
    child. The `--` is REQUIRED, since guessing would swallow the child command's first word.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--" not in args:
        raise SystemExit(_USAGE)
    child = args[args.index("--") + 1:]
    if not child:
        raise SystemExit("no program given after `--`")

    arm_parent_death_signal()
    try:
        os.execvp(child[0], child)
    except OSError as exc:
        sys.stderr.write(f"arm_exec: cannot exec {child[0]!r}: {exc}\n")
        sys.stderr.flush()
        os._exit(EXEC_FAILED_EXIT_CODE)
    return 0   # pragma: no cover — `execvp` does not return on success


if __name__ == "__main__":   # pragma: no cover — process entry point
    raise SystemExit(main())
