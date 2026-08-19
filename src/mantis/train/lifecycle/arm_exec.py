"""The supervisor's arming trampoline: arm `PR_SET_PDEATHSIG`, then BECOME the child.

``python -m mantis.train.lifecycle.arm_exec -- PROG ARG...``

WHY A TRAMPOLINE AND NOT A GATE IN THE RUN. `PR_SET_PDEATHSIG` is cleared across `fork` and
PRESERVED across `execve`, so a process that arms and then `execvp`s hands the arming to
whatever it becomes — WITHOUT that program knowing anything about it. That is the only way to
arm a wrapper this repo does not own: `uv run` does not `exec` (measured), so under a plain
`Popen` the supervisor's direct child is the WRAPPER and the run is a GRANDCHILD whose death
nobody promised. Measured before this file existed: `kill -9` on the supervisor of
``-- uv run python -m mantis.run`` left the run — the process holding the GPU — alive and
reparented to the host's subreaper (Q3 red-team A4b).

WHAT THIS CLOSES, AND EXACTLY HOW FAR — the claim is narrowed deliberately, because the
unqualified version is FALSE and was measured false. For a DIRECT exec (no wrapper: the
trampoline `execvp`s straight into `python -m mantis.run`) the arming is in place from the
first instruction of the run's own interpreter, so the run is protected through its whole
lifetime INCLUDING the `import torch` that precedes its entry point — that shape does close
Q3_DESIGN §9 residual 2. For a WRAPPED launch — a `uv run`-style wrapper that FORKS rather than
execs, which is the shape A4b is entirely about — the run is a fork-spawned grandchild and
`PR_SET_PDEATHSIG` does NOT survive `fork`: the run starts UNARMED and stays unarmed until its
own `arm_parent_death_if_supervised()` runs, which is still behind the full torch import,
exactly as before this packet. Residual 2 is therefore UNCHANGED for the wrapped case. What
the trampoline gives that case instead is the death of the WRAPPER, which is what makes the
run's own depth-2 arming fire once it is armed.

That residual's harm is BOUNDED, not silent: a supervisor that dies inside the unarmed window
does not orphan the run unnoticed — when `arm_parent_death_if_supervised()` does run, its own
check finds the stamped ppid already gone and exits `PARENT_VANISHED_EXIT_CODE` (71) at the
gate rather than proceeding armed against a parent that no longer exists (see
`train/lifecycle/signals.py`). The exposure is the window itself, never a leaf that runs on
forever believing it is supervised when it is not.

It lives in `mantis.train.lifecycle` and not in `mantis.monitor` because it must import the ONE
authority for the mechanism (`arm_parent_death_signal`, this package) and `monitor -> train` is
an illegal edge. The supervisor names it by the CONTRACT CONSTANT in `monitor/heartbeat.py` — a
string, never an import — which is exactly where the parent-death env stamp already lives, for
the reason that file already gives.

Deliberately tiny and torch-free: its one dependency costs tens of milliseconds cold, paid once
per spawn, and a babysitting trampoline that needed seconds and gigabytes to start would be a
worse failure than the orphan it prevents.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from mantis.train.lifecycle.signals import arm_parent_death_signal

#: The POSIX convention for "command not found", used when the `execvp` itself fails. NOT a
#: minted diagnosis and outside the reserved 42–48 band by construction: this process never
#: became the program it was asked to become, so it has no outcome of its own to report.
EXEC_FAILED_EXIT_CODE: int = 127

_USAGE = (
    "usage: python -m mantis.train.lifecycle.arm_exec -- PROG ARG...  "
    "(the program after `--` is required)"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Arm, then `execvp`. Returns only if the exec failed, and then not at all — the failure
    path is an `os._exit`, because a trampoline that returns to its caller would let a *Python
    traceback* stand in for the child the supervisor asked for.

    The `--` separator is REQUIRED and its absence is refused, mirroring
    `supervise._split_argv`'s refusal shape: a trampoline that guessed where its own flags end
    would silently swallow the first word of the child command.
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
