"""The supervisor half of the parent-death contract: `spawn_child`.

`tests/monitor/test_supervisor.py` drives the supervisor LOOP through an injected `spawn_fn`;
this file covers the real collaborator. Four claims, one contract: the child's environment
carries the supervisor's OWN pid, the only stamp that tells a run it is supervised and may arm
`PR_SET_PDEATHSIG`; `os.environ` in the supervisor is NOT mutated, or a leaked stamp would name
"some ancestor" to every later child; argv still arrives VERBATIM; and an off-main-thread spawn
is REFUSED, because `PR_SET_PDEATHSIG` fires on the death of the CREATING THREAD.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import Any

import pytest

from mantis.monitor.heartbeat import PARENT_DEATH_PPID_ENV
from mantis.monitor.supervise import spawn_child

_DEADLINE_SEC = 30.0


def _run_and_capture(script: str, argv_tail: list[str] | None = None) -> str:
    """Spawn `python -c <script>` through the PRODUCTION `spawn_child` and return its stdout.

    stdout is captured by redirecting the child's file descriptor rather than by passing
    `stdout=PIPE`, because `spawn_child` takes argv and nothing else.
    """
    argv = [sys.executable, "-c", script, *(argv_tail or [])]
    read_fd, write_fd = os.pipe()
    saved = os.dup(1)
    try:
        os.dup2(write_fd, 1)
        proc = spawn_child(argv)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(write_fd)
    try:
        proc.wait(timeout=_DEADLINE_SEC)
        with os.fdopen(read_fd, "r", encoding="utf-8") as fh:
            out = fh.read()
    finally:
        if proc.poll() is None:   # pragma: no cover — only on a wedged child
            proc.kill()
            proc.wait(timeout=_DEADLINE_SEC)
    assert proc.returncode == 0, f"the probe child exited {proc.returncode}"
    return out


def test_spawn_child_stamps_its_own_pid_in_the_child_environment() -> None:
    """The stamp equals THIS process's pid, not its parent's: the child's gate compares it
    against its own `getppid()`, so any other value never arms at all."""
    out = _run_and_capture(
        "import os,sys; sys.stdout.write(os.environ["
        f"{PARENT_DEATH_PPID_ENV!r}])"
    )
    assert out.strip() == str(os.getpid()), (
        f"the child must see {PARENT_DEATH_PPID_ENV}=<supervisor pid>; got {out.strip()!r}"
    )


def test_spawn_child_does_not_mutate_the_supervisors_own_environment() -> None:
    """The leak a COPY exists to prevent: a stamp written into `os.environ` would name an
    ancestor rather than a parent to every later child of this process."""
    before = dict(os.environ)
    _run_and_capture("pass")
    assert PARENT_DEATH_PPID_ENV not in os.environ, (
        f"{PARENT_DEATH_PPID_ENV} leaked into the supervisor's own environment"
    )
    assert dict(os.environ) == before, "spawn_child mutated the supervisor's environment"


def test_spawn_child_passes_argv_verbatim() -> None:
    """The documented "verbatim argv after `--`" contract survives the env change. An injected
    `--parent-death-...` flag would change the run's argv, which appears in provenance."""
    tail = ["--config", "some/config.yaml", "--out-dir", "some/out"]
    out = _run_and_capture("import json,sys; sys.stdout.write(json.dumps(sys.argv[1:]))", tail)
    assert json.loads(out) == tail, f"argv was not passed verbatim: {out!r}"


def test_spawn_child_refuses_to_launch_from_a_non_main_thread() -> None:
    """The refusal, driven from a real thread: a child spawned from a worker thread is SIGKILLed
    the moment that thread returns, a premature kill strictly worse than the orphan arming
    prevents. `RuntimeError` and not `assert`, because `python -O` strips asserts."""
    box: dict[str, Any] = {}

    def _attempt() -> None:
        try:
            proc = spawn_child([sys.executable, "-c", "pass"])
        except BaseException as exc:  # noqa: BLE001 — the exception IS the observation
            box["exc"] = exc
            return
        box["proc"] = proc   # pragma: no cover — only if the refusal is missing

    worker = threading.Thread(target=_attempt, name="q3-offthread-spawn")
    worker.start()
    worker.join(timeout=_DEADLINE_SEC)
    assert not worker.is_alive(), "the spawning thread never returned"

    proc: subprocess.Popen[bytes] | None = box.get("proc")
    if proc is not None:   # pragma: no cover — reached only against an unrefused mutant
        proc.kill()
        proc.wait(timeout=_DEADLINE_SEC)
    assert isinstance(box.get("exc"), RuntimeError), (
        f"an off-main-thread spawn must raise RuntimeError; got {box.get('exc')!r}"
    )
    assert "PR_SET_PDEATHSIG" in str(box["exc"]), (
        "the refusal message must name the mechanism — it is the operator's only route to the "
        f"reason; got {str(box['exc'])!r}"
    )


def test_spawn_child_from_the_main_thread_is_allowed() -> None:
    """POSITIVE CONTROL for the row above: without it the refusal could be "always raise" and
    the suite would stay green while the supervisor could launch nothing."""
    out = _run_and_capture("import sys; sys.stdout.write('ok')")
    assert out.strip() == "ok"


def test_the_supervisor_binds_the_real_spawn_child_as_its_spawn_fn() -> None:
    """The seam that makes every row above production-relevant: `main()` binds THIS function, so
    a stamp in a private helper the supervisor does not use cannot pass."""
    import inspect

    from mantis.monitor import supervise

    source = inspect.getsource(supervise.main)
    assert "spawn_fn=spawn_child" in source, (
        "the supervisor's CLI entry must bind the real `spawn_child`; a stamp in a function "
        "nothing calls is not a fix"
    )


@pytest.mark.parametrize("argv", [[], [""]])
def test_spawn_child_rejects_an_empty_command_the_same_way_it_always_did(argv) -> None:
    """A missing or empty program is still `Popen`'s own error, not a swallowed one."""
    with pytest.raises((IndexError, ValueError, OSError)):
        spawn_child(argv)
