"""F-816-19 (R285(h)) — the supervisor half of the parent-death contract: `spawn_child`.

`tests/monitor/test_supervisor.py` drives the supervisor LOOP through an injected `spawn_fn`
and never touches the real one; it is also a frozen oracle. This file covers the real
collaborator instead, and the four claims it must satisfy are one contract:

  * the child's environment carries the supervisor's OWN pid — that stamp is the only thing
    that tells a run it is supervised and may arm `PR_SET_PDEATHSIG`;
  * `os.environ` in the supervisor is NOT mutated — a leaked stamp would name "some ancestor"
    to every later child of this process, which is precisely the confusion the child's gate
    refuses;
  * argv still arrives VERBATIM — the module's documented, deliberate property, which the env
    change must not have bought its way past;
  * an off-main-thread spawn is REFUSED — `PR_SET_PDEATHSIG` fires on the death of the CREATING
    THREAD, so a child spawned from a worker thread dies when that thread returns.
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
    `stdout=PIPE`, because `spawn_child` takes argv and nothing else — reaching in to add a
    kwarg would be testing a different function from the one that ships.
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
    """The stamp is present in the child and equals THIS process's pid — not its parent's, not
    a placeholder. The child's gate compares the stamp against its own `getppid()`, so any
    other value would make every supervised run take the wrapper arm and never arm at all."""
    out = _run_and_capture(
        "import os,sys; sys.stdout.write(os.environ["
        f"{PARENT_DEATH_PPID_ENV!r}])"
    )
    assert out.strip() == str(os.getpid()), (
        f"the child must see {PARENT_DEATH_PPID_ENV}=<supervisor pid>; got {out.strip()!r}"
    )


def test_spawn_child_does_not_mutate_the_supervisors_own_environment() -> None:
    """The leak that a COPY exists to prevent. If `spawn_child` wrote into `os.environ`, every
    later child of this process — in a test session, children that are not runs at all — would
    inherit a stamp naming an ancestor rather than a parent, and a long-lived supervisor would
    hand the same stale value to processes it did not spawn."""
    before = dict(os.environ)
    _run_and_capture("pass")
    assert PARENT_DEATH_PPID_ENV not in os.environ, (
        f"{PARENT_DEATH_PPID_ENV} leaked into the supervisor's own environment"
    )
    assert dict(os.environ) == before, "spawn_child mutated the supervisor's environment"


def test_spawn_child_passes_argv_verbatim() -> None:
    """The documented contract — "the child command is the verbatim argv after `--`" — survives
    the env change. This is the row that would bite an injected `--parent-death-...` flag: a
    flag would change the run's argv, which appears in provenance."""
    tail = ["--config", "some/config.yaml", "--out-dir", "some/out"]
    out = _run_and_capture("import json,sys; sys.stdout.write(json.dumps(sys.argv[1:]))", tail)
    assert json.loads(out) == tail, f"argv was not passed verbatim: {out!r}"


def test_spawn_child_refuses_to_launch_from_a_non_main_thread() -> None:
    """The refusal, driven from a real thread. `PR_SET_PDEATHSIG` fires when the parent THREAD
    dies, so a child spawned from a worker thread is SIGKILLed the moment that thread returns —
    a premature kill of a healthy run, strictly worse than the orphan the arming prevents.

    `RuntimeError` and not `assert`: `python -O` strips asserts, and this is a production
    safety invariant rather than a test aid."""
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
    """THE POSITIVE CONTROL for the row above. Without it the refusal could be "always raise"
    and the suite would stay green while the supervisor could no longer launch anything."""
    out = _run_and_capture("import sys; sys.stdout.write('ok')")
    assert out.strip() == "ok"


def test_the_supervisor_binds_the_real_spawn_child_as_its_spawn_fn() -> None:
    """The seam that makes every row above production-relevant: `main()` binds THIS function.
    A mutant that stamps the environment in a private helper the supervisor does not use would
    pass all five rows above and ship the defect unchanged."""
    import inspect

    from mantis.monitor import supervise

    source = inspect.getsource(supervise.main)
    assert "spawn_fn=spawn_child" in source, (
        "the supervisor's CLI entry must bind the real `spawn_child`; a stamp in a function "
        "nothing calls is not a fix"
    )


@pytest.mark.parametrize("argv", [[], [""]])
def test_spawn_child_rejects_an_empty_command_the_same_way_it_always_did(argv) -> None:
    """The env change must not have altered the failure shape of a bad argv: a missing or empty
    program is still `Popen`'s own error, not a swallowed one."""
    with pytest.raises((IndexError, ValueError, OSError)):
        spawn_child(argv)
