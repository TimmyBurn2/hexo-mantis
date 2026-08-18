"""F-816-14 (R284(f)) — a child must not outlive its parent, even when the parent runs NOTHING.

THE DEFECT, MEASURED not hypothesised. Every teardown in this tree — `force_teardown_all`, the
eval pipeline's `terminate → join → kill`, the preflight tool's `os.killpg` — requires the parent
to EXECUTE. A parent killed outright (harness timeout, OOM kill, `kill -9`, an interrupted
session) runs none of them, and a child in its own session receives nothing, because a new
session is exactly what puts it beyond signals aimed at the parent's group. The kernel reparents
it to init and it runs without bound.

Found live on the development host 2026-08-18: a `preflight_mint.py --_boot` child spawned by a
test with `--timeout-sec 45.0`, at **PPID 1, 4 h 06 m old, 682% CPU, `VmHWM` 13.8 GB**. It
reproduced immediately when a pytest tier carrying a preflight row was killed. On the migration
box the same class held **458 MiB of a GPU** whose minted partition has 0.514 GiB of headroom.

THE TEST. A real grandchild process is started, armed, and then its parent is **SIGKILLed** —
the case no cooperative path can cover, because SIGKILL is precisely the signal a process cannot
handle. The assertion is that the grandchild is GONE and its memory with it.

The kill is `SIGKILL` and not `SIGTERM` deliberately: a SIGTERM'd parent could plausibly be
credited to Python's `atexit`/`daemon=True` machinery, and the row would then pass while proving
nothing about the mechanism it is named for.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from mantis.train.lifecycle.signals import arm_parent_death_signal

_LINUX = sys.platform.startswith("linux")
_DEADLINE_SEC = 20.0


def _alive(pid: int) -> bool:
    """True iff `pid` names a live, non-zombie process. A reaped-but-unwaited child stays in
    the table as a zombie; counting one as alive would make this test fail for the wrong reason
    (and counting one as dead would make it PASS for the wrong reason, which is worse)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def test_arming_reports_true_on_linux_and_false_elsewhere() -> None:
    """The contract of the return value, so a caller can log it truthfully (R169: a liveness
    claim ties to its instrument)."""
    assert arm_parent_death_signal() is _LINUX


def _write_scripts(tmp_path, marker, *, arm: bool, cooperative_sigterm: bool = False):
    """The two-generation harness as REAL FILES, not nested `-c` strings.

    Written this way after the string version produced an `IndentationError` inside the
    grandchild and both rows failed for a quoting reason rather than a process reason — which is
    exactly the failure mode a process test must not have, because it looks identical to the
    defect (no grandchild, no marker file).
    """
    stem = ("armed" if arm else "unarmed") + ("_coop" if cooperative_sigterm else "")
    child = tmp_path / f"child_{stem}.py"
    child.write_text(
        "import os, signal, time\n"
        # The production shape: a run installs a COOPERATIVE SIGTERM handler (save-then-exit),
        # so a SIGTERM does not end it. This is what made the first version of the fix fail.
        + ("signal.signal(signal.SIGTERM, lambda *a: None)\n" if cooperative_sigterm else "")
        + ("from mantis.train.lifecycle.signals import arm_parent_death_signal\n"
           "arm_parent_death_signal()\n" if arm else "")
        + f"open({str(marker)!r}, 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "while True:\n    time.sleep(0.2)\n",
        encoding="utf-8",
    )
    parent = tmp_path / ("parent_armed.py" if arm else "parent_unarmed.py")
    parent.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}], start_new_session=True)\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    return parent


def _await_marker(marker, deadline_sec: float = _DEADLINE_SEC) -> int:
    deadline = time.monotonic() + deadline_sec
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists(), "the grandchild never started; this row proved nothing"
    return int(marker.read_text(encoding="utf-8"))


@pytest.mark.skipif(not _LINUX, reason="PR_SET_PDEATHSIG is a Linux prctl; no equivalent here")
def test_a_SIGKILLED_parent_takes_its_armed_child_with_it(tmp_path) -> None:
    """THE producer test. Parent spawns a grandchild that arms and then sleeps; the parent is
    SIGKILLed; the grandchild must be gone.

    SIGKILL and not SIGTERM, deliberately: a SIGTERM'd parent could be credited to Python's
    `atexit` / `daemon=True` machinery, and the row would pass while proving nothing about the
    mechanism it is named for. SIGKILL is precisely the signal no cooperative path can cover."""
    marker = tmp_path / "grandchild.pid"
    parent_script = _write_scripts(tmp_path, marker, arm=True)
    parent = subprocess.Popen([sys.executable, str(parent_script)], cwd=os.getcwd())
    gc_pid = None
    try:
        gc_pid = _await_marker(marker)
        assert _alive(gc_pid), "the grandchild died before the parent was killed"

        parent.send_signal(signal.SIGKILL)   # the parent gets NO chance to clean up
        parent.wait(timeout=_DEADLINE_SEC)

        deadline = time.monotonic() + _DEADLINE_SEC
        while _alive(gc_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(gc_pid), (
            f"grandchild {gc_pid} SURVIVED its SIGKILLed parent — this is F-816-14: it is now "
            "reparented to init and will run without bound, holding whatever it holds"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=_DEADLINE_SEC)
        if gc_pid is not None and _alive(gc_pid):
            os.kill(gc_pid, signal.SIGKILL)


@pytest.mark.skipif(not _LINUX, reason="PR_SET_PDEATHSIG is a Linux prctl; no equivalent here")
def test_an_UNARMED_child_survives_the_same_kill(tmp_path) -> None:
    """THE MUTATION, and the load-bearing half of this file (LAW-07).

    Identical to the row above except the grandchild does NOT arm. It must SURVIVE — if it dies
    anyway, something else in the environment is reaping it and the row above is green without
    the mechanism it claims to test. A producer test whose negative control also passes is not a
    producer test.

    It reaps its own survivor, because leaving one would be this file writing the very defect it
    exists to detect."""
    marker = tmp_path / "unarmed.pid"
    parent_script = _write_scripts(tmp_path, marker, arm=False)
    parent = subprocess.Popen([sys.executable, str(parent_script)], cwd=os.getcwd())
    gc_pid = None
    try:
        gc_pid = _await_marker(marker)
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=_DEADLINE_SEC)
        time.sleep(2.0)
        assert _alive(gc_pid), (
            "the UNARMED grandchild also died — then the armed row above is not evidence for "
            "PR_SET_PDEATHSIG, and this file's mechanism claim is unproven"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=_DEADLINE_SEC)
        if gc_pid is not None and _alive(gc_pid):
            os.kill(gc_pid, signal.SIGKILL)   # this test's own orphan, reaped by this test


@pytest.mark.skipif(not _LINUX, reason="PR_SET_PDEATHSIG is a Linux prctl; no equivalent here")
def test_a_child_that_HANDLES_sigterm_is_still_taken_down(tmp_path) -> None:
    """THE REGRESSION ROW, and it exists because the first version of the fix failed here.

    `arm_parent_death_signal` originally defaulted to `SIGTERM`. Driven against a real
    `preflight_mint.py --_boot` child — which installs the cooperative save-then-exit handler —
    the signal ARRIVED and the child converted it into a PARK: `%CPU` decaying 408 → 133 over
    two minutes in state `Ssl`, still alive, still holding its memory. The orphan survived the
    fix that was supposed to end it.

    So this row gives the grandchild a SIGTERM handler that swallows the signal — the production
    shape — and asserts it dies anyway. It fails against the SIGTERM default and passes against
    the SIGKILL one, which is precisely the discrimination the first version lacked."""
    marker = tmp_path / "coop.pid"
    parent_script = _write_scripts(tmp_path, marker, arm=True, cooperative_sigterm=True)
    parent = subprocess.Popen([sys.executable, str(parent_script)], cwd=os.getcwd())
    gc_pid = None
    try:
        gc_pid = _await_marker(marker)
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=_DEADLINE_SEC)

        deadline = time.monotonic() + _DEADLINE_SEC
        while _alive(gc_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(gc_pid), (
            f"grandchild {gc_pid} swallowed the parent-death signal and SURVIVED. A death "
            "signal a process can turn into a park is not a death signal — this is the exact "
            "failure the SIGTERM default produced against a real preflight boot child"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=_DEADLINE_SEC)
        if gc_pid is not None and _alive(gc_pid):
            os.kill(gc_pid, signal.SIGKILL)
