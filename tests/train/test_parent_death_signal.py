"""Prove a child does not outlive its parent, even when the parent runs NOTHING.

Every teardown here needs the parent to EXECUTE, and a child in its own session is beyond signals
aimed at the parent's group. Measured live: an orphan at PPID 1, 4 h 06 m old, 682% CPU, 13.8 GB.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

_LINUX = sys.platform.startswith("linux")
_DEADLINE_SEC = 20.0


def _alive(pid: int) -> bool:
    """Report whether `pid` names a live, non-zombie process; counting a zombie as dead would
    make these rows pass for the wrong reason."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def test_arming_reports_true_on_linux_and_false_elsewhere(tmp_path) -> None:
    """Prove the arming call reports true on Linux and false elsewhere.

    Driven IN A SUBPROCESS: calling it in the pytest process permanently arms the RUNNER, and a
    detached tier then dies the instant its launcher exits (measured A/B).
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from mantis.train.lifecycle.signals import arm_parent_death_signal\n"
        "print(int(arm_parent_death_signal()))\n",
        encoding="utf-8",
    )
    out = subprocess.run([sys.executable, str(probe)], cwd=os.getcwd(),
                         capture_output=True, text=True, timeout=_DEADLINE_SEC)
    assert out.returncode == 0, out.stderr
    assert bool(int(out.stdout.strip())) is _LINUX


def _write_scripts(tmp_path, marker, *, arm: bool, cooperative_sigterm: bool = False):
    """Write the two-generation harness as real files: a quoting error inside a nested `-c`
    string looks identical to the defect, no grandchild and no marker file."""
    stem = ("armed" if arm else "unarmed") + ("_coop" if cooperative_sigterm else "")
    child = tmp_path / f"child_{stem}.py"
    child.write_text(
        "import os, signal, time\n"
        # The production shape: a cooperative SIGTERM handler, so a SIGTERM does not end it.
        + ("signal.signal(signal.SIGTERM, lambda *a: None)\n" if cooperative_sigterm else "")
        + ("from mantis.train.lifecycle.signals import arm_parent_death_signal\n"
           "arm_parent_death_signal()\n" if arm else "")
        + f"open({str(marker)!r}, 'w', encoding='utf-8').write(str(os.getpid()))\n"
        # Self-limited deliberately: the unarmed row's survivor is reaped only by that test's
        # `finally`, so an unbounded sleep would manufacture the defect this file detects.
        "time.sleep(90)\n",
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
    """Prove a SIGKILLed parent takes its armed grandchild with it — SIGTERM would be creditable
    to Python's own `atexit` machinery, and no cooperative path can cover SIGKILL."""
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
    """Prove an UNARMED child survives the same kill: if it dies anyway, something else is
    reaping it and the armed row is green without the mechanism it claims to test."""
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
    """Prove a child that HANDLES SIGTERM is still taken down.

    A SIGTERM-based arming was measured arriving at a real boot child and being converted into a
    park: %CPU decaying 408 to 133 over two minutes, still holding its memory.
    """
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
