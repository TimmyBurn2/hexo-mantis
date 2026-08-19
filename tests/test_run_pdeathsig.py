"""F-816-19 (R285(h)) — a run spawned by the supervisor must not outlive it, and a run
spawned by anything else must not be touched.

THE DEFECT. `mantis.monitor.supervise` launched the run with a plain `Popen` and no arming, so
a supervisor killed outright (`kill -9`, an OOM kill, a dropped session) ORPHANED the run: the
process holding the GPU, the trainer, the worker pool and the replay buffer kept running with
nobody watching it. `arm_parent_death_if_supervised` closes that first hop; `f6b4bb0` already
closed the second (run -> eval worker).

WHY THE GATE IS HALF THE FIX, and why this file has as many negative rows as positive ones.
`PR_SET_PDEATHSIG` on a TOP-LEVEL process ties that process's life to whatever launched it. An
unconditional arm at `mantis.run.main` would therefore end every unattended burn whose
launching shell exits after handing off — and would arm the PYTEST PROCESS itself, through the
five in-process `main()` calls in `tests/test_run_launcher.py`. That is not hypothetical: it is
the incident `eae0fc4` records, measured A/B on this suite. So the rows below prove the arming
FIRES when a supervisor stamped the environment, and — just as load-bearing — that it does
NOTHING in every other shape.

HARNESS SHAPE, re-created locally rather than imported (R5 bars cross-test imports): real
script FILES and not nested `-c` strings, because the string version once produced an
`IndentationError` inside a grandchild and the rows failed for a quoting reason that looks
identical to the defect; a `_alive()` that reads `/proc` and treats a zombie as dead; and every
child self-limited by a bounded sleep, so a killed runner cannot leave a permanent survivor —
this file must never manufacture the class it exists to detect.

>300 justify (R8): ONE claim — "the run arms itself exactly when a supervisor spawned it" —
whose evidence is irreducibly PAIRED. Every arming row is only evidence because a matching
non-arming row on the SAME harness survives the same kill, so the positive and its control
have to share one `_write_two_generation_harness`, one `_alive`, one reaper. Splitting the
file would either duplicate that harness (two copies free to drift, and a drifted control
silently stops controlling anything) or import it across files, which R5 bars.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any

import pytest

from mantis.monitor.heartbeat import PARENT_DEATH_PPID_ENV
from mantis.train.lifecycle.signals import (
    PARENT_VANISHED_EXIT_CODE,
    arm_parent_death_if_supervised,
)

_LINUX = sys.platform.startswith("linux")
_DEADLINE_SEC = 20.0
_LINUX_ONLY = pytest.mark.skipif(
    not _LINUX, reason="PR_SET_PDEATHSIG is a Linux prctl; there is no equivalent here"
)


def _alive(pid: int) -> bool:
    """True iff `pid` names a live, non-zombie process. A reaped-but-unwaited child stays in
    the table as a zombie; counting one as alive fails a row for the wrong reason, and counting
    one as dead PASSES a row for the wrong reason, which is worse."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def _write_two_generation_harness(tmp_path, marker, *, stamp: str | None,
                                  cooperative_sigterm: bool = False):
    """Write the parent + grandchild scripts and return the parent script path.

    The grandchild calls the PRODUCTION `arm_parent_death_if_supervised()` — never a
    hand-rolled prctl, which would test the test rather than the fix. It is started with a NEW
    SESSION, so nothing aimed at the parent's process group can reach it: whatever kills it has
    to be the kernel acting on the parent-death signal, which is the only thing that makes the
    negative control below mean anything.

    `stamp` is what the parent writes into the grandchild's environment under
    `PARENT_DEATH_PPID_ENV`: `"self"` means "the parent's own pid" (the supervised case) and
    `None` means the variable is absent (the direct-launch case).
    """
    child = tmp_path / f"child_{marker.stem}.py"
    child.write_text(
        "import os, signal, time\n"
        + ("signal.signal(signal.SIGTERM, lambda *a: None)\n" if cooperative_sigterm else "")
        + "from mantis.train.lifecycle.signals import arm_parent_death_if_supervised\n"
        "armed = arm_parent_death_if_supervised()\n"
        f"open({str(marker)!r}, 'w', encoding='utf-8').write("
        "str(os.getpid()) + ' ' + str(int(armed)))\n"
        # SELF-LIMITED, deliberately. The unstamped row MUST produce a process that outlives
        # its parent — that is the negative control — and its only reaper is that row's
        # `finally`. A bounded sleep is longer than any assertion here needs and short enough
        # that the worst case is litter with an expiry.
        "time.sleep(90)\n",
        encoding="utf-8",
    )
    stamp_line = (
        f"env[{PARENT_DEATH_PPID_ENV!r}] = str(os.getpid())\n" if stamp == "self"
        else (f"env[{PARENT_DEATH_PPID_ENV!r}] = {stamp!r}\n" if stamp is not None
              else f"env.pop({PARENT_DEATH_PPID_ENV!r}, None)\n")
    )
    parent = tmp_path / f"parent_{marker.stem}.py"
    parent.write_text(
        "import os, subprocess, sys, time\n"
        "env = dict(os.environ)\n"
        + stamp_line
        + f"subprocess.Popen([sys.executable, {str(child)!r}], env=env, "
        "start_new_session=True)\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    return parent


def _await_marker(marker, deadline_sec: float = _DEADLINE_SEC) -> tuple[int, bool]:
    deadline = time.monotonic() + deadline_sec
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists(), "the grandchild never started; this row proved nothing"
    raw = marker.read_text(encoding="utf-8").split()
    return int(raw[0]), bool(int(raw[1]))


def _kill_parent_and_wait(parent: subprocess.Popen[bytes]) -> None:
    parent.send_signal(signal.SIGKILL)   # the parent gets NO chance to clean up
    parent.wait(timeout=_DEADLINE_SEC)


def _reap(parent: subprocess.Popen[bytes], gc_pid: int | None) -> None:
    if parent.poll() is None:
        parent.kill()
        parent.wait(timeout=_DEADLINE_SEC)
    if gc_pid is not None and _alive(gc_pid):
        os.kill(gc_pid, signal.SIGKILL)


# ── the arming half ──────────────────────────────────────────────────────────────────────
@_LINUX_ONLY
def test_a_supervised_run_entry_dies_with_its_SIGKILLED_parent(tmp_path) -> None:
    """THE producer test. A parent stamps its own pid, spawns a grandchild that calls the
    production `arm_parent_death_if_supervised()` and then sleeps; the parent is SIGKILLed;
    the grandchild must be GONE.

    SIGKILL and not SIGTERM, deliberately: a SIGTERM'd parent could be credited to Python's
    `atexit`/`daemon=True` machinery and the row would pass while proving nothing about the
    mechanism it is named for."""
    marker = tmp_path / "supervised.pid"
    parent_script = _write_two_generation_harness(tmp_path, marker, stamp="self")
    parent = subprocess.Popen([sys.executable, str(parent_script)], cwd=os.getcwd())
    gc_pid = None
    try:
        gc_pid, armed = _await_marker(marker)
        assert armed, "the gate did not arm for a stamp naming the real parent"
        assert _alive(gc_pid), "the grandchild died before the parent was killed"

        _kill_parent_and_wait(parent)

        deadline = time.monotonic() + _DEADLINE_SEC
        while _alive(gc_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(gc_pid), (
            f"grandchild {gc_pid} SURVIVED its SIGKILLed supervisor — this is F-816-19: the "
            "run is now reparented and will hold the GPU, the pool and the buffer with nobody "
            "watching it"
        )
    finally:
        _reap(parent, gc_pid)


@_LINUX_ONLY
def test_an_UNSTAMPED_run_entry_SURVIVES_the_same_kill(tmp_path) -> None:
    """THE NEGATIVE CONTROL, and the load-bearing half of this file (LAW-07).

    Identical to the row above except the environment carries no stamp. It must SURVIVE. Two
    independent things ride on it: if it dies anyway, something else in the environment reaps
    every grandchild and the row above is green without the mechanism it claims to test; and it
    is the direct pin on the deliberate residual — a run launched WITHOUT a supervisor is still
    orphanable, because that detachment is the operator's own choice and the fix does not
    override it.

    It reaps its own survivor: leaving one would be this file writing the very defect it exists
    to detect."""
    marker = tmp_path / "unstamped.pid"
    parent_script = _write_two_generation_harness(tmp_path, marker, stamp=None)
    parent = subprocess.Popen([sys.executable, str(parent_script)], cwd=os.getcwd())
    gc_pid = None
    try:
        gc_pid, armed = _await_marker(marker)
        assert not armed, "the gate armed with no stamp present — the guard is inert"
        _kill_parent_and_wait(parent)
        time.sleep(2.0)
        assert _alive(gc_pid), (
            "the UNSTAMPED grandchild also died — then the supervised row above is not "
            "evidence for PR_SET_PDEATHSIG, and this file's mechanism claim is unproven"
        )
    finally:
        _reap(parent, gc_pid)


@_LINUX_ONLY
def test_a_stamp_naming_a_LIVE_non_parent_does_not_arm(tmp_path) -> None:
    """The wrapper case: a stamp that names a LIVE process which is not our parent (this is
    what an operator-supplied `-- sh -c "python -m mantis.run …"` produces). Arming there would
    tie the run's life to the WRAPPER, which nobody asked for.

    BOTH halves are asserted, and the second is why this row runs a real process rather than
    calling the function in-process: (i) the gate reports not-armed, and (ii) the grandchild
    SURVIVES its parent's SIGKILL — which is positive evidence that no `prctl` happened, where
    a bare return-value check would only show what the function said about itself. The stamp
    used is the pytest process's own pid, which is alive throughout and is the grandchild's
    grandparent, never its parent."""
    marker = tmp_path / "wrapper.pid"
    parent_script = _write_two_generation_harness(tmp_path, marker, stamp=str(os.getpid()))
    parent = subprocess.Popen([sys.executable, str(parent_script)], cwd=os.getcwd())
    gc_pid = None
    try:
        gc_pid, armed = _await_marker(marker)
        assert not armed, "a stamp naming a live NON-parent must not arm"
        _kill_parent_and_wait(parent)
        time.sleep(2.0)
        assert _alive(gc_pid), (
            "the grandchild died although its stamp named a live non-parent — the gate armed "
            "on mere PRESENCE of the variable, which ties a run's life to any wrapper in the "
            "chain"
        )
    finally:
        _reap(parent, gc_pid)


@_LINUX_ONLY
def test_a_stamp_naming_a_DEAD_parent_exits_71_instead_of_running_unsupervised(
    tmp_path,
) -> None:
    """The immediate-orphan case: the supervisor died between its `Popen` and this line. The
    run must exit `PARENT_VANISHED_EXIT_CODE` rather than come up unsupervised — nothing is
    built, nothing can be saved, and no supervisor is left to relaunch or to watch.

    The dead pid is minted by running a process to completion and reaping it, then VERIFIED
    free with signal 0; if the kernel has already recycled it the row skips rather than
    asserting against a live stranger."""
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=_DEADLINE_SEC)
    dead_pid = dead.pid
    try:
        os.kill(dead_pid, 0)
    except ProcessLookupError:
        pass
    else:
        pytest.skip(f"pid {dead_pid} was recycled before it could be used as a dead stamp")

    probe = tmp_path / "probe.py"
    probe.write_text(
        "from mantis.train.lifecycle.signals import arm_parent_death_if_supervised\n"
        "arm_parent_death_if_supervised()\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    env = {**os.environ, PARENT_DEATH_PPID_ENV: str(dead_pid)}
    out = subprocess.run([sys.executable, str(probe)], cwd=os.getcwd(), env=env,
                         capture_output=True, text=True, timeout=_DEADLINE_SEC)
    assert out.returncode == PARENT_VANISHED_EXIT_CODE, (
        f"a stamp naming a DEAD supervisor must exit {PARENT_VANISHED_EXIT_CODE} cleanly, got "
        f"rc {out.returncode}; stderr:\n{out.stderr}"
    )


@_LINUX_ONLY
def test_a_child_that_HANDLES_sigterm_is_still_taken_down(tmp_path) -> None:
    """THE REGRESSION ROW, and it exists because an earlier version of the mechanism failed
    here. `arm_parent_death_signal` originally defaulted to SIGTERM; driven against a real boot
    child — which installs the cooperative save-then-exit handler — the signal ARRIVED and the
    child converted it into a PARK, still alive, still holding its memory.

    A run is exactly that shape: `compose_run` installs cooperative SIGINT/SIGTERM handlers. So
    the grandchild here swallows SIGTERM and must die anyway."""
    marker = tmp_path / "coop.pid"
    parent_script = _write_two_generation_harness(
        tmp_path, marker, stamp="self", cooperative_sigterm=True,
    )
    parent = subprocess.Popen([sys.executable, str(parent_script)], cwd=os.getcwd())
    gc_pid = None
    try:
        gc_pid, armed = _await_marker(marker)
        assert armed
        _kill_parent_and_wait(parent)
        deadline = time.monotonic() + _DEADLINE_SEC
        while _alive(gc_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(gc_pid), (
            f"grandchild {gc_pid} swallowed the parent-death signal and SURVIVED. A death "
            "signal a process can turn into a park is not a death signal, and a run installs "
            "exactly that cooperative handler"
        )
    finally:
        _reap(parent, gc_pid)


# ── the launcher's ordering, and its self-protection ─────────────────────────────────────
def test_main_arms_before_it_reads_anything(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The arming is `main`'s FIRST statement, ahead of argparse and of every collaborator.

    Ordering is the claim, so ordering is what is asserted: three collaborators write their own
    token into ONE list and the arming token must be at index 0. A mutant that moves the call
    below `load_config`/`launch_run` — i.e. after the config read and after the GPU and the
    thread pool exist — leaves the exact window this fix closes wide open, and reds here."""
    import mantis.run as mantis_run

    order: list[str] = []
    monkeypatch.setattr(
        mantis_run, "arm_parent_death_if_supervised",
        lambda: (order.append("armed"), False)[1],
    )
    monkeypatch.setattr(
        mantis_run, "load_config", lambda path: (order.append("load_config"), {})[1],
    )

    def _fake_launch(**_kw: Any) -> Any:
        order.append("launch_run")

        class _H:
            shutdown = type("S", (), {"abort_rule": None})()

        return _H()

    monkeypatch.setattr(mantis_run, "launch_run", _fake_launch)

    rc = mantis_run.main(["--config", str(tmp_path / "cfg.yaml"), "--out-dir", str(tmp_path)])
    assert rc == 0
    assert order and order[0] == "armed", (
        f"the parent-death arming must be main's first statement; observed order {order}"
    )
    assert "launch_run" in order, "the launcher never ran; this row proved nothing about order"


def test_calling_main_in_this_process_does_not_arm_the_test_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog,
) -> None:
    """THE SELF-PROTECTION ROW — the one that keeps this whole tier runnable when its launcher
    exits, and the direct pin on why the arming is GATED rather than unconditional.

    `tests/test_run_launcher.py` calls `main()` in this process five times. With no stamp in
    the environment the gate must take the not-supervised arm: return False, perform no
    `prctl`, and leave `os.environ` byte-identical — a leaked stamp would arm every later child
    of the pytest process. An unconditional-arming mutant fails this row loudly instead of
    SIGKILLing the tier from outside, where no assertion can see it."""
    import mantis.run as mantis_run

    monkeypatch.delenv(PARENT_DEATH_PPID_ENV, raising=False)
    before = dict(os.environ)

    with caplog.at_level(logging.DEBUG, logger="mantis.train.lifecycle.signals"):
        assert arm_parent_death_if_supervised() is False

    monkeypatch.setattr(mantis_run, "load_config", lambda path: {})
    monkeypatch.setattr(
        mantis_run, "launch_run",
        lambda **_kw: type("H", (), {"shutdown": type("S", (), {"abort_rule": None})()})(),
    )
    assert mantis_run.main(
        ["--config", str(tmp_path / "cfg.yaml"), "--out-dir", str(tmp_path)]
    ) == 0
    assert dict(os.environ) == before, (
        "calling main() mutated the process environment; a leaked parent-death stamp arms "
        "every later child of this process"
    )
