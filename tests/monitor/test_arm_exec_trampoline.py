"""Q3 red-team A4b — the arming trampoline, driven as real process chains.

THE DEFECT, MEASURED. `uv run` does not `exec`. Under a plain `Popen` the supervisor's DIRECT
child was therefore the wrapper and the run — the process holding the GPU, the trainer, the
worker pool and the buffer — was a GRANDCHILD nobody had promised to kill: `kill -9` on the
supervisor of `-- uv run python -m mantis.run` left it alive and reparented.

THE FIX IS TWO HALVES THAT ARE INERT APART, which is why they are tested together here:
  * the trampoline arms `PR_SET_PDEATHSIG` and then `execvp`s the child, and the arming
    survives `execve` — so the WRAPPER dies with the supervisor;
  * the run's own gate arms against its DIRECT parent when the stamped supervisor is its
    grandparent — so the run dies with the wrapper.
Either half alone leaves the run alive, and both degenerate cases have their own row below.

CI MUST NOT DEPEND ON A THIRD-PARTY LAUNCHER'S SEMANTICS, so the non-exec wrapper is a
four-line Python spawner written to `tmp_path` — the exact shape the design measured `uv run`
to have (fork, do not exec, wait). Every process in every chain is self-limited by a bounded
sleep, so a row that dies mid-way cannot leave a permanent survivor: this file must never
manufacture the class it exists to detect.

>300 justify (R8): ONE claim — "a run launched through the trampoline dies with its supervisor,
and the design's own depth-2 boundary is where it stops" — whose rows are only evidence in
PAIRS. Each positive (the chain dies) means nothing without the negative control on the SAME
chain builder (the same chain WITHOUT the trampoline survives), and the residual row is the
same builder with one more wrapper. Splitting the file would duplicate the chain builder and
the pid reaper, and a drifted copy of the control silently stops controlling anything.
"""
from __future__ import annotations

import inspect
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mantis.monitor.heartbeat import PARENT_DEATH_ARM_EXEC_MODULE, PARENT_DEATH_PPID_ENV
from mantis.monitor.supervise import spawn_child

_LINUX = sys.platform.startswith("linux")
_DEADLINE_SEC = 30.0
_LINUX_ONLY = pytest.mark.skipif(
    not _LINUX, reason="PR_SET_PDEATHSIG and /proc ancestry are Linux-specific here"
)


def _alive(pid: int | None) -> bool:
    """True iff `pid` names a live, non-zombie process. Counting a zombie as alive fails a row
    for the wrong reason; counting one as dead PASSES a row for the wrong reason, which is
    worse."""
    if pid is None:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def _ppid_of(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _write_wrapper(tmp_path: Path) -> Path:
    """The non-exec wrapper: fork a child, wait for it, never `exec`. This is `uv run`'s
    measured shape and the entire reason the trampoline exists."""
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import subprocess, sys\n"
        "p = subprocess.Popen(sys.argv[1:])\n"
        "p.wait()\n",
        encoding="utf-8",
    )
    return wrapper


def _write_leaf(tmp_path: Path, marker: Path, *, arm: bool) -> Path:
    """The run stand-in. `arm=True` calls the PRODUCTION gate — never a hand-rolled prctl,
    which would test the test rather than the fix."""
    leaf = tmp_path / f"leaf_{marker.stem}.py"
    reasons = marker.parent / (marker.name + ".reasons")
    arm_lines = (
        f"import logging\nlogging.basicConfig(filename={str(reasons)!r}, "
        "level=logging.DEBUG, force=True)\n"
        "from mantis.train.lifecycle.signals import arm_parent_death_if_supervised\n"
        "armed = arm_parent_death_if_supervised()\n"
        if arm else "armed = False\n"
    )
    leaf.write_text(
        "import os, sys, time\n"
        + arm_lines
        + f"open({str(marker)!r}, 'w', encoding='utf-8').write("
        "str(os.getpid()) + ' ' + str(int(armed)))\n"
        "time.sleep(90)\n",
        encoding="utf-8",
    )
    return leaf


def _write_supervisor_stub(tmp_path: Path, pidfile: Path, *, trampoline: bool) -> Path:
    """A supervisor stand-in that launches its child through the PRODUCTION `spawn_child`
    (`trampoline=True`) or through the plain, pre-fix `Popen` shape (`trampoline=False`).

    The second is the NEGATIVE CONTROL and it is deliberately a hand-written copy of what
    `spawn_child` used to be: stamping the environment and starting a new session, i.e. every
    part of the fix EXCEPT the trampoline. Without it, a green positive row could be green
    because something else in the environment reaps the chain.
    """
    stub = tmp_path / f"sup_{'tramp' if trampoline else 'plain'}_{pidfile.stem}.py"
    launch = (
        "from mantis.monitor.supervise import spawn_child\n"
        "child = spawn_child(sys.argv[1:])\n"
        if trampoline else
        "env = {**os.environ, %r: str(os.getpid())}\n"
        "child = subprocess.Popen(sys.argv[1:], env=env, start_new_session=True)\n"
        % PARENT_DEATH_PPID_ENV
    )
    stub.write_text(
        "import os, subprocess, sys, time\n"
        + launch
        + f"open({str(pidfile)!r}, 'w', encoding='utf-8').write(str(child.pid))\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    return stub


def _await_file(path: Path, deadline_sec: float = _DEADLINE_SEC) -> str:
    deadline = time.monotonic() + deadline_sec
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return text
        time.sleep(0.05)
    raise AssertionError(f"{path.name} never appeared; the chain never came up, so this row "
                         "proved nothing")


def _await_gone(pid: int | None, deadline_sec: float = _DEADLINE_SEC) -> bool:
    deadline = time.monotonic() + deadline_sec
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _alive(pid)


def _reap(sup: subprocess.Popen[bytes], *pids: int | None) -> None:
    if sup.poll() is None:
        sup.kill()
        sup.wait(timeout=_DEADLINE_SEC)
    for pid in pids:
        if _alive(pid) and pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:      # pragma: no cover — already gone between poll and kill
                pass


def _run_chain(tmp_path: Path, chain: list[str], *, trampoline: bool,
               marker: Path) -> tuple[subprocess.Popen[bytes], int, int]:
    """Start `supervisor-stub -> chain…` and return (stub handle, first-child pid, leaf pid)."""
    pidfile = tmp_path / f"{marker.stem}.childpid"
    stub = _write_supervisor_stub(tmp_path, pidfile, trampoline=trampoline)
    sup = subprocess.Popen([sys.executable, str(stub), *chain], cwd=os.getcwd())
    first = int(_await_file(pidfile).strip())
    leaf_pid = int(_await_file(marker).split()[0])
    return sup, first, leaf_pid


def _armed(marker: Path) -> bool:
    return bool(int(marker.read_text(encoding="utf-8").split()[1]))


# ── the mechanism ────────────────────────────────────────────────────────────────────────
@_LINUX_ONLY
def test_the_trampoline_execs_into_the_program_it_was_given(tmp_path) -> None:
    """`execvp`, not a `Popen`: the program it becomes must report the trampoline's OWN pid and
    the verbatim tail as its `sys.argv`. A trampoline that SPAWNED its child instead would put
    a fork between the arming and the run — and `PR_SET_PDEATHSIG` is cleared across `fork`,
    so the whole mechanism would be inert while every other row still looked plausible."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "sys.stdout.write(json.dumps([os.getpid(), sys.argv[1:]]))\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", PARENT_DEATH_ARM_EXEC_MODULE, "--",
         sys.executable, str(probe), "--config", "x.yaml"],
        cwd=os.getcwd(), stdout=subprocess.PIPE,
    )
    out, _ = proc.communicate(timeout=_DEADLINE_SEC)
    import json
    pid, argv = json.loads(out.decode())
    assert pid == proc.pid, (
        f"the program reported pid {pid} but the trampoline was {proc.pid} — it forked instead "
        "of exec'ing, and an arming that a fork clears protects nothing"
    )
    assert argv == ["--config", "x.yaml"], f"the child argv was not passed verbatim: {argv}"


@_LINUX_ONLY
def test_a_wrapper_launched_through_the_trampoline_dies_with_the_supervisor(tmp_path) -> None:
    """THE A4b ROW, first half. The supervisor's direct child is the WRAPPER, and before this
    fix nothing killed it. Through the trampoline it carries the arming across its own `exec`
    and the kernel takes it down with the supervisor."""
    marker = tmp_path / "w1.marker"
    wrapper, leaf = _write_wrapper(tmp_path), _write_leaf(tmp_path, marker, arm=False)
    sup, wrapper_pid, leaf_pid = _run_chain(
        tmp_path, [sys.executable, str(wrapper), sys.executable, str(leaf)],
        trampoline=True, marker=marker,
    )
    try:
        sup.send_signal(signal.SIGKILL)
        sup.wait(timeout=_DEADLINE_SEC)
        assert _await_gone(wrapper_pid), (
            f"wrapper {wrapper_pid} survived the SIGKILLed supervisor — the trampoline did not "
            "arm, or the arming did not survive its exec"
        )
    finally:
        _reap(sup, wrapper_pid, leaf_pid)


@_LINUX_ONLY
def test_the_run_under_one_wrapper_dies_with_the_supervisor(tmp_path) -> None:
    """THE A4b ROW, end to end and with the production gate in the leaf. Supervisor SIGKILLed →
    kernel kills the armed wrapper → the wrapper's death fires the run's own depth-2 arming →
    the run is gone. This is the shape `-- uv run python -m mantis.run` produces."""
    marker = tmp_path / "w2.marker"
    wrapper, leaf = _write_wrapper(tmp_path), _write_leaf(tmp_path, marker, arm=True)
    sup, wrapper_pid, leaf_pid = _run_chain(
        tmp_path, [sys.executable, str(wrapper), sys.executable, str(leaf)],
        trampoline=True, marker=marker,
    )
    try:
        assert _armed(marker), "the run under one wrapper did not arm; the depth-2 gate is inert"
        sup.send_signal(signal.SIGKILL)
        sup.wait(timeout=_DEADLINE_SEC)
        assert _await_gone(leaf_pid), (
            f"run {leaf_pid} SURVIVED its SIGKILLed supervisor through one wrapper — this is "
            "A4b: the GPU holder is now reparented and running with nobody watching it"
        )
    finally:
        _reap(sup, wrapper_pid, leaf_pid)


@_LINUX_ONLY
def test_the_same_chain_WITHOUT_the_trampoline_leaves_the_run_alive(tmp_path) -> None:
    """THE NEGATIVE CONTROL, and the load-bearing half of this file. The identical chain
    launched by the pre-fix `Popen` shape must SURVIVE. Two things ride on it: if it dies
    anyway, the rows above are green because something else reaps the chain and prove nothing;
    and it is the direct measurement of the defect the packet claims to close."""
    marker = tmp_path / "w3.marker"
    wrapper, leaf = _write_wrapper(tmp_path), _write_leaf(tmp_path, marker, arm=True)
    sup, wrapper_pid, leaf_pid = _run_chain(
        tmp_path, [sys.executable, str(wrapper), sys.executable, str(leaf)],
        trampoline=False, marker=marker,
    )
    try:
        sup.send_signal(signal.SIGKILL)
        sup.wait(timeout=_DEADLINE_SEC)
        time.sleep(2.0)
        assert _alive(wrapper_pid) and _alive(leaf_pid), (
            "the un-trampolined chain died too — then the rows above are not evidence for the "
            "trampoline, and this file's mechanism claim is unproven"
        )
    finally:
        _reap(sup, wrapper_pid, leaf_pid)


@_LINUX_ONLY
def test_a_direct_launch_through_the_trampoline_still_dies(tmp_path) -> None:
    """The no-wrapper regression. The trampoline must not have broken the case that already
    worked — and here it is the STRONGER shape: the arming is in place from the leaf's first
    instruction, before any import it would otherwise be gated behind."""
    marker = tmp_path / "d1.marker"
    leaf = _write_leaf(tmp_path, marker, arm=True)
    sup, leaf_pid_reported, leaf_pid = _run_chain(
        tmp_path, [sys.executable, str(leaf)], trampoline=True, marker=marker,
    )
    try:
        assert leaf_pid_reported == leaf_pid, (
            "the trampoline's pid and the leaf's pid must be the SAME process after exec"
        )
        sup.send_signal(signal.SIGKILL)
        sup.wait(timeout=_DEADLINE_SEC)
        assert _await_gone(leaf_pid), f"directly launched run {leaf_pid} survived its supervisor"
    finally:
        _reap(sup, leaf_pid)


@_LINUX_ONLY
def test_two_stacked_wrappers_are_NOT_armed_and_say_so(tmp_path) -> None:
    """THE DISCLOSED RESIDUAL, measured and NAMED rather than assumed (RQ-9).

    With two non-exec wrappers the run is three hops from the supervisor, so there is an
    UNARMED process between it and the one the trampoline armed: the cascade cannot reach it.
    The gate must refuse to arm — arming against the nearer wrapper would tie the run to a
    process nobody promised to kill — and it must name `wrapper_chain_too_deep`, because the
    A6 finding was precisely that this decision was invisible after the fact."""
    marker = tmp_path / "deep.marker"
    wrapper, leaf = _write_wrapper(tmp_path), _write_leaf(tmp_path, marker, arm=True)
    sup, first_pid, leaf_pid = _run_chain(
        tmp_path,
        [sys.executable, str(wrapper), sys.executable, str(wrapper), sys.executable, str(leaf)],
        trampoline=True, marker=marker,
    )
    second_pid = _ppid_of(leaf_pid)
    try:
        assert not _armed(marker), (
            "a run three hops from its supervisor must NOT arm — the process it would arm "
            "against is one nothing has promised to kill"
        )
        reasons = (marker.parent / (marker.name + ".reasons")).read_text(encoding="utf-8")
        assert "wrapper_chain_too_deep" in reasons, (
            f"the residual must be NAMED in the run's own record; got:\n{reasons}"
        )
        sup.send_signal(signal.SIGKILL)
        sup.wait(timeout=_DEADLINE_SEC)
        time.sleep(2.0)
        assert _alive(leaf_pid), (
            "the depth-3 run died anyway — then this row is not measuring the residual it "
            "claims to measure and the boundary is somewhere else"
        )
    finally:
        _reap(sup, first_pid, second_pid, leaf_pid)


# ── the trampoline's own failure surface ─────────────────────────────────────────────────
@pytest.mark.parametrize("tail", [[], ["python", "-c", "pass"]])
def test_the_trampoline_refuses_an_argv_with_no_separator(tail) -> None:
    """Usage refusal, mirroring `supervise._split_argv`. A trampoline that guessed where its
    own flags end would silently swallow the first word of the child command."""
    out = subprocess.run(
        [sys.executable, "-m", PARENT_DEATH_ARM_EXEC_MODULE, *tail],
        cwd=os.getcwd(), capture_output=True, text=True, timeout=_DEADLINE_SEC,
    )
    assert out.returncode != 0, "an argv with no `--` must be refused, not guessed at"
    assert "usage" in out.stderr.lower() or "--" in out.stderr


def test_an_unresolvable_program_fails_in_the_supervisor_not_the_child() -> None:
    """With a trampoline in front, a typo'd child command would make `Popen` SUCCEED and move
    the failure into the child — turning a launcher typo into a `child_error` rc the supervisor
    reads as the run's own diagnosis. `spawn_child` resolves the program itself, so the failure
    stays exactly where it was before the trampoline existed: loud, and in the supervisor."""
    with pytest.raises(FileNotFoundError):
        spawn_child(["mantis-no-such-program-q3-a4b", "--config", "x"])


def test_the_trampoline_module_name_is_the_contract_constant() -> None:
    """One spelling of the contract. A hard-coded second copy of the module path in
    `spawn_child` is drift that no import-DAG gate can see — the coupling is a STRING, which
    is exactly why it needs a test of its own."""
    source = inspect.getsource(spawn_child)
    assert "PARENT_DEATH_ARM_EXEC_MODULE" in source, (
        "`spawn_child` must name the trampoline by the contract constant"
    )
    assert PARENT_DEATH_ARM_EXEC_MODULE not in source, (
        "the module path is hard-coded a second time; the constant is then decorative"
    )
