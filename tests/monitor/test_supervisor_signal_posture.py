"""Q3 red-team A2 — the supervisor's own signal posture, driven as real processes.

THE DEFECT. `mantis.monitor.supervise` installed NO signal handlers. Its own catchable death —
an operator's `kill`, a closed terminal, a `BrokenPipeError` out of `_emit` — therefore ended it
instantly, and with the run now armed against it (F-816-19) the KERNEL SIGKILLed a healthy run
mid-save. The run's own save-then-exit path (LAW-16) was reachable only by signalling the RUN
directly, which is not what an operator supervising a run does. The fix forwards ONE SIGTERM,
waits out the existing kill grace, escalates, and dies of the signal it was asked to die of.

THE SECOND HALF IS THE PROCESS GROUP, and it is why `test_ctrl_c_on_the_process_group_reaches_
the_child_exactly_once` is in this file. Fixing the handlers ALONE converts a `Ctrl-C` from
"SIGKILLed mid-save" into "force-exited mid-save": the tty delivers SIGINT to the run directly
AND the supervisor forwards SIGTERM, and the run's handler cannot tell two routes from an
operator's deliberate second press, which is LAW-16's `os._exit(1)`. `spawn_child`'s
`start_new_session=True` removes the double delivery at the source.

`main` IS DRIVEN ONLY IN A REAL SUBPROCESS, never in-process: it installs process-wide signal
handlers, and an in-process row would install them in the pytest runner. Every child here is
bounded by its own timer, so a row that dies mid-way cannot leave a permanent survivor — this
file must not manufacture the class it exists to detect.

>300 justify (R8): ONE claim — "a supervisor asked to stop stops its child cooperatively first"
— whose rows are inseparable because they share ONE harness. Each row is a four-process shape
(pytest -> supervisor -> arming trampoline -> child) parameterised only by how the child answers
SIGTERM, and every row's evidence is a log written by that shared child script. Splitting would
duplicate the child writer and the event reader, and a drifted copy of either silently stops
witnessing the thing the other half asserts.
"""
from __future__ import annotations

import ast
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_LINUX = sys.platform.startswith("linux")
_DEADLINE_SEC = 45.0
_LINUX_ONLY = pytest.mark.skipif(
    not _LINUX, reason="process-group and PR_SET_PDEATHSIG semantics are Linux-specific here"
)


def _write_child(tmp_path: Path, log: Path, marker: Path, *, drain_sec: float,
                 swallow: bool = False, exit_code: int = 0) -> Path:
    """A child that LOGS every signal it receives, drains, then exits with `exit_code`.

    The log is the instrument: it is the only way to tell "one SIGTERM from the supervisor"
    from "a SIGINT from the tty AND a forwarded SIGTERM", which is the whole of the F-2 claim.
    `swallow` makes the child ignore the request entirely, which is what the escalation row
    needs. The 180 s bound is the self-limit: nothing here may outlive the row that made it.
    """
    script = tmp_path / f"child_{marker.stem}.py"
    body = (
        "import os, signal, sys, time\n"
        f"LOG = {str(log)!r}\n"
        "def _log(text):\n"
        "    with open(LOG, 'a', encoding='utf-8') as fh:\n"
        "        fh.write(text + '\\n')\n"
        "        fh.flush()\n"
        "state = {'stop': False}\n"
        "def _on(signum, frame):\n"
        "    _log('SIG ' + str(int(signum)))\n"
        + ("" if swallow else "    state['stop'] = True\n")
        + "signal.signal(signal.SIGINT, _on)\n"
        "signal.signal(signal.SIGTERM, _on)\n"
        "_log('READY ' + str(os.getpid()))\n"
        "deadline = time.monotonic() + 180.0\n"
        "while not state['stop'] and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        f"time.sleep({float(drain_sec)!r})\n"
        f"_log('DRAINED')\n"
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('drained')\n"
        f"raise SystemExit({int(exit_code)})\n"
    )
    script.write_text(body, encoding="utf-8")
    return script


def _spawn_supervisor(tmp_path: Path, child_script: Path, err: Path, *,
                      kill_grace_sec: float = 15.0,
                      stale_after_sec: float = 600.0) -> subprocess.Popen[bytes]:
    """Run the REAL CLI in its own session, so a group signal in a row reaches it and nothing
    else — least of all the pytest process that sent it."""
    argv = [
        sys.executable, "-m", "mantis.monitor.supervise",
        "--heartbeat-file", str(tmp_path / "hb.json"),
        "--stale-after-sec", str(stale_after_sec),
        "--poll-interval-sec", "0.1",
        "--kill-grace-sec", str(kill_grace_sec),
        "--max-relaunches", "0",
        "--", sys.executable, str(child_script),
    ]
    handle = err.open("wb")
    return subprocess.Popen(argv, cwd=os.getcwd(), stderr=handle, start_new_session=True)


def _await_line(log: Path, needle: str, deadline_sec: float = _DEADLINE_SEC) -> str:
    deadline = time.monotonic() + deadline_sec
    while time.monotonic() < deadline:
        if log.exists():
            text = log.read_text(encoding="utf-8")
            if needle in text:
                return text
        time.sleep(0.05)
    got = log.read_text(encoding="utf-8") if log.exists() else "<no log>"
    raise AssertionError(f"{needle!r} never appeared in the child log; got:\n{got}")


def _events(err: Path) -> list[dict]:
    """The supervisor's own stream: one JSON line per action."""
    if not err.exists():
        return []
    rows = []
    for line in err.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _names(err: Path) -> list[str]:
    return [str(row.get("event")) for row in _events(err)]


def _signals(log: Path) -> list[int]:
    if not log.exists():
        return []
    return [int(line.split()[1]) for line in log.read_text(encoding="utf-8").splitlines()
            if line.startswith("SIG ")]


def _reap(proc: subprocess.Popen[bytes], *extra_pids: int) -> None:
    if proc.poll() is None:                     # pragma: no cover — only on a wedged row
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.kill()
        proc.wait(timeout=_DEADLINE_SEC)
    for pid in extra_pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def _child_pid(log: Path) -> int | None:
    for line in (log.read_text(encoding="utf-8").splitlines() if log.exists() else []):
        if line.startswith("READY "):
            return int(line.split()[1])
    return None


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


# ── the A2a rows ─────────────────────────────────────────────────────────────────────────
@_LINUX_ONLY
def test_a_SIGTERMed_supervisor_forwards_SIGTERM_and_waits_for_the_child(tmp_path) -> None:
    """THE A2a ROW. A SIGTERM to the supervisor must reach the child as ONE SIGTERM, and the
    supervisor must still be there when the child finishes draining.

    Two mutants die here: "install a handler and exit immediately" (the child is then SIGKILLed
    by the kernel through its own arming, and the marker never lands) and "forward SIGKILL"
    (the log shows no SIGTERM and, again, no marker)."""
    log, marker, err = tmp_path / "c.log", tmp_path / "c.marker", tmp_path / "sup.err"
    child = _write_child(tmp_path, log, marker, drain_sec=3.0)
    sup = _spawn_supervisor(tmp_path, child, err)
    try:
        _await_line(log, "READY")
        sup.send_signal(signal.SIGTERM)
        sup.wait(timeout=_DEADLINE_SEC)
        assert marker.exists(), (
            "the child never reached its post-drain marker — the supervisor did not wait for "
            "it, which is the A2a defect: a healthy run loses its save"
        )
        assert _signals(log) == [int(signal.SIGTERM)], (
            f"the child must receive exactly one SIGTERM; got {_signals(log)}"
        )
        assert "child_sigkilled" not in _names(err), (
            "a child that drained well inside the grace was SIGKILLed anyway"
        )
    finally:
        _reap(sup)


@_LINUX_ONLY
def test_the_supervisor_waits_for_a_child_that_takes_most_of_the_grace(tmp_path) -> None:
    """A nominal wait (`wait(timeout=0)`) would pass the row above on a fast child. Here the
    drain is most of the grace, so a nominal wait escalates and the marker never lands."""
    log, marker, err = tmp_path / "c.log", tmp_path / "c.marker", tmp_path / "sup.err"
    child = _write_child(tmp_path, log, marker, drain_sec=3.0)
    sup = _spawn_supervisor(tmp_path, child, err, kill_grace_sec=5.0)
    try:
        _await_line(log, "READY")
        sup.send_signal(signal.SIGTERM)
        sup.wait(timeout=_DEADLINE_SEC)
        assert marker.exists(), "the drain was truncated inside its own grace"
        assert "child_sigkilled" not in _names(err)
    finally:
        _reap(sup)


@_LINUX_ONLY
def test_a_child_that_ignores_SIGTERM_is_SIGKILLED_after_the_grace_and_reaped(tmp_path) -> None:
    """The bound. A child that swallows SIGTERM must be SIGKILLed once the grace expires and
    the supervisor must LEAVE — an unbounded wait is a hang this ladder must not introduce.

    THE RC ROW (RED-TEAM addendum 2, BROKE-IT). This is the grace-timeout escalation path, and
    the child's `wait()` code here is `-9` — this supervisor's OWN SIGKILL, not a diagnosis.
    Before the fix that `-9` was propagated as `reason="child_error"`, so `main` returned `-9`
    and `SystemExit(-9)` became exit status 247. The supervisor must still die OF the SIGTERM
    the operator sent IT, and say so honestly: `reason="signal"`, never `child_error`."""
    log, marker, err = tmp_path / "c.log", tmp_path / "c.marker", tmp_path / "sup.err"
    child = _write_child(tmp_path, log, marker, drain_sec=0.0, swallow=True)
    sup = _spawn_supervisor(tmp_path, child, err, kill_grace_sec=2.0)
    pid = None
    try:
        _await_line(log, "READY")
        pid = _child_pid(log)
        sup.send_signal(signal.SIGTERM)
        sup.wait(timeout=_DEADLINE_SEC)
        assert "child_sigkilled" in _names(err), (
            f"a child that ignored SIGTERM was never escalated; events={_names(err)}"
        )
        deadline = time.monotonic() + 10.0
        while pid is not None and _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid is None or not _alive(pid), "the SIGKILLed child was never reaped"
        assert not marker.exists(), "a swallowing child must not have reached its drain marker"
        assert sup.returncode == -int(signal.SIGTERM), (
            "the supervisor must die OF the SIGTERM it was sent, not return the child's own "
            f"-SIGKILL as a diagnosis (SystemExit(-9) is exit status 247); rc={sup.returncode}"
        )
        stops = [row for row in _events(err) if row.get("event") == "supervisor_stop"]
        assert stops and stops[-1].get("reason") == "signal", (
            f"a self-inflicted SIGKILL must not be mislabelled child_error; got {stops}"
        )
    finally:
        _reap(sup, *(p for p in (pid,) if p is not None))


# ── the F-2 row ──────────────────────────────────────────────────────────────────────────
@_LINUX_ONLY
def test_ctrl_c_on_the_process_group_reaches_the_child_exactly_once(tmp_path) -> None:
    """THE F-2 ROW, and the one that proves the handler fix ALONE is insufficient.

    A `Ctrl-C` is a signal to a process GROUP. Without `start_new_session=True` on the child,
    the run receives the tty's SIGINT directly AND the supervisor's forwarded SIGTERM — two
    entries in the log below — and the run's own handler reads the second as the operator's
    force-exit press: `force_teardown_all()` then `os._exit(1)`, mid-save. Against a build
    without the new session this row reds with two entries, which is exactly its job."""
    log, marker, err = tmp_path / "c.log", tmp_path / "c.marker", tmp_path / "sup.err"
    child = _write_child(tmp_path, log, marker, drain_sec=1.0)
    sup = _spawn_supervisor(tmp_path, child, err)
    try:
        _await_line(log, "READY")
        os.killpg(os.getpgid(sup.pid), signal.SIGINT)
        sup.wait(timeout=_DEADLINE_SEC)
        assert _signals(log) == [int(signal.SIGTERM)], (
            f"the child saw {_signals(log)} — a group signal reached it directly as well as "
            "through the supervisor, so the run counts two presses and force-exits mid-save"
        )
        assert marker.exists(), "the child never completed its cooperative drain"
    finally:
        _reap(sup)


# ── LAW-16 at the supervisor ─────────────────────────────────────────────────────────────
@_LINUX_ONLY
def test_a_second_signal_force_stops_and_the_supervisor_still_exits(tmp_path) -> None:
    """LAW-16's two-press affordance, mirrored. The second press re-forwards SIGTERM — the
    run's OWN second press, so its `force_teardown_all` still runs — and the ladder stays
    bounded, so everything is gone inside the deadline."""
    log, marker, err = tmp_path / "c.log", tmp_path / "c.marker", tmp_path / "sup.err"
    child = _write_child(tmp_path, log, marker, drain_sec=30.0)
    sup = _spawn_supervisor(tmp_path, child, err, kill_grace_sec=3.0)
    pid = None
    try:
        _await_line(log, "READY")
        pid = _child_pid(log)
        sup.send_signal(signal.SIGTERM)
        time.sleep(0.5)
        sup.send_signal(signal.SIGTERM)
        sup.wait(timeout=_DEADLINE_SEC)
        assert "supervisor_force_stop" in _names(err), (
            f"the second press was swallowed; events={_names(err)}"
        )
        assert _signals(log).count(int(signal.SIGTERM)) == 2, (
            f"the child must see the second press too; got {_signals(log)}"
        )
        deadline = time.monotonic() + 10.0
        while pid is not None and _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid is None or not _alive(pid), "the child outlived a force-stopped supervisor"
    finally:
        _reap(sup, *(p for p in (pid,) if p is not None))


@_LINUX_ONLY
def test_a_third_signal_force_kills_without_waiting_out_the_grace(tmp_path) -> None:
    """The THIRD press: `presses >= 3` short-circuits straight to SIGKILL instead of waiting out
    the (here, generous) grace — the operator's escalation must not be made to wait on it.

    THE RC ROW (RED-TEAM addendum 2, BROKE-IT). This is the second site of the same defect as
    the grace-timeout row above: the reaped child's code is `-9`, this supervisor's OWN SIGKILL,
    and must not be relabelled as the child's diagnosis. The supervisor still dies OF the
    SIGTERM the operator sent it on the FIRST press — the one that unwound the poll loop —
    regardless of how many further presses followed."""
    log, marker, err = tmp_path / "c.log", tmp_path / "c.marker", tmp_path / "sup.err"
    child = _write_child(tmp_path, log, marker, drain_sec=0.0, swallow=True)
    sup = _spawn_supervisor(tmp_path, child, err, kill_grace_sec=90.0)
    pid = None
    try:
        _await_line(log, "READY")
        pid = _child_pid(log)
        sup.send_signal(signal.SIGTERM)
        time.sleep(0.3)
        sup.send_signal(signal.SIGTERM)
        time.sleep(0.3)
        sup.send_signal(signal.SIGTERM)
        sup.wait(timeout=_DEADLINE_SEC)
        assert "supervisor_force_kill" in _names(err), (
            f"the third press must force-kill without waiting out the grace; "
            f"events={_names(err)}"
        )
        deadline = time.monotonic() + 10.0
        while pid is not None and _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid is None or not _alive(pid), "the third-press SIGKILL was never reaped"
        assert sup.returncode == -int(signal.SIGTERM), (
            "the supervisor must die OF the SIGTERM from the first press, not the child's own "
            f"-SIGKILL relabelled as a diagnosis (SystemExit(-9) is exit status 247); "
            f"rc={sup.returncode}"
        )
        stops = [row for row in _events(err) if row.get("event") == "supervisor_stop"]
        assert stops and stops[-1].get("reason") == "signal", (
            f"a third-press SIGKILL must not be mislabelled child_error; got {stops}"
        )
    finally:
        _reap(sup, *(p for p in (pid,) if p is not None))


# ── the rc contract ──────────────────────────────────────────────────────────────────────
@_LINUX_ONLY
def test_a_clean_signal_stop_leaves_the_run_resolving_to_zero(tmp_path) -> None:
    """§2.4: nothing is minted. The child resolved to 0, so the supervisor has no diagnosis to
    carry and dies OF the signal it was asked to die of — its waiter sees "terminated by
    SIGTERM", which is the truth. Mutants returning 0, 44 or 137 all red here."""
    log, marker, err = tmp_path / "c.log", tmp_path / "c.marker", tmp_path / "sup.err"
    child = _write_child(tmp_path, log, marker, drain_sec=0.2, exit_code=0)
    sup = _spawn_supervisor(tmp_path, child, err)
    try:
        _await_line(log, "READY")
        sup.send_signal(signal.SIGTERM)
        sup.wait(timeout=_DEADLINE_SEC)
        assert sup.returncode == -int(signal.SIGTERM), (
            f"the supervisor must die OF the signal, not return a number; rc={sup.returncode}"
        )
        stops = [row for row in _events(err) if row.get("event") == "supervisor_stop"]
        assert stops and stops[-1].get("reason") == "signal", (
            f"the stop must be named on the supervisor's own stream; got {stops}"
        )
    finally:
        _reap(sup)


@_LINUX_ONLY
def test_a_child_that_exits_nonzero_during_the_stop_propagates_that_code(tmp_path) -> None:
    """The child's diagnosis OUTRANKS the stop gesture. A disk-guard 47 recorded during the
    drain must not be erased by the fact that an operator also pressed Ctrl-C, so "always
    re-raise the signal" is the mutant this row kills."""
    log, marker, err = tmp_path / "c.log", tmp_path / "c.marker", tmp_path / "sup.err"
    child = _write_child(tmp_path, log, marker, drain_sec=0.2, exit_code=47)
    sup = _spawn_supervisor(tmp_path, child, err)
    try:
        _await_line(log, "READY")
        sup.send_signal(signal.SIGTERM)
        sup.wait(timeout=_DEADLINE_SEC)
        assert sup.returncode == 47, (
            f"a nonzero child code must be propagated through the stop; rc={sup.returncode}"
        )
    finally:
        _reap(sup)


# ── the escaping-exception path ──────────────────────────────────────────────────────────
@_LINUX_ONLY
def test_an_exception_escaping_the_loop_stops_the_child_cooperatively(tmp_path) -> None:
    """THE "`_emit` BrokenPipe MUST NOT SIGKILL THE RUN MID-SAVE" ROW.

    `_emit` writes to stderr on every spawn/exit/stale event. With the log consumer gone that
    write raises `BrokenPipeError`, which used to unwind the supervisor's main thread in
    milliseconds — and with the run armed against it, the kernel then SIGKILLed a healthy run
    mid-save. Here the consumer is closed only AFTER the child is up, and the heartbeat file is
    never written, so the next emit the supervisor attempts is the staleness one — a
    deterministic trigger that lands while a real, running child exists to be saved.

    The supervisor must still die LOUD: its stderr is the broken thing, so "loud" is a nonzero
    rc here, and the exception is re-raised unchanged rather than converted into a clean stop."""
    log, marker = tmp_path / "c.log", tmp_path / "c.marker"
    child = _write_child(tmp_path, log, marker, drain_sec=1.0)
    read_fd, write_fd = os.pipe()
    argv = [
        sys.executable, "-m", "mantis.monitor.supervise",
        "--heartbeat-file", str(tmp_path / "hb.json"),
        "--stale-after-sec", "3", "--poll-interval-sec", "0.1",
        "--kill-grace-sec", "10", "--max-relaunches", "0",
        "--", sys.executable, str(child),
    ]
    sup = subprocess.Popen(argv, cwd=os.getcwd(), stderr=write_fd, start_new_session=True)
    os.close(write_fd)
    try:
        _await_line(log, "READY")
        os.close(read_fd)                  # the log consumer goes away, mid-run
        sup.wait(timeout=_DEADLINE_SEC)
        assert sup.returncode not in (0, None), (
            f"a BrokenPipeError out of _emit must still be a loud death; rc={sup.returncode}"
        )
        _await_line(log, "DRAINED")
        assert _signals(log) == [int(signal.SIGTERM)], (
            f"the child must have been stopped COOPERATIVELY on the way out; got "
            f"{_signals(log)}"
        )
        assert marker.exists(), "the child was killed before it could finish its save"
    finally:
        _reap(sup)


@_LINUX_ONLY
def test_a_stop_with_no_child_yet_exits_without_touching_a_null_handle(tmp_path) -> None:
    """The null-handle guard for the attribute the fix adds. A stop that arrives before the
    first spawn must be a clean death of the signal, never an `AttributeError` on `child is
    None` — which is the crash the new `self.child` introduces if unguarded.

    Driven in a subprocess because the path ends in `SIG_DFL` + re-raise, which would kill the
    test runner."""
    probe = tmp_path / "nullchild.py"
    probe.write_text(
        "import signal, time\n"
        "from mantis.monitor import supervise\n"
        "sup = supervise.Supervisor(child_argv=['true'], heartbeat_file='hb.json',\n"
        "    stale_after_sec=1.0, poll_interval_sec=0.1, kill_grace_sec=1.0,\n"
        "    max_relaunches=0, spawn_fn=lambda argv: None, kill_fn=lambda c, s: None,\n"
        "    clock=time.monotonic)\n"
        "assert sup.child is None\n"
        "supervise._install_stop_handlers()\n"
        "supervise._stop_and_exit(sup, supervise._SupervisorStop(signal.SIGTERM, 1))\n",
        encoding="utf-8",
    )
    out = subprocess.run([sys.executable, str(probe)], cwd=os.getcwd(), capture_output=True,
                         text=True, timeout=_DEADLINE_SEC)
    assert out.returncode == -int(signal.SIGTERM), (
        f"a stop with no child must die of the signal; rc={out.returncode}\n{out.stderr}"
    )
    assert "AttributeError" not in out.stderr, out.stderr


# ── the import-time refusal ──────────────────────────────────────────────────────────────
def test_the_stop_handlers_are_installed_by_main_and_never_at_import(tmp_path) -> None:
    """`signal.signal` at module scope would install handlers in every process that imports
    this module — pytest included, where a raised `_SupervisorStop` would end the tier.

    Static half: no `signal.signal(` call at module scope. Dynamic half: a subprocess that
    imports the module and finds SIGTERM still on its default disposition."""
    from mantis.monitor import supervise

    source = Path(supervise.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_scope = [n for n in tree.body
                    if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)]
    for node in module_scope:
        for call in ast.walk(node):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "signal"
                    and getattr(call.func.value, "id", None) == "signal"):
                raise AssertionError(
                    "signal.signal() at module scope arms every importer of this module"
                )

    probe = tmp_path / "import_probe.py"
    probe.write_text(
        "import signal\n"
        "import mantis.monitor.supervise  # noqa: F401\n"
        "assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, 'import installed a handler'\n"
        "assert signal.getsignal(signal.SIGINT) in (signal.default_int_handler, signal.SIG_DFL)\n",
        encoding="utf-8",
    )
    out = subprocess.run([sys.executable, str(probe)], cwd=os.getcwd(), capture_output=True,
                         text=True, timeout=_DEADLINE_SEC)
    assert out.returncode == 0, out.stderr


def test_the_ladder_is_a_module_function_and_the_frozen_kill_path_is_untouched() -> None:
    """The structural claim the fix rests on: `Supervisor._kill` may NOT grow a `wait()`.

    `tests/monitor/test_supervisor.py` is frozen and drives `_kill` with a `FakeChild` that has
    `.pid` and `.poll()` and nothing else, so a `wait()` there is an `AttributeError` in a HELD
    oracle. The bounded, early-returning wait therefore lives in a module-level function over
    the real `Popen`. A future edit that moves it back into the class reds here BEFORE it reds
    the frozen file."""
    import inspect

    from mantis.monitor import supervise

    assert callable(supervise.stop_child_cooperatively)
    kill_src = inspect.getsource(supervise.Supervisor._kill)
    assert ".wait(" not in kill_src, (
        "`Supervisor._kill` grew a wait() — the frozen oracle's FakeChild has no such method"
    )
    assert ".wait(" in inspect.getsource(supervise.stop_child_cooperatively), (
        "the cooperative ladder must WAIT for the child; an unwaited SIGTERM saves nothing"
    )
