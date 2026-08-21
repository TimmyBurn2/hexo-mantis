"""F-816-24 — the supervisor reads the MINTED config, and the witness proves it in a live process.

ORACLE-FIRST (⊕): every test here is written and reviewed BEFORE the fix and is RED against
`dev` = `e5abab4`, where `main` builds a bare `MonitorConfig()` (`supervise.py:614`) and offers
no `--config` at all. The four minted `monitor.supervisor_*` keys reach no process on that head.

WHY THE WITNESS OBSERVES `supervisor_forwarding_stop.grace_sec` AND NOT THE CONFIG OBJECT. The
defect being fixed is precisely that the config and the process DISAGREED, so a test that reads
the value back out of the config object it just built proves nothing (R291(b)(iii)). That event
is emitted by `stop_child_cooperatively`, whose `grace_sec` argument is `supervisor._kill_grace`
passed AT THE STOP SITE by `_stop_and_exit` — so the assertion observes the bound where the
ladder CONSUMES it, in a real `-m` process, on the real SIGINT path. It is also not a timing
measurement: the event is emitted before the ladder waits, so nothing here races a clock.

>300 justify (R8): ONE claim — *the minted config reaches the running supervisor* — and every
test here is a face of it that only means something beside the others. They share one harness
(mint → real `-m` drive → SIGINT → read the supervisor's own stream), and the harness is the
expensive, defect-prone part: a split would copy it and the copies would drift, which is how the
witness stops witnessing. The refusal, the override and the two O-18 halves are not separate
subjects; they are the boundary conditions that make the central assertion mean what it says.

THE DISTINCTIVE VALUE IS THE POINT. Every committed config and both templates mint 30.0, which
is also the dataclass literal; `MonitorSchemaConfig.supervisor_kill_grace_sec` carries no schema
default. A witness using 30.0 would pass whether or not the fix works. `_GRACE` is a value no
authority in the tree can produce by accident.
"""
from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mantis.monitor.supervise import RELAUNCH_BUDGET_EXIT_CODE

REPO_ROOT = Path(__file__).resolve().parents[2]
MINT = REPO_ROOT / "tools" / "mint_config.py"
SUPERVISE_SRC = REPO_ROOT / "src" / "mantis" / "monitor" / "supervise.py"

#: Minted into the test config. Not 30.0 (every committed config + both templates + the
#: dataclass literal), not 900.0, not any schema default — there is none for this key.
_GRACE = 7.125
#: Supplied on the command line by the override test, and distinct from `_GRACE` so the two
#: cannot be confused for one another in an assertion.
_GRACE_OVERRIDE = 4.25
#: Also minted, and also distinctive: the drives need a sub-second cadence, and taking it from
#: the config rather than a flag means a SECOND minted key must reach the process.
_POLL = 0.075
#: ALL FOUR subject keys are minted distinctively, and all four are asserted. RED-TEAM broke
#: `stale_after_sec` and `max_relaunches` back to their dataclass literals (900.0 / 5) inside
#: `main` and the whole suite stayed green — 40/40 — because nothing here looked at them. The
#: packet's subject is FOUR keys and §7's promise is that the witness "would fail without the
#: fix"; that promise held for two of them. A witness that covers half its subject is the
#: coverage half of the phantom class: real, passing, and silent about what it never checked.
#: Deliberately LARGE so the staleness rule cannot fire during a drive whose subject is the stop
#: ladder — the child never writes a heartbeat file, so a small value here would kill and relaunch
#: it mid-test and quietly change what the other assertions are measuring.
_STALE = 611.25
_RELAUNCHES = 3
#: Small, distinctive, and used ONLY by the staleness witness below, which needs the rule to fire.
_STALE_FAST = 0.625
#: The staleness drive is bounded by the MINTED deadline plus slack, not by the generic deadline:
#: a supervisor waiting the dataclass 900.0 must be reported as a wrong bound within seconds
#: rather than surfacing as a minute-long `TimeoutExpired` from inside `subprocess`.
_STALE_DEADLINE_SEC = 20.0

_DEADLINE_SEC = 60.0
#: A refusal is a decision taken before anything is spawned, so it is fast or it is not a
#: refusal. Kept well under `_DEADLINE_SEC` so a supervisor that starts supervising instead of
#: refusing is reported as the wrong BEHAVIOUR rather than as a slow test.
_REFUSAL_DEADLINE_SEC = 20.0


def _mint(tmp_path: Path, **deltas: object) -> Path:
    """A MINTED config (R1: minted via the tool, never hand-varied), written to `tmp_path`.

    Never under `configs/` — this carries a test value, and §5 limit 2 of the packet puts test
    values in minted test fixtures and armed values nowhere near this file.
    """
    out = tmp_path / "witness.yaml"
    argv = [sys.executable, str(MINT), "--template", "dev", "--out", str(out)]
    for key, value in deltas.items():
        argv += ["--set", f"{key.replace('__', '.')}={value}"]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"mint failed: {proc.stderr}"
    return out


def _child_script(tmp_path: Path, log: Path) -> Path:
    """A child that announces itself and exits ON SIGTERM — it does not swallow it.

    Deliberately cooperative: the ladder emits `supervisor_forwarding_stop` BEFORE its bounded
    wait, so a prompt child keeps this test off the clock entirely.
    """
    script = tmp_path / "child.py"
    script.write_text(
        "import signal, sys, time, os\n"
        f"LOG = {str(log)!r}\n"
        "def _log(text):\n"
        "    with open(LOG, 'a', encoding='utf-8') as fh:\n"
        "        fh.write(text + '\\n')\n"
        "        fh.flush()\n"
        "state = {'stop': False}\n"
        "def _on(signum, frame):\n"
        "    state['stop'] = True\n"
        "signal.signal(signal.SIGTERM, _on)\n"
        "_log('READY ' + str(os.getpid()))\n"
        "deadline = time.monotonic() + 120.0\n"
        "while not state['stop'] and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    return script


def _spawn_supervisor(tmp_path: Path, child: Path, err: Path, *,
                      config: Path | None, extra: list[str] | None = None):
    """The REAL CLI in its own session, so a signal reaches it and nothing else."""
    argv = [sys.executable, "-m", "mantis.monitor.supervise"]
    if config is not None:
        argv += ["--config", str(config)]
    # NO FLAG OF THE HARNESS'S OWN. An earlier draft passed `--poll-interval-sec 0.1` here, which
    # made `overrides` legitimately non-empty and broke the two assertions that check exactly what
    # is and is not reported as overridden. The harness was wrong, not the assertions: the fix is
    # to MINT the cadence (`_POLL`) so a second minted key has to reach the process for these
    # drives to work at all, which widens the witness instead of weakening it.
    argv += ["--heartbeat-file", str(tmp_path / "hb.json")]
    argv += list(extra or [])
    argv += ["--", sys.executable, str(child)]
    handle = err.open("wb")
    return subprocess.Popen(argv, cwd=os.getcwd(), stderr=handle, start_new_session=True)


def _events(err: Path) -> list[dict]:
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


def _await_child_ready(proc, log: Path, err: Path) -> None:
    """Wait for the child's READY line — but stop the instant the SUPERVISOR dies.

    A healthy drive reaches READY in well under a second, so the interesting failure is not
    slowness, it is a supervisor that refused at argparse and exited before spawning anything.
    Polling only the log turns that into a generic 60-second timeout with no diagnosis; polling
    the process too turns it into an immediate failure carrying the supervisor's own stderr,
    which is the message a reader actually needs (ORACLE review #3).
    """
    deadline = time.monotonic() + _DEADLINE_SEC
    while time.monotonic() < deadline:
        if log.exists() and "READY" in log.read_text(encoding="utf-8"):
            return
        if proc.poll() is not None:
            tail = err.read_text(encoding="utf-8", errors="replace")[-800:] if err.exists() else ""
            raise AssertionError(
                f"the supervisor exited rc={proc.returncode} before its child ever ran; "
                f"its own stream said:\n{tail}"
            )
        time.sleep(0.02)
    raise AssertionError(f"the child never announced READY within {_DEADLINE_SEC}s")


def _drive(tmp_path: Path, *, config: Path, extra: list[str] | None = None) -> list[dict]:
    """Boot the supervisor on a minted config, SIGINT it, return its own event stream."""
    log = tmp_path / "child.log"
    err = tmp_path / "supervisor.err"
    child = _child_script(tmp_path, log)
    proc = _spawn_supervisor(tmp_path, child, err, config=config, extra=extra)
    try:
        _await_child_ready(proc, log, err)
        os.kill(proc.pid, signal.SIGINT)
        proc.wait(timeout=_DEADLINE_SEC)
    finally:
        if proc.poll() is None:   # pragma: no cover — only on a hung supervisor
            proc.kill()
            proc.wait(timeout=10)
    return _events(err)


# ── T1: the real-drive witness (LAW-07 producer) ────────────────────────────────────────
@pytest.mark.integration
def test_minted_kill_grace_reaches_the_running_supervisors_stop_ladder(tmp_path):
    """THE WITNESS. A minted `supervisor_kill_grace_sec` is the bound the LADDER uses.

    Mutation that proves it can fail (LAW-07, R289(u)): restore `defaults = MonitorConfig()`
    and drop the `--config` read, and the emitted `grace_sec` is 30.0 — the dataclass literal —
    not `_GRACE`. Run and recorded in the exit report; a producer test that cannot fail is the
    phantom class.
    """
    config = _mint(tmp_path, monitor__supervisor_kill_grace_sec=_GRACE,
                   monitor__supervisor_poll_interval_sec=_POLL,
                   monitor__supervisor_stale_after_sec=_STALE,
                   monitor__supervisor_max_relaunches=_RELAUNCHES)
    events = _drive(tmp_path, config=config)

    forwarding = [e for e in events if e.get("event") == "supervisor_forwarding_stop"]
    assert forwarding, f"no supervisor_forwarding_stop in the stream: {events}"
    assert forwarding[0]["grace_sec"] == pytest.approx(_GRACE), (
        "the ladder used a grace the minted config did not supply — the config and the process "
        f"disagree, which is F-816-24 itself: {forwarding[0]}"
    )


@pytest.mark.integration
def test_boot_identity_publishes_the_config_sha_and_the_effective_bounds(tmp_path):
    """The F-B1 parent-side twin: the supervisor publishes the identity of the file IT read.

    PUBLISH, NOT COMPARE (design D3): the supervisor never inspects the child's config, because
    learning it would mean parsing the verbatim child argv, which `spawn_child`'s contract
    forbids. Two shas in two streams that a reader CAN compare is the whole remedy.
    """
    from mantis.config.loader import config_identity_sha256, load_config

    config = _mint(tmp_path, monitor__supervisor_kill_grace_sec=_GRACE,
                   monitor__supervisor_poll_interval_sec=_POLL,
                   monitor__supervisor_stale_after_sec=_STALE,
                   monitor__supervisor_max_relaunches=_RELAUNCHES)
    events = _drive(tmp_path, config=config)

    boot = [e for e in events if e.get("event") == "supervisor_boot_identity"]
    assert boot, f"the supervisor published no boot identity: {[e.get('event') for e in events]}"
    assert boot[0]["config_sha256"] == config_identity_sha256(load_config(config))
    assert boot[0]["effective"]["kill_grace_sec"] == pytest.approx(_GRACE)
    assert boot[0]["effective"] == {
        "stale_after_sec": pytest.approx(_STALE),
        "poll_interval_sec": pytest.approx(_POLL),
        "kill_grace_sec": pytest.approx(_GRACE),
        "max_relaunches": _RELAUNCHES,
    }, (
        "ALL FOUR minted keys must reach the process. Asserted as a whole-dict equality rather "
        "than key by key, so a fifth published bound or a renamed one is a failure too — a "
        "subset assertion is what let two of these four go unchecked (RED-TEAM #1). Each value "
        "is read off the LIVE Supervisor at the attribute its own consumer uses."
    )
    assert boot[0]["overrides"] == {}, "no flag was passed; nothing may be reported as overridden"
    assert events.index(boot[0]) == 0, "the identity must be the FIRST thing published"


# ── T6: the override is PUBLISHED, never silent ─────────────────────────────────────────
@pytest.mark.integration
def test_a_command_line_override_is_published_and_is_the_effective_bound(tmp_path):
    """D2's mechanism, falsified. An override wins — and SAYS SO on the record.

    Without this, D2's central claim (an override is an event in the record, not an invisible
    hand-variation) would be unfalsified by this packet's own test set, which is the
    phantom-producer class the packet exists to close.
    """
    config = _mint(tmp_path, monitor__supervisor_kill_grace_sec=_GRACE,
                   monitor__supervisor_poll_interval_sec=_POLL,
                   monitor__supervisor_stale_after_sec=_STALE,
                   monitor__supervisor_max_relaunches=_RELAUNCHES)
    events = _drive(tmp_path, config=config,
                    extra=["--kill-grace-sec", str(_GRACE_OVERRIDE)])

    boot = [e for e in events if e.get("event") == "supervisor_boot_identity"]
    assert boot, f"no boot identity was published: {[e.get('event') for e in events]}"
    assert boot[0]["overrides"] == {"kill_grace_sec": pytest.approx(_GRACE_OVERRIDE)}
    assert boot[0]["effective"]["kill_grace_sec"] == pytest.approx(_GRACE_OVERRIDE)
    forwarding = [e for e in events if e.get("event") == "supervisor_forwarding_stop"]
    assert forwarding, f"the ladder never ran: {[e.get('event') for e in events]}"
    assert forwarding[0]["grace_sec"] == pytest.approx(_GRACE_OVERRIDE), (
        "the override must reach the ladder too — a published override that the ladder ignored "
        "would be a different lie from the one this packet fixes, not an improvement"
    )


@pytest.mark.integration
def test_the_minted_staleness_bound_is_the_one_the_liveness_rule_FIRES_on(tmp_path):
    """A SECOND behavioural witness, on a second key, at its own consumption site.

    RED-TEAM broke `stale_after_sec` back to its dataclass literal and every test still passed,
    because the only key observed behaviourally was the grace. Published bounds are read off the
    live object and that is strong; but two of the four keys had no observation that survives
    someone deciding the boot event should report the args namespace instead.

    So this drive lets the rule FIRE. The child never writes a heartbeat file, the minted
    `_STALE_FAST` elapses, and the supervisor emits `child_heartbeat_file_never_written` carrying
    `stale_after_sec=self._tracker.stale_after_sec` — the tracker's own copy, at the moment the
    tracker acted on it. `max_relaunches` is minted to 0 here so the budget is spent by that first
    kill and the drive terminates on `RELAUNCH_BUDGET_EXIT_CODE` instead of looping: that exit is
    itself the fourth key being consumed, so one drive observes both remaining keys behaving.
    """
    config = _mint(tmp_path, monitor__supervisor_stale_after_sec=_STALE_FAST,
                   monitor__supervisor_poll_interval_sec=_POLL,
                   monitor__supervisor_kill_grace_sec=_GRACE,
                   monitor__supervisor_max_relaunches=0)
    log, err = tmp_path / "child.log", tmp_path / "supervisor.err"
    child = _child_script(tmp_path, log)
    proc = _spawn_supervisor(tmp_path, child, err, config=config)
    timed_out = False
    try:
        proc.wait(timeout=_STALE_DEADLINE_SEC)
    except subprocess.TimeoutExpired:
        # NAME THE LIKELY CAUSE INSTEAD OF REPORTING A CLOCK. A supervisor that outlives this
        # bound is almost always one whose staleness deadline is NOT the minted one — the
        # dataclass literal is 900.0, which is what a broken delivery would leave here, so the
        # symptom of the defect under test is "this never finished". Reported as that, not as a
        # bare TimeoutExpired from deep inside subprocess.
        timed_out = True
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    events = _events(err)
    assert not timed_out, (
        f"the supervisor outlived a minted staleness bound of {_STALE_FAST}s by "
        f"{_STALE_DEADLINE_SEC}s — the deadline it acted on is not the one the config supplied "
        f"(the dataclass literal is 900.0). Events seen: {[e.get('event') for e in events]}"
    )
    stale = [e for e in events if e.get("event") == "child_heartbeat_file_never_written"]
    assert stale, (
        "the staleness rule never fired, so the minted bound was never exercised: "
        f"{[e.get('event') for e in events]}"
    )
    assert stale[0]["stale_after_sec"] == pytest.approx(_STALE_FAST), (
        "the liveness tracker acted on a deadline the minted config did not supply: "
        f"{stale[0]}"
    )
    assert proc.returncode == RELAUNCH_BUDGET_EXIT_CODE, (
        "a minted max_relaunches of 0 must spend the budget on the first kill; the supervisor "
        f"exited {proc.returncode} instead, so the fourth key did not reach the budget check"
    )


# ── T2: the refusal ─────────────────────────────────────────────────────────────────────
def test_a_missing_config_is_a_NAMED_refusal_not_a_default(tmp_path):
    """R1/LAW-11: absent is an error, never a default — and the error NAMES the input.

    `argparse`'s own "the following arguments are required: --config" is true and useless: it
    tells an operator nothing about what a config is or where one comes from. The refusal
    pre-scans argv before the parser runs, mirroring `_split_argv` one function away.
    """
    child = _child_script(tmp_path, tmp_path / "child.log")
    proc = subprocess.run(
        [sys.executable, "-m", "mantis.monitor.supervise",
         "--heartbeat-file", str(tmp_path / "hb.json"),
         "--", sys.executable, str(child)],
        capture_output=True, text=True, check=False, timeout=_REFUSAL_DEADLINE_SEC,
    )
    assert proc.returncode != 0, "a supervisor with no config must refuse to start"
    message = proc.stderr + proc.stdout
    assert "--config" in message, f"the refusal must NAME the missing input: {message!r}"
    # DESIGN D4 requires the message to point at where a minted config COMES FROM, and the
    # accepted markers are enumerated rather than left to one phrasing — the review's objection
    # was that a single required word prescribes wording the design never ordered (#2). The
    # requirement itself is the design's; this is the set of ways to satisfy it.
    remedies = ("mint", "configs/", "tools/mint_config.py", ".yaml")
    assert any(token in message.lower() for token in remedies), (
        "the refusal must point at where a config comes from — argparse's own 'the following "
        f"arguments are required' names the flag and no remedy: {message!r}"
    )


def test_a_malformed_override_is_refused_but_a_legitimate_zero_is_not(tmp_path):
    """Well-formedness is refused; POLICY is not (D2). The two must not be confused.

    RED-TEAM drove `--kill-grace-sec=nan` end to end: `Popen.wait(timeout=nan)` never raises
    `TimeoutExpired`, so the ladder's automatic SIGTERM -> grace -> SIGKILL escalation stopped
    escalating and only the operator's second and third signal recovered the child. A LAW-16
    bounded stop became an unbounded one through an input nothing validated — while the MINTED
    key for the same bound is schema-refused (`Field(ge=0)`).

    `0` is the control, and it is the one that matters: it is falsy, legitimate, and a naive
    truthiness check would silently drop it from `overrides` and from `effective`.
    """
    config = _mint(tmp_path, monitor__supervisor_kill_grace_sec=_GRACE)
    child = _child_script(tmp_path, tmp_path / "child.log")
    for bad in ("nan", "inf", "-1"):
        proc = subprocess.run(
            [sys.executable, "-m", "mantis.monitor.supervise", "--config", str(config),
             "--kill-grace-sec", bad, "--heartbeat-file", str(tmp_path / "hb.json"),
             "--", sys.executable, str(child)],
            capture_output=True, text=True, check=False, timeout=_REFUSAL_DEADLINE_SEC,
        )
        assert proc.returncode != 0, f"--kill-grace-sec {bad} was accepted"
        assert "finite and non-negative" in proc.stderr + proc.stdout, (
            f"the refusal for {bad!r} must name the property that failed: {proc.stderr[-300:]!r}"
        )

    events = _drive(tmp_path, config=config, extra=["--max-relaunches", "0"])
    boot = [e for e in events if e.get("event") == "supervisor_boot_identity"]
    assert boot and boot[0]["overrides"] == {"max_relaunches": 0}, (
        "a legitimate zero override was dropped — `0` is falsy, and a truthiness test here would "
        f"lose it from the record while it still governed the run: {boot}"
    )
    assert boot[0]["effective"]["max_relaunches"] == 0


# ── T4: O-18, on the path that is actually RUN ──────────────────────────────────────────
def test_importing_the_supervisor_does_not_import_torch():
    """O-18, import-time half. In a SUBPROCESS: in-process is polluted by pytest's own imports."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import mantis.monitor.supervise as s; "
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", "a liveness babysitter must not drag torch in"


@pytest.mark.integration
def test_running_main_with_a_real_config_does_not_import_torch(tmp_path):
    """O-18, RUN-TIME half — the one §7's exit criterion is actually about.

    The import-time half alone is a hole: if the config imports were deferred into `main`, an
    import-only check would pass while the exercised path went untested. This calls the REAL
    `main` with a REAL minted config in a subprocess and reports `sys.modules` from INSIDE that
    process, so the observation is of the path that ran, not of a proxy for it.
    """
    config = _mint(tmp_path, monitor__supervisor_kill_grace_sec=_GRACE,
                   monitor__supervisor_poll_interval_sec=_POLL,
                   monitor__supervisor_stale_after_sec=_STALE,
                   monitor__supervisor_max_relaunches=_RELAUNCHES)
    marker = tmp_path / "the_child_ran"
    touch = tmp_path / "touch.py"
    touch.write_text(f"open({str(marker)!r}, 'w', encoding='utf-8').close()\n", encoding="utf-8")
    argv = ["--config", str(config), "--heartbeat-file", str(tmp_path / "hb.json"),
            "--poll-interval-sec", "0.05", "--", sys.executable, str(touch)]
    probe = (
        "import json, sys\n"
        "from mantis.monitor import supervise\n"
        f"argv = {argv!r}\n"
        "try:\n"
        "    supervise.main(argv)\n"
        "except BaseException:\n"
        "    pass\n"
        "sys.stdout.write(json.dumps({'torch': 'torch' in sys.modules,"
        " 'mods': len(sys.modules)}) + '\\n')\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          check=False, timeout=_DEADLINE_SEC)
    assert proc.returncode == 0, proc.stderr
    # THE PROBE MUST PROVE IT RAN BEFORE ITS torch VERDICT MEANS ANYTHING. Without this, a
    # `main` that refused at argparse would report torch=False and pass — which is exactly what
    # this test did on its first ORACLE-WRITE run against the un-fixed head. A witness that
    # cannot distinguish "the path is clean" from "the path never executed" is the phantom class.
    assert marker.exists(), (
        "the supervisor never spawned its child, so this process never executed the config path "
        f"whose torch-freeness is under test: {proc.stdout!r} {proc.stderr[-400:]!r}"
    )
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdict["torch"] is False, (
        "reaching mantis.config from the supervisor dragged torch in — O-18 is the objection "
        "this design measured away, and a transitive edge would restore it"
    )


# ── T5: one construction authority in this module ───────────────────────────────────────
def test_the_supervisor_module_constructs_no_MonitorConfig_by_any_shape():
    """R1/R79: the resolver is the ONE construction authority reachable from this module.

    Broader than the packet's literal wording on purpose: `MonitorConfig()` is the shape that
    shipped, but `MonitorConfig(supervisor_kill_grace_sec=30.0)` would reintroduce the identical
    duplicate-authority defect while passing a zero-arg check. Derived by parsing the module, so
    a reformat cannot defeat it.
    """
    tree = ast.parse(SUPERVISE_SRC.read_text(encoding="utf-8"))

    # EVERY LOCAL NAME BOUND TO THE CLASS, derived from the module's own imports rather than
    # assumed to be the class's own spelling. `from … import MonitorConfig as MC` binds "MC";
    # `import mantis.monitor.config as _mc` reaches it as an attribute. An earlier draft matched
    # only a bare `ast.Name` called "MonitorConfig", which both of those escape — in the one
    # test written to stop this packet's defect from coming back, and right before a queued
    # packet generalises this mechanism to other sites, which is exactly when an import-shape
    # drift lands unnoticed (ORACLE review MUST-FIX #1).
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "MonitorConfig":
                    bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("monitor.config"):
                    bound.add(alias.asname or alias.name.split(".")[0])

    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and (func.id in bound or func.id == "MonitorConfig"):
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == "MonitorConfig":
            calls.append(node)

    assert not calls, (
        "supervise.py constructs a MonitorConfig directly at line(s) "
        f"{sorted(c.lineno for c in calls)}; construction goes through resolve_monitor_config, "
        "off a validated schema, or the minted keys reach no process again"
    )
