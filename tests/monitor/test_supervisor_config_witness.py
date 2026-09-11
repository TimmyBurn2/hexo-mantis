"""The supervisor reads the MINTED config, and the witness proves it in a live process.

ORACLE-FIRST: every test was written before the fix, against a head where `main` built a bare
`MonitorConfig()` and offered no `--config`, so the four minted `monitor.supervisor_*` keys
reached no process.

The witness observes `supervisor_forwarding_stop.grace_sec` and NOT the config object: the defect
is that the config and the process DISAGREED, so reading the value back out of the object just
built proves nothing. That event carries `supervisor._kill_grace` from the STOP SITE, in a real
`-m` process on the real SIGINT path, and is emitted before the ladder waits, so nothing races.

>300 justify (R8): ONE claim — the minted config reaches the running supervisor — whose tests
share one harness (mint → real `-m` drive → SIGINT → read the supervisor's own stream). The
harness is the expensive, defect-prone part, and a split would copy it and let the copies drift.
The distinctive values are the point: every committed config mints 30.0, which is also the
dataclass literal, so a witness using 30.0 would pass whether or not the fix works.
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
import yaml

from mantis.monitor.supervise import RELAUNCH_BUDGET_EXIT_CODE

REPO_ROOT = Path(__file__).resolve().parents[2]
MINT = REPO_ROOT / "tools" / "mint_config.py"
SUPERVISE_SRC = REPO_ROOT / "src" / "mantis" / "monitor" / "supervise.py"

#: Minted into the test config. Not 30.0 (every committed config, both templates and the dataclass
#: literal), not 900.0, and not a schema default — there is none for this key.
_GRACE = 7.125
#: Supplied on the command line by the override test, and distinct from `_GRACE` so the two
#: cannot be confused for one another in an assertion.
_GRACE_OVERRIDE = 4.25
#: Also minted, and also distinctive: the drives need a sub-second cadence, and taking it from
#: the config rather than a flag means a SECOND minted key must reach the process.
_POLL = 0.075
#: ALL FOUR subject keys are minted distinctively, and all four are asserted: a red-team break of
#: `stale_after_sec` and `max_relaunches` back to their dataclass literals left the whole suite
#: green, because nothing here looked at them. Deliberately LARGE so the staleness rule cannot fire
#: during a drive whose subject is the stop ladder — the child never writes a heartbeat file, so a
#: small value would kill and relaunch it mid-test.
_STALE = 611.25
_RELAUNCHES = 3
#: Small, distinctive, and used ONLY by the staleness witness below, which needs the rule to fire.
_STALE_FAST = 0.625
#: The staleness drive is bounded by the MINTED deadline plus slack, so a supervisor waiting the
#: dataclass 900.0 is reported as a wrong bound within seconds rather than as a `TimeoutExpired`.
_STALE_DEADLINE_SEC = 20.0

_DEADLINE_SEC = 60.0
#: A refusal is a decision taken before anything is spawned, so it is fast or it is not a refusal.
#: Kept well under `_DEADLINE_SEC`, so a supervisor that supervises instead of refusing is reported
#: as the wrong BEHAVIOUR rather than as a slow test.
_REFUSAL_DEADLINE_SEC = 20.0


def _in_template(template: dict, dotted: str) -> bool:
    node: object = template
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _mint(tmp_path: Path, **deltas: object) -> Path:
    """A MINTED config (R1: minted via the tool, never hand-varied), written to `tmp_path`."""
    out = tmp_path / "witness.yaml"
    argv = [sys.executable, str(MINT), "--template", "dev", "--out", str(out)]
    template = yaml.safe_load((REPO_ROOT / "tools" / "config_templates" / "dev.yaml")
                              .read_text(encoding="utf-8"))
    for key, value in deltas.items():
        dotted = key.replace("__", ".")
        # A key the template omits (a schema default since CONFIG-1) is minted as a ROW.
        flag = "--set" if _in_template(template, dotted) else "--mint-row"
        argv += [flag, f"{dotted}={value}"]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"mint failed: {proc.stderr}"
    return out


def _child_script(tmp_path: Path, log: Path) -> Path:
    """A child that announces itself and exits ON SIGTERM — it does not swallow it."""
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
    # NO FLAG OF THE HARNESS'S OWN: passing one here makes `overrides` legitimately non-empty and
    # breaks the assertions that check exactly what is and is not reported as overridden. The
    # cadence is MINTED instead, so a second minted key must reach the process.
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
    """Wait for the child's READY line — but stop the instant the SUPERVISOR dies."""
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


@pytest.mark.integration
def test_minted_kill_grace_reaches_the_running_supervisors_stop_ladder(tmp_path):
    """THE WITNESS."""
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
    """The F-B1 parent-side twin: the supervisor publishes the identity of the file IT read."""
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


@pytest.mark.integration
def test_a_command_line_override_is_published_and_is_the_effective_bound(tmp_path):
    """D2's mechanism, falsified."""
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
    """A SECOND behavioural witness, on a second key, at its own consumption site."""
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
        # NAME THE LIKELY CAUSE INSTEAD OF REPORTING A CLOCK: a supervisor that outlives this bound
        # is almost always one whose staleness deadline is the dataclass 900.0 rather than the
        # minted one, so the symptom of the defect under test is "this never finished".
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


def test_a_missing_config_is_a_NAMED_refusal_not_a_default(tmp_path):
    """R1/LAW-11: absent is an error, never a default — and the error NAMES the input."""
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
    # The message must point at where a minted config COMES FROM, and the accepted markers are
    # enumerated rather than left to one phrasing, since a single required word would prescribe
    # wording the design never ordered.
    remedies = ("mint", "configs/", "tools/mint_config.py", ".yaml")
    assert any(token in message.lower() for token in remedies), (
        "the refusal must point at where a config comes from — argparse's own 'the following "
        f"arguments are required' names the flag and no remedy: {message!r}"
    )


def test_a_malformed_override_is_refused_but_a_legitimate_zero_is_not(tmp_path):
    """Well-formedness is refused; POLICY is not (D2)."""
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
    """O-18, RUN-TIME half — the one §7's exit criterion is actually about."""
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
    # THE PROBE MUST PROVE IT RAN BEFORE ITS torch VERDICT MEANS ANYTHING: a `main` that refused at
    # argparse would report torch=False and pass, which is what this test did on its first run
    # against the un-fixed head.
    assert marker.exists(), (
        "the supervisor never spawned its child, so this process never executed the config path "
        f"whose torch-freeness is under test: {proc.stdout!r} {proc.stderr[-400:]!r}"
    )
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdict["torch"] is False, (
        "reaching mantis.config from the supervisor dragged torch in — O-18 is the objection "
        "this design measured away, and a transitive edge would restore it"
    )


def test_the_supervisor_module_constructs_no_MonitorConfig_by_any_shape():
    """R1/R79: the resolver is the ONE construction authority reachable from this module."""
    tree = ast.parse(SUPERVISE_SRC.read_text(encoding="utf-8"))

    # EVERY LOCAL NAME BOUND TO THE CLASS, derived from the module's own imports rather than from
    # the class's own spelling: `from … import MonitorConfig as MC` binds "MC" and `import
    # mantis.monitor.config as _mc` reaches it as an attribute, both of which a bare `ast.Name`
    # match escapes.
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
