# >300 justify (R8), stated at this file's MEASURED size of 333 lines (`wc -l`, never
# transcribed — SF-7). The file was 248 lines at ORACLE-WRITE and crossed the cap on the R129
# re-point, which replaced O-B1's one-line checkpoint assertion with the measured truth plus
# the reasoning a later reader needs to know WHY `checkpoints/` is empty and what would make
# that expire (CARD-CLEANSTOP-SAVE). A split is argued AGAINST: O-B1, O-B2 and O-B3 are three
# instruments on ONE surface — `mantis.run`'s launcher — and O-B2's AST census parses the very
# `run.py` the other two drive, so splitting would give each half its own answer to "what is
# the launcher's flag surface". Executable content stays a small minority; the rest is the
# per-oracle "what defect is this the only witness to" rationale LAW-07 requires.
"""⊕ WPMAIN ORACLE — `python -m mantis.run` is a real launcher (DESIGN §1.3/§9, O-B1..O-B3).

RED-at-import until IMPL lands `mantis.run.launch_run` + `UnregisteredAbortExitError`.

The audit headline this file retires: *`main()` validates a config and exits* — literally,
`print(f"config OK: …"); return 0` at `run.py:341`. Nothing in the repo executes
`mantis.run.main()` today (measured, DESIGN §0), so the entry point named by CLAUDE.md's own
`python -m mantis.*` law has never had a behavioural producer at all.

The defect each oracle is the only witness to:

- O-B1 — the launcher that composes but never RUNS. Success criterion 1, as amended by
  R121(c): boots through the one composer into the live run loop, bounded, clean stop.
  Real subsystems (R64) — real `init_trainer` -> `build_net`, real `WorkerPool`, real
  `build_run_safety`, real coordinator config. INTEGRATION tier: it is a real CPU burst,
  the same class as `tests/train/test_launch_path_smoke.py`. **Re-pointed by R129**: its
  "final checkpoint on disk" clause was written against a premise DESIGN §9 got wrong and
  IMPL measured false — a clean bounded stop saves nothing. See the row's own docstring.
- O-B2 — a code-side default sneaking onto a RUN INPUT at the CLI boundary. `--out-dir` with
  a default is a run-input default (R1), and it is the O-1 doctrine
  (`test_preflight_mint.py:866-882`) applied to the launcher, which never had it.
- O-B3 — the aborted run that exits 0. `_abort_rc`'s own docstring records this as OWED:
  "when a production launcher lands it must read this same resolver"
  (`preflight_mint.py:929-947`). A launcher that returns 0 after a hard abort fired reports
  a collapsed run as a clean one — and the supervisor above it relaunches into the wall.

Fakes: O-B1 fakes nothing about the RUN — every collaborator is real and the boot is real.
Its one substitution is a post-hoc rc read: `launch_run` is monkeypatched to return the
handles the real boot ALREADY produced, so `main`'s rc policy is driven on a real terminal
state instead of on a second 30 s boot (disclosed at the row). O-B3 substitutes
`mantis.run.launch_run` with a stub returning a
rigged `RunHandles`, disclosed: the SUBJECT is the rc policy, and the run is its harness —
the same seam the frozen `_abort_rc` trio (`test_preflight_mint_process.py:265-285`) reads
by calling the mapping directly.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

# RED-at-import anchor: neither name exists yet.
from mantis.run import RunHandles, UnregisteredAbortExitError, launch_run  # noqa: F401

import mantis.run as mantis_run
from mantis.config.armed_aborts import exit_code_for_abort
from mantis.train.lifecycle.signals import ShutdownState

_REPO = Path(__file__).resolve().parents[1]
_RUN_PY = _REPO / "src" / "mantis" / "run.py"
_CONFIGS = _REPO / "configs"

#: The launcher's whole flag surface (DESIGN §1.3, as amended by R126: NO `--device`).
_LAUNCHER_OPTIONS = {"--config", "--out-dir"}

#: The armed-smoke config is the one minted config with a burst-scale posture on CPU
#: (R103's grant); 16 is its minimum legal burst plus headroom, the same number
#: `tests/tools/test_preflight_armed_smoke.py` drives.
_SMOKE_CONFIG = "smoke_preflight_armed.yaml"
_BURST_STEPS = 16


def _add_argument_calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"]


def _handles(rule: str | None) -> RunHandles:
    """A `RunHandles` whose only live field is the one the rc policy reads."""
    return RunHandles(coordinator=None, run_safety=None, eval_pipeline=None,
                      shutdown=ShutdownState(running=False, abort_rule=rule))


# ══ O-B2 — the CLI contract ═══════════════════════════════════════════════════════════
def test_the_launcher_declares_exactly_config_and_out_dir_with_no_defaults() -> None:
    """O-B2, AST half — the O-1 twin census on the launcher's own parser.

    MUTATION THAT REDS IT: add `default="runs"` to `--out-dir` (the tempting convenience),
    or add a third flag. A defaulted out-dir is a run input the code decides — R1's exact
    subject — and every run that forgets the flag then writes into one shared directory,
    which is how two runs' checkpoints end up in one lineage.

    The option SET is pinned, not merely a floor: `>= N` censuses go on passing while the
    surface silently changes underneath them (measured on the tool's own `len(calls) >= 6`,
    DESIGN ADDENDUM C.1.3)."""
    tree = ast.parse(_RUN_PY.read_text(encoding="utf-8"))
    calls = _add_argument_calls(tree)
    declared = {arg.value for call in calls for arg in call.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    assert declared == _LAUNCHER_OPTIONS, (
        f"the launcher declares exactly {sorted(_LAUNCHER_OPTIONS)} (DESIGN §1.3); got "
        f"{sorted(declared)}"
    )
    for call in calls:
        names = [arg.value for arg in call.args if isinstance(arg, ast.Constant)]
        assert not [kw for kw in call.keywords if kw.arg == "default"], (
            f"add_argument{names} passes default= — R1: no code-side defaults for a run "
            "input; a default lives only in a schema field"
        )
        required = [kw for kw in call.keywords if kw.arg == "required"]
        assert required and isinstance(required[0].value, ast.Constant) \
            and required[0].value.value is True, (
            f"add_argument{names} must be required=True: both launcher inputs are REQUIRED "
            "and neither has a stand-in"
        )


@pytest.mark.parametrize("argv", [[], ["--config", "configs/run5.yaml"], ["--out-dir", "/tmp/x"]])
def test_omitting_a_launcher_input_is_a_usage_error_not_a_default(argv, capsys) -> None:
    """O-B2, behavioural half. Each omission is rc 2 (argparse's own), never a boot.

    MUTATION THAT REDS IT: make either flag optional. The AST half above cannot see a flag
    that is declared required and then re-read with a fallback; this one drives the refusal."""
    with pytest.raises(SystemExit) as exit_info:
        mantis_run.main(list(argv))
    assert exit_info.value.code == 2, (
        f"omitting an input must be a USAGE error (rc 2); got {exit_info.value.code!r}"
    )
    assert "usage" in capsys.readouterr().err.lower(), "argparse's own usage line must print"


def test_the_module_entry_point_reports_a_usage_error_at_the_process_boundary() -> None:
    """O-B2, process half — `python -m mantis.run` with no arguments exits 2.

    MUTATION THAT REDS IT: keep the retired positional surface
    (`python -m mantis.run <config.yaml>`), which prints its own usage line and returns 2
    from a hand-rolled `len(argv) != 1` check (`run.py:330-332`). The in-process assertions
    above pass on that shape too — argparse is not the only thing that can return 2 — so the
    surface itself is pinned in the AST half and the PROCESS rc is pinned here. This is the
    CLAUDE.md `python -m mantis.*` law's only executable producer."""
    result = subprocess.run([sys.executable, "-m", "mantis.run"], cwd=str(_REPO),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 2, (
        f"`python -m mantis.run` with no arguments is rc 2; got {result.returncode}\n"
        f"{(result.stdout + result.stderr)[-2000:]}"
    )
    assert "--config" in (result.stdout + result.stderr), (
        "the usage line must name the flag surface the operator has to supply"
    )


def test_the_launcher_prints_no_config_ok_readiness_line() -> None:
    """O-B2, the deletion arm — RED FOR A REASON OTHER THAN "not built yet": it asserts that
    a line which EXISTS at HEAD is gone.

    `run.py:341` prints `config OK: run_id=… encoding=…` and returns 0. That string is the
    whole of what `python -m mantis.run` does today, and it is the artefact a later reader
    cites as "the launcher works". Measured: the string appears exactly once in the repo and
    no oracle pins it (DESIGN §1.3), so deleting it costs nothing and keeping it means a
    validate-and-exit path survives beside the real boot. The run's OWN event stream
    (`run_boot_identity`, `resolved_config`) is the boot record now."""
    text = _RUN_PY.read_text(encoding="utf-8")
    assert "config OK" not in text, (
        "the validate-and-exit readiness print must be DELETED with the launcher rewrite — "
        "a boot record that is a stdout line nobody parses is not a boot record"
    )


# ══ O-B3 — the abort exit code, through THE resolver ══════════════════════════════════
def test_a_clean_run_exits_zero(monkeypatch, tmp_path) -> None:
    """O-B3, arm 1. `abort_rule is None` is the ONLY thing that means a clean run (R84).

    MUTATION THAT REDS IT: return a nonzero rc unconditionally, or read a different field."""
    monkeypatch.setattr(mantis_run, "launch_run", lambda **_kw: _handles(None))
    rc = mantis_run.main(["--config", str(_CONFIGS / "run5.yaml"), "--out-dir", str(tmp_path)])
    assert rc == 0, f"a run with no fired abort exits 0; got {rc}"


def test_a_fired_abort_exits_with_the_code_the_manifest_authors(monkeypatch, tmp_path) -> None:
    """O-B3, arm 2 — and the discharge of `_abort_rc`'s OWED paragraph
    (`preflight_mint.py:944-947`): the launcher reads the SAME resolver the child does.

    MUTATION THAT REDS IT: return 0 (or 1) after a hard abort fired — an aborted run
    reported as a clean one, which is the failure mode `ShutdownState.abort_rule` was added
    to make distinguishable at all. Writing the literal 46 in `run.py` instead of resolving
    it is caught too: the assertion is against `exit_code_for_abort`'s answer, so a second
    literal cannot agree with the manifest by accident — but 46 is ALSO stated here, so a
    silent manifest drift is loud rather than self-consistent (R92's pre-registered code)."""
    monkeypatch.setattr(mantis_run, "launch_run",
                        lambda **_kw: _handles("draw_rate_collapse"))
    rc = mantis_run.main(["--config", str(_CONFIGS / "run5.yaml"), "--out-dir", str(tmp_path)])
    assert rc == exit_code_for_abort("draw_rate_collapse") == 46, (
        f"a fired draw_rate_collapse exits 46, resolved from the manifest row; got {rc}"
    )


def test_a_fired_abort_with_no_authored_code_is_a_named_failure_never_an_invented_number(
    monkeypatch, tmp_path
) -> None:
    """O-B3, arm 3 — the middle outcome that must not be quietly rounded off.

    `grad_norm_hard_abort` and `sealbot_wr_abort` share `_fire_hard_abort` and neither is
    pre-registered with a code; R84 refused to invent one. MUTATION THAT REDS IT: map the
    unresolvable rule to 0 (an aborted run reads clean), or to a made-up number (a second
    exit-code authority beside the manifest). The error must NAME the rule, or the operator
    reading a supervisor log cannot tell which abort ended the run."""
    monkeypatch.setattr(mantis_run, "launch_run",
                        lambda **_kw: _handles("grad_norm_hard_abort"))
    assert exit_code_for_abort("grad_norm_hard_abort") is None, (
        "premise check: the rule genuinely has no authored code (if this flips, the arm "
        "above is testing nothing)"
    )
    with pytest.raises(UnregisteredAbortExitError) as exc_info:
        mantis_run.main(["--config", str(_CONFIGS / "run5.yaml"), "--out-dir", str(tmp_path)])
    assert "grad_norm_hard_abort" in str(exc_info.value), (
        f"the refusal must name the rule that fired; got {str(exc_info.value)!r}"
    )


# ══ O-B1 — success criterion 1 ════════════════════════════════════════════════════════
@pytest.mark.integration
def test_launch_run_boots_a_minted_config_into_the_live_loop_and_stops_clean(
    monkeypatch, tmp_path, smoke_run_config
) -> None:
    """O-B1 — success criterion 1 as amended by R121(c): "boots through the one composer
    into the live run loop, bounded, clean stop".

    R64 posture, nothing about the RUN routed around: real `init_trainer` -> `build_net`,
    real `WorkerPool` self-play on CPU, real graph replay buffer, real `build_run_safety`,
    real coordinator config, `eval_enabled` from the config's own value. The config is the
    minted armed smoke — the one config whose values make a burst-scale CPU boot legal
    (R103) — bounded to a 16-step burst, exactly as `test_preflight_armed_smoke.py` drives
    it.

    RE-POINTED BY R129 (an R43 grant against a byte-frozen ORACLE-WRITE artefact), because
    the version written at ORACLE-WRITE asserted a premise that is MEASURED FALSE. It read
    "the run must leave a stamped checkpoint", on DESIGN §9's reasoning that LAW-16's
    save-then-exit and LAW-12's stamp land on the same artefact. They do — on the SIGNAL
    path. A CLEAN bounded stop never takes it: `coordinator/step.py:238-241`'s O2
    iteration-limit arm sets `shutdown.running = False` and returns WITHOUT saving;
    `loop.py`'s `_final_save()` fires only on `shutdown_save`; and the trainer's periodic
    save is guarded by `if interval > 0` (`trainer/core.py:487-489`) against a
    `checkpoint_interval` this config — and every other minted config — mints at **0**. So
    `checkpoints/` is legitimately EMPTY here, and the oracle now asserts that, WITH its
    cause, rather than asserting a checkpoint nothing writes.

    Whether a clean completion SHOULD save is a deliberate lifecycle decision and is
    explicitly NOT this WP's work: R129 opened CARD-CLEANSTOP-SAVE for it (escalated to
    mint-blocking, because run5 mints `checkpoint_interval: 0` too). When that card lands,
    THIS row is what tells the next reader the emptiness expired.

    THE ORACLE MUST NOT GO VACUOUS, and does not: "no checkpoint" is a state a run that
    never booted would also produce, so the emptiness is asserted only ALONGSIDE the
    positive truth, and the positive truth is what carries the row —

    * `handles.shutdown.running is False`: `ShutdownState()` is born `running=True`, so this
      is the O2 arm having FIRED, i.e. the loop was entered and terminated itself;
    * `trainer.step == _BURST_STEPS`: the learner reached the bound EXACTLY — the loop did
      not merely start, it ran to its ceiling (measured: 16);
    * the run's own JSONL segment carries the boot + armed-watchdog witnesses (LAW-18);
    * `abort_rule is None` and the launcher's own rc policy answers **0** for this state.

    MUTATION THAT REDS IT: a launcher that composes and returns without driving the loop
    (`stop_step=0`, or a `compose_run` whose loop call is removed) — `running` never flips,
    `trainer.step` stays 0 and the segment carries no armed-watchdog rows. That is precisely
    the failure this WP exists to end, and no import-level or AST census can see it.

    DISCLOSED FAKE, and it is the ONLY one: the rc assertion re-enters `main` with
    `launch_run` monkeypatched to hand back **the handles this real boot just produced** —
    so nothing about the run is faked, and no second 30 s boot is spent to read one integer.
    Its subject is the composition "a real clean bounded run PRODUCES the state that maps to
    rc 0"; O-B3 arm 1 pins the mapping itself on a synthetic state, and the two together are
    what make "rc 0 means the run finished" a claim with a producer at both ends."""
    config = smoke_run_config(_SMOKE_CONFIG, train={"max_train_steps": _BURST_STEPS})
    assert int(config.train.checkpoint_interval) == 0, (
        "PREMISE CHECK for the emptiness arm below (R129's own instruction, applied to this "
        "config rather than to run5): the periodic save is guarded by `interval > 0`. If "
        f"this config ever mints a nonzero interval — got {config.train.checkpoint_interval!r} "
        "— the checkpoints-empty assertion stops being the truth and this row must be "
        "RE-POINTED again, never silenced"
    )
    handles = launch_run(config=config, out_dir=tmp_path)

    assert isinstance(handles, RunHandles), "the launcher returns the composed handles"
    assert handles.shutdown.running is False, (
        "a bounded run reaches its ceiling and stops — a still-running state means the loop "
        "was never entered or never terminated (`ShutdownState()` is born running=True, so "
        "this flip is the O2 arm having fired)"
    )
    assert handles.coordinator is not None, "a composed run hands back its coordinator"
    assert int(handles.coordinator.trainer.step) == _BURST_STEPS, (
        "the LOOP RAN TO ITS BOUND — the learner took exactly the burst's steps. This is the "
        "positive truth the emptiness assertion below must never stand alone against; got "
        f"step {handles.coordinator.trainer.step!r} against a {_BURST_STEPS}-step ceiling"
    )
    assert handles.shutdown.abort_rule is None, (
        f"the armed smoke completes its burst without firing an abort; got "
        f"{handles.shutdown.abort_rule!r}"
    )

    # The clean-vs-aborted distinction, asserted intact (R129) on the REAL handles.
    monkeypatch.setattr(mantis_run, "launch_run", lambda **_kw: handles)
    rc = mantis_run.main(["--config", str(_CONFIGS / _SMOKE_CONFIG), "--out-dir", str(tmp_path)])
    assert rc == 0, (
        "a run that completed its burst with no abort fired is a CLEAN run and exits 0; an "
        f"aborted one exits the manifest's code (O-B3 arm 2, 46). got {rc}"
    )

    residents = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert residents == [], (
        "a CLEAN bounded stop writes NOTHING under the derived checkpoint_dir: the O2 arm "
        "(`step.py:238-241`) returns without saving, `_final_save()` needs `shutdown_save`, "
        f"and `checkpoint_interval: 0` disables the periodic save. Found {residents}. If a "
        "checkpoint appears here, CARD-CLEANSTOP-SAVE has landed (or a save path grew a "
        "second authority) and this row is the notice — re-point it, do not delete it"
    )
    segments = sorted((tmp_path / "logs").glob("events_*.jsonl"))
    assert segments, (
        "…while the run's own JSONL event segment IS written under the ONE derived log_dir; "
        f"found {sorted(p.name for p in (tmp_path / 'logs').iterdir())}"
    )
    events = {json.loads(line)["event"]
              for segment in segments
              for line in segment.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert {"run_segment_started", "run_boot_identity", "resolved_config",
            "heartbeat_watchdog_armed", "selfplay_stall_watchdog_armed"} <= events, (
        "the boot must reach an ARMED training loop and publish its own identity, not merely "
        f"construct objects (LAW-18); saw {sorted(events)}"
    )
