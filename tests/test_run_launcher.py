# >300 justify (R8): a split is argued AGAINST — the three oracles are three instruments on ONE
# surface, `mantis.run`'s launcher, and the AST census parses the very `run.py` the other two
# drive, so splitting would give each half its own answer to "what is the launcher's flag surface".
"""`python -m mantis.run` is a real launcher, not a config validator.

The audit headline this file retires: `main()` validated a config and exited, and nothing in the
repo executed `mantis.run.main()` at all, so the entry point named by CLAUDE.md's own
`python -m mantis.*` law had never had a behavioural producer.

The defect each oracle is the only witness to: the launcher that composes but never RUNS; a
code-side default sneaking onto a RUN INPUT at the CLI boundary (`--out-dir` with a default is a
run-input default, R1); and the aborted run that exits 0, which reports a collapsed run as a clean
one so the supervisor above it relaunches into the wall.

The boot row fakes nothing about the RUN; its one substitution is a post-hoc rc read. The rc rows
substitute `launch_run` with a stub returning rigged `RunHandles`, because their SUBJECT is the rc
policy and the run is only its harness.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Neither name exists before the launcher lands.
from mantis.run import RunHandles, UnregisteredAbortExitError, launch_run  # noqa: F401

import mantis.run as mantis_run
from mantis.config.armed_aborts import exit_code_for_abort
from mantis.config.preflight_stamp import (
    PreflightStampMissingError,
    PreflightStampTreeMismatchError,
)
from mantis.train.lifecycle.signals import ShutdownState

_REPO = Path(__file__).resolve().parents[1]
_RUN_PY = _REPO / "src" / "mantis" / "run.py"
_CONFIGS = _REPO / "configs"

#: The launcher's whole flag surface. The two REQUIRED inputs; neither may carry a `default=`.
_LAUNCHER_REQUIRED_OPTIONS = {"--config", "--out-dir"}
#: The one OPTIONAL flag, enumerated by name so a SECOND optional flag still reds the census.
_LAUNCHER_OPTIONAL_OPTIONS = {"--resume-from"}
_LAUNCHER_OPTIONS = _LAUNCHER_REQUIRED_OPTIONS | _LAUNCHER_OPTIONAL_OPTIONS

#: The armed-smoke config is the one minted config with a burst-scale posture on CPU; 16 is its
#: minimum legal burst plus headroom, the same number `test_preflight_armed_smoke.py` drives.
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


def test_the_launcher_declares_exactly_config_and_out_dir_with_no_defaults() -> None:
    """The launcher's parser declares exactly the flag set, with no defaults on the required pair.

    A defaulted out-dir is a run input the code decides — R1's exact subject — and every run that
    forgets the flag then writes into one shared directory, which is how two runs' checkpoints end
    up in one lineage. `--resume-from` is the ONE optional flag: a resume target is a property of
    THIS invocation, not of the run's identity, so a schema key would make two runs differ by an
    identity key describing neither. The option SET is pinned, not merely a floor.
    """
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
        flag = next((n for n in names if n in _LAUNCHER_OPTIONS), None)
        if flag in _LAUNCHER_OPTIONAL_OPTIONS:
            # Its `default=None` must be exactly `None`, which selects no action. R1 bans a
            # default that picks a VALUE on the operator's behalf; a non-None default here would
            # be exactly that, so it still reds.
            default = [kw for kw in call.keywords if kw.arg == "default"]
            assert default and isinstance(default[0].value, ast.Constant) \
                and default[0].value.value is None, (
                f"add_argument{names} is the optional resume flag: its default must be "
                "exactly None (no action), never a stand-in path"
            )
            assert not [kw for kw in call.keywords if kw.arg == "required"], (
                f"add_argument{names} is optional; it must not declare required="
            )
            continue
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


@pytest.mark.parametrize("argv", [[], ["--config", "configs/run6.yaml"], ["--out-dir", "/tmp/x"]])
def test_omitting_a_launcher_input_is_a_usage_error_not_a_default(argv, capsys) -> None:
    """Each omission is rc 2 (argparse's own), never a boot. The AST census cannot see a flag
    declared required and then re-read with a fallback; this drives the refusal."""
    with pytest.raises(SystemExit) as exit_info:
        mantis_run.main(list(argv))
    assert exit_info.value.code == 2, (
        f"omitting an input must be a USAGE error (rc 2); got {exit_info.value.code!r}"
    )
    assert "usage" in capsys.readouterr().err.lower(), "argparse's own usage line must print"


def test_the_module_entry_point_reports_a_usage_error_at_the_process_boundary() -> None:
    """`python -m mantis.run` with no arguments exits 2 at the PROCESS boundary.

    The retired positional surface printed its own usage line and returned 2 from a hand-rolled
    length check, so the in-process assertions pass on that shape too — argparse is not the only
    thing that can return 2. This is the `python -m mantis.*` law's only executable producer.
    """
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
    """The validate-and-exit readiness print is DELETED — red for a reason other than "not built
    yet", since it asserts a line that EXISTS at HEAD is gone.

    That string was the whole of what `python -m mantis.run` did, and it is the artefact a later
    reader cites as "the launcher works". The run's OWN event stream is the boot record now.
    """
    text = _RUN_PY.read_text(encoding="utf-8")
    assert "config OK" not in text, (
        "the validate-and-exit readiness print must be DELETED with the launcher rewrite — "
        "a boot record that is a stdout line nobody parses is not a boot record"
    )


def test_a_clean_run_exits_zero(monkeypatch, tmp_path, preflight_stamped) -> None:
    """A clean run exits 0. `abort_rule is None` is the ONLY thing that means a clean run."""
    preflight_stamped(_CONFIGS / "run6.yaml")
    monkeypatch.setattr(mantis_run, "launch_run", lambda **_kw: _handles(None))
    rc = mantis_run.main(["--config", str(_CONFIGS / "run6.yaml"), "--out-dir", str(tmp_path)])
    assert rc == 0, f"a run with no fired abort exits 0; got {rc}"


def test_main_refuses_to_launch_a_config_with_no_preflight_stamp(
    monkeypatch, tmp_path, preflight_stamped,
) -> None:
    """R348(c): no stamp for this config on this tree means `main` refuses BEFORE `launch_run`."""
    launched: list[dict] = []
    monkeypatch.setattr(mantis_run, "launch_run",
                        lambda **kw: (launched.append(kw), _handles(None))[1])
    with pytest.raises(PreflightStampMissingError) as exc_info:
        mantis_run.main(["--config", str(_CONFIGS / "run6.yaml"), "--out-dir", str(tmp_path)])
    assert launched == [], "the launcher ran a run that was never preflighted"
    assert "run6" in str(exc_info.value) or "no preflight stamp" in str(exc_info.value)
    # and the SAME call launches once the config is stamped on this tree
    preflight_stamped(_CONFIGS / "run6.yaml")
    assert mantis_run.main(
        ["--config", str(_CONFIGS / "run6.yaml"), "--out-dir", str(tmp_path)]) == 0
    assert len(launched) == 1


def test_a_stamp_from_another_tree_does_not_launch(monkeypatch, tmp_path, preflight_stamped) -> None:
    """The stamp binds the config to the tree that preflighted it; a HEAD that moved refuses."""
    import json

    path = preflight_stamped(_CONFIGS / "run6.yaml")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["tree_sha"] = "0" * 40
    path.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(mantis_run, "launch_run", lambda **_kw: _handles(None))
    with pytest.raises(PreflightStampTreeMismatchError):
        mantis_run.main(["--config", str(_CONFIGS / "run6.yaml"), "--out-dir", str(tmp_path)])


def test_a_fired_abort_exits_with_the_code_the_manifest_authors(
    monkeypatch, tmp_path, preflight_stamped,
) -> None:
    """A fired abort exits with the manifest's code through the resolver the preflight child reads;
    46 is ALSO stated here so a literal in `run.py` and a silent manifest drift are both loud."""
    preflight_stamped(_CONFIGS / "run6.yaml")
    monkeypatch.setattr(mantis_run, "launch_run",
                        lambda **_kw: _handles("draw_rate_collapse"))
    rc = mantis_run.main(["--config", str(_CONFIGS / "run6.yaml"), "--out-dir", str(tmp_path)])
    assert rc == exit_code_for_abort("draw_rate_collapse") == 46, (
        f"a fired draw_rate_collapse exits 46, resolved from the manifest row; got {rc}"
    )


def test_a_fired_abort_with_no_authored_code_is_a_named_failure_never_an_invented_number(
    monkeypatch, tmp_path, preflight_stamped,
) -> None:
    """An abort with no authored code is a NAMED failure that names the rule — never 0, never an
    invented number that would be a second exit-code authority beside the manifest."""
    preflight_stamped(_CONFIGS / "run6.yaml")
    monkeypatch.setattr(mantis_run, "launch_run",
                        lambda **_kw: _handles("grad_norm_hard_abort"))
    assert exit_code_for_abort("grad_norm_hard_abort") is None, (
        "premise check: the rule genuinely has no authored code (if this flips, the arm "
        "above is testing nothing)"
    )
    with pytest.raises(UnregisteredAbortExitError) as exc_info:
        mantis_run.main(["--config", str(_CONFIGS / "run6.yaml"), "--out-dir", str(tmp_path)])
    assert "grad_norm_hard_abort" in str(exc_info.value), (
        f"the refusal must name the rule that fired; got {str(exc_info.value)!r}"
    )


@pytest.mark.integration
def test_launch_run_boots_a_minted_config_into_the_live_loop_and_stops_clean(
    monkeypatch, tmp_path, smoke_run_config, preflight_stamped
) -> None:
    """A minted config boots through the one composer into the live loop and stops clean.

    Everything on the RUN is real (trainer, CPU self-play pool, graph replay, run safety) on the
    armed smoke config, bounded to a 16-step burst. The checkpoint clause is EXACTLY ONE: 0 is the
    clean-completion leg absent, 2 a second write authority. The positive truths carry the row so
    an empty run dir cannot pass. The ONE fake: the rc assertion re-enters `main` with
    `launch_run` patched to hand back these handles, so no second boot reads one integer.
    """
    config = smoke_run_config(_SMOKE_CONFIG, train={"max_train_steps": _BURST_STEPS})
    assert int(config.train.checkpoint_interval) == 0, (
        "PREMISE CHECK, reason 1 of 2, for the EXACTNESS arm below (R129's own instruction, "
        "applied to this config rather than to run5): the periodic save is guarded by "
        f"`interval > 0`. If this config ever mints a nonzero interval — got "
        f"{config.train.checkpoint_interval!r} — a periodic checkpoint could join the "
        "clean-completion one, the `len(...) == 1` assertion stops being the truth, and this "
        "row must be RE-POINTED again, never silenced"
    )
    assert config.identity.representation == "graph", (
        "PREMISE CHECK, reason 2 of 2 — the ROUTE this config declares. It USED to be a "
        "second, interval-INDEPENDENT reason no periodic save could fire: on a `graph` "
        "representation the arm did not exist at all. WP12-R CARD-CS2 (R173) made both step "
        "tails call the ONE resolver `Trainer._maybe_periodic_checkpoint` "
        "(`trainer/core.py:562-600`), so the graph arm now EXISTS and evaluates. Got "
        f"{config.identity.representation!r}. This pin therefore records WHICH route the run "
        "takes, not a closed route — and reason 1 (the minted `0`) is now the SOLE reason a "
        "periodic artefact cannot join the clean-completion one asserted below"
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

    # The clean-vs-aborted distinction, on the REAL handles.
    preflight_stamped(_CONFIGS / _SMOKE_CONFIG)
    monkeypatch.setattr(mantis_run, "launch_run", lambda **_kw: handles)
    rc = mantis_run.main(["--config", str(_CONFIGS / _SMOKE_CONFIG), "--out-dir", str(tmp_path)])
    assert rc == 0, (
        "a run that completed its burst with no abort fired is a CLEAN run and exits 0; an "
        f"aborted one exits the manifest's code (O-B3 arm 2, 46). got {rc}"
    )

    residents = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())

    # `best_model.pt` — the promotion ANCHOR — lands here. It used to go to a CWD-RELATIVE
    # `checkpoints/best_model.pt` while the promotion WRITE side got the run's real path, so read
    # and write named different files. The `.ckpt` count is asserted separately below so the two
    # facts cannot mask each other.
    ckpts = [n for n in residents if n.endswith(".ckpt")]
    assert "best_model.pt" in residents, (
        "the promotion anchor is not under the run's own checkpoint_dir — it has gone back "
        f"to a CWD-relative path (item 5(a)). Found {residents}"
    )
    assert len(ckpts) == 1, (
        "a CLEAN bounded stop now writes EXACTLY ONE checkpoint under the derived "
        "checkpoint_dir — the clean-completion leg (R137/CARD-CLEANSTOP-SAVE, the O2 arm of "
        "`train/coordinator/step.py`). NOT zero: that was the pre-R137 truth this row "
        "recorded, and its own notice told the next reader to re-point rather than delete. "
        "NOT two: leg 2 is latched out by `clean_stop_saved`, and two saves at one step are "
        f"two DISTINCT files (the filename carries a content hash over a microsecond-"
        f"resolution `created_utc`, so there is no idempotence to lean on). Found {ckpts}"
    )
    # `ckpts[0]`, not `residents[0]`: `best_model.pt` sorts first and is not an envelope-v2
    # filename, so the stem parse below must read the CKPT.
    run_id, step_field, _sha8 = Path(ckpts[0]).stem.rsplit("_", 2)
    assert run_id == config.run_id and int(step_field) == _BURST_STEPS, (
        "…and the ONE artefact is stamped with THIS run's lineage at THIS run's terminus. "
        "The decomposition is production's own (`checkpoints.py`'s `_verify_provenance` "
        "reads `stem.rsplit('_', 2)`), not a parser re-derived here. A checkpoint written at "
        "some other step, or under some other run_id, would satisfy a bare count and is "
        f"exactly what a second write authority looks like; got run_id={run_id!r} "
        f"step={step_field!r} against {config.run_id!r} / {_BURST_STEPS}"
    )
    segments = sorted((tmp_path / "logs").glob("events_*.jsonl"))
    assert segments, (
        "…while the run's own JSONL event segment IS written under the ONE derived log_dir; "
        f"found {sorted(p.name for p in (tmp_path / 'logs').iterdir())}"
    )
    rows = [json.loads(line)
            for segment in segments
            for line in segment.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = {row["event"] for row in rows}
    assert {"run_segment_started", "run_boot_identity", "resolved_config",
            "heartbeat_watchdog_armed", "selfplay_stall_watchdog_armed"} <= events, (
        "the boot must reach an ARMED training loop and publish its own identity, not merely "
        f"construct objects (LAW-18); saw {sorted(events)}"
    )

    # The leg is READABLE from the ONE channel, not only from the filesystem (LAW-18).
    clean_stop = [row for row in rows if row["event"] == "clean_stop_save"]
    assert len(clean_stop) == 1, (
        "exactly ONE `clean_stop_save` — the run's own record that it wrote its FINAL "
        f"checkpoint, and the only in-stream answer to 'did the final save happen'; got "
        f"{clean_stop}"
    )
    assert clean_stop[0]["step"] == _BURST_STEPS, (
        "…at the terminus the loop actually reached, which is what makes the event about "
        f"THIS run's product rather than a number the emitter chose; got {clean_stop[0]}"
    )
    assert [row for row in rows if row["event"] == "shutdown_save"] == [], (
        "and ZERO `shutdown_save`: leg 2 means 'we were interrupted', which is FALSE of a run "
        "that finished. A stream carrying both would mean the two legs fired together, i.e. "
        f"the duplicate-final-artefact window is open; got "
        f"{[row for row in rows if row['event'] == 'shutdown_save']}"
    )


@pytest.mark.integration
def test_a_periodic_cadence_burst_streams_periodic_checkpoint_save(
    tmp_path, smoke_run_config
) -> None:
    """A composed run with a NONZERO cadence streams `periodic_checkpoint_save`.

    The event is authored at the ONE periodic seam and unit-pinned through a spy sink, but the
    production composition built the trainer with `sink=None`, so a live burn's stream carried ZERO
    checkpoint events while the .ckpt files appeared on disk (measured: grep count 0 over seg0001
    against a stamped step-25 artefact). LAW-18: the leg must log its own fires IN-RUN.

    One delta from the clean-stop row: `checkpoint_interval=5` against the 16-step burst, so the
    boundaries are 5/10/15 and the terminus is NOT one. The files-vs-events split is asserted in
    BOTH directions, so a fabricated event REDs the same run a dropped event does.
    """
    config = smoke_run_config(
        _SMOKE_CONFIG,
        train={"max_train_steps": _BURST_STEPS, "checkpoint_interval": 5},
    )
    handles = launch_run(config=config, out_dir=tmp_path)

    assert handles.shutdown.running is False and handles.shutdown.abort_rule is None, (
        "the cadence drive must still be a clean bounded run; got "
        f"running={handles.shutdown.running!r} abort_rule={handles.shutdown.abort_rule!r}"
    )
    assert int(handles.coordinator.trainer.step) == _BURST_STEPS

    rows = [json.loads(line)
            for segment in sorted((tmp_path / "logs").glob("events_*.jsonl"))
            for line in segment.read_text(encoding="utf-8").splitlines() if line.strip()]
    saves = [row for row in rows if row["event"] == "periodic_checkpoint_save"]

    assert {row["step"] for row in saves} == {5, 10, 15}, (
        "every periodic boundary the run crossed must be readable from the stream — an "
        "absent event with a present artefact is exactly the composed sink=None drop "
        f"(F-R-P2B-2); got steps {sorted(row['step'] for row in saves)}"
    )
    assert all(row["interval"] == 5 for row in saves)
    assert len(saves) == 3, f"one event per boundary, no duplicates; got {saves}"

    ckpt_names = {p.name for p in (tmp_path / "checkpoints").glob("*.ckpt")}
    for row in saves:
        assert row["path"] is not None and Path(row["path"]).name in ckpt_names, (
            "the event must carry the WRITER's returned path (LAW-18 post-write emit, "
            f"OP-9's ordering) and that artefact must exist; got {row}"
        )
    # 3 periodic + the clean-completion artefact at the terminus = exactly 4.
    assert len(ckpt_names) == 4, (
        "boundaries 5/10/15 plus the clean-stop save at 16 — fewer means a dropped WRITE "
        "(never this defect's shape), more means a second write authority; got "
        f"{sorted(ckpt_names)}"
    )

    # The trainer's OWN per-step diagnostic literal is delivered too, under its own name
    # (`trainer_step`, NEVER the coordinator's `training_step`): one row per learner step, and the
    # same `sink=None` revert that kills the periodic assertions kills this one.
    trainer_rows = [row for row in rows if row["event"] == "trainer_step"]
    assert {row["step"] for row in trainer_rows} == set(range(1, _BURST_STEPS + 1)), (
        "one trainer_step diagnostic row per learner step must ride the composed stream; "
        f"got steps {sorted({row['step'] for row in trainer_rows})}"
    )
    assert all(row["representation"] == "graph" for row in trainer_rows), (
        "this drive's declared route is graph — the diagnostic row carries its tail"
    )
