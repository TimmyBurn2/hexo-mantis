# >300 justify (R8): ONE theme — a run that died could not be recovered — and each arm's
# mutation self-test means nothing away from the arm it controls.
"""The survivability triangle: resume in, model out, no invented paths.

The resume branch dispatched on a `checkpoint_path` no production launch ever passed. The stall
abort snapshotted the replay buffer and dropped the WEIGHTS, the expensive half, back to a
periodic checkpoint every shipped config disables. And the snapshot path defaulted to a
CWD-relative literal, with the failure swallowed. The save arms carry mutation self-tests, since
"attempted" and "succeeded" look identical from outside unless the failure is counted.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.config.resolve.bootstrap import BootstrapNotFoundError
import mantis.run as mantis_run
from mantis.train.buffer_persist import canonical_buffer_path, try_save_buffer
from mantis.train.lifecycle.watchdog import (
    SELFPLAY_STALL_EXIT_CODE,
    StallWatchdog,
    watchdog_snapshot_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
#: What a stubbed R348(c) stamp check hands back: the three fields `main` logs.
_STAMP = {"config_sha256": "stub", "tree_sha": "stub", "preflight_utc": "stub"}


class _InitTrainerSpy:
    """Stand in for `init_trainer`, recording what the composition root passed."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(model=None, arch=None, step=0, checkpoint_dir=Path("."))


@pytest.mark.parametrize(
    ("passed", "expected"),
    [("/some/ckpt.pt", "/some/ckpt.pt"), (None, None)],
    ids=["resume-target-forwarded", "fresh-run-forwards-none"],
)
def test_build_run_collaborators_forwards_the_resume_target(
    monkeypatch: pytest.MonkeyPatch, passed: str | None, expected: str | None,
) -> None:
    """`checkpoint_path` REACHES `init_trainer`, which is the resume dispatch. The `None` case
    is the mutation half: a hardcoded path passes the first case and fails this one."""
    spy = _InitTrainerSpy()
    monkeypatch.setattr(mantis_run, "init_trainer", spy)
    monkeypatch.setattr(mantis_run, "_select_buffer", lambda *_a, **_k: object())
    monkeypatch.setattr(mantis_run, "WorkerPool", lambda **_k: object())

    config = _minted_config()
    mantis_run.build_run_collaborators(
        config=config, out_dir=_tmp_out_dir(), checkpoint_path=passed)

    assert spy.kwargs is not None, "init_trainer was never called"
    assert "checkpoint_path" in spy.kwargs, (
        "the composition root did not pass `checkpoint_path` to init_trainer at all — the "
        "resume branch is unreachable from production, which is defect (a)"
    )
    assert spy.kwargs["checkpoint_path"] == expected


def test_launch_run_forwards_the_resume_target_without_branching() -> None:
    """`launch_run` FORWARDS the resume target and does not branch on it. Structural, because
    a branch here would be a second boot path and is invisible on a green tier."""
    tree = ast.parse((REPO_ROOT / "src" / "mantis" / "run.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "launch_run")
    body = [s for s in fn.body if not (isinstance(s, ast.Expr)
                                       and isinstance(s.value, ast.Constant))]
    assert len(body) == 2, (
        "launch_run grew a third statement — the resume target is forwarded, never branched "
        f"on; found {[type(s).__name__ for s in body]}"
    )
    build_call = body[0].value  # pyright: ignore[reportAttributeAccessIssue]
    kwargs = {kw.arg for kw in build_call.keywords}
    assert "checkpoint_path" in kwargs, "launch_run does not forward checkpoint_path"


def test_cli_resume_flag_reaches_launch_run(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    """`--resume-from` is the operator's route in, and it reaches `launch_run`. The path must
    now EXIST: `main` validates it before launching, which the row below drives."""
    seen: dict[str, Any] = {}

    def _fake_launch(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(shutdown=SimpleNamespace(abort_rule=None))

    monkeypatch.setattr(mantis_run, "launch_run", _fake_launch)
    monkeypatch.setattr(mantis_run, "load_config", lambda _p: object())
    monkeypatch.setattr(mantis_run, "require_preflight_stamp", lambda _c, *, tree_root: _STAMP)

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"not a real checkpoint, but a real file")
    rc = mantis_run.main(["--config", "c.yaml", "--out-dir", "o", "--resume-from", str(ckpt)])
    assert rc == 0
    assert seen.get("checkpoint_path") == str(ckpt), (
        f"--resume-from did not reach launch_run; got {seen.get('checkpoint_path')!r}"
    )


def test_cli_resume_flag_with_a_STALE_path_refuses_before_it_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale `--resume-from` refuses BEFORE the launch, not as a torch error deep inside the
    resume branch. `launch_run` is monkeypatched to EXPLODE, so a green proves the order."""
    def _must_not_launch(**_kwargs: Any) -> Any:
        raise AssertionError("launch_run was reached with a nonexistent --resume-from path")

    monkeypatch.setattr(mantis_run, "launch_run", _must_not_launch)
    monkeypatch.setattr(mantis_run, "load_config", lambda _p: object())
    monkeypatch.setattr(mantis_run, "require_preflight_stamp", lambda _c, *, tree_root: _STAMP)

    with pytest.raises(BootstrapNotFoundError, match="does not exist"):
        mantis_run.main(["--config", "c.yaml", "--out-dir", "o",
                         "--resume-from", "/nonexistent/ckpt.pt"])


def test_cli_without_the_flag_launches_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation half: omitting the flag means FRESH, not some default path."""
    seen: dict[str, Any] = {}
    monkeypatch.setattr(mantis_run, "launch_run",
                        lambda **kw: (seen.update(kw),
                                      SimpleNamespace(shutdown=SimpleNamespace(
                                          abort_rule=None)))[1])
    monkeypatch.setattr(mantis_run, "load_config", lambda _p: object())
    monkeypatch.setattr(mantis_run, "require_preflight_stamp", lambda _c, *, tree_root: _STAMP)

    mantis_run.main(["--config", "c.yaml", "--out-dir", "o"])
    assert seen.get("checkpoint_path") is None, (
        "omitting --resume-from must launch fresh; a non-None value here is a code-side "
        "default choosing a resume target for the operator (R1)"
    )


def _fired_watchdog(**over: Any) -> tuple[StallWatchdog, list[str], list[int]]:
    """Build a watchdog wired with ordered spies, fired once."""
    order: list[str] = []
    exits: list[int] = []
    wd = StallWatchdog(
        timeout_sec=1800.0,
        clock=lambda: 0.0,
        sink=None,
        exit_fn=exits.append,
        save_snapshot=over.get("save_snapshot", lambda: order.append("buffer")),
        save_model=over.get("save_model", lambda: order.append("model")),
    )
    wd.arm(0)
    wd.tick(0, now=1801.0)
    return wd, order, exits


def test_stall_abort_saves_the_model_and_the_buffer() -> None:
    """The fire path saves the model as well as the buffer, and the model first."""
    _wd, order, exits = _fired_watchdog()
    assert "model" in order, (
        "the stall abort did not save model state — a wedged run still exits with its "
        "weights dropped, which is the whole of defect (b)"
    )
    assert "buffer" in order, "the buffer snapshot regressed"
    assert order.index("model") < order.index("buffer"), (
        "weights are the expensive half to regenerate and must be written first"
    )
    assert exits == [SELFPLAY_STALL_EXIT_CODE]


def test_a_failing_model_save_is_counted_and_still_exits() -> None:
    """A failing model save is COUNTED and the fire still exits: under a bare `except: pass` a
    fire that saved nothing and one that saved everything look identical."""
    def _boom() -> None:
        raise OSError("disk gone")

    wd, order, exits = _fired_watchdog(save_model=_boom)

    assert wd.counters.get("watchdog_model_save") == 1, (
        "a failed model save was not counted — this is the swallow LAW-14 bans"
    )
    assert "buffer" in order, "a failing model save must not block the buffer snapshot"
    assert exits == [SELFPLAY_STALL_EXIT_CODE], (
        "the fire path must still exit; best_effort never raises"
    )


def test_a_failing_buffer_save_is_counted_too() -> None:
    def _boom() -> None:
        raise OSError("disk gone")

    wd, _order, exits = _fired_watchdog(save_snapshot=_boom)
    assert wd.counters.get("watchdog_snapshot") == 1
    assert exits == [SELFPLAY_STALL_EXIT_CODE]


def test_a_clean_fire_counts_nothing() -> None:
    """The counters stay 0 when both saves succeed, or they report nothing."""
    wd, _order, _exits = _fired_watchdog()
    assert wd.counters.total() == 0, (
        "a healthy fire incremented a failure counter — the counter cannot distinguish a "
        "broken save from a working one"
    )


def test_canonical_buffer_path_is_derived_from_the_runs_own_directory(
    tmp_path: Path,
) -> None:
    """The path follows the run's own checkpoint dir and is never CWD-relative."""
    got = canonical_buffer_path(tmp_path / "checkpoints")
    assert got == tmp_path / "checkpoints" / "replay_buffer.bin"
    assert got.is_absolute(), (
        "a relative canonical path is the CWD-relative defect: a run launched from another "
        "directory writes its snapshot outside its own --out-dir"
    )
    assert watchdog_snapshot_path(got) != got, (
        "the watchdog snapshot must be a DISTINCT path so an abnormal-exit save can never "
        "truncate the resume buffer"
    )


def test_try_save_buffer_refuses_to_invent_a_path() -> None:
    """With persistence ON and no path configured, fail loud rather than guess: a snapshot
    nobody can find is worth what no snapshot is worth at the moment you need it."""
    with pytest.raises(KeyError, match="buffer_persist_path"):
        try_save_buffer(object(), {"buffer_persist": True}, trigger="test")


def test_try_save_buffer_is_inert_when_persistence_is_off() -> None:
    """Mutation half: the raise is conditional on persistence being ENABLED."""
    try_save_buffer(object(), {}, trigger="test")  # must not raise


@pytest.mark.parametrize("rel", ["src/mantis/train/coordinator/step.py",
                                 "src/mantis/train/buffer_persist.py"])
def test_the_cwd_relative_default_is_gone(rel: str) -> None:
    """Neither site reintroduces the CWD-relative literal, derived from the file rather than
    asserted from memory, so any `.get(..., default)` shape reds."""
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    offenders = [ln for ln in text.splitlines()
                 if '"checkpoints/replay_buffer.bin"' in ln and not ln.lstrip().startswith("#")]
    assert not offenders, (
        f"{rel} reintroduced the CWD-relative code-side default (R1):\n  "
        + "\n  ".join(offenders)
    )


def _minted_config() -> Any:
    """Load the real minted config — production parameters, not a hand-built stub.

    It mints a cuda device, so the root asserts the allocator posture before building; the drive
    states `default`, the regime CI runs in, and re-validates so the validators still run.
    """
    from mantis.config import load_config
    from mantis.config.schema import RunConfig

    base = load_config(str(REPO_ROOT / "configs" / "smoke_preflight_armed.yaml")).model_dump()
    base["allocator_posture"] = "default"
    return RunConfig.model_validate(base)


def _tmp_out_dir() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="mantis-survivability-"))
