"""Item 4 pins — the survivability triangle: resume in, model out, no invented paths.

Three defects, one theme: a run that died could not be recovered.

  (a) THE RESUME LEG WAS UNREACHABLE. `init_trainer` dispatches fresh-vs-resume on an
      explicit `checkpoint_path`, and `mantis.run` never passed one. The branch existed, was
      unit-tested, and no production launch could enter it.
  (b) THE STALL ABORT SAVED THE WRONG HALF. The watchdog fire path snapshotted the replay
      buffer and nothing else, so a wedged run exited with its positions kept and its
      WEIGHTS dropped back to the last periodic checkpoint — and `train.checkpoint_interval`
      is 0 in every shipped config, so that is routinely no checkpoint at all. The buffer is
      the cheap half to regenerate.
  (c) THE SNAPSHOT PATH WAS INVENTED, AND CWD-RELATIVE. Two sites defaulted to the literal
      `"checkpoints/replay_buffer.bin"`; the production root passes `mixing_cfg={}`, so the
      default always won, and a run launched from outside the repo root wrote its snapshot
      into an unrelated `./checkpoints/` rather than its own `--out-dir`. R1 bans the
      code-side default; the save failure was also swallowed by a bare `except: pass`,
      which LAW-14 bans.

Every test drives the real production objects. The (b) and (c) arms each carry a mutation
self-test, because "the save was attempted" and "the save succeeded" look identical from
outside unless the failure is counted (LAW-07/LAW-14).
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mantis.run as mantis_run
from mantis.train.buffer_persist import canonical_buffer_path, try_save_buffer
from mantis.train.lifecycle.watchdog import (
    SELFPLAY_STALL_EXIT_CODE,
    StallWatchdog,
    watchdog_snapshot_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── (a) the resume leg is reachable from production ────────────────────────────────────


class _InitTrainerSpy:
    """Stands in for `init_trainer`, recording exactly what the composition root passed."""

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
    """`checkpoint_path` must REACH `init_trainer` — that call is the resume dispatch.

    The `None` case is the mutation half: a builder that hardcoded a path, or dropped the
    parameter and always passed `None`, passes the first case and fails this one.
    """
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
    """O-A2 holds WITH the new parameter: `launch_run` forwards it, it does not branch on it.

    Structural, because a `if checkpoint_path:` branch here would be a second boot path and
    is behaviourally invisible on a green tier — the exact mutation O-A2 names.
    """
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


def test_cli_resume_flag_reaches_launch_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--resume-from` is the operator's route in. Without it the wiring below is dead."""
    seen: dict[str, Any] = {}

    def _fake_launch(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(shutdown=SimpleNamespace(abort_rule=None))

    monkeypatch.setattr(mantis_run, "launch_run", _fake_launch)
    monkeypatch.setattr(mantis_run, "load_config", lambda _p: object())

    rc = mantis_run.main(["--config", "c.yaml", "--out-dir", "o", "--resume-from", "ckpt.pt"])
    assert rc == 0
    assert seen.get("checkpoint_path") == "ckpt.pt", (
        f"--resume-from did not reach launch_run; got {seen.get('checkpoint_path')!r}"
    )


def test_cli_without_the_flag_launches_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation half of the above: omitting the flag must mean FRESH, not some default path."""
    seen: dict[str, Any] = {}
    monkeypatch.setattr(mantis_run, "launch_run",
                        lambda **kw: (seen.update(kw),
                                      SimpleNamespace(shutdown=SimpleNamespace(
                                          abort_rule=None)))[1])
    monkeypatch.setattr(mantis_run, "load_config", lambda _p: object())

    mantis_run.main(["--config", "c.yaml", "--out-dir", "o"])
    assert seen.get("checkpoint_path") is None, (
        "omitting --resume-from must launch fresh; a non-None value here is a code-side "
        "default choosing a resume target for the operator (R1)"
    )


# ── (b) the stall abort saves WEIGHTS, and failures are counted ────────────────────────


def _fired_watchdog(**over: Any) -> tuple[StallWatchdog, list[str], list[int]]:
    """A watchdog wired with ordered spies, fired once."""
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
    """Defect (b): the fire path used to save the buffer ONLY."""
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
    """LAW-14 mutation self-test: the failure must be COUNTED, not swallowed.

    Mechanism: `best_effort` is the only sanctioned optional-effect path and it increments a
    named counter on failure. The old code was `except Exception: pass`, under which a fire
    that saved nothing and a fire that saved everything produced identical observable state —
    so a broken save could never be detected from a run's own record.
    """
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
    """The counters must stay 0 when both saves succeed, or they report nothing."""
    wd, _order, _exits = _fired_watchdog()
    assert wd.counters.total() == 0, (
        "a healthy fire incremented a failure counter — the counter cannot distinguish a "
        "broken save from a working one"
    )


# ── (c) no invented paths ──────────────────────────────────────────────────────────────


def test_canonical_buffer_path_is_derived_from_the_runs_own_directory(
    tmp_path: Path,
) -> None:
    """The path follows the run's checkpoint dir — it is never CWD-relative."""
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
    """R1: with persistence ON and no path configured, fail loud rather than guess.

    The old default wrote a CWD-relative file. A snapshot nobody can find is worth exactly
    as much as no snapshot at the moment you need it, so silence is the worse failure.
    """
    with pytest.raises(KeyError, match="buffer_persist_path"):
        try_save_buffer(object(), {"buffer_persist": True}, trigger="test")


def test_try_save_buffer_is_inert_when_persistence_is_off() -> None:
    """Mutation half: the raise must be conditional on persistence being ENABLED."""
    try_save_buffer(object(), {}, trigger="test")  # must not raise


@pytest.mark.parametrize("rel", ["src/mantis/train/coordinator/step.py",
                                 "src/mantis/train/buffer_persist.py"])
def test_the_cwd_relative_default_is_gone(rel: str) -> None:
    """Source census: neither site may reintroduce the literal.

    Derived from the file rather than asserted from memory — if the string comes back under
    any `.get(..., default)` shape, this reds regardless of how it is spelled around.
    """
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    offenders = [ln for ln in text.splitlines()
                 if '"checkpoints/replay_buffer.bin"' in ln and not ln.lstrip().startswith("#")]
    assert not offenders, (
        f"{rel} reintroduced the CWD-relative code-side default (R1):\n  "
        + "\n  ".join(offenders)
    )


# ── helpers ────────────────────────────────────────────────────────────────────────────


def _minted_config() -> Any:
    """The real minted run5 config — production parameters, not a hand-built stub (R155)."""
    from mantis.config import load_config

    return load_config(str(REPO_ROOT / "configs" / "smoke_gnn.yaml"))


def _tmp_out_dir() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="mantis-survivability-"))
