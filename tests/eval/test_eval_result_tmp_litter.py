"""F-816-20 item 3a — the immortal `<round_id>_result.json.tmp`.

THE DEFECT. The eval worker writes its result as `tmp.write_text(...)` then
`tmp.replace(target)`. A kill between those two lines — a round timeout escalation, a drain
kill, `stop()`'s teardown, or the whole run being SIGKILLed — leaves the `.tmp` on disk
FOREVER, because round ids are unique (`r{round_idx:06d}_{step}`) and nothing ever writes that
name again. Atomicity is genuinely intact: a reader sees the complete old file or nothing. So
this is litter, never corruption — but litter with no expiry, in a directory a long run
revisits every round.

THE FIX IS PARENT-SIDE ONLY. `worker.py`'s write lines are not touched, moved or re-indented:
the cutover battery's `encoding=`/`fsync` half lands on exactly those lines later, and keeping
the two halves textually disjoint is what makes that a clean follow-on instead of a rebase.
The one shared FACT is how the `.tmp` name is spelled, and the last row here pins the parent's
derivation against the worker's so a future edit to either cannot drift unnoticed.

WHY THERE IS NO CROSS-PROCESS "SIGKILL MID-WRITE" ROW. There is no deterministic scheduling
point between `write_text` and `replace`, and reimplementing the write to manufacture one would
stop testing the real producer. What needs a test is the FIX, and the mechanism it turns on is
`is_alive()` — which reports whether the OS process exists at all, so it can only read False
once the interpreter is gone. The rows below drive that guard from both sides, which is the
whole of what a real mid-write kill would exercise; the real-instrument half is the box check
(`find <work_dir> -name '*.tmp'` after a supervisor kill), not a CI row.

>300 justify (R8): ONE fix — "the parent removes the litter its dead worker left" — expressed
at three sites (the construction sweep, the finalise route, the `stop()` teardown route) that
share ONE guard and ONE tmp-name derivation. Every row here is half of a pair: each removal
row is only safe because a matching row proves the same code does NOT remove a live writer's
tmp or a completed result. Splitting by site would put a removal in one file and its safety
control in another, and would need the `_FakeProc` whose liveness the rows drive duplicated
three ways — three copies free to drift, and a drifted control stops controlling anything.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.config.schema import EvalConfig, GateConfig, LadderConfig, LadderRung
from mantis.eval.pipeline import DrainCaps, EvalPipeline, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks

_ROUND_ID = "r000001_1000"


# ── fixtures, self-contained (R5 bars importing another test module) ─────────────────────
def _eval_cfg() -> EvalConfig:
    rungs = [
        LadderRung(name="sealbot_d5", bot="sealbot", variant="d5", depth=5, opponent_sims=None,
                   opening_book="book_v1_s20260625_p4", deploy_matched=True, games_max=32),
    ]
    gate = GateConfig(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625,
    )
    ladder = LadderConfig(
        rungs=rungs, round_games=64, min_games_per_active_rung=4,
        graduation_wr_lower_ci=0.75, graduation_consec_rounds=3, activation_wr_lower_ci=0.65,
        calibration_every_k_rounds=4, calibration_games=8, bootstrap_resamples=1000,
        bootstrap_ci_level=0.95, bt_prior_games=1.0, bootstrap_seed=1234,
    )
    return EvalConfig(
        random_model_sims=96, sealbot_model_sims=128, kraken_model_sims=128,
        strix_model_sims=128, random_floor_games=4, worker_device="cpu",
        round_timeout_sec=5.0, worker_kill_grace_sec=0.1, gate=gate, ladder=ladder,
        ply_cap_adjudication=None, strength_floor=None,
    )


#: The run id these fixtures build under, and the ONE place it is spelled. RQ-13 scoped the work
#: dir by `run_id`, so the sidecar path is `<out-dir>/spool.work/<run_id>` — derived here rather
#: than hardcoded at each assertion site, where three copies would have drifted independently.
_RUN_ID = "q3_tmp_litter"


def _work_dir(tmp_path: Path) -> Path:
    return tmp_path / "spool.work" / _RUN_ID


def _pipeline_kwargs(tmp_path: Path, **overrides: Any) -> dict:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(exist_ok=True)
    kwargs: dict[str, Any] = dict(
        eval_cfg=_eval_cfg(),
        coordinator_cfg_caps=DrainCaps(
            final_eval_drain_timeout_sec=5.0, eval_final_drain_safety_factor=1.0,
            eval_final_drain_hard_cap_sec=5.0, terminal_eval_hard_cap_sec=5.0,
        ),
        encoding="v6_live2_ls",
        amp_dtype="bf16",
        max_plies=128,
        run_id=_RUN_ID,
        spool_dir=spool_dir,
        ladder_state_path=tmp_path / "ladder_state.json",
        promotion=DeployTagHooks(
            anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
            best_model_path=tmp_path / "best_model.pt",
            run_id=_RUN_ID,
            encoding="v6_live2_ls",
            save_anchor=lambda *a, **k: None,
            guarded_load=lambda *a, **k: None,
        ),
        fused_graph_caps=None,
        inference_batching=None,
    )
    kwargs.update(overrides)
    return kwargs


class _FakeProc:
    """A worker stand-in whose liveness the row controls.

    `alive_until_killed=True` reproduces `stop()`'s own premise: the child is running, the
    parent terminates and then kills it, and only after that does the OS stop reporting it.
    """

    def __init__(self, *, alive: bool, alive_until_killed: bool = False) -> None:
        self._alive = alive
        self._alive_until_killed = alive_until_killed
        self.exitcode = 0 if not alive else None
        self.terminated = 0
        self.killed = 0

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1
        if self._alive_until_killed:
            self._alive = False
            self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
        return None


def _bare_pipeline(tmp_path: Path, **attrs: Any) -> Any:
    """A pipeline with only the state the finalise/stop paths read (the `__new__` house shape,
    technique cited from `tests/eval/test_promotion_integrity.py`)."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    pipe = EvalPipeline.__new__(EvalPipeline)
    pipe._lock = threading.Lock()
    pipe._inflight = None
    pipe._mailbox = []
    pipe._sink = None
    pipe._double_finalize_suppressed = 0
    pipe._clock = lambda: 0.0
    pipe._work_dir = work
    pipe._eval_cfg = _eval_cfg()
    for name, value in attrs.items():
        setattr(pipe, name, value)
    return pipe


def _inflight(work: Path, proc: Any, round_id: str = _ROUND_ID) -> dict[str, Any]:
    result_path = work / f"{round_id}_result.json"
    return {
        "round_id": round_id, "step": 1000, "proc": proc, "t0": 0.0, "round_idx": 1,
        "spec": SimpleNamespace(result_path=str(result_path), round_id=round_id),
    }


# ── the construction-time sweep ──────────────────────────────────────────────────────────
def test_a_stale_result_tmp_is_swept_when_a_pipeline_opens_the_work_dir(tmp_path) -> None:
    """The ONLY handle the whole-process-SIGKILL case has. Nothing runs in a killed run, so the
    only process that can ever clean its litter is a LATER pipeline over the same work dir —
    which is exactly what a `--resume-from` relaunch into the same out-dir is.

    Safe because at construction this pipeline provably has no live writer, and a second
    pipeline over one work dir does not happen: `build_eval_pipeline` has one call site, once
    per process, and the dir is derived from `--out-dir` AND `run_id` (RQ-13). The run_id half
    is what makes that structural rather than circumstantial — derived from the out-dir alone,
    the claim was false exactly when two runs shared one, and nothing locks an out-dir."""
    work = _work_dir(tmp_path)
    work.mkdir(parents=True)
    stale = work / f"{_ROUND_ID}_result.json.tmp"
    stale.write_text("{}", encoding="utf-8")

    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    try:
        assert pipeline._work_dir == work, "the work dir under test is not the one swept"
        assert not stale.exists(), (
            "a stale result tmp survived a fresh pipeline opening its work dir — nothing else "
            "will ever remove it, because round ids are unique and never reused"
        )
    finally:
        pipeline.stop()


def test_the_sweep_leaves_completed_results_and_snapshots_alone(tmp_path) -> None:
    """THE NEGATIVE CONTROL, and the most important row in this file: a sweep that eats results
    is far worse than the litter it was built to remove. Only the `.tmp` suffix is in scope —
    the completed result, the round spec, the progress sidecar and a candidate snapshot must
    all survive untouched, byte for byte."""
    work = _work_dir(tmp_path)
    work.mkdir(parents=True)
    keep = {
        work / f"{_ROUND_ID}_result.json": '{"ok": true}',
        work / f"{_ROUND_ID}_spec.json": '{"round_id": "x"}',
        work / f"{_ROUND_ID}_progress.txt": "12/64",
        work / f"{_ROUND_ID}_candidate.pt": "not really a tensor",
    }
    for path, body in keep.items():
        path.write_text(body, encoding="utf-8")

    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    try:
        for path, body in keep.items():
            assert path.exists(), f"the sweep deleted {path.name}, which it must never touch"
            assert path.read_text(encoding="utf-8") == body, f"the sweep rewrote {path.name}"
    finally:
        pipeline.stop()


# ── the per-round unlink at finalize ─────────────────────────────────────────────────────
def test_finalizing_a_round_removes_that_rounds_tmp(tmp_path) -> None:
    """Case 1 — the child died, the run lives. Every finalising route (`_poll_loop`,
    `_escalate_and_finalize`, `drain_pending`, `_run_terminal_sync`) converges here, so one
    unlink at this point covers all four. The completed result must survive: this row asserts
    both halves, because an unlink that took the target instead of the tmp would still make the
    `.tmp` assertion pass."""
    pipe = _bare_pipeline(tmp_path)
    proc = _FakeProc(alive=False)
    inflight = _inflight(pipe._work_dir, proc)
    result_path = Path(inflight["spec"].result_path)
    tmp = Path(str(result_path) + ".tmp")
    tmp.write_text("{}", encoding="utf-8")
    result_path.write_text(json.dumps({"round_id": _ROUND_ID}), encoding="utf-8")

    pipe._read_worker_result = lambda inflight, **_kw: {"round_id": inflight["round_id"]}
    pipe._finalize_round(inflight)

    assert not tmp.exists(), "the dead round's tmp survived its own finalise"
    assert result_path.exists(), "the finalise deleted the RESULT, not the tmp"


def test_a_LIVE_worker_keeps_its_tmp(tmp_path) -> None:
    """The guard, from the other side, and it is the half that makes the unlink safe rather
    than merely tidy. Unlinking a `.tmp` a LIVE writer is about to `replace()` would turn
    litter into a FAILED ROUND — the fix being worse than the defect. The check is re-run
    against the live process table inside `_finalize_round` itself, not trusted from a caller
    that only sent terminate/kill and joined with a bound."""
    pipe = _bare_pipeline(tmp_path)
    proc = _FakeProc(alive=True)
    inflight = _inflight(pipe._work_dir, proc)
    tmp = Path(inflight["spec"].result_path + ".tmp")
    tmp.write_text("{}", encoding="utf-8")

    pipe._read_worker_result = lambda inflight, **_kw: {"round_id": inflight["round_id"]}
    pipe._finalize_round(inflight)

    assert tmp.exists(), (
        "the tmp of a round whose writer is still ALIVE was unlinked — the next `replace()` "
        "will fail and a healthy round breaks"
    )


# ── RC-2: the stop() teardown route ──────────────────────────────────────────────────────
def test_stop_removes_the_inflight_rounds_tmp_on_teardown(tmp_path) -> None:
    """RC-2, the review's required change. `stop()` is called unconditionally from
    `compose_run`'s teardown on EVERY run exit, and it has its own terminate -> join -> kill ->
    join sequence that never reaches `_finalize_round`. A round in flight at ordinary shutdown
    is routine, not pathological, so before this fix the commonest producer of the litter was
    the one route the per-round unlink did not cover.

    The fake reproduces `stop()`'s own premise: alive on entry, gone after the kill."""
    pipe = _bare_pipeline(
        tmp_path,
        _stop_event=threading.Event(),
        _poller=SimpleNamespace(is_alive=lambda: False, join=lambda _t=None: None),
    )
    proc = _FakeProc(alive=True, alive_until_killed=True)
    inflight = _inflight(pipe._work_dir, proc)
    pipe._inflight = inflight
    tmp = Path(inflight["spec"].result_path + ".tmp")
    tmp.write_text("{}", encoding="utf-8")

    pipe.stop()

    assert proc.terminated and proc.killed, "stop() did not run its own teardown sequence"
    assert not tmp.exists(), (
        "the in-flight round's tmp survived `stop()` — this is the routine teardown route "
        "RC-2 found uncovered, not the disclosed whole-process-SIGKILL residual"
    )


def test_stop_keeps_the_tmp_of_a_worker_that_refuses_to_die(tmp_path) -> None:
    """The same guard on the `stop()` route. If terminate and kill both fail to end the writer
    — a D-state child, a join that timed out — the parent must NOT delete a file that process
    may still `replace()`. Loud litter beats a broken round."""
    pipe = _bare_pipeline(
        tmp_path,
        _stop_event=threading.Event(),
        _poller=SimpleNamespace(is_alive=lambda: False, join=lambda _t=None: None),
    )
    proc = _FakeProc(alive=True, alive_until_killed=False)
    inflight = _inflight(pipe._work_dir, proc)
    pipe._inflight = inflight
    tmp = Path(inflight["spec"].result_path + ".tmp")
    tmp.write_text("{}", encoding="utf-8")

    pipe.stop()

    assert tmp.exists(), "stop() unlinked the tmp of a worker that is still alive"


# ── the anti-drift pin against the worker's own spelling ─────────────────────────────────
def test_the_parent_derives_the_same_tmp_name_the_worker_writes(tmp_path) -> None:
    """THE SEAM ROW (design §5.3). The parent spells the temp name `result_path + ".tmp"`; the
    worker spells it `target.with_suffix(target.suffix + ".tmp")`. Those are two derivations of
    one fact, in two files, and the cutover battery is going to edit the worker's side later.
    This row is what makes that edit safe: change either spelling and the parent stops removing
    the file the worker writes, silently, with no other test noticing."""
    target = tmp_path / f"{_ROUND_ID}_result.json"
    worker_side = target.with_suffix(target.suffix + ".tmp")
    parent_side = Path(str(target) + ".tmp")
    assert worker_side == parent_side, (
        f"the parent and the worker disagree on the tmp name: {parent_side} vs {worker_side}"
    )

    src = Path(__file__).resolve().parents[2] / "src" / "mantis" / "eval" / "worker.py"
    text = src.read_text(encoding="utf-8")
    assert 'with_suffix(target.suffix + ".tmp")' in text, (
        "the worker's tmp-name derivation has moved; the parent's unlink is keyed on it and "
        "this pin is the only thing that would have said so"
    )


def test_a_round_whose_writer_left_no_tmp_finalises_normally(tmp_path) -> None:
    """The ordinary case — a clean round already `replace()`d its tmp away — must not become an
    error. `missing_ok=True` is the whole of it, and this row is what keeps a future
    "tighten the unlink" edit from converting every healthy round into a broken one."""
    pipe = _bare_pipeline(tmp_path)
    inflight = _inflight(pipe._work_dir, _FakeProc(alive=False))
    Path(inflight["spec"].result_path).write_text("{}", encoding="utf-8")

    pipe._read_worker_result = lambda inflight, **_kw: {"round_id": inflight["round_id"]}
    result = pipe._finalize_round(inflight)
    assert result == {"round_id": _ROUND_ID}


def test_the_unlink_never_manufactures_a_broken_round(tmp_path, monkeypatch) -> None:
    """A deletion failure must not fabricate an `eval_broken(round_completion_error)`. The call
    sits in `_finalize_round`'s UN-CAUGHT prologue precisely because it has been made
    non-raising by construction; if that ever stops being true, the poller thread dies silently
    — the F1 class this file must not reintroduce."""
    pipe = _bare_pipeline(tmp_path)
    inflight = _inflight(pipe._work_dir, _FakeProc(alive=False))
    tmp = Path(inflight["spec"].result_path + ".tmp")
    tmp.write_text("{}", encoding="utf-8")

    real_unlink = Path.unlink

    def _boom(self: Path, *a: Any, **kw: Any) -> None:
        if self.name.endswith(".tmp"):
            raise PermissionError("read-only work dir")
        real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", _boom)
    pipe._read_worker_result = lambda inflight, **_kw: {"round_id": inflight["round_id"]}
    assert pipe._finalize_round(inflight) == {"round_id": _ROUND_ID}


def test_the_construction_sweep_survives_an_unlinkable_stale_tmp(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same never-fatal contract at the other site. A work dir the run cannot delete from
    is a real condition (a read-only mount, a stale NFS handle), and a pipeline that refuses to
    BUILD because it could not remove litter would be a far worse failure than the litter."""
    work = _work_dir(tmp_path)
    work.mkdir(parents=True)
    (work / f"{_ROUND_ID}_result.json.tmp").write_text("{}", encoding="utf-8")

    real_unlink = Path.unlink

    def _boom(self: Path, *a: Any, **kw: Any) -> None:
        if self.name.endswith(".tmp"):
            raise PermissionError("read-only work dir")
        real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", _boom)
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    pipeline.stop()
