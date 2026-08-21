"""RQ-13 / clause (l) — the eval work dir is `run_id`-scoped, not just `--out-dir`-scoped.

THE ROW, AND WHAT IT IS ACTUALLY ABOUT. A5's caveat: *"the eval `_work_dir` is `--out-dir`-scoped,
not `run_id`-scoped, and nothing locks an out-dir."* Two runs pointed at one out-dir therefore
shared one sidecar directory for their specs, results and progress files. Round ids are
`r{round_idx:06d}_{step}` — a per-run counter, not a global one — so two runs collide on the SAME
filenames, and a `.tmp` sweep written for one run's litter can reach the other's live writer.

WHY THIS IS A STRENGTHENING OF THE SWEEP AND NOT A RISK TO IT. The construction sweep
(`F-816-20` item 3a) is the only handle the whole-process-SIGKILL case has, and its stated
precondition is that *"at CONSTRUCTION this pipeline has no live writer"*. That precondition was
argued from `build_eval_pipeline` having one call site per process — true — plus the dir being
derived from `--out-dir`, which is exactly the half that does NOT hold when two runs share one.
Scoping the dir by `run_id` makes the precondition structural instead of circumstantial.

WHY A RELAUNCH STILL SWEEPS ITS OWN LITTER — measured, not assumed. `run.py:691` takes
`run_id = config.run_id` and `run.py:477` takes `log_dir = out_dir / "logs"`; both are
config-derived and neither carries a timestamp. A `--resume-from` relaunch supplies the same
`--config`, so it lands on the same out-dir AND the same run_id, and therefore on the same work
dir. The equivalence class the sweep depends on is unchanged; only the path spelling moved.

PATH SAFETY IS A TYPE PROPERTY HERE, NOT A CHECK. `run_id` is schema-constrained to
`^[a-z0-9][a-z0-9_\\-]*$` (`config/schema/core.py:331`), so it cannot contain `/`, `.` or `..` and
cannot escape the directory it names. That is why no sanitizer is written: one would be a second
authority for a constraint the schema already enforces, and R296(f) is explicit that verification
derives from structure rather than from text handling.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mantis.config.loader import load_config
from mantis.eval.pipeline import DrainCaps, EvalPipeline

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "configs" / "dev_example.yaml"


def _pipeline(tmp_path: Path, run_id: str, spool_name: str = "spool") -> Any:
    cfg = load_config(_CONFIG)
    spool = tmp_path / spool_name
    spool.mkdir(parents=True, exist_ok=True)
    return EvalPipeline(
        eval_cfg=cfg.eval,
        caps=DrainCaps(final_eval_drain_timeout_sec=1.0, eval_final_drain_safety_factor=1.0,
                       eval_final_drain_hard_cap_sec=1.0, terminal_eval_hard_cap_sec=1.0),
        encoding=cfg.identity.encoding, run_id=run_id, spool_dir=spool,
        ladder_state_path=tmp_path / f"ladder_{run_id}.json", promotion=None, sink=None,
        fused_graph_caps=None,
    )


def test_the_work_dir_names_the_run(tmp_path):
    """The row's subject in one line: the run is identifiable from the sidecar path."""
    pipeline = _pipeline(tmp_path, "alpha_run")
    try:
        assert "alpha_run" in pipeline._work_dir.parts, (
            f"the work dir does not name its run: {pipeline._work_dir}"
        )
    finally:
        pipeline.stop()


def test_two_runs_sharing_one_out_dir_do_not_share_a_work_dir(tmp_path):
    """THE DEFECT. Nothing locks an out-dir, and round ids are per-run counters — so before this
    scoping, two runs in one out-dir wrote `r000001_1_result.json` to the same path."""
    a = _pipeline(tmp_path, "run_alpha")
    b = _pipeline(tmp_path, "run_beta")
    try:
        assert a._work_dir != b._work_dir, (
            "two runs sharing an out-dir still share a sidecar directory; their per-run round ids "
            f"collide on identical filenames in {a._work_dir}"
        )
    finally:
        a.stop()
        b.stop()


def test_the_same_run_relaunched_into_the_same_out_dir_gets_the_SAME_work_dir(tmp_path):
    """The sweep's precondition, pinned. `--resume-from` supplies the same `--config`, so the same
    run_id and the same out-dir — a relaunch must land on its own litter or the SIGKILL case loses
    the only handle it has."""
    first = _pipeline(tmp_path, "same_run")
    first_dir = first._work_dir
    first.stop()
    second = _pipeline(tmp_path, "same_run")
    try:
        assert second._work_dir == first_dir, (
            "a relaunch of the same run landed on a different work dir, so the construction sweep "
            "can no longer reach the litter its own previous process left"
        )
    finally:
        second.stop()


def test_the_work_dir_is_still_a_SIBLING_of_the_spool_dir(tmp_path):
    """LAW-12 carve-out, unchanged by this scoping and pinned so it stays that way.

    `spool_dir` holds ONLY model snapshot `.pt` files — `test_snapshots_are_not_checkpoints`
    walks every file under it and `torch.load()`s it. A spec or a progress sidecar nested there
    would break that walk, so the sidecar tree is a sibling. Adding a `run_id` level must not
    quietly move it inside.
    """
    pipeline = _pipeline(tmp_path, "sibling_check")
    try:
        assert pipeline._spool_dir not in pipeline._work_dir.parents, (
            f"the work dir moved INSIDE the spool dir: {pipeline._work_dir}"
        )
    finally:
        pipeline.stop()


def test_the_construction_sweep_does_not_reach_another_runs_litter(tmp_path):
    """The strengthening, driven. The sweep deletes `*_result.json.tmp` at construction on the
    argument that no live writer exists. With one shared dir that argument was false across runs:
    a second run booting could delete a FIRST run's in-flight tmp. Scoping makes it true.
    """
    other = _pipeline(tmp_path, "other_run")
    other_litter = other._work_dir / "r000001_1_result.json.tmp"
    other_litter.parent.mkdir(parents=True, exist_ok=True)
    other_litter.write_text("{}", encoding="utf-8")
    other.stop()

    mine = _pipeline(tmp_path, "my_run")
    try:
        assert other_litter.exists(), (
            "booting one run swept a DIFFERENT run's result tmp — with a shared out-dir that file "
            "may belong to a live writer, which is the precondition the sweep claims to have"
        )
    finally:
        mine.stop()
