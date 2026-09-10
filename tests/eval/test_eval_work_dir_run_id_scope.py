"""The eval work dir is `run_id`-scoped, not just `--out-dir`-scoped.

Nothing locks an out-dir, and round ids are per-run counters, so two runs sharing one out-dir
once shared one sidecar directory and collided on identical filenames. Scoping by `run_id`
makes the construction sweep's precondition — no live writer at construction — structural
rather than circumstantial, while a `--resume-from` relaunch still lands on its own litter
(same config, so same out-dir and same run_id).

No path sanitizer is written: `run_id` is schema-constrained to `^[a-z0-9][a-z0-9_\\-]*$`, so it
cannot contain `/`, `.` or `..`, and a sanitizer would be a second authority for that.
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
        leaf_batch_size=1,
        max_plies=128,
        c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, eval_cfg=cfg.eval,
        caps=DrainCaps(final_eval_drain_timeout_sec=1.0, eval_final_drain_safety_factor=1.0,
                       eval_final_drain_hard_cap_sec=1.0, terminal_eval_hard_cap_sec=1.0),
        encoding=cfg.identity.encoding, run_id=run_id, spool_dir=spool, game_record_dir=str(spool) + "_games",
        ladder_state_path=tmp_path / f"ladder_{run_id}.json", promotion=None, sink=None,
        fused_graph_caps=None,
        inference_batching=None,
    )


def test_the_work_dir_names_the_run(tmp_path):
    """Prove the run is identifiable from its sidecar path."""
    pipeline = _pipeline(tmp_path, "alpha_run")
    try:
        assert "alpha_run" in pipeline._work_dir.parts, (
            f"the work dir does not name its run: {pipeline._work_dir}"
        )
    finally:
        pipeline.stop()


def test_two_runs_sharing_one_out_dir_do_not_share_a_work_dir(tmp_path):
    """Prove two runs sharing one out-dir get different work dirs.

    Round ids are per-run counters, so before this scoping both wrote `r000001_1_result.json`
    to the same path.
    """
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
    """Prove a relaunch of the same run lands on the same work dir, and so on its own litter."""
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
    """Prove the work dir stays a sibling of the spool dir, not nested inside it.

    `spool_dir` holds only model snapshot `.pt` files, and another test walks every file under
    it and `torch.load()`s it.
    """
    pipeline = _pipeline(tmp_path, "sibling_check")
    try:
        assert pipeline._spool_dir not in pipeline._work_dir.parents, (
            f"the work dir moved INSIDE the spool dir: {pipeline._work_dir}"
        )
    finally:
        pipeline.stop()


def test_the_construction_sweep_does_not_reach_another_runs_litter(tmp_path):
    """Prove the construction sweep cannot delete another run's in-flight result tmp."""
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
